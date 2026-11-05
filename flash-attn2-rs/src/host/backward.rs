use std::ffi::c_void;

use tvm_ffi::error::{Error, RUNTIME_ERROR, Result, VALUE_ERROR};
use tvm_ffi::{Any, AnyView, Function, Tensor};
use tvm_ffi_sys::TVMFFIEnvGetStream;
use tvm_ffi_sys::dlpack::{DLDataType, DLDataTypeCode, DLDeviceType};

use crate::host::cuda::{launch_kernel, sys};
use crate::host::layout::{Strides, prepare_qkv, tensor_span};
use crate::host::runtime::{Runtime, WIDE_D32, WIDE_MODE, generic_smem, generic_tile, runtime};
use crate::host::tiles::{HEAD_DIMS, combined_smem};
use crate::host::{EMPTY_SLICE, MAX_GRID_YZ, alloc_tensor, any_into_raw_owned, arg, err};

/// With `pre`, argument 5 is the forward output instead of delta: delta is
/// computed here, clearing the fused dQ accumulator in the same launch, and
/// appended to the outputs.
fn backward_component(args: &[AnyView], kv: bool, all: bool, pre: bool) -> Result<Vec<Tensor>> {
    if args.len() != 14 {
        return Err(err(
            VALUE_ERROR,
            "backward component expects q, k, v, dout, lse, delta, alibi, scale, causal, left, right, softcap, dropout, rng_state",
        ));
    }
    let (q, bf16, qs) = prepare_qkv("q", arg(args, 0, "q")?)?;
    let (k, kb, ks) = prepare_qkv("k", arg(args, 1, "k")?)?;
    let (v, vb, vs) = prepare_qkv("v", arg(args, 2, "v")?)?;
    let (dout, db, ds) = prepare_qkv("dout", arg(args, 3, "dout")?)?;
    let lse: Tensor = arg(args, 4, "lse")?;
    let pre_out = if pre {
        Some(prepare_qkv("out", arg(args, 5, "out")?)?)
    } else {
        None
    };
    let delta: Tensor = if let Some((out, _, _)) = &pre_out {
        let [b, sq, h, _] = *out.shape() else {
            unreachable!()
        };
        alloc_tensor(
            &[b, h, (sq + 127) / 128 * 128],
            DLDataType::new(DLDataTypeCode::kDLFloat, 32, 1),
            out.device(),
        )?
    } else {
        arg(args, 5, "delta")?
    };
    let alibi: Option<Tensor> = arg(args, 6, "alibi")?;
    let scale: Option<f64> = arg(args, 7, "scale")?;
    let causal: bool = arg(args, 8, "causal")?;
    let mut left: i64 = arg(args, 9, "left")?;
    let mut right: i64 = arg(args, 10, "right")?;
    let cap: f64 = arg(args, 11, "softcap")?;
    let p: f64 = arg(args, 12, "dropout")?;
    let rng: Option<Tensor> = arg(args, 13, "rng_state")?;
    let [b, sq, h, d] = *q.shape() else {
        unreachable!()
    };
    let sk = k.shape()[1];
    let hk = k.shape()[2];
    let device = q.device();
    if bf16 != kb
        || bf16 != vb
        || bf16 != db
        || k.device() != device
        || v.device() != device
        || dout.device() != device
    {
        return Err(err(
            VALUE_ERROR,
            "backward inputs must share q's dtype and device",
        ));
    }
    if dout.shape() != q.shape() || k.shape() != v.shape() || k.shape()[0] != b || k.shape()[3] != d
    {
        return Err(err(VALUE_ERROR, "inconsistent backward input shapes"));
    }
    if b <= 0
        || h <= 0
        || b > MAX_GRID_YZ
        || h > MAX_GRID_YZ
        || hk <= 0
        || h % hk != 0
        || d < 8
        || d > 256
        || d % 8 != 0
        || sq < 0
        || sq > i32::MAX as i64 - 127
        || sk < 0
        || sk > i32::MAX as i64
    {
        return Err(err(VALUE_ERROR, "unsupported backward dimensions"));
    }
    if !p.is_finite()
        || !(0.0..1.0).contains(&p)
        || p as f32 >= 1.0
        || (p as f32 > 0.0 && cap > 0.0)
    {
        return Err(err(
            VALUE_ERROR,
            "invalid dropout or dropout combined with softcap",
        ));
    }
    let rounded_q = (sq + 127) / 128 * 128;
    let float32 = DLDataType::new(DLDataTypeCode::kDLFloat, 32, 1);
    for (name, tensor, shape) in [
        ("lse", &lse, [b, h, sq]),
        ("delta", &delta, [b, h, rounded_q]),
    ] {
        if tensor.shape() != shape
            || tensor.dtype() != float32
            || tensor.device() != device
            || (tensor.numel() != 0 && !tensor.is_contiguous())
        {
            return Err(err(
                VALUE_ERROR,
                &format!("{name} must be contiguous float32 with shape {shape:?} on q's device"),
            ));
        }
    }
    if p as f32 > 0.0 {
        let state = rng
            .as_ref()
            .ok_or_else(|| err(VALUE_ERROR, "dropout backward requires saved RNG state"))?;
        if state.shape() != [2]
            || state.dtype() != DLDataType::new(DLDataTypeCode::kDLInt, 64, 1)
            || state.device() != device
            || !state.is_contiguous()
            || state.data_ptr() as u64 % 8 != 0
        {
            return Err(err(
                VALUE_ERROR,
                "RNG state must be contiguous CUDA int64 [2] on q's device",
            ));
        }
    }
    let alibi_bs = if let Some(a) = &alibi {
        if a.dtype() != float32 || a.device() != device || !a.is_contiguous() {
            return Err(err(
                VALUE_ERROR,
                "alibi must be contiguous CUDA float32 on q's device",
            ));
        }
        match *a.shape() {
            [n] if n == h => 0,
            [bb, n] if bb == b && n == h => h,
            _ => return Err(err(VALUE_ERROR, "alibi must have shape [H] or [B,H]")),
        }
    } else {
        0
    };
    if i32::try_from(left).is_err() || i32::try_from(right).is_err() {
        return Err(err(VALUE_ERROR, "window bounds must fit int32"));
    }
    if left >= sk {
        left = -1;
    }
    if right >= sk {
        right = -1;
    }
    if causal {
        right = 0;
    }
    let alibi_causal = left < 0 && right == 0;
    if left < 0 && right >= 0 {
        left = sk;
    }
    if left >= 0 && right < 0 {
        right = sk;
    }
    let original_scale = scale.unwrap_or(1.0 / (d as f64).sqrt());
    let (mut score_scale, mut tanh_scale) = if cap > 0.0 && original_scale != 0.0 {
        (
            cap.copysign(original_scale) as f32,
            (original_scale.abs() / cap) as f32,
        )
    } else {
        (original_scale as f32, 0.0f32)
    };
    let mut keep = 1.0f32 - p as f32;
    let mut gradient_scale = original_scale as f32 / keep;
    let fused = if all {
        let rt = runtime(device.device_id)?;
        let q_bytes = q.numel().checked_mul(4);
        let kv_bytes = b
            .checked_mul(sk)
            .and_then(|n| n.checked_mul(h))
            .and_then(|n| n.checked_mul(d))
            .and_then(|n| n.checked_mul(8));
        let key_blocks = b * h * ((sk + 63) / 64);
        // Tiny query tiles benefit from fewer launches on underfilled grids.
        let short_query = sq <= if d <= 64 { 128 } else { 32 };
        let enough_parallelism = (short_query || key_blocks >= rt.sm_count / 2)
            && (d == 32 || sq <= sk || key_blocks >= rt.sm_count);
        // Extend the fused D64 path for square, dropout-free MHA. The split
        // fallback recomputes attention probabilities separately for dQ and dKV.
        let long_d64 = d == 64
            && p == 0.0
            && h == hk
            && sq == sk
            && alibi.is_none()
            && ((left < 0 && right < 0) || (left >= sk && right == 0));
        let max_fused_seq = 4096;
        // D192/D256 use the generic fused kernel.
        let wide = matches!(d, 192 | 256);
        rt.sm89
            && (matches!(d, 32 | 64 | 96 | 128) || wide)
            && enough_parallelism
            && cap <= 0.0
            && (32..=max_fused_seq).contains(&sq)
            && (128..=max_fused_seq).contains(&sk)
            && ((left < 0 && right < 0)
                || (alibi.is_none() && left >= sk && right == 0)
                || left + right + 1 >= 256)
            && (wide
                || (q_bytes.is_some_and(|n| n <= (rt.l2_bytes / 2).max(0) as usize)
                    && (h == hk || kv_bytes.is_some_and(|n| n <= rt.l2_bytes / 2))))
    } else {
        false
    };
    // The generic kernel wins for wide and D32 heads, D64 dropout, and D96
    // except short dropout; D64 plain and D128 keep their tuned kernels.
    let generic = fused
        && match d {
            32 | 192 | 256 => true,
            64 => true,
            128 => true,
            96 => p == 0.0 || sq > 1024,
            _ => false,
        };
    // Matches the kernel's plain mode: no dropout, mask or bias.
    let generic_plain = generic && p == 0.0 && alibi.is_none() && left < 0 && right < 0;
    let wide = generic_plain && d == 32 && sq <= 1024 && sk <= 1024;
    let layout = if wide {
        WIDE_D32
    } else {
        generic_tile(d, p > 0.0, generic_plain, false)
    };
    let fused_pipeline = fused
        && !generic
        && d == 64
        && p == 0.0
        && h == hk
        && alibi.is_none()
        && ((left < 0 && right < 0) || (left >= sk && right == 0));
    // Larger causal tiles amortize gradient work once their grid fills the GPU.
    // Keep the smaller tile for underfilled grids and partial query/key tiles.
    let causal_query = fused_pipeline
        && alibi_causal
        && sq % 64 == 0
        && sk % 128 == 0
        && b * h * ((sk + 127) / 128) >= runtime(device.device_id)?.sm_count;
    let query_pipeline = fused_pipeline && (!alibi_causal || causal_query);
    let causal_permuted = fused_pipeline && !query_pipeline && sq % 32 == 0;
    // The preprocess launch below needs dout/out to match q's layout.
    let preprocess = |accum: Option<(&Tensor, i64)>| -> Result<()> {
        let (out, out_bf16, os) = pre_out.as_ref().unwrap();
        if out.shape() != q.shape() || *out_bf16 != bf16 || out.device() != device {
            return Err(err(
                VALUE_ERROR,
                "out must match q's shape, dtype and device",
            ));
        }
        if sq == 0 {
            return Ok(());
        }
        let rt = runtime(device.device_id)?;
        rt.ctx
            .bind_to_thread()
            .map_err(|e| err(RUNTIME_ERROR, &format!("bind backward context: {e:?}")))?;
        let stream = unsafe { TVMFFIEnvGetStream(DLDeviceType::kDLCUDA as i32, device.device_id) };
        launch_preprocess(rt, stream.cast(), &dout, ds, out, *os, &delta, p, accum)
    };
    if all && !fused {
        if pre {
            preprocess(None)?;
            let mut forwarded: Vec<AnyView> = args.to_vec();
            forwarded[5] = AnyView::from(&delta);
            let mut gradients = backward_component(&forwarded, false, false, false)?;
            gradients.extend(backward_component(&forwarded, true, false, false)?);
            gradients.push(delta);
            return Ok(gradients);
        }
        let mut gradients = backward_component(args, false, false, false)?;
        gradients.extend(backward_component(args, true, false, false)?);
        return Ok(gradients);
    }
    let fused_q = if fused {
        Some(alloc_tensor(q.shape(), q.dtype(), device)?)
    } else {
        None
    };
    let dq = alloc_tensor(if kv { k.shape() } else { q.shape() }, q.dtype(), device)?;
    let dv = if kv {
        Some(alloc_tensor(v.shape(), v.dtype(), device)?)
    } else {
        None
    };
    let finish = || -> Result<Vec<Tensor>> {
        let mut outputs = Vec::new();
        if let Some(output) = &fused_q {
            outputs.push(output.clone());
        }
        outputs.push(dq.clone());
        if let Some(dv) = &dv {
            outputs.push(dv.clone());
        }
        if pre {
            outputs.push(delta.clone());
        }
        Ok(outputs)
    };
    if (if kv { sk } else { sq }) == 0 {
        return finish();
    }
    let bucket = *HEAD_DIMS.iter().find(|&&bucket| bucket >= d).unwrap();
    let rt = runtime(device.device_id)?;
    // Eight-warps with coalesced dQ atomics outperform query splits for this
    // full-wave MHA shape. GQA retains the existing query-split path.
    let dense128 = fused
        && !generic
        && h == hk
        && d == 128
        && p == 0.0
        && sq == 512
        && sk == 512
        && b * h * ((sk + 63) / 64) == rt.sm_count
        && alibi.is_none()
        && ((left < 0 && right < 0) || (left >= sk && right == 0));
    // Each wide block does roughly twice the work. Avoid paying for a
    // partially occupied extra wave when the narrower grid fills the device.
    let wide_blocks = b * h * ((sk + 255) / 256);
    let narrow_blocks = b * h * ((sk + 127) / 128);
    let wide_waves = (wide_blocks + rt.sm_count - 1) / rt.sm_count;
    let narrow_waves = (narrow_blocks + rt.sm_count - 1) / rt.sm_count;
    let query_wide =
        query_pipeline && !alibi_causal && sq >= 2048 && 2 * wide_waves <= narrow_waves;

    rt.ctx
        .bind_to_thread()
        .map_err(|e| err(RUNTIME_ERROR, &format!("bind backward context: {e:?}")))?;
    let stream = unsafe { TVMFFIEnvGetStream(DLDeviceType::kDLCUDA as i32, device.device_id) };
    if (if kv { sq } else { sk }) == 0 {
        for output in std::iter::once(&dq).chain(dv.iter()) {
            unsafe {
                sys::cuMemsetD8Async(
                    output.data_ptr() as u64,
                    0,
                    (output.numel() * 2) as usize,
                    stream.cast(),
                )
            }
            .result()
            .map_err(|e| {
                err(
                    RUNTIME_ERROR,
                    &format!("clear empty attention gradients: {e:?}"),
                )
            })?;
        }
        return finish();
    }
    // Expand underfilled GQA grids; FP32 scratch bounded to half of L2.
    let partial_bytes = b
        .checked_mul(sk)
        .and_then(|n| n.checked_mul(h))
        .and_then(|n| n.checked_mul(d))
        .and_then(|n| n.checked_mul(8));
    // Query partitions fill the two resident CTA slots on small D128
    // grids. Four partitions help only when the original grid uses <= half
    // the SMs. Combined FP32 scratch still fits within half of L2.
    // Shorter query ranges do not amortize the extra partial-output traffic.
    let query_partitions = if p == 0.0 && b * h * ((sk + 63) / 64) <= rt.sm_count / 2 {
        4i64
    } else {
        2i64
    };
    let split_fused_queries = fused
        && !generic
        && !dense128
        && d == 128
        && (256..=512).contains(&sq)
        && (h > hk || sq == 512)
        && h <= 65535 / query_partitions
        && b * h * ((sk + 63) / 64) <= rt.sm_count
        && partial_bytes.is_some_and(|n| n <= rt.l2_bytes / (2 * query_partitions));
    let parallel_gqa = (fused && (h > hk || split_fused_queries))
        || (kv
            && rt.sm89
            && bucket == 128
            && h > hk
            && sq >= 64
            && (sq >= 256 || sk <= sq || (left < 0 && right < 0))
            && b * hk * ((sk + 31) / 32) <= 2 * rt.sm_count
            && partial_bytes.is_some_and(|bytes| bytes <= rt.l2_bytes / 2));
    let partial_mode = if p as f32 > 0.0 {
        2
    } else if cap <= 0.0 && alibi.is_none() && left < 0 && right < 0 {
        0
    } else {
        1
    };
    let both_masked = alibi.is_some() || left >= 0 || right >= 0;
    // Normalized causal windows (left=Sk, right=0) stay eligible.
    let wide_window = (left < 0 && right < 0) || left + right + 1 >= 256;
    // Count only key blocks intersecting some row's window.
    let first_key = if left >= 0 {
        (sk - sq - left).max(0)
    } else {
        0
    };
    let active_key_blocks = (sk + 63) / 64 - first_key / 64;
    let both_kv = kv
        && rt.sm89
        && bucket == 128
        && cap <= 0.0
        && wide_window
        && sq >= 256
        && sk >= 64
        && b * (if parallel_gqa { h } else { hk }) * active_key_blocks >= rt.sm_count;
    let kv_query_splits = if split_fused_queries {
        query_partitions
    } else {
        1i64
    };
    let partials = if parallel_gqa {
        Some((
            alloc_tensor(&[b, sk, h * kv_query_splits, d], float32, device)?,
            alloc_tensor(&[b, sk, h * kv_query_splits, d], float32, device)?,
        ))
    } else {
        None
    };
    // Split keys for underfilled dQ grids, >= 4 tiles per split.
    let query_blocks = b * h * ((sq + 63) / 64);
    let mut q_splits = 1u32;
    if !kv && rt.sm89 && bucket == 128 && sk >= 256 && query_blocks <= rt.sm_count / 2 {
        // Mirrors device KvRange rounding.
        let min_key_tiles = if left < 0 && right < 0 {
            (sk + 31) / 32
        } else {
            [0, ((sq - 1) / 64) * 64]
                .into_iter()
                .map(|first_q| {
                    let last_q = (first_q + 63).min(sq - 1);
                    let first_k = if left >= 0 {
                        (first_q + sk - sq - left).clamp(0, sk) / 32
                    } else {
                        0
                    };
                    let end_k = if right >= 0 {
                        ((last_q + 1 + sk - sq + right).clamp(0, sk) + 31) / 32
                    } else {
                        (sk + 31) / 32
                    };
                    (end_k - first_k).max(0)
                })
                .min()
                .unwrap()
        };
        for splits in [2u32, 4, 8] {
            let bytes = dq.numel().checked_mul(4 * splits as usize);
            if query_blocks * splits as i64 <= rt.sm_count
                && min_key_tiles / splits as i64 >= 4
                && bytes.is_some_and(|n| n <= (rt.l2_bytes / 2).max(0) as usize)
            {
                q_splits = splits;
            }
        }
    }
    let q_partials = if fused || q_splits > 1 {
        // The generic kernel's permuted reductions cover whole 128-row tiles.
        Some(alloc_tensor(
            &[
                q_splits as i64,
                b,
                if generic { (sq + 127) / 128 * 128 } else { sq },
                h,
                d,
            ],
            float32,
            device,
        )?)
    } else {
        None
    };
    if fused && pre {
        let partial = q_partials.as_ref().unwrap();
        preprocess(Some((partial, partial.shape()[2])))?;
    } else if fused {
        let partial = q_partials.as_ref().unwrap();
        unsafe {
            sys::cuMemsetD8Async(
                partial.data_ptr() as u64,
                0,
                partial.numel() * 4,
                stream.cast(),
            )
        }
        .result()
        .map_err(|e| err(RUNTIME_ERROR, &format!("clear fused dQ: {e:?}")))?;
    }
    let swizzled_q = !kv && rt.sm89 && bucket == 128 && b * h * ((sq + 63) / 64) > rt.sm_count;
    // Slice ABI: device pointer followed by length.
    let mut buffers = Vec::<u64>::new();
    for t in [&q, &k, &v, &dout] {
        buffers.extend([t.data_ptr() as u64, tensor_span(t) / 2]);
    }
    for t in [&lse, &delta] {
        buffers.extend([t.data_ptr() as u64, t.numel() as u64]);
    }
    if let Some((pk, pv)) = &partials {
        buffers.extend([pk.data_ptr() as u64, pk.numel() as u64]);
        buffers.extend([pv.data_ptr() as u64, pv.numel() as u64]);
    } else if !kv && let Some(partial) = &q_partials {
        buffers.extend([partial.data_ptr() as u64, partial.numel() as u64]);
    } else {
        buffers.extend([dq.data_ptr() as u64, dq.numel() as u64 / 2]);
        if let Some(dv) = &dv {
            buffers.extend([dv.data_ptr() as u64, dv.numel() as u64 / 2]);
        }
    }
    if fused {
        buffers.push(q_partials.as_ref().unwrap().data_ptr() as u64);
    }
    buffers.extend(
        alibi
            .as_ref()
            .map_or(EMPTY_SLICE, |t| [t.data_ptr() as u64, t.numel() as u64]),
    );
    buffers.extend(
        rng.as_ref()
            .map_or(EMPTY_SLICE, |t| [t.data_ptr() as u64, t.numel() as u64]),
    );
    let mut params: Vec<*mut c_void> = buffers.iter_mut().map(|v| (v as *mut u64).cast()).collect();
    params.push((&mut score_scale as *mut f32).cast());
    params.push((&mut tanh_scale as *mut f32).cast());
    let mut config = [
        left as u32,
        right as u32,
        sq as u32,
        sk as u32,
        alibi.is_some() as u32,
        alibi_causal as u32,
        h as u32,
        (h / hk) as u32,
        (d / 2) as u32,
        qs.batch,
        qs.seq,
        qs.head,
        ks.batch,
        ks.seq,
        ks.head,
        vs.batch,
        vs.seq,
        vs.head,
        alibi_bs as u32,
        ds.batch,
        ds.seq,
        ds.head,
    ];
    for value in &mut config {
        params.push((value as *mut u32).cast());
    }
    params.push((&mut gradient_scale as *mut f32).cast());
    params.push((&mut keep as *mut f32).cast());
    unsafe {
        launch_kernel(
            if dense128 {
                rt.bwd_dense128_fns
                    .get(&(bf16, if right == 0 { 1 } else { 0 }))
            } else if causal_query {
                rt.bwd_causal_query_fns.get(&bf16)
            } else if causal_permuted {
                rt.bwd_causal_permuted_fns.get(&bf16)
            } else if query_wide {
                rt.bwd_fused_wide_pipeline_fns.get(&bf16)
            } else if fused_pipeline {
                rt.bwd_fused_pipeline_fns.get(&(bf16, alibi_causal))
            } else if fused {
                // Normalized causal left=Sk never excludes a valid key.
                let mode = if alibi.is_none() && left < 0 && right < 0 {
                    0
                } else if alibi.is_none() && left >= sk && right == 0 {
                    // Pair long and short causal tiles across two batches.
                    // Unequal D128 dropout grids need enough varying work to
                    // offset the balanced kernel's overhead on short queries.
                    if b == 2
                        && (sq == sk || (d == 128 && p > 0.0 && sq >= 512 && 2 * sq >= sk + 128))
                        && sq >= 256
                        && h * ((sk + 63) / 64) == rt.sm_count
                    {
                        3
                    } else {
                        1
                    }
                } else {
                    2
                };
                if generic {
                    let mode = if wide {
                        WIDE_MODE
                    } else if mode == 3 {
                        1
                    } else {
                        mode
                    };
                    rt.bwd_generic_fns
                        .get(&(d, bf16, h > hk, p as f32 > 0.0, mode))
                } else {
                    let kernels = if d == 32 {
                        &rt.bwd_fused_d32_fns
                    } else if d == 64 {
                        &rt.bwd_fused_d64_fns
                    } else {
                        &rt.bwd_fused_fns
                    };
                    // Modes 4..=6 and 8..=10 select two and four query partitions.
                    let mode = mode
                        + match kv_query_splits {
                            4 => 8,
                            2 => 4,
                            _ => 0,
                        };
                    // Two query partitions still benefit from opposite key order
                    // when the two batches fill both resident CTA slots. Short
                    // query ranges and smaller key grids favor ordinary traversal.
                    let mode = if d == 128
                        && p > 0.0
                        && mode == 5
                        && b == 2
                        && sq >= 384
                        && (sk + 63) / 64 >= 8
                        && 4 * sq > 3 * sk
                        && 2 * h * ((sk + 63) / 64) == rt.sm_count
                    {
                        7
                    } else {
                        mode
                    };
                    // Sign encoding wins on full D128 tiles with one original
                    // key CTA per SM; shorter grids retain the regular mask path.
                    let mode = if d == 128
                        && p > 0.0
                        && mode == 4
                        && sq == 512
                        && sk == 512
                        && b * h * (sk / 64) == rt.sm_count
                    {
                        12
                    } else {
                        mode
                    };
                    kernels.get(&(bf16, parallel_gqa, p as f32 > 0.0, mode))
                }
            } else if both_kv {
                rt.bwd_dkv_both_fns
                    .get(&(bf16, parallel_gqa, p as f32 > 0.0, both_masked))
            } else if parallel_gqa {
                rt.bwd_dkv_specialized_fns.get(&(bf16, true, partial_mode))
            } else if kv {
                if rt.sm89 && bucket == 128 && partial_mode != 2 {
                    rt.bwd_dkv_specialized_fns.get(&(bf16, false, partial_mode))
                } else {
                    rt.bwd_dkv_fns.get(&(bucket, bf16))
                }
            } else if q_splits > 1 {
                rt.bwd_dq_split_fns.get(&(bf16, q_splits))
            } else if swizzled_q {
                rt.bwd_dq_swizzled_fns.get(&(
                    bf16,
                    if cap <= 0.0 && partial_mode != 0 {
                        if p as f32 > 0.0 { 4 } else { 3 }
                    } else {
                        partial_mode
                    },
                ))
            } else {
                rt.bwd_dq_fns.get(&(bucket, bf16))
            }
            .unwrap()
            .cu_function(),
            if kv {
                (
                    (if query_wide {
                        (sk + 255) / 256
                    } else if query_pipeline {
                        (sk + 127) / 128
                    } else if generic {
                        (sk + layout.keys - 1) / layout.keys
                    } else if fused || both_kv {
                        (sk + 63) / 64
                    } else {
                        (sk + 31) / 32
                    }) as u32,
                    (if parallel_gqa {
                        h * kv_query_splits
                    } else {
                        hk
                    }) as u32,
                    b as u32,
                )
            } else {
                (((sq + 63) / 64) as u32 * q_splits, h as u32, b as u32)
            },
            (
                if generic {
                    layout.warps * 32
                } else if query_pipeline || dense128 {
                    256
                } else {
                    128
                },
                1,
                1,
            ),
            if dense128 {
                98304
            } else if query_wide {
                90112
            } else if query_pipeline {
                98304
            } else if fused_pipeline {
                34816
            } else if generic {
                generic_smem(d, layout)
            } else if fused {
                if d == 32 {
                    21504
                } else if d == 64 {
                    25600
                } else {
                    50176
                }
            } else if both_kv {
                combined_smem(p as f32 > 0.0, both_masked)
            } else if swizzled_q {
                50176
            } else {
                ((if kv { 128 } else { 192 }) * (bucket / 2 + 4) * 4) as u32
            },
            stream.cast(),
            &mut params,
        )
    }
    .map_err(|e| err(RUNTIME_ERROR, &format!("launch backward component: {e:?}")))?;
    if let (false, Some((pk, pv)), Some(pq), Some(out_q)) =
        (generic, &partials, &q_partials, &fused_q)
    {
        let out_v = dv.as_ref().unwrap();
        let kv_pairs = dq.numel() as u64 / 2;
        let q_pairs = out_q.numel() as u64 / 2;
        let mut buffers = [
            pk.data_ptr() as u64,
            pk.numel() as u64,
            pv.data_ptr() as u64,
            pv.numel() as u64,
            pq.data_ptr() as u64,
            pq.numel() as u64,
            dq.data_ptr() as u64,
            kv_pairs,
            out_v.data_ptr() as u64,
            out_v.numel() as u64 / 2,
            out_q.data_ptr() as u64,
            q_pairs,
            kv_pairs,
            q_pairs,
        ];
        let mut shape = [
            hk as u32,
            (h / hk * kv_query_splits) as u32,
            (d / 2) as u32,
            q_splits,
        ];
        let mut params: Vec<*mut c_void> =
            buffers.iter_mut().map(|x| (x as *mut u64).cast()).collect();
        params.extend(shape.iter_mut().map(|x| (x as *mut u32).cast()));
        params.push((&mut gradient_scale as *mut f32).cast());
        unsafe {
            launch_kernel(
                rt.bwd_grad_merge_fns.get(&bf16).unwrap().cu_function(),
                (kv_pairs.max(q_pairs).div_ceil(256).min(65535) as u32, 1, 1),
                (256, 1, 1),
                0,
                stream.cast(),
                &mut params,
            )
        }
        .map_err(|e| {
            err(
                RUNTIME_ERROR,
                &format!("merge fused backward gradients: {e:?}"),
            )
        })?;
        return finish();
    }
    if let Some((pk, pv)) = &partials {
        let dv = dv.as_ref().unwrap();
        let mut buffers = [
            pk.data_ptr() as u64,
            pk.numel() as u64,
            pv.data_ptr() as u64,
            pv.numel() as u64,
            dq.data_ptr() as u64,
            dq.numel() as u64 / 2,
            dv.data_ptr() as u64,
            dv.numel() as u64 / 2,
            dq.numel() as u64 / 2,
        ];
        let mut shape = [hk as u32, (h / hk * kv_query_splits) as u32, (d / 2) as u32];
        let mut params: Vec<*mut c_void> =
            buffers.iter_mut().map(|x| (x as *mut u64).cast()).collect();
        params.extend(shape.iter_mut().map(|x| (x as *mut u32).cast()));
        unsafe {
            launch_kernel(
                rt.bwd_dkv_merge_fns.get(&bf16).unwrap().cu_function(),
                (
                    ((dq.numel() as u64 / 2).div_ceil(256).min(65535)) as u32,
                    1,
                    1,
                ),
                (256, 1, 1),
                0,
                stream.cast(),
                &mut params,
            )
        }
        .map_err(|e| err(RUNTIME_ERROR, &format!("merge GQA gradients: {e:?}")))?;
    }
    if generic {
        let partial = q_partials.as_ref().unwrap();
        let out_q = fused_q.as_ref().unwrap();
        let rows = layout.rows;
        let mut buffers = [
            partial.data_ptr() as u64,
            partial.numel() as u64,
            out_q.data_ptr() as u64,
            out_q.numel() as u64 / 2,
        ];
        let mut params: Vec<*mut c_void> =
            buffers.iter_mut().map(|x| (x as *mut u64).cast()).collect();
        let mut shape = [sq as u32, h as u32];
        params.extend(shape.iter_mut().map(|x| (x as *mut u32).cast()));
        params.push((&mut gradient_scale as *mut f32).cast());
        unsafe {
            launch_kernel(
                if wide {
                    rt.bwd_wide_dq_fns.get(&bf16)
                } else {
                    rt.bwd_generic_dq_fns
                        .get(&(d, bf16, p > 0.0, generic_plain))
                }
                .unwrap()
                .cu_function(),
                (((sq + rows - 1) / rows) as u32, h as u32, b as u32),
                (layout.warps * 32, 1, 1),
                (rows * (d + 4) * 4) as u32,
                stream.cast(),
                &mut params,
            )
        }
        .map_err(|e| err(RUNTIME_ERROR, &format!("convert generic dQ: {e:?}")))?;
        return finish();
    }
    if let Some(partial) = &q_partials {
        let dq = fused_q.as_ref().unwrap_or(&dq);
        let mut buffers = [
            partial.data_ptr() as u64,
            partial.numel() as u64,
            dq.data_ptr() as u64,
            dq.numel() as u64 / 2,
            dq.numel() as u64 / 2,
        ];
        let mut params: Vec<*mut c_void> =
            buffers.iter_mut().map(|x| (x as *mut u64).cast()).collect();
        params.push((&mut q_splits as *mut u32).cast());
        let mut q_heads = h as u32;
        params.push((&mut q_heads as *mut u32).cast());
        params.push((&mut gradient_scale as *mut f32).cast());
        let wide_permuted = query_wide && sq % 16 == 0 && sk % 256 == 0;
        let d64_permuted = query_pipeline && !query_wide && sq % 64 == 0 && sk % 128 == 0;
        unsafe {
            launch_kernel(
                rt.bwd_dq_merge_fns
                    .get(&(
                        bf16,
                        if dense128 {
                            1
                        } else if causal_permuted {
                            4
                        } else if wide_permuted {
                            3
                        } else if d64_permuted {
                            2
                        } else {
                            0
                        },
                    ))
                    .unwrap()
                    .cu_function(),
                (
                    ((dq.numel() as u64 / 2)
                        .div_ceil(if dense128 {
                            4096
                        } else if causal_permuted {
                            1024
                        } else if wide_permuted {
                            512
                        } else if d64_permuted {
                            2048
                        } else {
                            256
                        })
                        .min(65535)) as u32,
                    1,
                    1,
                ),
                (if causal_permuted { 128 } else { 256 }, 1, 1),
                if dense128 {
                    16384
                } else if causal_permuted {
                    4096
                } else if wide_permuted {
                    2048
                } else if d64_permuted {
                    8192
                } else {
                    0
                },
                stream.cast(),
                &mut params,
            )
        }
        .map_err(|e| err(RUNTIME_ERROR, &format!("merge dQ key splits: {e:?}")))?;
    }
    finish()
}

fn backward_result(args: &[AnyView], kv: bool, all: bool, pre: bool) -> Result<Any> {
    let outputs = backward_component(args, kv, all, pre)?;
    if kv {
        let views: Vec<AnyView> = outputs.iter().map(AnyView::from).collect();
        Function::get_global("ffi.Array")?.call_packed(&views)
    } else {
        Ok(Any::from(outputs.into_iter().next().unwrap()))
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tvm_ffi_bwd_dq(
    _handle: *mut c_void,
    args: *const tvm_ffi_sys::TVMFFIAny,
    num_args: i32,
    result: *mut tvm_ffi_sys::TVMFFIAny,
) -> i32 {
    let packed = unsafe { std::slice::from_raw_parts(args as *const AnyView, num_args as usize) };
    match backward_result(packed, false, false, false) {
        Ok(value) => {
            unsafe { *result = any_into_raw_owned(value) };
            0
        }
        Err(error) => {
            Error::set_raised(&error);
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tvm_ffi_bwd_dkv(
    _handle: *mut c_void,
    args: *const tvm_ffi_sys::TVMFFIAny,
    num_args: i32,
    result: *mut tvm_ffi_sys::TVMFFIAny,
) -> i32 {
    let packed = unsafe { std::slice::from_raw_parts(args as *const AnyView, num_args as usize) };
    match backward_result(packed, true, false, false) {
        Ok(value) => {
            unsafe { *result = any_into_raw_owned(value) };
            0
        }
        Err(error) => {
            Error::set_raised(&error);
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tvm_ffi_bwd_all(
    _handle: *mut c_void,
    args: *const tvm_ffi_sys::TVMFFIAny,
    num_args: i32,
    result: *mut tvm_ffi_sys::TVMFFIAny,
) -> i32 {
    let packed = unsafe { std::slice::from_raw_parts(args as *const AnyView, num_args as usize) };
    match backward_result(packed, true, true, false) {
        Ok(value) => {
            unsafe { *result = any_into_raw_owned(value) };
            0
        }
        Err(error) => {
            Error::set_raised(&error);
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tvm_ffi_bwd_all_preprocess(
    _handle: *mut c_void,
    args: *const tvm_ffi_sys::TVMFFIAny,
    num_args: i32,
    result: *mut tvm_ffi_sys::TVMFFIAny,
) -> i32 {
    let packed = unsafe { std::slice::from_raw_parts(args as *const AnyView, num_args as usize) };
    match backward_result(packed, true, true, true) {
        Ok(value) => {
            unsafe { *result = any_into_raw_owned(value) };
            0
        }
        Err(error) => {
            Error::set_raised(&error);
            -1
        }
    }
}

fn bwd_preprocess(args: &[AnyView]) -> Result<Any> {
    if args.len() != 3 {
        return Err(err(
            VALUE_ERROR,
            "bwd_preprocess expects dout, out, p_dropout",
        ));
    }
    let p: f64 = arg(args, 2, "p_dropout")?;
    if !p.is_finite() || !(0.0..1.0).contains(&p) || p as f32 >= 1.0 {
        return Err(err(VALUE_ERROR, "p_dropout must be in [0, 1) as float32"));
    }
    let (dout, bf16, ds) = prepare_qkv("dout", arg(args, 0, "dout")?)?;
    let (out, out_bf16, os) = prepare_qkv("out", arg(args, 1, "out")?)?;
    if dout.shape() != out.shape() || bf16 != out_bf16 || dout.device() != out.device() {
        return Err(err(
            VALUE_ERROR,
            "dout and out must share shape, dtype and device",
        ));
    }
    let [b, seq, h, d] = *out.shape() else {
        unreachable!()
    };
    if b <= 0
        || h <= 0
        || b > MAX_GRID_YZ
        || h > MAX_GRID_YZ
        || seq < 0
        || seq > u32::MAX as i64 - 127
        || d < 8
        || d > 256
        || d % 8 != 0
    {
        return Err(err(VALUE_ERROR, "unsupported backward tensor dimensions"));
    }
    let rounded_seq = (seq + 127) / 128 * 128;
    let delta = alloc_tensor(
        &[b, h, rounded_seq],
        DLDataType::new(DLDataTypeCode::kDLFloat, 32, 1),
        out.device(),
    )?;
    if seq == 0 {
        return Ok(Any::from(delta));
    }
    let rt = runtime(out.device().device_id)?;
    rt.ctx
        .bind_to_thread()
        .map_err(|e| err(RUNTIME_ERROR, &format!("bind backward context: {e:?}")))?;
    let stream =
        unsafe { TVMFFIEnvGetStream(DLDeviceType::kDLCUDA as i32, out.device().device_id) };
    launch_preprocess(rt, stream, &dout, ds, &out, os, &delta, p, None)?;
    Ok(Any::from(delta))
}

/// Launch delta = rowsum(dO * O) * keep, optionally clearing a fused dQ
/// accumulator [b, rows, h, d] in the same kernel.
fn launch_preprocess(
    rt: &Runtime,
    stream: *mut c_void,
    dout: &Tensor,
    ds: Strides,
    out: &Tensor,
    os: Strides,
    delta: &Tensor,
    p: f64,
    accum: Option<(&Tensor, i64)>,
) -> Result<()> {
    let [b, seq, h, d] = *out.shape() else {
        unreachable!()
    };
    let bf16 = out.dtype().bits == 16 && out.dtype().code == DLDataTypeCode::kDLBfloat as u8;
    let rounded_seq = (seq + 127) / 128 * 128;
    let lanes = if d <= 64 {
        8
    } else if d <= 128 {
        16
    } else {
        32
    };
    let grid = rounded_seq / (256 / lanes);
    if grid > i32::MAX as i64 {
        return Err(err(
            VALUE_ERROR,
            "backward grid exceeds CUDA indexing range",
        ));
    }
    let mut dp = dout.data_ptr() as u64;
    let mut dl = tensor_span(dout) / 2;
    let mut op = out.data_ptr() as u64;
    let mut ol = tensor_span(out) / 2;
    let mut dst = delta.data_ptr() as u64;
    let mut len = delta.numel() as u64;
    let mut dims = [
        seq as u32,
        rounded_seq as u32,
        h as u32,
        (d / 2) as u32,
        ds.batch,
        ds.seq,
        ds.head,
        os.batch,
        os.seq,
        os.head,
    ];
    let mut keep = 1.0f32 - p as f32;
    let mut params = vec![
        (&mut dp as *mut u64).cast::<c_void>(),
        (&mut dl as *mut u64).cast(),
        (&mut op as *mut u64).cast(),
        (&mut ol as *mut u64).cast(),
        (&mut dst as *mut u64).cast(),
        (&mut len as *mut u64).cast(),
    ];
    for d in &mut dims {
        params.push((d as *mut u32).cast());
    }
    params.push((&mut keep as *mut f32).cast());
    let mut accum_ptr = accum.map_or(0u64, |(t, _)| t.data_ptr() as u64);
    let mut accum_rows = accum.map_or(0u32, |(_, rows)| rows as u32);
    params.push((&mut accum_ptr as *mut u64).cast());
    params.push((&mut accum_rows as *mut u32).cast());
    unsafe {
        launch_kernel(
            rt.bwd_preprocess_fns
                .get(&(bf16, lanes as u32))
                .unwrap()
                .cu_function(),
            (grid as u32, h as u32, b as u32),
            (256, 1, 1),
            0,
            stream.cast(),
            &mut params,
        )
    }
    .map_err(|e| err(RUNTIME_ERROR, &format!("backward preprocess: {e:?}")))?;
    Ok(())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn __tvm_ffi_bwd_preprocess(
    _handle: *mut c_void,
    args: *const tvm_ffi_sys::TVMFFIAny,
    num_args: i32,
    result: *mut tvm_ffi_sys::TVMFFIAny,
) -> i32 {
    let packed = unsafe { std::slice::from_raw_parts(args as *const AnyView, num_args as usize) };
    match bwd_preprocess(packed) {
        Ok(value) => {
            unsafe { *result = any_into_raw_owned(value) };
            0
        }
        Err(error) => {
            Error::set_raised(&error);
            -1
        }
    }
}
