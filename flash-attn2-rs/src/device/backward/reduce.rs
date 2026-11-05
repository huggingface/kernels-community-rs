use crate::device::inlined::{cvt, unpack};
use cuda_device::warp::shuffle_xor_f32;
use cuda_device::{DisjointSlice, kernel, launch_bounds, thread};

/// Output precision and temporary layout of the fixed-order partial-sum merges.
pub trait MergeSpec {
    const BF16: bool;
    const PERMUTE: bool;
    const D64: bool;
    const WIDE: bool;
    const CAUSAL: bool;
}
pub struct Merge<
    const BF16: bool,
    const PERMUTE: bool = false,
    const D64: bool = false,
    const WIDE: bool = false,
    const CAUSAL: bool = false,
>;
impl<const BF16: bool, const PERMUTE: bool, const D64: bool, const WIDE: bool, const CAUSAL: bool>
    MergeSpec for Merge<BF16, PERMUTE, D64, WIDE, CAUSAL>
{
    const BF16: bool = BF16;
    const PERMUTE: bool = PERMUTE;
    const D64: bool = D64;
    const WIDE: bool = WIDE;
    const CAUSAL: bool = CAUSAL;
}
#[allow(non_camel_case_types)]
pub type f16 = Merge<false>;
#[allow(non_camel_case_types)]
pub type bf16 = Merge<true>;
#[allow(non_camel_case_types)]
pub type f16_permuted = Merge<false, true>;
#[allow(non_camel_case_types)]
pub type bf16_permuted = Merge<true, true>;
#[allow(non_camel_case_types)]
pub type f16_permuted_d64 = Merge<false, true, true>;
#[allow(non_camel_case_types)]
pub type bf16_permuted_d64 = Merge<true, true, true>;
#[allow(non_camel_case_types)]
pub type f16_permuted_wide = Merge<false, true, true, true>;
#[allow(non_camel_case_types)]
pub type bf16_permuted_wide = Merge<true, true, true, true>;
#[allow(non_camel_case_types)]
pub type f16_permuted_causal = Merge<false, true, true, false, true>;
#[allow(non_camel_case_types)]
pub type bf16_permuted_causal = Merge<true, true, true, false, true>;

pub trait ReduceSpec {
    const BF16: bool;
    const LANES: u32;
}
pub struct Reduce<const BF16: bool, const LANES: u32>;
impl<const BF16: bool, const LANES: u32> ReduceSpec for Reduce<BF16, LANES> {
    const BF16: bool = BF16;
    const LANES: u32 = LANES;
}
#[allow(non_camel_case_types)]
pub type f16_8 = Reduce<false, 8>;
#[allow(non_camel_case_types)]
pub type f16_16 = Reduce<false, 16>;
#[allow(non_camel_case_types)]
pub type f16_32 = Reduce<false, 32>;
#[allow(non_camel_case_types)]
pub type bf16_8 = Reduce<true, 8>;
#[allow(non_camel_case_types)]
pub type bf16_16 = Reduce<true, 16>;
#[allow(non_camel_case_types)]
pub type bf16_32 = Reduce<true, 32>;

#[inline(always)]
unsafe fn load_four(ptr: *const u32) -> [u32; 4] {
    let a: u32;
    let b: u32;
    let c: u32;
    let d: u32;
    unsafe {
        cuda_device::ptx_asm!(
            "ld.global.v4.u32 {%0, %1, %2, %3}, [%4];",
            out("=r") a, out("=r") b, out("=r") c, out("=r") d,
            in("l") ptr, clobber("memory")
        );
    }
    [a, b, c, d]
}

#[kernel(f16_8, f16_16, f16_32, bf16_8, bf16_16, bf16_32)]
#[launch_bounds(256)]
pub fn bwd_preprocess<S: ReduceSpec>(
    dout: &[u32],
    out: &[u32],
    delta: DisjointSlice<f32>,
    seq: u32,
    rounded_seq: u32,
    heads: u32,
    dim_pairs: u32,
    do_batch: u32,
    do_seq: u32,
    do_head: u32,
    out_batch: u32,
    out_seq: u32,
    out_head: u32,
    keep: f32,
    accum: *mut f32,
    accum_rows: u32,
) {
    let mut delta = delta;
    let tid = thread::threadIdx_x();
    // Clear this block's rows of the fused dQ accumulator, [b, rows, h, d]
    // in FP32, saving the separate memset launch.
    if !accum.is_null() {
        let rows = 256 / S::LANES as usize;
        let first = thread::blockIdx_x() as usize * rows;
        let width = dim_pairs as usize * 2;
        let base = ((thread::blockIdx_z() as usize * accum_rows as usize + first) * heads as usize
            + thread::blockIdx_y() as usize)
            * width;
        let row_stride = heads as usize * width;
        let mut i = tid as usize * 4;
        while i < rows * width {
            let row = i / width;
            if first + row < accum_rows as usize {
                unsafe {
                    let ptr = accum.add(base + row * row_stride + i % width);
                    cuda_device::ptx_asm!(
                        "st.global.v4.f32 [%0], {%1, %1, %1, %1};",
                        in("l") ptr as u64, in("f") 0.0f32, clobber("memory")
                    );
                }
            }
            i += 1024;
        }
    }
    // Keep sequence blocks fastest in the grid, followed by head and batch.
    // Direct coordinates avoid dividing a flattened row index in every thread.
    let s = thread::blockIdx_x() as u64 * (256 / S::LANES) as u64 + (tid / S::LANES) as u64;
    let h = thread::blockIdx_y() as u64;
    let b = thread::blockIdx_z() as u64;
    let row = (b * heads as u64 + h) * rounded_seq as u64 + s;
    let lane = tid % S::LANES;
    let mut sum = 0.0f32;
    if s < seq as u64 {
        let a_base = b * do_batch as u64 + s * do_seq as u64 + h * do_head as u64;
        let b_base = b * out_batch as u64 + s * out_seq as u64 + h * out_head as u64;
        let ap = unsafe { dout.as_ptr().add(a_base as usize) };
        let bp = unsafe { out.as_ptr().add(b_base as usize) };
        // Eight lanes cover D64 with one aligned 16-byte load per input.
        // Unaligned rows and other dimensions retain the scalar path.
        if S::LANES == 8 && dim_pairs == 32 && ((ap as usize | bp as usize) & 15) == 0 {
            let aa = unsafe { load_four(ap.add(lane as usize * 4)) };
            let bb = unsafe { load_four(bp.add(lane as usize * 4)) };
            let mut i = 0;
            while i < 4 {
                thread::__unroll_config::<0>();
                let a = aa[i];
                let b = bb[i];
                let (a0, a1, b0, b1) = if S::BF16 {
                    (
                        unpack::<true>(a),
                        unpack::<true>(a >> 16),
                        unpack::<true>(b),
                        unpack::<true>(b >> 16),
                    )
                } else {
                    (
                        unpack::<false>(a),
                        unpack::<false>(a >> 16),
                        unpack::<false>(b),
                        unpack::<false>(b >> 16),
                    )
                };
                sum += a0 * b0;
                sum += a1 * b1;
                i += 1;
            }
        } else {
            let mut d = lane;
            while d < dim_pairs {
                let (a, b) = unsafe {
                    (
                        *dout.as_ptr().add((a_base + d as u64) as usize),
                        *out.as_ptr().add((b_base + d as u64) as usize),
                    )
                };
                let (a0, a1, b0, b1) = if S::BF16 {
                    (
                        unpack::<true>(a),
                        unpack::<true>(a >> 16),
                        unpack::<true>(b),
                        unpack::<true>(b >> 16),
                    )
                } else {
                    (
                        unpack::<false>(a),
                        unpack::<false>(a >> 16),
                        unpack::<false>(b),
                        unpack::<false>(b >> 16),
                    )
                };
                sum += a0 * b0;
                sum += a1 * b1;
                d += S::LANES;
            }
        }
    }
    let mut distance = S::LANES / 2;
    while distance > 0 {
        sum += shuffle_xor_f32(sum, distance);
        distance /= 2;
    }
    if lane == 0 {
        unsafe {
            *delta.as_mut_ptr().add(row as usize) = sum * keep;
        }
    }
}

// Fixed-order FP32 sum over heads, rounded once for determinism.
#[kernel(f16, bf16)]
#[launch_bounds(256)]
pub fn backward_kv_merge<S: MergeSpec>(
    partial_k: &[f32],
    partial_v: &[f32],
    dk: DisjointSlice<u32>,
    dv: DisjointSlice<u32>,
    pairs: u64,
    heads_k: u32,
    groups: u32,
    dim_pairs: u32,
) {
    let mut dk = dk;
    let mut dv = dv;
    let mut i = thread::blockIdx_x() as u64 * 256 + thread::threadIdx_x() as u64;
    let step = thread::gridDim_x() as u64 * 256;
    while i < pairs {
        unsafe {
            merge_kv_pair::<S>(
                partial_k.as_ptr(),
                partial_v.as_ptr(),
                dk.as_mut_ptr(),
                dv.as_mut_ptr(),
                i,
                heads_k,
                groups,
                dim_pairs,
            );
        }
        i += step;
    }
}

// Increasing split order, rounded once, so the result is deterministic.
#[kernel(
    f16,
    bf16,
    f16_permuted,
    bf16_permuted,
    f16_permuted_d64,
    bf16_permuted_d64,
    f16_permuted_wide,
    bf16_permuted_wide,
    f16_permuted_causal,
    bf16_permuted_causal
)]
#[launch_bounds(256)]
pub fn backward_dq_merge<S: MergeSpec>(
    partial: &[f32],
    dq: DisjointSlice<u32>,
    pairs: u64,
    splits: u32,
    q_heads: u32,
    scale: f32,
) {
    let mut dq = dq;
    if S::D64 {
        unsafe {
            merge_d64_tile::<S>(
                partial.as_ptr(),
                dq.as_mut_ptr(),
                pairs,
                q_heads,
                scale,
                u64::MAX,
            );
        }
        return;
    }
    if S::PERMUTE {
        unsafe {
            merge_d128_tile::<S>(partial.as_ptr(), dq.as_mut_ptr(), pairs, q_heads, scale);
        }
        return;
    }
    let mut i = thread::blockIdx_x() as u64 * 256 + thread::threadIdx_x() as u64;
    let step = thread::gridDim_x() as u64 * 256;
    while i < pairs {
        unsafe {
            merge_q_pair::<S>(
                partial.as_ptr(),
                dq.as_mut_ptr(),
                i,
                pairs,
                splits,
                q_heads,
                scale,
            );
        }
        i += step;
    }
}

#[inline(always)]
unsafe fn merge_kv_pair<S: MergeSpec>(
    partial_k: *const f32,
    partial_v: *const f32,
    dk: *mut u32,
    dv: *mut u32,
    i: u64,
    _heads_k: u32,
    groups: u32,
    dim_pairs: u32,
) {
    let col = if dim_pairs.is_power_of_two() {
        i & (dim_pairs as u64 - 1)
    } else {
        i % dim_pairs as u64
    };
    // Flatten row/head together: the input inserts groups before the dimension.
    let base = ((i - col) * groups as u64 + col) * 2;
    let mut k0 = 0.0f32;
    let mut k1 = 0.0f32;
    let mut v0 = 0.0f32;
    let mut v1 = 0.0f32;
    let mut g = 0;
    while g < groups {
        let from = base as usize + g as usize * dim_pairs as usize * 2;
        unsafe {
            k0 += *partial_k.add(from);
            k1 += *partial_k.add(from + 1);
            v0 += *partial_v.add(from);
            v1 += *partial_v.add(from + 1);
        }
        g += 1;
    }
    unsafe {
        *dk.add(i as usize) = if S::BF16 {
            cvt::<true>(k0, k1)
        } else {
            cvt::<false>(k0, k1)
        };
        *dv.add(i as usize) = if S::BF16 {
            cvt::<true>(v0, v1)
        } else {
            cvt::<false>(v0, v1)
        };
    }
}
#[inline(always)]
unsafe fn merge_d64_tile<S: MergeSpec>(
    partial: *const f32,
    dq: *mut u32,
    pairs: u64,
    heads: u32,
    scale: f32,
    valid_rows: u64,
) {
    unsafe {
        let staging =
            cuda_device::shared::DynamicSharedArray::<u32, 16>::get() as usize as *mut u32;
        let tid = thread::threadIdx_x();
        let lane = tid % 32;
        let warp = tid / 32;
        let rows = if S::WIDE {
            16
        } else if S::CAUSAL {
            32
        } else {
            64
        };
        let lane_rows = if S::CAUSAL { 2 } else { 4 };
        let block_threads = if S::CAUSAL { 128 } else { 256 };
        let tile_pairs = rows * 32;
        let mut tile = thread::blockIdx_x() as u64;
        while tile < pairs / tile_pairs {
            let head = tile % heads as u64;
            let base = (tile / heads as u64 * rows * heads as u64 + head) * 64;
            let stride = heads as usize * 64;
            let row = if S::WIDE {
                lane / 4
            } else {
                (warp % lane_rows) * 16 + lane / 4
            };
            let mut d = 0;
            while d < if S::WIDE { 1 } else { 4 } {
                thread::__unroll_config::<0>();
                let source = partial.add(
                    base as usize
                        + (d * 4 * lane_rows + tid / 64) as usize * stride
                        + (tid % 64) as usize,
                );
                let x0 = *source * scale;
                let x1 = *source.add(lane_rows as usize * stride) * scale;
                let x2 = *source.add(2 * lane_rows as usize * stride) * scale;
                let x3 = *source.add(3 * lane_rows as usize * stride) * scale;
                let col = if S::WIDE {
                    warp * 4 + lane % 4
                } else {
                    (warp / lane_rows) * 16 + d * 4 + lane % 4
                };
                let lo = row * 32 + (col ^ ((row & 7) * 4));
                let hi = (row + 8) * 32 + (col ^ ((row & 7) * 4));
                if S::BF16 {
                    *staging.add(lo as usize) = cvt::<true>(x0, x1);
                    *staging.add(hi as usize) = cvt::<true>(x2, x3);
                } else {
                    *staging.add(lo as usize) = cvt::<false>(x0, x1);
                    *staging.add(hi as usize) = cvt::<false>(x2, x3);
                }
                d += 1;
            }
            thread::sync_threads();
            let mut i = tid;
            while i < tile_pairs as u32 {
                thread::__unroll_config::<0>();
                let row = i / 32;
                let col = i % 32;
                let value = *staging.add((row * 32 + (col ^ ((row & 7) * 4))) as usize);
                if tile / heads as u64 * rows + (row as u64) < valid_rows {
                    *dq.add(
                        base as usize / 2 + row as usize * heads as usize * 32 + col as usize,
                    ) = value;
                }
                i += block_threads;
            }
            thread::sync_threads();
            tile += thread::gridDim_x() as u64;
        }
    }
}

// Read the accumulator layout with contiguous warp loads, then transpose
// converted pairs in shared memory for contiguous logical output stores.
#[inline(always)]
unsafe fn merge_d128_tile<S: MergeSpec>(
    partial: *const f32,
    dq: *mut u32,
    pairs: u64,
    heads: u32,
    scale: f32,
) {
    unsafe {
        let staging =
            cuda_device::shared::DynamicSharedArray::<u32, 16>::get() as usize as *mut u32;
        let tid = thread::threadIdx_x();
        let lane = tid % 32;
        let warp = tid / 32;
        let mut tile = thread::blockIdx_x() as u64;
        while tile < pairs / 4096 {
            let head = tile % heads as u64;
            let base = (tile / heads as u64 * 64 * heads as u64 + head) * 128;
            let stride = heads as usize * 128;
            let mut d = 0;
            while d < 8 {
                thread::__unroll_config::<0>();
                let source = partial.add(
                    base as usize + (d * 8 + tid / 128) as usize * stride + (tid % 128) as usize,
                );
                let x0 = *source * scale;
                let x1 = *source.add(2 * stride) * scale;
                let x2 = *source.add(4 * stride) * scale;
                let x3 = *source.add(6 * stride) * scale;
                let row = (warp % 2) * 32 + lane / 4 + (d / 4) * 16;
                let col = (warp / 2) * 16 + (d % 4) * 4 + lane % 4;
                let lo = row * 64 + (col ^ ((row & 7) * 4));
                let hi = (row + 8) * 64 + (col ^ ((row & 7) * 4));
                *staging.add(lo as usize) = cvt::<false>(x0, x1);
                *staging.add(hi as usize) = cvt::<false>(x2, x3);
                if S::BF16 {
                    *staging.add(lo as usize) = cvt::<true>(x0, x1);
                    *staging.add(hi as usize) = cvt::<true>(x2, x3);
                }
                d += 1;
            }
            thread::sync_threads();
            let mut i = tid;
            while i < 4096 {
                thread::__unroll_config::<0>();
                let row = i / 64;
                let col = i % 64;
                let value = *staging.add((row * 64 + (col ^ ((row & 7) * 4))) as usize);
                *dq.add(base as usize / 2 + row as usize * heads as usize * 64 + col as usize) =
                    value;
                i += 256;
            }
            thread::sync_threads();
            tile += thread::gridDim_x() as u64;
        }
    }
}

#[inline(always)]
unsafe fn merge_q_pair<S: MergeSpec>(
    partial: *const f32,
    dq: *mut u32,
    i: u64,
    pairs: u64,
    splits: u32,
    q_heads: u32,
    scale: f32,
) {
    let mut x = 0.0f32;
    let mut y = 0.0f32;
    let mut split = 0;
    while split < splits {
        let from = ((split as u64 * pairs + i) * 2) as usize;
        unsafe {
            if S::PERMUTE {
                // The 64-query MHA kernel writes contiguous warp atomics to a
                // temporary layout; invert its (warp, lane, fragment) mapping.
                // Its dispatch guarantees full tiles and a single Q split.
                let index = i * 2;
                let col = index % 128;
                let head = index / 128 % q_heads as u64;
                let row = index / (128 * q_heads as u64);
                let r = row % 64;
                let warp = r / 32 + 2 * (col / 32);
                let lane = (r % 8) * 4 + (col % 8) / 2;
                let d = (r % 32) / 16 * 4 + (col % 32) / 8;
                let e = (r % 16) / 8 * 2;
                let temp_row = row / 64 * 64 + (d * 4 + e) * 2 + warp / 4;
                let temp_col = (warp % 4) * 32 + lane;
                let offset = ((temp_row * q_heads as u64 + head) * 128 + temp_col) as usize;
                x += *partial.add(offset);
                y += *partial.add(offset + 2 * q_heads as usize * 128);
            } else {
                x += *partial.add(from);
                y += *partial.add(from + 1);
            }
        }
        split += 1;
    }
    unsafe {
        *dq.add(i as usize) = if S::BF16 {
            cvt::<true>(x * scale, y * scale)
        } else {
            cvt::<false>(x * scale, y * scale)
        };
    }
}

// Merge all fused-backward outputs in one launch, preserving each sum's order.
#[kernel(f16, bf16)]
#[launch_bounds(256)]
pub fn backward_grad_merge<S: MergeSpec>(
    partial_k: &[f32],
    partial_v: &[f32],
    partial_q: &[f32],
    dk: DisjointSlice<u32>,
    dv: DisjointSlice<u32>,
    dq: DisjointSlice<u32>,
    kv_pairs: u64,
    q_pairs: u64,
    heads_k: u32,
    groups: u32,
    dim_pairs: u32,
    q_splits: u32,
    scale: f32,
) {
    let (mut dk, mut dv, mut dq) = (dk, dv, dq);
    let mut i = thread::blockIdx_x() as u64 * 256 + thread::threadIdx_x() as u64;
    let step = thread::gridDim_x() as u64 * 256;
    while i < kv_pairs.max(q_pairs) {
        unsafe {
            if i < kv_pairs {
                merge_kv_pair::<S>(
                    partial_k.as_ptr(),
                    partial_v.as_ptr(),
                    dk.as_mut_ptr(),
                    dv.as_mut_ptr(),
                    i,
                    heads_k,
                    groups,
                    dim_pairs,
                );
            }
            if i < q_pairs {
                merge_q_pair::<S>(
                    partial_q.as_ptr(),
                    dq.as_mut_ptr(),
                    i,
                    q_pairs,
                    q_splits,
                    0,
                    scale,
                );
            }
        }
        i += step;
    }
}

// Each packed sequence reserves one extra accumulator tile, allowing its final
// partial tile to use the same coalesced layout as every full tile.
#[kernel(
    f16_permuted_causal,
    bf16_permuted_causal,
    f16_permuted_d64,
    bf16_permuted_d64,
    f16_permuted_wide,
    bf16_permuted_wide
)]
#[launch_bounds(256)]
pub fn backward_packed_dq_merge<S: MergeSpec>(
    partial: &[f32],
    output: DisjointSlice<u32>,
    cuq: &[i32],
    heads: u32,
    scale: f32,
) {
    let mut output = output;
    unsafe {
        let batch = thread::blockIdx_y() as usize;
        let start = *cuq.as_ptr().add(batch) as usize;
        let rows = (*cuq.as_ptr().add(batch + 1) as usize) - start;
        let tile_rows = if S::WIDE {
            16
        } else if S::CAUSAL {
            32
        } else {
            64
        };
        let pairs = rows.div_ceil(tile_rows) as u64 * tile_rows as u64 * heads as u64 * 32;
        let input = partial
            .as_ptr()
            .add((start + batch * tile_rows) * heads as usize * 64);
        let dst = output.as_mut_ptr().add(start * heads as usize * 32);
        merge_d64_tile::<S>(input, dst, pairs, heads, scale, rows as u64);
    }
}
