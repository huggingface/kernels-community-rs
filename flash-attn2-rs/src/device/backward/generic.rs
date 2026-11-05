//! Head-dimension-generic fused backward in the FlashAttention v2 layout.
//!
//! Eight warps own one key tile and walk BM-row query blocks. S and dP are
//! tiled BM/16 (query rows) x 8/(BM/16) (keys); P and dS go through padded shared memory
//! so dK/dV can be tiled across the head dimension instead, which keeps
//! D192/D256 accumulators in registers. dQ = dS K is added to the FP32
//! accumulator with relaxed reductions while the next Q/dO tile streams in.
use super::common::probability_gradient;
use super::fused::FusedSpec;
use crate::device::config::{FwdConfig, Strides};
use crate::device::dropout::Dropout;
use crate::device::forward::LOG2_E;
use crate::device::inlined::{cvt, mma, shared_address};
use cuda_device::async_copy::cp_async_ca_zfill_16;
use cuda_device::shared::DynamicSharedArray;
use cuda_device::wmma::{
    ldmatrix_x2_trans_shared_u32 as ldmatrix_x2_trans_shared,
    ldmatrix_x4_shared_u32 as ldmatrix_x4_shared,
    ldmatrix_x4_trans_shared_u32 as ldmatrix_x4_trans_shared,
};
use cuda_device::{DisjointSlice, kernel, launch_bounds, ptx_asm, thread};

#[inline(always)]
unsafe fn copy_tile<const HD: usize, const ROWS: usize, const WARPS: usize>(
    src: *const u32,
    dst: *mut u32,
    base: usize,
    row_start: u32,
    stride: u32,
    seq: u32,
    dim_pairs: u32,
    tid: u32,
    row_words: usize,
) {
    unsafe {
        // Full tiles whose rows split evenly across the threads skip the
        // per-chunk bounds checks and step fixed pointer offsets (D32 measured
        // faster with the checked loop).
        let per_row = HD / 8;
        let step = WARPS * 32 / per_row;
        if HD >= 64
            && (WARPS * 32) % per_row == 0
            && ROWS % step == 0
            && row_start as usize + ROWS <= seq as usize
            && dim_pairs as usize == HD / 2
        {
            let row = tid as usize / per_row;
            let col = (tid as usize % per_row) * 4;
            let mut from = src.add(base + (row_start as usize + row) * stride as usize + col);
            let mut to = dst.add(row * row_words + col);
            let mut r = 0;
            while r < ROWS {
                thread::__unroll_config::<0>();
                cp_async_ca_zfill_16(to, from as *const u8, 16);
                from = from.add(step * stride as usize);
                to = to.add(step * row_words);
                r += step;
            }
            return;
        }
        let mut i = tid as usize * 4;
        while i < ROWS * (HD / 2) {
            let row = i / (HD / 2);
            let col = i % (HD / 2);
            let global_row = row_start as usize + row;
            let valid = global_row < seq as usize && col < dim_pairs as usize;
            let index = if valid {
                base + global_row * stride as usize + col
            } else {
                0
            };
            cp_async_ca_zfill_16(
                dst.add(row * row_words + col),
                src.add(index) as *const u8,
                if valid { 16 } else { 0 },
            );
            i += WARPS * 128;
        }
    }
}

/// (key tile, head, batch) of this CTA. Causal key tiles near the start see
/// the most queries, so `heavy_first` launches them first for every head
/// instead of letting them trail the wave.
#[inline(always)]
pub(crate) fn grid_coords(heavy_first: bool) -> (u32, u32, u32) {
    if heavy_first {
        let slots = thread::gridDim_y() * thread::gridDim_z();
        let linear = thread::blockIdx_x()
            + thread::gridDim_x()
                * (thread::blockIdx_y() + thread::gridDim_y() * thread::blockIdx_z());
        let slot = linear % slots;
        (
            linear / slots,
            slot % thread::gridDim_y(),
            slot / thread::gridDim_y(),
        )
    } else {
        (
            thread::blockIdx_x(),
            thread::blockIdx_y(),
            thread::blockIdx_z(),
        )
    }
}

/// exp2 flushing denormals, as the reference does; avoids the fix-up the
/// non-ftz form needs around each MUFU.
#[inline(always)]
fn exp2_ftz(x: f32) -> f32 {
    let y: f32;
    unsafe {
        ptx_asm!("ex2.approx.ftz.f32 %0, %1;", out("=f") y, in("f") x, options(register_only));
    }
    y
}

#[inline(always)]
unsafe fn reduce_add(ptr: *mut f32, value: f32) {
    unsafe {
        ptx_asm!("red.relaxed.gpu.global.add.f32 [%0], %1;", in("l") ptr as u64, in("f") value, clobber("memory"));
    }
}

#[inline(always)]
unsafe fn store_pair<const BF16: bool, const PARTIAL: bool>(
    output: *mut u32,
    index: usize,
    a: f32,
    b: f32,
) {
    unsafe {
        if PARTIAL {
            let partial = output.cast::<f32>().add(index * 2);
            *partial = a;
            *partial.add(1) = b;
        } else {
            *output.add(index) = cvt::<BF16>(a, b);
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn core<
    const BF16: bool,
    const DROP: bool,
    const PARTIAL: bool,
    const HD: usize,
    const BM: usize,
    const BN: usize,
    const SM: usize,
    const VREG: bool,
    const KS: usize,
    const NT: usize,
    const KVN: usize,
    const QN: usize,
    const VK: usize,
    const DB: bool,
    const WARPS: usize,
>(
    q: *const u32,
    k: *const u32,
    v: *const u32,
    dout: *const u32,
    lse: *const f32,
    delta: *const f32,
    dk: *mut u32,
    dv: *mut u32,
    dq: *mut f32,
    alibi: *const f32,
    cfg: FwdConfig,
    do_strides: Strides,
    gradient_scale: f32,
    dropout: Dropout,
    packed: Option<(u32, u32)>,
) {
    unsafe {
        let rs = HD / 2 + 4;
        let ps = BN / 2 + 4;
        let tid = thread::threadIdx_x();
        let lane = tid % 32;
        let warp = tid / 32;
        let group = lane >> 2;
        let pair = lane & 3;
        // Packed callers offset every pointer to their sequence, so batch
        // strides vanish and LSE/delta use the packed row strides.
        // D256 measured slower with the reordered grid.
        let (key_block, grid_head, grid_batch) = grid_coords(HD != 256 && cfg.window_right >= 0);
        // CTAs launched 128 apart share an SM on the 4090. Give each such pair
        // adjacent key tiles of one head so they walk the same Q/dO rows and
        // the second one hits in L1. Other SM counts only lose the locality.
        // D32 has too little compute to hide lockstep dQ atomics on shared rows:
        // pairing there made run times swing by up to 30%.
        let pair_map = packed.is_none()
            && cfg.window_left < 0
            && cfg.window_right < 0
            && BN == 64
            && HD > 32
            && HD <= 128
            && thread::gridDim_x() % 2 == 0
            && (thread::gridDim_x() * thread::gridDim_y() * thread::gridDim_z()) % 256 == 0;
        let (key_block, grid_head, grid_batch) = if pair_map {
            let linear = thread::blockIdx_x()
                + thread::gridDim_x()
                    * (thread::blockIdx_y() + thread::gridDim_y() * thread::blockIdx_z());
            let wave = linear / 256;
            let half = (linear % 256) / 128;
            let pair = wave * 128 + linear % 128;
            let key_pairs = thread::gridDim_x() / 2;
            let slot = pair / key_pairs;
            (
                (pair % key_pairs) * 2 + half,
                slot % thread::gridDim_y(),
                slot / thread::gridDim_y(),
            )
        } else {
            (key_block, grid_head, grid_batch)
        };
        let bh = grid_batch * cfg.h + grid_head;
        let b = if packed.is_some() { 0 } else { grid_batch };
        let head = grid_head;
        let kh = head / cfg.gqa;
        let row_head = (b * cfg.h + head) as usize;
        let key_start = key_block * BN as u32;

        let scratch = DynamicSharedArray::<u32, 16>::get() as usize as *mut u32;
        let q_s = scratch;
        let do_s = q_s.add(BM * rs);
        let k_s = do_s.add(BM * rs);
        let v_s = k_s.add(BN * rs);
        let p_s = if VREG { v_s } else { v_s.add(BN * rs) };
        let ds_s = p_s.add(BM * ps);
        // Double-buffered Q/dO follow the other tiles.
        let end_words = if VREG {
            (BN * rs).max(2 * BM * ps)
        } else {
            BN * rs + 2 * BM * ps
        };
        let q_s2 = v_s.add(end_words);
        let k_a = shared_address(k_s);
        let v_a = shared_address(v_s);
        let p_a = shared_address(p_s);
        let ds_a = shared_address(ds_s);

        let k_base = b as usize * cfg.k.batch as usize + kh as usize * cfg.k.head as usize;
        let v_base = b as usize * cfg.v.batch as usize + kh as usize * cfg.v.head as usize;
        let q_base = b as usize * cfg.q.batch as usize + head as usize * cfg.q.head as usize;
        let do_base =
            b as usize * do_strides.batch as usize + head as usize * do_strides.head as usize;
        copy_tile::<HD, BN, WARPS>(
            v,
            v_s,
            v_base,
            key_start,
            cfg.v.seq,
            cfg.seqlen_k,
            cfg.hs_u32,
            tid,
            rs,
        );
        // With V bound for registers, commit it alone so its fragments load
        // while K, Q and dO are still in flight, as the reference does.
        if VREG {
            ptx_asm!("cp.async.commit_group;", clobber("memory"));
        }
        copy_tile::<HD, BN, WARPS>(
            k,
            k_s,
            k_base,
            key_start,
            cfg.k.seq,
            cfg.seqlen_k,
            cfg.hs_u32,
            tid,
            rs,
        );

        // Query blocks that can attend to this key tile.
        let off = cfg.seqlen_k as i64 - cfg.seqlen_q as i64;
        let first = if cfg.window_right >= 0 {
            (key_start as i64 - off - cfg.window_right as i64).clamp(0, cfg.seqlen_q as i64) as u32
                / BM as u32
        } else {
            0
        };
        let end = if cfg.window_left >= 0 {
            ((key_start + BN as u32).min(cfg.seqlen_k) as i64 - off + cfg.window_left as i64)
                .clamp(0, cfg.seqlen_q as i64) as u32
        } else {
            cfg.seqlen_q
        }
        .div_ceil(BM as u32);
        // Rotating the start keeps concurrent tiles off the same dQ rows. D256
        // does this for causal tiles too, whose ranges differ per key tile.
        let count = end.saturating_sub(first);
        let shift = if cfg.window_left < 0 && (cfg.window_right < 0 || HD == 256) && count > 0 {
            (if pair_map { key_block / 2 } else { key_block }) % count
        } else {
            0
        };
        let rotate = |step: u32| {
            let i = step + shift;
            first + if i >= count { i - count } else { i }
        };
        if first < end {
            let start = rotate(0) * BM as u32;
            copy_tile::<HD, BM, WARPS>(
                q,
                q_s,
                q_base,
                start,
                cfg.q.seq,
                cfg.seqlen_q,
                cfg.hs_u32,
                tid,
                rs,
            );
            copy_tile::<HD, BM, WARPS>(
                dout,
                do_s,
                do_base,
                start,
                do_strides.seq,
                cfg.seqlen_q,
                cfg.hs_u32,
                tid,
                rs,
            );
        }
        ptx_asm!("cp.async.commit_group;", clobber("memory"));
        if VREG {
            ptx_asm!("cp.async.wait_group 1;", clobber("memory"));
        } else {
            ptx_asm!("cp.async.wait_group 0;", clobber("memory"));
        }
        thread::sync_threads();

        // Score tiles: 16 query rows x BN/2 keys per warp.
        let row_warps = BM as u32 / (16 * SM as u32);
        let wm = warp % row_warps;
        let wn = warp / row_warps;
        let score_row = wm * 16 * SM as u32;
        let score_key = wn * (BN as u32 / (WARPS as u32 / row_warps));
        // dK/dV tiles: 32 keys x KVN*8 head columns per warp.
        let kv_groups = WARPS as u32 / (BN as u32 / 32);
        let kg = warp / kv_groups;
        let hg = warp % kv_groups;
        // dQ tiles: 32 query rows x QN*8 head columns per warp.
        let q_cols = WARPS as u32 / (BM as u32 / 32);
        let qm = warp / q_cols;
        let qh = warp % q_cols;
        // Per-lane byte offsets; unrolled loads add compile-time immediates.
        let (rs4, ps4) = (rs as u32 * 4, ps as u32 * 4);
        let score_a = (score_row + lane % 16) * rs4 + (lane / 16) * 16;
        let score_b = (score_key + (lane / 16) * 8 + lane % 8) * rs4 + ((lane / 8) % 2) * 16;
        let mat = lane / 8;
        let staged_t = ((mat / 2) * 8 + lane % 8) * ps4 + (kg * 32 + (mat % 2) * 8) * 2;
        let kv_b = (lane % 16) * rs4 + hg * KVN as u32 * 16;
        let dq_a = (qm * 32 + lane % 16) * ps4 + (lane / 16) * 16;
        let dq_b = (lane % 16) * rs4 + qh * QN as u32 * 16;
        let store = (score_row + group) * ps as u32 + score_key / 2 + pair;

        let mut v_frag = [[[0u32; 2]; NT]; VK];
        if VREG {
            let mut kk = 0;
            while kk < KS {
                thread::__unroll_config::<0>();
                let mut j = 0;
                while j < NT {
                    thread::__unroll_config::<0>();
                    let row = score_key + j as u32 * 8 + (lane / 16) * 8 + lane % 8;
                    let r = ldmatrix_x4_shared(
                        v_a + (row * rs as u32 + kk as u32 * 8 + ((lane / 8) % 2) * 4) * 4,
                    );
                    v_frag[kk][j] = [r[0], r[1]];
                    v_frag[kk][j + 1] = [r[2], r[3]];
                    j += 2;
                }
                kk += 1;
            }
            // P/dS reuse V's shared memory; the other tiles land here.
            ptx_asm!("cp.async.wait_group 0;", clobber("memory"));
            thread::sync_threads();
        }

        let slope = if cfg.has_alibi != 0 {
            *alibi.add(b as usize * cfg.alibi_bs as usize + head as usize)
        } else {
            0.0
        };
        let rounded_q = cfg.seqlen_q.div_ceil(128) * 128;
        let (lse_stride, delta_stride) = packed.unwrap_or((cfg.seqlen_q, rounded_q));
        let mut acc_k = [[[0.0f32; 4]; KVN]; 2];
        let mut acc_v = [[[0.0f32; 4]; KVN]; 2];
        // Full Q/dO tiles copy from per-thread pointers computed once, adding
        // one row offset per block instead of per-chunk addressing and checks.
        let per_row = HD / 8;
        let pass = WARPS * 32 / per_row;
        let fast_copy = HD >= 64
            && (WARPS * 32) % per_row == 0
            && BM % pass == 0
            && cfg.hs_u32 as usize == HD / 2;
        let thread_row = tid as usize / per_row;
        let thread_col = (tid as usize % per_row) * 4;
        let q_from = q.add(q_base + thread_row * cfg.q.seq as usize + thread_col);
        let do_from = dout.add(do_base + thread_row * do_strides.seq as usize + thread_col);
        let thread_to = thread_row * rs + thread_col;
        // Rows whose chunks do not split evenly across threads (D96, D192)
        // still give each thread a fixed set of up to four chunks per tile.
        let threads = WARPS * 32;
        let chunks = BM * per_row / threads;
        let spread_copy = !fast_copy
            && HD >= 64
            && (BM * per_row) % threads == 0
            && chunks <= 4
            && cfg.hs_u32 as usize == HD / 2;
        let mut q_offsets = [0usize; 4];
        let mut do_offsets = [0usize; 4];
        let mut to_offsets = [0usize; 4];
        let mut k = 0;
        while k < 4 {
            thread::__unroll_config::<0>();
            let i = tid as usize + k * threads;
            let (row, col) = (i / per_row, (i % per_row) * 4);
            q_offsets[k] = row * cfg.q.seq as usize + col;
            do_offsets[k] = row * do_strides.seq as usize + col;
            to_offsets[k] = row * rs + col;
            k += 1;
        }
        let copy = |start: u32,
                    to: *mut u32,
                    from: *const u32,
                    src: *const u32,
                    base: usize,
                    stride: u32,
                    offsets: &[usize; 4]| {
            if spread_copy && start + BM as u32 <= cfg.seqlen_q {
                let from = src.add(base + start as usize * stride as usize);
                let mut k = 0;
                while k < 4 {
                    thread::__unroll_config::<0>();
                    if k < chunks {
                        cp_async_ca_zfill_16(
                            to.add(to_offsets[k]),
                            from.add(offsets[k]) as *const u8,
                            16,
                        );
                    }
                    k += 1;
                }
            } else if fast_copy && start + BM as u32 <= cfg.seqlen_q {
                let mut from = from.add(start as usize * stride as usize);
                let mut to = to.add(thread_to);
                let mut r = 0;
                while r < BM {
                    thread::__unroll_config::<0>();
                    cp_async_ca_zfill_16(to, from as *const u8, 16);
                    from = from.add(pass * stride as usize);
                    to = to.add(pass * rs);
                    r += pass;
                }
            } else {
                copy_tile::<HD, BM, WARPS>(
                    src,
                    to,
                    base,
                    start,
                    stride,
                    cfg.seqlen_q,
                    cfg.hs_u32,
                    tid,
                    rs,
                );
            }
        };
        let stream_q =
            |start: u32, to: *mut u32| copy(start, to, q_from, q, q_base, cfg.q.seq, &q_offsets);
        let stream_do = |start: u32, to: *mut u32| {
            copy(
                start,
                to,
                do_from,
                dout,
                do_base,
                do_strides.seq,
                &do_offsets,
            )
        };
        macro_rules! load_lses {
            ($start:expr, $to:ident) => {
                let mut mt = 0;
                while mt < SM {
                    thread::__unroll_config::<0>();
                    let q0 = $start + score_row + mt as u32 * 16 + group;
                    let row_lse = |row: u32| {
                        if row < cfg.seqlen_q {
                            *lse.add(row_head * lse_stride as usize + row as usize)
                        } else {
                            f32::INFINITY
                        }
                    };
                    $to[mt] = [row_lse(q0), row_lse(q0 + 8)];
                    mt += 1;
                }
            };
        }
        macro_rules! load_deltas {
            ($start:expr, $to:ident) => {
                let mut mt = 0;
                while mt < SM {
                    thread::__unroll_config::<0>();
                    let q0 = $start + score_row + mt as u32 * 16 + group;
                    $to[mt] = [
                        *delta.add(row_head * delta_stride as usize + q0 as usize),
                        *delta.add(row_head * delta_stride as usize + q0 as usize + 8),
                    ];
                    mt += 1;
                }
            };
        }
        let mut next_lses = [[0.0f32; 2]; SM];
        if DB && count > 0 {
            load_lses!(rotate(0) * BM as u32, next_lses);
        }
        let mut step = 0;
        while step < count {
            let block = rotate(step);
            let query_start = block * BM as u32;
            let second = DB && step % 2 == 1;
            let (q_cur, do_cur) = if second { (q_s2, do_s) } else { (q_s, do_s) };
            let q_a = shared_address(q_cur);
            let do_a = shared_address(do_cur);
            // Row statistics load ahead of the S product, or for DB around it:
            // LSE an iteration early and delta behind the S product.
            let mut rows = [[0u32; 2]; SM];
            let mut lses = next_lses;
            let mut deltas = [[0.0f32; 2]; SM];
            let mut mt = 0;
            while mt < SM {
                thread::__unroll_config::<0>();
                let q0 = query_start + score_row + mt as u32 * 16 + group;
                rows[mt] = [q0, q0 + 8];
                mt += 1;
            }
            if !DB {
                load_lses!(query_start, lses);
                load_deltas!(query_start, deltas);
            }
            // S = Q K^T first; its exponentials then overlap the dP = dO V^T
            // products, as in the reference.
            let mut s = [[[0.0f32; 4]; NT]; SM];
            let mut kk = 0;
            while kk < KS {
                thread::__unroll_config::<0>();
                let mut a_q = [[0u32; 4]; SM];
                let mut mt = 0;
                while mt < SM {
                    thread::__unroll_config::<0>();
                    a_q[mt] =
                        ldmatrix_x4_shared(q_a + score_a + mt as u32 * 16 * rs4 + kk as u32 * 32);
                    mt += 1;
                }
                let mut j = 0;
                while j < NT {
                    thread::__unroll_config::<0>();
                    let bk =
                        ldmatrix_x4_shared(k_a + score_b + j as u32 * 8 * rs4 + kk as u32 * 32);
                    let mut mt = 0;
                    while mt < SM {
                        thread::__unroll_config::<0>();
                        s[mt][j] = mma::<BF16>(s[mt][j], a_q[mt], [bk[0], bk[1]]);
                        s[mt][j + 1] = mma::<BF16>(s[mt][j + 1], a_q[mt], [bk[2], bk[3]]);
                        mt += 1;
                    }
                    j += 2;
                }
                kk += 1;
            }
            if DB {
                load_deltas!(query_start, deltas);
            }
            // Tiles fully inside both sequences and below any causal diagonal
            // need no per-element masking.
            let unmasked = query_start + BM as u32 <= cfg.seqlen_q
                && key_start + BN as u32 <= cfg.seqlen_k
                && cfg.has_alibi == 0
                && cfg.window_left < 0
                && (cfg.window_right < 0
                    || (cfg.window_right == 0
                        && (key_start + BN as u32) as i64 <= query_start as i64 + off + 1));
            let scale = cfg.scale * LOG2_E;
            // Probabilities stay in registers until dS is known.
            let mut probs = [[[0.0f32; 4]; NT]; SM];
            macro_rules! probabilities {
                ($fast:expr) => {
                    let mut mt = 0;
                    while mt < SM {
                        thread::__unroll_config::<0>();
                        let [q0, q1] = rows[mt];
                        let [l0, l1] = lses[mt];
                        let (e0, e1) = (l0 * LOG2_E, l1 * LOG2_E);
                        let mut nt = 0;
                        while nt < NT {
                            thread::__unroll_config::<0>();
                            let key = key_start + score_key + nt as u32 * 8 + pair * 2;
                            let f = s[mt][nt];
                            probs[mt][nt] = if $fast {
                                [
                                    exp2_ftz(f[0] * scale - e0),
                                    exp2_ftz(f[1] * scale - e0),
                                    exp2_ftz(f[2] * scale - e1),
                                    exp2_ftz(f[3] * scale - e1),
                                ]
                            } else {
                                [
                                    probability_gradient(f[0], q0, key, l0, &cfg, slope, false),
                                    probability_gradient(f[1], q0, key + 1, l0, &cfg, slope, false),
                                    probability_gradient(f[2], q1, key, l1, &cfg, slope, false),
                                    probability_gradient(f[3], q1, key + 1, l1, &cfg, slope, false),
                                ]
                            };
                            nt += 1;
                        }
                        mt += 1;
                    }
                };
            }
            // Full tiles take a straight-line body with no per-element masking.
            if unmasked {
                probabilities!(true);
            } else {
                probabilities!(false);
            }
            let mut dp = [[[0.0f32; 4]; NT]; SM];
            let mut kk = 0;
            while kk < KS {
                thread::__unroll_config::<0>();
                let mut a_o = [[0u32; 4]; SM];
                let mut mt = 0;
                while mt < SM {
                    thread::__unroll_config::<0>();
                    a_o[mt] =
                        ldmatrix_x4_shared(do_a + score_a + mt as u32 * 16 * rs4 + kk as u32 * 32);
                    mt += 1;
                }
                let mut j = 0;
                while j < NT {
                    thread::__unroll_config::<0>();
                    let bv = if VREG {
                        [
                            v_frag[kk][j][0],
                            v_frag[kk][j][1],
                            v_frag[kk][j + 1][0],
                            v_frag[kk][j + 1][1],
                        ]
                    } else {
                        ldmatrix_x4_shared(v_a + score_b + j as u32 * 8 * rs4 + kk as u32 * 32)
                    };
                    let mut mt = 0;
                    while mt < SM {
                        thread::__unroll_config::<0>();
                        dp[mt][j] = mma::<BF16>(dp[mt][j], a_o[mt], [bv[0], bv[1]]);
                        dp[mt][j + 1] = mma::<BF16>(dp[mt][j + 1], a_o[mt], [bv[2], bv[3]]);
                        mt += 1;
                    }
                    j += 2;
                }
                kk += 1;
            }
            let mut mt = 0;
            while mt < SM {
                thread::__unroll_config::<0>();
                let [q0, _] = rows[mt];
                let [d0, d1] = deltas[mt];
                // One Philox call covers 16 rows x 32 keys.
                let mut random = [0u32; 4];
                let mut nt = 0;
                while nt < NT {
                    thread::__unroll_config::<0>();
                    let local = score_key + nt as u32 * 8;
                    let p = probs[mt][nt];
                    if DROP && local % 32 == score_key % 32 {
                        random = dropout.random(
                            (q0 - group) / 16,
                            (key_start + local) / 32,
                            dropout.offset.wrapping_add(bh as u64 * 32 + lane as u64),
                        );
                    }
                    let mask = if DROP {
                        let word = super::common::random_word(random, (local % 32) / 8);
                        [
                            ((word & 255) <= dropout.threshold) as u32 as f32,
                            ((word >> 8 & 255) <= dropout.threshold) as u32 as f32,
                            ((word >> 16 & 255) <= dropout.threshold) as u32 as f32,
                            ((word >> 24) <= dropout.threshold) as u32 as f32,
                        ]
                    } else {
                        [1.0f32; 4]
                    };
                    let x = dp[mt][nt];
                    let ds = [
                        p[0] * (x[0] * mask[0] - d0),
                        p[1] * (x[1] * mask[1] - d0),
                        p[2] * (x[2] * mask[2] - d1),
                        p[3] * (x[3] * mask[3] - d1),
                    ];
                    let at = (store + mt as u32 * 16 * ps as u32 + nt as u32 * 4) as usize;
                    let below = at + 8 * ps;
                    *p_s.add(at) = cvt::<BF16>(p[0] * mask[0], p[1] * mask[1]);
                    *p_s.add(below) = cvt::<BF16>(p[2] * mask[2], p[3] * mask[3]);
                    *ds_s.add(at) = cvt::<BF16>(ds[0], ds[1]);
                    *ds_s.add(below) = cvt::<BF16>(ds[2], ds[3]);
                    nt += 1;
                }
                mt += 1;
            }
            // DB pipelines as the reference does: issuing the next Q here keeps it
            // clear of the S product's shared-memory burst.
            if DB && step + 1 < count {
                stream_q(
                    rotate(step + 1) * BM as u32,
                    if second { q_s } else { q_s2 },
                );
            }
            thread::sync_threads();

            // acc += A^T B over this block's queries, A staged at `staged` (P or
            // dS) and B the Q or dO tile at `tile`.
            macro_rules! key_product {
                ($acc:ident, $staged:expr, $tile:expr) => {
                    let mut kq = 0;
                    while kq < BM {
                        thread::__unroll_config::<0>();
                        let mut a = [[0u32; 4]; 2];
                        let mut mt = 0;
                        while mt < 2 {
                            thread::__unroll_config::<0>();
                            a[mt] = ldmatrix_x4_trans_shared(
                                $staged + staged_t + kq as u32 * ps4 + mt as u32 * 32,
                            );
                            mt += 1;
                        }
                        let mut n = 0;
                        while n + 1 < KVN {
                            thread::__unroll_config::<0>();
                            let b = ldmatrix_x4_trans_shared(
                                $tile + kv_b + kq as u32 * rs4 + n as u32 * 16 + (lane / 16) * 16,
                            );
                            let mut mt = 0;
                            while mt < 2 {
                                thread::__unroll_config::<0>();
                                $acc[mt][n] = mma::<BF16>($acc[mt][n], a[mt], [b[0], b[1]]);
                                $acc[mt][n + 1] = mma::<BF16>($acc[mt][n + 1], a[mt], [b[2], b[3]]);
                                mt += 1;
                            }
                            n += 2;
                        }
                        if KVN % 2 == 1 {
                            let n = KVN - 1;
                            let b = ldmatrix_x2_trans_shared(
                                $tile + kv_b + kq as u32 * rs4 + n as u32 * 16,
                            );
                            let mut mt = 0;
                            while mt < 2 {
                                thread::__unroll_config::<0>();
                                $acc[mt][n] = mma::<BF16>($acc[mt][n], a[mt], b);
                                mt += 1;
                            }
                        }
                        kq += 16;
                    }
                };
            }
            if DB {
                key_product!(acc_v, p_a, do_a);
            } else {
                // dV += P^T dO and dK += dS^T Q.
                let mut kq = 0;
                while kq < BM {
                    thread::__unroll_config::<0>();
                    let mut a_p = [[0u32; 4]; 2];
                    let mut a_ds = [[0u32; 4]; 2];
                    let mut mt = 0;
                    while mt < 2 {
                        thread::__unroll_config::<0>();
                        let offset = staged_t + kq as u32 * ps4 + mt as u32 * 32;
                        a_p[mt] = ldmatrix_x4_trans_shared(p_a + offset);
                        a_ds[mt] = ldmatrix_x4_trans_shared(ds_a + offset);
                        mt += 1;
                    }
                    let mut n = 0;
                    while n + 1 < KVN {
                        thread::__unroll_config::<0>();
                        let offset = kv_b + kq as u32 * rs4 + n as u32 * 16 + (lane / 16) * 16;
                        let bo = ldmatrix_x4_trans_shared(do_a + offset);
                        let bq = ldmatrix_x4_trans_shared(q_a + offset);
                        let mut mt = 0;
                        while mt < 2 {
                            thread::__unroll_config::<0>();
                            acc_v[mt][n] = mma::<BF16>(acc_v[mt][n], a_p[mt], [bo[0], bo[1]]);
                            acc_v[mt][n + 1] =
                                mma::<BF16>(acc_v[mt][n + 1], a_p[mt], [bo[2], bo[3]]);
                            acc_k[mt][n] = mma::<BF16>(acc_k[mt][n], a_ds[mt], [bq[0], bq[1]]);
                            acc_k[mt][n + 1] =
                                mma::<BF16>(acc_k[mt][n + 1], a_ds[mt], [bq[2], bq[3]]);
                            mt += 1;
                        }
                        n += 2;
                    }
                    if KVN % 2 == 1 {
                        let n = KVN - 1;
                        let offset = kv_b + kq as u32 * rs4 + n as u32 * 16;
                        let bo = ldmatrix_x2_trans_shared(do_a + offset);
                        let bq = ldmatrix_x2_trans_shared(q_a + offset);
                        let mut mt = 0;
                        while mt < 2 {
                            thread::__unroll_config::<0>();
                            acc_v[mt][n] = mma::<BF16>(acc_v[mt][n], a_p[mt], bo);
                            acc_k[mt][n] = mma::<BF16>(acc_k[mt][n], a_ds[mt], bq);
                            mt += 1;
                        }
                    }
                    kq += 16;
                }
            }
            thread::sync_threads();

            // Q and dO are free: stream the next block during dQ.
            if step + 1 < count {
                let start = rotate(step + 1) * BM as u32;
                if !DB {
                    stream_q(start, q_s);
                }
                if HD != 256 {
                    stream_do(start, do_s);
                }
            }
            if HD != 256 {
                ptx_asm!("cp.async.commit_group;", clobber("memory"));
            }
            if DB && step + 1 < count {
                load_lses!(rotate(step + 1) * BM as u32, next_lses);
            }

            // dQ += dS K.
            let mut acc_q = [[[0.0f32; 4]; QN]; 2];
            let mut kk = 0;
            while kk < BN {
                thread::__unroll_config::<0>();
                let mut a = [[0u32; 4]; 2];
                let mut mt = 0;
                while mt < 2 {
                    thread::__unroll_config::<0>();
                    a[mt] = ldmatrix_x4_shared(ds_a + dq_a + mt as u32 * 16 * ps4 + kk as u32 * 2);
                    mt += 1;
                }
                let mut n = 0;
                while n + 1 < QN {
                    thread::__unroll_config::<0>();
                    let bk = ldmatrix_x4_trans_shared(
                        k_a + dq_b + kk as u32 * rs4 + n as u32 * 16 + (lane / 16) * 16,
                    );
                    let mut mt = 0;
                    while mt < 2 {
                        thread::__unroll_config::<0>();
                        acc_q[mt][n] = mma::<BF16>(acc_q[mt][n], a[mt], [bk[0], bk[1]]);
                        acc_q[mt][n + 1] = mma::<BF16>(acc_q[mt][n + 1], a[mt], [bk[2], bk[3]]);
                        mt += 1;
                    }
                    n += 2;
                }
                if QN % 2 == 1 {
                    let n = QN - 1;
                    let bk = ldmatrix_x2_trans_shared(k_a + dq_b + kk as u32 * rs4 + n as u32 * 16);
                    let mut mt = 0;
                    while mt < 2 {
                        thread::__unroll_config::<0>();
                        acc_q[mt][n] = mma::<BF16>(acc_q[mt][n], a[mt], bk);
                        mt += 1;
                    }
                }
                // D256's two copy bursts would stall the first dQ fragments.
                if HD == 256 && kk == 0 {
                    if step + 1 < count {
                        stream_do(rotate(step + 1) * BM as u32, do_s);
                    }
                    ptx_asm!("cp.async.commit_group;", clobber("memory"));
                }
                kk += 16;
            }
            // As in the reference, add dQ through a fixed CTA-wide permutation
            // instead of the MMA layout: value i of warp w lands on tile row
            // w + WARPS * (i / (HD / 32)), columns 32 * (i % (HD / 32)) + lane, so
            // each reduction covers one 128-byte line. Rows are padded to a
            // multiple of 128, and `convert` undoes the permutation.
            let tile = ((b as usize * rounded_q as usize + query_start as usize) * cfg.h as usize
                + head as usize)
                * HD;
            let row_stride = cfg.h as usize * HD;
            let lane_ptr = dq.add(tile + warp as usize * row_stride + lane as usize);
            let mut mt = 0;
            while mt < 2 {
                thread::__unroll_config::<0>();
                let mut n = 0;
                while n < QN {
                    thread::__unroll_config::<0>();
                    let mut e = 0;
                    while e < 4 {
                        thread::__unroll_config::<0>();
                        let i = (mt * QN + n) * 4 + e;
                        reduce_add(
                            lane_ptr
                                .add((i / (HD / 32)) * WARPS * row_stride + 32 * (i % (HD / 32))),
                            acc_q[mt][n][e],
                        );
                        e += 1;
                    }
                    n += 1;
                }
                mt += 1;
            }
            // The dK product covers the reductions' latency.
            if DB {
                key_product!(acc_k, ds_a, q_a);
            }
            ptx_asm!("cp.async.wait_group 0;", clobber("memory"));
            thread::sync_threads();
            step += 1;
        }

        let output_heads = if PARTIAL { cfg.h } else { cfg.h / cfg.gqa };
        let output_slot = if PARTIAL { head } else { kh };
        let output_stride = output_heads as usize * cfg.hs_u32 as usize;
        let output_base = b as usize * cfg.seqlen_k as usize * output_stride
            + output_slot as usize * cfg.hs_u32 as usize;
        let v_scale = if DROP { dropout.inv_keep } else { 1.0 };
        // Stage dK/dV through the now idle shared memory so each thread
        // writes whole 16-byte row segments.
        if !PARTIAL && cfg.hs_u32 as usize == HD / 2 {
            let k_out = scratch;
            let v_out = scratch.add(BN * rs);
            let mut mt = 0;
            while mt < 2 {
                thread::__unroll_config::<0>();
                let row = (kg * 32 + mt as u32 * 16 + group) as usize;
                let mut n = 0;
                while n < KVN {
                    thread::__unroll_config::<0>();
                    let at = row * rs + (hg as usize * KVN + n) * 4 + pair as usize;
                    let (a, c) = (acc_k[mt][n], acc_v[mt][n]);
                    *k_out.add(at) = cvt::<BF16>(a[0] * gradient_scale, a[1] * gradient_scale);
                    *k_out.add(at + 8 * rs) =
                        cvt::<BF16>(a[2] * gradient_scale, a[3] * gradient_scale);
                    *v_out.add(at) = cvt::<BF16>(c[0] * v_scale, c[1] * v_scale);
                    *v_out.add(at + 8 * rs) = cvt::<BF16>(c[2] * v_scale, c[3] * v_scale);
                    n += 1;
                }
                mt += 1;
            }
            thread::sync_threads();
            let k_a = shared_address(k_out);
            let v_a = shared_address(v_out);
            // At most eight passes: BN * HD / 2 words over WARPS * 128 per pass.
            let mut pass = 0;
            while pass < 8 {
                thread::__unroll_config::<0>();
                let i = tid as usize * 4 + pass * WARPS * 128;
                if i < BN * (HD / 2) {
                    let (row, col) = (i / (HD / 2), i % (HD / 2));
                    let key = key_start as usize + row;
                    if key < cfg.seqlen_k as usize {
                        let index = output_base + key * output_stride + col;
                        let at = ((row * rs + col) * 4) as u32;
                        let (x0, x1, x2, x3): (u32, u32, u32, u32);
                        ptx_asm!(
                            "ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];",
                            out("=r") x0, out("=r") x1, out("=r") x2, out("=r") x3,
                            in("r") k_a + at, clobber("memory")
                        );
                        let (y0, y1, y2, y3): (u32, u32, u32, u32);
                        ptx_asm!(
                            "ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];",
                            out("=r") y0, out("=r") y1, out("=r") y2, out("=r") y3,
                            in("r") v_a + at, clobber("memory")
                        );
                        ptx_asm!(
                            "st.global.v4.u32 [%0], {%1, %2, %3, %4};",
                            in("l") dk.add(index), in("r") x0, in("r") x1,
                            in("r") x2, in("r") x3, clobber("memory")
                        );
                        ptx_asm!(
                            "st.global.v4.u32 [%0], {%1, %2, %3, %4};",
                            in("l") dv.add(index), in("r") y0, in("r") y1,
                            in("r") y2, in("r") y3, clobber("memory")
                        );
                    }
                }
                pass += 1;
            }
            return;
        }
        let mut mt = 0;
        while mt < 2 {
            thread::__unroll_config::<0>();
            let key = key_start + kg * 32 + mt as u32 * 16 + group;
            let mut n = 0;
            while n < KVN {
                thread::__unroll_config::<0>();
                let col = (hg * KVN as u32 * 8 + n as u32 * 8) / 2 + pair;
                if col < cfg.hs_u32 {
                    if key < cfg.seqlen_k {
                        let index = output_base + key as usize * output_stride + col as usize;
                        let a = acc_k[mt][n];
                        store_pair::<BF16, PARTIAL>(
                            dk,
                            index,
                            a[0] * gradient_scale,
                            a[1] * gradient_scale,
                        );
                        let a = acc_v[mt][n];
                        store_pair::<BF16, PARTIAL>(dv, index, a[0] * v_scale, a[1] * v_scale);
                    }
                    if key + 8 < cfg.seqlen_k {
                        let index = output_base + (key as usize + 8) * output_stride + col as usize;
                        let a = acc_k[mt][n];
                        store_pair::<BF16, PARTIAL>(
                            dk,
                            index,
                            a[2] * gradient_scale,
                            a[3] * gradient_scale,
                        );
                        let a = acc_v[mt][n];
                        store_pair::<BF16, PARTIAL>(dv, index, a[2] * v_scale, a[3] * v_scale);
                    }
                }
                n += 1;
            }
            mt += 1;
        }
    }
}

pub struct Generic<
    const BF16: bool,
    const DROP: bool,
    const PARTIAL: bool,
    const MODE: u32,
    const HD: usize,
    const BM: usize,
    const BN: usize,
    const SM: usize,
    const VREG: bool,
    const KS: usize,
    const NT: usize,
    const KVN: usize,
    const QN: usize,
    const VK: usize,
    const DB: bool,
    const WARPS: usize,
>;
impl<
    const BF16: bool,
    const DROP: bool,
    const PARTIAL: bool,
    const MODE: u32,
    const HD: usize,
    const BM: usize,
    const BN: usize,
    const SM: usize,
    const VREG: bool,
    const KS: usize,
    const NT: usize,
    const KVN: usize,
    const QN: usize,
    const VK: usize,
    const DB: bool,
    const WARPS: usize,
> FusedSpec
    for Generic<BF16, DROP, PARTIAL, MODE, HD, BM, BN, SM, VREG, KS, NT, KVN, QN, VK, DB, WARPS>
{
    const DROPOUT: bool = DROP;
    const MODE: u32 = MODE;
    #[inline(always)]
    unsafe fn run(
        q: *const u32,
        k: *const u32,
        v: *const u32,
        dout: *const u32,
        lse: *const f32,
        delta: *const f32,
        dk: *mut u32,
        dv: *mut u32,
        dq: *mut f32,
        alibi: *const f32,
        cfg: FwdConfig,
        do_strides: Strides,
        gradient_scale: f32,
        dropout: Dropout,
    ) {
        unsafe {
            core::<BF16, DROP, PARTIAL, HD, BM, BN, SM, VREG, KS, NT, KVN, QN, VK, DB, WARPS>(
                q,
                k,
                v,
                dout,
                lse,
                delta,
                dk,
                dv,
                dq,
                alibi,
                cfg,
                do_strides,
                gradient_scale,
                dropout,
                None,
            );
        }
    }
}

#[allow(non_camel_case_types)]
pub type generic_d32_f16_plain =
    Generic<false, false, false, 0, 32, 64, 64, 1, false, 2, 4, 1, 1, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d32_f16_causal =
    Generic<false, false, false, 1, 32, 64, 64, 1, false, 2, 4, 1, 1, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d32_f16_masked =
    Generic<false, false, false, 2, 32, 64, 64, 1, false, 2, 4, 1, 1, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d32_f16_partial_plain =
    Generic<false, false, true, 0, 32, 64, 64, 1, false, 2, 4, 1, 1, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d32_f16_partial_causal =
    Generic<false, false, true, 1, 32, 64, 64, 1, false, 2, 4, 1, 1, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d32_f16_partial_masked =
    Generic<false, false, true, 2, 32, 64, 64, 1, false, 2, 4, 1, 1, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d32_f16_drop_plain =
    Generic<false, true, false, 0, 32, 64, 64, 1, false, 2, 4, 1, 1, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d32_f16_drop_causal =
    Generic<false, true, false, 1, 32, 64, 64, 1, false, 2, 4, 1, 1, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d32_f16_drop_masked =
    Generic<false, true, false, 2, 32, 64, 64, 1, false, 2, 4, 1, 1, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d32_f16_drop_partial_plain =
    Generic<false, true, true, 0, 32, 64, 64, 1, false, 2, 4, 1, 1, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d32_f16_drop_partial_causal =
    Generic<false, true, true, 1, 32, 64, 64, 1, false, 2, 4, 1, 1, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d32_f16_drop_partial_masked =
    Generic<false, true, true, 2, 32, 64, 64, 1, false, 2, 4, 1, 1, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d32_bf16_plain =
    Generic<true, false, false, 0, 32, 64, 64, 1, false, 2, 4, 1, 1, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d32_bf16_causal =
    Generic<true, false, false, 1, 32, 64, 64, 1, false, 2, 4, 1, 1, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d32_bf16_masked =
    Generic<true, false, false, 2, 32, 64, 64, 1, false, 2, 4, 1, 1, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d32_bf16_partial_plain =
    Generic<true, false, true, 0, 32, 64, 64, 1, false, 2, 4, 1, 1, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d32_bf16_partial_causal =
    Generic<true, false, true, 1, 32, 64, 64, 1, false, 2, 4, 1, 1, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d32_bf16_partial_masked =
    Generic<true, false, true, 2, 32, 64, 64, 1, false, 2, 4, 1, 1, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d32_bf16_drop_plain =
    Generic<true, true, false, 0, 32, 64, 64, 1, false, 2, 4, 1, 1, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d32_bf16_drop_causal =
    Generic<true, true, false, 1, 32, 64, 64, 1, false, 2, 4, 1, 1, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d32_bf16_drop_masked =
    Generic<true, true, false, 2, 32, 64, 64, 1, false, 2, 4, 1, 1, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d32_bf16_drop_partial_plain =
    Generic<true, true, true, 0, 32, 64, 64, 1, false, 2, 4, 1, 1, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d32_bf16_drop_partial_causal =
    Generic<true, true, true, 1, 32, 64, 64, 1, false, 2, 4, 1, 1, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d32_bf16_drop_partial_masked =
    Generic<true, true, true, 2, 32, 64, 64, 1, false, 2, 4, 1, 1, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d64_f16_plain =
    Generic<false, false, false, 0, 64, 64, 128, 2, true, 4, 4, 4, 2, 4, true, 8>;
#[allow(non_camel_case_types)]
pub type generic_d64_f16_causal =
    Generic<false, false, false, 1, 64, 64, 64, 1, false, 4, 4, 2, 2, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d64_f16_masked =
    Generic<false, false, false, 2, 64, 64, 64, 1, false, 4, 4, 2, 2, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d64_f16_partial_plain =
    Generic<false, false, true, 0, 64, 64, 128, 2, true, 4, 4, 4, 2, 4, true, 8>;
#[allow(non_camel_case_types)]
pub type generic_d64_f16_partial_causal =
    Generic<false, false, true, 1, 64, 64, 64, 1, false, 4, 4, 2, 2, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d64_f16_partial_masked =
    Generic<false, false, true, 2, 64, 64, 64, 1, false, 4, 4, 2, 2, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d64_f16_drop_plain =
    Generic<false, true, false, 0, 64, 32, 64, 1, false, 4, 4, 4, 2, 1, false, 4>;
#[allow(non_camel_case_types)]
pub type generic_d64_f16_drop_causal =
    Generic<false, true, false, 1, 64, 32, 64, 1, false, 4, 4, 4, 2, 1, false, 4>;
#[allow(non_camel_case_types)]
pub type generic_d64_f16_drop_masked =
    Generic<false, true, false, 2, 64, 32, 64, 1, false, 4, 4, 4, 2, 1, false, 4>;
#[allow(non_camel_case_types)]
pub type generic_d64_f16_drop_partial_plain =
    Generic<false, true, true, 0, 64, 32, 64, 1, false, 4, 4, 4, 2, 1, false, 4>;
#[allow(non_camel_case_types)]
pub type generic_d64_f16_drop_partial_causal =
    Generic<false, true, true, 1, 64, 32, 64, 1, false, 4, 4, 4, 2, 1, false, 4>;
#[allow(non_camel_case_types)]
pub type generic_d64_f16_drop_partial_masked =
    Generic<false, true, true, 2, 64, 32, 64, 1, false, 4, 4, 4, 2, 1, false, 4>;
#[allow(non_camel_case_types)]
pub type generic_d64_bf16_plain =
    Generic<true, false, false, 0, 64, 64, 128, 2, true, 4, 4, 4, 2, 4, true, 8>;
#[allow(non_camel_case_types)]
pub type generic_d64_bf16_causal =
    Generic<true, false, false, 1, 64, 64, 64, 1, false, 4, 4, 2, 2, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d64_bf16_masked =
    Generic<true, false, false, 2, 64, 64, 64, 1, false, 4, 4, 2, 2, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d64_bf16_partial_plain =
    Generic<true, false, true, 0, 64, 64, 128, 2, true, 4, 4, 4, 2, 4, true, 8>;
#[allow(non_camel_case_types)]
pub type generic_d64_bf16_partial_causal =
    Generic<true, false, true, 1, 64, 64, 64, 1, false, 4, 4, 2, 2, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d64_bf16_partial_masked =
    Generic<true, false, true, 2, 64, 64, 64, 1, false, 4, 4, 2, 2, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d64_bf16_drop_plain =
    Generic<true, true, false, 0, 64, 32, 64, 1, false, 4, 4, 4, 2, 1, false, 4>;
#[allow(non_camel_case_types)]
pub type generic_d64_bf16_drop_causal =
    Generic<true, true, false, 1, 64, 32, 64, 1, false, 4, 4, 4, 2, 1, false, 4>;
#[allow(non_camel_case_types)]
pub type generic_d64_bf16_drop_masked =
    Generic<true, true, false, 2, 64, 32, 64, 1, false, 4, 4, 4, 2, 1, false, 4>;
#[allow(non_camel_case_types)]
pub type generic_d64_bf16_drop_partial_plain =
    Generic<true, true, true, 0, 64, 32, 64, 1, false, 4, 4, 4, 2, 1, false, 4>;
#[allow(non_camel_case_types)]
pub type generic_d64_bf16_drop_partial_causal =
    Generic<true, true, true, 1, 64, 32, 64, 1, false, 4, 4, 4, 2, 1, false, 4>;
#[allow(non_camel_case_types)]
pub type generic_d64_bf16_drop_partial_masked =
    Generic<true, true, true, 2, 64, 32, 64, 1, false, 4, 4, 4, 2, 1, false, 4>;
#[allow(non_camel_case_types)]
pub type generic_d96_f16_plain =
    Generic<false, false, false, 0, 96, 32, 64, 1, true, 6, 4, 6, 3, 6, true, 4>;
#[allow(non_camel_case_types)]
pub type generic_d96_f16_causal =
    Generic<false, false, false, 1, 96, 64, 64, 1, false, 6, 4, 3, 3, 1, true, 8>;
#[allow(non_camel_case_types)]
pub type generic_d96_f16_masked =
    Generic<false, false, false, 2, 96, 64, 64, 1, false, 6, 4, 3, 3, 1, true, 8>;
#[allow(non_camel_case_types)]
pub type generic_d96_f16_partial_plain =
    Generic<false, false, true, 0, 96, 32, 64, 1, true, 6, 4, 6, 3, 6, true, 4>;
#[allow(non_camel_case_types)]
pub type generic_d96_f16_partial_causal =
    Generic<false, false, true, 1, 96, 64, 64, 1, false, 6, 4, 3, 3, 1, true, 8>;
#[allow(non_camel_case_types)]
pub type generic_d96_f16_partial_masked =
    Generic<false, false, true, 2, 96, 64, 64, 1, false, 6, 4, 3, 3, 1, true, 8>;
#[allow(non_camel_case_types)]
pub type generic_d96_f16_drop_plain =
    Generic<false, true, false, 0, 96, 32, 64, 1, false, 6, 4, 6, 3, 1, false, 4>;
#[allow(non_camel_case_types)]
pub type generic_d96_f16_drop_causal =
    Generic<false, true, false, 1, 96, 32, 64, 1, false, 6, 4, 6, 3, 1, false, 4>;
#[allow(non_camel_case_types)]
pub type generic_d96_f16_drop_masked =
    Generic<false, true, false, 2, 96, 32, 64, 1, false, 6, 4, 6, 3, 1, false, 4>;
#[allow(non_camel_case_types)]
pub type generic_d96_f16_drop_partial_plain =
    Generic<false, true, true, 0, 96, 32, 64, 1, false, 6, 4, 6, 3, 1, false, 4>;
#[allow(non_camel_case_types)]
pub type generic_d96_f16_drop_partial_causal =
    Generic<false, true, true, 1, 96, 32, 64, 1, false, 6, 4, 6, 3, 1, false, 4>;
#[allow(non_camel_case_types)]
pub type generic_d96_f16_drop_partial_masked =
    Generic<false, true, true, 2, 96, 32, 64, 1, false, 6, 4, 6, 3, 1, false, 4>;
#[allow(non_camel_case_types)]
pub type generic_d96_bf16_plain =
    Generic<true, false, false, 0, 96, 32, 64, 1, true, 6, 4, 6, 3, 6, true, 4>;
#[allow(non_camel_case_types)]
pub type generic_d96_bf16_causal =
    Generic<true, false, false, 1, 96, 64, 64, 1, false, 6, 4, 3, 3, 1, true, 8>;
#[allow(non_camel_case_types)]
pub type generic_d96_bf16_masked =
    Generic<true, false, false, 2, 96, 64, 64, 1, false, 6, 4, 3, 3, 1, true, 8>;
#[allow(non_camel_case_types)]
pub type generic_d96_bf16_partial_plain =
    Generic<true, false, true, 0, 96, 32, 64, 1, true, 6, 4, 6, 3, 6, true, 4>;
#[allow(non_camel_case_types)]
pub type generic_d96_bf16_partial_causal =
    Generic<true, false, true, 1, 96, 64, 64, 1, false, 6, 4, 3, 3, 1, true, 8>;
#[allow(non_camel_case_types)]
pub type generic_d96_bf16_partial_masked =
    Generic<true, false, true, 2, 96, 64, 64, 1, false, 6, 4, 3, 3, 1, true, 8>;
#[allow(non_camel_case_types)]
pub type generic_d96_bf16_drop_plain =
    Generic<true, true, false, 0, 96, 32, 64, 1, false, 6, 4, 6, 3, 1, false, 4>;
#[allow(non_camel_case_types)]
pub type generic_d96_bf16_drop_causal =
    Generic<true, true, false, 1, 96, 32, 64, 1, false, 6, 4, 6, 3, 1, false, 4>;
#[allow(non_camel_case_types)]
pub type generic_d96_bf16_drop_masked =
    Generic<true, true, false, 2, 96, 32, 64, 1, false, 6, 4, 6, 3, 1, false, 4>;
#[allow(non_camel_case_types)]
pub type generic_d96_bf16_drop_partial_plain =
    Generic<true, true, true, 0, 96, 32, 64, 1, false, 6, 4, 6, 3, 1, false, 4>;
#[allow(non_camel_case_types)]
pub type generic_d96_bf16_drop_partial_causal =
    Generic<true, true, true, 1, 96, 32, 64, 1, false, 6, 4, 6, 3, 1, false, 4>;
#[allow(non_camel_case_types)]
pub type generic_d96_bf16_drop_partial_masked =
    Generic<true, true, true, 2, 96, 32, 64, 1, false, 6, 4, 6, 3, 1, false, 4>;
#[allow(non_camel_case_types)]
pub type generic_d128_f16_plain =
    Generic<false, false, false, 0, 128, 64, 64, 1, true, 8, 4, 4, 4, 8, true, 8>;
#[allow(non_camel_case_types)]
pub type generic_d128_f16_causal =
    Generic<false, false, false, 1, 128, 64, 64, 1, true, 8, 4, 4, 4, 8, true, 8>;
#[allow(non_camel_case_types)]
pub type generic_d128_f16_masked =
    Generic<false, false, false, 2, 128, 64, 64, 1, true, 8, 4, 4, 4, 8, true, 8>;
#[allow(non_camel_case_types)]
pub type generic_d128_f16_partial_plain =
    Generic<false, false, true, 0, 128, 64, 64, 1, true, 8, 4, 4, 4, 8, true, 8>;
#[allow(non_camel_case_types)]
pub type generic_d128_f16_partial_causal =
    Generic<false, false, true, 1, 128, 64, 64, 1, true, 8, 4, 4, 4, 8, true, 8>;
#[allow(non_camel_case_types)]
pub type generic_d128_f16_partial_masked =
    Generic<false, false, true, 2, 128, 64, 64, 1, true, 8, 4, 4, 4, 8, true, 8>;
#[allow(non_camel_case_types)]
pub type generic_d128_f16_drop_plain =
    Generic<false, true, false, 0, 128, 64, 64, 1, true, 8, 4, 4, 4, 8, true, 8>;
#[allow(non_camel_case_types)]
pub type generic_d128_f16_drop_causal =
    Generic<false, true, false, 1, 128, 64, 64, 1, true, 8, 4, 4, 4, 8, true, 8>;
#[allow(non_camel_case_types)]
pub type generic_d128_f16_drop_masked =
    Generic<false, true, false, 2, 128, 64, 64, 1, true, 8, 4, 4, 4, 8, true, 8>;
#[allow(non_camel_case_types)]
pub type generic_d128_f16_drop_partial_plain =
    Generic<false, true, true, 0, 128, 64, 64, 1, true, 8, 4, 4, 4, 8, true, 8>;
#[allow(non_camel_case_types)]
pub type generic_d128_f16_drop_partial_causal =
    Generic<false, true, true, 1, 128, 64, 64, 1, true, 8, 4, 4, 4, 8, true, 8>;
#[allow(non_camel_case_types)]
pub type generic_d128_f16_drop_partial_masked =
    Generic<false, true, true, 2, 128, 64, 64, 1, true, 8, 4, 4, 4, 8, true, 8>;
#[allow(non_camel_case_types)]
pub type generic_d128_bf16_plain =
    Generic<true, false, false, 0, 128, 64, 64, 1, true, 8, 4, 4, 4, 8, true, 8>;
#[allow(non_camel_case_types)]
pub type generic_d128_bf16_causal =
    Generic<true, false, false, 1, 128, 64, 64, 1, true, 8, 4, 4, 4, 8, true, 8>;
#[allow(non_camel_case_types)]
pub type generic_d128_bf16_masked =
    Generic<true, false, false, 2, 128, 64, 64, 1, true, 8, 4, 4, 4, 8, true, 8>;
#[allow(non_camel_case_types)]
pub type generic_d128_bf16_partial_plain =
    Generic<true, false, true, 0, 128, 64, 64, 1, true, 8, 4, 4, 4, 8, true, 8>;
#[allow(non_camel_case_types)]
pub type generic_d128_bf16_partial_causal =
    Generic<true, false, true, 1, 128, 64, 64, 1, true, 8, 4, 4, 4, 8, true, 8>;
#[allow(non_camel_case_types)]
pub type generic_d128_bf16_partial_masked =
    Generic<true, false, true, 2, 128, 64, 64, 1, true, 8, 4, 4, 4, 8, true, 8>;
#[allow(non_camel_case_types)]
pub type generic_d128_bf16_drop_plain =
    Generic<true, true, false, 0, 128, 64, 64, 1, true, 8, 4, 4, 4, 8, true, 8>;
#[allow(non_camel_case_types)]
pub type generic_d128_bf16_drop_causal =
    Generic<true, true, false, 1, 128, 64, 64, 1, true, 8, 4, 4, 4, 8, true, 8>;
#[allow(non_camel_case_types)]
pub type generic_d128_bf16_drop_masked =
    Generic<true, true, false, 2, 128, 64, 64, 1, true, 8, 4, 4, 4, 8, true, 8>;
#[allow(non_camel_case_types)]
pub type generic_d128_bf16_drop_partial_plain =
    Generic<true, true, true, 0, 128, 64, 64, 1, true, 8, 4, 4, 4, 8, true, 8>;
#[allow(non_camel_case_types)]
pub type generic_d128_bf16_drop_partial_causal =
    Generic<true, true, true, 1, 128, 64, 64, 1, true, 8, 4, 4, 4, 8, true, 8>;
#[allow(non_camel_case_types)]
pub type generic_d128_bf16_drop_partial_masked =
    Generic<true, true, true, 2, 128, 64, 64, 1, true, 8, 4, 4, 4, 8, true, 8>;
#[allow(non_camel_case_types)]
pub type generic_d192_f16_plain =
    Generic<false, false, false, 0, 192, 32, 64, 1, false, 12, 2, 6, 3, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d192_f16_causal =
    Generic<false, false, false, 1, 192, 32, 64, 1, false, 12, 2, 6, 3, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d192_f16_masked =
    Generic<false, false, false, 2, 192, 32, 64, 1, false, 12, 2, 6, 3, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d192_f16_partial_plain =
    Generic<false, false, true, 0, 192, 32, 64, 1, false, 12, 2, 6, 3, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d192_f16_partial_causal =
    Generic<false, false, true, 1, 192, 32, 64, 1, false, 12, 2, 6, 3, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d192_f16_partial_masked =
    Generic<false, false, true, 2, 192, 32, 64, 1, false, 12, 2, 6, 3, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d192_f16_drop_plain =
    Generic<false, true, false, 0, 192, 32, 64, 1, false, 12, 2, 6, 3, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d192_f16_drop_causal =
    Generic<false, true, false, 1, 192, 32, 64, 1, false, 12, 2, 6, 3, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d192_f16_drop_masked =
    Generic<false, true, false, 2, 192, 32, 64, 1, false, 12, 2, 6, 3, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d192_f16_drop_partial_plain =
    Generic<false, true, true, 0, 192, 32, 64, 1, false, 12, 2, 6, 3, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d192_f16_drop_partial_causal =
    Generic<false, true, true, 1, 192, 32, 64, 1, false, 12, 2, 6, 3, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d192_f16_drop_partial_masked =
    Generic<false, true, true, 2, 192, 32, 64, 1, false, 12, 2, 6, 3, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d192_bf16_plain =
    Generic<true, false, false, 0, 192, 32, 64, 1, false, 12, 2, 6, 3, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d192_bf16_causal =
    Generic<true, false, false, 1, 192, 32, 64, 1, false, 12, 2, 6, 3, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d192_bf16_masked =
    Generic<true, false, false, 2, 192, 32, 64, 1, false, 12, 2, 6, 3, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d192_bf16_partial_plain =
    Generic<true, false, true, 0, 192, 32, 64, 1, false, 12, 2, 6, 3, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d192_bf16_partial_causal =
    Generic<true, false, true, 1, 192, 32, 64, 1, false, 12, 2, 6, 3, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d192_bf16_partial_masked =
    Generic<true, false, true, 2, 192, 32, 64, 1, false, 12, 2, 6, 3, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d192_bf16_drop_plain =
    Generic<true, true, false, 0, 192, 32, 64, 1, false, 12, 2, 6, 3, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d192_bf16_drop_causal =
    Generic<true, true, false, 1, 192, 32, 64, 1, false, 12, 2, 6, 3, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d192_bf16_drop_masked =
    Generic<true, true, false, 2, 192, 32, 64, 1, false, 12, 2, 6, 3, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d192_bf16_drop_partial_plain =
    Generic<true, true, true, 0, 192, 32, 64, 1, false, 12, 2, 6, 3, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d192_bf16_drop_partial_causal =
    Generic<true, true, true, 1, 192, 32, 64, 1, false, 12, 2, 6, 3, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d192_bf16_drop_partial_masked =
    Generic<true, true, true, 2, 192, 32, 64, 1, false, 12, 2, 6, 3, 1, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d256_f16_drop_plain =
    Generic<false, true, false, 0, 256, 64, 32, 1, true, 16, 2, 4, 8, 16, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d256_f16_drop_causal =
    Generic<false, true, false, 1, 256, 64, 32, 1, true, 16, 2, 4, 8, 16, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d256_f16_drop_masked =
    Generic<false, true, false, 2, 256, 64, 32, 1, true, 16, 2, 4, 8, 16, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d256_f16_drop_partial_plain =
    Generic<false, true, true, 0, 256, 64, 32, 1, true, 16, 2, 4, 8, 16, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d256_f16_drop_partial_causal =
    Generic<false, true, true, 1, 256, 64, 32, 1, true, 16, 2, 4, 8, 16, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d256_f16_drop_partial_masked =
    Generic<false, true, true, 2, 256, 64, 32, 1, true, 16, 2, 4, 8, 16, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d256_bf16_drop_plain =
    Generic<true, true, false, 0, 256, 64, 32, 1, true, 16, 2, 4, 8, 16, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d256_bf16_drop_causal =
    Generic<true, true, false, 1, 256, 64, 32, 1, true, 16, 2, 4, 8, 16, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d256_bf16_drop_masked =
    Generic<true, true, false, 2, 256, 64, 32, 1, true, 16, 2, 4, 8, 16, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d256_bf16_drop_partial_plain =
    Generic<true, true, true, 0, 256, 64, 32, 1, true, 16, 2, 4, 8, 16, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d256_bf16_drop_partial_causal =
    Generic<true, true, true, 1, 256, 64, 32, 1, true, 16, 2, 4, 8, 16, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d256_bf16_drop_partial_masked =
    Generic<true, true, true, 2, 256, 64, 32, 1, true, 16, 2, 4, 8, 16, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d256_f16_plain =
    Generic<false, false, false, 0, 256, 64, 32, 1, true, 16, 2, 4, 8, 16, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d256_f16_causal =
    Generic<false, false, false, 1, 256, 64, 32, 1, true, 16, 2, 4, 8, 16, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d256_f16_masked =
    Generic<false, false, false, 2, 256, 64, 32, 1, true, 16, 2, 4, 8, 16, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d256_f16_partial_plain =
    Generic<false, false, true, 0, 256, 64, 32, 1, true, 16, 2, 4, 8, 16, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d256_f16_partial_causal =
    Generic<false, false, true, 1, 256, 64, 32, 1, true, 16, 2, 4, 8, 16, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d256_f16_partial_masked =
    Generic<false, false, true, 2, 256, 64, 32, 1, true, 16, 2, 4, 8, 16, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d256_bf16_plain =
    Generic<true, false, false, 0, 256, 64, 32, 1, true, 16, 2, 4, 8, 16, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d256_bf16_causal =
    Generic<true, false, false, 1, 256, 64, 32, 1, true, 16, 2, 4, 8, 16, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d256_bf16_masked =
    Generic<true, false, false, 2, 256, 64, 32, 1, true, 16, 2, 4, 8, 16, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d256_bf16_partial_plain =
    Generic<true, false, true, 0, 256, 64, 32, 1, true, 16, 2, 4, 8, 16, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d256_bf16_partial_causal =
    Generic<true, false, true, 1, 256, 64, 32, 1, true, 16, 2, 4, 8, 16, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d256_bf16_partial_masked =
    Generic<true, false, true, 2, 256, 64, 32, 1, true, 16, 2, 4, 8, 16, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d32_f16_wide =
    Generic<false, false, false, 0, 32, 128, 128, 2, true, 2, 8, 2, 2, 2, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d32_f16_partial_wide =
    Generic<false, false, true, 0, 32, 128, 128, 2, true, 2, 8, 2, 2, 2, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d32_bf16_wide =
    Generic<true, false, false, 0, 32, 128, 128, 2, true, 2, 8, 2, 2, 2, false, 8>;
#[allow(non_camel_case_types)]
pub type generic_d32_bf16_partial_wide =
    Generic<true, false, true, 0, 32, 128, 128, 2, true, 2, 8, 2, 2, 2, false, 8>;

pub trait ConvertSpec {
    const BF16: bool;
    const HD: usize;
    const BM: usize;
    const QN: usize;
    const WARPS: usize;
}
pub struct Convert<
    const BF16: bool,
    const HD: usize,
    const BM: usize,
    const QN: usize,
    const WARPS: usize = 8,
>;
impl<const BF16: bool, const HD: usize, const BM: usize, const QN: usize, const WARPS: usize>
    ConvertSpec for Convert<BF16, HD, BM, QN, WARPS>
{
    const WARPS: usize = WARPS;
    const BF16: bool = BF16;
    const HD: usize = HD;
    const BM: usize = BM;
    const QN: usize = QN;
}
#[allow(non_camel_case_types)]
pub type convert_d32_m64w8_f16 = Convert<false, 32, 64, 1, 8>;
#[allow(non_camel_case_types)]
pub type convert_d32_m64w8_bf16 = Convert<true, 32, 64, 1, 8>;
#[allow(non_camel_case_types)]
pub type convert_d32_m128w8_f16 = Convert<false, 32, 128, 2, 8>;
#[allow(non_camel_case_types)]
pub type convert_d32_m128w8_bf16 = Convert<true, 32, 128, 2, 8>;
#[allow(non_camel_case_types)]
pub type convert_d64_m32w4_f16 = Convert<false, 64, 32, 2, 4>;
#[allow(non_camel_case_types)]
pub type convert_d64_m32w4_bf16 = Convert<true, 64, 32, 2, 4>;
#[allow(non_camel_case_types)]
pub type convert_d64_m64w8_f16 = Convert<false, 64, 64, 2, 8>;
#[allow(non_camel_case_types)]
pub type convert_d64_m64w8_bf16 = Convert<true, 64, 64, 2, 8>;
#[allow(non_camel_case_types)]
pub type convert_d96_m32w4_f16 = Convert<false, 96, 32, 3, 4>;
#[allow(non_camel_case_types)]
pub type convert_d96_m32w4_bf16 = Convert<true, 96, 32, 3, 4>;
#[allow(non_camel_case_types)]
pub type convert_d96_m64w8_f16 = Convert<false, 96, 64, 3, 8>;
#[allow(non_camel_case_types)]
pub type convert_d96_m64w8_bf16 = Convert<true, 96, 64, 3, 8>;
#[allow(non_camel_case_types)]
pub type convert_d128_m64w8_f16 = Convert<false, 128, 64, 4, 8>;
#[allow(non_camel_case_types)]
pub type convert_d128_m64w8_bf16 = Convert<true, 128, 64, 4, 8>;
#[allow(non_camel_case_types)]
pub type convert_d192_m32w8_f16 = Convert<false, 192, 32, 3, 8>;
#[allow(non_camel_case_types)]
pub type convert_d192_m32w8_bf16 = Convert<true, 192, 32, 3, 8>;
#[allow(non_camel_case_types)]
pub type convert_d256_m64w8_f16 = Convert<false, 256, 64, 8, 8>;
#[allow(non_camel_case_types)]
pub type convert_d256_m64w8_bf16 = Convert<true, 256, 64, 8, 8>;

/// Undo the dQ reduction permutation of one BM-row tile at `partial` (row
/// stride `row_stride` floats) and write `rows` scaled half-precision rows
/// starting at pair index `out` with row stride `out_stride` pairs.
#[inline(always)]
unsafe fn convert_tile<S: ConvertSpec>(
    partial: *const f32,
    row_stride: usize,
    out: *mut u32,
    out_stride: usize,
    rows: usize,
    scale: f32,
) {
    unsafe {
        let tid = thread::threadIdx_x();
        let lane = tid % 32;
        let warp = tid / 32;
        let group = lane >> 2;
        let pair = lane & 3;
        let q_cols = S::WARPS as u32 / (S::BM as u32 / 32);
        let qm = warp / q_cols;
        let qh = warp % q_cols;
        let staged = DynamicSharedArray::<f32, 16>::get() as usize as *mut f32;
        let mut mt = 0;
        while mt < 2 {
            thread::__unroll_config::<0>();
            let mut n = 0;
            while n < S::QN {
                thread::__unroll_config::<0>();
                let mut e = 0;
                while e < 4 {
                    thread::__unroll_config::<0>();
                    let i = (mt * S::QN + n) * 4 + e;
                    let row = warp as usize + S::WARPS * (i / (S::HD / 32));
                    let col = 32 * (i % (S::HD / 32)) + lane as usize;
                    let logical_row = qm * 32 + mt as u32 * 16 + group + (e as u32 / 2) * 8;
                    let logical_col =
                        qh * S::QN as u32 * 8 + n as u32 * 8 + pair * 2 + e as u32 % 2;
                    *staged.add(logical_row as usize * (S::HD + 4) + logical_col as usize) =
                        *partial.add(row * row_stride + col);
                    e += 1;
                }
                n += 1;
            }
            mt += 1;
        }
        thread::sync_threads();
        let mut index = tid as usize;
        while index < S::BM * S::HD / 2 {
            let row = index / (S::HD / 2);
            let pair_col = index % (S::HD / 2);
            if row < rows {
                let x = *staged.add(row * (S::HD + 4) + 2 * pair_col) * scale;
                let y = *staged.add(row * (S::HD + 4) + 2 * pair_col + 1) * scale;
                *out.add(row * out_stride + pair_col) = if S::BF16 {
                    cvt::<true>(x, y)
                } else {
                    cvt::<false>(x, y)
                };
            }
            index += S::WARPS * 32;
        }
    }
}

/// Dense dQ conversion. Grid: (query tiles, heads, batch).
#[kernel(
    convert_d32_m64w8_f16,
    convert_d32_m64w8_bf16,
    convert_d32_m128w8_f16,
    convert_d32_m128w8_bf16,
    convert_d64_m32w4_f16,
    convert_d64_m32w4_bf16,
    convert_d64_m64w8_f16,
    convert_d64_m64w8_bf16,
    convert_d96_m32w4_f16,
    convert_d96_m32w4_bf16,
    convert_d96_m64w8_f16,
    convert_d96_m64w8_bf16,
    convert_d128_m64w8_f16,
    convert_d128_m64w8_bf16,
    convert_d192_m32w8_f16,
    convert_d192_m32w8_bf16,
    convert_d256_m64w8_f16,
    convert_d256_m64w8_bf16
)]
#[launch_bounds(256)]
pub fn generic_dq<S: ConvertSpec>(
    partial: &[f32],
    dq: DisjointSlice<u32>,
    seqlen_q: u32,
    heads: u32,
    scale: f32,
) {
    let mut dq = dq;
    unsafe {
        let query_start = thread::blockIdx_x() as usize * S::BM;
        let head = thread::blockIdx_y() as usize;
        let b = thread::blockIdx_z() as usize;
        let heads = heads as usize;
        let rounded_q = (seqlen_q as usize).div_ceil(128) * 128;
        convert_tile::<S>(
            partial
                .as_ptr()
                .add(((b * rounded_q + query_start) * heads + head) * S::HD),
            heads * S::HD,
            dq.as_mut_ptr()
                .add(((b * seqlen_q as usize + query_start) * heads + head) * (S::HD / 2)),
            heads * S::HD / 2,
            (seqlen_q as usize - query_start).min(S::BM),
            scale,
        );
    }
}

/// Packed dQ conversion; each sequence's accumulator rows start 128 rows
/// after the previous one's. Grid: (query tiles, heads, batch).
#[kernel(
    convert_d32_m64w8_f16,
    convert_d32_m64w8_bf16,
    convert_d32_m128w8_f16,
    convert_d32_m128w8_bf16,
    convert_d64_m32w4_f16,
    convert_d64_m32w4_bf16,
    convert_d64_m64w8_f16,
    convert_d64_m64w8_bf16,
    convert_d96_m32w4_f16,
    convert_d96_m32w4_bf16,
    convert_d96_m64w8_f16,
    convert_d96_m64w8_bf16,
    convert_d128_m64w8_f16,
    convert_d128_m64w8_bf16,
    convert_d192_m32w8_f16,
    convert_d192_m32w8_bf16,
    convert_d256_m64w8_f16,
    convert_d256_m64w8_bf16
)]
#[launch_bounds(256)]
pub fn generic_packed_dq<S: ConvertSpec>(
    partial: &[f32],
    dq: DisjointSlice<u32>,
    cuq: &[i32],
    heads: u32,
    scale: f32,
) {
    let mut dq = dq;
    unsafe {
        let query_start = thread::blockIdx_x() as usize * S::BM;
        let head = thread::blockIdx_y() as usize;
        let b = thread::blockIdx_z() as usize;
        let heads = heads as usize;
        let start = *cuq.as_ptr().add(b) as usize;
        let rows = *cuq.as_ptr().add(b + 1) as usize - start;
        if query_start >= rows {
            return;
        }
        convert_tile::<S>(
            partial
                .as_ptr()
                .add(((start + b * 128 + query_start) * heads + head) * S::HD),
            heads * S::HD,
            dq.as_mut_ptr()
                .add(((start + query_start) * heads + head) * (S::HD / 2)),
            heads * S::HD / 2,
            (rows - query_start).min(S::BM),
            scale,
        );
    }
}
