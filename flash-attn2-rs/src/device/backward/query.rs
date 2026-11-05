//! Query-oriented D64 score tiles, with shared P/dS for the gradient products.
use super::common::probability_gradient;
use crate::device::config::{FwdConfig, Strides};
use crate::device::inlined::{
    cvt, ldmatrix_x2_shared, ldmatrix_x4_shared, ldmatrix_x4_trans_shared, mma, shared_address,
};
use cuda_device::shared::DynamicSharedArray;
use cuda_device::{ptx_asm, thread};

// Host dispatch guarantees D64 MHA without dropout, bias, or softcap.
#[inline(always)]
fn probability<const EVEN: bool>(
    raw: f32,
    row: u32,
    col: u32,
    lse: f32,
    cfg: &FwdConfig,
    slope: f32,
    gradient: bool,
) -> f32 {
    if EVEN {
        let center = row as i64 + cfg.seqlen_k as i64 - cfg.seqlen_q as i64;
        if cfg.window_right >= 0 && col as i64 > center {
            0.0
        } else {
            crate::device::inlined::fast_exp2(
                raw * (cfg.scale * crate::device::forward::LOG2_E)
                    - lse * crate::device::forward::LOG2_E,
            )
        }
    } else {
        probability_gradient(raw, row, col, lse, cfg, slope, gradient)
    }
}

#[inline(always)]
unsafe fn copy_rows<const ROWS: usize, const EVEN: bool>(
    src: *const u32,
    dst: *mut u32,
    base: usize,
    row_start: u32,
    stride: u32,
    seq: u32,
    _dim_pairs: u32,
    tid: u32,
) {
    unsafe {
        let mut i = tid as usize * 4;
        while i < ROWS * 32 {
            let row = i / 32;
            let col = i % 32;
            let global_row = row_start as usize + row;
            let valid = EVEN || global_row < seq as usize;
            let index = if valid {
                base + global_row * stride as usize + col
            } else {
                0
            };
            cuda_device::async_copy::cp_async_ca_zfill_16(
                dst.add(row * 32 + (col ^ ((row & 7) * 4))),
                src.add(index).cast(),
                if valid { 16 } else { 0 },
            );
            i += 1024;
        }
    }
}

#[inline(always)]
fn offset(row: u32, col: u32, stride: u32) -> u32 {
    (row * stride + (col ^ ((row & 7) * 4))) * 4
}

#[inline(always)]
unsafe fn load_b(addr: u32, group: u32, lane: u32, kc: u32, nt: u32) -> [u32; 2] {
    unsafe {
        ldmatrix_x2_shared(
            addr + offset(
                group + nt * 8 + (lane & 7),
                kc * 8 + ((lane >> 3) & 1) * 4,
                32,
            ),
        )
    }
}

#[inline(always)]
unsafe fn score_row<const BF16: bool, const VREGS: bool>(
    queries: *const u32,
    keys: *const u32,
    tid: u32,
    row_fragment: u32,
    value_fragments: &[[[u32; 2]; 4]; 4],
) -> [[f32; 4]; 4] {
    unsafe {
        let lane = tid % 32;
        let qa = shared_address(queries);
        let ka = shared_address(keys);
        let group = (tid / 64) * 32;
        let row = ((tid / 32) % 2) * 16 + (lane & 15) + row_fragment * 32;
        let mut acc = [[0.0f32; 4]; 4];
        macro_rules! chunk {
            ($kc:expr) => {
                let a = ldmatrix_x4_shared(qa + offset(row, $kc * 8 + (lane / 16) * 4, 32));
                acc[0] = mma::<BF16>(
                    acc[0],
                    a,
                    if VREGS {
                        value_fragments[$kc][0]
                    } else {
                        load_b(ka, group, lane, $kc, 0)
                    },
                );
                acc[1] = mma::<BF16>(
                    acc[1],
                    a,
                    if VREGS {
                        value_fragments[$kc][1]
                    } else {
                        load_b(ka, group, lane, $kc, 1)
                    },
                );
                acc[2] = mma::<BF16>(
                    acc[2],
                    a,
                    if VREGS {
                        value_fragments[$kc][2]
                    } else {
                        load_b(ka, group, lane, $kc, 2)
                    },
                );
                acc[3] = mma::<BF16>(
                    acc[3],
                    a,
                    if VREGS {
                        value_fragments[$kc][3]
                    } else {
                        load_b(ka, group, lane, $kc, 3)
                    },
                );
            };
        }
        chunk!(0);
        chunk!(1);
        chunk!(2);
        chunk!(3);
        acc
    }
}

// P/dS are [query, key]; each warp accumulates 32 key rows and 32 dimensions.
#[inline(always)]
unsafe fn accumulate_kv<const BF16: bool>(
    acc: &mut [[f32; 4]; 8],
    probabilities: *const u32,
    queries: *const u32,
    tid: u32,
) {
    unsafe {
        let lane = tid % 32;
        let warp = tid / 32;
        let pa = shared_address(probabilities);
        let qa = shared_address(queries);
        let mut kc = 0;
        while kc < 4 {
            thread::__unroll_config::<0>();
            let row = kc * 16 + lane % 8 + (lane / 16) * 8;
            let col = (warp % 4) * 8 + ((lane / 8) % 2) * 4;
            let a0 = ldmatrix_x4_trans_shared(pa + offset(row, col, 64));
            let a1 = ldmatrix_x4_trans_shared(pa + offset(row, col + 32, 64));
            let mut d = 0;
            while d < 4 {
                thread::__unroll_config::<0>();
                let b = ldmatrix_x4_trans_shared(
                    qa + offset(
                        kc * 16 + lane % 16,
                        (warp / 4) * 16 + d * 4 + (lane / 16) * 4,
                        32,
                    ),
                );
                acc[d as usize] = mma::<BF16>(acc[d as usize], a0, [b[0], b[1]]);
                acc[d as usize + 1] = mma::<BF16>(acc[d as usize + 1], a0, [b[2], b[3]]);
                acc[d as usize + 4] = mma::<BF16>(acc[d as usize + 4], a1, [b[0], b[1]]);
                acc[d as usize + 5] = mma::<BF16>(acc[d as usize + 5], a1, [b[2], b[3]]);
                d += 2;
            }
            kc += 1;
        }
    }
}

#[inline(always)]
unsafe fn accumulate_q<const BF16: bool, const EVEN: bool, const PERMUTE: bool>(
    ds: *const u32,
    keys: *const u32,
    staging: *mut f32,
    dq: *mut f32,
    cfg: &FwdConfig,
    b: u32,
    head: u32,
    query_start: u32,
    tid: u32,
) {
    unsafe {
        let lane = tid % 32;
        let warp = tid / 32;
        let da = shared_address(ds);
        let ka = shared_address(keys);
        let mut acc = [[0.0f32; 4]; 4];
        let mut kc = 0;
        while kc < 8 {
            thread::__unroll_config::<0>();
            let a = ldmatrix_x4_shared(
                da + offset((warp % 4) * 16 + lane % 16, kc * 8 + (lane / 16) * 4, 64),
            );
            let mut d = 0;
            while d < 4 {
                thread::__unroll_config::<0>();
                let col = (warp / 4) * 16 + d * 4 + (lane / 16) * 4;
                let bv = ldmatrix_x4_trans_shared(ka + offset(kc * 16 + lane % 16, col, 32));
                acc[d as usize] = mma::<BF16>(acc[d as usize], a, [bv[0], bv[1]]);
                acc[d as usize + 1] = mma::<BF16>(acc[d as usize + 1], a, [bv[2], bv[3]]);
                d += 2;
            }
            kc += 1;
        }
        if PERMUTE {
            let stride = cfg.h as usize * 64;
            let base = ((b as usize * cfg.seqlen_q as usize + query_start as usize)
                * cfg.h as usize
                + head as usize)
                * 64;
            let mut d = 0;
            while d < 4 {
                thread::__unroll_config::<0>();
                let x = acc[d as usize];
                let row = d * 16 + tid / 64;
                let dst = dq.add(base + row as usize * stride + (tid % 64) as usize);
                ptx_asm!(
                    "{ .reg .u64 p; red.relaxed.gpu.global.add.f32 [%0], %1; add.u64 p, %0, %5; red.relaxed.gpu.global.add.f32 [p], %2; add.u64 p, p, %5; red.relaxed.gpu.global.add.f32 [p], %3; add.u64 p, p, %5; red.relaxed.gpu.global.add.f32 [p], %4; }",
                    in("l") dst, in("f") x[0], in("f") x[1], in("f") x[2], in("f") x[3], in("l") (4 * stride * 4) as u64, clobber("memory")
                );
                d += 1;
            }
            return;
        }
        thread::sync_threads();
        let staging_address = shared_address(staging.cast());
        let row = (warp % 4) * 16 + lane / 4;
        // Pair-aligned swizzle keeps each 128-byte vector-store transaction conflict-free.
        let swizzle = (row & 3) * 8;
        macro_rules! store {
            ($d:expr) => {
                let col = (warp / 4) * 32 + $d * 8 + (lane % 4) * 2;
                let v = acc[$d];
                let address = staging_address + (row * 64 + (col ^ swizzle)) * 4;
                let upper_address = address + 2048;
                ptx_asm!(
                    "st.shared.v2.f32 [%0], {%1, %2}; st.shared.v2.f32 [%3], {%4, %5};",
                    in("r") address, in("f") v[0], in("f") v[1],
                    in("r") upper_address, in("f") v[2], in("f") v[3],
                    clobber("memory"),
                );
            };
        }
        store!(0);
        store!(1);
        store!(2);
        store!(3);
        thread::sync_threads();
        let col = tid % 64;
        let rows = (cfg.seqlen_q - query_start).min(64);
        let mut i = 0;
        while i < 16 {
            thread::__unroll_config::<0>();
            let row = tid / 64 + i * 4;
            if EVEN || row < rows {
                let swizzle = (row & 3) * 8;
                let address = staging_address + (row * 64 + (col ^ swizzle)) * 4;
                let value: f32;
                ptx_asm!("ld.shared.f32 %0, [%1];", out("=f") value,
                    in("r") address, clobber("memory"));
                let index = (((b as usize * cfg.seqlen_q as usize + (query_start + row) as usize)
                    * cfg.h as usize
                    + head as usize)
                    * 64)
                    + col as usize;
                let dest = dq.add(index);
                ptx_asm!("red.global.add.f32 [%0], %1;", in("l") dest, in("f") value, clobber("memory"));
            }
            i += 1;
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn core<const BF16: bool, const EVEN: bool>(
    q: *const u32,
    k: *const u32,
    v: *const u32,
    dout: *const u32,
    lse: *const f32,
    delta: *const f32,
    dk: *mut u32,
    dv: *mut u32,
    dq: *mut f32,
    mut cfg: FwdConfig,
    do_strides: Strides,
    gradient_scale: f32,
    packed_strides: Option<(u32, u32)>,
) {
    unsafe {
        cfg.hs_u32 = 32;
        cfg.gqa = 1;
        let tid = thread::threadIdx_x();
        let lane = tid % 32;
        let warp = tid / 32;
        let b = if packed_strides.is_some() {
            0
        } else {
            thread::blockIdx_z()
        };
        let head = thread::blockIdx_y();
        let bh = b * cfg.h + head;
        let key_start = thread::blockIdx_x() * 128;
        let group = (warp / 2) * 32;
        let scratch = DynamicSharedArray::<u32, 16>::get() as usize as *mut u32;
        let ks = scratch;
        let ps = scratch.add(4096);
        let ds = scratch.add(8192);
        let qs = scratch.add(12288);
        let vs = scratch.add(20480);
        copy_rows::<128, EVEN>(
            k,
            ks,
            b as usize * cfg.k.batch as usize + head as usize * cfg.k.head as usize,
            key_start,
            cfg.k.seq,
            cfg.seqlen_k,
            32,
            tid,
        );
        copy_rows::<128, EVEN>(
            v,
            vs,
            b as usize * cfg.v.batch as usize + head as usize * cfg.v.head as usize,
            key_start,
            cfg.v.seq,
            cfg.seqlen_k,
            32,
            tid,
        );
        ptx_asm!(
            "cp.async.commit_group; cp.async.wait_group 0;",
            clobber("memory")
        );
        thread::sync_threads();
        let va = shared_address(vs);
        let mut value_fragments = [[[0u32; 2]; 4]; 4];
        let mut kc = 0;
        while kc < 4 {
            thread::__unroll_config::<0>();
            let mut nt = 0;
            while nt < 4 {
                thread::__unroll_config::<0>();
                value_fragments[kc][nt] = load_b(va, group, lane, kc as u32, nt as u32);
                nt += 1;
            }
            kc += 1;
        }
        let (lse_stride, delta_stride) =
            packed_strides.unwrap_or((cfg.seqlen_q, cfg.seqlen_q.div_ceil(128) * 128));
        let off = cfg.seqlen_k as i64 - cfg.seqlen_q as i64;
        let first = if cfg.window_right >= 0 {
            (key_start as i64 - off - cfg.window_right as i64).clamp(0, cfg.seqlen_q as i64) as u32
                / 64
        } else {
            0
        };
        let end = cfg.seqlen_q.div_ceil(64);
        let shift = if cfg.window_right < 0 && cfg.seqlen_q >= 2048 && end > first {
            (key_start / 64) % (end - first)
        } else {
            0
        };
        let block_id = |block: u32| {
            let n = block + shift;
            if n >= end { n - (end - first) } else { n }
        };
        let qb = b as usize * cfg.q.batch as usize + head as usize * cfg.q.head as usize;
        let dob = b as usize * do_strides.batch as usize + head as usize * do_strides.head as usize;
        let mut acc_k = [[0.0f32; 4]; 8];
        let mut acc_v = [[0.0f32; 4]; 8];
        if first < end {
            copy_rows::<64, EVEN>(
                q,
                qs,
                qb,
                block_id(first) * 64,
                cfg.q.seq,
                cfg.seqlen_q,
                32,
                tid,
            );
            copy_rows::<64, EVEN>(
                dout,
                qs.add(2048),
                dob,
                block_id(first) * 64,
                do_strides.seq,
                cfg.seqlen_q,
                32,
                tid,
            );
            ptx_asm!("cp.async.commit_group;", clobber("memory"));
        }
        let mut block = first;
        while block < end {
            let query_start = block_id(block) * 64;
            let qtile = qs.add(((block - first) % 2) as usize * 4096);
            let dotile = qtile.add(2048);
            ptx_asm!("cp.async.wait_group 0;", clobber("memory"));
            thread::sync_threads();
            if block + 1 < end {
                let next = qs.add(((block - first + 1) % 2) as usize * 4096);
                copy_rows::<64, EVEN>(
                    q,
                    next,
                    qb,
                    block_id(block + 1) * 64,
                    cfg.q.seq,
                    cfg.seqlen_q,
                    32,
                    tid,
                );
                copy_rows::<64, EVEN>(
                    dout,
                    next.add(2048),
                    dob,
                    block_id(block + 1) * 64,
                    do_strides.seq,
                    cfg.seqlen_q,
                    32,
                    tid,
                );
                ptx_asm!("cp.async.commit_group;", clobber("memory"));
            }
            let mut rf = 0;
            while rf < 2 {
                thread::__unroll_config::<0>();
                let mut p = score_row::<BF16, false>(qtile, ks, tid, rf as u32, &value_fragments);
                let row = query_start + (warp % 2) * 16 + rf as u32 * 32 + lane / 4;
                let l0 = if EVEN || row < cfg.seqlen_q {
                    *lse.add(bh as usize * lse_stride as usize + row as usize)
                } else {
                    f32::INFINITY
                };
                let l1 = if EVEN || row + 8 < cfg.seqlen_q {
                    *lse.add(bh as usize * lse_stride as usize + row as usize + 8)
                } else {
                    f32::INFINITY
                };
                // Only full key tiles below the diagonal can omit masking.
                // Padded query rows already have infinite LSE and yield zero.
                if (EVEN || key_start + 128 <= cfg.seqlen_k)
                    && cfg.window_right >= 0
                    && query_start as i64 + off >= key_start as i64 + 128
                {
                    let mut nt = 0;
                    while nt < 4 {
                        thread::__unroll_config::<0>();
                        let f = p[nt];
                        let scale = cfg.scale * crate::device::forward::LOG2_E;
                        let lo = l0 * crate::device::forward::LOG2_E;
                        let hi = l1 * crate::device::forward::LOG2_E;
                        p[nt] = [
                            crate::device::inlined::fast_exp2(f[0] * scale - lo),
                            crate::device::inlined::fast_exp2(f[1] * scale - lo),
                            crate::device::inlined::fast_exp2(f[2] * scale - hi),
                            crate::device::inlined::fast_exp2(f[3] * scale - hi),
                        ];
                        nt += 1;
                    }
                } else {
                    let mut nt = 0;
                    while nt < 4 {
                        thread::__unroll_config::<0>();
                        let col = key_start + group + nt as u32 * 8 + (lane % 4) * 2;
                        let f = p[nt];
                        p[nt] = [
                            probability::<EVEN>(f[0], row, col, l0, &cfg, 0.0, false),
                            probability::<EVEN>(f[1], row, col + 1, l0, &cfg, 0.0, false),
                            probability::<EVEN>(f[2], row + 8, col, l1, &cfg, 0.0, false),
                            probability::<EVEN>(f[3], row + 8, col + 1, l1, &cfg, 0.0, false),
                        ];
                        nt += 1;
                    }
                }
                let dp = score_row::<BF16, true>(dotile, vs, tid, rf as u32, &value_fragments);
                let row = (warp % 2) * 16 + rf as u32 * 32 + lane / 4;
                let d0 =
                    *delta.add(bh as usize * delta_stride as usize + (query_start + row) as usize);
                let d1 = *delta
                    .add(bh as usize * delta_stride as usize + (query_start + row + 8) as usize);
                let mut nt = 0;
                while nt < 4 {
                    thread::__unroll_config::<0>();
                    let col = (group + nt as u32 * 8) / 2 + lane % 4;
                    let f = p[nt];
                    let g = dp[nt];
                    let i0 = (offset(row, col, 64) / 4) as usize;
                    let i1 = (offset(row + 8, col, 64) / 4) as usize;
                    *ps.add(i0) = cvt::<BF16>(f[0], f[1]);
                    *ps.add(i1) = cvt::<BF16>(f[2], f[3]);
                    *ds.add(i0) = cvt::<BF16>(f[0] * (g[0] - d0), f[1] * (g[1] - d0));
                    *ds.add(i1) = cvt::<BF16>(f[2] * (g[2] - d1), f[3] * (g[3] - d1));
                    nt += 1;
                }
                rf += 1;
            }
            thread::sync_threads();
            accumulate_kv::<BF16>(&mut acc_v, ps, dotile, tid);
            // Packed tiles benefit from completing dK before issuing dQ
            // atomics; dense tiles retain their measured overlap schedule.
            if packed_strides.is_some() {
                accumulate_kv::<BF16>(&mut acc_k, ds, qtile, tid);
            }
            // Padded packed tiles and full dense tiles write coalesced dQ
            // accumulators directly. Unpadded tails reuse P for staging.
            if EVEN || packed_strides.is_some() {
                accumulate_q::<BF16, EVEN, true>(
                    ds,
                    ks,
                    ps.cast(),
                    dq,
                    &cfg,
                    b,
                    head,
                    query_start,
                    tid,
                );
            } else {
                accumulate_q::<BF16, EVEN, false>(
                    ds,
                    ks,
                    ps.cast(),
                    dq,
                    &cfg,
                    b,
                    head,
                    query_start,
                    tid,
                );
            }
            if packed_strides.is_none() {
                accumulate_kv::<BF16>(&mut acc_k, ds, qtile, tid);
            }
            block += 1;
        }
        let key_base = key_start + (warp % 4) * 16 + lane / 4;
        let base = (b as usize * cfg.seqlen_k as usize * cfg.h as usize + head as usize) * 32;
        let mut d = 0;
        while d < 8 {
            thread::__unroll_config::<0>();
            let key = key_base + (d as u32 / 4) * 64;
            let col = (warp / 4) * 16 + (d as u32 % 4) * 4 + lane % 4;
            if EVEN || key < cfg.seqlen_k {
                let i = base + key as usize * cfg.h as usize * 32 + col as usize;
                *dk.add(i) =
                    cvt::<BF16>(acc_k[d][0] * gradient_scale, acc_k[d][1] * gradient_scale);
                *dv.add(i) = cvt::<BF16>(acc_v[d][0], acc_v[d][1]);
            }
            if EVEN || key + 8 < cfg.seqlen_k {
                let i = base + (key + 8) as usize * cfg.h as usize * 32 + col as usize;
                *dk.add(i) =
                    cvt::<BF16>(acc_k[d][2] * gradient_scale, acc_k[d][3] * gradient_scale);
                *dv.add(i) = cvt::<BF16>(acc_v[d][2], acc_v[d][3]);
            }
            d += 1;
        }
    }
}
