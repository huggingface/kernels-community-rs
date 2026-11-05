//! D64 backward with 256-key, 16-query tiles to reduce partial dQ updates.
use super::common::probability_gradient;
use crate::device::config::{FwdConfig, Strides};
use crate::device::inlined::{
    cvt, ldmatrix_x2_shared, ldmatrix_x2_trans_shared, ldmatrix_x4_shared,
    ldmatrix_x4_trans_shared, mma, shared_address,
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
                (raw * cfg.scale - lse) * crate::device::forward::LOG2_E,
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
unsafe fn score_row<const BF16: bool>(
    queries: *const u32,
    keys: *const u32,
    tid: u32,
    row_fragment: u32,
) -> [[f32; 4]; 4] {
    unsafe {
        let lane = tid % 32;
        let qa = shared_address(queries);
        let ka = shared_address(keys);
        let group = (tid / 32) * 32;
        let row = (lane & 15) + row_fragment * 16;
        let mut acc = [[0.0f32; 4]; 4];
        macro_rules! chunk {
            ($kc:expr) => {
                let a = ldmatrix_x4_shared(qa + offset(row, $kc * 8 + (lane / 16) * 4, 32));
                acc[0] = mma::<BF16>(acc[0], a, load_b(ka, group, lane, $kc, 0));
                acc[1] = mma::<BF16>(acc[1], a, load_b(ka, group, lane, $kc, 1));
                acc[2] = mma::<BF16>(acc[2], a, load_b(ka, group, lane, $kc, 2));
                acc[3] = mma::<BF16>(acc[3], a, load_b(ka, group, lane, $kc, 3));
            };
        }
        chunk!(0);
        chunk!(1);
        chunk!(2);
        chunk!(3);
        acc
    }
}

// P/dS are [query, key]; each warp accumulates 32 key rows and 64 dimensions.
#[inline(always)]
unsafe fn accumulate_kv<const BF16: bool>(
    acc: &mut [[f32; 4]; 16],
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
        while kc < 1 {
            thread::__unroll_config::<0>();
            let row = kc * 16 + lane % 8 + (lane / 16) * 8;
            let col = warp * 8 + ((lane / 8) % 2) * 4;
            let a0 = ldmatrix_x4_trans_shared(pa + offset(row, col, 128));
            let a1 = ldmatrix_x4_trans_shared(pa + offset(row, col + 64, 128));
            let mut d = 0;
            while d < 8 {
                thread::__unroll_config::<0>();
                let b = ldmatrix_x4_trans_shared(
                    qa + offset(kc * 16 + lane % 16, d * 4 + (lane / 16) * 4, 32),
                );
                acc[d as usize] = mma::<BF16>(acc[d as usize], a0, [b[0], b[1]]);
                acc[d as usize + 1] = mma::<BF16>(acc[d as usize + 1], a0, [b[2], b[3]]);
                acc[d as usize + 8] = mma::<BF16>(acc[d as usize + 8], a1, [b[0], b[1]]);
                acc[d as usize + 9] = mma::<BF16>(acc[d as usize + 9], a1, [b[2], b[3]]);
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
        let mut acc = [[0.0f32; 4]; 1];
        let mut kc = 0;
        while kc < 16 {
            thread::__unroll_config::<0>();
            let a = ldmatrix_x4_shared(da + offset(lane % 16, kc * 8 + (lane / 16) * 4, 128));
            let bv = ldmatrix_x2_trans_shared(ka + offset(kc * 16 + lane % 16, warp * 4, 32));
            acc[0] = mma::<BF16>(acc[0], a, bv);
            kc += 1;
        }
        // Full dense tiles use a temporary layout with contiguous warp atomics.
        // The wide D64 merge restores the logical 16-query by 64-dimension tile.
        if PERMUTE {
            let stride = cfg.h as usize * 64;
            let base = ((b as usize * cfg.seqlen_q as usize + query_start as usize)
                * cfg.h as usize
                + head as usize)
                * 64;
            let x = acc[0];
            let dst = dq.add(base + (tid / 64) as usize * stride + (tid % 64) as usize);
            ptx_asm!(
                "{ .reg .u64 p; red.relaxed.gpu.global.add.f32 [%0], %1; add.u64 p, %0, %5; red.relaxed.gpu.global.add.f32 [p], %2; add.u64 p, p, %5; red.relaxed.gpu.global.add.f32 [p], %3; add.u64 p, p, %5; red.relaxed.gpu.global.add.f32 [p], %4; }",
                in("l") dst, in("f") x[0], in("f") x[1], in("f") x[2], in("f") x[3], in("l") (4 * stride * 4) as u64, clobber("memory")
            );
            return;
        }
        thread::sync_threads();
        let row = lane / 4;
        let swizzle = ((row & 3) * 8) | ((row >> 2) & 1);
        macro_rules! store {
            ($d:expr) => {
                let col = warp * 8 + $d * 8 + (lane % 4) * 2;
                let v = acc[$d];
                *staging.add((row * 64 + (col ^ swizzle)) as usize) = v[0];
                *staging.add((row * 64 + ((col + 1) ^ swizzle)) as usize) = v[1];
                *staging.add(((row + 8) * 64 + (col ^ swizzle)) as usize) = v[2];
                *staging.add(((row + 8) * 64 + ((col + 1) ^ swizzle)) as usize) = v[3];
            };
        }
        store!(0);
        thread::sync_threads();
        let col = tid % 64;
        let rows = (cfg.seqlen_q - query_start).min(16);
        let mut i = 0;
        while i < 4 {
            thread::__unroll_config::<0>();
            let row = tid / 64 + i * 4;
            if EVEN || row < rows {
                let swizzle = ((row & 3) * 8) | ((row >> 2) & 1);
                let value = *staging.add((row * 64 + (col ^ swizzle)) as usize);
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
        let key_start = thread::blockIdx_x() * 256;
        let group = warp * 32;
        let scratch = DynamicSharedArray::<u32, 16>::get() as usize as *mut u32;
        let ks = scratch;
        let ps = scratch.add(8192);
        let ds = scratch.add(10240);
        let qs = scratch.add(12288);
        let vs = scratch.add(14336);
        copy_rows::<256, EVEN>(
            k,
            ks,
            b as usize * cfg.k.batch as usize + head as usize * cfg.k.head as usize,
            key_start,
            cfg.k.seq,
            cfg.seqlen_k,
            32,
            tid,
        );
        copy_rows::<256, EVEN>(
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
        let (lse_stride, delta_stride) =
            packed_strides.unwrap_or((cfg.seqlen_q, cfg.seqlen_q.div_ceil(128) * 128));
        let off = cfg.seqlen_k as i64 - cfg.seqlen_q as i64;
        let first = if cfg.window_right >= 0 {
            (key_start as i64 - off - cfg.window_right as i64).clamp(0, cfg.seqlen_q as i64) as u32
                / 16
        } else {
            0
        };
        let end = cfg.seqlen_q.div_ceil(16);
        let shift = if cfg.window_right < 0 && cfg.seqlen_q >= 2048 && end > first {
            (key_start / 16) % (end - first)
        } else {
            0
        };
        let block_id = |block: u32| {
            let n = block + shift;
            if n >= end { n - (end - first) } else { n }
        };
        let qb = b as usize * cfg.q.batch as usize + head as usize * cfg.q.head as usize;
        let dob = b as usize * do_strides.batch as usize + head as usize * do_strides.head as usize;
        let mut acc_k = [[0.0f32; 4]; 16];
        let mut acc_v = [[0.0f32; 4]; 16];
        if first < end {
            copy_rows::<16, EVEN>(
                q,
                qs,
                qb,
                block_id(first) * 16,
                cfg.q.seq,
                cfg.seqlen_q,
                32,
                tid,
            );
            copy_rows::<16, EVEN>(
                dout,
                qs.add(512),
                dob,
                block_id(first) * 16,
                do_strides.seq,
                cfg.seqlen_q,
                32,
                tid,
            );
            ptx_asm!("cp.async.commit_group;", clobber("memory"));
        }
        let mut block = first;
        while block < end {
            let query_start = block_id(block) * 16;
            let qtile = qs.add(((block - first) % 2) as usize * 1024);
            let dotile = qtile.add(512);
            ptx_asm!("cp.async.wait_group 0;", clobber("memory"));
            thread::sync_threads();
            if block + 1 < end {
                let next = qs.add(((block - first + 1) % 2) as usize * 1024);
                copy_rows::<16, EVEN>(
                    q,
                    next,
                    qb,
                    block_id(block + 1) * 16,
                    cfg.q.seq,
                    cfg.seqlen_q,
                    32,
                    tid,
                );
                copy_rows::<16, EVEN>(
                    dout,
                    next.add(512),
                    dob,
                    block_id(block + 1) * 16,
                    do_strides.seq,
                    cfg.seqlen_q,
                    32,
                    tid,
                );
                ptx_asm!("cp.async.commit_group;", clobber("memory"));
            }
            let mut rf = 0;
            while rf < 1 {
                thread::__unroll_config::<0>();
                let mut p = score_row::<BF16>(qtile, ks, tid, rf as u32);
                let row = query_start + rf as u32 * 16 + lane / 4;
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
                let dp = score_row::<BF16>(dotile, vs, tid, rf as u32);
                let row = rf as u32 * 16 + lane / 4;
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
                    let i0 = (offset(row, col, 128) / 4) as usize;
                    let i1 = (offset(row + 8, col, 128) / 4) as usize;
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
            if packed_strides.is_some() {
                // Packed dQ has per-sequence padding, so every tile can write
                // coalesced accumulators before the dK GEMM.
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
                accumulate_kv::<BF16>(&mut acc_k, ds, qtile, tid);
            } else {
                accumulate_kv::<BF16>(&mut acc_k, ds, qtile, tid);
                accumulate_q::<BF16, EVEN, EVEN>(
                    ds,
                    ks,
                    qtile.cast(),
                    dq,
                    &cfg,
                    b,
                    head,
                    query_start,
                    tid,
                );
            }
            block += 1;
        }
        let key_base = key_start + warp * 16 + lane / 4;
        let base = (b as usize * cfg.seqlen_k as usize * cfg.h as usize + head as usize) * 32;
        let mut d = 0;
        while d < 16 {
            thread::__unroll_config::<0>();
            let key = key_base + (d as u32 / 8) * 128;
            let col = (d as u32 % 8) * 4 + lane % 4;
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
