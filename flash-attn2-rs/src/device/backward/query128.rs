use super::common::copy_rows_swizzled_cta;
use crate::device::config::{FwdConfig, LaunchTile, Strides};
use crate::device::forward::LOG2_E;
use crate::device::inlined::{
    cvt, fast_exp2, ldmatrix_x4_shared, ldmatrix_x4_trans_shared, mma, shared_address,
};
use cuda_device::shared::DynamicSharedArray;
use cuda_device::{ptx_asm, thread};

// Host dispatch requires full 64-row query/key tiles, no ALiBi or softcap.
// Eight warps: scores use four key groups by two query groups; dK/dV
// use two key groups by four dimension groups. P/dS staging bridges the layouts.
#[inline(always)]
pub(super) unsafe fn core<const BF16: bool, const MODE: u32>(
    q: *const u32,
    k: *const u32,
    v: *const u32,
    dout: *const u32,
    lse: *const f32,
    delta: *const f32,
    dk: *mut u32,
    dv: *mut u32,
    dq: *mut f32,
    cfg: FwdConfig,
    do_strides: Strides,
    scale: f32,
) {
    unsafe {
        let tid = thread::threadIdx_x();
        let lane = tid % 32;
        let warp = tid / 32;
        let half = warp / 4;
        let b = thread::blockIdx_z();
        let head = thread::blockIdx_y();
        let kh = head / cfg.gqa;
        let key_start = thread::blockIdx_x() * 64;
        let tile = LaunchTile {
            tid,
            lane_id: lane,
            bh: b * cfg.h + head,
            group: lane >> 2,
            pair: lane & 3,
            q_row_base: key_start,
            warp_row: (warp % 4) * 16,
            b,
            head,
            kv_head: kh,
        };
        let key = key_start + tile.warp_row + tile.group;
        let scratch = DynamicSharedArray::<u32, 16>::get() as usize as *mut u32;
        let ks = scratch;
        let vs = ks.add(64 * 64);
        let qs = vs.add(64 * 64);
        let dos = qs.add(64 * 64);
        let ps = dos.add(64 * 64);
        let ds = ps.add(64 * 32);
        copy_rows_swizzled_cta::<64, 64, 128>(
            k,
            ks,
            b as usize * cfg.k.batch as usize + kh as usize * cfg.k.head as usize,
            key_start,
            cfg.k.seq,
            cfg.seqlen_k,
            64,
            tid,
        );
        copy_rows_swizzled_cta::<64, 64, 128>(
            v,
            vs,
            b as usize * cfg.v.batch as usize + kh as usize * cfg.v.head as usize,
            key_start,
            cfg.v.seq,
            cfg.seqlen_k,
            64,
            tid,
        );
        ptx_asm!(
            "cp.async.commit_group; cp.async.wait_group 0;",
            clobber("memory")
        );
        thread::sync_threads();
        let mut ak = [[0.0f32; 4]; 8];
        let mut av = [[0.0f32; 4]; 8];
        let off = cfg.seqlen_k as i64 - cfg.seqlen_q as i64;
        let first = if MODE == 1 {
            (key_start as i64 - off).clamp(0, cfg.seqlen_q as i64) as u32 / 64
        } else {
            0
        };
        let end = cfg.seqlen_q.div_ceil(64);
        let rounded_q = cfg.seqlen_q.div_ceil(128) * 128;
        // Prime both inputs. Later iterations overlap their copies with the
        // gradient GEMMs, using a second Q tile after P/dS in shared memory.
        if first < end {
            copy_rows_swizzled_cta::<64, 64, 128>(
                q,
                qs,
                b as usize * cfg.q.batch as usize + head as usize * cfg.q.head as usize,
                first * 64,
                cfg.q.seq,
                cfg.seqlen_q,
                64,
                tid,
            );

            copy_rows_swizzled_cta::<64, 64, 128>(
                dout,
                dos,
                b as usize * do_strides.batch as usize + head as usize * do_strides.head as usize,
                first * 64,
                do_strides.seq,
                cfg.seqlen_q,
                64,
                tid,
            );
            ptx_asm!("cp.async.commit_group;", clobber("memory"));
        }
        let mut block = first;
        while block < end {
            let query_start = block * 64;
            let qtile = if (block - first) % 2 == 0 {
                qs
            } else {
                scratch.add(20480)
            };
            ptx_asm!("cp.async.wait_group 0;", clobber("memory"));
            thread::sync_threads();
            let mut scores = [[0.0f32; 4]; 4];
            score_tile::<BF16>(&mut scores, ks, qtile.add(half as usize * 32 * 64), &tile);
            // Keep the noncausal schedule intact; causal tiles below the
            // diagonal can skip all per-score mask comparisons.
            if MODE == 0 {
                let mut nt = 0;
                while nt < 4 {
                    thread::__unroll_config::<0>();
                    let row = query_start + half * 32 + nt as u32 * 8 + tile.pair * 2;
                    let l0 = *lse.add(tile.bh as usize * cfg.seqlen_q as usize + row as usize);
                    let l1 = *lse.add(tile.bh as usize * cfg.seqlen_q as usize + row as usize + 1);
                    let x = scores[nt];
                    scores[nt] = [
                        probability::<MODE>(x[0], row, key, l0, &cfg),
                        probability::<MODE>(x[1], row + 1, key, l1, &cfg),
                        probability::<MODE>(x[2], row, key + 8, l0, &cfg),
                        probability::<MODE>(x[3], row + 1, key + 8, l1, &cfg),
                    ];
                    nt += 1;
                }
            } else {
                if MODE == 1 && query_start as i64 >= key_start as i64 + 64 - off {
                    scale_scores::<0>(&mut scores, lse, &cfg, &tile, query_start, key);
                } else {
                    scale_scores::<MODE>(&mut scores, lse, &cfg, &tile, query_start, key);
                }
            }
            let mut dp = [[0.0f32; 4]; 4];
            score_tile::<BF16>(&mut dp, vs, dos.add(half as usize * 32 * 64), &tile);
            let mut nt = 0;
            while nt < 4 {
                thread::__unroll_config::<0>();
                let row = query_start + half * 32 + nt as u32 * 8 + tile.pair * 2;
                let d0 = *delta.add(tile.bh as usize * rounded_q as usize + row as usize);
                let d1 = *delta.add(tile.bh as usize * rounded_q as usize + row as usize + 1);
                let p = scores[nt];
                let x = dp[nt];
                let local_key = tile.warp_row + tile.group;
                let col = half * 16 + nt as u32 * 4 + tile.pair;
                let lo = local_key * 32 + (col ^ ((local_key & 7) * 4));
                let hi = (local_key + 8) * 32 + (col ^ (((local_key + 8) & 7) * 4));
                *ps.add(lo as usize) = cvt::<BF16>(p[0], p[1]);
                *ps.add(hi as usize) = cvt::<BF16>(p[2], p[3]);
                *ds.add(lo as usize) = cvt::<BF16>(p[0] * (x[0] - d0), p[1] * (x[1] - d1));
                *ds.add(hi as usize) = cvt::<BF16>(p[2] * (x[2] - d0), p[3] * (x[3] - d1));
                nt += 1;
            }
            if block + 1 < end {
                copy_rows_swizzled_cta::<64, 64, 128>(
                    q,
                    if (block - first) % 2 == 0 {
                        scratch.add(20480)
                    } else {
                        qs
                    },
                    b as usize * cfg.q.batch as usize + head as usize * cfg.q.head as usize,
                    (block + 1) * 64,
                    cfg.q.seq,
                    cfg.seqlen_q,
                    64,
                    tid,
                );
                ptx_asm!("cp.async.commit_group;", clobber("memory"));
            }
            thread::sync_threads();
            value_tile::<BF16>(&mut av, ps, dos, &tile);
            // Every warp must finish reading dO before its shared tile is reused.
            thread::sync_threads();
            if block + 1 < end {
                copy_rows_swizzled_cta::<64, 64, 128>(
                    dout,
                    dos,
                    b as usize * do_strides.batch as usize
                        + head as usize * do_strides.head as usize,
                    (block + 1) * 64,
                    do_strides.seq,
                    cfg.seqlen_q,
                    64,
                    tid,
                );
                ptx_asm!("cp.async.commit_group;", clobber("memory"));
            }
            dq_tile::<BF16>(ks, ds, dq, &cfg, &tile, query_start);
            value_tile::<BF16>(&mut ak, ds, qtile, &tile);
            block += 1;
        }
        let heads = cfg.h;
        let stride = heads as usize * 64;
        let base = b as usize * cfg.seqlen_k as usize * stride + head as usize * 64;
        let mut dim = 0;
        while dim < 8 {
            thread::__unroll_config::<0>();
            let key = key_start + (warp % 2) * 32 + tile.group + (dim / 4) as u32 * 16;
            let col = (warp / 2) as usize * 16 + (dim % 4) * 4 + tile.pair as usize;
            let x = ak[dim];
            let y = av[dim];
            let index = base + key as usize * stride + col;
            *dk.add(index) = cvt::<BF16>(x[0] * scale, x[1] * scale);
            *dv.add(index) = cvt::<BF16>(y[0], y[1]);
            let index = base + (key + 8) as usize * stride + col;
            *dk.add(index) = cvt::<BF16>(x[2] * scale, x[3] * scale);
            *dv.add(index) = cvt::<BF16>(y[2], y[3]);
            dim += 1;
        }
    }
}

#[inline(always)]
unsafe fn dq_tile<const BF16: bool>(
    keys: *const u32,
    ds: *const u32,
    dq: *mut f32,
    cfg: &FwdConfig,
    tile: &LaunchTile,
    query_start: u32,
) {
    unsafe {
        let lane = tile.lane_id;
        let warp = tile.tid / 32;
        let ka = shared_address(keys);
        let sa = shared_address(ds);
        let mut acc = [[0.0f32; 4]; 8];
        let mut chunk = 0;
        while chunk < 4 {
            thread::__unroll_config::<0>();
            let kr = chunk * 16 + lane % 8 + (lane / 16) * 8;
            let qr = (warp % 2) * 32 + ((lane / 8) % 2) * 8;
            let a = ldmatrix_x4_trans_shared(sa + (kr * 32 + ((qr / 2) ^ ((kr & 7) * 4))) * 4);
            let a1 =
                ldmatrix_x4_trans_shared(sa + (kr * 32 + (((qr + 16) / 2) ^ ((kr & 7) * 4))) * 4);
            let mut d = 0;
            while d < 4 {
                thread::__unroll_config::<0>();
                let row = chunk * 16 + lane % 16;
                let col = (warp / 2) * 16 + d * 4 + (lane / 16) * 4;
                let b = ldmatrix_x4_trans_shared(ka + (row * 64 + (col ^ ((row & 7) * 4))) * 4);
                acc[d as usize] = mma::<BF16>(acc[d as usize], a, [b[0], b[1]]);
                acc[d as usize + 1] = mma::<BF16>(acc[d as usize + 1], a, [b[2], b[3]]);
                acc[d as usize + 4] = mma::<BF16>(acc[d as usize + 4], a1, [b[0], b[1]]);
                acc[d as usize + 5] = mma::<BF16>(acc[d as usize + 5], a1, [b[2], b[3]]);
                d += 2;
            }
            chunk += 1;
        }
        // Keep each atomic instruction contiguous across all lanes. The merge
        // converts this temporary accumulator layout back to logical Q rows.
        let stride = cfg.h as usize * 128;
        let base = ((tile.b as usize * cfg.seqlen_q as usize + query_start as usize)
            * cfg.h as usize
            + tile.head as usize)
            * 128;
        let mut d = 0;
        while d < 8 {
            thread::__unroll_config::<0>();
            let x = acc[d as usize];
            let row = d * 8 + tile.tid / 128;
            let dst = dq.add(base + row as usize * stride + (tile.tid % 128) as usize);
            ptx_asm!(
                "{ .reg .u64 p; red.relaxed.gpu.global.add.f32 [%0], %1; add.u64 p, %0, %5; red.relaxed.gpu.global.add.f32 [p], %2; add.u64 p, p, %5; red.relaxed.gpu.global.add.f32 [p], %3; add.u64 p, p, %5; red.relaxed.gpu.global.add.f32 [p], %4; }",
                in("l") dst, in("f") x[0], in("f") x[1], in("f") x[2], in("f") x[3], in("l") (2 * stride * 4) as u64, clobber("memory")
            );
            d += 1;
        }
    }
}

#[inline(always)]
unsafe fn score_tile<const BF16: bool>(
    scores: &mut [[f32; 4]; 4],
    a_shared: *const u32,
    b_shared: *const u32,
    tile: &LaunchTile,
) {
    unsafe {
        let aa = shared_address(a_shared);
        let ba = shared_address(b_shared);
        let lane = tile.lane_id;
        let ar = tile.warp_row + (lane & 7) + ((lane >> 3) & 1) * 8;
        let mut kc = 0;
        while kc < 8 {
            thread::__unroll_config::<0>();
            let ac = kc * 8 + (lane >> 4) * 4;
            let a = ldmatrix_x4_shared(aa + (ar * 64 + (ac ^ ((ar & 7) * 4))) * 4);
            let mut nt = 0;
            while nt < 4 {
                thread::__unroll_config::<0>();
                let row = nt * 8 + (lane & 7) + (lane >> 4) * 8;
                let col = kc * 8 + ((lane & 15) >> 3) * 4;
                let b = ldmatrix_x4_shared(ba + (row * 64 + (col ^ ((row & 7) * 4))) * 4);
                scores[nt as usize] = mma::<BF16>(scores[nt as usize], a, [b[0], b[1]]);
                scores[nt as usize + 1] = mma::<BF16>(scores[nt as usize + 1], a, [b[2], b[3]]);
                nt += 2;
            }
            kc += 1;
        }
    }
}

#[inline(always)]
unsafe fn value_tile<const BF16: bool>(
    acc: &mut [[f32; 4]; 8],
    a_shared: *const u32,
    b_shared: *const u32,
    tile: &LaunchTile,
) {
    unsafe {
        let aa = shared_address(a_shared);
        let ba = shared_address(b_shared);
        let lane = tile.lane_id;
        let warp = tile.tid / 32;
        let ar = (warp % 2) * 32 + (lane & 7) + ((lane >> 3) & 1) * 8;
        let mut kc = 0;
        while kc < 4 {
            thread::__unroll_config::<0>();
            let ac = kc * 8 + (lane >> 4) * 4;
            let a = ldmatrix_x4_shared(aa + (ar * 32 + (ac ^ ((ar & 7) * 4))) * 4);
            let a1 = ldmatrix_x4_shared(aa + ((ar + 16) * 32 + (ac ^ (((ar + 16) & 7) * 4))) * 4);
            let mut d = 0;
            while d < 4 {
                thread::__unroll_config::<0>();
                let row = kc * 16 + (lane & 15);
                let col = (warp / 2) * 16 + d * 4 + (lane >> 4) * 4;
                let b = ldmatrix_x4_trans_shared(ba + (row * 64 + (col ^ ((row & 7) * 4))) * 4);
                acc[d as usize] = mma::<BF16>(acc[d as usize], a, [b[0], b[1]]);
                acc[d as usize + 1] = mma::<BF16>(acc[d as usize + 1], a, [b[2], b[3]]);
                acc[d as usize + 4] = mma::<BF16>(acc[d as usize + 4], a1, [b[0], b[1]]);
                acc[d as usize + 5] = mma::<BF16>(acc[d as usize + 5], a1, [b[2], b[3]]);
                d += 2;
            }
            kc += 1;
        }
    }
}

#[inline(always)]
unsafe fn scale_scores<const MODE: u32>(
    scores: &mut [[f32; 4]; 4],
    lse: *const f32,
    cfg: &FwdConfig,
    tile: &LaunchTile,
    query_start: u32,
    key: u32,
) {
    unsafe {
        let mut nt = 0;
        while nt < 4 {
            thread::__unroll_config::<0>();
            let row = query_start + (tile.tid / 128) * 32 + nt as u32 * 8 + tile.pair * 2;
            let l0 = *lse.add(tile.bh as usize * cfg.seqlen_q as usize + row as usize);
            let l1 = *lse.add(tile.bh as usize * cfg.seqlen_q as usize + row as usize + 1);
            let x = scores[nt];
            scores[nt] = [
                probability::<MODE>(x[0], row, key, l0, cfg),
                probability::<MODE>(x[1], row + 1, key, l1, cfg),
                probability::<MODE>(x[2], row, key + 8, l0, cfg),
                probability::<MODE>(x[3], row + 1, key + 8, l1, cfg),
            ];
            nt += 1;
        }
    }
}

#[inline(always)]
fn probability<const MODE: u32>(raw: f32, row: u32, col: u32, lse: f32, cfg: &FwdConfig) -> f32 {
    if MODE == 1 && col as i64 > row as i64 + cfg.seqlen_k as i64 - cfg.seqlen_q as i64 {
        return 0.0;
    }
    fast_exp2((raw * cfg.scale - lse) * LOG2_E)
}
