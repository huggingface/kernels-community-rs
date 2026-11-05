use crate::device::config::{FwdConfig, LaunchTile};
use crate::device::dropout::Dropout;
use crate::device::forward::{LOG2_E, SmemLayout};
use crate::device::inlined::{fast_exp2, ldmatrix_x4_shared, shared_address, tanh_approx};
use cuda_device::async_copy::cp_async_ca_zfill_16;
use cuda_device::{ptx_asm, thread};

#[inline(always)]
pub(crate) unsafe fn copy_rows<const HD: usize, const ROWS: usize>(
    src: *const u32,
    dst: *mut u32,
    base: usize,
    row_start: u32,
    stride: u32,
    seq: u32,
    dim_pairs: u32,
    tid: u32,
) {
    unsafe { copy_rows_cta::<HD, ROWS, 64>(src, dst, base, row_start, stride, seq, dim_pairs, tid) }
}

#[inline(always)]
pub(crate) unsafe fn copy_rows_cta<const HD: usize, const ROWS: usize, const KEY_ROWS: usize>(
    src: *const u32,
    dst: *mut u32,
    base: usize,
    row_start: u32,
    stride: u32,
    seq: u32,
    dim_pairs: u32,
    tid: u32,
) {
    unsafe {
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
                dst.add(row * (HD / 2 + 4) + col),
                src.add(index) as *const u8,
                if valid { 16 } else { 0 },
            );
            i += KEY_ROWS * 8;
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn query_fragments<const NFQ: usize>(
    shared: *const u32,
    tile: &LaunchTile,
    layout: SmemLayout,
) -> [[u32; 4]; NFQ] {
    let mut fragments = [[0; 4]; NFQ];
    let row = tile.warp_row + (tile.lane_id & 7) + ((tile.lane_id >> 3) & 1) * 8;
    let col = (tile.lane_id >> 4) * 4;
    let address = shared_address(shared);
    let mut kc = 0;
    while kc < NFQ {
        thread::__unroll_config::<0>();
        // Caller supplies a synchronized tile matching layout and NFQ.
        fragments[kc] = unsafe {
            ldmatrix_x4_shared(
                address + ((row as usize * layout.stride + kc * 8 + col as usize) * 4) as u32,
            )
        };
        kc += 1;
    }
    fragments
}

// XOR swizzle of 16-byte groups; stores and ldmatrix share it.
#[inline(always)]
pub(super) unsafe fn copy_rows_swizzled<const ROWS: usize, const KSTRIDE: usize>(
    src: *const u32,
    dst: *mut u32,
    base: usize,
    row_start: u32,
    stride: u32,
    seq: u32,
    dim_pairs: u32,
    tid: u32,
) {
    unsafe {
        copy_rows_swizzled_cta::<ROWS, KSTRIDE, 64>(
            src, dst, base, row_start, stride, seq, dim_pairs, tid,
        )
    }
}

#[inline(always)]
pub(super) unsafe fn copy_rows_swizzled_cta<
    const ROWS: usize,
    const KSTRIDE: usize,
    const KEY_ROWS: usize,
>(
    src: *const u32,
    dst: *mut u32,
    base: usize,
    row_start: u32,
    stride: u32,
    seq: u32,
    dim_pairs: u32,
    tid: u32,
) {
    unsafe {
        let mut i = tid as usize * 4;
        while i < ROWS * KSTRIDE {
            let row = i / KSTRIDE;
            let col = i % KSTRIDE;
            let global_row = row_start as usize + row;
            let valid = global_row < seq as usize && col < dim_pairs as usize;
            let index = if valid {
                base + global_row * stride as usize + col
            } else {
                0
            };
            cp_async_ca_zfill_16(
                dst.add(row * KSTRIDE + (col ^ ((row & 7) * 4))),
                src.add(index).cast(),
                if valid { 16 } else { 0 },
            );
            i += KEY_ROWS * 8;
        }
    }
}

#[inline(always)]
pub(super) unsafe fn query_fragments_swizzled<const NFQ: usize, const KSTRIDE: usize>(
    shared: *const u32,
    tile: &LaunchTile,
) -> [[u32; 4]; NFQ] {
    let mut fragments = [[0; 4]; NFQ];
    let row = tile.warp_row + (tile.lane_id & 7) + ((tile.lane_id >> 3) & 1) * 8;
    let col = (tile.lane_id >> 4) * 4;
    let address = shared_address(shared);
    let mut kc = 0;
    while kc < NFQ {
        thread::__unroll_config::<0>();
        let index = row as usize * KSTRIDE + ((kc * 8 + col as usize) ^ ((row as usize & 7) * 4));
        // Caller provides a synchronized tile written with this XOR mapping.
        fragments[kc] = unsafe { ldmatrix_x4_shared(address + (index * 4) as u32) };
        kc += 1;
    }
    fragments
}

// Combined to avoid a second live score tile during dout*V.
#[inline(always)]
pub(crate) fn probability_gradient(
    raw: f32,
    row: u32,
    col: u32,
    lse: f32,
    cfg: &FwdConfig,
    slope: f32,
    gradient: bool,
) -> f32 {
    let center = row as i64 + cfg.seqlen_k as i64 - cfg.seqlen_q as i64;
    if row >= cfg.seqlen_q
        || col >= cfg.seqlen_k
        || (cfg.window_right >= 0 && col as i64 > center + cfg.window_right as i64)
        || (cfg.window_left >= 0 && (col as i64) < center - cfg.window_left as i64)
    {
        return 0.0;
    }
    let (mut score, derivative) = if cfg.tanh_scale > 0.0 {
        let t = tanh_approx(raw * cfg.tanh_scale);
        (t * cfg.scale, if gradient { 1.0 - t * t } else { 1.0 })
    } else {
        (raw * cfg.scale, 1.0)
    };
    if cfg.has_alibi != 0 {
        score += if cfg.alibi_causal != 0 {
            slope * col as f32
        } else {
            -slope * (center - col as i64).abs() as f32
        };
    }
    fast_exp2((score - lse) * LOG2_E) * derivative
}

// Forward Philox stream read in transposed score coordinates.
#[inline(always)]
pub(super) fn random_word(random: [u32; 4], index: u32) -> u32 {
    if index == 0 {
        random[0]
    } else if index == 1 {
        random[1]
    } else if index == 2 {
        random[2]
    } else {
        random[3]
    }
}

#[inline(always)]
pub(crate) fn transposed_keep(dropout: Dropout, bh: u32, query: u32, key: u32) -> [f32; 4] {
    if dropout.threshold == 255 {
        return [1.0; 4];
    }
    let lane = (query % 8) * 4 + (key % 8) / 2;
    let offset = dropout.offset.wrapping_add(bh as u64 * 32 + lane as u64);
    let a = dropout.random(query / 16, key / 32, offset);
    let b = dropout.random(query / 16, key / 32, offset.wrapping_add(4));
    let word = (key % 32) / 8;
    let shift = ((query % 16) / 8 * 2 + key % 2) * 8;
    [
        (((random_word(a, word) >> shift) & 255) <= dropout.threshold) as u32 as f32,
        (((random_word(b, word) >> shift) & 255) <= dropout.threshold) as u32 as f32,
        (((random_word(a, word + 1) >> shift) & 255) <= dropout.threshold) as u32 as f32,
        (((random_word(b, word + 1) >> shift) & 255) <= dropout.threshold) as u32 as f32,
    ]
}

// All lanes must participate; xor4 lanes share Philox counters.
#[inline(always)]
pub(crate) fn transposed_paired_words(dropout: Dropout, bh: u32, query: u32, key: u32) -> [u32; 4] {
    if dropout.threshold == 255 {
        return [0; 4];
    }
    let parity = key & 1;
    let lane = (query % 8) * 4 + (key % 8) / 2;
    let offset = dropout
        .offset
        .wrapping_add(bh as u64 * 32 + lane as u64 + parity as u64 * 4);
    let random = dropout.random(query / 16, key / 32, offset);
    let word = (key % 32) / 8;
    let own0 = random_word(random, word);
    let own1 = random_word(random, word + 1);
    let other0: u32;
    let other1: u32;
    unsafe {
        ptx_asm!("shfl.sync.bfly.b32 %0, %1, 4, 31, -1;", out("=r") other0, in("r") own0);
        ptx_asm!("shfl.sync.bfly.b32 %0, %1, 4, 31, -1;", out("=r") other1, in("r") own1);
    }
    let a0 = if parity == 0 { own0 } else { other0 };
    let b0 = if parity == 0 { other0 } else { own0 };
    let a1 = if parity == 0 { own1 } else { other1 };
    let b1 = if parity == 0 { other1 } else { own1 };
    [a0, b0, a1, b1]
}

#[inline(always)]
pub(crate) fn transposed_word_mask(
    words: [u32; 4],
    threshold: u32,
    query: u32,
    key: u32,
) -> [f32; 4] {
    let shift = ((query % 16) / 8 * 2 + (key & 1)) * 8;
    [
        ((words[0] >> shift & 255) <= threshold) as u32 as f32,
        ((words[1] >> shift & 255) <= threshold) as u32 as f32,
        ((words[2] >> shift & 255) <= threshold) as u32 as f32,
        ((words[3] >> shift & 255) <= threshold) as u32 as f32,
    ]
}

// dQ tiles and dS staging shared by the fused and dK/dV kernels.
#[inline(always)]
pub(super) unsafe fn fused_store_ds<const NQT: usize>(
    packed: &[[u32; 2]; NQT],
    shared: *mut u32,
    tid: u32,
) {
    unsafe {
        let lane = tid % 32;
        let key = (tid / 32) * 16 + lane / 4;
        let mut nt = 0;
        while nt < NQT {
            thread::__unroll_config::<0>();
            let query = nt as u32 * 8 + (lane % 4) * 2;
            *shared.add(key as usize * (NQT * 4) + (query / 2) as usize) = packed[nt as usize][0];
            *shared.add((key as usize + 8) * (NQT * 4) + (query / 2) as usize) =
                packed[nt as usize][1];
            nt += 1;
        }
    }
}
#[inline(always)]
unsafe fn fused_atomic_add(ptr: *mut f32, value: f32) {
    unsafe {
        ptx_asm!("red.relaxed.gpu.global.add.f32 [%0], %1;", in("l") ptr as u64, in("f") value, clobber("memory"));
    }
}
#[inline(always)]
pub(super) unsafe fn fused_dq_tile_d64<
    const BF16: bool,
    const HD: usize,
    const NFQ: usize,
    const KSTRIDE: usize,
    const KEY_ROWS: usize,
    const QUERY_ROWS: usize,
>(
    keys: *const u32,
    ds: *const u32,
    dq: *mut f32,
    cfg: &FwdConfig,
    tile: &LaunchTile,
    query_start: u32,
    permuted: bool,
) {
    unsafe {
        let lane = tile.lane_id;
        let warp = tile.tid / 32;
        let ds_addr = shared_address(ds);
        let k_addr = shared_address(keys);
        let mut acc = [[0.0f32; 4]; NFQ];
        let mut chunk = 0;
        while chunk < (KEY_ROWS / 16) as u32 {
            thread::__unroll_config::<0>();
            let key_row = chunk * 16 + lane % 8 + (lane / 16) * 8;
            let query = (warp % (QUERY_ROWS as u32 / 16)) * 16 + ((lane / 8) % 2) * 8;
            let a = crate::device::inlined::ldmatrix_x4_trans_shared(
                ds_addr + (key_row * (QUERY_ROWS as u32 / 2) + (query / 2)) * 4,
            );
            let mut d = 0;
            while d < (NFQ * 2 * QUERY_ROWS / KEY_ROWS) as u32 {
                thread::__unroll_config::<0>();
                let row = chunk * 16 + lane % 16;
                let col = (warp / (QUERY_ROWS as u32 / 16))
                    * (HD as u32 * QUERY_ROWS as u32 / (2 * KEY_ROWS as u32))
                    + d * 4
                    + (lane / 16) * 4;
                let b = crate::device::inlined::ldmatrix_x4_trans_shared(
                    k_addr + (row * KSTRIDE as u32 + (col ^ ((row & 7) * 4))) * 4,
                );
                acc[d as usize] =
                    crate::device::inlined::mma::<BF16>(acc[d as usize], a, [b[0], b[1]]);
                acc[d as usize + 1] =
                    crate::device::inlined::mma::<BF16>(acc[d as usize + 1], a, [b[2], b[3]]);
                d += 2;
            }
            chunk += 1;
        }
        if permuted {
            let stride = cfg.h as usize * 64;
            let base = ((tile.b as usize * cfg.seqlen_q as usize + query_start as usize)
                * cfg.h as usize
                + tile.head as usize)
                * 64;
            let mut d = 0;
            while d < 4 {
                thread::__unroll_config::<0>();
                let x = acc[d];
                let row = d * 8 + tile.tid as usize / 64;
                let dst = dq.add(base + row * stride + (tile.tid % 64) as usize);
                ptx_asm!(
                    "{ .reg .u64 p; red.relaxed.gpu.global.add.f32 [%0], %1; add.u64 p, %0, %5; red.relaxed.gpu.global.add.f32 [p], %2; add.u64 p, p, %5; red.relaxed.gpu.global.add.f32 [p], %3; add.u64 p, p, %5; red.relaxed.gpu.global.add.f32 [p], %4; }",
                    in("l") dst, in("f") x[0], in("f") x[1], in("f") x[2], in("f") x[3], in("l") (2 * stride * 4) as u64, clobber("memory")
                );
                d += 1;
            }
            return;
        }
        // Reuse the full Q/dOut staging region after all dS reads finish.
        thread::sync_threads();
        let staged = ds as *mut f32;
        let row = (warp % (QUERY_ROWS as u32 / 16)) * 16 + lane / 4;
        let mut d = 0;
        while d < (NFQ * 2 * QUERY_ROWS / KEY_ROWS) as u32 {
            thread::__unroll_config::<0>();
            let col = (warp / (QUERY_ROWS as u32 / 16))
                * (HD as u32 * QUERY_ROWS as u32 / KEY_ROWS as u32)
                + d * 8
                + (lane % 4) * 2;
            let values = acc[d as usize];
            let swizzle = ((row & 3) * 8) | ((row >> 2) & 1);
            *staged.add((row * HD as u32 + (col ^ swizzle)) as usize) = values[0];
            *staged.add((row * HD as u32 + ((col + 1) ^ swizzle)) as usize) = values[1];
            *staged.add(((row + 8) * HD as u32 + (col ^ swizzle)) as usize) = values[2];
            *staged.add(((row + 8) * HD as u32 + ((col + 1) ^ swizzle)) as usize) = values[3];
            d += 1;
        }
        thread::sync_threads();
        let col = tile.tid % HD as u32;
        let first_row = tile.tid / HD as u32;
        if col < cfg.hs_u32 * 2 {
            let rows = (cfg.seqlen_q - query_start).min(QUERY_ROWS as u32);
            let stride = cfg.h as usize * cfg.hs_u32 as usize * 2;
            let mut base = ((tile.b as usize * cfg.seqlen_q as usize
                + (query_start + first_row) as usize)
                * cfg.h as usize
                + tile.head as usize)
                * cfg.hs_u32 as usize
                * 2
                + col as usize;
            let mut row = first_row;
            while row < rows {
                let swizzle = ((row & 3) * 8) | ((row >> 2) & 1);
                let value = *staged.add((row * HD as u32 + (col ^ swizzle)) as usize);
                fused_atomic_add(dq.add(base), value);
                base += stride * (KEY_ROWS * 2 / HD);
                row += (KEY_ROWS * 2 / HD) as u32;
            }
        }
    }
}

#[inline(always)]
pub(super) unsafe fn fused_dq_tile<const BF16: bool, const DROP: bool, const MODE: u32>(
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
        let ds_addr = shared_address(ds);
        let k_addr = shared_address(keys);
        let mut acc = [[0.0f32; 4]; 8];
        let mut chunk = 0;
        while chunk < 4 {
            thread::__unroll_config::<0>();
            let key_row = chunk * 16 + lane % 8 + (lane / 16) * 8;
            let query = (warp % 2) * 16 + ((lane / 8) % 2) * 8;
            let a = crate::device::inlined::ldmatrix_x4_trans_shared(
                ds_addr + (key_row * 16 + query / 2) * 4,
            );
            let mut d = 0;
            while d < 8 {
                thread::__unroll_config::<0>();
                let row = chunk * 16 + lane % 16;
                let col = (warp / 2) * 32 + d * 4 + (lane / 16) * 4;
                let b = crate::device::inlined::ldmatrix_x4_trans_shared(
                    k_addr + (row * 64 + (col ^ ((row & 7) * 4))) * 4,
                );
                acc[d as usize] =
                    crate::device::inlined::mma::<BF16>(acc[d as usize], a, [b[0], b[1]]);
                acc[d as usize + 1] =
                    crate::device::inlined::mma::<BF16>(acc[d as usize + 1], a, [b[2], b[3]]);
                d += 2;
            }
            chunk += 1;
        }
        // Reuse the full Q/dOut staging region after all dS reads finish.
        thread::sync_threads();
        // Short general kernels favor scalar staging. Balanced traversal is
        // selected for long sequences, so its paired path can be unconditional.
        let paired = DROP && (MODE == 3 || cfg.seqlen_q > 64);
        let staged = ds as *mut f32;
        let staged_address = shared_address(ds);
        let row = (warp % 2) * 16 + lane / 4;
        let mut d = 0;
        while d < 8 {
            thread::__unroll_config::<0>();
            let col = (warp / 2) * 64 + d * 8 + (lane % 4) * 2;
            let values = acc[d as usize];
            // Paired stores reduce staging work and spills in dropout variants.
            // The scalar layout remains faster for non-dropout kernels.
            let swizzle = if paired {
                (row & 3) * 8
            } else {
                ((row & 3) * 8) | ((row >> 2) & 1)
            };
            if paired {
                let address = staged_address + (row * 128 + (col ^ swizzle)) * 4;
                let upper_address = address + 4096;
                ptx_asm!(
                    "st.shared.v2.f32 [%0], {%1, %2}; st.shared.v2.f32 [%3], {%4, %5};",
                    in("r") address, in("f") values[0], in("f") values[1],
                    in("r") upper_address, in("f") values[2], in("f") values[3],
                    clobber("memory"),
                );
            } else {
                *staged.add((row * 128 + (col ^ swizzle)) as usize) = values[0];
                *staged.add((row * 128 + ((col + 1) ^ swizzle)) as usize) = values[1];
                *staged.add(((row + 8) * 128 + (col ^ swizzle)) as usize) = values[2];
                *staged.add(((row + 8) * 128 + ((col + 1) ^ swizzle)) as usize) = values[3];
            }
            d += 1;
        }
        thread::sync_threads();
        let col = tile.tid;
        if col < cfg.hs_u32 * 2 {
            let rows = (cfg.seqlen_q - query_start).min(32);
            let stride = cfg.h as usize * cfg.hs_u32 as usize * 2;
            let mut base = ((tile.b as usize * cfg.seqlen_q as usize + query_start as usize)
                * cfg.h as usize
                + tile.head as usize)
                * cfg.hs_u32 as usize
                * 2
                + col as usize;
            // Full causal-dropout tiles avoid per-row bounds checks and loop control.
            // Keep partial tiles bounded; other variants perform better with the loop.
            if DROP && (MODE == 1 || MODE == 3) && rows == 32 {
                let mut row = 0;
                while row < 32 {
                    thread::__unroll_config::<0>();
                    let swizzle = if paired {
                        (row & 3) * 8
                    } else {
                        ((row & 3) * 8) | ((row >> 2) & 1)
                    };
                    let value = *staged.add((row * 128 + (col ^ swizzle)) as usize);
                    fused_atomic_add(dq.add(base), value);
                    base += stride;
                    row += 1;
                }
            } else {
                let mut row = 0;
                while row < rows {
                    let swizzle = if paired {
                        (row & 3) * 8
                    } else {
                        ((row & 3) * 8) | ((row >> 2) & 1)
                    };
                    let value = *staged.add((row * 128 + (col ^ swizzle)) as usize);
                    fused_atomic_add(dq.add(base), value);
                    base += stride;
                    row += 1;
                }
            }
        }
    }
}
