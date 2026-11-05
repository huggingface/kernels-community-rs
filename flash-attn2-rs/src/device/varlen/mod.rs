mod backward;
mod merge;
mod paged;

use crate::device::config::{FwdConfig, GRID_PLAIN, KvRange, LaunchTile, TensorBases};
use crate::device::dropout::Dropout;
use crate::device::forward::*;
use crate::device::inlined::{cvt, lg2_approx};
use crate::device::varlen::paged::Pages;
use cuda_device::warp::shuffle_xor_f32;
use cuda_device::{DisjointSlice, kernel, launch_bounds, ptx_asm, thread};

#[derive(Clone, Copy)]
pub struct Cache {
    pub indices: *const i32,
    pub batches: u32,
    pub new_keys: u32,
    pub splits: u32,
    pub append_k: *const u32,
    pub append_v: *const u32,
    pub append_ks: crate::device::config::Strides,
    pub append_vs: crate::device::config::Strides,
}
impl Cache {
    pub const NONE: Self = Self {
        indices: core::ptr::null(),
        batches: 0,
        new_keys: 0,
        splits: 1,
        append_k: core::ptr::null(),
        append_v: core::ptr::null(),
        append_ks: crate::device::config::Strides {
            batch: 0,
            seq: 0,
            head: 0,
        },
        append_vs: crate::device::config::Strides {
            batch: 0,
            seq: 0,
            head: 0,
        },
    };
}

#[inline(always)]
unsafe fn core<
    const BF16: bool,
    const HD: usize,
    const NFQ: usize,
    const NFO: usize,
    const KEYS: u32,
    const NKT: usize,
    const PV_CHUNKS: usize,
    const DROPOUT: bool,
    const PAGED: bool,
    const CACHED: bool,
    const SPLIT: bool,
    const GROUPED: bool,
    const VECTOR_OUTPUT: bool,
    const WIDE: bool,
>(
    q: *const u32,
    k: *const u32,
    v: *const u32,
    o: *mut u32,
    lse: *mut f32,
    alibi: *const f32,
    cuq: *const i32,
    cuk: *const i32,
    used: *const i32,
    leftpad: *const i32,
    total_q: u32,
    total_k: u32,
    mut cfg: FwdConfig,
    mut dropout: Dropout,
    pages: Pages,
    cache: Cache,
) {
    unsafe {
        let groups = if GROUPED { cfg.gqa } else { 1 };
        let heads = if GROUPED { cfg.h / groups } else { cfg.h };
        let query_rows = if GROUPED { groups } else { cfg.seqlen_q };
        // Prefill causal grids put the longest query blocks of every head first.
        let causal_grid = !CACHED
            && !SPLIT
            && cfg.window_right == 0
            && (cfg.window_left < 0 || cfg.window_left >= cfg.seqlen_k as i32);
        let mut tile = if CACHED || SPLIT {
            LaunchTile::new::<64, 1, GRID_PLAIN>(
                heads,
                if GROUPED { 1 } else { cfg.gqa },
                query_rows,
                false,
            )
        } else {
            LaunchTile::new::<64, 1, 0>(heads, cfg.gqa, query_rows, causal_grid)
        };
        let split = if SPLIT {
            // Fold splits into grid.x so batch remains within CUDA's grid.z limit.
            tile.q_row_base =
                (query_rows.div_ceil(64) - 1 - thread::blockIdx_x() / cache.splits) * 64;
            thread::blockIdx_x() % cache.splits
        } else {
            0
        };
        let qs = if CACHED {
            (tile.b * cfg.seqlen_q) as i32
        } else {
            *cuq.add(tile.b as usize)
        };
        let qe = if CACHED {
            qs + cfg.seqlen_q as i32
        } else {
            *cuq.add(tile.b as usize + 1)
        };
        let ks = if CACHED { 0 } else { *cuk.add(tile.b as usize) };
        let ke = if CACHED {
            cfg.seqlen_k as i32
        } else {
            *cuk.add(tile.b as usize + 1)
        };
        let pad = if leftpad.is_null() {
            0
        } else {
            *leftpad.add(tile.b as usize)
        };
        let length = if CACHED {
            (if cuk.is_null() {
                cfg.seqlen_k as i32
            } else {
                *cuk.add(tile.b as usize)
            })
            .wrapping_add(cache.new_keys as i32)
        } else if used.is_null() {
            ke - ks
        } else {
            *used.add(tile.b as usize)
        };
        // As in v2, metadata is trusted; only memory bounds are guarded.
        if qs < 0
            || qe < qs
            || qe as u32 > total_q
            || (qe - qs) as u32 > cfg.seqlen_q
            || ks < 0
            || ke < ks
            || (!PAGED && ke as u32 > total_k)
            || (PAGED && length as u32 > pages.columns.saturating_mul(pages.size))
            || pad < 0
            || length < pad
            || length > ke - ks
            || (length - pad) as u32 > cfg.seqlen_k
        {
            return;
        }
        cfg.seqlen_q = (qe - qs) as u32 * groups;
        cfg.seqlen_k = (length - pad) as u32;
        if GROUPED {
            cfg.h = heads;
            cfg.gqa = 1;
        }
        if tile.q_row_base >= cfg.seqlen_q {
            return;
        }
        dropout.seqlen_q = cfg.seqlen_q;
        dropout.seqlen_k = cfg.seqlen_k;
        let cache_batch = if CACHED && !PAGED && !cache.indices.is_null() {
            *cache.indices.add(tile.b as usize)
        } else {
            tile.b as i32
        };
        if CACHED && !PAGED && (cache_batch < 0 || cache_batch as u32 >= cache.batches) {
            return;
        }
        let direct_grouped = CACHED && GROUPED && !SPLIT;
        let o_ss = if direct_grouped {
            cfg.hs_u32 as usize
        } else {
            cfg.h as usize * cfg.hs_u32 as usize
        };
        let bases = TensorBases {
            q: (if CACHED {
                tile.b as usize * cfg.q.batch as usize
            } else {
                qs as usize * cfg.q.seq as usize
            }) + tile.head as usize * groups as usize * cfg.q.head as usize,
            k: (if CACHED && !PAGED {
                cache_batch as usize * cfg.k.batch as usize
            } else {
                0
            }) + (if PAGED { 0 } else { ks + pad }) as usize * cfg.k.seq as usize
                + tile.kv_head as usize * cfg.k.head as usize,
            v: (if CACHED && !PAGED {
                cache_batch as usize * cfg.v.batch as usize
            } else {
                0
            }) + (if PAGED { 0 } else { ks + pad }) as usize * cfg.v.seq as usize
                + tile.kv_head as usize * cfg.v.head as usize,
            o: if direct_grouped {
                (tile.b as usize * heads as usize + tile.head as usize)
                    * groups as usize
                    * cfg.hs_u32 as usize
            } else {
                qs as usize * o_ss + tile.head as usize * cfg.hs_u32 as usize
            },
            o_ss,
        };
        // The fused launch owns one cache head per block; aliased sources were snapshotted.
        if WIDE && !cache.append_k.is_null() {
            if cfg.seqlen_k < cache.new_keys {
                return;
            }
            let start = cfg.seqlen_k - cache.new_keys;
            let cols = (HD / 2) as u32;
            let mut i = tile.tid;
            while i < cache.new_keys * cols {
                let row = i / cols;
                let col = i % cols;
                let ki = tile.b as usize * cache.append_ks.batch as usize
                    + row as usize * cache.append_ks.seq as usize
                    + tile.kv_head as usize * cache.append_ks.head as usize
                    + col as usize;
                let vi = tile.b as usize * cache.append_vs.batch as usize
                    + row as usize * cache.append_vs.seq as usize
                    + tile.kv_head as usize * cache.append_vs.head as usize
                    + col as usize;
                *(k as *mut u32)
                    .add(bases.k + (start + row) as usize * cfg.k.seq as usize + col as usize) =
                    *cache.append_k.add(ki);
                *(v as *mut u32)
                    .add(bases.v + (start + row) as usize * cfg.v.seq as usize + col as usize) =
                    *cache.append_v.add(vi);
                i += 128;
            }
            thread::sync_threads();
        }
        if GROUPED {
            // Query rows are adjacent GQA heads at the same position.
            cfg.q.seq = cfg.q.head;
        }
        // Constant copy indices measured faster for these packed tiles; d96 and
        // d128 lost ~5-10% and the wide d32 decode tile ran out of registers.
        let unroll_copy = (HD == 32 && !WIDE) || HD == 192;
        let mut layout = SmemLayout::new::<HD, KEYS, 64>();
        // Unsplit decode streams the next K tile behind the current one.
        if WIDE
            && !layout.double
            && !layout.triple
            && 3 * KEYS as usize * layout.stride * 4 <= SMEM_BUDGET
        {
            layout.triple = true;
        }
        let smem = SmemBuffers::new(layout);
        let off = cfg.seqlen_k as i32 - cfg.seqlen_q as i32;
        let slope = if cfg.has_alibi != 0 {
            *alibi.add(tile.b as usize * cfg.alibi_bs as usize + tile.head as usize)
        } else {
            0.0
        };

        let scale = cfg.scale * LOG2_E;
        let raw_scores = WIDE;
        let tanh_scale = cfg.tanh_scale;
        let slope = slope * LOG2_E;

        let q_frag = load_q_frag::<NFQ, 1, NFQ, 64, 128>(q, smem.k0, &cfg, &tile, &bases, layout);

        let mut o_acc = [[0.0f32; 4]; NFO];
        let mut m_row = [f32::NEG_INFINITY; 2];
        let mut l_row = [0.0f32; 2];

        let mut kv_range = KvRange::new::<64, KEYS>(&cfg, tile.q_row_base, off);
        if SPLIT {
            // Preserve logical positions for masks/ALiBi and page-table lookup.
            let tiles_per_split = cfg.seqlen_k.div_ceil(KEYS).div_ceil(cache.splits);
            kv_range.start_tile = kv_range.start_tile.max(split * tiles_per_split);
            kv_range.num_tiles = kv_range.num_tiles.min((split + 1) * tiles_per_split);
        }
        let reverse = false;
        // Page-table entries for tiles j and j+1; j+2 is fetched a tile early.
        let mut page = 0i32;
        let mut next_page = 0i32;
        if PAGED {
            page = paged::lookup_page(pages, tile.b, kv_range.start_tile * KEYS);
            next_page = paged::lookup_page(pages, tile.b, (kv_range.start_tile + 1) * KEYS);
            paged::prefetch_first_kv::<HD, KEYS, 128, false, false>(
                k, v, &cfg, &tile, &bases, kv_range, smem, layout, reverse, pages, page,
            );
        } else {
            if unroll_copy {
                prefetch_first_kv::<HD, KEYS, 128, true, false>(
                    k, v, &cfg, &tile, &bases, kv_range, smem, layout, reverse,
                );
            } else {
                prefetch_first_kv::<HD, KEYS, 128, false, false>(
                    k, v, &cfg, &tile, &bases, kv_range, smem, layout, reverse,
                );
            }
        }

        let mut j = kv_range.start_tile;
        while j < kv_range.num_tiles {
            let kv_base = j * KEYS;
            let after_next_page = if PAGED {
                paged::lookup_page(pages, tile.b, (j + 2) * KEYS)
            } else {
                0
            };
            let (k_smem_base, v_smem_base) = if PAGED {
                paged::stage_kv_tile::<HD, KEYS, 128, false, false>(
                    k, v, &cfg, &tile, &bases, kv_range, smem, layout, j, reverse, pages, page,
                    next_page,
                )
            } else {
                if unroll_copy {
                    stage_kv_tile::<HD, KEYS, 128, true, false>(
                        k, v, &cfg, &tile, &bases, kv_range, smem, layout, j, reverse,
                    )
                } else {
                    stage_kv_tile::<HD, KEYS, 128, false, false>(
                        k, v, &cfg, &tile, &bases, kv_range, smem, layout, j, reverse,
                    )
                }
            };
            let mut s = [[0.0f32; 4]; NKT];

            compute_qk::<BF16, 1, NFQ, NKT>(&mut s, &q_frag, k_smem_base, &tile, layout);

            if !layout.double && !layout.triple {
                // One barrier can finish V staging and release the consumed K tile.
                if HD == 64 || raw_scores {
                    ptx_asm!("cp.async.wait_group 0;", clobber("memory"));
                }
                thread::sync_threads();
                if j + 1 < kv_range.num_tiles {
                    if PAGED {
                        paged::stage_tile::<HD, KEYS, 128>(
                            smem.k0,
                            k,
                            (j + 1) * KEYS,
                            next_page,
                            bases.k,
                            cfg.k.seq,
                            cfg.k.batch,
                            &cfg,
                            &tile,
                            layout,
                            pages,
                        );
                    } else {
                        if unroll_copy {
                            stage_tile::<HD, KEYS, 128, true, false>(
                                smem.k0,
                                k,
                                (j + 1) * KEYS,
                                bases.k,
                                cfg.k.seq,
                                &cfg,
                                &tile,
                                layout,
                            );
                        } else {
                            stage_tile::<HD, KEYS, 128, false, false>(
                                smem.k0,
                                k,
                                (j + 1) * KEYS,
                                bases.k,
                                cfg.k.seq,
                                &cfg,
                                &tile,
                                layout,
                            );
                        }
                    }
                    ptx_asm!("cp.async.commit_group;", clobber("memory"));
                }
            }
            let mut p_pack = [[0u32; 2]; NKT];

            apply_scores::<1, NKT, 64, KEYS>(
                &mut s,
                &cfg,
                &tile,
                kv_base,
                off,
                if raw_scores { 1.0 } else { scale },
                tanh_scale,
                slope,
            );
            if raw_scores {
                online_softmax_pack_raw::<BF16, 1, NKT, NFO>(
                    &s,
                    &mut p_pack,
                    &mut o_acc,
                    &mut m_row,
                    &mut l_row,
                    scale,
                );
            } else {
                online_softmax_pack::<BF16, 1, NKT, NFO>(
                    &s,
                    &mut p_pack,
                    &mut o_acc,
                    &mut m_row,
                    &mut l_row,
                );
            }
            if DROPOUT {
                dropout.apply::<1, NKT>(&mut p_pack, &tile, kv_base);
            }

            if (!layout.double && HD != 64 && !raw_scores) || (layout.triple && raw_scores) {
                // V is the older committed group; next K may remain in flight.
                if j + 1 < kv_range.num_tiles {
                    ptx_asm!("cp.async.wait_group 1;", clobber("memory"));
                } else {
                    ptx_asm!("cp.async.wait_group 0;", clobber("memory"));
                }
                thread::sync_threads();
            }

            accumulate_values::<BF16, 1, NKT, NFO, PV_CHUNKS>(
                &mut o_acc,
                &p_pack,
                v_smem_base,
                &tile,
                layout,
            );

            // Double buffering may enqueue the V overwrite early.
            if layout.double {
                thread::sync_threads();
            }
            page = next_page;
            next_page = after_next_page;
            j += 1;
        }

        if raw_scores {
            m_row[0] *= scale;
            m_row[1] *= scale;
        }
        let lse_base = if SPLIT {
            ((tile.b as usize * cache.splits as usize + split as usize) * cfg.h as usize
                + tile.head as usize)
                * query_rows as usize
        } else if CACHED {
            (tile.b as usize * cfg.h as usize + tile.head as usize) * cfg.seqlen_q as usize
        } else {
            tile.head as usize * total_q as usize + qs as usize
        };
        let output_scale = if DROPOUT { dropout.inv_keep } else { 1.0 };
        if VECTOR_OUTPUT {
            write_output::<BF16, 1, NFO, false, 64, 128, true>(
                o,
                lse,
                &o_acc,
                &m_row,
                &l_row,
                &cfg,
                &tile,
                &bases,
                smem.k0,
                layout,
                output_scale,
                lse_base,
            );
        } else {
            write_packed_output::<BF16, NFO, SPLIT>(
                o,
                lse,
                &o_acc,
                &m_row,
                &l_row,
                &cfg,
                &tile,
                &bases,
                lse_base,
                output_scale,
            );
        }
    }
}

pub trait Spec {
    unsafe fn run(
        q: *const u32,
        k: *const u32,
        v: *const u32,
        o: *mut u32,
        lse: *mut f32,
        alibi: *const f32,
        cuq: *const i32,
        cuk: *const i32,
        used: *const i32,
        leftpad: *const i32,
        total_q: u32,
        total_k: u32,
        cfg: FwdConfig,
        dropout: Dropout,
        pages: Pages,
        cache: Cache,
    );
}
pub struct Varlen<
    const BF16: bool,
    const HD: usize,
    const NFQ: usize,
    const NFO: usize,
    const KEYS: u32,
    const NKT: usize,
    const PV_CHUNKS: usize,
    const DROPOUT: bool,
    const PAGED: bool,
    const CACHED: bool,
    const SPLIT: bool = false,
    const GROUPED: bool = false,
    const PAGE_SIZE: u32 = 0,
    const CAUSAL256: bool = false,
    const WIDE: bool = false,
>;
impl<
    const BF16: bool,
    const HD: usize,
    const NFQ: usize,
    const NFO: usize,
    const KEYS: u32,
    const NKT: usize,
    const PV_CHUNKS: usize,
    const DROPOUT: bool,
    const PAGED: bool,
    const CACHED: bool,
    const SPLIT: bool,
    const GROUPED: bool,
    const PAGE_SIZE: u32,
    const CAUSAL256: bool,
    const WIDE: bool,
> Spec
    for Varlen<
        BF16,
        HD,
        NFQ,
        NFO,
        KEYS,
        NKT,
        PV_CHUNKS,
        DROPOUT,
        PAGED,
        CACHED,
        SPLIT,
        GROUPED,
        PAGE_SIZE,
        CAUSAL256,
        WIDE,
    >
{
    #[inline(always)]
    unsafe fn run(
        q: *const u32,
        k: *const u32,
        v: *const u32,
        o: *mut u32,
        lse: *mut f32,
        alibi: *const f32,
        cuq: *const i32,
        cuk: *const i32,
        used: *const i32,
        leftpad: *const i32,
        total_q: u32,
        total_k: u32,
        mut cfg: FwdConfig,
        dropout: Dropout,
        mut pages: Pages,
        cache: Cache,
    ) {
        if PAGE_SIZE != 0 {
            pages.size = PAGE_SIZE;
        }
        // Both folded variants serve exact head dims without masks or ALiBi.
        if WIDE {
            cfg.hs_u32 = (HD / 2) as u32;
            cfg.tanh_scale = 0.0;
            cfg.has_alibi = 0;
            cfg.alibi_causal = 0;
            cfg.window_left = -1;
            cfg.window_right = -1;
        }
        if CAUSAL256 {
            cfg.hs_u32 = (HD / 2) as u32;
            cfg.tanh_scale = 0.0;
            cfg.has_alibi = 0;
            cfg.alibi_causal = 0;
            cfg.window_left = -1;
            cfg.window_right = 0;
        }
        unsafe {
            core::<
                BF16,
                HD,
                NFQ,
                NFO,
                KEYS,
                NKT,
                PV_CHUNKS,
                DROPOUT,
                PAGED,
                CACHED,
                SPLIT,
                GROUPED,
                CAUSAL256,
                WIDE,
            >(
                q, k, v, o, lse, alibi, cuq, cuk, used, leftpad, total_q, total_k, cfg, dropout,
                pages, cache,
            );
        }
    }
}
#[allow(non_camel_case_types)]
pub type d32 = Varlen<false, 32, 2, 4, 64, 8, 4, false, false, false>;
#[allow(non_camel_case_types)]
pub type d32_dropout = Varlen<false, 32, 2, 4, 64, 8, 4, true, false, false>;
#[allow(non_camel_case_types)]
pub type d32_bf16 = Varlen<true, 32, 2, 4, 64, 8, 4, false, false, false>;
#[allow(non_camel_case_types)]
pub type d32_bf16_dropout = Varlen<true, 32, 2, 4, 64, 8, 4, true, false, false>;
#[allow(non_camel_case_types)]
pub type d64 = Varlen<false, 64, 4, 8, 64, 8, 4, false, false, false>;
#[allow(non_camel_case_types)]
pub type d64_dropout = Varlen<false, 64, 4, 8, 64, 8, 4, true, false, false>;
#[allow(non_camel_case_types)]
pub type d64_bf16 = Varlen<true, 64, 4, 8, 64, 8, 4, false, false, false>;
#[allow(non_camel_case_types)]
pub type d64_bf16_dropout = Varlen<true, 64, 4, 8, 64, 8, 4, true, false, false>;
#[allow(non_camel_case_types)]
pub type d96 = Varlen<false, 96, 6, 12, 32, 4, 2, false, false, false>;
#[allow(non_camel_case_types)]
pub type d96_dropout = Varlen<false, 96, 6, 12, 32, 4, 2, true, false, false>;
#[allow(non_camel_case_types)]
pub type d96_bf16 = Varlen<true, 96, 6, 12, 32, 4, 2, false, false, false>;
#[allow(non_camel_case_types)]
pub type d96_bf16_dropout = Varlen<true, 96, 6, 12, 32, 4, 2, true, false, false>;
#[allow(non_camel_case_types)]
pub type d128 = Varlen<false, 128, 8, 16, 32, 4, 2, false, false, false>;
#[allow(non_camel_case_types)]
pub type d128_dropout = Varlen<false, 128, 8, 16, 32, 4, 2, true, false, false>;
#[allow(non_camel_case_types)]
pub type d128_bf16 = Varlen<true, 128, 8, 16, 32, 4, 2, false, false, false>;
#[allow(non_camel_case_types)]
pub type d128_bf16_dropout = Varlen<true, 128, 8, 16, 32, 4, 2, true, false, false>;
#[allow(non_camel_case_types)]
pub type d192 = Varlen<false, 192, 12, 24, 32, 4, 2, false, false, false>;
#[allow(non_camel_case_types)]
pub type d192_dropout = Varlen<false, 192, 12, 24, 32, 4, 2, true, false, false>;
#[allow(non_camel_case_types)]
pub type d192_bf16 = Varlen<true, 192, 12, 24, 32, 4, 2, false, false, false>;
#[allow(non_camel_case_types)]
pub type d192_bf16_dropout = Varlen<true, 192, 12, 24, 32, 4, 2, true, false, false>;
#[allow(non_camel_case_types)]
pub type d256 = Varlen<false, 256, 16, 32, 32, 4, 2, false, false, false>;
#[allow(non_camel_case_types)]
pub type d256_dropout = Varlen<false, 256, 16, 32, 32, 4, 2, true, false, false>;
#[allow(non_camel_case_types)]
pub type d256_bf16 = Varlen<true, 256, 16, 32, 32, 4, 2, false, false, false>;
#[allow(non_camel_case_types)]
pub type d256_bf16_dropout = Varlen<true, 256, 16, 32, 32, 4, 2, true, false, false>;
#[kernel(
    d32,
    d32_dropout,
    d32_bf16,
    d32_bf16_dropout,
    d64,
    d64_dropout,
    d64_bf16,
    d64_bf16_dropout,
    d96,
    d96_dropout,
    d96_bf16,
    d96_bf16_dropout,
    d128,
    d128_dropout,
    d128_bf16,
    d128_bf16_dropout,
    d192,
    d192_dropout,
    d192_bf16,
    d192_bf16_dropout,
    d256,
    d256_dropout,
    d256_bf16,
    d256_bf16_dropout
)]
#[launch_bounds(128)]
pub fn varlen_fwd<S: Spec>(
    q: &[u32],
    k: &[u32],
    v: &[u32],
    out: DisjointSlice<u32>,
    lse: DisjointSlice<f32>,
    alibi: &[f32],
    cuq: &[i32],
    cuk: &[i32],
    used: &[i32],
    leftpad: &[i32],
    rng: &[u64],
    probabilities: DisjointSlice<u32>,
    scale: f32,
    tanh_scale: f32,
    window_left: i32,
    window_right: i32,
    max_q: u32,
    max_k: u32,
    has_alibi: u32,
    alibi_causal: u32,
    h: u32,
    gqa: u32,
    hs_u32: u32,
    q_ss: u32,
    q_hs: u32,
    k_ss: u32,
    k_hs: u32,
    v_ss: u32,
    v_hs: u32,
    alibi_bs: u32,
    total_q: u32,
    total_k: u32,
    keep: f32,
) {
    let mut out = out;
    let mut lse = lse;
    let mut probabilities = probabilities;
    let cfg = FwdConfig::new(
        scale,
        tanh_scale,
        window_left,
        window_right,
        max_q,
        max_k,
        has_alibi,
        alibi_causal,
        h,
        gqa,
        hs_u32,
        0,
        q_ss,
        q_hs,
        0,
        k_ss,
        k_hs,
        0,
        v_ss,
        v_hs,
        alibi_bs,
    );
    let seed = if keep < 1.0 {
        unsafe { *rng.as_ptr() }
    } else {
        0
    };
    let offset = if keep < 1.0 {
        unsafe { *rng.as_ptr().add(1) }
    } else {
        0
    };
    let dropout = Dropout {
        seed,
        offset,
        threshold: (keep * 255.0) as u32,
        inv_keep: 1.0 / keep,
        probabilities: if probabilities.len() == 0 {
            core::ptr::null_mut()
        } else {
            probabilities.as_mut_ptr()
        },
        rounded_q: max_q.div_ceil(128) as u64 * 128,
        rounded_k: max_k.div_ceil(128) as u64 * 128,
        seqlen_q: max_q,
        seqlen_k: max_k,
    };
    unsafe {
        S::run(
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            out.as_mut_ptr(),
            lse.as_mut_ptr(),
            alibi.as_ptr(),
            cuq.as_ptr(),
            crate::device::inlined::optional_ptr(cuk),
            crate::device::inlined::optional_ptr(used),
            crate::device::inlined::optional_ptr(leftpad),
            total_q,
            total_k,
            cfg,
            dropout,
            Pages::NONE,
            Cache::NONE,
        );
    }
}

// Packed dropout resolves replay-time Philox state without a separate launch.
#[kernel(d64_dropout, d64_bf16_dropout)]
#[launch_bounds(128)]
pub fn varlen_rng_descriptor<S: Spec>(
    q: &[u32],
    k: &[u32],
    v: &[u32],
    out: DisjointSlice<u32>,
    lse: DisjointSlice<f32>,
    alibi: &[f32],
    cuq: &[i32],
    cuk: &[i32],
    used: &[i32],
    leftpad: &[i32],
    rng: DisjointSlice<u64>,
    probabilities: DisjointSlice<u32>,
    scale: f32,
    tanh_scale: f32,
    window_left: i32,
    window_right: i32,
    max_q: u32,
    max_k: u32,
    has_alibi: u32,
    alibi_causal: u32,
    h: u32,
    gqa: u32,
    hs_u32: u32,
    q_ss: u32,
    q_hs: u32,
    k_ss: u32,
    k_hs: u32,
    v_ss: u32,
    v_hs: u32,
    alibi_bs: u32,
    total_q: u32,
    total_k: u32,
    keep: f32,
    descriptor_seed: u64,
    descriptor_offset: u64,
    intragraph: u64,
    captured: u32,
) {
    let mut out = out;
    let mut lse = lse;
    let mut probabilities = probabilities;
    let cfg = FwdConfig::new(
        scale,
        tanh_scale,
        window_left,
        window_right,
        max_q,
        max_k,
        has_alibi,
        alibi_causal,
        h,
        gqa,
        hs_u32,
        0,
        q_ss,
        q_hs,
        0,
        k_ss,
        k_hs,
        0,
        v_ss,
        v_hs,
        alibi_bs,
    );
    let mut rng = rng;
    let (seed, offset) = unsafe {
        if captured != 0 {
            (
                *(descriptor_seed as *const u64),
                (*(descriptor_offset as *const u64)).wrapping_add(intragraph),
            )
        } else {
            (descriptor_seed, descriptor_offset)
        }
    };
    // Save before early returns; no block depends on this write.
    if thread::blockIdx_x() == 0
        && thread::blockIdx_y() == 0
        && thread::blockIdx_z() == 0
        && thread::threadIdx_x() == 0
    {
        unsafe {
            *rng.as_mut_ptr() = seed;
            *rng.as_mut_ptr().add(1) = offset;
        }
    }
    let dropout = Dropout {
        seed,
        offset,
        threshold: (keep * 255.0) as u32,
        inv_keep: 1.0 / keep,
        probabilities: if probabilities.len() == 0 {
            core::ptr::null_mut()
        } else {
            probabilities.as_mut_ptr()
        },
        rounded_q: max_q.div_ceil(128) as u64 * 128,
        rounded_k: max_k.div_ceil(128) as u64 * 128,
        seqlen_q: max_q,
        seqlen_k: max_k,
    };
    unsafe {
        S::run(
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            out.as_mut_ptr(),
            lse.as_mut_ptr(),
            alibi.as_ptr(),
            cuq.as_ptr(),
            crate::device::inlined::optional_ptr(cuk),
            crate::device::inlined::optional_ptr(used),
            crate::device::inlined::optional_ptr(leftpad),
            total_q,
            total_k,
            cfg,
            dropout,
            Pages::NONE,
            Cache::NONE,
        );
    }
}

// Separate path so dense output register allocation stays unchanged.
#[inline(always)]
unsafe fn write_packed_output<const BF16: bool, const NFO: usize, const PARTIAL: bool>(
    out: *mut u32,
    lse: *mut f32,
    acc: &[[f32; 4]],
    maxima: &[f32],
    sums: &[f32],
    cfg: &FwdConfig,
    tile: &LaunchTile,
    bases: &TensorBases,
    lse_base: usize,
    output_scale: f32,
) {
    unsafe {
        let mut l0 = sums[0];
        let mut l1 = sums[1];
        l0 += shuffle_xor_f32(l0, 1);
        l0 += shuffle_xor_f32(l0, 2);
        l1 += shuffle_xor_f32(l1, 1);
        l1 += shuffle_xor_f32(l1, 2);
        let inv0 = if l0 > 0.0 {
            1.0 / l0 * output_scale
        } else {
            0.0
        };
        let inv1 = if l1 > 0.0 {
            1.0 / l1 * output_scale
        } else {
            0.0
        };
        let row = (tile.q_row_base + tile.warp_row + tile.group) as usize;
        let mut dim = 0;
        while dim < NFO {
            thread::__unroll_config::<0>();
            let col = dim * 4 + tile.pair as usize;
            let f = acc[dim];
            if col < cfg.hs_u32 as usize {
                if row < cfg.seqlen_q as usize {
                    if PARTIAL {
                        let target = out
                            .cast::<f32>()
                            .add((lse_base + row) * cfg.hs_u32 as usize * 2 + col * 2);
                        *target = f[0] * inv0;
                        *target.add(1) = f[1] * inv0;
                    } else {
                        *out.add(bases.o + row * bases.o_ss + col) =
                            cvt::<BF16>(f[0] * inv0, f[1] * inv0);
                    }
                }
                if row + 8 < cfg.seqlen_q as usize {
                    if PARTIAL {
                        let target = out
                            .cast::<f32>()
                            .add((lse_base + row + 8) * cfg.hs_u32 as usize * 2 + col * 2);
                        *target = f[2] * inv1;
                        *target.add(1) = f[3] * inv1;
                    } else {
                        *out.add(bases.o + (row + 8) * bases.o_ss + col) =
                            cvt::<BF16>(f[2] * inv1, f[3] * inv1);
                    }
                }
            }
            dim += 1;
        }
        if tile.pair == 0 {
            if row < cfg.seqlen_q as usize {
                *lse.add(lse_base + row) = if l0 > 0.0 {
                    (maxima[0] + lg2_approx(l0)) * core::f32::consts::LN_2
                } else {
                    if PARTIAL {
                        f32::NEG_INFINITY
                    } else {
                        f32::INFINITY
                    }
                };
            }
            if row + 8 < cfg.seqlen_q as usize {
                *lse.add(lse_base + row + 8) = if l1 > 0.0 {
                    (maxima[1] + lg2_approx(l1)) * core::f32::consts::LN_2
                } else {
                    if PARTIAL {
                        f32::NEG_INFINITY
                    } else {
                        f32::INFINITY
                    }
                };
            }
        }
    }
}

// Causal prefill over 256-row pages: folded config and v2's split-KV key tiles.
#[allow(non_camel_case_types)]
pub type paged_d32_causal256 =
    Varlen<false, 32, 2, 4, 128, 16, 8, false, true, false, false, false, 256, true>;
#[allow(non_camel_case_types)]
pub type paged_d32_causal256_bf16 =
    Varlen<true, 32, 2, 4, 128, 16, 8, false, true, false, false, false, 256, true>;
#[allow(non_camel_case_types)]
pub type paged_d32 = Varlen<false, 32, 2, 4, 64, 8, 4, false, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d32_dropout = Varlen<false, 32, 2, 4, 64, 8, 4, true, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d32_bf16 = Varlen<true, 32, 2, 4, 64, 8, 4, false, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d32_bf16_dropout = Varlen<true, 32, 2, 4, 64, 8, 4, true, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d64_causal256 =
    Varlen<false, 64, 4, 8, 64, 8, 4, false, true, false, false, false, 256, true>;
#[allow(non_camel_case_types)]
pub type paged_d64_causal256_bf16 =
    Varlen<true, 64, 4, 8, 64, 8, 4, false, true, false, false, false, 256, true>;
#[allow(non_camel_case_types)]
pub type paged_d64 = Varlen<false, 64, 4, 8, 64, 8, 4, false, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d64_dropout = Varlen<false, 64, 4, 8, 64, 8, 4, true, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d64_bf16 = Varlen<true, 64, 4, 8, 64, 8, 4, false, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d64_bf16_dropout = Varlen<true, 64, 4, 8, 64, 8, 4, true, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d96_causal256 =
    Varlen<false, 96, 6, 12, 128, 16, 8, false, true, false, false, false, 256, true>;
#[allow(non_camel_case_types)]
pub type paged_d96_causal256_bf16 =
    Varlen<true, 96, 6, 12, 128, 16, 8, false, true, false, false, false, 256, true>;
#[allow(non_camel_case_types)]
pub type paged_d96 = Varlen<false, 96, 6, 12, 32, 4, 2, false, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d96_dropout = Varlen<false, 96, 6, 12, 32, 4, 2, true, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d96_bf16 = Varlen<true, 96, 6, 12, 32, 4, 2, false, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d96_bf16_dropout = Varlen<true, 96, 6, 12, 32, 4, 2, true, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d128_causal256 =
    Varlen<false, 128, 8, 16, 128, 16, 8, false, true, false, false, false, 256, true>;
#[allow(non_camel_case_types)]
pub type paged_d128_causal256_bf16 =
    Varlen<true, 128, 8, 16, 128, 16, 8, false, true, false, false, false, 256, true>;
#[allow(non_camel_case_types)]
pub type paged_d128 = Varlen<false, 128, 8, 16, 32, 4, 2, false, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d128_dropout = Varlen<false, 128, 8, 16, 32, 4, 2, true, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d128_bf16 = Varlen<true, 128, 8, 16, 32, 4, 2, false, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d128_bf16_dropout = Varlen<true, 128, 8, 16, 32, 4, 2, true, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d192_causal256 =
    Varlen<false, 192, 12, 24, 64, 8, 4, false, true, false, false, false, 256, true>;
#[allow(non_camel_case_types)]
pub type paged_d192_causal256_bf16 =
    Varlen<true, 192, 12, 24, 64, 8, 4, false, true, false, false, false, 256, true>;
#[allow(non_camel_case_types)]
pub type paged_d192 = Varlen<false, 192, 12, 24, 32, 4, 2, false, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d192_dropout = Varlen<false, 192, 12, 24, 32, 4, 2, true, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d192_bf16 = Varlen<true, 192, 12, 24, 32, 4, 2, false, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d192_bf16_dropout = Varlen<true, 192, 12, 24, 32, 4, 2, true, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d256_causal256 =
    Varlen<false, 256, 16, 32, 64, 8, 4, false, true, false, false, false, 256, true>;
#[allow(non_camel_case_types)]
pub type paged_d256_causal256_bf16 =
    Varlen<true, 256, 16, 32, 64, 8, 4, false, true, false, false, false, 256, true>;
#[allow(non_camel_case_types)]
pub type paged_d256 = Varlen<false, 256, 16, 32, 32, 4, 2, false, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d256_dropout = Varlen<false, 256, 16, 32, 32, 4, 2, true, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d256_bf16 = Varlen<true, 256, 16, 32, 32, 4, 2, false, true, false>;
#[allow(non_camel_case_types)]
pub type paged_d256_bf16_dropout = Varlen<true, 256, 16, 32, 32, 4, 2, true, true, false>;
#[kernel(
    paged_d32_causal256,
    paged_d32_causal256_bf16,
    paged_d32,
    paged_d32_dropout,
    paged_d32_bf16,
    paged_d32_bf16_dropout,
    paged_d64_causal256,
    paged_d64_causal256_bf16,
    paged_d64,
    paged_d64_dropout,
    paged_d64_bf16,
    paged_d64_bf16_dropout,
    paged_d96_causal256,
    paged_d96_causal256_bf16,
    paged_d96,
    paged_d96_dropout,
    paged_d96_bf16,
    paged_d96_bf16_dropout,
    paged_d128_causal256,
    paged_d128_causal256_bf16,
    paged_d128,
    paged_d128_dropout,
    paged_d128_bf16,
    paged_d128_bf16_dropout,
    paged_d192_causal256,
    paged_d192_causal256_bf16,
    paged_d192,
    paged_d192_dropout,
    paged_d192_bf16,
    paged_d192_bf16_dropout,
    paged_d256_causal256,
    paged_d256_causal256_bf16,
    paged_d256,
    paged_d256_dropout,
    paged_d256_bf16,
    paged_d256_bf16_dropout
)]
#[launch_bounds(128)]
pub fn paged_fwd<S: Spec>(
    q: &[u32],
    k: &[u32],
    v: &[u32],
    out: DisjointSlice<u32>,
    lse: DisjointSlice<f32>,
    alibi: &[f32],
    cuq: &[i32],
    cuk: &[i32],
    used: &[i32],
    leftpad: &[i32],
    rng: &[u64],
    probabilities: DisjointSlice<u32>,
    scale: f32,
    tanh_scale: f32,
    window_left: i32,
    window_right: i32,
    max_q: u32,
    max_k: u32,
    has_alibi: u32,
    alibi_causal: u32,
    h: u32,
    gqa: u32,
    hs_u32: u32,
    q_ss: u32,
    q_hs: u32,
    k_ss: u32,
    k_hs: u32,
    v_ss: u32,
    v_hs: u32,
    alibi_bs: u32,
    total_q: u32,
    total_k: u32,
    keep: f32,
    table: &[i32],
    table_stride: u32,
    table_columns: u32,
    page_size: u32,
    page_count: u32,
    k_bs: u32,
    v_bs: u32,
) {
    let mut out = out;
    let mut lse = lse;
    let mut probabilities = probabilities;
    let cfg = FwdConfig::new(
        scale,
        tanh_scale,
        window_left,
        window_right,
        max_q,
        max_k,
        has_alibi,
        alibi_causal,
        h,
        gqa,
        hs_u32,
        0,
        q_ss,
        q_hs,
        k_bs,
        k_ss,
        k_hs,
        v_bs,
        v_ss,
        v_hs,
        alibi_bs,
    );
    let seed = if keep < 1.0 {
        unsafe { *rng.as_ptr() }
    } else {
        0
    };
    let offset = if keep < 1.0 {
        unsafe { *rng.as_ptr().add(1) }
    } else {
        0
    };
    let dropout = Dropout {
        seed,
        offset,
        threshold: (keep * 255.0) as u32,
        inv_keep: 1.0 / keep,
        probabilities: if probabilities.len() == 0 {
            core::ptr::null_mut()
        } else {
            probabilities.as_mut_ptr()
        },
        rounded_q: max_q.div_ceil(128) as u64 * 128,
        rounded_k: max_k.div_ceil(128) as u64 * 128,
        seqlen_q: max_q,
        seqlen_k: max_k,
    };
    unsafe {
        S::run(
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            out.as_mut_ptr(),
            lse.as_mut_ptr(),
            alibi.as_ptr(),
            cuq.as_ptr(),
            crate::device::inlined::optional_ptr(cuk),
            crate::device::inlined::optional_ptr(used),
            crate::device::inlined::optional_ptr(leftpad),
            total_q,
            total_k,
            cfg,
            dropout,
            Pages {
                table: table.as_ptr(),
                row_stride: table_stride,
                columns: table_columns,
                size: page_size,
                count: page_count,
            },
            Cache::NONE,
        );
    }
}

#[allow(non_camel_case_types)]
pub type cache_d32 = Varlen<false, 32, 2, 4, 64, 8, 4, false, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d32_paged = Varlen<false, 32, 2, 4, 64, 8, 4, false, true, true>;
#[allow(non_camel_case_types)]
pub type cache_d32_bf16 = Varlen<true, 32, 2, 4, 64, 8, 4, false, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d32_bf16_paged = Varlen<true, 32, 2, 4, 64, 8, 4, false, true, true>;
#[allow(non_camel_case_types)]
pub type cache_d64 = Varlen<false, 64, 4, 8, 64, 8, 4, false, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d64_paged = Varlen<false, 64, 4, 8, 64, 8, 4, false, true, true>;
#[allow(non_camel_case_types)]
pub type cache_d64_bf16 = Varlen<true, 64, 4, 8, 64, 8, 4, false, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d64_bf16_paged = Varlen<true, 64, 4, 8, 64, 8, 4, false, true, true>;
#[allow(non_camel_case_types)]
pub type cache_d96 = Varlen<false, 96, 6, 12, 32, 4, 2, false, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d96_paged = Varlen<false, 96, 6, 12, 32, 4, 2, false, true, true>;
#[allow(non_camel_case_types)]
pub type cache_d96_bf16 = Varlen<true, 96, 6, 12, 32, 4, 2, false, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d96_bf16_paged = Varlen<true, 96, 6, 12, 32, 4, 2, false, true, true>;
#[allow(non_camel_case_types)]
pub type cache_d128 = Varlen<false, 128, 8, 16, 32, 4, 2, false, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d128_paged = Varlen<false, 128, 8, 16, 32, 4, 2, false, true, true>;
#[allow(non_camel_case_types)]
pub type cache_d128_bf16 = Varlen<true, 128, 8, 16, 32, 4, 2, false, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d128_bf16_paged = Varlen<true, 128, 8, 16, 32, 4, 2, false, true, true>;
#[allow(non_camel_case_types)]
pub type cache_d192 = Varlen<false, 192, 12, 24, 32, 4, 2, false, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d192_paged = Varlen<false, 192, 12, 24, 32, 4, 2, false, true, true>;
#[allow(non_camel_case_types)]
pub type cache_d192_bf16 = Varlen<true, 192, 12, 24, 32, 4, 2, false, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d192_bf16_paged = Varlen<true, 192, 12, 24, 32, 4, 2, false, true, true>;
#[allow(non_camel_case_types)]
pub type cache_d256 = Varlen<false, 256, 16, 32, 32, 4, 2, false, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d256_paged = Varlen<false, 256, 16, 32, 32, 4, 2, false, true, true>;
#[allow(non_camel_case_types)]
pub type cache_d256_bf16 = Varlen<true, 256, 16, 32, 32, 4, 2, false, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d256_bf16_paged = Varlen<true, 256, 16, 32, 32, 4, 2, false, true, true>;
// Unsplit decode with v2's split-KV key tiles (256/128/64 by head dim), raw
// scores and an optional fused append; grouped variants own one KV head.
#[allow(non_camel_case_types)]
pub type cache_d32_wide =
    Varlen<false, 32, 2, 4, 256, 32, 16, false, false, true, false, false, 0, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d32_wide_bf16 =
    Varlen<true, 32, 2, 4, 256, 32, 16, false, false, true, false, false, 0, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d32_wide_grouped =
    Varlen<false, 32, 2, 4, 256, 32, 16, false, false, true, false, true, 0, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d32_wide_grouped_bf16 =
    Varlen<true, 32, 2, 4, 256, 32, 16, false, false, true, false, true, 0, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d64_wide =
    Varlen<false, 64, 4, 8, 256, 32, 16, false, false, true, false, false, 0, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d64_wide_bf16 =
    Varlen<true, 64, 4, 8, 256, 32, 16, false, false, true, false, false, 0, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d64_wide_grouped =
    Varlen<false, 64, 4, 8, 256, 32, 16, false, false, true, false, true, 0, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d64_wide_grouped_bf16 =
    Varlen<true, 64, 4, 8, 256, 32, 16, false, false, true, false, true, 0, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d96_wide =
    Varlen<false, 96, 6, 12, 128, 16, 8, false, false, true, false, false, 0, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d96_wide_bf16 =
    Varlen<true, 96, 6, 12, 128, 16, 8, false, false, true, false, false, 0, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d96_wide_grouped =
    Varlen<false, 96, 6, 12, 128, 16, 8, false, false, true, false, true, 0, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d96_wide_grouped_bf16 =
    Varlen<true, 96, 6, 12, 128, 16, 8, false, false, true, false, true, 0, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d128_wide =
    Varlen<false, 128, 8, 16, 128, 16, 8, false, false, true, false, false, 0, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d128_wide_bf16 =
    Varlen<true, 128, 8, 16, 128, 16, 8, false, false, true, false, false, 0, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d128_wide_grouped =
    Varlen<false, 128, 8, 16, 128, 16, 8, false, false, true, false, true, 0, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d128_wide_grouped_bf16 =
    Varlen<true, 128, 8, 16, 128, 16, 8, false, false, true, false, true, 0, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d192_wide =
    Varlen<false, 192, 12, 24, 64, 8, 4, false, false, true, false, false, 0, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d192_wide_bf16 =
    Varlen<true, 192, 12, 24, 64, 8, 4, false, false, true, false, false, 0, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d192_wide_grouped =
    Varlen<false, 192, 12, 24, 64, 8, 4, false, false, true, false, true, 0, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d192_wide_grouped_bf16 =
    Varlen<true, 192, 12, 24, 64, 8, 4, false, false, true, false, true, 0, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d256_wide =
    Varlen<false, 256, 16, 32, 64, 8, 4, false, false, true, false, false, 0, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d256_wide_bf16 =
    Varlen<true, 256, 16, 32, 64, 8, 4, false, false, true, false, false, 0, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d256_wide_grouped =
    Varlen<false, 256, 16, 32, 64, 8, 4, false, false, true, false, true, 0, false, true>;
#[allow(non_camel_case_types)]
pub type cache_d256_wide_grouped_bf16 =
    Varlen<true, 256, 16, 32, 64, 8, 4, false, false, true, false, true, 0, false, true>;
#[kernel(
    cache_d32,
    cache_d32_paged,
    cache_d32_bf16,
    cache_d32_bf16_paged,
    cache_d64,
    cache_d64_paged,
    cache_d64_bf16,
    cache_d64_bf16_paged,
    cache_d96,
    cache_d96_paged,
    cache_d96_bf16,
    cache_d96_bf16_paged,
    cache_d128,
    cache_d128_paged,
    cache_d128_bf16,
    cache_d128_bf16_paged,
    cache_d192,
    cache_d192_paged,
    cache_d192_bf16,
    cache_d192_bf16_paged,
    cache_d256,
    cache_d256_paged,
    cache_d256_bf16,
    cache_d256_bf16_paged,
    cache_d32_wide,
    cache_d32_wide_bf16,
    cache_d32_wide_grouped,
    cache_d32_wide_grouped_bf16,
    cache_d64_wide,
    cache_d64_wide_bf16,
    cache_d64_wide_grouped,
    cache_d64_wide_grouped_bf16,
    cache_d96_wide,
    cache_d96_wide_bf16,
    cache_d96_wide_grouped,
    cache_d96_wide_grouped_bf16,
    cache_d128_wide,
    cache_d128_wide_bf16,
    cache_d128_wide_grouped,
    cache_d128_wide_grouped_bf16,
    cache_d192_wide,
    cache_d192_wide_bf16,
    cache_d192_wide_grouped,
    cache_d192_wide_grouped_bf16,
    cache_d256_wide,
    cache_d256_wide_bf16,
    cache_d256_wide_grouped,
    cache_d256_wide_grouped_bf16
)]
#[launch_bounds(128)]
pub fn cached_fwd<S: Spec>(
    q: &[u32],
    k: DisjointSlice<u32>,
    v: DisjointSlice<u32>,
    out: DisjointSlice<u32>,
    lse: DisjointSlice<f32>,
    alibi: &[f32],
    cuq: &[i32],
    cuk: &[i32],
    used: &[i32],
    leftpad: &[i32],
    rng: &[u64],
    probabilities: DisjointSlice<u32>,
    scale: f32,
    tanh_scale: f32,
    window_left: i32,
    window_right: i32,
    max_q: u32,
    max_k: u32,
    has_alibi: u32,
    alibi_causal: u32,
    h: u32,
    gqa: u32,
    hs_u32: u32,
    q_ss: u32,
    q_hs: u32,
    k_ss: u32,
    k_hs: u32,
    v_ss: u32,
    v_hs: u32,
    alibi_bs: u32,
    total_q: u32,
    total_k: u32,
    keep: f32,
    table: &[i32],
    table_stride: u32,
    table_columns: u32,
    page_size: u32,
    page_count: u32,
    k_bs: u32,
    v_bs: u32,
    indices: &[i32],
    cache_batches: u32,
    new_keys: u32,
    q_bs: u32,
    append_k: &[u32],
    append_v: &[u32],
    append_k_bs: u32,
    append_k_ss: u32,
    append_k_hs: u32,
    append_v_bs: u32,
    append_v_ss: u32,
    append_v_hs: u32,
) {
    let mut k = k;
    let mut v = v;
    let mut out = out;
    let mut lse = lse;
    let mut probabilities = probabilities;
    let cfg = FwdConfig::new(
        scale,
        tanh_scale,
        window_left,
        window_right,
        max_q,
        max_k,
        has_alibi,
        alibi_causal,
        h,
        gqa,
        hs_u32,
        q_bs,
        q_ss,
        q_hs,
        k_bs,
        k_ss,
        k_hs,
        v_bs,
        v_ss,
        v_hs,
        alibi_bs,
    );
    let seed = if keep < 1.0 {
        unsafe { *rng.as_ptr() }
    } else {
        0
    };
    let offset = if keep < 1.0 {
        unsafe { *rng.as_ptr().add(1) }
    } else {
        0
    };
    let dropout = Dropout {
        seed,
        offset,
        threshold: (keep * 255.0) as u32,
        inv_keep: 1.0 / keep,
        probabilities: if probabilities.len() == 0 {
            core::ptr::null_mut()
        } else {
            probabilities.as_mut_ptr()
        },
        rounded_q: max_q.div_ceil(128) as u64 * 128,
        rounded_k: max_k.div_ceil(128) as u64 * 128,
        seqlen_q: max_q,
        seqlen_k: max_k,
    };
    unsafe {
        S::run(
            q.as_ptr(),
            k.as_mut_ptr(),
            v.as_mut_ptr(),
            out.as_mut_ptr(),
            lse.as_mut_ptr(),
            alibi.as_ptr(),
            cuq.as_ptr(),
            crate::device::inlined::optional_ptr(cuk),
            crate::device::inlined::optional_ptr(used),
            crate::device::inlined::optional_ptr(leftpad),
            total_q,
            total_k,
            cfg,
            dropout,
            Pages {
                table: table.as_ptr(),
                row_stride: table_stride,
                columns: table_columns,
                size: page_size,
                count: page_count,
            },
            Cache {
                indices: crate::device::inlined::optional_ptr(indices),
                batches: cache_batches,
                new_keys,
                splits: 1,
                append_k: crate::device::inlined::optional_ptr(append_k),
                append_v: crate::device::inlined::optional_ptr(append_v),
                append_ks: crate::device::config::Strides {
                    batch: append_k_bs,
                    seq: append_k_ss,
                    head: append_k_hs,
                },
                append_vs: crate::device::config::Strides {
                    batch: append_v_bs,
                    seq: append_v_ss,
                    head: append_v_hs,
                },
            },
        );
    }
}

// FP32 partial-output specializations for cached split-K attention.
#[allow(non_camel_case_types)]
pub type split_d32 = Varlen<false, 32, 2, 4, 64, 8, 4, false, false, true, true>;
#[allow(non_camel_case_types)]
pub type split_d32_paged = Varlen<false, 32, 2, 4, 64, 8, 4, false, true, true, true>;
#[allow(non_camel_case_types)]
pub type split_d32_bf16 = Varlen<true, 32, 2, 4, 64, 8, 4, false, false, true, true>;
#[allow(non_camel_case_types)]
pub type split_d32_bf16_paged = Varlen<true, 32, 2, 4, 64, 8, 4, false, true, true, true>;
#[allow(non_camel_case_types)]
pub type split_d64 = Varlen<false, 64, 4, 8, 64, 8, 4, false, false, true, true>;
#[allow(non_camel_case_types)]
pub type split_d64_paged = Varlen<false, 64, 4, 8, 64, 8, 4, false, true, true, true>;
#[allow(non_camel_case_types)]
pub type split_d64_bf16 = Varlen<true, 64, 4, 8, 64, 8, 4, false, false, true, true>;
#[allow(non_camel_case_types)]
pub type split_d64_bf16_paged = Varlen<true, 64, 4, 8, 64, 8, 4, false, true, true, true>;
#[allow(non_camel_case_types)]
pub type split_d96 = Varlen<false, 96, 6, 12, 32, 4, 2, false, false, true, true>;
#[allow(non_camel_case_types)]
pub type split_d96_paged = Varlen<false, 96, 6, 12, 32, 4, 2, false, true, true, true>;
#[allow(non_camel_case_types)]
pub type split_d96_bf16 = Varlen<true, 96, 6, 12, 32, 4, 2, false, false, true, true>;
#[allow(non_camel_case_types)]
pub type split_d96_bf16_paged = Varlen<true, 96, 6, 12, 32, 4, 2, false, true, true, true>;
#[allow(non_camel_case_types)]
pub type split_d128 = Varlen<false, 128, 8, 16, 32, 4, 2, false, false, true, true>;
#[allow(non_camel_case_types)]
pub type split_d128_paged = Varlen<false, 128, 8, 16, 32, 4, 2, false, true, true, true>;
#[allow(non_camel_case_types)]
pub type split_d128_bf16 = Varlen<true, 128, 8, 16, 32, 4, 2, false, false, true, true>;
#[allow(non_camel_case_types)]
pub type split_d128_bf16_paged = Varlen<true, 128, 8, 16, 32, 4, 2, false, true, true, true>;
#[allow(non_camel_case_types)]
pub type split_d192 = Varlen<false, 192, 12, 24, 32, 4, 2, false, false, true, true>;
#[allow(non_camel_case_types)]
pub type split_d192_paged = Varlen<false, 192, 12, 24, 32, 4, 2, false, true, true, true>;
#[allow(non_camel_case_types)]
pub type split_d192_bf16 = Varlen<true, 192, 12, 24, 32, 4, 2, false, false, true, true>;
#[allow(non_camel_case_types)]
pub type split_d192_bf16_paged = Varlen<true, 192, 12, 24, 32, 4, 2, false, true, true, true>;
#[allow(non_camel_case_types)]
pub type split_d256 = Varlen<false, 256, 16, 32, 32, 4, 2, false, false, true, true>;
#[allow(non_camel_case_types)]
pub type split_d256_paged = Varlen<false, 256, 16, 32, 32, 4, 2, false, true, true, true>;
#[allow(non_camel_case_types)]
pub type split_d256_bf16 = Varlen<true, 256, 16, 32, 32, 4, 2, false, false, true, true>;
#[allow(non_camel_case_types)]
pub type split_d256_bf16_paged = Varlen<true, 256, 16, 32, 32, 4, 2, false, true, true, true>;
// Packed decode reuses the split core while reading cumulative lengths.
#[allow(non_camel_case_types)]
pub type packed_split_d64 = Varlen<false, 64, 4, 8, 64, 8, 4, false, false, false, true>;
#[allow(non_camel_case_types)]
pub type packed_split_d64_paged = Varlen<false, 64, 4, 8, 64, 8, 4, false, true, false, true>;
#[allow(non_camel_case_types)]
pub type packed_split_d64_bf16 = Varlen<true, 64, 4, 8, 64, 8, 4, false, false, false, true>;
#[allow(non_camel_case_types)]
pub type packed_split_d64_bf16_paged = Varlen<true, 64, 4, 8, 64, 8, 4, false, true, false, true>;
#[allow(non_camel_case_types)]
pub type packed_split_d128 = Varlen<false, 128, 8, 16, 32, 4, 2, false, false, false, true>;
#[allow(non_camel_case_types)]
pub type packed_split_d128_paged = Varlen<false, 128, 8, 16, 32, 4, 2, false, true, false, true>;
#[allow(non_camel_case_types)]
pub type packed_split_d128_bf16 = Varlen<true, 128, 8, 16, 32, 4, 2, false, false, false, true>;
#[allow(non_camel_case_types)]
pub type packed_split_d128_bf16_paged =
    Varlen<true, 128, 8, 16, 32, 4, 2, false, true, false, true>;

#[allow(non_camel_case_types)]
pub type packed_grouped_split_d128 =
    Varlen<false, 128, 8, 16, 32, 4, 2, false, false, false, true, true>;
#[allow(non_camel_case_types)]
pub type packed_grouped_split_d128_paged =
    Varlen<false, 128, 8, 16, 32, 4, 2, false, true, false, true, true>;
#[allow(non_camel_case_types)]
pub type packed_grouped_split_d128_bf16 =
    Varlen<true, 128, 8, 16, 32, 4, 2, false, false, false, true, true>;
#[allow(non_camel_case_types)]
pub type packed_grouped_split_d128_bf16_paged =
    Varlen<true, 128, 8, 16, 32, 4, 2, false, true, false, true, true>;

#[kernel(
    packed_grouped_split_d128,
    packed_grouped_split_d128_paged,
    packed_grouped_split_d128_bf16,
    packed_grouped_split_d128_bf16_paged,
    packed_split_d64,
    packed_split_d64_paged,
    packed_split_d64_bf16,
    packed_split_d64_bf16_paged,
    packed_split_d128,
    packed_split_d128_paged,
    packed_split_d128_bf16,
    packed_split_d128_bf16_paged,
    split_d32,
    split_d32_paged,
    split_d32_bf16,
    split_d32_bf16_paged,
    split_d64,
    split_d64_paged,
    split_d64_bf16,
    split_d64_bf16_paged,
    split_d96,
    split_d96_paged,
    split_d96_bf16,
    split_d96_bf16_paged,
    split_d128,
    split_d128_paged,
    split_d128_bf16,
    split_d128_bf16_paged,
    split_d192,
    split_d192_paged,
    split_d192_bf16,
    split_d192_bf16_paged,
    split_d256,
    split_d256_paged,
    split_d256_bf16,
    split_d256_bf16_paged
)]
#[launch_bounds(128)]
pub fn cached_split_fwd<S: Spec>(
    q: &[u32],
    k: &[u32],
    v: &[u32],
    out: DisjointSlice<u32>,
    lse: DisjointSlice<f32>,
    alibi: &[f32],
    cuq: &[i32],
    cuk: &[i32],
    used: &[i32],
    leftpad: &[i32],
    rng: &[u64],
    probabilities: DisjointSlice<u32>,
    scale: f32,
    tanh_scale: f32,
    window_left: i32,
    window_right: i32,
    max_q: u32,
    max_k: u32,
    has_alibi: u32,
    alibi_causal: u32,
    h: u32,
    gqa: u32,
    hs_u32: u32,
    q_ss: u32,
    q_hs: u32,
    k_ss: u32,
    k_hs: u32,
    v_ss: u32,
    v_hs: u32,
    alibi_bs: u32,
    total_q: u32,
    total_k: u32,
    keep: f32,
    table: &[i32],
    table_stride: u32,
    table_columns: u32,
    page_size: u32,
    page_count: u32,
    k_bs: u32,
    v_bs: u32,
    indices: &[i32],
    cache_batches: u32,
    new_keys: u32,
    q_bs: u32,
    splits: u32,
) {
    let mut out = out;
    let mut lse = lse;
    let mut probabilities = probabilities;
    let cfg = FwdConfig::new(
        scale,
        tanh_scale,
        window_left,
        window_right,
        max_q,
        max_k,
        has_alibi,
        alibi_causal,
        h,
        gqa,
        hs_u32,
        q_bs,
        q_ss,
        q_hs,
        k_bs,
        k_ss,
        k_hs,
        v_bs,
        v_ss,
        v_hs,
        alibi_bs,
    );
    let seed = if keep < 1.0 {
        unsafe { *rng.as_ptr() }
    } else {
        0
    };
    let offset = if keep < 1.0 {
        unsafe { *rng.as_ptr().add(1) }
    } else {
        0
    };
    let dropout = Dropout {
        seed,
        offset,
        threshold: (keep * 255.0) as u32,
        inv_keep: 1.0 / keep,
        probabilities: if probabilities.len() == 0 {
            core::ptr::null_mut()
        } else {
            probabilities.as_mut_ptr()
        },
        rounded_q: max_q.div_ceil(128) as u64 * 128,
        rounded_k: max_k.div_ceil(128) as u64 * 128,
        seqlen_q: max_q,
        seqlen_k: max_k,
    };
    unsafe {
        S::run(
            q.as_ptr(),
            k.as_ptr(),
            v.as_ptr(),
            out.as_mut_ptr(),
            lse.as_mut_ptr(),
            alibi.as_ptr(),
            cuq.as_ptr(),
            crate::device::inlined::optional_ptr(cuk),
            crate::device::inlined::optional_ptr(used),
            crate::device::inlined::optional_ptr(leftpad),
            total_q,
            total_k,
            cfg,
            dropout,
            Pages {
                table: table.as_ptr(),
                row_stride: table_stride,
                columns: table_columns,
                size: page_size,
                count: page_count,
            },
            Cache {
                indices: crate::device::inlined::optional_ptr(indices),
                batches: cache_batches,
                new_keys,
                splits,
                ..Cache::NONE
            },
        );
    }
}
