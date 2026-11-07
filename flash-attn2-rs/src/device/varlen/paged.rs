// Tiles never cross pages: v2 pages are multiples of 256 rows.
use crate::device::config::{FwdConfig, KvRange, LaunchTile, TensorBases};
use crate::device::forward::{SmemBuffers, SmemLayout};
use cuda_device::async_copy::cp_async_ca_zfill_16;
use cuda_device::{ptx_asm, thread};

#[derive(Clone, Copy)]
pub struct Pages {
    pub table: *const i32,
    pub row_stride: u32,
    pub columns: u32,
    pub size: u32,
    pub count: u32,
}
impl Pages {
    pub const NONE: Self = Self {
        table: core::ptr::null(),
        row_stride: 0,
        columns: 0,
        size: 0,
        count: 0,
    };
}

// Page-table entry for the tile at `base`, or -1 past the table row.
// Loaded one tile ahead so the global latency hides behind a tile of MMAs.
#[inline(always)]
pub unsafe fn lookup_page(pages: Pages, batch: u32, base: u32) -> i32 {
    unsafe {
        let logical_page = base / pages.size;
        if logical_page < pages.columns {
            *pages
                .table
                .add(batch as usize * pages.row_stride as usize + logical_page as usize)
        } else {
            -1
        }
    }
}

#[inline(always)]
pub unsafe fn stage_tile<const HD: usize, const KEYS: u32, const THREADS: u32>(
    smem: *mut u32,
    src: *const u32,
    base: u32,
    page: i32,
    head_base: usize,
    seq_stride: u32,
    page_stride: u32,
    cfg: &FwdConfig,
    tile: &LaunchTile,
    layout: SmemLayout,
    pages: Pages,
) {
    unsafe {
        let valid_page = page >= 0 && (page as u32) < pages.count;
        let page_base = if valid_page {
            page as usize * page_stride as usize + head_base
        } else {
            0
        };
        let within = base % pages.size;
        let copies = (KEYS as usize * (HD / 2)) / (THREADS as usize * 4);
        // Constant copy indices keep row/col arithmetic out of the loop.
        let unrolled = copies * THREADS as usize * 4 == KEYS as usize * (HD / 2);
        let mut copy = 0usize;
        while copy < copies || (!unrolled && copy * THREADS as usize * 4 < KEYS as usize * (HD / 2))
        {
            thread::__unroll_config::<0>();
            let i = tile.tid as usize * 4 + copy * THREADS as usize * 4;
            let row = i / (HD / 2);
            let col = i % (HD / 2);
            let valid = valid_page
                && base as usize + row < cfg.seqlen_k as usize
                && col < cfg.hs_u32 as usize;
            let offset = if valid {
                page_base + (within as usize + row) * seq_stride as usize + col
            } else {
                0
            };
            cp_async_ca_zfill_16(
                smem.add(row * layout.stride + col),
                src.add(offset) as *const u8,
                if valid { 16 } else { 0 },
            );
            copy += 1;
        }
    }
}

#[inline(always)]
unsafe fn stage_kv_pair<
    const HD: usize,
    const KEYS: u32,
    const THREADS: u32,
    const UNROLL_COPY: bool,
    const RECURRENT_COPY: bool,
>(
    k_smem: *mut u32,
    v_smem: *mut u32,
    k: *const u32,
    v: *const u32,
    base: u32,
    page: i32,
    cfg: &FwdConfig,
    tile: &LaunchTile,
    bases: &TensorBases,
    layout: SmemLayout,
    pages: Pages,
) {
    unsafe {
        stage_tile::<HD, KEYS, THREADS>(
            k_smem,
            k,
            base,
            page,
            bases.k,
            cfg.k.seq,
            cfg.k.batch,
            cfg,
            tile,
            layout,
            pages,
        );
        stage_tile::<HD, KEYS, THREADS>(
            v_smem,
            v,
            base,
            page,
            bases.v,
            cfg.v.seq,
            cfg.v.batch,
            cfg,
            tile,
            layout,
            pages,
        );
    }
}

// `page` belongs to the first staged tile (forward or reverse order).
#[inline(always)]
pub unsafe fn prefetch_first_kv<
    const HD: usize,
    const KEYS: u32,
    const THREADS: u32,
    const UNROLL_COPY: bool,
    const RECURRENT_COPY: bool,
>(
    k: *const u32,
    v: *const u32,
    cfg: &FwdConfig,
    tile: &LaunchTile,
    bases: &TensorBases,
    range: KvRange,
    smem: SmemBuffers,
    layout: SmemLayout,
    reverse: bool,
    pages: Pages,
    page: i32,
) {
    unsafe {
        if !layout.double && range.start_tile < range.num_tiles {
            stage_tile::<HD, KEYS, THREADS>(
                smem.k0,
                k,
                (if reverse {
                    range.num_tiles - 1
                } else {
                    range.start_tile
                }) * KEYS,
                page,
                bases.k,
                cfg.k.seq,
                cfg.k.batch,
                cfg,
                tile,
                layout,
                pages,
            );
            ptx_asm!("cp.async.commit_group;", clobber("memory"));
        }
        if layout.double && range.start_tile < range.num_tiles {
            stage_kv_pair::<HD, KEYS, THREADS, UNROLL_COPY, RECURRENT_COPY>(
                smem.k0,
                smem.v0,
                k,
                v,
                (if reverse {
                    range.num_tiles - 1
                } else {
                    range.start_tile
                }) * KEYS,
                page,
                cfg,
                tile,
                bases,
                layout,
                pages,
            );
            ptx_asm!("cp.async.commit_group;", clobber("memory"));
        }
    }
}

// `page` is tile j's table entry and `next_page` the following tile's.
#[inline(always)]
pub unsafe fn stage_kv_tile<
    const HD: usize,
    const KEYS: u32,
    const THREADS: u32,
    const UNROLL_COPY: bool,
    const RECURRENT_COPY: bool,
>(
    k: *const u32,
    v: *const u32,
    cfg: &FwdConfig,
    tile: &LaunchTile,
    bases: &TensorBases,
    range: KvRange,
    smem: SmemBuffers,
    layout: SmemLayout,
    j: u32,
    reverse: bool,
    pages: Pages,
    page: i32,
    next_page: i32,
) -> (*const u32, *const u32) {
    unsafe {
        let kv_base = j * KEYS;
        if layout.triple {
            ptx_asm!("cp.async.wait_group 0;", clobber("memory"));
            // Finish reading the previous V tile and the reusable K buffer.
            thread::sync_threads();
            let odd = ((if reverse {
                range.num_tiles - 1 - j
            } else {
                j - range.start_tile
            }) & 1)
                == 1;
            let current_k = if odd { smem.k1 } else { smem.k0 };
            stage_tile::<HD, KEYS, THREADS>(
                smem.v0,
                v,
                kv_base,
                page,
                bases.v,
                cfg.v.seq,
                cfg.v.batch,
                cfg,
                tile,
                layout,
                pages,
            );
            ptx_asm!("cp.async.commit_group;", clobber("memory"));
            if if reverse {
                j > range.start_tile
            } else {
                j + 1 < range.num_tiles
            } {
                let next_k = if odd { smem.k0 } else { smem.k1 };
                let next_base = (if reverse { j - 1 } else { j + 1 }) * KEYS;
                stage_tile::<HD, KEYS, THREADS>(
                    next_k,
                    k,
                    next_base,
                    next_page,
                    bases.k,
                    cfg.k.seq,
                    cfg.k.batch,
                    cfg,
                    tile,
                    layout,
                    pages,
                );
                ptx_asm!("cp.async.commit_group;", clobber("memory"));
            }
            return (current_k as *const u32, smem.v0 as *const u32);
        }
        if layout.double {
            let odd = ((if reverse {
                range.num_tiles - 1 - j
            } else {
                j - range.start_tile
            }) & 1)
                == 1;
            if if reverse {
                j > range.start_tile
            } else {
                j + 1 < range.num_tiles
            } {
                let next_base = (if reverse { j - 1 } else { j + 1 }) * KEYS;
                let nxt_k = if odd { smem.k0 } else { smem.k1 };
                let nxt_v = if odd { smem.v0 } else { smem.v1 };
                stage_kv_pair::<HD, KEYS, THREADS, UNROLL_COPY, RECURRENT_COPY>(
                    nxt_k, nxt_v, k, v, next_base, next_page, cfg, tile, bases, layout, pages,
                );
                ptx_asm!("cp.async.commit_group;", clobber("memory"));
                ptx_asm!("cp.async.wait_group 1;", clobber("memory"));
            } else {
                ptx_asm!("cp.async.wait_group 0;", clobber("memory"));
            }
            thread::sync_threads();
            let kb = if odd { smem.k1 } else { smem.k0 };
            let vb = if odd { smem.v1 } else { smem.v0 };
            (kb as *const u32, vb as *const u32)
        } else {
            ptx_asm!("cp.async.wait_group 0;", clobber("memory"));
            thread::sync_threads();
            stage_tile::<HD, KEYS, THREADS>(
                smem.v0,
                v,
                kv_base,
                page,
                bases.v,
                cfg.v.seq,
                cfg.v.batch,
                cfg,
                tile,
                layout,
                pages,
            );
            ptx_asm!("cp.async.commit_group;", clobber("memory"));
            (smem.k0 as *const u32, smem.v0 as *const u32)
        }
    }
}
