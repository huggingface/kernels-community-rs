//! Device kernels.
//!
//! The input is viewed as a matrix of `rows` x `n`, where `n` is the hidden
//! size. Both the forward and backward pass use one block per row (or per
//! range of rows) and stride the block's threads over the row's columns. Each
//! row needs one block-wide sum, which [`kernels::block_sum`] computes with
//! warp shuffles and a little shared memory.

use cuda_device::cuda_module;

#[cuda_module]
pub mod kernels {
    use cuda_device::{DisjointSlice, SharedArray, kernel, thread, warp};

    use crate::casting::CastingMode;
    use crate::element::{Element, GROUP, Group, round, round_if};

    // Bind to libdevice for rsqrtf.
    #[cuda_device::device]
    unsafe extern "C" {
        fn __nv_rsqrtf(x: f32) -> f32;
    }

    /// Inverse square root.
    #[inline(always)]
    fn rsqrt(x: f32) -> f32 {
        unsafe { __nv_rsqrtf(x) }
    }

    /// One partial sum per warp; a block has at most 32 warps.
    type Scratch = SharedArray<f32, 32>;

    /// The sum of `value` over all threads of the block, returned to every
    /// thread. The block size must be a multiple of 32.
    ///
    /// # Safety
    ///
    /// Every thread of the block must call this with the same `scratch`, and
    /// reusing `scratch` requires a barrier after the previous call.
    #[inline(always)]
    unsafe fn block_sum(value: f32, scratch: *mut Scratch) -> f32 {
        let scratch = unsafe { SharedArray::as_raw_mut_ptr(scratch) };
        // Every lane gets its warp's sum.
        let warp_sum = warp::reduce_sum_f32(value);
        if warp::lane_id() == 0 {
            unsafe {
                scratch
                    .add(thread::threadIdx_x() as usize / 32)
                    .write(warp_sum)
            };
        }
        thread::sync_threads();
        // Every warp reduces the warp sums the same way, so all threads get
        // the same result.
        let lane = warp::lane_id();
        let warps = thread::blockDim_x() / 32;
        let partial = if lane < warps {
            unsafe { scratch.add(lane as usize).read() }
        } else {
            0.0
        };
        warp::reduce_sum_f32(partial)
    }

    /// `y = x / rms(x) * (offset + weight)` for each row, also storing the
    /// inverse RMS `rstd` of each row for the backward pass.
    ///
    /// Launched with one block per row. Without `AFFINE` there is no weight
    /// and `weight` is never read.
    ///
    /// With `vectorized`, the threads load and store groups of [`GROUP`]
    /// elements with 16-byte instructions, and keep their first group of `x`
    /// and `weight` in registers between the two passes. The remaining
    /// columns are handled one at a time.
    #[kernel]
    pub fn rms_norm_fwd<X: Element, W: Element, M: CastingMode, const AFFINE: bool>(
        mut y: DisjointSlice<X>,
        mut rstd: DisjointSlice<f32>,
        x: &[X],
        weight: &[W],
        n: u32,
        inv_n: f32,
        eps: f32,
        offset: f32,
        vectorized: u32,
    ) {
        static mut SCRATCH: Scratch = SharedArray::UNINIT;

        // Columns are 32-bit (`n` is at most 65536); only the row offset
        // needs 64 bits.
        let row = thread::blockIdx_x() as usize;
        let row_start = row * n as usize;
        let first = thread::threadIdx_x();
        let step = thread::blockDim_x();
        const GROUP_U32: u32 = GROUP as u32;
        // Columns before `tail` are handled in groups.
        let tail = if vectorized != 0 {
            n / GROUP_U32 * GROUP_U32
        } else {
            0
        };
        // SAFETY (all accesses through these and `weight`): the host checks
        // that `x` and `y` have `n` elements per launched row and `weight` has
        // `n` with `AFFINE`, and with `vectorized` that they are 16-byte
        // aligned.
        let x_row = unsafe { x.as_ptr().add(row_start) };
        let y_row = unsafe { y.as_mut_ptr().add(row_start) };
        let w_row = weight.as_ptr();

        let load_weights = |col: u32| -> Group {
            if AFFINE {
                unsafe { W::load_group(w_row.add(col as usize)) }
            } else {
                [0.0; GROUP]
            }
        };
        let sum_squares = |values: Group| -> f32 {
            let mut sum = 0.0f32;
            let mut i = 0;
            #[unroll]
            while i < GROUP {
                sum += values[i] * values[i];
                i += 1;
            }
            sum
        };

        // Pass 1: the sum of squares of the row. Each thread keeps its first
        // group (and its weights) in registers for pass 2; later groups are
        // loaded again.
        let first_col = first * GROUP_U32;
        let (first_x, first_w) = if first_col < tail {
            let x = unsafe { X::load_group(x_row.add(first_col as usize)) };
            (x, load_weights(first_col))
        } else {
            ([0.0; GROUP], [0.0; GROUP])
        };
        let mut sum_sq = sum_squares(first_x);
        let mut col = first_col + step * GROUP_U32;
        while col < tail {
            sum_sq += sum_squares(unsafe { X::load_group(x_row.add(col as usize)) });
            col += step * GROUP_U32;
        }
        let mut col = tail + first;
        while col < n {
            let value = unsafe { *x_row.add(col as usize) }.to_f32();
            sum_sq += value * value;
            col += step;
        }
        // SAFETY: every thread of the block calls this once, with this scratch.
        let sum_sq = unsafe { block_sum(sum_sq, &raw mut SCRATCH) };

        // rstd = 1 / sqrt(mean(x^2) + eps)
        let mean_square = sum_sq * inv_n;
        let row_rstd = if M::INPUT_DTYPE_MATH {
            round::<X>(rsqrt(round::<X>(mean_square) + round::<X>(eps)))
        } else {
            rsqrt(mean_square + eps)
        };
        if first == 0 {
            // SAFETY: the host checks that `rstd` has one element per row.
            unsafe { *rstd.get_unchecked_mut(row) = row_rstd };
        }

        // y = x * rstd * (offset + w)
        let normalize = |x: f32, w: f32| -> f32 {
            let x_hat = round_if::<X>(M::ROUND_X_HAT, x * row_rstd);
            if !AFFINE {
                return x_hat;
            }
            let factor = if M::INPUT_DTYPE_MATH {
                // Liger casts the offset to the input dtype and adds in the
                // weight dtype.
                round::<W>(round::<X>(offset) + w)
            } else {
                offset + w
            };
            x_hat * factor
        };
        let normalize_group = |values: Group, weights: Group| -> Group {
            core::array::from_fn(|i| normalize(values[i], weights[i]))
        };

        // Pass 2: normalize and scale. Groups after the first are still in
        // the L1 cache.
        if first_col < tail {
            let out = normalize_group(first_x, first_w);
            unsafe { X::store_group(y_row.add(first_col as usize), out) };
        }
        let mut col = first_col + step * GROUP_U32;
        while col < tail {
            unsafe {
                let values = X::load_group(x_row.add(col as usize));
                let out = normalize_group(values, load_weights(col));
                X::store_group(y_row.add(col as usize), out);
            }
            col += step * GROUP_U32;
        }
        let mut col = tail + first;
        while col < n {
            unsafe {
                let x = (*x_row.add(col as usize)).to_f32();
                let w = if AFFINE {
                    (*w_row.add(col as usize)).to_f32()
                } else {
                    0.0
                };
                *y_row.add(col as usize) = X::from_f32(normalize(x, w));
            }
            col += step;
        }
    }

    /// Gradients of [`rms_norm_fwd`]:
    ///
    /// ```text
    /// m  = dy * (offset + weight)
    /// dx = rstd * m - rstd^3 / n * dot(m, x) * x
    /// dw = sum over rows of dy * x * rstd
    /// ```
    ///
    /// The `rows` are split into one contiguous range per block. With `AFFINE`
    /// each block sums its rows' weight gradients into its own row of
    /// `dw_partial` (`gridDim.x` x `n`), which [`reduce_weight_grad`] then sums.
    /// Without `AFFINE`, `weight` and `dw_partial` are never accessed.
    ///
    /// With `in_place`, the gradient is read from `dx` and `dy` is ignored.
    #[kernel]
    pub fn rms_norm_bwd<X: Element, W: Element, M: CastingMode, const AFFINE: bool>(
        mut dx: DisjointSlice<X>,
        mut dw_partial: DisjointSlice<f32>,
        dy: &[X],
        x: &[X],
        weight: &[W],
        rstd: &[f32],
        rows: u32,
        n: u32,
        inv_n: f32,
        offset: f32,
        in_place: u32,
    ) {
        static mut SCRATCH: Scratch = SharedArray::UNINIT;

        // Columns are 32-bit (`n` is at most 65536); only row offsets need
        // 64 bits.
        let block_id = thread::blockIdx_x();
        let first = thread::threadIdx_x();
        let step = thread::blockDim_x();

        let rows_per_block = rows.div_ceil(thread::gridDim_x());
        let mut row = block_id * rows_per_block;
        let row_end = core::cmp::min(row + rows_per_block, rows);

        // SAFETY (all accesses through these pointers): the host checks that
        // `dy`, `dx` and `x` have `n` elements per row, `rstd` one element per
        // row, and with `AFFINE` that `weight` has `n` elements and
        // `dw_partial` `n` per block.
        //
        // Each thread reads and writes only its own columns, so reading the
        // gradient from `dx` while overwriting it is safe.
        let dx_ptr = dx.as_mut_ptr();
        let dy_ptr: *const X = if in_place != 0 { dx_ptr } else { dy.as_ptr() };
        let x_ptr = x.as_ptr();
        let w_ptr = weight.as_ptr();
        let dw_row = dw_partial
            .as_mut_ptr()
            .wrapping_add(block_id as usize * n as usize);

        if AFFINE {
            let mut col = first;
            while col < n {
                unsafe { *dw_row.add(col as usize) = 0.0 };
                col += step;
            }
        }

        // The gradient `m` of the normalized input, before the weight.
        let grad = |dy: f32, col: u32| -> f32 {
            if AFFINE {
                dy * (unsafe { *w_ptr.add(col as usize) }.to_f32() + offset)
            } else {
                dy
            }
        };

        while row < row_end {
            let row_start = row as usize * n as usize;
            let (dy_row, x_row, dx_row) = unsafe {
                (
                    dy_ptr.add(row_start),
                    x_ptr.add(row_start),
                    dx_ptr.add(row_start),
                )
            };
            let row_rstd = unsafe { *rstd.get_unchecked(row as usize) };

            // Pass 1: dot(m, x).
            let mut dot = 0.0f32;
            let mut col = first;
            while col < n {
                unsafe {
                    let dy = (*dy_row.add(col as usize)).to_f32();
                    let x = (*x_row.add(col as usize)).to_f32();
                    dot += grad(dy, col) * x;
                }
                col += step;
            }
            // SAFETY: every thread of the block calls this once per row, with a
            // barrier before the next row reuses the scratch.
            let dot = unsafe { block_sum(dot, &raw mut SCRATCH) };
            let correction = -inv_n * row_rstd * row_rstd * dot;

            // Pass 2: dx, and the weight gradient of this row.
            let mut col = first;
            while col < n {
                unsafe {
                    let c = col as usize;
                    let dy = (*dy_row.add(c)).to_f32();
                    let x = (*x_row.add(c)).to_f32();
                    let dx = row_rstd * grad(dy, col) + row_rstd * (correction * x);
                    *dx_row.add(c) = X::from_f32(dx);
                    if AFFINE {
                        // dw += dy * x * rstd
                        let x_hat = round_if::<X>(M::ROUND_WEIGHT_GRAD, x * row_rstd);
                        *dw_row.add(c) += round_if::<X>(M::ROUND_WEIGHT_GRAD, dy * x_hat);
                    }
                }
                col += step;
            }

            // The next row's `block_sum` reuses the scratch.
            thread::sync_threads();
            row += 1;
        }
    }

    /// Sums the `blocks` partial weight gradients of [`rms_norm_bwd`] per
    /// column. Launched with one thread per column.
    #[kernel]
    pub fn reduce_weight_grad<W: Element>(
        mut dw: DisjointSlice<W>,
        dw_partial: &[f32],
        blocks: u32,
    ) {
        let n = dw.len();
        let idx = thread::index_1d();
        let col = idx.get();
        if let Some(out) = dw.get_mut(idx) {
            let mut sum = 0.0f32;
            let mut block = 0;
            while block < blocks as usize {
                // SAFETY: the host checks that `dw_partial` has `blocks * n` elements.
                sum += unsafe { *dw_partial.get_unchecked(block * n + col) };
                block += 1;
            }
            *out = W::from_f32(sum);
        }
    }
}
