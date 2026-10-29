//! Host side: argument validation, kernel selection and launches behind the
//! tvm-ffi exports. Every op writes into preallocated outputs and launches on
//! the caller's current stream.

use cuda_core::simt::LaunchConfig;
use half::{bf16, f16};
use kernels_cuda::{Gpu, cuda_error};
use tvm_ffi::tvm_ffi_sys::dlpack::DLDataType;
use tvm_ffi::{Error, Result, Tensor, VALUE_ERROR};

use crate::casting::{CastingMode, Gemma, Kind, Llama, NoCasting};
use crate::device::kernels::{self, LoadedModule};
use crate::element::Element;

/// The largest supported hidden size, as in Liger.
const MAX_COLS: i64 = 65536;

/// The CUDA limit for `gridDim.x`.
const MAX_GRID_X: i64 = i32::MAX as i64;

/// Threads per block of [`kernels::reduce_weight_grad`].
const REDUCE_THREADS: u32 = 256;

fn value_error(message: &str) -> Error {
    Error::new(VALUE_ERROR, message, "")
}

fn casting_mode(name: &str) -> Result<Kind> {
    match name {
        "llama" => Ok(Kind::Llama),
        "gemma" => Ok(Kind::Gemma),
        "none" => Ok(Kind::None),
        _ => Err(value_error(&format!(
            "invalid casting mode {name:?}, expected \"llama\", \"gemma\" or \"none\""
        ))),
    }
}

fn same_dtype(a: DLDataType, b: DLDataType) -> bool {
    (a.code, a.bits, a.lanes) == (b.code, b.bits, b.lanes)
}

fn has_dtype<T: kernels_cuda::Element>(tensor: &Tensor) -> bool {
    let dtype = tensor.dtype();
    dtype.code == T::CODE as u8 && dtype.bits == T::BITS && dtype.lanes == 1
}

/// The `[rows, n]` matrix view of `x` over its last dimension.
fn rows_and_cols(x: &Tensor) -> Result<(i64, i64)> {
    let Some((&n, batch)) = x.shape().split_last() else {
        return Err(value_error("x must have at least one dimension"));
    };
    if n > MAX_COLS {
        return Err(value_error(&format!(
            "the hidden size of x is {n}, at most {MAX_COLS} is supported"
        )));
    }
    let rows: i64 = batch.iter().product();
    if rows > MAX_GRID_X {
        return Err(value_error(&format!(
            "x has {rows} rows, at most {MAX_GRID_X} are supported"
        )));
    }
    Ok((rows, n))
}

/// Checks that `tensor` has the shape and dtype of `like`.
fn check_like(tensor: &Tensor, like: &Tensor, name: &str, like_name: &str) -> Result<()> {
    if tensor.shape() != like.shape() || !same_dtype(tensor.dtype(), like.dtype()) {
        return Err(value_error(&format!(
            "{name} must have the shape and dtype of {like_name}"
        )));
    }
    Ok(())
}

fn check_f32(tensor: &Tensor, numel: i64, name: &str) -> Result<()> {
    if !has_dtype::<f32>(tensor) || tensor.numel() as i64 != numel {
        return Err(value_error(&format!(
            "{name} must be a float32 tensor with {numel} elements"
        )));
    }
    Ok(())
}

/// Checks that `weight` is `[n]`; its dtype is checked by [`dispatch`].
fn check_weight(weight: Option<&Tensor>, n: i64) -> Result<()> {
    match weight {
        Some(weight) if weight.shape() != [n] => Err(value_error(&format!(
            "weight must have shape [{n}], got {:?}",
            weight.shape()
        ))),
        _ => Ok(()),
    }
}

/// Threads per block for rows of `n` elements, the warp counts Liger uses.
fn block_threads(n: i64) -> u32 {
    let warps = match n {
        ..2048 => 4,
        ..8192 => 8,
        ..32768 => 16,
        _ => 32,
    };
    warps * 32
}

/// Whether the kernels can load and store rows of `n` elements of `X` with
/// 16-byte instructions: every row and tensor starts at a 16-byte boundary.
fn can_vectorize<X>(n: i64, tensors: &[&Tensor]) -> bool {
    (n as usize * size_of::<X>()).is_multiple_of(16)
        && tensors
            .iter()
            .all(|t| (t.data_ptr() as usize).is_multiple_of(16))
}

fn config(blocks: u32, threads: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// A launch of the generic kernels, for the casting mode and element types
/// chosen at runtime by [`dispatch`]. `X` is the input dtype, `W` the weight
/// dtype, and `AFFINE` whether there is a weight.
trait Launch {
    fn launch<X, W, M, const AFFINE: bool>(self, gpu: &Gpu<LoadedModule>) -> Result<()>
    where
        X: Element + kernels_cuda::Element,
        W: Element + kernels_cuda::Element,
        M: CastingMode;
}

fn dispatch(launch: impl Launch, mode: Kind, x: &Tensor, weight: Option<&Tensor>) -> Result<()> {
    // `view` checks that all tensors are contiguous, on this device, and of
    // the element type.
    let gpu = kernels_cuda::load(x.device(), kernels::from_module)?;
    match mode {
        Kind::Llama => with_input::<Llama>(launch, &gpu, x, weight),
        Kind::Gemma => with_input::<Gemma>(launch, &gpu, x, weight),
        Kind::None => with_input::<NoCasting>(launch, &gpu, x, weight),
    }
}

fn with_input<M: CastingMode>(
    launch: impl Launch,
    gpu: &Gpu<LoadedModule>,
    x: &Tensor,
    weight: Option<&Tensor>,
) -> Result<()> {
    if has_dtype::<f16>(x) {
        with_weight::<f16, M>(launch, gpu, x, weight)
    } else if has_dtype::<bf16>(x) {
        with_weight::<bf16, M>(launch, gpu, x, weight)
    } else if has_dtype::<f32>(x) {
        with_weight::<f32, M>(launch, gpu, x, weight)
    } else {
        Err(value_error(
            "unsupported dtype, expected float16, bfloat16 or float32",
        ))
    }
}

fn with_weight<X, M>(
    launch: impl Launch,
    gpu: &Gpu<LoadedModule>,
    x: &Tensor,
    weight: Option<&Tensor>,
) -> Result<()>
where
    X: Element + kernels_cuda::Element,
    M: CastingMode,
{
    match weight {
        None => launch.launch::<X, X, M, false>(gpu),
        Some(weight) if same_dtype(weight.dtype(), x.dtype()) => {
            launch.launch::<X, X, M, true>(gpu)
        }
        Some(weight) if has_dtype::<f32>(weight) => launch.launch::<X, f32, M, true>(gpu),
        Some(_) => Err(value_error("weight must have the dtype of x or be float32")),
    }
}

struct Forward<'a> {
    y: &'a Tensor,
    rstd: &'a Tensor,
    x: &'a Tensor,
    weight: Option<&'a Tensor>,
    rows: i64,
    n: i64,
    eps: f32,
    offset: f32,
}

impl Launch for Forward<'_> {
    fn launch<X, W, M, const AFFINE: bool>(self, gpu: &Gpu<LoadedModule>) -> Result<()>
    where
        X: Element + kernels_cuda::Element,
        W: Element + kernels_cuda::Element,
        M: CastingMode,
    {
        // Without a weight the kernel never reads it: pass `x` instead.
        let weight_or_x = self.weight.unwrap_or(self.x);
        let vectorized = can_vectorize::<X>(self.n, &[self.x, self.y, weight_or_x]);
        let x = gpu.view::<X>(self.x)?;
        let weight = gpu.view::<W>(weight_or_x)?;
        let mut y = gpu.view::<X>(self.y)?;
        let mut rstd = gpu.view::<f32>(self.rstd)?;
        // SAFETY: `rms_norm_forward` checked that `x` and `y` have `rows * n`
        // elements, `rstd` has `rows`, and `weight` has `n` with `AFFINE`;
        // `can_vectorize` checked the alignment for `vectorized`.
        unsafe {
            gpu.module.rms_norm_fwd::<X, W, M, AFFINE>(
                &gpu.stream,
                config(self.rows as u32, block_threads(self.n)),
                &mut y,
                &mut rstd,
                &x,
                &weight,
                self.n as u32,
                // A multiply in the kernel is cheaper than a division.
                1.0 / self.n as f32,
                self.eps,
                self.offset,
                vectorized as u32,
            )
        }
        .map_err(cuda_error)
    }
}

/// `y = x / rms(x) * (offset + weight)` over the last dimension of `x`, and
/// the inverse RMS of every row in `rstd` (float32, one element per row).
/// `weight` is optional and may be float32 for 16-bit `x`.
fn rms_norm_forward(
    y: Tensor,
    rstd: Tensor,
    x: Tensor,
    weight: Option<Tensor>,
    eps: f64,
    offset: f64,
    casting_mode: tvm_ffi::String,
) -> Result<()> {
    let mode = self::casting_mode(&casting_mode)?;
    let (rows, n) = rows_and_cols(&x)?;
    check_like(&y, &x, "y", "x")?;
    check_f32(&rstd, rows, "rstd")?;
    check_weight(weight.as_ref(), n)?;
    if rows == 0 || n == 0 {
        return Ok(());
    }
    let forward = Forward {
        y: &y,
        rstd: &rstd,
        x: &x,
        weight: weight.as_ref(),
        rows,
        n,
        eps: eps as f32,
        offset: offset as f32,
    };
    dispatch(forward, mode, &x, weight.as_ref())
}

struct Backward<'a> {
    dx: &'a Tensor,
    dw_partial: Option<&'a Tensor>,
    dy: Option<&'a Tensor>,
    x: &'a Tensor,
    weight: Option<&'a Tensor>,
    rstd: &'a Tensor,
    blocks: i64,
    rows: i64,
    n: i64,
    offset: f32,
}

impl Launch for Backward<'_> {
    fn launch<X, W, M, const AFFINE: bool>(self, gpu: &Gpu<LoadedModule>) -> Result<()>
    where
        X: Element + kernels_cuda::Element,
        W: Element + kernels_cuda::Element,
        M: CastingMode,
    {
        let x = gpu.view::<X>(self.x)?;
        // Placeholders for arguments the kernel never reads: `dy` when in
        // place, and `weight` and `dw_partial` without a weight.
        let dy = gpu.view::<X>(self.dy.unwrap_or(self.x))?;
        let weight = gpu.view::<W>(self.weight.unwrap_or(self.x))?;
        let mut dw_partial = gpu.view::<f32>(self.dw_partial.unwrap_or(self.rstd))?;
        let rstd = gpu.view::<f32>(self.rstd)?;
        let mut dx = gpu.view::<X>(self.dx)?;
        // SAFETY: `rms_norm_backward` checked that `dx`, `x` and `dy` (unless
        // in place) have `rows * n` elements, `rstd` has `rows`, and with
        // `AFFINE` that `weight` has `n` and `dw_partial` `blocks * n`.
        unsafe {
            gpu.module.rms_norm_bwd::<X, W, M, AFFINE>(
                &gpu.stream,
                config(self.blocks as u32, block_threads(self.n)),
                &mut dx,
                &mut dw_partial,
                &dy,
                &x,
                &weight,
                &rstd,
                self.rows as u32,
                self.n as u32,
                1.0 / self.n as f32,
                self.offset,
                self.dy.is_none() as u32,
            )
        }
        .map_err(cuda_error)
    }
}

/// The input gradient `dx` of [`rms_norm_forward`] for the output gradient
/// `dy`, and with a weight its partial gradients in `dw_partial` (float32,
/// `[blocks, n]` for any number of `blocks`), to be summed by
/// [`rms_norm_reduce_weight_grad`].
///
/// Without `dy`, the output gradient is read from `dx` and overwritten.
#[allow(clippy::too_many_arguments)]
fn rms_norm_backward(
    dx: Tensor,
    dw_partial: Option<Tensor>,
    dy: Option<Tensor>,
    x: Tensor,
    weight: Option<Tensor>,
    rstd: Tensor,
    offset: f64,
    casting_mode: tvm_ffi::String,
) -> Result<()> {
    let mode = self::casting_mode(&casting_mode)?;
    let (rows, n) = rows_and_cols(&x)?;
    check_like(&dx, &x, "dx", "x")?;
    if let Some(dy) = &dy {
        check_like(dy, &x, "dy", "x")?;
    }
    check_f32(&rstd, rows, "rstd")?;
    check_weight(weight.as_ref(), n)?;
    if n == 0 {
        return Ok(());
    }

    let blocks = match (&weight, &dw_partial) {
        (Some(_), Some(dw_partial)) => {
            let blocks = match dw_partial.shape() {
                &[blocks, cols] if cols == n && (1..=MAX_GRID_X).contains(&blocks) => blocks,
                shape => {
                    return Err(value_error(&format!(
                        "dw_partial must have shape [blocks, {n}] with blocks > 0, got {shape:?}"
                    )));
                }
            };
            check_f32(dw_partial, blocks * n, "dw_partial")?;
            blocks
        }
        (Some(_), None) => return Err(value_error("dw_partial is required with a weight")),
        // Without a weight, every row gets its own block.
        (None, _) if rows == 0 => return Ok(()),
        (None, _) => rows,
    };

    let backward = Backward {
        dx: &dx,
        dw_partial: dw_partial.as_ref(),
        dy: dy.as_ref(),
        x: &x,
        weight: weight.as_ref(),
        rstd: &rstd,
        blocks,
        rows,
        n,
        offset: offset as f32,
    };
    dispatch(backward, mode, &x, weight.as_ref())
}

/// `dw = dw_partial.sum(0)`, rounded to the dtype of `dw`.
fn rms_norm_reduce_weight_grad(dw: Tensor, dw_partial: Tensor) -> Result<()> {
    let n = match dw.shape() {
        &[n] => n,
        shape => {
            return Err(value_error(&format!(
                "dw must be one-dimensional, got shape {shape:?}"
            )));
        }
    };
    let blocks = match dw_partial.shape() {
        &[blocks, cols] if cols == n && blocks <= u32::MAX as i64 => blocks,
        shape => {
            return Err(value_error(&format!(
                "dw_partial must have shape [blocks, {n}], got {shape:?}"
            )));
        }
    };
    check_f32(&dw_partial, blocks * n, "dw_partial")?;
    if n == 0 {
        return Ok(());
    }

    let gpu = kernels_cuda::load(dw.device(), kernels::from_module)?;
    let config = LaunchConfig {
        grid_dim: ((n as u32).div_ceil(REDUCE_THREADS), 1, 1),
        block_dim: (REDUCE_THREADS, 1, 1),
        shared_mem_bytes: 0,
    };
    let dw_partial_view = gpu.view::<f32>(&dw_partial)?;
    // SAFETY: `dw_partial` has `blocks * n` elements, and the kernel writes
    // `dw` only through bounds-checked `get_mut`.
    macro_rules! reduce {
        ($w:ty) => {{
            let mut dw = gpu.view::<$w>(&dw)?;
            unsafe {
                gpu.module.reduce_weight_grad::<$w>(
                    &gpu.stream,
                    config,
                    &mut dw,
                    &dw_partial_view,
                    blocks as u32,
                )
            }
            .map_err(cuda_error)
        }};
    }
    if has_dtype::<f16>(&dw) {
        reduce!(f16)
    } else if has_dtype::<bf16>(&dw) {
        reduce!(bf16)
    } else if has_dtype::<f32>(&dw) {
        reduce!(f32)
    } else {
        Err(value_error(
            "unsupported dtype for dw, expected float16, bfloat16 or float32",
        ))
    }
}

tvm_ffi::tvm_ffi_dll_export_typed_func!(rms_norm_forward, rms_norm_forward);
tvm_ffi::tvm_ffi_dll_export_typed_func!(rms_norm_backward, rms_norm_backward);
tvm_ffi::tvm_ffi_dll_export_typed_func!(rms_norm_reduce_weight_grad, rms_norm_reduce_weight_grad);
