//! Host-side functions for launching activation kernels.

use cutile::DType;
use cutile::half::{bf16, f16};
use cutile::prelude::{IntoPartition, Reshape, TileKernel};
use kernels_cutile::{Gpu, cuda_error};
use tvm_ffi::{Error, Result, Tensor, VALUE_ERROR};

use crate::device::kernels;

/// Number of elements handled by one tile of the element-wise kernels.
const TILE: usize = 1024;

/// Bounds of the number of elements per tile of the gated kernels; smaller
/// rows use one tile of the next power of two.
const MIN_GATED_TILE: usize = 32;
const MAX_GATED_TILE: usize = 1024;

// DLPack type codes.
const DL_FLOAT: u8 = 2;
const DL_BFLOAT: u8 = 4;

fn value_error(message: &str) -> Error {
    Error::new(VALUE_ERROR, message, "")
}

fn has_dtype(tensor: &Tensor, code: u8, bits: u8) -> bool {
    let dtype = tensor.dtype();
    dtype.code == code && dtype.bits == bits && dtype.lanes == 1
}

/// Validate `input` and `out`.
///
/// Returns the number of tokens (rows) and the row length of `out`, or `None`
/// when there is nothing to compute.
fn check_args(out: &Tensor, input: &Tensor, gated: bool) -> Result<Option<(usize, usize)>> {
    let input_shape = input.shape();
    let Some((&last, batch)) = input_shape.split_last() else {
        return Err(value_error("input must have at least one dimension"));
    };

    if gated && last % 2 != 0 {
        return Err(value_error(&format!(
            "the last dimension of input must be even, got {last}"
        )));
    }

    let d = if gated { last / 2 } else { last };

    let out_shape = out.shape();
    if out_shape.split_last() != Some((&d, batch)) {
        let mut expected = batch.to_vec();
        expected.push(d);
        return Err(value_error(&format!(
            "out must have shape {expected:?}, got {out_shape:?}"
        )));
    }

    let (input_dtype, out_dtype) = (input.dtype(), out.dtype());
    if (input_dtype.code, input_dtype.bits, input_dtype.lanes)
        != (out_dtype.code, out_dtype.bits, out_dtype.lanes)
    {
        return Err(value_error("input and out must have the same dtype"));
    }

    let tokens: i64 = batch.iter().product();
    if tokens == 0 || d == 0 {
        return Ok(None);
    }

    Ok(Some((tokens as usize, d as usize)))
}

/// An element-wise activation, one of the activation constants of
/// [`kernels`].
#[derive(Clone, Copy)]
struct Elementwise {
    act: i32,
}

impl Elementwise {
    fn launch<T: DType>(self, gpu: &Gpu, out: &Tensor, input: &Tensor) -> Result<()> {
        let numel = input.numel();
        // SAFETY: the caller keeps `input` and `out` alive and orders accesses
        // to them through Torch's current stream, on which the kernel is
        // queued.
        unsafe {
            let input = gpu
                .tensor::<T>(input)?
                .reshape(&[numel])
                .map_err(cuda_error)?;
            let out = gpu
                .tensor::<T>(out)?
                .reshape(&[numel])
                .map_err(cuda_error)?;
            // `ACT` cannot be inferred, so all generics are given explicitly.
            let generics = vec![
                T::DTYPE.as_str().to_string(),
                TILE.to_string(),
                self.act.to_string(),
            ];
            gpu.launch(kernels::activation(out.partition([TILE]), input).generics(generics))?;
        }
        Ok(())
    }
}

/// The gated activations.
#[derive(Clone, Copy)]
enum Gated {
    /// `act(input[..., :d]) * input[..., d:]`, or `input[..., :d] *
    /// act(input[..., d:])` when `act_first` is false.
    ActAndMul { act: i32, act_first: bool },
    /// `fatrelu(input[..., :d], threshold) * input[..., d:]`.
    FatreluAndMul { threshold: f32 },
}

impl Gated {
    fn launch<T: DType>(
        self,
        gpu: &Gpu,
        out: &Tensor,
        input: &Tensor,
        tokens: usize,
        d: usize,
    ) -> Result<()> {
        let tile = d.next_power_of_two().clamp(MIN_GATED_TILE, MAX_GATED_TILE);
        // SAFETY: as for `Elementwise::launch`.
        unsafe {
            // The two halves of each row become `[token, 0, ..]` and
            // `[token, 1, ..]`.
            let input = gpu
                .tensor::<T>(input)?
                .reshape(&[tokens, 2, d])
                .map_err(cuda_error)?;
            let out = gpu
                .tensor::<T>(out)?
                .reshape(&[tokens, d])
                .map_err(cuda_error)?;
            let out = out.partition([1, tile]);
            match self {
                Self::ActAndMul { act, act_first } => {
                    let generics = vec![
                        T::DTYPE.as_str().to_string(),
                        tile.to_string(),
                        act.to_string(),
                        act_first.to_string(),
                    ];
                    gpu.launch(kernels::act_and_mul(out, input).generics(generics))?;
                }
                Self::FatreluAndMul { threshold } => {
                    gpu.launch(kernels::fatrelu_and_mul(out, input, threshold))?;
                }
            }
        }
        Ok(())
    }
}

/// Run `launch::<T>` for the element type of `input`.
macro_rules! dispatch_dtype {
    ($input:expr, $launch:ident) => {
        if has_dtype($input, DL_FLOAT, 16) {
            $launch!(f16)
        } else if has_dtype($input, DL_BFLOAT, 16) {
            $launch!(bf16)
        } else if has_dtype($input, DL_FLOAT, 32) {
            $launch!(f32)
        } else {
            Err(value_error(
                "unsupported dtype, expected float16, bfloat16 or float32",
            ))
        }
    };
}

fn elementwise(act: i32, out: Tensor, input: Tensor) -> Result<()> {
    if check_args(&out, &input, false)?.is_none() {
        return Ok(());
    }
    let op = Elementwise { act };
    let gpu = Gpu::new(input.device())?;
    macro_rules! launch {
        ($t:ty) => {
            op.launch::<$t>(&gpu, &out, &input)
        };
    }
    dispatch_dtype!(&input, launch)
}

fn gated(op: Gated, out: Tensor, input: Tensor) -> Result<()> {
    let Some((tokens, d)) = check_args(&out, &input, true)? else {
        return Ok(());
    };
    let gpu = Gpu::new(input.device())?;
    macro_rules! launch {
        ($t:ty) => {
            op.launch::<$t>(&gpu, &out, &input, tokens, d)
        };
    }
    dispatch_dtype!(&input, launch)
}

fn act_and_mul(act: i32, act_first: bool, out: Tensor, input: Tensor) -> Result<()> {
    gated(Gated::ActAndMul { act, act_first }, out, input)
}

/// SwiGLU: `silu(x[..., :d]) * x[..., d:]`.
fn silu_and_mul(out: Tensor, input: Tensor) -> Result<()> {
    act_and_mul(kernels::SILU, true, out, input)
}

/// `x[..., :d] * silu(x[..., d:])`.
fn mul_and_silu(out: Tensor, input: Tensor) -> Result<()> {
    act_and_mul(kernels::SILU, false, out, input)
}

/// GeGLU: `gelu(x[..., :d]) * x[..., d:]`.
fn gelu_and_mul(out: Tensor, input: Tensor) -> Result<()> {
    act_and_mul(kernels::GELU, true, out, input)
}

/// GeGLU with the tanh approximation.
fn gelu_tanh_and_mul(out: Tensor, input: Tensor) -> Result<()> {
    act_and_mul(kernels::GELU_TANH, true, out, input)
}

/// `fatrelu(x[..., :d], threshold) * x[..., d:]`.
fn fatrelu_and_mul(out: Tensor, input: Tensor, threshold: f64) -> Result<()> {
    let threshold = threshold as f32;
    gated(Gated::FatreluAndMul { threshold }, out, input)
}

/// GELU, `x * cdf(x)`, exact (PyTorch `approximate="none"`).
fn gelu(out: Tensor, input: Tensor) -> Result<()> {
    elementwise(kernels::GELU, out, input)
}

/// GELU with the tanh approximation (PyTorch `approximate="tanh"`).
fn gelu_tanh(out: Tensor, input: Tensor) -> Result<()> {
    elementwise(kernels::GELU_TANH, out, input)
}

/// The GELU approximation used by GPT-2.
fn gelu_new(out: Tensor, input: Tensor) -> Result<()> {
    elementwise(kernels::GELU_NEW, out, input)
}

/// A faster tanh-based GELU approximation.
fn gelu_fast(out: Tensor, input: Tensor) -> Result<()> {
    elementwise(kernels::GELU_FAST, out, input)
}

/// `x * sigmoid(1.702 * x)`.
fn gelu_quick(out: Tensor, input: Tensor) -> Result<()> {
    elementwise(kernels::GELU_QUICK, out, input)
}

/// SiLU, `x * sigmoid(x)`.
fn silu(out: Tensor, input: Tensor) -> Result<()> {
    elementwise(kernels::SILU, out, input)
}

tvm_ffi::tvm_ffi_dll_export_typed_func!(silu_and_mul, silu_and_mul);
tvm_ffi::tvm_ffi_dll_export_typed_func!(mul_and_silu, mul_and_silu);
tvm_ffi::tvm_ffi_dll_export_typed_func!(gelu_and_mul, gelu_and_mul);
tvm_ffi::tvm_ffi_dll_export_typed_func!(gelu_tanh_and_mul, gelu_tanh_and_mul);
tvm_ffi::tvm_ffi_dll_export_typed_func!(fatrelu_and_mul, fatrelu_and_mul);
tvm_ffi::tvm_ffi_dll_export_typed_func!(gelu, gelu);
tvm_ffi::tvm_ffi_dll_export_typed_func!(gelu_tanh, gelu_tanh);
tvm_ffi::tvm_ffi_dll_export_typed_func!(gelu_new, gelu_new);
tvm_ffi::tvm_ffi_dll_export_typed_func!(gelu_fast, gelu_fast);
tvm_ffi::tvm_ffi_dll_export_typed_func!(gelu_quick, gelu_quick);
tvm_ffi::tvm_ffi_dll_export_typed_func!(silu, silu);
