//! Host-side functions for launching activation kernels.

use std::marker::PhantomData;

use cuda_core::simt::LaunchConfig;
use half::{bf16, f16};
use kernels_cuda::{Gpu, cuda_error};
use tvm_ffi::{Error, Result, Tensor, VALUE_ERROR};

use crate::activation::{Activation, Gelu, GeluFast, GeluNew, GeluQuick, GeluTanh, Silu};
use crate::device::kernels::{self, LoadedModule};
use crate::element::Element;

/// Upper bound of the block size; smaller rows use one thread per element.
const MAX_BLOCK_THREADS: u32 = 1024;

/// The CUDA limit for `gridDim.x`.
const MAX_GRID_X: i64 = i32::MAX as i64;

fn value_error(message: &str) -> Error {
    Error::new(VALUE_ERROR, message, "")
}

/// Validate `input` and `out` and return the launch config and dimensionality.
///
/// Returns `None` when there is nothing to compute.
fn launch_config(out: &Tensor, input: &Tensor, gated: bool) -> Result<Option<(LaunchConfig, u32)>> {
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

    if tokens > MAX_GRID_X {
        return Err(value_error(&format!(
            "input has {tokens} rows, at most {MAX_GRID_X} are supported"
        )));
    }

    let d = u32::try_from(d)
        .map_err(|_| value_error(&format!("rows of {d} elements are not supported")))?;

    let config = LaunchConfig {
        grid_dim: (tokens as u32, 1, 1),
        block_dim: (d.min(MAX_BLOCK_THREADS), 1, 1),
        shared_mem_bytes: 0,
    };

    Ok(Some((config, d)))
}

/// Kernel launch for a given element type.
trait Launch {
    fn launch<T: Element + kernels_cuda::Element>(
        self,
        gpu: &Gpu<LoadedModule>,
        out: &Tensor,
        input: &Tensor,
        config: LaunchConfig,
        d: u32,
    ) -> Result<()>;
}

struct ActAndMul<A, const ACT_FIRST: bool>(PhantomData<A>);

impl<A: Activation, const ACT_FIRST: bool> Launch for ActAndMul<A, ACT_FIRST> {
    fn launch<T: Element + kernels_cuda::Element>(
        self,
        gpu: &Gpu<LoadedModule>,
        out: &Tensor,
        input: &Tensor,
        config: LaunchConfig,
        d: u32,
    ) -> Result<()> {
        let input = gpu.view::<T>(input)?;
        let mut out = gpu.view::<T>(out)?;
        // SAFETY: `launch_config` checked that `input` has `2 * d` and `out`
        // has `d` elements for each block of `config`.
        unsafe {
            gpu.module
                .act_and_mul::<T, A, ACT_FIRST>(&gpu.stream, config, &mut out, &input, d)
        }
        .map_err(cuda_error)
    }
}

struct FatreluAndMul {
    threshold: f32,
}

impl Launch for FatreluAndMul {
    fn launch<T: Element + kernels_cuda::Element>(
        self,
        gpu: &Gpu<LoadedModule>,
        out: &Tensor,
        input: &Tensor,
        config: LaunchConfig,
        d: u32,
    ) -> Result<()> {
        let input = gpu.view::<T>(input)?;
        let mut out = gpu.view::<T>(out)?;
        // SAFETY: as for `ActAndMul`.
        unsafe {
            gpu.module.fatrelu_and_mul::<T>(
                &gpu.stream,
                config,
                &mut out,
                &input,
                d,
                self.threshold,
            )
        }
        .map_err(cuda_error)
    }
}

struct Elementwise<A>(PhantomData<A>);

impl<A: Activation> Launch for Elementwise<A> {
    fn launch<T: Element + kernels_cuda::Element>(
        self,
        gpu: &Gpu<LoadedModule>,
        out: &Tensor,
        input: &Tensor,
        config: LaunchConfig,
        d: u32,
    ) -> Result<()> {
        let input = gpu.view::<T>(input)?;
        let mut out = gpu.view::<T>(out)?;
        // SAFETY: `launch_config` checked that `input` and `out` have `d`
        // elements for each block of `config`.
        unsafe {
            gpu.module
                .activation::<T, A>(&gpu.stream, config, &mut out, &input, d)
        }
        .map_err(cuda_error)
    }
}

fn has_dtype<T: kernels_cuda::Element>(tensor: &Tensor) -> bool {
    let dtype = tensor.dtype();
    dtype.code == T::CODE as u8 && dtype.bits == T::BITS && dtype.lanes == 1
}

fn run(launch: impl Launch, out: Tensor, input: Tensor, gated: bool) -> Result<()> {
    let Some((config, d)) = launch_config(&out, &input, gated)? else {
        return Ok(());
    };
    let gpu = kernels_cuda::load(input.device(), kernels::from_module)?;
    if has_dtype::<f16>(&input) {
        launch.launch::<f16>(&gpu, &out, &input, config, d)
    } else if has_dtype::<bf16>(&input) {
        launch.launch::<bf16>(&gpu, &out, &input, config, d)
    } else if has_dtype::<f32>(&input) {
        launch.launch::<f32>(&gpu, &out, &input, config, d)
    } else {
        Err(value_error(
            "unsupported dtype, expected float16, bfloat16 or float32",
        ))
    }
}

fn act_and_mul<A: Activation, const ACT_FIRST: bool>(out: Tensor, input: Tensor) -> Result<()> {
    run(ActAndMul::<A, ACT_FIRST>(PhantomData), out, input, true)
}

fn elementwise<A: Activation>(out: Tensor, input: Tensor) -> Result<()> {
    run(Elementwise::<A>(PhantomData), out, input, false)
}

/// SwiGLU: `silu(x[..., :d]) * x[..., d:]`.
fn silu_and_mul(out: Tensor, input: Tensor) -> Result<()> {
    act_and_mul::<Silu, true>(out, input)
}

/// `x[..., :d] * silu(x[..., d:])`.
fn mul_and_silu(out: Tensor, input: Tensor) -> Result<()> {
    act_and_mul::<Silu, false>(out, input)
}

/// GeGLU: `gelu(x[..., :d]) * x[..., d:]`.
fn gelu_and_mul(out: Tensor, input: Tensor) -> Result<()> {
    act_and_mul::<Gelu, true>(out, input)
}

/// GeGLU with the tanh approximation.
fn gelu_tanh_and_mul(out: Tensor, input: Tensor) -> Result<()> {
    act_and_mul::<GeluTanh, true>(out, input)
}

/// `fatrelu(x[..., :d], threshold) * x[..., d:]`.
fn fatrelu_and_mul(out: Tensor, input: Tensor, threshold: f64) -> Result<()> {
    let threshold = threshold as f32;
    run(FatreluAndMul { threshold }, out, input, true)
}

/// GELU, `x * cdf(x)`, exact (PyTorch `approximate="none"`).
fn gelu(out: Tensor, input: Tensor) -> Result<()> {
    elementwise::<Gelu>(out, input)
}

/// GELU with the tanh approximation (PyTorch `approximate="tanh"`).
fn gelu_tanh(out: Tensor, input: Tensor) -> Result<()> {
    elementwise::<GeluTanh>(out, input)
}

/// The GELU approximation used by GPT-2.
fn gelu_new(out: Tensor, input: Tensor) -> Result<()> {
    elementwise::<GeluNew>(out, input)
}

/// A faster tanh-based GELU approximation.
fn gelu_fast(out: Tensor, input: Tensor) -> Result<()> {
    elementwise::<GeluFast>(out, input)
}

/// `x * sigmoid(1.702 * x)`.
fn gelu_quick(out: Tensor, input: Tensor) -> Result<()> {
    elementwise::<GeluQuick>(out, input)
}

/// SiLU, `x * sigmoid(x)`.
fn silu(out: Tensor, input: Tensor) -> Result<()> {
    elementwise::<Silu>(out, input)
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
