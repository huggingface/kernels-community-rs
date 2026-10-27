//! Activation functions defined on a single element.

// Note: looping is done by the generic functions in the device module.

use core::f64::consts::{FRAC_1_SQRT_2, FRAC_2_SQRT_PI, SQRT_2};

use cuda_device::device;

use crate::element::{Element, add, mul};

// libdevice symbols for elementary functions.
#[device]
unsafe extern "C" {
    fn __nv_erff(x: f32) -> f32;
    fn __nv_expf(x: f32) -> f32;
    fn __nv_tanhf(x: f32) -> f32;
}

#[inline(always)]
fn erf(x: f32) -> f32 {
    unsafe { __nv_erff(x) }
}

#[inline(always)]
fn exp(x: f32) -> f32 {
    unsafe { __nv_expf(x) }
}

#[inline(always)]
fn tanh(x: f32) -> f32 {
    unsafe { __nv_tanhf(x) }
}

pub trait Activation: 'static {
    fn apply<T: Element>(x: T) -> T;
}

/// SiLU, `x * sigmoid(x)`.
pub struct Silu;

impl Activation for Silu {
    #[inline(always)]
    fn apply<T: Element>(x: T) -> T {
        let f = x.to_f32();
        T::from_f32(f / (1.0 + exp(-f)))
    }
}

/// GELU, `x * cdf(x)`, exact (PyTorch `approximate="none"`).
pub struct Gelu;

impl Activation for Gelu {
    #[inline(always)]
    fn apply<T: Element>(x: T) -> T {
        const ALPHA: f32 = FRAC_1_SQRT_2 as f32;
        let f = x.to_f32();
        T::from_f32(f * 0.5 * (1.0 + erf(f * ALPHA)))
    }
}

/// GELU with the tanh approximation (PyTorch `approximate="tanh"`).
pub struct GeluTanh;

impl Activation for GeluTanh {
    #[inline(always)]
    fn apply<T: Element>(x: T) -> T {
        // Like the C++ constants: computed in f64, then rounded to f32.
        const BETA: f32 = (SQRT_2 * FRAC_2_SQRT_PI * 0.5) as f32;
        const KAPPA: f32 = 0.044715;
        let f = x.to_f32();
        let x_cube = f * f * f;
        // nvcc contracts `f + KAPPA * x_cube` to an FMA. cuda-oxide does too by
        // default, but the exact results depend on it, so don't leave it to luck.
        let inner = BETA * KAPPA.mul_add(x_cube, f);
        T::from_f32(0.5 * f * (1.0 + tanh(inner)))
    }
}

/// The GELU approximation used by GPT-2.
pub struct GeluNew;

impl Activation for GeluNew {
    #[inline(always)]
    fn apply<T: Element>(x: T) -> T {
        let x3 = mul(mul(x, x), x).to_f32();
        let inner = add(x, T::from_f32(0.044715 * x3));
        let t = T::from_f32(tanh(T::from_f32(0.797_884_56 * inner.to_f32()).to_f32()));
        mul(mul(T::from_f32(0.5), x), add(T::from_f32(1.0), t))
    }
}

/// A faster tanh-based GELU approximation.
pub struct GeluFast;

impl Activation for GeluFast {
    #[inline(always)]
    fn apply<T: Element>(x: T) -> T {
        let f = x.to_f32();
        let scaled = T::from_f32(f * 0.797_884_56);
        let cubic = add(T::from_f32(1.0), mul(T::from_f32(0.044715 * f), x));
        let t = T::from_f32(tanh(mul(scaled, cubic).to_f32()));
        mul(mul(T::from_f32(0.5), x), add(T::from_f32(1.0), t))
    }
}

/// `x * sigmoid(1.702 * x)`.
pub struct GeluQuick;

impl Activation for GeluQuick {
    #[inline(always)]
    fn apply<T: Element>(x: T) -> T {
        let f = x.to_f32();
        T::from_f32(f / (1.0 + exp(-1.702 * f)))
    }
}

/// FATReLU: `x` above `threshold`, zero otherwise.
#[inline(always)]
pub fn fatrelu<T: Element>(x: T, threshold: f32) -> T {
    let f = x.to_f32();
    T::from_f32(if f > threshold { f } else { 0.0 })
}
