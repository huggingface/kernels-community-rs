//! Tensor element types.
//!
//! The kernels are generic over `f32`, `half::f16` and `half::bf16`. This
//! module defines some traits for narrowing/widening these types from/to
//! f32.

use cuda_device::ptx_asm;
use half::{bf16, f16};

pub trait Element: Copy + Send + Sync + 'static {
    fn to_f32(self) -> f32;

    fn from_f32(value: f32) -> Self;
}

impl Element for f32 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }

    #[inline(always)]
    fn from_f32(value: f32) -> Self {
        value
    }
}

// cuda-oxide cannot lower the scalar `f16` primitive, so convert in PTX.
impl Element for f16 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        let value: f32;
        unsafe {
            ptx_asm!(
                "{ .reg .b16 h; cvt.u16.u32 h, %1; cvt.f32.f16 %0, h; }",
                out("=f") value,
                in("r") u32::from(self.to_bits()),
                options(register_only)
            );
        }
        value
    }

    #[inline(always)]
    fn from_f32(value: f32) -> Self {
        let bits: u32;
        unsafe {
            ptx_asm!(
                "{ .reg .b16 h; cvt.rn.f16.f32 h, %1; cvt.u32.u16 %0, h; }",
                out("=r") bits,
                in("f") value,
                options(register_only)
            );
        }
        f16::from_bits(bits as u16)
    }
}

impl Element for bf16 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        // Plain Rust (`f32::from_bits(bits << 16)`) lets llc sink the second
        // load of the gated kernels past the inline-asm conversions and the
        // division, which serializes the two loads (~20% slower bf16
        // `silu_and_mul` on L4).
        let value: f32;
        unsafe {
            ptx_asm!(
                "shl.b32 %0, %1, 16;",
                out("=f") value,
                in("r") u32::from(self.to_bits()),
                options(register_only)
            );
        }
        value
    }

    #[inline(always)]
    fn from_f32(value: f32) -> Self {
        bf16::from_bits(f32_to_bf16_bits(value))
    }
}

// The bf16 conversion instruction requires sm_80.
#[cfg(cuda_arch = "80")]
#[inline(always)]
fn f32_to_bf16_bits(value: f32) -> u16 {
    // `cvt.rn.bf16x2.f32`; the first operand goes in the low half.
    cuda_device::convert::cvt_bf16x2_f32(value, 0.0) as u16
}

#[cfg(not(cuda_arch = "80"))]
#[inline(always)]
fn f32_to_bf16_bits(value: f32) -> u16 {
    cuda_device::convert::f32_to_bf16_rne(value)
}

#[inline(always)]
pub fn add<T: Element>(a: T, b: T) -> T {
    T::from_f32(a.to_f32() + b.to_f32())
}

#[inline(always)]
pub fn mul<T: Element>(a: T, b: T) -> T {
    T::from_f32(a.to_f32() * b.to_f32())
}
