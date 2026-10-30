//! Tensor element types.
//!
//! This module defines some traits for narrowing/widening tensor elements
//! from/to `f32`.

use cuda_device::ptx_asm;
use half::{bf16, f16};

/// Elements per group.
pub const GROUP: usize = 8;

/// A group of elements, widened to f32.
pub type Group = [f32; GROUP];

pub trait Element: Copy + Send + Sync + 'static {
    fn to_f32(self) -> f32;

    fn from_f32(value: f32) -> Self;

    /// For 16-bit types only: rounds `lo` and `hi` and packs them into
    /// a `u32`, `lo` in the low half.
    fn pack_pair(lo: f32, hi: f32) -> u32;

    /// For 16-bit types only: the inverse of [`Element::pack_pair`].
    fn unpack_pair(word: u32) -> (f32, f32);

    /// Loads `GROUP` elements with 16-byte loads.
    ///
    /// # Safety
    ///
    /// `ptr` must be 16-byte aligned global memory with `GROUP` elements.
    unsafe fn load_group(ptr: *const Self) -> Group;

    /// Stores `GROUP` elements with 16-byte stores.
    ///
    /// # Safety
    ///
    /// `ptr` must be 16-byte aligned global memory with `GROUP` elements.
    unsafe fn store_group(ptr: *mut Self, values: Group);
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

    fn pack_pair(_lo: f32, _hi: f32) -> u32 {
        unreachable!("f32 is not a 16-bit type")
    }

    fn unpack_pair(_word: u32) -> (f32, f32) {
        unreachable!("f32 is not a 16-bit type")
    }

    #[inline(always)]
    unsafe fn load_group(ptr: *const Self) -> Group {
        let [a, b] = unsafe { [load_16_bytes(ptr.cast()), load_16_bytes(ptr.add(4).cast())] };
        core::array::from_fn(|i| f32::from_bits(if i < 4 { a[i] } else { b[i - 4] }))
    }

    #[inline(always)]
    unsafe fn store_group(ptr: *mut Self, values: Group) {
        let bits = values.map(f32::to_bits);
        unsafe {
            store_16_bytes(ptr.cast(), [bits[0], bits[1], bits[2], bits[3]]);
            store_16_bytes(ptr.add(4).cast(), [bits[4], bits[5], bits[6], bits[7]]);
        }
    }
}

/// Loads 8 16-bit elements with a 16-byte load: element `2 * i` is the low
/// half of word `i`.
#[inline(always)]
unsafe fn load_halves<T: Element>(ptr: *const T) -> Group {
    let words = unsafe { load_16_bytes(ptr.cast()) };
    let pairs = words.map(T::unpack_pair);
    core::array::from_fn(|i| {
        if i % 2 == 0 {
            pairs[i / 2].0
        } else {
            pairs[i / 2].1
        }
    })
}

#[inline(always)]
unsafe fn store_halves<T: Element>(ptr: *mut T, values: Group) {
    let words = core::array::from_fn(|i| T::pack_pair(values[2 * i], values[2 * i + 1]));
    unsafe { store_16_bytes(ptr.cast(), words) };
}

/// `ld.global.v4.u32`: one 16-byte load. (The read-only `ld.global.nc`
/// variant measured no faster, and is only valid for memory that the kernel
/// does not write.)
#[inline(always)]
unsafe fn load_16_bytes(ptr: *const u32) -> [u32; 4] {
    let (a, b, c, d): (u32, u32, u32, u32);
    unsafe {
        ptx_asm!(
            "ld.global.v4.u32 {%0, %1, %2, %3}, [%4];",
            out("=r") a,
            out("=r") b,
            out("=r") c,
            out("=r") d,
            in("l") ptr as u64,
        );
    }
    [a, b, c, d]
}

/// `st.global.v4.u32`: one 16-byte store.
#[inline(always)]
unsafe fn store_16_bytes(ptr: *mut u32, words: [u32; 4]) {
    unsafe {
        ptx_asm!(
            "st.global.v4.u32 [%0], {%1, %2, %3, %4};",
            in("l") ptr as u64,
            in("r") words[0],
            in("r") words[1],
            in("r") words[2],
            in("r") words[3],
            clobber("memory"),
        );
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

    #[cfg(cuda_arch = "80")]
    #[inline(always)]
    fn pack_pair(lo: f32, hi: f32) -> u32 {
        // `cvt.rn.f16x2.f32` converts and packs both in one instruction.
        cuda_device::convert::cvt_f16x2_f32(lo, hi)
    }

    #[cfg(not(cuda_arch = "80"))]
    #[inline(always)]
    fn pack_pair(lo: f32, hi: f32) -> u32 {
        // `<Self as Element>`: `half`'s inherent `from_f32` is host-only.
        let bits = |value| u32::from(<Self as Element>::from_f32(value).to_bits());
        bits(lo) | (bits(hi) << 16)
    }

    #[inline(always)]
    fn unpack_pair(word: u32) -> (f32, f32) {
        let (lo, hi): (f32, f32);
        unsafe {
            ptx_asm!(
                "{ .reg .b16 l, h; mov.b32 {l, h}, %2; cvt.f32.f16 %0, l; cvt.f32.f16 %1, h; }",
                out("=f") lo,
                out("=f") hi,
                in("r") word,
                options(register_only)
            );
        }
        (lo, hi)
    }

    #[inline(always)]
    unsafe fn load_group(ptr: *const Self) -> Group {
        unsafe { load_halves(ptr) }
    }

    #[inline(always)]
    unsafe fn store_group(ptr: *mut Self, values: Group) {
        unsafe { store_halves(ptr, values) }
    }
}

impl Element for bf16 {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        // Plain Rust (`f32::from_bits(bits << 16)`) lets llc sink loads past
        // the inline-asm conversions, which can serialize independent loads.
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

    #[cfg(cuda_arch = "80")]
    #[inline(always)]
    fn pack_pair(lo: f32, hi: f32) -> u32 {
        // `cvt.rn.bf16x2.f32` converts and packs both in one instruction.
        cuda_device::convert::cvt_bf16x2_f32(lo, hi)
    }

    #[cfg(not(cuda_arch = "80"))]
    #[inline(always)]
    fn pack_pair(lo: f32, hi: f32) -> u32 {
        u32::from(f32_to_bf16_bits(lo)) | (u32::from(f32_to_bf16_bits(hi)) << 16)
    }

    #[inline(always)]
    fn unpack_pair(word: u32) -> (f32, f32) {
        // A bf16 is the high half of an f32.
        let (lo, hi): (f32, f32);
        unsafe {
            ptx_asm!(
                "shl.b32 %0, %2, 16; and.b32 %1, %2, 0xffff0000;",
                out("=f") lo,
                out("=f") hi,
                in("r") word,
                options(register_only)
            );
        }
        (lo, hi)
    }

    #[inline(always)]
    unsafe fn load_group(ptr: *const Self) -> Group {
        unsafe { load_halves(ptr) }
    }

    #[inline(always)]
    unsafe fn store_group(ptr: *mut Self, values: Group) {
        unsafe { store_halves(ptr, values) }
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

/// Rounds `value` to the precision of `T` (to nearest even), keeping it as f32.
#[inline(always)]
pub fn round<T: Element>(value: f32) -> f32 {
    T::from_f32(value).to_f32()
}

/// `value` rounded to `T` if `cond`, else unchanged.
#[inline(always)]
pub fn round_if<T: Element>(cond: bool, value: f32) -> f32 {
    if cond { round::<T>(value) } else { value }
}
