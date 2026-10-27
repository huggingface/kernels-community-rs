//! Generic device kernels.
//!
//! The kernels (except [`fatrelu_and_mul`]) are generic over the activation
//! that is used. Following vLLM kernels, each block handles one token (and
//! threads the dimensions).

use cuda_device::cuda_module;

#[cuda_module]
pub mod kernels {
    use cuda_device::{DisjointSlice, kernel, thread};

    use crate::activation::{Activation, fatrelu};
    use crate::element::{Element, mul};

    /// Gated activation: `out[t, i] = act(x) * y` with `x = input[t, i]` and
    /// `y = input[t, d + i]`, or `x * act(y)` when `ACT_FIRST` is false.
    #[kernel]
    pub fn act_and_mul<T: Element, A: Activation, const ACT_FIRST: bool>(
        mut out: DisjointSlice<T>,
        input: &[T],
        d: u32,
    ) {
        let d = d as usize;
        let token = thread::blockIdx_x() as usize;
        let mut idx = thread::threadIdx_x() as usize;
        while idx < d {
            // SAFETY: the host checks `input` has `2 * d` and `out` has `d`
            // elements per launched token.
            unsafe {
                let x = *input.get_unchecked(token * 2 * d + idx);
                let y = *input.get_unchecked(token * 2 * d + d + idx);
                *out.get_unchecked_mut(token * d + idx) = if ACT_FIRST {
                    mul(A::apply(x), y)
                } else {
                    mul(x, A::apply(y))
                };
            }
            idx += thread::blockDim_x() as usize;
        }
    }

    /// `out[t, i] = fatrelu(input[t, i], threshold) * input[t, d + i]`.
    #[kernel]
    pub fn fatrelu_and_mul<T: Element>(
        mut out: DisjointSlice<T>,
        input: &[T],
        d: u32,
        threshold: f32,
    ) {
        let d = d as usize;
        let token = thread::blockIdx_x() as usize;
        let mut idx = thread::threadIdx_x() as usize;
        while idx < d {
            // SAFETY: as in `act_and_mul`.
            unsafe {
                let x = *input.get_unchecked(token * 2 * d + idx);
                let y = *input.get_unchecked(token * 2 * d + d + idx);
                *out.get_unchecked_mut(token * d + idx) = mul(fatrelu(x, threshold), y);
            }
            idx += thread::blockDim_x() as usize;
        }
    }

    /// Element-wise activation: `out[t, i] = act(input[t, i])`.
    #[kernel]
    pub fn activation<T: Element, A: Activation>(mut out: DisjointSlice<T>, input: &[T], d: u32) {
        let d = d as usize;
        let token = thread::blockIdx_x() as usize;
        let mut idx = thread::threadIdx_x() as usize;
        while idx < d {
            // SAFETY: the host checks `input` and `out` have `d` elements per
            // launched token.
            unsafe {
                let x = *input.get_unchecked(token * d + idx);
                *out.get_unchecked_mut(token * d + idx) = A::apply(x);
            }
            idx += thread::blockDim_x() as usize;
        }
    }
}
