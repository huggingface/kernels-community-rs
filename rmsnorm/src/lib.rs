//! RMSNorm, forward and backward, ported from Liger's Triton kernels to Rust
//! with cuda-oxide.
//!
//! - [`element`]: the f16/bf16/f32 element types and their f32 round trips.
//! - [`casting`]: the casting modes (`llama`, `gemma`, `none`).
//! - [`device`]: the device kernels.
//! - [`ops`]: argument validation, launches and the tvm-ffi exports.

mod casting;
mod device;
mod element;
mod ops;
