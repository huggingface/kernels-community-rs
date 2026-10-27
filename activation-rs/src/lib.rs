//! Activation kernels from vLLM, ported to Rust with cuda-oxide.
//!
//! - [`element`]: the f16/bf16/f32 element types and their f32 round trips.
//! - [`activation`]: the activation functions.
//! - [`device`]: the device kernels.
//! - [`ops`]: argument validation, launches and the tvm-ffi exports.

mod activation;
mod device;
mod element;
mod ops;
