//! Activation kernels from vLLM, ported to Rust with cuTile Rust.
//!
//! - [`device`]: the device kernels.
//! - [`ops`]: argument validation, launches and the tvm-ffi exports.

mod device;
mod ops;
