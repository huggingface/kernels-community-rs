//! Casting modes
//!
//! Models compute RMSNorm with different precisions, and the kernels follow
//! Liger's three modes, which match the transformers implementations:
//!
//! - `llama`: compute the inverse RMS in f32, round the normalized input
//!   to the input dtype, then multiply by the weight.
//! - `gemma`: compute everything in f32 and round only the output.
//! - `none`: compute in the input dtype.
//!
//! All modes accumulate the sum of squares in f32 and compute the backward pass
//! in f32 apart from the rounding of the weight gradient in `llama` mode.
//!
//! The kernels do all arithmetic in f32 and round to a lower precision where
//! the [`CastingMode`] flags say so.

/// Casting mode.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Llama,
    Gemma,
    None,
}

/// Casting mode as a type, so that every mode gets its own kernel and the
/// flags are compile-time constants.
///
/// Each flag says where the mode rounds to a lower precision.
pub trait CastingMode: 'static {
    /// Use the input dtype for math.
    const INPUT_DTYPE_MATH: bool;

    /// Round normalized input to the input dtype before applying weight.
    const ROUND_X_HAT: bool;

    /// Compute weight gradient in input dtype.
    const ROUND_WEIGHT_GRAD: bool;
}

pub struct Llama;
pub struct Gemma;
pub struct NoCasting;

impl CastingMode for Llama {
    const INPUT_DTYPE_MATH: bool = false;
    const ROUND_X_HAT: bool = true;
    const ROUND_WEIGHT_GRAD: bool = true;
}

impl CastingMode for Gemma {
    const INPUT_DTYPE_MATH: bool = false;
    const ROUND_X_HAT: bool = false;
    const ROUND_WEIGHT_GRAD: bool = false;
}

impl CastingMode for NoCasting {
    const INPUT_DTYPE_MATH: bool = true;
    const ROUND_X_HAT: bool = true;
    const ROUND_WEIGHT_GRAD: bool = false;
}
