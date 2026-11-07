//! Dense backward: dQ, dK/dV, the fused single-pass variant and the preprocessing reductions.

mod common;
mod dkv;
mod dq;
mod fused;
mod generic;
mod reduce;

pub(crate) use common::{
    copy_rows, probability_gradient, query_fragments, transposed_paired_words, transposed_word_mask,
};
pub(crate) use dkv::dkv_both_core;
pub(crate) use generic::{core as generic_core, grid_coords as generic_grid_coords};
pub(crate) use reduce::{ReduceSpec, bf16_8, bf16_16, bf16_32, f16_8, f16_16, f16_32};
