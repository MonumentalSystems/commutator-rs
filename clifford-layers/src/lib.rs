//! Framework-neutral neural-network layers over Clifford multivectors.
//!
//! The crate provides geometric-product linear layers, spatial Clifford
//! convolutions, blade-aware normalization, and an optional spectral
//! convolution implementation. It depends only on [`clifford_core`] for
//! algebra conventions; it contains no model runtime, autodiff, networking,
//! GPU, or experiment orchestration code.
//!
//! The initial implementation was extracted from HarmonicRust, where these
//! layers were originally ported from the ideas in Microsoft's
//! `cliffordlayers` project. The Rust implementation and tests are MIT-licensed.
//!
//! ```
//! use clifford_core::CliffordAlgebra;
//! use clifford_layers::CliffordLinear;
//!
//! let mut layer = CliffordLinear::new(CliffordAlgebra::cl3(), 1, 1, false);
//! layer.weights.fill(0.0);
//! layer.weights[0] = 1.0;
//!
//! let input = [1.0, 0.0, 0.5, -0.5, 0.25, 0.0, 0.0, 0.0];
//! assert_eq!(layer.forward(&input), input);
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod conv;
#[cfg(feature = "fourier")]
pub mod fourier;
pub mod linear;
pub mod norm;

pub use conv::{
    build_clifford_kernel, build_rotation_kernel, deinterleave_blades, interleave_blades,
    vector_silu, CliffordConjugateLinear, CliffordConv1d, CliffordConv2d, VectorSiLUMode,
};
#[cfg(feature = "fourier")]
pub use fourier::{CliffordFourierConv1d, CliffordFourierConvConfig};
pub use linear::CliffordLinear;
pub use norm::{CliffordBatchNorm, CliffordGroupNorm};
