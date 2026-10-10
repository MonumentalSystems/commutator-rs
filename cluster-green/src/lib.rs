//! Checked small-matrix foundations for cluster perturbation theory.
//!
//! This crate consumes one-particle cluster Green functions. It deliberately
//! does not prescribe how an interacting cluster Green function is obtained.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod error;
mod green;
mod matrix;

pub use error::GreenError;
pub use green::{CausalityReport, ClusterGreenGrid, EmbeddedGreenGrid, GreenPoint};
pub use matrix::DenseMatrix;
pub use num_complex::Complex64;

/// Result type used by checked Green-function operations.
pub type Result<T> = core::result::Result<T, GreenError>;
