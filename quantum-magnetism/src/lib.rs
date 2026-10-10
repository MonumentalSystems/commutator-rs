//! Exact reference tools for finite frustrated spin-1/2 magnets.
//!
//! The crate uses a matrix-free computational-basis Hamiltonian and has no
//! external dependencies. See [`SpinModel`] for the sign and basis
//! conventions.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod complex;
mod error;
mod lanczos;
mod model;
mod observables;

pub use complex::Complex64;
pub use error::MagnetismError;
pub use lanczos::{GroundState, LanczosConfig};
pub use model::{Bond, SpinModel, SpinModelBuilder};
pub use observables::SpinAxis;

/// Result type used by checked magnetic-model operations.
pub type Result<T> = core::result::Result<T, MagnetismError>;
