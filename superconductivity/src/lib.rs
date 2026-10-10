//! Checked lattice Bogoliubov-de Gennes models and observables.
//!
//! The crate assembles spinful onsite s-wave BdG Hamiltonians without tying
//! applications to a particular eigensolver. It validates the physical
//! symmetries of inputs and externally computed eigensystems, then provides
//! local spectral and anomalous-pairing observables.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod complex;
mod error;
mod matrix;
mod model;
mod observables;

pub use complex::Complex64;
pub use error::SuperconductivityError;
pub use matrix::{spectrum_particle_hole_residual, BdGMatrix};
pub use model::{orbital_index, Hopping, NormalHamiltonian, OnsiteSWaveModel, Spin};
pub use observables::BdGEigensystem;

/// Result type used by checked model and observable operations.
pub type Result<T> = core::result::Result<T, SuperconductivityError>;
