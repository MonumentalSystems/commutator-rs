//! Gauge-covariant time-dependent Ginzburg--Landau and Josephson dynamics.
//!
//! The crate evolves a complex order parameter on a checked finite graph. It
//! deliberately complements [`superconductivity`](https://docs.rs/superconductivity)
//! instead of assembling another Bogoliubov--de Gennes Hamiltonian. Its
//! [`TdglModel::scaled_pairing_gaps`] output passes directly to that crate's
//! onsite s-wave model.
//!
//! # Convention
//!
//! An oriented link `i -> j` stores the dimensionless Peierls phase
//! `A_ij = q_pair integral(A . dl) / hbar`. Its covariant difference is
//! `psi_j - exp(i A_ij) psi_i`. Under `psi_i -> exp(i chi_i) psi_i`, the link
//! transforms as `A_ij -> A_ij + chi_j - chi_i`. Energies use one consistent
//! arbitrary unit; time uses the corresponding TDGL unit, and site mobility
//! has units `1 / (energy * time)`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod dynamics;
mod error;
mod junction;
mod model;

pub use dynamics::{StepReport, TdglIntegrator};
pub use error::DynamicsError;
pub use junction::JosephsonJunction;
pub use model::{GaugeLink, LinkDirection, LoopEdge, OrderParameter, SiteParameters, TdglModel};
pub use num_complex::Complex64;

/// Result type for checked model and evolution operations.
pub type Result<T> = core::result::Result<T, DynamicsError>;
