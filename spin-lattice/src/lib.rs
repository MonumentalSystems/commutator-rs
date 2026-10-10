//! Checked reference dynamics for coupled classical spins and lattices.
//!
//! The crate evaluates forces and effective spin fields as derivatives of one
//! explicit Hamiltonian, then provides velocity-Verlet lattice motion and
//! norm-preserving frozen-field spin precession.
//!
//! ```
//! use spin_lattice::{Bond, LinearExchange, SpinLatticeModel, SpinLatticeState, Vec3};
//!
//! let model = SpinLatticeModel::try_new(
//!     vec![1.0, 1.0],
//!     vec![Bond::new(
//!         0,
//!         1,
//!         Vec3::new(1.0, 0.0, 0.0),
//!         4.0,
//!         LinearExchange::new(1.0, -0.1),
//!     )],
//! )?;
//! let state = SpinLatticeState::stationary(
//!     &model,
//!     vec![Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, 1.0)],
//! )?;
//! assert!(model.evaluate(&state)?.energy().total().is_finite());
//! # Ok::<(), spin_lattice::SpinLatticeError>(())
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod error;
mod evaluate;
mod integrator;
mod model;
mod vector;

pub use error::SpinLatticeError;
pub use evaluate::{Energy, Evaluation};
pub use model::{Bond, LinearExchange, SpinLatticeModel, SpinLatticeState, SPIN_NORM_TOLERANCE};
pub use vector::Vec3;

/// Result type for checked model and integration operations.
pub type Result<T> = core::result::Result<T, SpinLatticeError>;
