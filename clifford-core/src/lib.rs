//! Precision-generic Clifford algebra primitives.
//!
//! `clifford-core` provides a runtime-signature `Cl(p,q)` engine and the
//! scalar abstraction shared by geometric simulation crates. It contains
//! no networking, orchestration, database, GPU, or domain-experiment code.
//!
//! ```
//! use clifford_core::CliffordAlgebra;
//!
//! let algebra = CliffordAlgebra::cl3();
//! let e1 = algebra.basis::<f64>(1);
//! let e2 = algebra.basis::<f64>(2);
//! let e12 = algebra.geometric_product(&e1, &e2);
//! assert_eq!(algebra.blade_bitmap(4), 0b011);
//! assert_eq!(e12[4], 1.0);
//! ```

#![forbid(unsafe_code)]

pub mod clifford_algebra;
pub mod scalar;

pub use clifford_algebra::{CliffordAlgebra, CliffordAlgebraError};
pub use scalar::CliffordScalar;
