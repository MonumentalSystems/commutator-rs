//! Sparse Majorana and complex fermionic operator algebra.
//!
//! Majorana generators satisfy
//! `gamma_j gamma_k + gamma_k gamma_j = 2 delta_jk`. A
//! [`MajoranaMonomial`] stores only its strictly increasing generator indices,
//! so its size scales with its grade rather than with the dimension of the full
//! Clifford algebra. [`FermionOperator`] is a deterministic sparse linear
//! combination of those monomials with [`Complex64`] coefficients.
//!
//! ```
//! use majorana_fermions::{annihilation, creation, FermionOperator};
//!
//! let c = annihilation(0)?;
//! let c_dagger = creation(0)?;
//! let car = c.anticommutator(&c_dagger)?;
//! assert_eq!(car, FermionOperator::identity());
//! # Ok::<(), majorana_fermions::FermionError>(())
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod error;
mod monomial;
mod operator;
mod quadratic;

pub use error::FermionError;
pub use monomial::{MajoranaMonomial, SignedMonomial};
pub use num_complex::Complex64;
pub use operator::{
    annihilation, creation, fermion_parity, majorana, mode_parity, number_operator, FermionOperator,
};
pub use quadratic::{kitaev_chain, QuadraticMajoranaHamiltonian};

/// Result type used by checked algebra operations in this crate.
pub type Result<T> = core::result::Result<T, FermionError>;
