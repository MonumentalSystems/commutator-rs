//! Dense reference algorithms for real-time nonequilibrium Green functions.
//!
//! The crate fixes the fermionic conventions
//! `G^<(t,t') = i <c†(t') c(t)>` and
//! `G^>(t,t') = -i <c(t) c†(t')>`.  It is intended as a checked reference
//! layer: two-time storage is quadratic in the number of time points and the
//! direct convolution and Dyson algorithms are cubic in that number.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod error;
mod grid;
mod keldysh;
mod matrix_ops;
mod observables;

pub use cluster_green::{Complex64, DenseMatrix};
pub use error::KeldyshError;
pub use grid::RealTimeGrid;
pub use keldysh::{
    causal_volterra_convolution, langreth_product, retarded_dyson, time_convolution,
    ConsistencyReport, KeldyshGreen, TwoTimeMatrix,
};
pub use observables::{bond_flow_into, density_matrix, one_body_expectation, particle_number};

/// Result type for checked nonequilibrium Green-function operations.
pub type Result<T> = core::result::Result<T, KeldyshError>;
