//! Checked coherent NEGF and Landauer transport reference kernels.
//!
//! Matrix and complex primitives come from [`cluster_green`], allowing direct
//! composition with cluster Green-function and CPT calculations.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod error;
mod landauer;
mod negf;

pub use cluster_green::{Complex64, DenseMatrix};
pub use error::TransportError;
pub use landauer::{fermi_function, landauer_current_ev, landauer_integral, Reservoir};
pub use negf::{
    broadening, caroli_transmission, retarded_device_green, two_terminal_transmission,
    TransmissionPoint,
};

/// Result type for checked coherent-transport operations.
pub type Result<T> = core::result::Result<T, TransportError>;

/// Exact elementary charge in coulombs.
pub const ELEMENTARY_CHARGE_COULOMBS: f64 = 1.602_176_634e-19;

/// Exact Planck constant in joule-seconds.
pub const PLANCK_CONSTANT_JOULE_SECONDS: f64 = 6.626_070_15e-34;
