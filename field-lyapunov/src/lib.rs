//! Matrix-free Lyapunov diagnostics for high-dimensional dynamical systems.
//!
//! [`ContinuousSystem`] represents an ODE through its vector field and a
//! Jacobian-vector product (JVP); callers never need to form a dense Jacobian.
//! [`Rk4`] advances the state and any number of tangent vectors together.
//! Existing field solvers can instead implement [`TangentStepper`] directly.
//!
//! The spectrum estimator implements the Benettin algorithm with twice-applied
//! modified Gram-Schmidt reorthogonalization. It stores `top_k * dimension`
//! tangent values, making it suitable for leading-exponent calculations on
//! large lattice and PDE states when `top_k` is small.
//!
//! ```
//! use field_lyapunov::{estimate_spectrum, BenettinConfig, ContinuousSystem, Rk4};
//!
//! struct Diagonal {
//!     rates: Vec<f64>,
//! }
//!
//! impl ContinuousSystem for Diagonal {
//!     fn dimension(&self) -> usize { self.rates.len() }
//!
//!     fn vector_field(
//!         &self,
//!         _time: f64,
//!         state: &[f64],
//!         output: &mut [f64],
//!     ) -> field_lyapunov::Result<()> {
//!         for ((out, value), rate) in output.iter_mut().zip(state).zip(&self.rates) {
//!             *out = rate * value;
//!         }
//!         Ok(())
//!     }
//!
//!     fn jacobian_vector_product(
//!         &self,
//!         _time: f64,
//!         _state: &[f64],
//!         direction: &[f64],
//!         output: &mut [f64],
//!     ) -> field_lyapunov::Result<()> {
//!         for ((out, value), rate) in output.iter_mut().zip(direction).zip(&self.rates) {
//!             *out = rate * value;
//!         }
//!         Ok(())
//!     }
//! }
//!
//! let mut stepper = Rk4::new(Diagonal { rates: vec![0.4, -0.2] });
//! let config = BenettinConfig::new(2, 0.01, 10, 400)?.with_transient_windows(20);
//! let estimate = estimate_spectrum(&mut stepper, &[1.0, 1.0], &config)?;
//! assert!((estimate.exponents()[0] - 0.4).abs() < 0.02);
//! # Ok::<(), field_lyapunov::LyapunovError>(())
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod diagnostics;
mod error;
mod estimator;
mod system;

pub use diagnostics::{kaplan_yorke_dimension, kolmogorov_sinai_entropy};
pub use error::LyapunovError;
pub use estimator::{
    estimate_largest, estimate_spectrum, estimate_spectrum_at, BenettinConfig, FiniteTimeEstimate,
    SpectrumEstimate,
};
pub use system::{ContinuousSystem, Rk4, TangentStepper};

/// Result type used by fallible operations in this crate.
pub type Result<T> = core::result::Result<T, LyapunovError>;
