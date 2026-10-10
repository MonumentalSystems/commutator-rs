//! Safe, dependency-free kernels for harmonic and spherical dynamics.
//!
//! The crate exposes typed quaternion algebra, pure unit-sphere geometry,
//! Lohe and Kuramoto synchronization, and recurrent harmonic sequence
//! transforms. All public slice-based functions validate their dimensions
//! before indexing.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod error;

pub mod kuramoto;
pub mod lohe;
pub mod quaternion;
pub mod sequence;
pub mod sphere;

pub use error::DynamicsError;

/// Result type used by fallible numerical kernels in this crate.
pub type Result<T> = core::result::Result<T, DynamicsError>;
