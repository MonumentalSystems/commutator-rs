#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod error;

pub mod cl6;
pub mod lattice;
pub mod metropolis;
pub mod rng;

pub use error::LatticeError;

/// Result type used by fallible lattice and rotor operations.
pub type Result<T> = core::result::Result<T, LatticeError>;
