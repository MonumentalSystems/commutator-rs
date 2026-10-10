#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod error;
mod serde_bounded;

pub mod merkle;
pub mod spot_check;

pub use error::MerkleError;

/// Result type used by commitment and spot-check operations.
pub type Result<T> = core::result::Result<T, MerkleError>;
