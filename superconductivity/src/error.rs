//! Error types for checked superconductivity models.

use core::fmt;

/// An error caused by an invalid model, matrix, eigenbasis, or observable query.
#[derive(Clone, Debug, PartialEq)]
pub enum SuperconductivityError {
    /// A lattice must contain at least one site.
    EmptyLattice,
    /// A matrix or vector has the wrong number of elements.
    DimensionMismatch {
        /// Description of the affected input.
        context: &'static str,
        /// Required element count.
        expected: usize,
        /// Supplied element count.
        actual: usize,
    },
    /// A scalar contains NaN or infinity.
    NonFinite {
        /// Description of the affected input.
        context: &'static str,
        /// Flat element index, or zero for scalar inputs.
        index: usize,
    },
    /// A matrix expected to be Hermitian exceeds the selected tolerance.
    NotHermitian {
        /// Largest absolute element residual.
        residual: f64,
        /// Selected tolerance.
        tolerance: f64,
    },
    /// A hopping references a site outside the lattice.
    SiteOutOfBounds {
        /// Supplied site index.
        site: usize,
        /// Total site count.
        site_count: usize,
    },
    /// A hopping joins a site to itself; onsite terms must be supplied separately.
    SelfHopping {
        /// Site index used by both ends.
        site: usize,
    },
    /// A tolerance or broadening parameter is not finite and strictly positive.
    InvalidPositiveParameter {
        /// Name of the invalid parameter.
        parameter: &'static str,
    },
    /// A tolerance is not finite and nonnegative.
    InvalidTolerance {
        /// Name of the invalid tolerance.
        parameter: &'static str,
    },
    /// Eigenvectors are not orthonormal within the selected tolerance.
    NonOrthonormalEigenvectors {
        /// Largest orthonormality residual.
        residual: f64,
        /// Selected tolerance.
        tolerance: f64,
    },
    /// A supplied eigenpair does not solve the BdG eigenproblem.
    InvalidEigenpair {
        /// Zero-based eigenpair index.
        eigenpair: usize,
        /// Largest component residual for that eigenpair.
        residual: f64,
        /// Selected tolerance.
        tolerance: f64,
    },
    /// An observable requires a nonnegative finite temperature.
    InvalidTemperature,
}

impl fmt::Display for SuperconductivityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyLattice => write!(formatter, "the lattice must contain at least one site"),
            Self::DimensionMismatch {
                context,
                expected,
                actual,
            } => write!(
                formatter,
                "{context} has {actual} elements, expected {expected}"
            ),
            Self::NonFinite { context, index } => {
                write!(
                    formatter,
                    "{context} contains a non-finite value at index {index}"
                )
            }
            Self::NotHermitian {
                residual,
                tolerance,
            } => write!(
                formatter,
                "normal Hamiltonian is not Hermitian: residual {residual:e} exceeds {tolerance:e}"
            ),
            Self::SiteOutOfBounds { site, site_count } => write!(
                formatter,
                "site {site} is outside a lattice containing {site_count} sites"
            ),
            Self::SelfHopping { site } => write!(
                formatter,
                "hopping at site {site} is diagonal; use its onsite energy instead"
            ),
            Self::InvalidPositiveParameter { parameter } => {
                write!(
                    formatter,
                    "{parameter} must be finite and strictly positive"
                )
            }
            Self::InvalidTolerance { parameter } => {
                write!(formatter, "{parameter} must be finite and nonnegative")
            }
            Self::NonOrthonormalEigenvectors {
                residual,
                tolerance,
            } => write!(
                formatter,
                "eigenvectors are not orthonormal: residual {residual:e} exceeds {tolerance:e}"
            ),
            Self::InvalidEigenpair {
                eigenpair,
                residual,
                tolerance,
            } => write!(
                formatter,
                "eigenpair {eigenpair} has residual {residual:e}, exceeding {tolerance:e}"
            ),
            Self::InvalidTemperature => {
                write!(formatter, "temperature must be finite and nonnegative")
            }
        }
    }
}

impl std::error::Error for SuperconductivityError {}
