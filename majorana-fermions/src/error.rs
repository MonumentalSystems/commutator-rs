use core::fmt;

/// Errors returned by checked Majorana and fermionic operations.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum FermionError {
    /// Monomial indices were not strictly increasing.
    NonCanonicalIndices {
        /// Position of the offending index.
        position: usize,
        /// Previous generator index.
        previous: usize,
        /// Offending generator index.
        current: usize,
    },
    /// A Majorana generator index was outside a declared Hamiltonian dimension.
    MajoranaIndexOutOfBounds {
        /// Supplied generator index.
        index: usize,
        /// Number of generators in the Hamiltonian.
        majorana_count: usize,
    },
    /// A quadratic term used the same generator twice.
    RepeatedBilinearIndex {
        /// Repeated generator index.
        index: usize,
    },
    /// Converting a fermionic mode to two Majorana indices overflowed `usize`.
    ModeIndexOverflow {
        /// Fermionic mode whose conversion overflowed.
        mode: usize,
    },
    /// A model requiring at least one site was given zero sites.
    ZeroSites,
    /// A real or complex coefficient was not finite.
    NonFiniteCoefficient,
    /// A tolerance was negative or non-finite.
    InvalidTolerance {
        /// Supplied tolerance.
        tolerance: f64,
    },
}

impl fmt::Display for FermionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonCanonicalIndices {
                position,
                previous,
                current,
            } => write!(
                formatter,
                "Majorana index {current} at position {position} does not follow {previous}"
            ),
            Self::MajoranaIndexOutOfBounds {
                index,
                majorana_count,
            } => write!(
                formatter,
                "Majorana index {index} is outside a {majorana_count}-generator Hamiltonian"
            ),
            Self::RepeatedBilinearIndex { index } => {
                write!(formatter, "quadratic term repeats Majorana index {index}")
            }
            Self::ModeIndexOverflow { mode } => {
                write!(formatter, "fermionic mode {mode} cannot be represented")
            }
            Self::ZeroSites => write!(formatter, "site count must be non-zero"),
            Self::NonFiniteCoefficient => write!(formatter, "coefficient must be finite"),
            Self::InvalidTolerance { tolerance } => {
                write!(formatter, "tolerance has invalid value {tolerance}")
            }
        }
    }
}

impl std::error::Error for FermionError {}
