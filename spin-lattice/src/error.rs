//! Error type for checked model construction and evolution.

use core::fmt;

/// An invalid model, state, or integration request.
#[derive(Clone, Debug, PartialEq)]
pub enum SpinLatticeError {
    /// A model must contain at least one atom.
    EmptySystem,
    /// A state array has the wrong length.
    DimensionMismatch {
        /// State component that has the wrong length.
        context: &'static str,
        /// Required length.
        expected: usize,
        /// Supplied length.
        actual: usize,
    },
    /// An input contains NaN or infinity.
    NonFinite {
        /// Input containing the invalid value.
        context: &'static str,
        /// Atom or bond index of the value.
        index: usize,
    },
    /// An atom mass is not finite and strictly positive.
    InvalidMass {
        /// Atom index.
        atom: usize,
    },
    /// A bond endpoint or state update references an absent atom.
    AtomOutOfBounds {
        /// Supplied atom index.
        atom: usize,
        /// Number of atoms in the model or state.
        atom_count: usize,
    },
    /// A bond connects an atom to itself.
    SelfBond {
        /// Bond index.
        bond: usize,
        /// Repeated atom index.
        atom: usize,
    },
    /// A spring constant is negative or non-finite.
    InvalidSpringConstant {
        /// Bond index.
        bond: usize,
    },
    /// An equilibrium bond vector has zero length.
    DegenerateEquilibriumBond {
        /// Bond index.
        bond: usize,
    },
    /// Atomic displacement collapsed a current bond to zero length.
    DegenerateCurrentBond {
        /// Bond index.
        bond: usize,
    },
    /// A spin is not a finite unit vector within the crate tolerance.
    InvalidSpin {
        /// Atom index.
        atom: usize,
        /// Measured spin norm.
        norm: f64,
    },
    /// An integration time step is not finite and strictly positive.
    InvalidTimeStep,
    /// A requested integration rotation overflowed.
    RotationOverflow {
        /// Atom index whose rotation angle overflowed.
        atom: usize,
    },
}

impl fmt::Display for SpinLatticeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySystem => write!(formatter, "the model must contain at least one atom"),
            Self::DimensionMismatch {
                context,
                expected,
                actual,
            } => write!(
                formatter,
                "{context} has length {actual}, expected {expected}"
            ),
            Self::NonFinite { context, index } => {
                write!(formatter, "{context} is non-finite at index {index}")
            }
            Self::InvalidMass { atom } => {
                write!(
                    formatter,
                    "mass for atom {atom} must be finite and positive"
                )
            }
            Self::AtomOutOfBounds { atom, atom_count } => write!(
                formatter,
                "atom {atom} is outside a system containing {atom_count} atoms"
            ),
            Self::SelfBond { bond, atom } => {
                write!(formatter, "bond {bond} connects atom {atom} to itself")
            }
            Self::InvalidSpringConstant { bond } => write!(
                formatter,
                "spring constant for bond {bond} must be finite and nonnegative"
            ),
            Self::DegenerateEquilibriumBond { bond } => {
                write!(
                    formatter,
                    "equilibrium vector for bond {bond} has zero length"
                )
            }
            Self::DegenerateCurrentBond { bond } => {
                write!(formatter, "current vector for bond {bond} has zero length")
            }
            Self::InvalidSpin { atom, norm } => {
                write!(formatter, "spin {atom} has invalid norm {norm}")
            }
            Self::InvalidTimeStep => {
                write!(formatter, "time step must be finite and strictly positive")
            }
            Self::RotationOverflow { atom } => {
                write!(formatter, "spin rotation angle overflowed for atom {atom}")
            }
        }
    }
}

impl std::error::Error for SpinLatticeError {}
