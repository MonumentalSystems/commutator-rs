use core::fmt;

/// Validation and numerical failures returned by lattice kernels.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum LatticeError {
    /// A required lattice extent or collection was empty.
    Empty(&'static str),
    /// A flat buffer did not have the required number of elements.
    Shape {
        /// Name of the invalid input.
        name: &'static str,
        /// Required element count.
        expected: usize,
        /// Supplied element count.
        actual: usize,
    },
    /// A dimension product overflowed `usize`.
    SizeOverflow(&'static str),
    /// A coordinate or flat index was outside the lattice.
    IndexOutOfBounds,
    /// An input or computed quantity was not finite.
    NonFinite(&'static str),
    /// A value did not satisfy the documented mathematical domain.
    InvalidDomain(&'static str),
    /// A multivector that must be nonzero had zero or negligible norm.
    ZeroNorm(&'static str),
    /// A 64-component input contained an odd-grade coefficient.
    OddGradeComponent,
    /// An even multivector did not satisfy the checked Spin(6) invariant.
    NotUnitRotor,
}

impl fmt::Display for LatticeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty(name) => write!(formatter, "{name} must not be empty"),
            Self::Shape {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "{name} has length {actual}, but length {expected} is required"
            ),
            Self::SizeOverflow(name) => {
                write!(formatter, "dimensions for {name} overflow usize")
            }
            Self::IndexOutOfBounds => formatter.write_str("lattice index is out of bounds"),
            Self::NonFinite(name) => write!(formatter, "{name} must be finite"),
            Self::InvalidDomain(name) => write!(formatter, "{name} is outside its valid domain"),
            Self::ZeroNorm(name) => write!(formatter, "{name} must have nonzero norm"),
            Self::OddGradeComponent => {
                formatter.write_str("full multivector contains an odd-grade component")
            }
            Self::NotUnitRotor => {
                formatter.write_str("even multivector is not a checked Spin(6) rotor")
            }
        }
    }
}

impl std::error::Error for LatticeError {}

pub(crate) fn checked_product(
    name: &'static str,
    left: usize,
    right: usize,
) -> crate::Result<usize> {
    left.checked_mul(right)
        .ok_or(LatticeError::SizeOverflow(name))
}

pub(crate) fn finite_slice(name: &'static str, values: &[f64]) -> crate::Result<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(LatticeError::NonFinite(name))
    }
}

pub(crate) fn finite_scalar(name: &'static str, value: f64) -> crate::Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(LatticeError::NonFinite(name))
    }
}
