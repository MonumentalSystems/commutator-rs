use core::fmt;

/// Validation and numerical-domain failures returned by dynamics kernels.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DynamicsError {
    /// A required dimension or collection was empty.
    Empty(&'static str),
    /// A flat buffer did not have the required length.
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
    /// An input or computed value was not finite.
    NonFinite(&'static str),
    /// A value did not satisfy the documented mathematical domain.
    InvalidDomain(&'static str),
    /// A direction or state that must be nonzero had zero norm.
    ZeroNorm(&'static str),
}

impl fmt::Display for DynamicsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty(name) => write!(f, "{name} must not be empty"),
            Self::Shape {
                name,
                expected,
                actual,
            } => write!(
                f,
                "{name} has length {actual}, but length {expected} is required"
            ),
            Self::SizeOverflow(name) => write!(f, "dimensions for {name} overflow usize"),
            Self::NonFinite(name) => write!(f, "{name} must contain only finite values"),
            Self::InvalidDomain(name) => write!(f, "{name} is outside its valid domain"),
            Self::ZeroNorm(name) => write!(f, "{name} must have nonzero norm"),
        }
    }
}

impl std::error::Error for DynamicsError {}

pub(crate) fn checked_product(name: &'static str, factors: &[usize]) -> crate::Result<usize> {
    factors.iter().try_fold(1usize, |value, factor| {
        value
            .checked_mul(*factor)
            .ok_or(DynamicsError::SizeOverflow(name))
    })
}

pub(crate) fn expect_len(name: &'static str, actual: usize, expected: usize) -> crate::Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(DynamicsError::Shape {
            name,
            expected,
            actual,
        })
    }
}

pub(crate) fn finite_slice(name: &'static str, values: &[f32]) -> crate::Result<()> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(DynamicsError::NonFinite(name))
    }
}

pub(crate) fn finite_scalar(name: &'static str, value: f32) -> crate::Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(DynamicsError::NonFinite(name))
    }
}
