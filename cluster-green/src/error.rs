use core::fmt;

/// Validation and numerical failures returned by Green-function operations.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum GreenError {
    /// A matrix or frequency grid was empty.
    Empty(&'static str),
    /// A dimension product overflowed `usize`.
    SizeOverflow,
    /// A flat input had an invalid length.
    Shape {
        /// Name of the invalid input.
        name: &'static str,
        /// Required element count.
        expected: usize,
        /// Supplied element count.
        actual: usize,
    },
    /// Matrix row and column dimensions did not match an operation.
    Dimensions {
        /// Name of the invalid matrix.
        name: &'static str,
        /// Required `(rows, columns)`.
        expected: (usize, usize),
        /// Supplied `(rows, columns)`.
        actual: (usize, usize),
    },
    /// An index was outside its valid range.
    IndexOutOfBounds {
        /// Name of the indexed collection.
        name: &'static str,
        /// Invalid index.
        index: usize,
        /// Collection length.
        length: usize,
    },
    /// An input contained NaN or infinity.
    NonFinite(&'static str),
    /// Retarded broadening was not strictly positive.
    NonPositiveBroadening {
        /// Frequency-grid index.
        index: usize,
    },
    /// Real frequencies were not strictly increasing.
    UnorderedFrequencyGrid {
        /// Index of the first point that did not increase.
        index: usize,
    },
    /// A matrix expected to be Hermitian exceeded the requested tolerance.
    NonHermitian(&'static str),
    /// A Green function violated retarded causality.
    NonCausal {
        /// Frequency-grid index.
        index: usize,
        /// Most negative pivot found in the spectral matrix.
        minimum_pivot: f64,
    },
    /// A matrix was singular to working precision.
    SingularMatrix {
        /// Elimination column where the failure occurred.
        pivot: usize,
    },
    /// A tolerance was negative, zero, or non-finite.
    InvalidTolerance,
}

impl fmt::Display for GreenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty(name) => write!(f, "{name} must not be empty"),
            Self::SizeOverflow => write!(f, "matrix dimensions overflow usize"),
            Self::Shape {
                name,
                expected,
                actual,
            } => write!(
                f,
                "{name} has length {actual}, but length {expected} is required"
            ),
            Self::Dimensions {
                name,
                expected,
                actual,
            } => write!(
                f,
                "{name} has shape {}x{}, but shape {}x{} is required",
                actual.0, actual.1, expected.0, expected.1
            ),
            Self::IndexOutOfBounds {
                name,
                index,
                length,
            } => write!(f, "{name} index {index} is outside length {length}"),
            Self::NonFinite(name) => write!(f, "{name} must contain only finite values"),
            Self::NonPositiveBroadening { index } => {
                write!(f, "frequency point {index} has non-positive broadening")
            }
            Self::UnorderedFrequencyGrid { index } => write!(
                f,
                "real frequency at index {index} is not strictly increasing"
            ),
            Self::NonHermitian(name) => write!(f, "{name} must be Hermitian"),
            Self::NonCausal {
                index,
                minimum_pivot,
            } => write!(
                f,
                "Green function at index {index} is noncausal (spectral pivot {minimum_pivot})"
            ),
            Self::SingularMatrix { pivot } => {
                write!(f, "matrix is singular at elimination pivot {pivot}")
            }
            Self::InvalidTolerance => write!(f, "tolerance must be finite and positive"),
        }
    }
}

impl std::error::Error for GreenError {}
