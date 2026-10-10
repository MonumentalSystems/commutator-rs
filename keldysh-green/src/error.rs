use core::fmt;

/// Validation and numerical failures from Keldysh operations.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum KeldyshError {
    /// A required collection was empty or too short.
    Empty(&'static str),
    /// An input contained NaN or infinity.
    NonFinite(&'static str),
    /// Time points were not strictly increasing.
    UnorderedTimeGrid {
        /// Index of the first non-increasing time.
        index: usize,
    },
    /// A flat input had the wrong number of entries.
    Shape {
        /// Name of the invalid input.
        name: &'static str,
        /// Required entry count.
        expected: usize,
        /// Supplied entry count.
        actual: usize,
    },
    /// A matrix had the wrong dimensions.
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
        /// Name of the indexed object.
        name: &'static str,
        /// Invalid index.
        index: usize,
        /// Collection length.
        length: usize,
    },
    /// A dimension product overflowed `usize`.
    SizeOverflow,
    /// A tolerance was non-positive or non-finite.
    InvalidTolerance,
    /// Components did not use the same grid or orbital dimension.
    Incompatible(&'static str),
    /// Matrix inversion failed in a Dyson step.
    SingularDysonStep {
        /// Time index at which the solve failed.
        time_index: usize,
    },
    /// A one-body observable was not real within tolerance.
    NonRealObservable {
        /// Residual imaginary component.
        imaginary: f64,
    },
    /// An operation supplied by `cluster-green` failed.
    Matrix(String),
}

impl fmt::Display for KeldyshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty(name) => write!(f, "{name} must contain at least two entries"),
            Self::NonFinite(name) => write!(f, "{name} must contain only finite values"),
            Self::UnorderedTimeGrid { index } => {
                write!(f, "time at index {index} is not strictly increasing")
            }
            Self::Shape {
                name,
                expected,
                actual,
            } => {
                write!(f, "{name} has length {actual}, but {expected} is required")
            }
            Self::Dimensions {
                name,
                expected,
                actual,
            } => write!(
                f,
                "{name} has shape {}x{}, but {}x{} is required",
                actual.0, actual.1, expected.0, expected.1
            ),
            Self::IndexOutOfBounds {
                name,
                index,
                length,
            } => {
                write!(f, "{name} index {index} is outside length {length}")
            }
            Self::SizeOverflow => write!(f, "dimension product overflows usize"),
            Self::InvalidTolerance => write!(f, "tolerance must be finite and positive"),
            Self::Incompatible(name) => write!(f, "incompatible {name}"),
            Self::SingularDysonStep { time_index } => {
                write!(f, "singular retarded Dyson step at time index {time_index}")
            }
            Self::NonRealObservable { imaginary } => {
                write!(f, "observable has residual imaginary component {imaginary}")
            }
            Self::Matrix(message) => write!(f, "matrix operation failed: {message}"),
        }
    }
}

impl std::error::Error for KeldyshError {}

impl From<cluster_green::GreenError> for KeldyshError {
    fn from(value: cluster_green::GreenError) -> Self {
        Self::Matrix(value.to_string())
    }
}
