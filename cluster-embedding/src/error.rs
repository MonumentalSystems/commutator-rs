use core::fmt;

/// Validation, solver, and numerical errors from embedding operations.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum EmbeddingError {
    /// An input collection was empty.
    Empty(&'static str),
    /// An input contained NaN or infinity.
    NonFinite(&'static str),
    /// A scalar was outside its allowed interval.
    InvalidValue(&'static str),
    /// A flat collection had an unexpected length.
    Length {
        /// Input name.
        name: &'static str,
        /// Required length.
        expected: usize,
        /// Supplied length.
        actual: usize,
    },
    /// A matrix had unexpected dimensions.
    Dimensions {
        /// Matrix name.
        name: &'static str,
        /// Required dimensions.
        expected: (usize, usize),
        /// Supplied dimensions.
        actual: (usize, usize),
    },
    /// A required matrix was not Hermitian.
    NonHermitian(&'static str),
    /// A retarded quantity had a positive anti-Hermitian component.
    NonCausal {
        /// Quantity being checked.
        name: &'static str,
        /// Frequency index.
        index: usize,
        /// Largest positive Rayleigh-test violation.
        violation: f64,
    },
    /// A frequency grid was not strictly ordered or did not use positive broadening.
    InvalidFrequencyGrid,
    /// A matrix was singular.
    SingularMatrix,
    /// A named external solver failed.
    Solver(String),
    /// A lower-level `cluster-green` operation failed.
    Green(cluster_green::GreenError),
}

impl fmt::Display for EmbeddingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty(name) => write!(f, "{name} must not be empty"),
            Self::NonFinite(name) => write!(f, "{name} must be finite"),
            Self::InvalidValue(name) => write!(f, "{name} is outside its allowed range"),
            Self::Length {
                name,
                expected,
                actual,
            } => {
                write!(f, "{name} has length {actual}; expected {expected}")
            }
            Self::Dimensions {
                name,
                expected,
                actual,
            } => write!(
                f,
                "{name} has shape {}x{}; expected {}x{}",
                actual.0, actual.1, expected.0, expected.1
            ),
            Self::NonHermitian(name) => write!(f, "{name} must be Hermitian"),
            Self::NonCausal {
                name,
                index,
                violation,
            } => write!(
                f,
                "{name} is noncausal at frequency {index} (violation {violation})"
            ),
            Self::InvalidFrequencyGrid => write!(
                f,
                "frequencies must be finite, strictly ordered, and have positive broadening"
            ),
            Self::SingularMatrix => write!(f, "matrix is singular to working precision"),
            Self::Solver(message) => write!(f, "external solver failed: {message}"),
            Self::Green(error) => write!(f, "cluster Green-function error: {error}"),
        }
    }
}

impl std::error::Error for EmbeddingError {}

impl From<cluster_green::GreenError> for EmbeddingError {
    fn from(value: cluster_green::GreenError) -> Self {
        Self::Green(value)
    }
}

/// Result type for checked embedding operations.
pub type Result<T> = core::result::Result<T, EmbeddingError>;
