use core::fmt;

use cluster_green::GreenError;

/// Validation and numerical failures in coherent transport calculations.
#[derive(Debug)]
#[non_exhaustive]
pub enum TransportError {
    /// A matrix operation from `cluster-green` failed.
    LinearAlgebra(GreenError),
    /// A matrix had invalid row and column dimensions.
    Dimensions {
        /// Name of the matrix.
        name: &'static str,
        /// Required `(rows, columns)`.
        expected: (usize, usize),
        /// Supplied `(rows, columns)`.
        actual: (usize, usize),
    },
    /// A required Hermitian matrix exceeded tolerance.
    NonHermitian(&'static str),
    /// A broadening matrix was not positive semidefinite.
    NonCausalSelfEnergy {
        /// Smallest LDL† pivot encountered.
        minimum_pivot: f64,
    },
    /// A scalar or slice contained NaN or infinity.
    NonFinite(&'static str),
    /// The retarded regulator was not strictly positive.
    NonPositiveRegulator,
    /// A tolerance was zero, negative, or non-finite.
    InvalidTolerance,
    /// The Caroli trace retained a significant imaginary component.
    ComplexTransmission {
        /// Imaginary residual.
        imaginary: f64,
    },
    /// Transmission was negative beyond numerical tolerance.
    NegativeTransmission {
        /// Computed real transmission.
        value: f64,
    },
    /// A supplied energy grid was too short.
    EnergyGridTooShort,
    /// Energy-grid values were not strictly increasing.
    UnorderedEnergyGrid {
        /// First index that did not increase.
        index: usize,
    },
    /// Parallel input slices had different lengths.
    LengthMismatch {
        /// Required length.
        expected: usize,
        /// Supplied length.
        actual: usize,
    },
    /// Temperature in energy units or degeneracy was negative.
    NegativeParameter(&'static str),
}

impl fmt::Display for TransportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LinearAlgebra(error) => write!(f, "matrix operation failed: {error}"),
            Self::Dimensions {
                name,
                expected,
                actual,
            } => write!(
                f,
                "{name} has shape {}x{}, but shape {}x{} is required",
                actual.0, actual.1, expected.0, expected.1
            ),
            Self::NonHermitian(name) => write!(f, "{name} must be Hermitian"),
            Self::NonCausalSelfEnergy { minimum_pivot } => write!(
                f,
                "lead self-energy is noncausal (broadening pivot {minimum_pivot})"
            ),
            Self::NonFinite(name) => write!(f, "{name} must contain only finite values"),
            Self::NonPositiveRegulator => {
                write!(f, "the retarded regulator eta must be finite and positive")
            }
            Self::InvalidTolerance => write!(f, "tolerance must be finite and positive"),
            Self::ComplexTransmission { imaginary } => {
                write!(f, "Caroli transmission has imaginary residual {imaginary}")
            }
            Self::NegativeTransmission { value } => {
                write!(f, "Caroli transmission is negative ({value})")
            }
            Self::EnergyGridTooShort => write!(f, "energy grid must contain at least two points"),
            Self::UnorderedEnergyGrid { index } => {
                write!(f, "energy grid is not strictly increasing at index {index}")
            }
            Self::LengthMismatch { expected, actual } => write!(
                f,
                "parallel input has length {actual}, but length {expected} is required"
            ),
            Self::NegativeParameter(name) => write!(f, "{name} must be nonnegative"),
        }
    }
}

impl std::error::Error for TransportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::LinearAlgebra(error) => Some(error),
            _ => None,
        }
    }
}

impl From<GreenError> for TransportError {
    fn from(value: GreenError) -> Self {
        Self::LinearAlgebra(value)
    }
}
