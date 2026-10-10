use core::fmt;

/// Errors returned by Lyapunov estimators and tangent integrators.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum LyapunovError {
    /// A dynamical system reported a zero-dimensional state.
    ZeroDimension,
    /// A slice did not match the system's declared dimension.
    DimensionMismatch {
        /// Operation whose input had the wrong size.
        context: &'static str,
        /// Required number of scalar values.
        expected: usize,
        /// Supplied number of scalar values.
        actual: usize,
    },
    /// The requested number of exponents exceeded the state dimension.
    InvalidExponentCount {
        /// Requested number of exponents.
        requested: usize,
        /// Dimension of the dynamical system.
        dimension: usize,
    },
    /// A configuration count was zero.
    ZeroCount {
        /// Name of the invalid configuration field.
        field: &'static str,
    },
    /// A floating-point parameter was not finite or was outside its required domain.
    InvalidParameter {
        /// Name of the invalid parameter.
        field: &'static str,
        /// Supplied value.
        value: f64,
    },
    /// A state, tangent, or model output contained a non-finite value.
    NonFiniteValue {
        /// Data in which the value was observed.
        context: &'static str,
        /// Scalar index of the first non-finite value.
        index: usize,
    },
    /// A tangent vector collapsed during orthonormalization.
    DegenerateTangent {
        /// Index of the collapsed tangent vector.
        vector: usize,
    },
    /// A user-supplied system rejected an evaluation.
    System {
        /// Static explanation supplied by the system implementation.
        message: &'static str,
    },
}

impl LyapunovError {
    /// Creates an error for a model-specific evaluation failure.
    #[must_use]
    pub const fn system(message: &'static str) -> Self {
        Self::System { message }
    }
}

impl fmt::Display for LyapunovError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimension => write!(formatter, "system dimension must be non-zero"),
            Self::DimensionMismatch {
                context,
                expected,
                actual,
            } => write!(
                formatter,
                "{context} has length {actual}, expected {expected}"
            ),
            Self::InvalidExponentCount {
                requested,
                dimension,
            } => write!(
                formatter,
                "requested {requested} exponents for a system of dimension {dimension}"
            ),
            Self::ZeroCount { field } => write!(formatter, "{field} must be non-zero"),
            Self::InvalidParameter { field, value } => {
                write!(formatter, "{field} has invalid value {value}")
            }
            Self::NonFiniteValue { context, index } => {
                write!(
                    formatter,
                    "{context} contains a non-finite value at index {index}"
                )
            }
            Self::DegenerateTangent { vector } => {
                write!(formatter, "tangent vector {vector} collapsed to zero")
            }
            Self::System { message } => write!(formatter, "system evaluation failed: {message}"),
        }
    }
}

impl std::error::Error for LyapunovError {}
