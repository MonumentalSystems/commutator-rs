use core::fmt;

/// Validation and numerical failures returned by magnetic-model operations.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum MagnetismError {
    /// The model contains no spin sites.
    EmptyModel,
    /// The Hilbert-space dimension cannot be represented by `usize`.
    HilbertSpaceOverflow {
        /// Number of spin-1/2 sites requested.
        spins: usize,
    },
    /// A site index lies outside the model.
    SiteOutOfBounds {
        /// Invalid site index.
        site: usize,
        /// Number of sites in the model.
        spins: usize,
    },
    /// A bond connects a site to itself.
    SelfBond {
        /// Site used as both bond endpoints.
        site: usize,
    },
    /// An input contains a NaN or infinity.
    NonFinite(&'static str),
    /// A slice length does not match the Hilbert-space dimension or site count.
    Shape {
        /// Name of the invalid input.
        name: &'static str,
        /// Required number of elements.
        expected: usize,
        /// Supplied number of elements.
        actual: usize,
    },
    /// A state has zero norm.
    ZeroNorm,
    /// A solver parameter lies outside its documented domain.
    InvalidSolverParameter(&'static str),
    /// A numerical iteration produced an invalid value.
    NumericalFailure(&'static str),
}

impl fmt::Display for MagnetismError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyModel => write!(f, "a spin model must contain at least one site"),
            Self::HilbertSpaceOverflow { spins } => {
                write!(f, "the 2^{spins} Hilbert-space dimension overflows usize")
            }
            Self::SiteOutOfBounds { site, spins } => {
                write!(f, "site {site} is outside a model with {spins} sites")
            }
            Self::SelfBond { site } => write!(f, "bond at site {site} is a self-bond"),
            Self::NonFinite(name) => write!(f, "{name} must contain only finite values"),
            Self::Shape {
                name,
                expected,
                actual,
            } => write!(
                f,
                "{name} has length {actual}, but length {expected} is required"
            ),
            Self::ZeroNorm => write!(f, "quantum state must have nonzero norm"),
            Self::InvalidSolverParameter(name) => {
                write!(f, "invalid Lanczos parameter: {name}")
            }
            Self::NumericalFailure(name) => write!(f, "numerical failure: {name}"),
        }
    }
}

impl std::error::Error for MagnetismError {}
