//! Error types for checked superconducting dynamics.

use core::fmt;

/// An invalid model, state, gauge operation, or integration request.
#[derive(Clone, Debug, PartialEq)]
pub enum DynamicsError {
    /// A graph must contain at least one site.
    EmptyGraph,
    /// Rectangular lattice dimensions overflow the addressable site count.
    SizeOverflow,
    /// An input has the wrong number of elements.
    DimensionMismatch {
        /// Input being checked.
        context: &'static str,
        /// Required length.
        expected: usize,
        /// Supplied length.
        actual: usize,
    },
    /// A scalar or complex component is not finite.
    NonFinite {
        /// Input being checked.
        context: &'static str,
        /// Flat element index, or zero for a scalar.
        index: usize,
    },
    /// A named scalar must be finite and strictly positive.
    InvalidPositiveParameter {
        /// Parameter name.
        parameter: &'static str,
    },
    /// A named scalar must be finite and nonnegative.
    InvalidNonnegativeParameter {
        /// Parameter name.
        parameter: &'static str,
    },
    /// A link endpoint lies outside the graph.
    SiteOutOfBounds {
        /// Invalid site index.
        site: usize,
        /// Number of graph sites.
        site_count: usize,
    },
    /// A graph link joins a site to itself.
    SelfLink {
        /// Repeated endpoint.
        site: usize,
    },
    /// A loop references a nonexistent link.
    LinkOutOfBounds {
        /// Invalid link index.
        link: usize,
        /// Number of graph links.
        link_count: usize,
    },
    /// An oriented edge sequence is not a continuous closed loop.
    OpenLoop {
        /// Position at which continuity or closure fails.
        position: usize,
    },
    /// Backtracking did not find a finite, non-increasing energy step.
    NoDescentStep {
        /// Smallest time step attempted.
        attempted_dt: f64,
        /// Number of rejected trials.
        backtracks: usize,
    },
}

impl fmt::Display for DynamicsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyGraph => write!(formatter, "the graph must contain at least one site"),
            Self::SizeOverflow => write!(formatter, "lattice dimensions overflow usize"),
            Self::DimensionMismatch {
                context,
                expected,
                actual,
            } => write!(
                formatter,
                "{context} has {actual} elements, expected {expected}"
            ),
            Self::NonFinite { context, index } => {
                write!(formatter, "{context} is non-finite at index {index}")
            }
            Self::InvalidPositiveParameter { parameter } => {
                write!(formatter, "{parameter} must be finite and strictly positive")
            }
            Self::InvalidNonnegativeParameter { parameter } => {
                write!(formatter, "{parameter} must be finite and nonnegative")
            }
            Self::SiteOutOfBounds { site, site_count } => {
                write!(formatter, "site {site} is outside a graph of {site_count} sites")
            }
            Self::SelfLink { site } => write!(formatter, "link at site {site} is a self-link"),
            Self::LinkOutOfBounds { link, link_count } => {
                write!(formatter, "link {link} is outside a graph of {link_count} links")
            }
            Self::OpenLoop { position } => {
                write!(formatter, "oriented edge sequence is open at position {position}")
            }
            Self::NoDescentStep {
                attempted_dt,
                backtracks,
            } => write!(
                formatter,
                "no energy-descent step found after {backtracks} backtracks (last dt {attempted_dt:e})"
            ),
        }
    }
}

impl std::error::Error for DynamicsError {}
