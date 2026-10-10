use core::fmt;

/// Invalid geometry, topology, or tessellation request.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum MeshError {
    /// A named scalar or vector contained NaN or infinity.
    NonFinite(&'static str),
    /// A named length or radius was not strictly positive.
    NonPositive(&'static str),
    /// A named direction or normal had zero length.
    ZeroVector(&'static str),
    /// Two otherwise valid dimensions did not satisfy a required relation.
    InvalidRelation(&'static str),
    /// Resolution was outside the supported inclusive range.
    ResolutionOutOfRange {
        /// Requested resolution.
        requested: u32,
        /// Smallest supported resolution.
        minimum: u32,
        /// Largest supported resolution.
        maximum: u32,
    },
    /// Icosphere subdivision count exceeded the supported maximum.
    SubdivisionOutOfRange {
        /// Requested subdivision count.
        requested: u32,
        /// Largest supported subdivision count.
        maximum: u32,
    },
    /// Mesh size arithmetic overflowed or exceeded the `u32` index domain.
    SizeOverflow,
    /// An index referenced a vertex outside its buffer.
    IndexOutOfRange,
    /// A generated or supplied triangle had zero geometric area.
    DegenerateTriangle,
    /// A supplied line segment had coincident endpoints.
    DegenerateSegment,
}

impl fmt::Display for MeshError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite(name) => write!(formatter, "{name} must contain only finite values"),
            Self::NonPositive(name) => write!(formatter, "{name} must be strictly positive"),
            Self::ZeroVector(name) => write!(formatter, "{name} must have nonzero length"),
            Self::InvalidRelation(requirement) => formatter.write_str(requirement),
            Self::ResolutionOutOfRange {
                requested,
                minimum,
                maximum,
            } => write!(
                formatter,
                "resolution {requested} is outside {minimum}..={maximum}"
            ),
            Self::SubdivisionOutOfRange { requested, maximum } => write!(
                formatter,
                "subdivision count {requested} exceeds maximum {maximum}"
            ),
            Self::SizeOverflow => formatter.write_str("mesh size exceeds supported limits"),
            Self::IndexOutOfRange => formatter.write_str("mesh index is out of range"),
            Self::DegenerateTriangle => formatter.write_str("mesh contains a degenerate triangle"),
            Self::DegenerateSegment => formatter.write_str("line segment has coincident endpoints"),
        }
    }
}

impl std::error::Error for MeshError {}
