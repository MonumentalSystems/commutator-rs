use core::fmt;

/// Validation, resource-limit, and proof-verification failures.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum MerkleError {
    /// A required context identifier was empty.
    EmptyIdentifier(&'static str),
    /// A context identifier exceeded the protocol limit.
    IdentifierTooLong(&'static str),
    /// The tree exceeded the supported number of leaves.
    TooManyLeaves,
    /// One leaf exceeded the supported byte length.
    LeafTooLarge,
    /// A fixed-size chunk adapter was given a zero chunk size.
    InvalidChunkSize,
    /// Input bytes were not divisible into complete fixed-size leaves.
    NonDivisibleData,
    /// The requested leaf index is outside the committed tree.
    IndexOutOfRange,
    /// The proof claims a different position than the caller expected.
    UnexpectedProofIndex,
    /// An inclusion proof was requested from an empty tree.
    EmptyTree,
    /// The proof depth does not match the committed leaf count.
    InvalidProofDepth,
    /// A commitment or proof uses an unsupported protocol version.
    UnsupportedVersion(u8),
    /// The supplied experiment context does not match the commitment.
    ContextMismatch,
    /// The reconstructed commitment root does not match.
    RootMismatch,
    /// The requested challenge count exceeds the committed leaf count.
    ChallengeCountOutOfRange,
    /// Challenge openings are missing, duplicated, or unexpected.
    InvalidOpenings,
    /// Opened leaf bytes exceed the aggregate challenge budget.
    ChallengePayloadTooLarge,
    /// Checked size or index arithmetic overflowed.
    SizeOverflow,
}

impl fmt::Display for MerkleError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyIdentifier(name) => write!(formatter, "{name} must not be empty"),
            Self::IdentifierTooLong(name) => write!(formatter, "{name} exceeds the byte limit"),
            Self::TooManyLeaves => formatter.write_str("Merkle tree has too many leaves"),
            Self::LeafTooLarge => formatter.write_str("Merkle leaf exceeds the byte limit"),
            Self::InvalidChunkSize => formatter.write_str("Merkle chunk size must be nonzero"),
            Self::NonDivisibleData => {
                formatter.write_str("input bytes do not contain complete Merkle leaves")
            }
            Self::IndexOutOfRange => formatter.write_str("Merkle leaf index is out of range"),
            Self::UnexpectedProofIndex => {
                formatter.write_str("Merkle proof does not match the expected leaf index")
            }
            Self::EmptyTree => formatter.write_str("empty Merkle tree has no inclusion proofs"),
            Self::InvalidProofDepth => {
                formatter.write_str("Merkle proof depth does not match the commitment")
            }
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported experiment-merkle version {version}")
            }
            Self::ContextMismatch => {
                formatter.write_str("commitment does not match the experiment context")
            }
            Self::RootMismatch => formatter.write_str("Merkle opening does not match the root"),
            Self::ChallengeCountOutOfRange => {
                formatter.write_str("challenge count exceeds committed leaves")
            }
            Self::InvalidOpenings => formatter.write_str("challenge openings are invalid"),
            Self::ChallengePayloadTooLarge => {
                formatter.write_str("challenge opening payload exceeds the byte limit")
            }
            Self::SizeOverflow => formatter.write_str("Merkle size arithmetic overflowed"),
        }
    }
}

impl std::error::Error for MerkleError {}
