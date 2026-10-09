//! Versioned, context-bound SHA-256 Merkle commitments.

use core::fmt;

use experiment_core::WorkUnit;
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest as _, Sha256};

use crate::{MerkleError, Result};

/// Wire and hashing specification version implemented by this crate.
pub const PROTOCOL_VERSION: u8 = 1;
/// Maximum leaves accepted by one tree.
pub const MAX_LEAVES: u64 = 1 << 20;
/// Maximum bytes accepted in one leaf.
pub const MAX_LEAF_BYTES: usize = 16 * 1024 * 1024;
/// Maximum UTF-8 byte length of one context identifier.
pub const MAX_CONTEXT_FIELD_BYTES: usize = 4 * 1024;
/// SHA-256 digest used throughout the protocol.
pub type Digest = [u8; 32];

const CONTEXT_DOMAIN: &[u8] = b"experiment-merkle:v1:context\0";
const LEAF_DOMAIN: &[u8] = b"experiment-merkle:v1:leaf\0";
const PADDING_DOMAIN: &[u8] = b"experiment-merkle:v1:padding\0";
const NODE_DOMAIN: &[u8] = b"experiment-merkle:v1:node\0";
const EMPTY_DOMAIN: &[u8] = b"experiment-merkle:v1:empty\0";
const ROOT_DOMAIN: &[u8] = b"experiment-merkle:v1:root\0";

/// Experiment/run identity cryptographically bound into a commitment.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct CommitmentContext {
    experiment_id: String,
    instance_id: String,
    unit_id: String,
    generation: u64,
    schema_version: u32,
}

#[derive(Deserialize)]
struct CommitmentContextWire {
    #[serde(deserialize_with = "crate::serde_bounded::context_string")]
    experiment_id: String,
    #[serde(deserialize_with = "crate::serde_bounded::context_string")]
    instance_id: String,
    #[serde(deserialize_with = "crate::serde_bounded::context_string")]
    unit_id: String,
    generation: u64,
    schema_version: u32,
}

impl<'de> Deserialize<'de> for CommitmentContext {
    fn deserialize<D>(deserializer: D) -> core::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = CommitmentContextWire::deserialize(deserializer)?;
        Self::new(
            wire.experiment_id,
            wire.instance_id,
            wire.unit_id,
            wire.generation,
            wire.schema_version,
        )
        .map_err(D::Error::custom)
    }
}

impl CommitmentContext {
    /// Constructs and validates a commitment context.
    pub fn new(
        experiment_id: impl Into<String>,
        instance_id: impl Into<String>,
        unit_id: impl Into<String>,
        generation: u64,
        schema_version: u32,
    ) -> Result<Self> {
        let context = Self {
            experiment_id: experiment_id.into(),
            instance_id: instance_id.into(),
            unit_id: unit_id.into(),
            generation,
            schema_version,
        };
        context.validate()?;
        Ok(context)
    }

    /// Derives context from a work unit's IDs, generation, and schema version.
    ///
    /// This deliberately does not bind the work-unit payload, seed, backend,
    /// precision, implementation version, or attributes. A host that needs
    /// those values committed must encode them into leaves or use an
    /// application-level commitment alongside this context.
    pub fn from_work_unit<T>(unit: &WorkUnit<T>) -> Result<Self> {
        Self::new(
            unit.experiment_id.clone(),
            unit.instance_id.clone(),
            unit.unit_id.clone(),
            unit.generation,
            unit.run.schema_version,
        )
    }

    /// Stable experiment-definition identifier.
    pub fn experiment_id(&self) -> &str {
        &self.experiment_id
    }

    /// Stable experiment-run identifier.
    pub fn instance_id(&self) -> &str {
        &self.instance_id
    }

    /// Work-unit identifier within the run.
    pub fn unit_id(&self) -> &str {
        &self.unit_id
    }

    /// Run-state revision that issued the work.
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Version of the experiment payload schema.
    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Returns the canonical domain-separated context digest.
    pub fn digest(&self) -> Result<Digest> {
        self.validate()?;
        let mut hasher = Sha256::new();
        hasher.update(CONTEXT_DOMAIN);
        update_length_prefixed(&mut hasher, self.experiment_id.as_bytes())?;
        update_length_prefixed(&mut hasher, self.instance_id.as_bytes())?;
        update_length_prefixed(&mut hasher, self.unit_id.as_bytes())?;
        hasher.update(self.generation.to_le_bytes());
        hasher.update(self.schema_version.to_le_bytes());
        Ok(finalize(hasher))
    }

    pub(crate) fn validate(&self) -> Result<()> {
        validate_identifier("experiment_id", &self.experiment_id)?;
        validate_identifier("instance_id", &self.instance_id)?;
        validate_identifier("unit_id", &self.unit_id)
    }
}

/// Root of one context-bound Merkle commitment.
#[derive(Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct MerkleRoot(Digest);

impl MerkleRoot {
    /// Constructs a root from its exact 32-byte representation.
    pub const fn from_bytes(bytes: Digest) -> Self {
        Self(bytes)
    }

    /// Returns the exact 32-byte representation.
    pub const fn as_bytes(&self) -> &Digest {
        &self.0
    }

    /// Returns lowercase hexadecimal without allocating protocol state.
    pub fn to_hex(self) -> String {
        let mut output = String::with_capacity(64);
        for byte in self.0 {
            use fmt::Write as _;
            let _ = write!(output, "{byte:02x}");
        }
        output
    }
}

impl fmt::Debug for MerkleRoot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("MerkleRoot")
            .field(&self.to_hex())
            .finish()
    }
}

impl fmt::Display for MerkleRoot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.to_hex())
    }
}

/// Public commitment to an ordered leaf collection and experiment context.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct MerkleCommitment {
    version: u8,
    root: MerkleRoot,
    leaf_count: u64,
    context_digest: Digest,
}

#[derive(Deserialize)]
struct MerkleCommitmentWire {
    version: u8,
    root: MerkleRoot,
    leaf_count: u64,
    context_digest: Digest,
}

impl<'de> Deserialize<'de> for MerkleCommitment {
    fn deserialize<D>(deserializer: D) -> core::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = MerkleCommitmentWire::deserialize(deserializer)?;
        if wire.version != PROTOCOL_VERSION {
            return Err(D::Error::custom("unsupported experiment-merkle version"));
        }
        if wire.leaf_count > MAX_LEAVES {
            return Err(D::Error::custom("Merkle tree has too many leaves"));
        }
        Ok(Self {
            version: wire.version,
            root: wire.root,
            leaf_count: wire.leaf_count,
            context_digest: wire.context_digest,
        })
    }
}

impl MerkleCommitment {
    /// Protocol version used to construct the commitment.
    pub const fn version(&self) -> u8 {
        self.version
    }

    /// Context-bound Merkle root.
    pub const fn root(&self) -> MerkleRoot {
        self.root
    }

    /// Number of real leaves committed before deterministic padding.
    pub const fn leaf_count(&self) -> u64 {
        self.leaf_count
    }

    /// Digest of the experiment/run context.
    pub const fn context_digest(&self) -> &Digest {
        &self.context_digest
    }
}

/// Inclusion path for one leaf.
///
/// Sibling orientation is deliberately absent: verification derives it from
/// `leaf_index`, preventing a proof from claiming a different position.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct MerkleProof {
    version: u8,
    leaf_index: u64,
    siblings: Vec<Digest>,
}

#[derive(Deserialize)]
struct MerkleProofWire {
    version: u8,
    leaf_index: u64,
    #[serde(deserialize_with = "crate::serde_bounded::proof_siblings")]
    siblings: Vec<Digest>,
}

impl<'de> Deserialize<'de> for MerkleProof {
    fn deserialize<D>(deserializer: D) -> core::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = MerkleProofWire::deserialize(deserializer)?;
        if wire.version != PROTOCOL_VERSION {
            return Err(D::Error::custom("unsupported experiment-merkle version"));
        }
        Ok(Self {
            version: wire.version,
            leaf_index: wire.leaf_index,
            siblings: wire.siblings,
        })
    }
}

impl MerkleProof {
    /// Protocol version used to construct the proof.
    pub const fn version(&self) -> u8 {
        self.version
    }

    /// Position of the opened leaf in the committed ordered collection.
    pub const fn leaf_index(&self) -> u64 {
        self.leaf_index
    }

    /// Sibling hashes ordered from the leaf layer toward the root.
    pub fn siblings(&self) -> &[Digest] {
        &self.siblings
    }
}

/// Opaque in-memory tree capable of producing v1 inclusion proofs.
#[derive(Debug, Clone)]
pub struct MerkleTree {
    commitment: MerkleCommitment,
    nodes: Vec<Digest>,
    capacity: usize,
}

impl MerkleTree {
    /// Builds a tree over ordered, variable-length byte leaves.
    pub fn from_leaves<B: AsRef<[u8]>>(context: CommitmentContext, leaves: &[B]) -> Result<Self> {
        context.validate()?;
        let leaf_count = u64::try_from(leaves.len()).map_err(|_| MerkleError::TooManyLeaves)?;
        if leaf_count > MAX_LEAVES {
            return Err(MerkleError::TooManyLeaves);
        }
        let context_digest = context.digest()?;

        for leaf in leaves {
            if leaf.as_ref().len() > MAX_LEAF_BYTES {
                return Err(MerkleError::LeafTooLarge);
            }
        }

        if leaves.is_empty() {
            let internal_root = hash_empty(&context_digest);
            return Ok(Self {
                commitment: make_commitment(context_digest, 0, internal_root),
                nodes: Vec::new(),
                capacity: 0,
            });
        }

        let capacity = leaves
            .len()
            .checked_next_power_of_two()
            .ok_or(MerkleError::SizeOverflow)?;
        let node_count = capacity.checked_mul(2).ok_or(MerkleError::SizeOverflow)?;
        let mut nodes = vec![[0_u8; 32]; node_count];

        for (index, leaf) in leaves.iter().enumerate() {
            let bytes = leaf.as_ref();
            let index_u64 = u64::try_from(index).map_err(|_| MerkleError::SizeOverflow)?;
            nodes[capacity + index] = hash_leaf(&context_digest, index_u64, bytes)?;
        }
        for index in leaves.len()..capacity {
            let index_u64 = u64::try_from(index).map_err(|_| MerkleError::SizeOverflow)?;
            nodes[capacity + index] = hash_padding(&context_digest, index_u64);
        }

        let mut child_start = capacity;
        let mut level = 0_u32;
        while child_start > 1 {
            let parent_start = child_start / 2;
            for parent in parent_start..child_start {
                nodes[parent] = hash_node(level, &nodes[parent * 2], &nodes[parent * 2 + 1]);
            }
            child_start = parent_start;
            level = level.checked_add(1).ok_or(MerkleError::SizeOverflow)?;
        }

        let commitment = make_commitment(context_digest, leaf_count, nodes[1]);
        Ok(Self {
            commitment,
            nodes,
            capacity,
        })
    }

    /// Builds a tree by dividing one byte slice into equal, complete leaves.
    pub fn from_chunks(
        context: CommitmentContext,
        bytes: &[u8],
        bytes_per_leaf: usize,
    ) -> Result<Self> {
        if bytes_per_leaf == 0 {
            return Err(MerkleError::InvalidChunkSize);
        }
        if bytes.len() % bytes_per_leaf != 0 {
            return Err(MerkleError::NonDivisibleData);
        }
        if bytes_per_leaf > MAX_LEAF_BYTES {
            return Err(MerkleError::LeafTooLarge);
        }
        let leaf_count = bytes.len() / bytes_per_leaf;
        if u64::try_from(leaf_count).map_err(|_| MerkleError::TooManyLeaves)? > MAX_LEAVES {
            return Err(MerkleError::TooManyLeaves);
        }
        let leaves = bytes.chunks_exact(bytes_per_leaf).collect::<Vec<_>>();
        Self::from_leaves(context, &leaves)
    }

    /// Returns the public commitment.
    pub const fn commitment(&self) -> &MerkleCommitment {
        &self.commitment
    }

    /// Produces an inclusion proof for a real leaf.
    pub fn proof(&self, leaf_index: u64) -> Result<MerkleProof> {
        if self.commitment.leaf_count == 0 {
            return Err(MerkleError::EmptyTree);
        }
        if leaf_index >= self.commitment.leaf_count {
            return Err(MerkleError::IndexOutOfRange);
        }
        let leaf_index_usize =
            usize::try_from(leaf_index).map_err(|_| MerkleError::IndexOutOfRange)?;
        let mut position = self
            .capacity
            .checked_add(leaf_index_usize)
            .ok_or(MerkleError::SizeOverflow)?;
        let mut siblings = Vec::with_capacity(expected_depth(self.commitment.leaf_count)?);
        while position > 1 {
            siblings.push(self.nodes[position ^ 1]);
            position /= 2;
        }
        Ok(MerkleProof {
            version: PROTOCOL_VERSION,
            leaf_index,
            siblings,
        })
    }
}

/// Verifies that `leaf_bytes` occupies `expected_index` in the commitment.
pub fn verify_opening(
    commitment: &MerkleCommitment,
    context: &CommitmentContext,
    expected_index: u64,
    leaf_bytes: &[u8],
    proof: &MerkleProof,
) -> Result<()> {
    validate_commitment_context(commitment, context)?;
    if proof.version != PROTOCOL_VERSION {
        return Err(MerkleError::UnsupportedVersion(proof.version));
    }
    if proof.leaf_index != expected_index {
        return Err(MerkleError::UnexpectedProofIndex);
    }
    if commitment.leaf_count == 0 || proof.leaf_index >= commitment.leaf_count {
        return Err(MerkleError::IndexOutOfRange);
    }
    if leaf_bytes.len() > MAX_LEAF_BYTES {
        return Err(MerkleError::LeafTooLarge);
    }
    let depth = expected_depth(commitment.leaf_count)?;
    if proof.siblings.len() != depth {
        return Err(MerkleError::InvalidProofDepth);
    }

    let mut current = hash_leaf(&commitment.context_digest, proof.leaf_index, leaf_bytes)?;
    let mut position = proof.leaf_index;
    for (level, sibling) in proof.siblings.iter().enumerate() {
        let level = u32::try_from(level).map_err(|_| MerkleError::SizeOverflow)?;
        current = if position & 1 == 0 {
            hash_node(level, &current, sibling)
        } else {
            hash_node(level, sibling, &current)
        };
        position >>= 1;
    }
    let root = hash_root(&commitment.context_digest, commitment.leaf_count, &current);
    if root == *commitment.root.as_bytes() {
        Ok(())
    } else {
        Err(MerkleError::RootMismatch)
    }
}

pub(crate) fn validate_commitment_context(
    commitment: &MerkleCommitment,
    context: &CommitmentContext,
) -> Result<()> {
    if commitment.version != PROTOCOL_VERSION {
        return Err(MerkleError::UnsupportedVersion(commitment.version));
    }
    if commitment.leaf_count > MAX_LEAVES {
        return Err(MerkleError::TooManyLeaves);
    }
    if context.digest()? != commitment.context_digest {
        return Err(MerkleError::ContextMismatch);
    }
    Ok(())
}

fn make_commitment(
    context_digest: Digest,
    leaf_count: u64,
    internal_root: Digest,
) -> MerkleCommitment {
    MerkleCommitment {
        version: PROTOCOL_VERSION,
        root: MerkleRoot(hash_root(&context_digest, leaf_count, &internal_root)),
        leaf_count,
        context_digest,
    }
}

fn expected_depth(leaf_count: u64) -> Result<usize> {
    if leaf_count == 0 {
        return Ok(0);
    }
    let capacity = leaf_count
        .checked_next_power_of_two()
        .ok_or(MerkleError::SizeOverflow)?;
    usize::try_from(capacity.trailing_zeros()).map_err(|_| MerkleError::SizeOverflow)
}

fn validate_identifier(name: &'static str, value: &str) -> Result<()> {
    if value.is_empty() {
        return Err(MerkleError::EmptyIdentifier(name));
    }
    if value.len() > MAX_CONTEXT_FIELD_BYTES {
        return Err(MerkleError::IdentifierTooLong(name));
    }
    Ok(())
}

fn update_length_prefixed(hasher: &mut Sha256, bytes: &[u8]) -> Result<()> {
    let length = u64::try_from(bytes.len()).map_err(|_| MerkleError::SizeOverflow)?;
    hasher.update(length.to_le_bytes());
    hasher.update(bytes);
    Ok(())
}

fn hash_leaf(context_digest: &Digest, index: u64, bytes: &[u8]) -> Result<Digest> {
    let mut hasher = Sha256::new();
    hasher.update(LEAF_DOMAIN);
    hasher.update(context_digest);
    hasher.update(index.to_le_bytes());
    update_length_prefixed(&mut hasher, bytes)?;
    Ok(finalize(hasher))
}

fn hash_padding(context_digest: &Digest, index: u64) -> Digest {
    let mut hasher = Sha256::new();
    hasher.update(PADDING_DOMAIN);
    hasher.update(context_digest);
    hasher.update(index.to_le_bytes());
    finalize(hasher)
}

fn hash_node(level: u32, left: &Digest, right: &Digest) -> Digest {
    let mut hasher = Sha256::new();
    hasher.update(NODE_DOMAIN);
    hasher.update(level.to_le_bytes());
    hasher.update(left);
    hasher.update(right);
    finalize(hasher)
}

fn hash_empty(context_digest: &Digest) -> Digest {
    let mut hasher = Sha256::new();
    hasher.update(EMPTY_DOMAIN);
    hasher.update(context_digest);
    finalize(hasher)
}

fn hash_root(context_digest: &Digest, leaf_count: u64, internal_root: &Digest) -> Digest {
    let mut hasher = Sha256::new();
    hasher.update(ROOT_DOMAIN);
    hasher.update([PROTOCOL_VERSION]);
    hasher.update(context_digest);
    hasher.update(leaf_count.to_le_bytes());
    hasher.update(internal_root);
    finalize(hasher)
}

fn finalize(hasher: Sha256) -> Digest {
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> CommitmentContext {
        CommitmentContext::new("demo", "run-1", "unit-7", 3, 1).unwrap()
    }

    #[test]
    fn all_real_leaves_verify_for_non_power_of_two_tree() {
        let leaves = [b"zero".as_slice(), b"one", b"two", b"three", b"four"];
        let tree = MerkleTree::from_leaves(context(), &leaves).unwrap();
        assert_eq!(tree.commitment().leaf_count(), 5);
        for (index, leaf) in leaves.iter().enumerate() {
            let proof = tree.proof(index as u64).unwrap();
            verify_opening(tree.commitment(), &context(), index as u64, leaf, &proof).unwrap();
        }
    }

    #[test]
    fn empty_and_single_leaf_trees_are_defined() {
        let empty = MerkleTree::from_leaves::<&[u8]>(context(), &[]).unwrap();
        assert_eq!(empty.commitment().leaf_count(), 0);
        assert!(matches!(empty.proof(0), Err(MerkleError::EmptyTree)));

        let one = MerkleTree::from_leaves(context(), &[b"only".as_slice()]).unwrap();
        let proof = one.proof(0).unwrap();
        assert!(proof.siblings().is_empty());
        verify_opening(one.commitment(), &context(), 0, b"only", &proof).unwrap();
    }

    #[test]
    fn index_value_context_and_path_are_bound() {
        let leaves = [b"a".as_slice(), b"b", b"c"];
        let tree = MerkleTree::from_leaves(context(), &leaves).unwrap();
        let proof = tree.proof(1).unwrap();
        assert!(matches!(
            verify_opening(tree.commitment(), &context(), 1, b"changed", &proof),
            Err(MerkleError::RootMismatch)
        ));

        let mut wrong_index = proof.clone();
        wrong_index.leaf_index = 2;
        assert!(matches!(
            verify_opening(tree.commitment(), &context(), 1, b"b", &wrong_index),
            Err(MerkleError::UnexpectedProofIndex)
        ));

        let other_context = CommitmentContext::new("demo", "run-2", "unit-7", 3, 1).unwrap();
        assert!(matches!(
            verify_opening(tree.commitment(), &other_context, 1, b"b", &proof),
            Err(MerkleError::ContextMismatch)
        ));

        let mut truncated = proof.clone();
        truncated.siblings.pop();
        assert!(matches!(
            verify_opening(tree.commitment(), &context(), 1, b"b", &truncated),
            Err(MerkleError::InvalidProofDepth)
        ));
        let mut appended = proof.clone();
        appended.siblings.push([0_u8; 32]);
        assert!(matches!(
            verify_opening(tree.commitment(), &context(), 1, b"b", &appended),
            Err(MerkleError::InvalidProofDepth)
        ));

        let mut wrong_proof_version = proof.clone();
        wrong_proof_version.version = 2;
        assert!(matches!(
            verify_opening(tree.commitment(), &context(), 1, b"b", &wrong_proof_version),
            Err(MerkleError::UnsupportedVersion(2))
        ));

        let mut mutated_sibling = proof.clone();
        mutated_sibling.siblings[0][0] ^= 1;
        assert!(matches!(
            verify_opening(tree.commitment(), &context(), 1, b"b", &mutated_sibling),
            Err(MerkleError::RootMismatch)
        ));
        let mut swapped_siblings = proof.clone();
        swapped_siblings.siblings.swap(0, 1);
        assert!(matches!(
            verify_opening(tree.commitment(), &context(), 1, b"b", &swapped_siblings),
            Err(MerkleError::RootMismatch)
        ));

        let mut wrong_version = tree.commitment().clone();
        wrong_version.version = 2;
        assert!(matches!(
            verify_opening(&wrong_version, &context(), 1, b"b", &proof),
            Err(MerkleError::UnsupportedVersion(2))
        ));
        let mut wrong_root = tree.commitment().clone();
        wrong_root.root.0[0] ^= 1;
        assert!(matches!(
            verify_opening(&wrong_root, &context(), 1, b"b", &proof),
            Err(MerkleError::RootMismatch)
        ));
        let mut wrong_count = tree.commitment().clone();
        wrong_count.leaf_count = 4;
        assert!(matches!(
            verify_opening(&wrong_count, &context(), 1, b"b", &proof),
            Err(MerkleError::RootMismatch)
        ));
        let mut wrong_context_digest = tree.commitment().clone();
        wrong_context_digest.context_digest[0] ^= 1;
        assert!(matches!(
            verify_opening(&wrong_context_digest, &context(), 1, b"b", &proof),
            Err(MerkleError::ContextMismatch)
        ));
    }

    #[test]
    fn every_context_field_changes_the_commitment() {
        let leaves = [b"value".as_slice()];
        let base = MerkleTree::from_leaves(context(), &leaves)
            .unwrap()
            .commitment()
            .root();
        let variants = [
            CommitmentContext::new("other", "run-1", "unit-7", 3, 1).unwrap(),
            CommitmentContext::new("demo", "other", "unit-7", 3, 1).unwrap(),
            CommitmentContext::new("demo", "run-1", "other", 3, 1).unwrap(),
            CommitmentContext::new("demo", "run-1", "unit-7", 4, 1).unwrap(),
            CommitmentContext::new("demo", "run-1", "unit-7", 3, 2).unwrap(),
        ];
        for variant in variants {
            let root = MerkleTree::from_leaves(variant, &leaves)
                .unwrap()
                .commitment()
                .root();
            assert_ne!(root, base);
        }
    }

    #[test]
    fn fixed_chunk_adapter_is_checked() {
        assert!(matches!(
            MerkleTree::from_chunks(context(), b"abcd", 0),
            Err(MerkleError::InvalidChunkSize)
        ));
        assert!(matches!(
            MerkleTree::from_chunks(context(), b"abcde", 2),
            Err(MerkleError::NonDivisibleData)
        ));
        let tree = MerkleTree::from_chunks(context(), b"abcdef", 2).unwrap();
        assert_eq!(tree.commitment().leaf_count(), 3);
    }
}
