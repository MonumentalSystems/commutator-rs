//! Transport-neutral post-commitment spot-check challenges.

use std::collections::{BTreeMap, BTreeSet};

use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest as _, Sha256};

use crate::merkle::{
    validate_commitment_context, verify_opening, CommitmentContext, Digest, MerkleCommitment,
    MerkleProof, MAX_LEAF_BYTES,
};
use crate::{MerkleError, Result};

const CHALLENGE_DOMAIN: &[u8] = b"experiment-merkle:v1:challenge\0";

/// Maximum leaves sampled by one spot-check challenge.
pub const MAX_CHALLENGE_OPENINGS: usize = 4_096;
/// Maximum aggregate opened leaf bytes accepted by one verification call.
pub const MAX_CHALLENGE_LEAF_BYTES: usize = 64 * 1024 * 1024;

/// A verifier-selected challenge issued after a worker commitment.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct SpotCheckChallenge {
    commitment: MerkleCommitment,
    verifier_nonce: Digest,
    indices: Vec<u64>,
}

#[derive(Deserialize)]
struct SpotCheckChallengeWire {
    commitment: MerkleCommitment,
    verifier_nonce: Digest,
    #[serde(deserialize_with = "crate::serde_bounded::challenge_indices")]
    indices: Vec<u64>,
}

impl<'de> Deserialize<'de> for SpotCheckChallenge {
    fn deserialize<D>(deserializer: D) -> core::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = SpotCheckChallengeWire::deserialize(deserializer)?;
        if wire.indices.is_empty()
            || wire.indices.len()
                > usize::try_from(wire.commitment.leaf_count()).map_err(D::Error::custom)?
        {
            return Err(D::Error::custom("challenge index count is invalid"));
        }
        let mut unique = BTreeSet::new();
        for index in &wire.indices {
            if *index >= wire.commitment.leaf_count() || !unique.insert(*index) {
                return Err(D::Error::custom("challenge indices are invalid"));
            }
        }
        Ok(Self {
            commitment: wire.commitment,
            verifier_nonce: wire.verifier_nonce,
            indices: wire.indices,
        })
    }
}

impl SpotCheckChallenge {
    /// Commitment whose leaves are being challenged.
    pub const fn commitment(&self) -> &MerkleCommitment {
        &self.commitment
    }

    /// Verifier-controlled nonce used to derive unpredictable indices.
    pub const fn verifier_nonce(&self) -> &Digest {
        &self.verifier_nonce
    }

    /// Unique challenged indices in deterministic derivation order.
    pub fn indices(&self) -> &[u64] {
        &self.indices
    }
}

/// One challenged leaf value and its inclusion proof.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct MerkleOpening {
    leaf_bytes: Vec<u8>,
    proof: MerkleProof,
}

#[derive(Deserialize)]
struct MerkleOpeningWire {
    #[serde(deserialize_with = "crate::serde_bounded::leaf_bytes")]
    leaf_bytes: Vec<u8>,
    proof: MerkleProof,
}

impl<'de> Deserialize<'de> for MerkleOpening {
    fn deserialize<D>(deserializer: D) -> core::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = MerkleOpeningWire::deserialize(deserializer)?;
        Self::new(wire.leaf_bytes, wire.proof).map_err(D::Error::custom)
    }
}

impl MerkleOpening {
    /// Constructs an opening after enforcing the public leaf-size limit.
    pub fn new(leaf_bytes: Vec<u8>, proof: MerkleProof) -> Result<Self> {
        if leaf_bytes.len() > MAX_LEAF_BYTES {
            return Err(MerkleError::LeafTooLarge);
        }
        Ok(Self { leaf_bytes, proof })
    }

    /// Opened leaf bytes supplied by the worker.
    pub fn leaf_bytes(&self) -> &[u8] {
        &self.leaf_bytes
    }

    /// Inclusion proof for the opened leaf.
    pub const fn proof(&self) -> &MerkleProof {
        &self.proof
    }
}

/// Deterministically derives unique, unbiased challenge indices.
///
/// `verifier_nonce` must be generated unpredictably after the commitment is
/// fixed. This function supplies deterministic sampling, not nonce entropy.
pub fn derive_challenge(
    commitment: &MerkleCommitment,
    context: &CommitmentContext,
    verifier_nonce: Digest,
    challenge_count: usize,
) -> Result<SpotCheckChallenge> {
    validate_commitment_context(commitment, context)?;
    let challenge_count_u64 =
        u64::try_from(challenge_count).map_err(|_| MerkleError::ChallengeCountOutOfRange)?;
    if challenge_count == 0
        || challenge_count > MAX_CHALLENGE_OPENINGS
        || challenge_count_u64 > commitment.leaf_count()
    {
        return Err(MerkleError::ChallengeCountOutOfRange);
    }

    let mut stream = ChallengeStream {
        commitment,
        context_digest: context.digest()?,
        nonce: verifier_nonce,
        requested: challenge_count_u64,
        counter: 0,
    };
    let mut selected = BTreeSet::new();
    let mut indices = Vec::with_capacity(challenge_count);
    let start = commitment.leaf_count() - challenge_count_u64;
    for upper in start..commitment.leaf_count() {
        let candidate = stream.sample_below(upper + 1)?;
        let index = if selected.contains(&candidate) {
            upper
        } else {
            candidate
        };
        selected.insert(index);
        indices.push(index);
    }

    Ok(SpotCheckChallenge {
        commitment: commitment.clone(),
        verifier_nonce,
        indices,
    })
}

/// Re-derives a verifier's challenge and verifies every opening exactly once.
///
/// This proves membership of returned bytes. It does not prove the scientific
/// computation was correct; the host must independently recompute or otherwise
/// validate the opened values. The verifier must supply the commitment, nonce,
/// and count from its own trusted state rather than accepting echoed values from
/// a worker.
pub fn verify_challenge_openings(
    commitment: &MerkleCommitment,
    context: &CommitmentContext,
    verifier_nonce: Digest,
    challenge_count: usize,
    openings: &[MerkleOpening],
) -> Result<()> {
    let challenge = derive_challenge(commitment, context, verifier_nonce, challenge_count)?;
    if openings.len() != challenge.indices.len() {
        return Err(MerkleError::InvalidOpenings);
    }
    let total_leaf_bytes = openings.iter().try_fold(0_usize, |total, opening| {
        total
            .checked_add(opening.leaf_bytes.len())
            .ok_or(MerkleError::SizeOverflow)
    })?;
    if total_leaf_bytes > MAX_CHALLENGE_LEAF_BYTES {
        return Err(MerkleError::ChallengePayloadTooLarge);
    }

    let mut by_index = BTreeMap::new();
    for opening in openings {
        let index = opening.proof.leaf_index();
        if by_index.insert(index, opening).is_some() {
            return Err(MerkleError::InvalidOpenings);
        }
    }
    for index in &challenge.indices {
        let opening = by_index.remove(index).ok_or(MerkleError::InvalidOpenings)?;
        verify_opening(
            commitment,
            context,
            *index,
            &opening.leaf_bytes,
            &opening.proof,
        )?;
    }
    if by_index.is_empty() {
        Ok(())
    } else {
        Err(MerkleError::InvalidOpenings)
    }
}

struct ChallengeStream<'a> {
    commitment: &'a MerkleCommitment,
    context_digest: Digest,
    nonce: Digest,
    requested: u64,
    counter: u64,
}

impl ChallengeStream<'_> {
    fn sample_below(&mut self, bound: u64) -> Result<u64> {
        debug_assert!(bound > 0);
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let mut hasher = Sha256::new();
            hasher.update(CHALLENGE_DOMAIN);
            hasher.update([crate::merkle::PROTOCOL_VERSION]);
            hasher.update(self.commitment.root().as_bytes());
            hasher.update(self.context_digest);
            hasher.update(self.commitment.leaf_count().to_le_bytes());
            hasher.update(self.nonce);
            hasher.update(self.requested.to_le_bytes());
            hasher.update(self.counter.to_le_bytes());
            let digest: Digest = hasher.finalize().into();
            self.counter = self
                .counter
                .checked_add(1)
                .ok_or(MerkleError::SizeOverflow)?;
            let candidate = u64::from_le_bytes(
                digest[..8]
                    .try_into()
                    .map_err(|_| MerkleError::SizeOverflow)?,
            );
            if candidate >= threshold {
                return Ok(candidate % bound);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merkle::MerkleTree;

    fn context() -> CommitmentContext {
        CommitmentContext::new("demo", "run-1", "unit-7", 3, 1).unwrap()
    }

    #[test]
    fn challenges_are_deterministic_unique_and_bound() {
        let leaves = (0_u8..16).map(|value| vec![value]).collect::<Vec<_>>();
        let tree = MerkleTree::from_leaves(context(), &leaves).unwrap();
        let nonce = [7_u8; 32];
        let first = derive_challenge(tree.commitment(), &context(), nonce, 8).unwrap();
        let second = derive_challenge(tree.commitment(), &context(), nonce, 8).unwrap();
        assert_eq!(first, second);
        assert_eq!(
            first.indices.iter().copied().collect::<BTreeSet<_>>().len(),
            8
        );
        assert!(first.indices.iter().all(|index| *index < 16));

        let other_nonce = derive_challenge(tree.commitment(), &context(), [8_u8; 32], 8).unwrap();
        assert_ne!(first.indices, other_nonce.indices);

        let other_context = CommitmentContext::new("demo", "run-2", "unit-7", 3, 1).unwrap();
        let other_tree = MerkleTree::from_leaves(other_context.clone(), &leaves).unwrap();
        let other = derive_challenge(other_tree.commitment(), &other_context, nonce, 8).unwrap();
        assert_ne!(first.indices, other.indices);
    }

    #[test]
    fn complete_opening_set_verifies() {
        let leaves = (0_u8..8).map(|value| vec![value]).collect::<Vec<_>>();
        let tree = MerkleTree::from_leaves(context(), &leaves).unwrap();
        let challenge = derive_challenge(tree.commitment(), &context(), [9_u8; 32], 4).unwrap();
        let openings = challenge
            .indices()
            .iter()
            .map(|index| {
                MerkleOpening::new(leaves[*index as usize].clone(), tree.proof(*index).unwrap())
                    .unwrap()
            })
            .collect::<Vec<_>>();
        verify_challenge_openings(tree.commitment(), &context(), [9_u8; 32], 4, &openings).unwrap();

        assert!(matches!(
            verify_challenge_openings(tree.commitment(), &context(), [9_u8; 32], 4, &openings[..3]),
            Err(MerkleError::InvalidOpenings)
        ));
        let mut duplicated = openings.clone();
        duplicated[1] = duplicated[0].clone();
        assert!(matches!(
            verify_challenge_openings(tree.commitment(), &context(), [9_u8; 32], 4, &duplicated),
            Err(MerkleError::InvalidOpenings)
        ));

        assert!(verify_challenge_openings(
            tree.commitment(),
            &context(),
            [10_u8; 32],
            4,
            &openings
        )
        .is_err());
    }

    #[test]
    fn challenge_count_is_checked() {
        let tree = MerkleTree::from_leaves(context(), &[b"one".as_slice()]).unwrap();
        assert!(derive_challenge(tree.commitment(), &context(), [0_u8; 32], 0).is_err());
        assert!(derive_challenge(tree.commitment(), &context(), [0_u8; 32], 2).is_err());
    }
}
