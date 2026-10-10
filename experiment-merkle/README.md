# experiment-merkle

`experiment-merkle` provides transport-neutral, context-bound SHA-256 Merkle
commitments for verifiable distributed experiments.

The v1 format binds every commitment to:

- the experiment, run, work unit, generation, and payload-schema version;
- the ordered leaf count;
- each leaf's index, byte length, and exact bytes;
- deterministic padding positions and every internal-tree level.

Proof direction is derived from the committed leaf index. Workers cannot
substitute a proof-carried hash or orientation for the challenged bytes and
position.

```rust
use experiment_merkle::merkle::{verify_opening, CommitmentContext, MerkleTree};
use experiment_merkle::spot_check::{
    derive_challenge, verify_challenge_openings, MerkleOpening,
};

let context = CommitmentContext::new("ising", "run-42", "tile-7", 3, 1)?;
let leaves = [b"site-0".as_slice(), b"site-1", b"site-2"];
let tree = MerkleTree::from_leaves(context.clone(), &leaves)?;

let proof = tree.proof(1)?;
verify_opening(tree.commitment(), &context, 1, b"site-1", &proof)?;

// Test-only nonce. Production verifiers must generate this freshly and
// unpredictably only after receiving the commitment.
let challenge = derive_challenge(tree.commitment(), &context, [7_u8; 32], 2)?;
let openings = challenge
    .indices()
    .iter()
    .map(|&index| {
        MerkleOpening::new(leaves[index as usize].to_vec(), tree.proof(index)?)
    })
    .collect::<Result<Vec<_>, experiment_merkle::MerkleError>>()?;
verify_challenge_openings(
    tree.commitment(),
    &context,
    [7_u8; 32],
    2,
    &openings,
)?;
# Ok::<(), experiment_merkle::MerkleError>(())
```

## Security boundary

A valid opening proves only that specific bytes occupied a specific position
in the committed result. It does not prove the computation was correct.
Sound spot checking additionally requires a fresh, unpredictable post-commitment
nonce, an appropriate sampling policy, and independent scientific
recomputation or validation by the host.

The commitment is binding only under the usual SHA-256 collision-resistance
assumption. It is not hiding and it does not authenticate a worker; openings
reveal their exact leaf bytes. The host must authenticate and retain the
commitment, then fix it before disclosing the nonce.

For verification, keep the commitment, nonce, and challenge count in trusted
verifier state. `verify_challenge_openings` re-derives the sample from those
inputs and does not trust a worker-echoed challenge object.

The crate contains no identity scheme, network transport, scheduler,
database, reputation system, physics kernel, or random-number generator.

## Compatibility

The v1 format is intentionally not compatible with Commutator's legacy Merkle
format. See [`SPEC.md`](SPEC.md) for the byte-level hashing contract and
[`PROVENANCE.md`](PROVENANCE.md) for the extraction rationale.
Frozen roots, proofs, and challenge indices live in `tests/fixtures/v1.json`
with a checked SHA-256 checksum. Serde's representation is transport-neutral
convenience, not a canonical v1 wire encoding.

The library applies streaming collection limits during deserialization where
the data format permits it. Network hosts must still impose request-body
limits: some deserializers allocate a complete string or byte buffer before
invoking a Serde visitor.

## License

MIT.
