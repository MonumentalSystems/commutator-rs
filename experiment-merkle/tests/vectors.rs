use experiment_merkle::merkle::{
    verify_opening, CommitmentContext, MerkleCommitment, MerkleProof, MerkleTree, MAX_LEAVES,
    PROTOCOL_VERSION,
};
use experiment_merkle::spot_check::{derive_challenge, SpotCheckChallenge};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    protocol_version: u8,
    context: CommitmentContext,
    context_digest_hex: String,
    challenge: Challenge,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Challenge {
    leaf_count: usize,
    nonce_hex: String,
    requested_count: usize,
    indices: Vec<u64>,
}

#[derive(Deserialize)]
struct Case {
    leaf_count: usize,
    leaves_hex: Vec<String>,
    root_hex: String,
    proofs: Vec<Proof>,
}

#[derive(Deserialize)]
struct Proof {
    leaf_index: u64,
    siblings_hex: Vec<String>,
}

fn decode_hex(value: &str) -> Vec<u8> {
    assert_eq!(value.len() % 2, 0, "fixture hex must have whole bytes");
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let pair = core::str::from_utf8(pair).unwrap();
            u8::from_str_radix(pair, 16).unwrap()
        })
        .collect()
}

fn encode_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[test]
fn frozen_v1_vectors_match_roots_proofs_and_serde_fields() {
    let fixture: Fixture = serde_json::from_str(include_str!("fixtures/v1.json")).unwrap();
    assert_eq!(fixture.protocol_version, PROTOCOL_VERSION);
    assert_eq!(
        encode_hex(&fixture.context.digest().unwrap()),
        fixture.context_digest_hex
    );

    for case in fixture.cases {
        let leaves = case
            .leaves_hex
            .iter()
            .map(|leaf| decode_hex(leaf))
            .collect::<Vec<_>>();
        assert_eq!(leaves.len(), case.leaf_count);
        let tree = MerkleTree::from_leaves(fixture.context.clone(), &leaves).unwrap();
        assert_eq!(tree.commitment().root().to_hex(), case.root_hex);
        assert_eq!(tree.commitment().leaf_count(), case.leaf_count as u64);
        assert_eq!(case.proofs.len(), case.leaf_count);

        for expected in case.proofs {
            let proof = tree.proof(expected.leaf_index).unwrap();
            assert_eq!(proof.version(), PROTOCOL_VERSION);
            assert_eq!(
                proof
                    .siblings()
                    .iter()
                    .map(|sibling| encode_hex(sibling))
                    .collect::<Vec<_>>(),
                expected.siblings_hex
            );
            verify_opening(
                tree.commitment(),
                &fixture.context,
                expected.leaf_index,
                &leaves[expected.leaf_index as usize],
                &proof,
            )
            .unwrap();
        }

        if case.leaf_count == 3 {
            let proof = tree.proof(2).unwrap();
            let commitment_json = serde_json::to_value(tree.commitment()).unwrap();
            let proof_json = serde_json::to_value(&proof).unwrap();
            assert_eq!(commitment_json["version"], 1);
            assert_eq!(commitment_json["leaf_count"], 3);
            assert_eq!(
                commitment_json["root"],
                serde_json::json!(decode_hex(&case.root_hex))
            );
            assert_eq!(proof_json["version"], 1);
            assert_eq!(proof_json["leaf_index"], 2);
            assert_eq!(proof_json["siblings"].as_array().unwrap().len(), 2);
        }
    }
}

#[test]
fn frozen_v1_challenge_vector_matches() {
    let fixture: Fixture = serde_json::from_str(include_str!("fixtures/v1.json")).unwrap();
    let case = fixture
        .cases
        .iter()
        .find(|case| case.leaf_count == fixture.challenge.leaf_count)
        .unwrap();
    let leaves = case
        .leaves_hex
        .iter()
        .map(|leaf| decode_hex(leaf))
        .collect::<Vec<_>>();
    let tree = MerkleTree::from_leaves(fixture.context.clone(), &leaves).unwrap();
    let nonce: [u8; 32] = decode_hex(&fixture.challenge.nonce_hex).try_into().unwrap();
    let challenge = derive_challenge(
        tree.commitment(),
        &fixture.context,
        nonce,
        fixture.challenge.requested_count,
    )
    .unwrap();
    assert_eq!(challenge.indices(), fixture.challenge.indices);
}

#[test]
fn bounded_wire_types_reject_invalid_state() {
    let fixture: Fixture = serde_json::from_str(include_str!("fixtures/v1.json")).unwrap();
    let tree = MerkleTree::from_leaves(
        fixture.context.clone(),
        &[b"zero".as_slice(), b"one", b"two"],
    )
    .unwrap();

    let mut context_json = serde_json::to_value(&fixture.context).unwrap();
    context_json["unit_id"] = serde_json::json!("");
    assert!(serde_json::from_value::<CommitmentContext>(context_json).is_err());
    let oversized = "x".repeat(4_097);
    let mut context_json = serde_json::to_value(&fixture.context).unwrap();
    context_json["experiment_id"] = serde_json::json!(oversized);
    assert!(serde_json::from_value::<CommitmentContext>(context_json).is_err());

    let mut commitment_json = serde_json::to_value(tree.commitment()).unwrap();
    commitment_json["leaf_count"] = serde_json::json!(MAX_LEAVES + 1);
    assert!(serde_json::from_value::<MerkleCommitment>(commitment_json).is_err());

    let proof = tree.proof(1).unwrap();
    let mut proof_json = serde_json::to_value(&proof).unwrap();
    proof_json["version"] = serde_json::json!(2);
    assert!(serde_json::from_value::<MerkleProof>(proof_json).is_err());
    let mut proof_json = serde_json::to_value(&proof).unwrap();
    proof_json["siblings"] = serde_json::json!(vec![[0_u8; 32]; 21]);
    assert!(serde_json::from_value::<MerkleProof>(proof_json).is_err());

    let challenge = derive_challenge(tree.commitment(), &fixture.context, [3_u8; 32], 2).unwrap();
    let mut challenge_json = serde_json::to_value(&challenge).unwrap();
    let first = challenge.indices()[0];
    challenge_json["indices"] = serde_json::json!([first, first]);
    assert!(serde_json::from_value::<SpotCheckChallenge>(challenge_json).is_err());
}

#[test]
fn frozen_utf8_and_binary_vector_matches() {
    let context = CommitmentContext::new("μ-field", "試験-1", "tile-🦀", 11, 3).unwrap();
    let leaves = ["Δ".as_bytes(), &[0_u8, 255, 1][..]];
    let tree = MerkleTree::from_leaves(context.clone(), &leaves).unwrap();
    assert_eq!(
        encode_hex(&context.digest().unwrap()),
        "751e629212328f3d305e0b58ef65f1291885f49a7c7909fee1308dcb81bd6e1f"
    );
    assert_eq!(
        tree.commitment().root().to_hex(),
        "6b9bd9cb99822dca52595f55ab9ab7319bb66d2321825b8219ccee9bd57a8893"
    );
}
