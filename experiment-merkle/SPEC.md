# experiment-merkle v1 specification

All integers are unsigned little-endian. All hashes are SHA-256 and produce
32 bytes. `LP(x)` means `u64_le(len(x)) || x`. Strings are their exact UTF-8
bytes. `||` denotes concatenation.

## Context

```text
context_digest = SHA256(
  "experiment-merkle:v1:context\0" ||
  LP(experiment_id) ||
  LP(instance_id) ||
  LP(unit_id) ||
  u64_le(generation) ||
  u32_le(schema_version)
)
```

## Leaves and padding

For a real leaf at zero-based position `i`:

```text
leaf_hash = SHA256(
  "experiment-merkle:v1:leaf\0" ||
  context_digest ||
  u64_le(i) ||
  LP(leaf_bytes)
)
```

The tree capacity is the least power of two greater than or equal to the real
leaf count. For each padding position `i`:

```text
padding_hash = SHA256(
  "experiment-merkle:v1:padding\0" ||
  context_digest ||
  u64_le(i)
)
```

## Internal nodes

Level zero combines leaf-layer children. Higher levels increment `level`:

```text
node_hash = SHA256(
  "experiment-merkle:v1:node\0" ||
  u32_le(level) ||
  left_hash ||
  right_hash
)
```

For an empty tree:

```text
internal_root = SHA256(
  "experiment-merkle:v1:empty\0" || context_digest
)
```

For a nonempty tree, `internal_root` is the final internal node, or the single
leaf hash when the leaf count is one.

## Public root

```text
root = SHA256(
  "experiment-merkle:v1:root\0" ||
  0x01 ||
  context_digest ||
  u64_le(real_leaf_count) ||
  internal_root
)
```

An inclusion proof contains protocol version 1, the leaf index, and sibling
hashes from the leaf layer toward the root. It contains neither orientation
flags nor a trusted leaf hash. At level `k`, bit `k` of the leaf index selects
whether the current value is the left or right child.

## Challenge derivation

The verifier supplies a fresh, unpredictable 32-byte nonce after receiving the
commitment. Each candidate word is the first eight bytes, interpreted as
`u64_le`, of:

```text
SHA256(
  "experiment-merkle:v1:challenge\0" ||
  0x01 ||
  root ||
  context_digest ||
  u64_le(real_leaf_count) ||
  verifier_nonce ||
  u64_le(requested_count) ||
  u64_le(counter)
)
```

Modulo bias is removed with rejection sampling. Unique indices are selected
with Floyd sampling without replacement and returned in derivation order.
When checking a response, the verifier re-derives this ordered sample from its
own retained commitment, nonce, and requested count; echoed challenge state is
not trusted.

Exact sampling procedure for `N = real_leaf_count` and requested count `k`:

1. Require `1 <= k <= min(N, 4096)`, set `counter = 0`, and start with an empty
   selected set and output list.
2. For each `j` from `N - k` through `N - 1`, draw a candidate below `j + 1`.
3. To draw below `bound`, compute `threshold = (-bound mod 2^64) mod bound`.
   Hash the transcript above with the current counter, interpret its first
   eight bytes as `u64_le(candidate)`, then increment the counter. Rejected
   candidates still consume a counter value. Accept when
   `candidate >= threshold`, returning `candidate mod bound`.
4. If the returned candidate is already selected, choose `j`; otherwise choose
   the candidate. Insert and append the chosen index.

Serde data-model representations are not a canonical wire encoding. Protocol
v1 fixes only the hash inputs, proof semantics, and challenge derivation above.

## Versioning

The protocol version is independent of the crate's semantic version. Any
change to domain labels, byte order, context fields, leaf or padding rules,
tree shape, public-root binding, proof semantics, or challenge sampling
requires a new protocol version and new frozen vectors. Compatible API or
transport-representation changes do not alter the protocol version.

## Resource limits

- at most 2^20 real leaves per tree;
- at most 16 MiB per leaf;
- at most 4 KiB per context identifier;
- at most 4,096 openings and 64 MiB of opened leaf data per challenge.

These are v1 library limits, not recommendations for transport-layer request
sizes. Hosts should enforce smaller serialized-message limits where needed;
some data formats may allocate a token before bounded Serde visitors run.
