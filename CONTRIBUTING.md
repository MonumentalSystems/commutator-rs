# Contributing

Thank you for helping make reusable scientific software easier to inspect,
test, and compose.

## Before opening a change

1. Keep networking, storage, identity, and scheduler concerns out of the
   foundational crates.
2. Treat public basis ordering, normalization, serialization, and numerical
   behavior as compatibility contracts.
3. Add a focused regression test for every numerical or API change.
4. Prefer portable reference implementations here; specialized SIMD and GPU
   implementations should live behind downstream adapters and differential
   tests.

## Local checks

```bash
cargo fmt --all --check
cargo test --workspace --all-targets --all-features --locked
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
RUSTDOCFLAGS="-D warnings -D missing-docs" cargo doc --workspace --all-features --no-deps --locked
cargo package -p clifford-core -p clifford-field -p clifford-geometry -p clifford-lattice -p clifford-layers -p clifford-mesh -p experiment-core -p experiment-merkle -p harmonic-dynamics --locked
```

## Compatibility fixtures

Changes to Clifford conventions require an explicit versioned fixture update,
an explanation in `clifford-core/CONVENTIONS.md`, and independent comparison
against the cited implementation. Do not silently regenerate golden values.

Changes to the `experiment-merkle` hash or challenge contract require a new
protocol version, regenerated versioned vectors, and an updated fixture
checksum. Do not rewrite an already published protocol version's vectors.

## Pull requests

Keep pull requests narrow, explain user-visible behavior, and call out any
floating-point or reproducibility impact. By contributing, you agree that your
work is licensed under the target crate's declared license.

