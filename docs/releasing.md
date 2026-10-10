# Release and provenance

This repository is a multi-package Rust workspace. A release is not complete
merely because a Git tag exists: every package must be built from the same
reviewed source revision, registry dependencies must be published in order,
and the resulting archives and scientific fixtures must remain attributable to
that revision.

The crate manifests currently describe release candidates. Nothing in this
document asserts that a crate has already been published.

## Release gates

Start from a clean, signed-off commit on the protected release branch. Record
the output of `git rev-parse HEAD`, `git status --porcelain`, `rustc -Vv`, and
`cargo -V`. Then run the same gates as CI:

```bash
cargo test --workspace --all-targets --all-features --locked
cargo +1.80 test --workspace --all-targets --all-features --locked
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --doc --all-features --locked
RUSTDOCFLAGS="-D warnings -D missing-docs" \
  cargo doc --workspace --all-features --no-deps --locked
sha256sum --check clifford-core/tests/fixtures/clifford-golden-v1.sha256
sha256sum --check experiment-merkle/tests/fixtures/SHA256SUMS
cargo package --workspace --exclude physics-conformance --locked
```

The single multi-package `cargo package` invocation is intentional. Cargo can
resolve every selected path dependency from the same workspace revision while
verifying the generated archives. `physics-conformance` is excluded because it
is a workspace integration-test consumer with `publish = false`.

Before uploading, inspect every generated archive with `cargo package --list
-p <crate>`. The package must contain its README, license, tests, and all source
needed to build without undeclared repository files.

## Dependency-safe publication waves

Crates.io resolves dependency versions from the registry, not sibling paths.
Publish a wave only after every crate in the preceding wave is visible through
the crates.io API. A dry-run is required immediately before each upload.

### Wave 1: no workspace package dependencies

- `clifford-core`
- `cluster-green`
- `experiment-core`
- `field-lyapunov`
- `open-quantum-systems`
- `phonon-transport`
- `quantum-chaos`
- `quantum-light`
- `quantum-magnetism`
- `quantum-shadows`
- `spin-lattice`
- `superconducting-dynamics`
- `superconductivity`

### Wave 2: dependencies supplied by wave 1

- `clifford-field`
- `clifford-geometry`
- `clifford-lattice`
- `clifford-layers`
- `cluster-embedding`
- `experiment-accelerator`
- `experiment-merkle`
- `harmonic-dynamics`
- `keldysh-green`
- `majorana-fermions`
- `quantum-tomography`
- `quantum-transport`

### Wave 3: dependencies supplied by wave 2

- `clifford-mesh` (`clifford-geometry` is an optional registry dependency)

Within a wave, crates are mutually independent at the workspace-package level.
Use explicit commands such as `cargo publish -p <crate> --locked --dry-run`
followed by the corresponding non-dry-run command. Never use an unattended loop
for uploads: crates.io publication is irreversible for that name and version.

`physics-conformance` must never be published. Run it against the final
registry-resolved versions in a temporary consumer project as a post-publish
check before announcing the release.

## Provenance bundle

Attach one provenance bundle to the GitHub release and link it from the website
release page. The bundle should contain:

- the exact Git commit and signed tag;
- `Cargo.lock` and its SHA-256 digest;
- `rustc -Vv`, Cargo version, target triple, operating system, and CI run URL;
- one SHA-256 digest for every `target/package/*.crate` archive;
- the two checked scientific-fixture checksum files;
- the test, Clippy, rustdoc, MSRV, and package job conclusions;
- authenticated serialize-only backend qualification audit snapshots,
  including descriptor digests, precision, versioned policy identity, and
  absolute and relative tolerances, for any separately released accelerated
  implementation;
- links or immutable identifiers for associated papers, datasets, model
  weights, and reproduction inputs when they exist.

A machine-readable `release-provenance.json` in the bundle should use this
minimum shape:

```json
{
  "schema": "commutator.release-provenance.v1",
  "git_commit": "40 lowercase hexadecimal characters",
  "git_tag": "signed release tag",
  "cargo_lock_sha256": "64 lowercase hexadecimal characters",
  "rustc": "complete rustc -Vv output",
  "ci_run": "immutable CI run URL",
  "packages": [
    {
      "name": "crate name",
      "version": "semver",
      "archive_sha256": "64 lowercase hexadecimal characters",
      "crates_io": "registry URL after publication",
      "docs_rs": "documentation URL after successful docs.rs build"
    }
  ]
}
```

Generate archive checksums from the packaged bytes, not from source
directories. Keep the checksum file beside the JSON and sign both with the
project's documented release-signing identity. Publication tokens and signing
private keys must never be placed in the repository, CI logs, or provenance
bundle.

## Citation and website update

[`CITATION.cff`](../CITATION.cff) cites the software family. A tagged release
may add a version and release date to that file in the same commit as the
version bumps; do not add them speculatively. Method papers should be linked
from the relevant crate and release record rather than presented as coverage
for unrelated crates.

Only after crates.io accepts a package and docs.rs finishes its build should
`commutator.science` add the corresponding registry, API-documentation, and
download links. Paper, model, dataset, and reproducibility links should resolve
to immutable versions wherever the hosting service supports them.
