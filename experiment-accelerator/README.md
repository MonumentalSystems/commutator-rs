# experiment-accelerator

Transport-neutral adapters between [`experiment-core`](https://docs.rs/experiment-core)
work units and CPU/GPU compute backends.

The crate keeps three concerns explicit:

- backend identity, precision, determinism, and worker requirements;
- conversion of typed `WorkUnit<W>` values into provenance-preserving
  `WorkResult<R>` values;
- differential validation of an optimized backend against a portable reference
  backend before results are admitted to a distributed experiment.

It also supplies deterministic independent-replica and contiguous range plans.
The default feature set has no dependency on CUDA, Metal, WebGPU, an async
runtime, or a network transport. Hardware crates can implement `ComputeBackend`
and use the same validation harness in their own CI or startup qualification.
Accepted qualification reports are constructible only by the differential
check API and are bound to the serialized work unit, worker identity, backend
descriptors, concrete execution fingerprints, and an explicit versioned policy
with absolute and relative error limits. Fingerprints are checked again after
admitted execution to close mutation during a call. Reports remain opaque
in-process admission capabilities. A separate serialize-only audit snapshot
records a commitment summary for logs but does not contain its raw preimages
and cannot be deserialized or used as an admission token.

```rust
use experiment_accelerator::{partition_range, BackendDescriptor, BackendKind, Precision,
                             WorkerCapability, GPU_API_LABEL, GPU_AVAILABLE_CAPABILITY};
use std::collections::BTreeMap;

let gpu = BackendDescriptor::try_new(
    "metal-m3",
    "kernel-2026.10",
    BackendKind::Gpu,
    Precision::F32,
    true,
    vec![WorkerCapability { name: GPU_AVAILABLE_CAPABILITY.into(), min_value: Some(1.0) }],
    BTreeMap::from([(GPU_API_LABEL.into(), "metal".into())]),
)?;
assert_eq!(gpu.id(), "metal-m3");
assert_eq!(partition_range(10, 4)?.len(), 3);
# Ok::<(), experiment_accelerator::AdapterError>(())
```

This adapter validates integration and numerical agreement; it does not imply
that a backend is deterministic merely because it runs on a GPU. Backends must
declare that property honestly. Hosts supply a checked `QualificationPolicy`
with a stable identity, version, and finite nonnegative absolute and relative
tolerances appropriate to the backend precision and reduction order. Both
numeric limits and the comparator's scientific-invariant decision must pass.

Qualification reports are opaque in-process capabilities, not portable trust
certificates. They cannot be serialized or deserialized. `audit_snapshot()`
produces a serialize-only commitment summary; a distributed host should retain
the raw descriptor, worker, work, and implementation inputs and sign or
authenticate the resulting artifact when persisting or transmitting it. The
host must recreate authorization from a trusted reference allowlist after
restart. An audit summary cannot recreate an admission capability.
Work binding uses domain-separated SHA-256 over recursively key-sorted canonical
JSON; payload types therefore need stable Serde value semantics.

## Concrete adapters

`ThreadedShardedBackend<B, W, R>` is a deterministically partitioned,
transport-neutral distributed reference. It assigns at most one contiguous
shard to each child backend,
derives a stable per-shard seed, executes children on scoped threads, checks
each result length, namespaces child metrics, and rejoins results in input order
regardless of completion order. A child error or panic identifies its stable
shard ordinal and rejects the whole result. Its descriptor derives determinism,
precision, numeric requirements, and exact labels from the child descriptors;
mixed precision is reported honestly and conflicting child labels are rejected.
Its execution fingerprint recursively binds the ordered child descriptors and
fingerprints, child count, partition algorithm, and seed-derivation version.
Every scoped handle is joined; if multiple shards fail, the lowest shard
ordinal determines the returned error.

The optional `cuda` feature provides `CudaAffineBackend`, a real f64 CUDA
backend for the checked `AffineVectorWork` contract. It uses cudarc/NVRTC to
compile and launch an embedded `scale * x + bias` kernel and can be qualified
against `AffineCpuBackend` through the same policy-bound differential API:

```bash
cargo test -p experiment-accelerator --all-targets --all-features
cargo test -p experiment-accelerator --features cuda --test cuda_hardware \
  -- --ignored --nocapture
NVIDIA_DRIVER_VERSION=580.173.02 \
  cargo run -p experiment-accelerator --features cuda --example cuda_validate
```

CUDA is optional and dynamically loaded; default builds remain portable. The
module has narrow documented unsafe boundaries for the kernel launch and raw
driver/NVRTC version queries. The checked buffer length, element type, context
ownership, and kernel signature are kept together at the launch boundary. Its
execution fingerprint binds CUDA source and generated PTX digests, algorithm
version, device UUID/name/ordinal/compute capability, and driver/NVRTC API
versions.

The committed GB10 validation record contains the raw descriptors, worker,
work unit and payload, policy, outputs, fingerprints, kernel digests, and
device/software metadata needed to inspect or reproduce its commitments. It is
an unauthenticated hardware observation until covered by signed release
provenance and remains unusable for admission:
[`evidence/cuda-gb10-driver-580.173.02.json`](evidence/cuda-gb10-driver-580.173.02.json).
