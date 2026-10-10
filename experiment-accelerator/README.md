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
There is no dependency on CUDA, Metal, WebGPU, an async runtime, or a network
transport. Hardware crates implement `ComputeBackend` and use the same
validation harness in their own CI or startup qualification.
Accepted qualification reports are constructible only by the differential
check API and are bound to the serialized work unit, worker identity, backend
IDs, and implementation versions. They serialize for audit logs but cannot be
deserialized into an admission token from an untrusted worker.

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
declare that property honestly and publish differential tolerances appropriate
to their precision and reduction order.

Qualification reports are opaque in-process capabilities, not portable trust
certificates. They cannot be deserialized. A distributed host that persists or
transmits qualification must wrap the report in its own authenticated record
and recreate authorization from a trusted reference allowlist after restart.
Work binding uses domain-separated SHA-256 over recursively key-sorted canonical
JSON; payload types therefore need stable Serde value semantics.
