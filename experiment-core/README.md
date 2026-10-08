# experiment-core

Transport-neutral contracts for reproducible computational experiments.

This crate defines typed experiment configuration, work, result, status, and
error boundaries together with topology, verification, worker capability,
seed, and run-metadata types. It intentionally contains no HTTP, async runtime,
database, identity, reputation, blob-store, or physics dependencies.

A host runtime can implement these contracts using an in-process loop, a
cluster scheduler, or a volunteer network without changing the experiment's
typed configuration and results.

```rust
use experiment_core::{ExperimentTopology, RunMetadata, WorkUnit};

let work = WorkUnit {
    experiment_id: "ising-scan".into(),
    instance_id: "run-001".into(),
    unit_id: "temperature-2.2".into(),
    generation: 0,
    run: RunMetadata { seed: 42, schema_version: 1, ..Default::default() },
    payload: vec![2.2_f64],
};
let topology = ExperimentTopology::IndependentReplica;
assert_eq!(topology.tag(), "independent_replica");
assert_eq!(work.run.seed, 42);
```

## License

MIT

