# Distributed science

Distribution is a deployment choice, not a property of the mathematics.
`experiment-core` makes the boundary explicit so the same experiment can run
in-process, on one accelerated workstation, or through a distributed host.

A distributed adapter can map the neutral contracts onto:

- domain-decomposed tiles with explicit halo dependencies;
- independent replicas for parameter scans or parallel tempering;
- map/reduce work with a deterministic reducer;
- capability-aware worker selection;
- Merkle commitments, challenges, or redundant execution;
- checkpoints and generation-aware retries;
- signed source and release manifests.

The adapter owns transport, authentication, persistence, scheduling, and
reputation. The experiment owns configuration, deterministic seeds, numerical
work, validation policy, and reproducibility metadata. This separation makes
it possible to test the scientific core without starting a network and to
replace infrastructure without rewriting the experiment.

