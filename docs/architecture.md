# Architecture

`commutator-rs` is organized around one rule: scientific definitions should
not depend on the infrastructure that executes them.

## Algebra layer

`clifford-core` owns runtime signatures, blade ordering, products, reverse,
grade projection, sandwich products, rotor construction, and the open scalar
contract. It has no runtime dependencies.

## Field layer

`clifford-field` owns precision-selected Cl(1,3) bivectors, regular-grid
storage, boundary conditions, portable spatial operators, reference Euler and
Strang steppers, and array-based analysis. It depends inward on
`clifford-core`; it does not know about workers, networks, or experiments.

## Experiment layer

`experiment-core` owns typed descriptions of work and results, topology,
verification policy, checkpoints, worker capabilities, and run metadata. It
does not choose an async runtime, wire format, database, or identity system.

## Host adapters

A host may connect those layers to SIMD, Metal, CUDA, Accelerate, WebGPU,
HTTP, databases, Merkle commitments, or volunteer-worker scheduling. Those
adapters remain outside the foundational crates so local experiments do not
inherit operational dependencies.

