# commutator-rs

Reusable Rust foundations for Clifford algebra, field simulation, and
reproducible computational experiments.

This repository separates the mathematics of an experiment from the machinery
used to distribute it. Each crate is useful on one machine. Together, they
provide clean boundaries for simulations that may later grow across CPU, GPU,
or volunteer-computing backends.

## Crates

| Crate | Purpose | Runtime dependencies |
| --- | --- | --- |
| [`clifford-core`](clifford-core) | Runtime-signature `Cl(p,q)`, geometric products, grades, rotors, and frozen cross-implementation conventions | none |
| [`clifford-field`](clifford-field) | Cl(1,3) field storage, checked analytic initial conditions, boundary operators, reference steppers, and portable numerical analysis | `clifford-core`, Rayon, Serde |
| [`clifford-geometry`](clifford-geometry) | Fixed-size EGA, PGA, CGA, STA, and Cl(6,0) geometry with motors, conformal primitives, frames, and kinematic chains | `clifford-core` |
| [`clifford-lattice`](clifford-lattice) | Checked Spin(6) rotors, periodic lattice storage, and a portable sequential Metropolis reference kernel | `clifford-core` |
| [`clifford-layers`](clifford-layers) | Framework-neutral Clifford linear, convolution, normalization, and optional Fourier layers | `clifford-core`, optional RustFFT |
| [`clifford-mesh`](clifford-mesh) | Checked renderer-independent triangle, line, and point topology with Euclidean primitive tessellators | none |
| [`experiment-core`](experiment-core) | Transport-neutral work, result, topology, verification, checkpoint-reference, and reproducibility contracts | Serde |
| [`experiment-merkle`](experiment-merkle) | Context-bound SHA-256 commitments, inclusion proofs, and deterministic post-commitment spot checks | `experiment-core`, Serde, SHA-2 |
| [`harmonic-dynamics`](harmonic-dynamics) | Safe `S¹`/Kuramoto, quaternion and sphere geometry, Lohe synchronization, gated scans, and Helmholtz sequence fibers | none |

The foundational crates deliberately do not contain HTTP, databases, identity,
scheduling, or a particular scientific model. Those capabilities belong in
host runtimes and backend adapters.

## Quick start

```rust
use clifford_core::CliffordAlgebra;

let algebra = CliffordAlgebra::cl3();
let e1 = algebra.basis::<f64>(1);
let e2 = algebra.basis::<f64>(2);
let e12 = algebra.geometric_product(&e1, &e2);
assert_eq!(algebra.grade_project(&e12, 2), e12);
```

```rust
use clifford_field::{
    step_strang_reference, BivectorField, BoundaryCondition, StaBivector,
};

let mut field = BivectorField::new_1d(
    64,
    0.1_f64,
    1.0,
    BoundaryCondition::Periodic,
);
field.data[0] = StaBivector::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
step_strang_reference(&mut field, 0.001);
assert!(field.total_energy().is_finite());
```

Run the complete workspace locally:

```bash
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings -D missing-docs" cargo doc --workspace --no-deps
```

## Design

The dependency direction is intentionally one-way:

```text
clifford-core
├── clifford-field
├── clifford-geometry
├── clifford-lattice
└── clifford-layers

clifford-mesh
(independent geometry buffers and tessellation)

experiment-core                 harmonic-dynamics
      └── experiment-merkle      (independent numerical dynamics)
                 ^                           ^
                 +-------------+-------------+
                               |
        host runtimes, optimized backends, and distributed schedulers
```

See [Architecture](docs/architecture.md) for the crate boundaries and
[Distributed science](docs/distributed-science.md) for the adapter model.
The algebra conventions and golden vectors are versioned in
[`clifford-core/CONVENTIONS.md`](clifford-core/CONVENTIONS.md).

## Status

The crates are pre-1.0 and their APIs are being prepared for their first
crates.io releases. The portable CPU implementations and conformance fixtures
are the reference surface; optimized backends remain host-side adapters until
their differential tests are equally strong.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before proposing an API or numerical
change. Security reports should follow [SECURITY.md](SECURITY.md).

The workspace is licensed under the [MIT License](LICENSE), except
`clifford-geometry` and `clifford-mesh`, which preserve Versor's BSD-2-Clause
license and attribution.

