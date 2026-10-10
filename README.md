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
| [`clifford-mesh`](clifford-mesh) | Checked renderer-independent topology, Euclidean tessellators, and optional CGA adapters | optional `clifford-geometry` |
| [`experiment-core`](experiment-core) | Transport-neutral work, result, topology, verification, checkpoint-reference, and reproducibility contracts | Serde |
| [`experiment-accelerator`](experiment-accelerator) | Backend descriptors, deterministic partition plans, and differential qualification of CPU/GPU/distributed experiment adapters | `experiment-core`, Serde, SHA-2 |
| [`experiment-merkle`](experiment-merkle) | Context-bound SHA-256 commitments, inclusion proofs, and deterministic post-commitment spot checks | `experiment-core`, Serde, SHA-2 |
| [`harmonic-dynamics`](harmonic-dynamics) | Safe `S¹`/Kuramoto, quaternion and sphere geometry, Lohe synchronization, gated scans, and Clifford-compatible Helmholtz sequence fibers | none (Clifford conformance in dev tests) |
| [`field-lyapunov`](field-lyapunov) | Matrix-free top-k Lyapunov spectra, finite-time estimates, RK4 tangent flow, and chaos diagnostics | none |
| [`quantum-chaos`](quantum-chaos) | Level-spacing ratios, spectral form factors, number variance, and explicit unfolding policies | `num-complex` |
| [`quantum-magnetism`](quantum-magnetism) | Matrix-free finite spin-1/2 models, frustrated exchange, observables, and reference Lanczos | `num-complex` |
| [`spin-lattice`](spin-lattice) | Coupled harmonic lattice and distance-dependent Heisenberg dynamics from one checked Hamiltonian | none |
| [`phonon-transport`](phonon-transport) | Harmonic-chain bands, branch-tracked group velocities, heat capacity, and ballistic thermal conductance | `num-complex` |
| [`superconductivity`](superconductivity) | Spinful onsite s-wave BdG assembly, symmetry checks, LDOS, and pairing observables | `num-complex` |
| [`superconducting-dynamics`](superconducting-dynamics) | Gauge-covariant graph TDGL, energy-monotone reference evolution, flux, and Josephson observables | `num-complex` |
| [`majorana-fermions`](majorana-fermions) | Sparse Majorana/fermion operator algebra and quadratic Kitaev-chain Hamiltonians | `num-complex` |
| [`cluster-green`](cluster-green) | Checked cluster Green functions, causality, CPT/Dyson embedding, and periodization | `num-complex` |
| [`cluster-embedding`](cluster-embedding) | Solver-injected Hubbard reference systems, VCA stationarity, and retarded DMFT embedding | `cluster-green`, `num-complex` |
| [`keldysh-green`](keldysh-green) | Real-time Keldysh components, Langreth products, observables, and a dense retarded Dyson reference solver | `cluster-green` |
| [`quantum-transport`](quantum-transport) | Coherent NEGF device Green functions, broadenings, Caroli transmission, and Landauer current | `cluster-green` |
| [`open-quantum-systems`](open-quantum-systems) | Dense Lindblad reference dynamics, physical density matrices, and fixed-step RK4 | `num-complex` |
| [`quantum-light`](quantum-light) | Finite Fock states, passive optics, photon statistics, and reduced density matrices | `num-complex` |
| [`quantum-shadows`](quantum-shadows) | Local-Pauli classical-shadow estimators and robust uncertainty reductions | none |
| [`quantum-tomography`](quantum-tomography) | Local-Pauli state reconstruction, PTM/Choi channel diagnostics, and focused gate-set likelihood tools | `open-quantum-systems` |

[`physics-conformance`](physics-conformance) is a non-publishable workspace
crate. It exercises convention and interoperability boundaries across the
independently versioned physics crates using only their public APIs.

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
(independent by default; optional CGA adapters consume clifford-geometry)

experiment-core                    clifford-core
├── experiment-merkle              ├── harmonic-dynamics
└── experiment-accelerator         └── majorana-fermions
          ^
          |
host runtimes and hardware backends

field-lyapunov          quantum-chaos
      |                       |
      +------ dynamics -------+

quantum-magnetism      spin-lattice      phonon-transport

superconductivity ---- interoperability ---- superconducting-dynamics

cluster-green
├── cluster-embedding
├── keldysh-green
└── quantum-transport

open-quantum-systems ---- quantum-tomography

quantum-light           quantum-shadows

physics-conformance (workspace-only cross-crate tests)
```

See [Architecture](docs/architecture.md) for the crate boundaries and
[Distributed science](docs/distributed-science.md) for the adapter model.
See [Research suite](docs/research-suite.md) for scientific workflows,
interoperability boundaries, and deliberate non-goals.
See [Release and provenance](docs/releasing.md) for dependency-safe publication
waves, package verification, checksums, and citation metadata.
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

