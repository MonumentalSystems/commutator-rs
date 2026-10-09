# clifford-lattice

`clifford-lattice` provides portable, safe reference kernels for periodic
lattice experiments in Rust:

- checked row-major `PeriodicLattice2<T>` storage;
- a generic sequential Metropolis sweep with explicit interaction and proposal
  policies;
- pure-f64 Cl⁺(6,0) even-multivector products;
- checked Spin(6) rotors with robust bivector exponentiation;
- full-rotor scalar interactions and caller-configured Gaussian bivector
  proposals.

```rust
use clifford_lattice::cl6::{
    GaussianBivectorProposal, RotorScalarInteraction, Spin6Rotor,
};
use clifford_lattice::lattice::PeriodicLattice2;
use clifford_lattice::metropolis::sequential_metropolis_sweep;
use clifford_lattice::rng::SplitMix64;

let mut lattice = PeriodicLattice2::filled(4, 4, Spin6Rotor::identity())?;
let proposal = GaussianBivectorProposal::canonical(0.1)?;
let mut random = SplitMix64::new(42);
let stats = sequential_metropolis_sweep(
    &mut lattice,
    0.5, // dimensionless beta * coupling
    &RotorScalarInteraction,
    &proposal,
    &mut random,
)?;
assert_eq!(stats.attempted(), 16);
# Ok::<(), clifford_lattice::LatticeError>(())
```

## Numerical contract

The total action counts each periodic `+x` and `+y` bond once. Sequential
sweeps visit sites in row-major order and see accepted updates from earlier
sites in the same sweep. The coupling parameter is explicitly dimensionless
`βJ`; the crate does not choose a temperature or experiment-specific default.

Cl(6,0) proposals take an explicit ordered bivector basis. Equal Gaussian
coefficients are isotropic only when those directions are orthonormal. This
keeps Commutator's eight-direction CPU policy distinct from its weighted
fifteen-direction GPU policy.

Public rotor imports use a fixed invariant tolerance and verify that sandwich
action maps the six vector basis elements to a proper orthonormal vector
frame. Checking `R reverse(R) = 1` alone is not sufficient for Spin(6)
membership.

The initial release intentionally excludes checkerboard parallelism,
deterministic rotor-flow integration, and Langevin dynamics. The source CPU
and GPU implementations do not yet share a single validated physical kernel;
see [`PROVENANCE.md`](PROVENANCE.md).

## Boundary

The crate contains no Rayon, Serde, GPU backend, neural-network runtime,
experiment schema, networking, identity, Merkle, database, or scheduler code.
Its only runtime dependency is `clifford-core`.

## License

MIT.
