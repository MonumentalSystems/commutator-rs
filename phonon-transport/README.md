# phonon-transport

Small, checked reference kernels for one-dimensional harmonic lattice dynamics
and ballistic phonon heat transport. A unit cell may contain multiple scalar
displacement degrees of freedom with arbitrary inter-cell harmonic bonds.

The crate provides:

- mass-weighted complex Hermitian dynamical matrices;
- acoustic and optical phonon frequencies across the Brillouin zone;
- eigenvector-tracked finite-difference group velocities (exact degeneracies
  are reported rather than assigned basis-dependent branch velocities);
- harmonic heat capacity and ballistic Landauer thermal conductance;
- a dependency-light Hermitian Jacobi eigensolver suitable for validation.

```rust
use phonon_transport::{HarmonicBond, HarmonicChain};

let chain = HarmonicChain::try_new(
    1.0,
    vec![1.0],
    vec![HarmonicBond::try_new(0, 0, 1, 1.0)?],
)?;
let omega = chain.frequencies(core::f64::consts::PI, 1e-12)?[0];
assert!((omega - 2.0).abs() < 1e-10);
# Ok::<(), phonon_transport::PhononError>(())
```

All frequencies are angular frequencies. Inputs are SI-compatible when masses,
force constants, lattice spacing, wavevectors, and temperature are supplied in
SI units, but the eigensystem itself is equally useful in consistent reduced
units.

This is a harmonic reference model, not a density-functional force-constant
parser or anharmonic Boltzmann solver. Those tools can export reduced bond
models into this crate for regression tests.
