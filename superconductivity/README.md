# superconductivity

`superconductivity` provides small, checked building blocks for lattice
Bogoliubov-de Gennes (BdG) calculations. It constructs a dense spinful BdG
Hamiltonian from any Hermitian tight-binding normal Hamiltonian and onsite
spin-singlet gaps, validates particle-hole symmetry, validates externally
computed eigenpairs, and evaluates local density of states and anomalous
pairing amplitudes.

The crate intentionally does not select a dense or sparse eigensolver. This
keeps its numerical model interoperable with existing Rust linear-algebra
ecosystems and lets callers choose a backend appropriate to their lattice.
It has no dependencies and supports Rust 1.80.

```rust
use superconductivity::{Complex64, NormalHamiltonian, OnsiteSWaveModel};

let normal = NormalHamiltonian::spin_independent(&[0.0], &[])?;
let model = OnsiteSWaveModel::try_new(normal, 0.25, vec![Complex64::new(0.4, 0.0)])?;
let bdg = model.hamiltonian();

assert_eq!(bdg.dimension(), 4); // particle/hole x spin x one site
assert!(bdg.is_hermitian(1e-12));
assert!(bdg.particle_hole_residual() < 1e-12);
# Ok::<(), superconductivity::SuperconductivityError>(())
```

## Conventions

Single-particle orbitals are ordered by site, then spin: `(0 up, 0 down,
1 up, 1 down, ...)`. The Nambu basis contains all particle orbitals followed
by all hole orbitals:

`(c_up, c_down, ..., c_up^dagger, c_down^dagger, ...)`.

For `h_mu = h - mu I`, the constructed matrix is

```text
H_BdG = [ h_mu       Delta  ]
        [ Delta^dag  -h_mu^T]
```

with `Delta_(site up, site down) = gap` and
`Delta_(site down, site up) = -gap`. This explicitly enforces the fermionic
antisymmetry of onsite spin-singlet pairing and gives particle-hole operator
`C = tau_x K`.

## Scope

This first release is a reference model for finite lattices and for testing
larger matrix-free solvers. It does not yet perform self-consistent gap
iteration, sparse assembly, time evolution, or electromagnetic gauge-field
coupling.

## License

MIT.
