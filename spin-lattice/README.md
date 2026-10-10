# spin-lattice

`spin-lattice` is a dependency-free reference kernel for coupled classical
atomistic spins and harmonic lattice motion. A single checked Hamiltonian
produces the potential energy, atom forces, and effective spin fields, keeping
magnetoelastic signs and derivatives consistent.

For a bond from atom `i` to atom `j`, the current bond vector is
`r_ij = r_ij^0 + u_j - u_i` and the Hamiltonian contribution is

```text
H_ij = k_ij / 2 (|r_ij| - |r_ij^0|)^2
       - J_ij(|r_ij|) s_i . s_j

J_ij(r) = J0_ij + dJdr_ij (r - |r_ij^0|).
```

The lattice kinetic energy is `sum_i |p_i|^2 / (2 m_i)`. Spins are
dimensionless unit vectors. The crate supplies velocity Verlet for lattice
motion, exact frozen-field Rodrigues rotations for spin precession, and a
palindromic half-spin/full-lattice/half-spin coupled step.

```rust
use spin_lattice::{Bond, LinearExchange, SpinLatticeModel, SpinLatticeState, Vec3};

let bond = Bond::new(
    0,
    1,
    Vec3::new(1.0, 0.0, 0.0),
    2.0,
    LinearExchange::new(1.0, -0.2),
);
let model = SpinLatticeModel::try_new(vec![1.0, 1.0], vec![bond])?;
let state = SpinLatticeState::stationary(
    &model,
    vec![Vec3::new(0.0, 0.0, 1.0), Vec3::new(1.0, 0.0, 0.0)],
)?;

let evaluation = model.evaluate(&state)?;
assert!(evaluation.energy().total().is_finite());
# Ok::<(), spin_lattice::SpinLatticeError>(())
```

## Numerical scope

This crate is deliberately narrower than a molecular-dynamics engine. It
does not provide neighbor lists, thermostats, constraints, electronic
structure, or long-range interactions. It is intended as an auditable model,
a finite-difference validation target, and a base for domain-specific
magnetoelastic experiments.

The spin step solves precession in fields frozen at the beginning of each
substep. It exactly preserves each spin norm but is not an exact many-spin
flow. The coupled step uses palindromic ordering: half spin, full lattice,
half spin. Because the interacting-spin substep freezes all effective fields at
its start, that substep is first-order and is not self-adjoint. Consequently,
the coupled composition is not generally time-reversible or a second-order
Strang integrator despite its palindromic ordering. Time, energy, and
gyromagnetic units are chosen consistently by the caller.

## License

MIT.
