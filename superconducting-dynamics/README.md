# superconducting-dynamics

`superconducting-dynamics` provides checked, reference-quality dissipative
time-dependent Ginzburg--Landau (TDGL) dynamics and Josephson observables on
finite graphs. It is intended for reproducible small systems, validation of
accelerated solvers, and coupling to the `superconductivity` BdG crate.

```rust
use superconducting_dynamics::{Complex64, OrderParameter, SiteParameters,
    TdglIntegrator, TdglModel};

let local = SiteParameters::try_new(-1.0, 1.0, 1.0)?;
let model = TdglModel::uniform_rectangular(8, 8, local, 0.5)?;
let mut psi = OrderParameter::uniform(&model, Complex64::new(0.2, 0.1))?;
let integrator = TdglIntegrator::reference(0.1)?;
let report = integrator.step(&model, &mut psi)?;
assert!(report.energy_after <= report.energy_before + 1e-12);
# Ok::<(), superconducting_dynamics::DynamicsError>(())
```

## Physics and units

The implemented free energy is

```text
F = sum_i [alpha_i |psi_i|^2 + beta_i |psi_i|^4 / 2]
  + sum_(i->j) K_ij |psi_j - exp(i A_ij) psi_i|^2.
```

`A_ij = q_pair integral(A . dl) / hbar` is dimensionless. Under a local gauge
change, `psi_i' = exp(i chi_i) psi_i` and
`A_ij' = A_ij + chi_j - chi_i`. Link current is oriented from `i` to `j` and
uses `J_ij = -dF/dA_ij`. All energy-like coefficients must share one unit;
time is set by the site mobility in
`d psi_i/dt = -Gamma_i dF/dpsi_i*`.

The deterministic integrator uses forward-Euler trials with energy
backtracking. It is intentionally a stable reference, not a high-order or
high-throughput production solver. A failed step leaves the state unchanged.

## BdG interoperability

`TdglModel::scaled_pairing_gaps` maps a TDGL snapshot to a vector accepted by
`superconductivity::OnsiteSWaveModel::try_new`. This crate does not duplicate
BdG assembly, eigensolvers, or self-consistency, and the dependency-neutral
bridge keeps the two crates independently publishable.

## Current limits

- Scalar, dissipative model-A TDGL only; no inertial or charge-density mode.
- Static compact U(1) links; Maxwell-field evolution is outside this crate.
- Deterministic dynamics only; thermal noise requires a documented stochastic
  discretization and is deliberately not approximated here.
- Finite graphs with in-memory state; GPU and distributed backends can validate
  against this implementation.

## License

MIT.

