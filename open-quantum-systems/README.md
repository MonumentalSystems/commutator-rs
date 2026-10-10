# open-quantum-systems

Checked, dependency-light reference dynamics for finite-dimensional Markovian
open quantum systems. The crate implements the Gorini–Kossakowski–Sudarshan–
Lindblad equation with dense row-major complex operators and fixed-step RK4.

The intended role is a transparent validation kernel for small systems. It is
not an operator-construction framework: mature operator vocabularies such as
`struqture` can export their dense matrices through `Operator::try_new`.

```rust
use open_quantum_systems::{
    CollapseOperator, Complex64, DensityMatrix, LindbladModel, Operator,
};

let zero = Complex64::new(0.0, 0.0);
let one = Complex64::new(1.0, 0.0);
let h = Operator::try_new(2, vec![zero; 4])?;
let lowering = Operator::try_new(2, vec![zero, one, zero, zero])?;
let model = LindbladModel::try_new(h, vec![CollapseOperator::try_new(lowering, 1.0)?], 1e-12)?;
let excited = DensityMatrix::basis(2, 1)?;
let evolved = model.evolve_rk4(&excited, 0.001, 1_000, 1e-9)?;
assert!(evolved.get(1, 1).unwrap().re < 0.38);
# Ok::<(), open_quantum_systems::QuantumError>(())
```

## Numerical contract

- constructors reject non-finite, malformed, non-Hermitian, non-normalized, or
  non-positive density matrices;
- Hamiltonians are checked for Hermiticity and collapse rates for positivity;
- RK4 steps are re-Hermitized and normalized to control roundoff, then checked
  for positivity;
- `diagnostics` exposes trace, purity, Hermiticity residual, and the minimum
  eigenvalue from a complex Hermitian positivity check.

For large Hilbert spaces, sparse Liouvillians, tensor networks, or production
quantum trajectories, use a specialized simulator and keep this crate as a
small-system reference oracle.
