# quantum-transport

`quantum-transport` provides small, checked reference kernels for coherent
nonequilibrium Green-function (NEGF) and Landauer calculations. It reuses
`cluster-green`'s `DenseMatrix` and the ecosystem-standard
`num_complex::Complex64` type re-exported there.

The crate includes:

- retarded device Green functions
  `G^R(E) = [(E+i eta)I - H - sum_l Sigma_l^R(E)]^-1`;
- causal lead broadening matrices `Gamma_l = i(Sigma_l-Sigma_l†)`;
- Caroli transmission `Tr[Gamma_L G^R Gamma_R G^A]`;
- stable finite- and zero-temperature Fermi functions;
- trapezoidal Landauer integrals and currents in amperes for electron-volt
  energy grids.

```rust
use cluster_green::{Complex64, DenseMatrix};
use quantum_transport::two_terminal_transmission;

let h = DenseMatrix::from_scalar(0.0.into())?;
let left = DenseMatrix::from_scalar(Complex64::new(0.0, -0.25))?;
let right = DenseMatrix::from_scalar(Complex64::new(0.0, -0.25))?;
let point = two_terminal_transmission(0.0, 1.0e-9, &h, &left, &right, 1.0e-12)?;
assert!((point.transmission - 1.0).abs() < 1.0e-7);
# Ok::<(), quantum_transport::TransportError>(())
```

## Numerical conventions

Hamiltonians and broadening matrices must be Hermitian. Retarded lead
self-energies must give positive-semidefinite `Gamma`. A positive `eta` is
required even when physical lead broadening already makes the inverse regular.
Transmission is accepted only when its residual imaginary part and negative
roundoff are within the caller's scale-aware tolerance.
Positive-semidefinite broadening checks scale only with `Gamma` itself and do
not impose an absolute unit-scale floor, including for very weak couplings.

`landauer_integral` returns the energy integral in the same units as its grid.
`landauer_current_ev` interprets all energies as electron-volts and returns
amperes using `I = g e^2/h integral`, where `g` is the supplied degeneracy.

This is a dense small-system reference implementation. Recursive Green
functions, sparse solvers, inelastic self-energies, and self-consistent lesser
Green functions are intentionally outside its initial scope.

## License

MIT.
