# cluster-green

`cluster-green` is a lightweight reference layer for cluster perturbation
theory (CPT). It starts from an already-computed cluster one-particle Green
function; it does not duplicate exact diagonalization or impurity solvers.

The crate provides:

- checked retarded Green-function grids at `z = omega + i eta`, `eta > 0`;
- small dense complex matrices with pivoted inversion;
- matrix-valued causality checks using the positive-semidefinite spectral
  matrix `(G† - G)/(2i)`;
- Dyson/CPT embedding `G(k,z) = [G_c(z)^-1 - V(k)]^-1`;
- trace spectral functions and optional position-aware Green-function
  periodization.

```rust
use cluster_green::{ClusterGreenGrid, DenseMatrix, GreenPoint};
use num_complex::Complex64;

let z = Complex64::new(0.2, 0.05);
let cluster = ClusterGreenGrid::try_new(
    1,
    vec![GreenPoint::new(z, DenseMatrix::from_scalar(Complex64::new(1.0, 0.0) / z)?)],
)?;
let embedded = cluster.cpt_at(&DenseMatrix::from_scalar(0.3.into())?)?;
assert!(embedded.spectral_function(0)? > 0.0);
# Ok::<(), cluster_green::GreenError>(())
```

## Scope and conventions

Matrices use row-major site order. Frequency points must have strictly
increasing real parts and positive broadening. `V(k)` must be Hermitian. The
spectral function is `A(k,omega) = -Im Tr G(k,z)/(pi N)`.

Periodization uses
`G_periodized(k,z) = sum_ab exp(-i k·(r_a-r_b)) G_ab(k,z) / N`.
Explicit site positions make the phase and normalization convention visible.

Causality tolerance is relative to the spectral Hermitian matrix itself, not
to the full magnitude of `G`; a large dispersive real part therefore cannot
hide a wrong-sign imaginary part. Dense inversion first normalizes the input,
so singularity decisions are invariant under uniform rescaling.

This is deliberately a small dense reference implementation. Production
solvers should use optimized linear algebra while retaining the same checked
boundary conventions.

## License

MIT.
