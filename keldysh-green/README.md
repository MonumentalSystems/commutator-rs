# keldysh-green

`keldysh-green` is a checked, dense Rust reference layer for real-time
nonequilibrium Green functions. It complements `cluster-green` by reusing its
`DenseMatrix` and `Complex64` primitives instead of defining a second matrix
stack.

The crate provides:

- strictly increasing, possibly nonuniform real-time grids with
  composite-trapezoid weights;
- matrix-valued retarded, advanced, lesser, and greater components;
- diagnostics for causal support, adjoint/anti-Hermitian symmetries, and
  `G^R-G^A = G^>-G^<`;
- full time convolution and causal Volterra convolution;
- the real-time Langreth rules for contour products;
- a discretized retarded Dyson/Volterra reference solver;
- equal-time density, one-body expectation, particle number, and explicitly
  oriented bond-flow helpers.

```rust
use keldysh_green::{Complex64, DenseMatrix, RealTimeGrid, TwoTimeMatrix,
                    retarded_dyson};

let grid = RealTimeGrid::try_new(vec![0.0, 0.5, 1.0])?;
let free = TwoTimeMatrix::try_from_fn(grid.len(), 1, |i, j| {
    DenseMatrix::from_scalar(if i >= j {
        Complex64::new(0.0, -1.0)
    } else {
        Complex64::new(0.0, 0.0)
    }).map_err(Into::into)
})?;
let zero_sigma = TwoTimeMatrix::try_from_fn(grid.len(), 1, |_, _| {
    DenseMatrix::from_scalar(0.0.into()).map_err(Into::into)
})?;
assert_eq!(retarded_dyson(&grid, &free, &zero_sigma)?, free);
# Ok::<(), keldysh_green::KeldyshError>(())
```

## Conventions and scope

The fermionic definitions are
`G^<(t,t') = i <c†(t')c(t)>` and
`G^>(t,t') = -i <c(t)c†(t')>`. Therefore
`rho(t) = -i G^<(t,t)`. `bond_flow_into` returns the contribution flowing into
orbital `a` from `b`,
`2(q/hbar) Im[h_ab rho_ba]`; its orientation and physical prefactor are never
implicit.

This is a transparent reference implementation. Storing a component costs
`O(N_t^2 n^2)` and direct dense convolution/Dyson work costs
`O(N_t^3 n^3)` for `N_t` time points and `n` orbitals. It is suitable for
validation and modest grids, not production-scale propagation. Optimized
backends can preserve these checked boundary conventions.

The crate does not reimplement sparse intermediate-representation or DLR
bases. Those representations should be supplied through future adapters to
specialized libraries. Imaginary-branch initial-correlation terms, collision
self-energy approximations, and conserving self-consistency loops are also
outside this first real-time reference layer.

## License

MIT.
