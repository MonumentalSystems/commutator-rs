# cluster-embedding

`cluster-embedding` is a checked, solver-agnostic foundation for variational
cluster approximation (VCA) and dynamical mean-field theory (DMFT). It accepts
cluster Green functions and self-energies produced by an external exact
diagonalization, tensor-network, Monte Carlo, or other impurity solver. It does
not implement another many-body eigensolver.

The crate provides:

- checked Hubbard and reference-system one-body parameter types;
- CPT-compatible lattice embedding from either `cluster-green` data or a
  supplied self-energy;
- Potthoff self-energy-functional quadrature with explicitly weighted real
  frequencies and momenta;
- continuous log-determinant phase unwrapping and a reported imaginary
  residual rather than silently discarding branch errors;
- finite-difference VCA stationarity diagnostics and a conservative
  trust-region search;
- retarded DMFT Weiss-field/Dyson updates, causal hybridization validation,
  linear mixing, convergence histories, and Bethe-lattice self-consistency;
- injected `ReferenceSolver` and `ImpuritySolver` traits.

## Conventions

All matrices are row-major in the orbital/site basis. Retarded frequencies are
`z = omega + i eta`, with `eta > 0`; retarded Green functions, self-energies,
and hybridizations use negative-semidefinite anti-Hermitian parts. Hopping
matrices contain the coefficients in the one-body Hamiltonian `H_0`; lattice
Green functions are

`G(k,z) = [(z + mu) I - h(k) - Sigma(z)]^-1`.

The VCA correction is evaluated as

`Omega = Omega' + integral (log det[-G] - log det[-G'])`,

using caller-supplied quadrature weights. The caller owns temperature/contour
normalizations. Phase is unwrapped independently along each ordered frequency
path. The first phase remains on the principal branch, making the unavoidable
additive branch convention explicit.

## Non-goals

This crate is not an exact-diagonalization package, a quantum Monte Carlo
solver, a production sparse linear-algebra backend, or a claim that a finite
real-frequency quadrature is thermodynamically converged. Solver accuracy,
tail corrections, symmetry breaking fields, and thermodynamic extrapolation
remain explicit responsibilities of the application.

## License

MIT.
