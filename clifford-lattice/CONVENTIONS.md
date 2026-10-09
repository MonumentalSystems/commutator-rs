# Numerical conventions

`clifford-lattice` treats these choices as part of its public compatibility
contract.

## Cl(6,0) layout

Full multivectors use the `clifford-core` ShortLex blade order. The compact
even-subalgebra layout contains, in order:

1. the scalar;
2. the 15 bivectors;
3. the 15 grade-four blades;
4. the pseudoscalar.

`Spin6Rotor::exp_bivector(B)` computes the literal positive-sign exponential
`exp(B)`. Callers using the common geometric-rotation convention `exp(-B/2)`
must apply that sign and scale before calling it. A rotor's grade-two
coefficients are not presented as a logarithm map.

## Periodic lattice

Two-dimensional sites use `index = x * height + y`. Neighbor order is
`(+x, -x, +y, -y)`. Total action counts the `+x` and `+y` bonds from every
site exactly once:

```text
S = -beta_coupling * sum(score(site, positive_neighbor))
```

The reference Metropolis sweep visits sites in ascending flat-index order.
Each proposal therefore observes accepted changes from earlier sites in the
same sweep.

## Proposal coordinates

Gaussian proposals preserve the caller-supplied direction vectors and their
norms. Equal coefficient variance is isotropic only for an orthonormal basis.
The crate does not silently equate the different proposal bases used by the
source CPU and GPU experiments.
