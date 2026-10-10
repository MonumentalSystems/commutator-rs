# majorana-fermions

`majorana-fermions` is a sparse, deterministic operator toolkit for Majorana
and canonical complex fermions. It provides exact Clifford product signs,
complex sparse operators, canonical anticommutation-relation constructors,
quadratic Hamiltonians, and an open Kitaev-chain model.

```rust
use majorana_fermions::{annihilation, creation, number_operator, FermionOperator};

let c = annihilation(2)?;
let c_dagger = creation(2)?;
assert_eq!(c.anticommutator(&c_dagger)?, FermionOperator::identity());

let number = number_operator(2)?;
assert_eq!(number.multiply(&number)?, number);
# Ok::<(), majorana_fermions::FermionError>(())
```

## Representation

Majorana generators obey

```text
gamma_j gamma_k + gamma_k gamma_j = 2 delta_jk.
```

A monomial stores a sorted `Vec<usize>` of distinct generator indices rather
than a fixed-width bitmap. Products merge those sparse index sets and count
the exact permutation parity. Consequently, the algebra is not limited to the
small dimensions practical for a dense `2^n` Clifford multiplication table.

`FermionOperator` stores nonzero `num_complex::Complex64` coefficients in a
`BTreeMap`, giving reproducible iteration and accumulation order. Arithmetic
is checked for non-finite coefficients. Constructors use

```text
c_j       = (gamma_(2j) + i gamma_(2j+1)) / 2
c_j^dagger = (gamma_(2j) - i gamma_(2j+1)) / 2.
```

The resulting creation, annihilation, number, and parity operators satisfy
the canonical anticommutation relations exactly in floating-point arithmetic.

## Quadratic Hamiltonians

`QuadraticMajoranaHamiltonian` represents

```text
H = constant + i sum_(j<k) K_jk gamma_j gamma_k
```

with sparse real couplings. `kitaev_chain` constructs an open real-parameter
chain and exposes the unpaired edge Majoranas at `mu = 0` and `pairing =
hopping` directly in its coupling graph.

The sparse core is independent of `clifford-core`. A development-only
conformance test exhaustively compares every product in `Cl(5,0)` against
`clifford-core`'s dense Cayley table, keeping the sign conventions aligned
without imposing that implementation's deliberate small-rank bound.

## Scope

This crate manipulates symbolic sparse operators. It does not choose a Fock
basis, allocate `2^L` state vectors, diagonalize Hamiltonians, perform
Jordan-Wigner matrix construction, or solve interacting many-body systems.
Those operations belong in solver crates layered above this algebra.

## License

MIT.
