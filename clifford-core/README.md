# clifford-core

Precision-generic Clifford algebra primitives for Rust.

The crate provides:

- a runtime-signature `Cl(p,q)` algebra with ShortLex blade ordering;
- geometric product, commutator, reverse, grade projection, sandwich product,
  and rotor construction;
- predefined complex, quaternion, Cl(3,0), Cl(3,1), STA, and Cl(6,0)
  signatures;
- an open `CliffordScalar` algebra trait, with built-in `f32` and `f64`
  implementations.

It deliberately contains no distributed-computing, persistence, GPU, or
application-specific experiment dependencies.

The precise blade ordering, metric ordering, commutator normalization, rotor
orientation, and cross-project adapter rules are frozen in
[`CONVENTIONS.md`](CONVENTIONS.md). Machine-readable reference cases live in
[`tests/fixtures/clifford-golden-v1.json`](tests/fixtures/clifford-golden-v1.json).

## Runtime model

This is a dense, standard-library implementation. Constructing `Cl(p,q)`
precomputes a table with `4^(p+q)` entries, and dense geometric products have
the same worst-case operation count. The supported vector-space dimension is
therefore capped at 10. Use `CliffordAlgebra::try_new` when the signature is
not a compile-time choice.

Operations returning multivectors allocate `Vec`s. Invalid dense lengths and
blade indices panic; this makes shape mistakes fail immediately rather than
silently corrupting a calculation. Specialized fixed-size engines remain the
right choice for simulation hot loops.

```rust
use clifford_core::CliffordAlgebra;

let algebra = CliffordAlgebra::cl3();
let e1 = algebra.basis::<f64>(1);
let e2 = algebra.basis::<f64>(2);
let e12 = algebra.geometric_product(&e1, &e2);

assert_eq!(algebra.grade_project(&e12, 2), e12);
```

## Composition

`clifford-field` builds its sealed simulation-scalar contract on this crate.
Host crates can layer specialized field storage and accelerated kernels on the
same algebra conventions without adding those concerns to this core.

## License

MIT

