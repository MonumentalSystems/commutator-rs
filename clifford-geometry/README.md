# clifford-geometry

Sparse, fixed-size geometric-algebra types and geometry operations extracted
from [`versor-rs`](https://github.com/DavinciDreams/versor-rs) at commit
`93a5a1334ab77131e5ef6b88105cdaac57d267e6`.

The crate contains:

- bitmap blade and product-table primitives;
- Euclidean 3D, projective 3D, conformal 3D, and spacetime algebras;
- dense compatibility types for Cl(3,0) and Cl(6,0);
- bivector exponentials and rotors;
- conformal frames, shapes, kinematic chains, twists, interpolation, and
  three-dimensional scalar/vector fields;
- checked decomposition of conformal points and real dual spheres into
  Euclidean parameters, with homogeneous normalization when scaled `f32`
  coefficients retain the represented semantics;
- explicit adapters to the dense, precision-generic `clifford-core` engine.

It deliberately excludes lattice dynamics, simulation dispatch, SIMD, GPU
backends, mesh generation, and research binaries.

```rust
use clifford_geometry::cga3d::{decomposition::decompose_point, point};

let p = point(1.0, 2.0, 3.0);
assert_eq!(decompose_point(&p)?.position(), [1.0, 2.0, 3.0]);
# Ok::<(), clifford_geometry::cga3d::decomposition::DecompositionError>(())
```

For input that crosses an API or storage boundary, prefer the checked semantic
decomposition functions. They reject zero-weight, non-finite, non-null, and
imaginary-radius values rather than applying a silent fallback:

```rust
use clifford_geometry::cga3d::{
    decomposition::decompose_real_dual_sphere,
    point,
    Round,
};

let sphere = Round::dls(&point(1.0, 2.0, 3.0), 2.5);
let parameters = decompose_real_dual_sphere(&sphere)?;
assert_eq!(parameters.center(), [1.0, 2.0, 3.0]);
assert!((parameters.radius() - 2.5).abs() < 1e-5);
# Ok::<(), clifford_geometry::cga3d::decomposition::DecompositionError>(())
```

The older `Round::location`, `Round::center`, and `Round::radius_squared`
helpers remain available for source compatibility, but are unchecked and keep
their historical degenerate-value fallbacks.

## Compatibility conventions

This crate preserves Versor's established formulas and storage layouts. Its
commutators return raw `AB - BA`, while `clifford-core` returns
`(AB - BA) / 2`. Versor rotors use `exp(-B/2)`, which has the opposite sign
from `clifford-core::CliffordAlgebra::rotor_from_bivector`. The
[`adapter`](https://docs.rs/clifford-geometry/latest/clifford_geometry/adapter/)
module makes those differences explicit.

The named sparse types remain aliases of `Multivector<N>`. For example, two
different three-component aliases are not nominally distinct Rust types.
Replacing them with semantic newtypes is deferred because it would be a broad
API and representation change; this extraction intentionally preserves the
validated Versor behavior.

True projective Cl(3,0,1) is implemented locally with a degenerate metric.
`clifford-core` currently supports only non-degenerate signatures, so no PGA
adapter is exposed.

## License and attribution

BSD-2-Clause, preserving the Versor Contributors copyright and attribution.
