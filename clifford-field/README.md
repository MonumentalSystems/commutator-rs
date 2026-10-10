# clifford-field

Reusable Clifford field storage and portable CPU simulation kernels in Rust.

This first extraction layer provides:

- `FieldPrecision`, the runtime f32/f64 selector;
- `FieldScalar`, the sealed, monomorphized simulation scalar contract;
- stable little-endian serialization and bit-conversion behavior for f32/f64;
- `StaBivector` and one-, two-, or three-dimensional `BivectorField` storage;
- checked, transactional harmonic-wave, tanh-interface, and cored-vortex
  initial conditions in physical grid coordinates;
- periodic, fixed, and free boundary conditions;
- gradient, Laplacian, raw STA commutator, energy, and chiral observables;
- portable reference Euler and three-pass Strang steppers;
- portable correlation, vortex-core, winding-map spectrum, and Wilson/Creutz
  post-processing;
- a configurable Rayon crossover for larger CPU fields.

It depends on [`clifford-core`](https://crates.io/crates/clifford-core) for the
open `CliffordScalar` algebra contract. Specialized fused SIMD and GPU backends
remain in their host crates until differential trajectory tests cover those
execution paths. Distributed scheduling, Merkle verification, identity, and
experiment orchestration are intentionally out of scope.

```rust
use clifford_field::{
    step_strang_reference, BivectorField, BoundaryCondition, StaBivector,
};

let mut field = BivectorField::new_1d(64, 0.1_f64, 1.0, BoundaryCondition::Periodic);
field.data[0] = StaBivector::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
step_strang_reference(&mut field, 0.001);
assert!(field.total_energy().is_finite());
```

Analytic initializers write only field data and leave geometry, boundary
selection, units, solver choice, and timestep policy to the caller. They
validate the selected boundary where the topology requires it:

```rust
use clifford_field::{BivectorField, BoundaryCondition, StaBivector};
use clifford_field::initial_conditions::HarmonicWave1d;

let mut field = BivectorField::new_1d(64, 1.0 / 64.0, 1.0, BoundaryCondition::Periodic);
let wave = HarmonicWave1d::try_new(
    StaBivector::new(0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
    StaBivector::new(0.1, 0.0, 0.0, 0.0, 0.3, 0.0),
    StaBivector::new(0.0, -0.1, 0.0, 0.3, 0.0, 0.0),
    std::f64::consts::TAU,
    0.0,
    0.0,
)?;
wave.apply(&mut field)?;
# Ok::<(), clifford_field::initial_conditions::InitialConditionError>(())
```

Formulas are evaluated in the field's `f32` or `f64` scalar type. They are
deterministic for a given build and platform, but transcendental results are
not promised to be bit-identical across precisions or hardware. See
[`PROVENANCE.md`](PROVENANCE.md) for the extraction boundary and source-history
anchors.

Analysis is intentionally array-based, so experiment workers can process
results without depending on a distributed runtime:

```rust
use clifford_field::analysis::{correlation_raw_f64, extract_xi_single};

let samples = vec![1.0_f64; 8 * 8 * 6];
let correlation = correlation_raw_f64(&samples, 8, 8, 4, 6);
let xi = extract_xi_single(&correlation, 2);
assert_eq!(xi, 0.0); // a uniform field has no finite decay length
```

## License

MIT
