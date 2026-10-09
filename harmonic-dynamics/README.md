# harmonic-dynamics

`harmonic-dynamics` is a dependency-free collection of safe numerical kernels
for unit-quaternion geometry, spherical interpolation, Lohe mean-field
synchronization, gated sequence scans, and multi-head Helmholtz fibers.

The crate deliberately contains no neural-network framework integration,
model configuration, diagnostics, unsafe code, or platform-specific SIMD.
Public operations validate shapes and numerical domains and return
[`DynamicsError`](https://docs.rs/harmonic-dynamics/latest/harmonic_dynamics/enum.DynamicsError.html)
instead of relying on indexing assertions.

```rust
use harmonic_dynamics::quaternion::Quaternion;

let z = Quaternion::from_axis_angle(
    core::f32::consts::FRAC_PI_2,
    [0.0, 0.0, 1.0],
)?;
assert!((z.norm() - 1.0).abs() < 1e-6);
# Ok::<(), harmonic_dynamics::DynamicsError>(())
```

## Modules

- `quaternion`: typed Hamilton algebra and maps on `S^3`
- `sphere`: dimension-checked operations on `S^(d-1)`
- `lohe`: real, quaternion, and phase-shifted Lohe synchronization
- `sequence`: gated scans and fixed/adaptive Helmholtz fibers

## Numerical policy

This crate treats the extracted formulas as a new, validated reference API,
not as bit-for-bit compatibility wrappers. Lohe updates retain the magnitude
of the mean field, apply coupling strengths consistently in phase-shifted
paths, and use exponential-map sphere steps. These choices intentionally
differ from HarmonicRust paths that normalize the mean direction or use a
first-order Euler update. Stable direct recurrences likewise replace the
source's inverse-power scan formulas.

## Provenance

The numerical formulas were extracted and reworked from HarmonicRust commit
`c20fd04956f987f9a00d53c78728d8069f0a1589`. See
`THIRD_PARTY_NOTICES.md` for licensing provenance and adaptation details.

## License

MIT.
