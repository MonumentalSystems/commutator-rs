# field-lyapunov

`field-lyapunov` provides safe, matrix-free Lyapunov diagnostics for
high-dimensional dynamical systems, lattices, and discretized PDEs.

The crate deliberately does not build or store a dense Jacobian. A model
supplies a vector field and Jacobian-vector product, or an existing simulator
implements the lower-level tangent-step interface. The Benettin estimator then
tracks only the requested leading tangent subspace.

```rust
use field_lyapunov::{estimate_spectrum, BenettinConfig, ContinuousSystem, Rk4};

struct Linear([f64; 3]);

impl ContinuousSystem for Linear {
    fn dimension(&self) -> usize { 3 }

    fn vector_field(
        &self,
        _time: f64,
        state: &[f64],
        output: &mut [f64],
    ) -> field_lyapunov::Result<()> {
        for ((out, x), rate) in output.iter_mut().zip(state).zip(self.0) {
            *out = rate * x;
        }
        Ok(())
    }

    fn jacobian_vector_product(
        &self,
        _time: f64,
        _state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> field_lyapunov::Result<()> {
        for ((out, v), rate) in output.iter_mut().zip(direction).zip(self.0) {
            *out = rate * v;
        }
        Ok(())
    }
}

let mut model = Rk4::new(Linear([0.3, 0.0, -0.2]));
let config = BenettinConfig::new(3, 0.01, 10, 500)?
    .with_transient_windows(25)
    .with_seed(42);
let result = estimate_spectrum(&mut model, &[1.0; 3], &config)?;
println!("{:?}", result.exponents());
# Ok::<(), field_lyapunov::LyapunovError>(())
```

## Features

- Matrix-free `ContinuousSystem` interface with explicit JVP evaluation
- `TangentStepper` interface for optimized CPU, GPU, or distributed solvers
- Reusable fourth-order Runge-Kutta state/tangent adapter
- Deterministically seeded top-*k* Benettin spectrum estimation
- Twice-applied modified Gram-Schmidt reorthogonalization
- Local finite-time and cumulative convergence estimates
- Largest-exponent convenience API
- Kaplan-Yorke dimension and positive-exponent entropy-rate helpers
- No runtime dependencies and no unsafe code

## Numerical scope

The estimator uses `O(kN)` tangent storage for state dimension `N` and `k`
requested exponents. The bundled RK4 adapter uses additional reusable
`O(kN)` work buffers. Implementing `TangentStepper` directly allows a field
solver to control integration, memory placement, and accelerator dispatch.

Reported finite-time values depend on the integration step, window length,
warm-up, initial tangent basis, and trajectory. Convergence should be checked
by varying each of them. Stochastic systems require a tangent evolution that
uses the same noise realization as the base trajectory. Discrete maps can
implement `TangentStepper` directly and conventionally use a step size of one.

Kaplan-Yorke dimension computed from a truncated spectrum is relative to that
partial spectrum. Summing positive exponents equals metric entropy only where
the hypotheses of Pesin's identity hold. A positive Lyapunov exponent shows
sensitive dependence; it does not establish cryptographic one-wayness.

Covariant Lyapunov vectors, adjoint modes, SALI/GALI/MEGNO, stochastic error
models, checkpointing, and distributed reductions are intentionally outside
this initial crate surface.

## License

MIT.
