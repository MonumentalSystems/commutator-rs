# quantum-light

`quantum-light` provides small, checked reference kernels for discrete-variable
quantum photonics. It complements electromagnetic and nonlinear-source solvers:
those tools can determine modes and coupling parameters, while this crate owns
the finite Fock-space state, passive optical transformations, and photon-counting
observables.

The first release includes:

- bounded multimode Fock spaces with explicit basis ordering;
- complex state vectors, normalization, creation, and annihilation operators;
- phase shifters and exact two-mode lossless beam splitters;
- photon-number distributions, coincidences, and second-order coherence;
- pure-state density matrices, one-mode reductions, and purity; and
- truncated two-mode squeezed-vacuum states.

```rust
use quantum_light::{FockSpace, StateVector};

let space = FockSpace::try_new(2, 2)?;
let input = StateVector::basis(space, &[1, 1])?;
let output = input.beam_splitter(0, 1, core::f64::consts::FRAC_PI_4)?;

// Hong-Ou-Mandel interference removes the |1,1> coincidence.
assert!(output.probability(&[1, 1])? < 1e-24);
# Ok::<(), quantum_light::QuantumLightError>(())
```

This is a deterministic CPU reference implementation, not a large-scale
Gaussian-boson sampler, Maxwell solver, or hardware-control package. Per-mode
cutoffs are explicit; transformations fail instead of silently discarding
amplitude outside the represented space.

## Basis convention

Occupation vectors are ordered `[n0, n1, ...]`. Mode zero is the least
significant digit in the mixed-radix basis index. `max_occupation` is inclusive,
so a cutoff of two represents occupations 0, 1, and 2.

## License

MIT.

