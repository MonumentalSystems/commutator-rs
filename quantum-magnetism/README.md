# quantum-magnetism

`quantum-magnetism` is a dependency-free exact/reference toolkit for finite
frustrated spin-1/2 models. It is intended for analytic checks, small-cluster
studies, and validation of scalable solvers—not large many-body production
runs.

The crate provides:

- checked finite graphs with anisotropic Heisenberg exchange, oriented
  Dzyaloshinskii–Moriya interactions, and longitudinal fields;
- matrix-free Hamiltonian application in the computational basis;
- energy, local and total magnetization, ordered spin correlations, and
  static structure factors;
- a deterministic, fully reorthogonalized Lanczos ground-state reference
  solver.

```rust
use quantum_magnetism::{Bond, LanczosConfig, SpinModel};

let model = SpinModel::builder(2)
    .bond(Bond::heisenberg(0, 1, 1.0))
    .build()?;
let ground = model.ground_state(LanczosConfig::default())?;
assert!((ground.energy + 0.75).abs() < 1.0e-10);
# Ok::<(), quantum_magnetism::MagnetismError>(())
```

## Conventions

Basis index bit `i = 1` denotes spin-up at site `i`; bit `0` denotes
spin-down. Spin operators are `S = sigma/2`. An oriented bond `(i, j)` adds

`Jx Sx_i Sx_j + Jy Sy_i Sy_j + Jz Sz_i Sz_j + D · (S_i × S_j)`.

Reversing a bond requires reversing `D`. A field value `h_i` adds
`-h_i Sz_i`. The structure factor is
`S_a(q) = <O_a(q)^dagger O_a(q)> / N`, where
`O_a(q) = sum_j exp(i q·r_j) S^a_j`.

The exact Hilbert space has dimension `2^N`. The API checks arithmetic and
buffer shapes, but callers remain responsible for choosing a tractable `N`.
Lanczos stores its Krylov basis and is consequently a small-system reference
implementation.

## License

MIT.

