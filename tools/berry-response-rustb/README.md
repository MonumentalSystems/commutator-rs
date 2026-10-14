# Rustb qualification boundary

This standalone executable qualifies pinned Rustb commit
`ba04a2a0746f9798433cb4d9826675d10177fbc7` against an independent analytic
periodic two-band Hamiltonian. It then imports the resulting integrated dipole
into `berry-response` and checks the SI and field-order boundary.

This package has its own workspace and lockfile. Its Rust 1.90 minimum and
BLAS/LAPACK dependencies do not raise the main workspace's Rust 1.80 minimum.
The pinned backend was actually built with Rust 1.94.0 on being.local using
`netlib-system`. Exactly this scope is qualified; no material accuracy is implied.

## Run

Install a compatible Rust toolchain and a Netlib BLAS/LAPACK implementation with
CBLAS and Fortran runtime linking available. From the repository root:

```powershell
cargo run --release --locked --manifest-path tools/berry-response-rustb/Cargo.toml
cargo test --release --locked --manifest-path tools/berry-response-rustb/Cargo.toml
cargo fmt --check --manifest-path tools/berry-response-rustb/Cargo.toml
```

The executable writes JSON to stdout and exits unsuccessfully if any check fails.
`results/being-local.json` is an observed successful run, not a fabricated fixture.
`results/provenance.json` records toolchain and linked-library hashes. A locked
repeat on the same machine produced byte-identical JSON.

On being.local the runtime libraries existed without development linker names.
We created symlinks **only in an isolated temporary directory**, pointing
`libblas.so` and `libcblas.so` to the existing `libblas.so.3`, `liblapack.so` to
`liblapack.so.3`, and `libgfortran.so` to `libgfortran.so.5`; then set
`RUSTFLAGS='-L native=/tmp/rustb-qualification-20261014/lib'`. No system libraries,
packages, global toolchain, or services were changed. `remote-workspace.toml`
is the minimal workspace manifest used for the isolated remote core dependency.

## Independently checked physics

The mathematical fixture is

`H(q) = 0.3 sin(qx) I + sin(qx) sigma_x + sin(qy) sigma_y
       + (1 + cos(qx) + cos(qy)) sigma_z`,

with energies in eV, `q = a k` dimensionless, and a positive square lattice
constant `a` in Angstrom. This model generally breaks time reversal; it is not
a fitted Hamiltonian for Te or WTe2. Reversing the sign of the `sigma_y` term
provides a Berry-sign control.

The reference uses the closed-form two-level Berry curvature, independently of
Rustb eigenvectors and Kubo kernels. For `A = i <u|grad u>`, the upper-band
curvature is `-d dot (partial_x d cross partial_y d)/(2|d|^3)`.

We compare both periodic integrals:

* Fermi-surface: `D_x = sum integral (-df/dE) (partial_kx E) Omega_xy`.
* Fermi-sea: `D_x = sum integral f partial_kx Omega_xy`.

Both use `d^2k/(2pi)^2`. Integration by parts equates them on a periodic BZ.
The direct meshes are 64, 128, 256 and 512 per axis at 300 K and mu = 1.5 eV.
Band and curvature values are independently checked at generic k points.
Zero tilt, reversed tilt, reversed Berry sign, lattice length scaling, and
full-field contraction after symmetrization are also checked. Finite grid
errors are reported, not silently replaced by the analytic answer.

## Exact boundary conventions

Rustb's method `extrinsic_nonlinear_hall` returns an integrated BCD kernel here;
its label differs from the name often used for the intrinsic Berry-dipole
mechanism. It does **not** include `e^3 tau/hbar^2`, AC factors, or lead geometry.
`intrinsic_nonlinear_hall` in Rustb refers to a different Berry-connection-polarizability
mechanism and must not be substituted.

* Rustb directions are `(current, Berry field, velocity field)` and its raw
  kernel is `Omega[current,Berry field] * velocity[velocity field]`.
* The core's ordered tensor is `(current, velocity field, Berry field)`.
  Therefore `core[a][b][c] = K * Rustb[a][c][b]`, after length conversion.
* `Rustb[x,y,x] = D_x` and `Rustb[y,x,x] = -D_x` for a right-handed xy sheet.
* Lattice coordinates in Angstrom give an integrated 2D dipole in Angstrom.
  Multiply by `1e-10` exactly once before `Dipole2D::new`. Rustb already divides
  the uniform reduced-BZ average by the real-space cell area; do not add another
  cell-area or `(2pi)^2` normalization.
* Symmetrization averages the field permutations; the complete contraction
  includes both. Doubling one unsymmetrized mixed component is not general.

The SI handoff in the JSON uses the 128-square mesh dipole, tau = 1e-13 seconds,
fundamental angular frequency = 1e12 rad/s, and complex **peak** electric fields.
It yields a peak second-harmonic **sheet** current in A/m, with the core's
explicit positive-e charge/sign contract. This does not determine a measured
voltage without independent linear transport and lead geometry.

Unqualified paths: zero-temperature energy-cut integration, nonorthogonal or
negative-orientation cells, multiband/spin currents, position-matrix imports,
extrinsic scattering, and material-specific accuracy. Full Wannier position
matrices and convergence checks remain necessary for a material calculation.
