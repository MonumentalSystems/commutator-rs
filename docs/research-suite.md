# Research suite

The research crates are small reference implementations organized around
composable scientific boundaries. They do not attempt to replace broad linear
algebra, electronic-structure, Maxwell, or many-body packages.

## Dynamics and chaos

- `field-lyapunov` accepts a matrix-free vector field and Jacobian-vector
  product, or a caller-supplied tangent stepper. It computes deterministic
  top-k Benettin spectra and finite-time windows without materializing a dense
  Jacobian. This is the intended bridge from `clifford-field`, `spin-lattice`,
  and downstream GPU/PDE solvers.
- `quantum-chaos` consumes energy levels and provides spectral diagnostics. It
  is deliberately not an eigensolver and keeps unfolding policy explicit.
- `harmonic-dynamics` remains the reference layer for Kuramoto, Lohe, sphere,
  quaternion, and harmonic-sequence dynamics.

## Quantum and magnetic matter

- `quantum-magnetism` provides matrix-free finite spin-1/2 Hamiltonians for
  anisotropic Heisenberg and Dzyaloshinskii-Moriya models, measured observables,
  and a small deterministic Lanczos reference solver.
- `spin-lattice` derives harmonic forces and magnetic effective fields from one
  magnetoelastic Hamiltonian, then supplies checked reference integrators.
- `superconductivity` constructs spinful onsite singlet BdG matrices and checks
  Hermiticity, particle-hole symmetry, supplied eigenpairs, LDOS, and anomalous
  pairing observables. Large eigensolvers remain caller-selected.
- `majorana-fermions` supplies sparse Clifford-sign-exact Majorana monomials,
  canonical fermion operators, parity, quadratic Hamiltonians, and an open
  Kitaev chain. Its small-system product convention is tested against
  `clifford-core`.
- `cluster-green` starts from a cluster one-particle Green function. It checks
  retarded causality and performs CPT/Dyson embedding and position-aware
  periodization; it does not duplicate an exact-diagonalization solver.
- `quantum-transport` reuses `cluster-green` matrices for coherent NEGF and
  Landauer calculations rather than introducing a second matrix convention.

## Quantum light and measurement

- `quantum-light` starts after optical modes and source parameters have been
  determined. It provides bounded Fock states, creation and annihilation,
  passive phase/beam-splitter transformations, photon correlations, and
  reduced density matrices. Maxwell propagation and nonlinear material models
  remain upstream.
- `quantum-shadows` consumes externally generated local-Pauli measurement
  records. It estimates Pauli strings and linear observables with mean,
  standard-error, median-of-means, and conservative Hoeffding reductions. RNG,
  device control, and detector calibration stay with the experiment host.

## Interoperability contracts

Quantum crates expose `num_complex::Complex64` rather than crate-local complex
types. Numerical crates accept caller-owned slices or checked row-major
containers. Solvers that are expensive, hardware-specific, or already mature
elsewhere are injected as state vectors, eigensystems, cluster Green functions,
or tangent steppers.

Every model keeps ordering and sign conventions in its README and rustdoc.
Reference kernels reject non-finite values, incompatible dimensions, and
silent cutoff loss. Optimized implementations should be added as downstream
adapters and checked differentially against these kernels.

## Scope and maturity

These crates are pre-1.0 reference surfaces. Current boundaries intentionally
exclude density-functional theory, a general Hubbard exact-diagonalization
framework, production molecular dynamics, full Maxwell/FDTD simulation,
quantum process or gate-set tomography, open-system master equations, and
accelerator kernels. Those are integration targets or later focused crates,
not reasons to expand one foundational crate into a framework.

Distributed execution is also separate. Any model can be wrapped in
`experiment-core` contracts and hosted by Commutator without acquiring HTTP,
identity, database, or scheduler dependencies.

## Publication order

Independent crates can be published after their package checks pass.
`clifford-core` must precede packages whose conformance tests reference its
published version. `cluster-green` must precede `quantum-transport`. Website
catalog entries should link to crates.io API documentation, the repository
subdirectory, and any associated paper or reproducibility dataset.
