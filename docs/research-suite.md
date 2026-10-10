# Research suite

The research crates are small reference implementations organized around
composable scientific boundaries. They do not attempt to replace broad linear
algebra, electronic-structure, Maxwell, or many-body packages. All crates in
this document are pre-1.0 release candidates; the repository does not claim
that they are already available from crates.io.

## Dynamics and chaos

- `field-lyapunov` accepts a matrix-free vector field and Jacobian-vector
  product, or a caller-supplied tangent stepper. It computes deterministic
  top-k Benettin spectra and finite-time windows without materializing a dense
  Jacobian. This is the intended bridge from `clifford-field`, `spin-lattice`,
  and downstream GPU/PDE solvers.
- `quantum-chaos` consumes supplied energy levels and provides level-spacing,
  form-factor, number-variance, and unfolding diagnostics. It is deliberately
  not an eigensolver.
- `harmonic-dynamics` is the reference layer for Kuramoto, Lohe, sphere,
  quaternion, and harmonic-sequence dynamics, with `clifford-core` providing
  its algebra interoperability boundary.

## Quantum, magnetic, and thermal matter

- `quantum-magnetism` provides matrix-free finite spin-1/2 Hamiltonians for
  anisotropic Heisenberg and Dzyaloshinskii--Moriya models, measured
  observables, and a small deterministic Lanczos reference solver.
- `spin-lattice` derives harmonic forces and magnetic effective fields from one
  magnetoelastic Hamiltonian, then supplies checked reference integrators.
- `phonon-transport` supplies one-dimensional harmonic-chain dynamical
  matrices, bands, branch-tracked finite-difference group velocities, harmonic
  heat capacity, and ballistic Landauer thermal conductance. Force constants
  and reduced models remain caller supplied.
- `superconductivity` constructs spinful onsite-singlet BdG matrices and checks
  Hermiticity, particle-hole symmetry, supplied eigenpairs, LDOS, and anomalous
  pairing observables. Large eigensolvers remain caller selected.
- `superconducting-dynamics` owns scalar model-A TDGL on checked finite graphs,
  compact U(1) links, an analytic free-energy gradient, energy-backtracked
  evolution, loop flux, and Josephson current conventions. Its scaled order
  parameters can be passed to `superconductivity` as onsite gaps without
  introducing a package dependency.
- `majorana-fermions` supplies sparse Clifford-sign-exact Majorana monomials,
  canonical fermion operators, parity, quadratic Hamiltonians, and an open
  Kitaev chain. Its small-system product convention is checked against
  `clifford-core`.

## Green functions and correlated matter

- `cluster-green` starts from a cluster one-particle Green function. It checks
  retarded causality and performs CPT/Dyson embedding and position-aware
  periodization; it does not duplicate an exact-diagonalization solver.
- `cluster-embedding` adds solver-injected Hubbard reference systems, CPT
  self-energy embedding, Potthoff-functional quadrature, VCA stationarity, and
  retarded Bethe-lattice DMFT iteration. Exact diagonalization, tensor-network,
  Monte Carlo, and impurity solvers enter through traits.
- `keldysh-green` owns dense real-time retarded, advanced, lesser, and greater
  components, nonuniform-grid convolution, Langreth products, consistency
  diagnostics, equal-time observables, and a retarded Volterra/Dyson reference
  solver. It does not model imaginary-branch initial correlations or sampled
  delta-function self-energies.
- `quantum-transport` reuses `cluster-green` matrices for coherent device NEGF,
  lead broadenings, Caroli transmission, and Landauer current rather than
  introducing a second matrix convention.

## Quantum light, open systems, and measurement

- `quantum-light` starts after optical modes and source parameters have been
  determined. It provides bounded Fock states, creation and annihilation,
  passive phase/beam-splitter transformations, photon correlations, and
  reduced density matrices. Maxwell propagation and nonlinear material models
  remain upstream.
- `open-quantum-systems` provides checked dense operators and density matrices,
  Gorini--Kossakowski--Sudarshan--Lindblad generators, diagnostics, and
  fixed-step RK4 for small-system validation. Sparse Liouvillians, trajectories,
  and tensor networks remain downstream concerns.
- `quantum-shadows` consumes externally generated local-Pauli measurement
  records. It estimates Pauli strings and linear observables with mean,
  standard-error, median-of-means, and conservative Hoeffding reductions. RNG,
  device control, and detector calibration stay with the experiment host.
- `quantum-tomography` provides local-Pauli measurement plans, trace-one linear
  inversion, Euclidean PSD trace-one projection, normalized Choi conversion,
  PTM TP/unital/CP checks, and a focused gate-set sequence likelihood and gauge
  model. It deliberately leaves maximum-likelihood state/channel reconstruction
  and nonlinear GST optimization to injected specialist solvers.

## Accelerator and distributed validation

- `experiment-core` defines transport-neutral work units, results, topology,
  verification policy, capabilities, checkpoints, and run metadata.
- `experiment-merkle` binds ordered result bytes to experiment context and
  supports deterministic post-commitment spot-check messages.
- `experiment-accelerator` describes backend identity, precision,
  determinism, and scheduling requirements; partitions work deterministically;
  and differentially qualifies an optimized backend against a portable
  reference. It contains no CUDA, Metal, WebGPU, transport, or scheduler.
- `physics-conformance` is intentionally not published. It tests cross-crate
  conventions and conversion paths through public APIs inside this workspace.

The adapter crate validates the boundary around accelerated and distributed
work. It does not itself claim that any hardware implementation is correct or
deterministic; each backend must publish its own qualification evidence,
precision, reduction order, implementation version, and tolerance.

`experiment-accelerator` includes two executable reference boundaries for that
policy. `ThreadedShardedBackend` provides deterministic ordered sharding across
in-process child workers, suitable as the oracle for a network transport.
With the optional `cuda` feature, `CudaAffineBackend` performs real f64 device
execution of a checked vector-affine kernel and is differentially qualified
against its CPU reference. Hardware evidence belongs under the crate's
`evidence/` directory and is not itself an admission capability.

## Interoperability contracts

Quantum and Green-function crates expose `num_complex::Complex64` or re-export
the canonical type from their inward dependency. Numerical crates accept
caller-owned slices or checked row-major containers. Expensive,
hardware-specific, or ecosystem-mature solvers are injected as state vectors,
eigensystems, cluster/impurity solutions, Green functions, force constants, or
tangent steppers.

Every model keeps ordering, sign, unit, and normalization conventions in its
README and rustdoc. Reference kernels reject non-finite values, incompatible
dimensions, and silent cutoff loss. Optimized implementations belong in
downstream adapters and should be checked differentially against these
kernels.

## Scope and maturity

These crates are pre-1.0 reference surfaces. Current boundaries intentionally
exclude density-functional electronic structure, a general Hubbard
exact-diagonalization package, production molecular dynamics, full
Maxwell/FDTD simulation, production tensor-network solvers, turnkey
maximum-likelihood/GST optimization, anharmonic phonon Boltzmann transport,
and hardware kernels. The focused VCA/DMFT, real-time Keldysh, TDGL/Josephson,
Lindblad, tomography, and harmonic-phonon foundations are present without
absorbing those mature or application-specific systems.

Distributed execution remains separate. Any model can be wrapped in
`experiment-core` contracts and hosted by Commutator without acquiring HTTP,
identity, database, or scheduler dependencies.

## Publication status

All publishable manifests target crates.io, but publication is not implied by
their presence here. CI packages the complete publishable workspace together
so unpublished path dependencies can be verified from one source revision.
Actual registry publication must follow the dependency waves documented in
[Release and provenance](releasing.md). Website catalogue entries should be
added only after the corresponding crates.io and docs.rs URLs resolve.
