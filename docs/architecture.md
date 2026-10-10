# Architecture

`commutator-rs` is organized around one rule: scientific definitions should
not depend on the infrastructure that executes them.

## Algebra layer

`clifford-core` owns runtime signatures, blade ordering, products, reverse,
grade projection, sandwich products, rotor construction, and the open scalar
contract. It has no runtime dependencies.

## Field layer

`clifford-field` owns precision-selected Cl(1,3) bivectors, regular-grid
storage, boundary conditions, portable spatial operators, reference Euler and
Strang steppers, and array-based analysis. It depends inward on
`clifford-core`; it does not know about workers, networks, or experiments.

## Geometry layer

`clifford-geometry` owns fixed-size, geometry-specific EGA, PGA, CGA, STA,
and Cl(6,0) types together with rotors, motors, meet/join operations, frames,
interpolation, and kinematic chains. It preserves Versor's sparse typed
geometry model while using `clifford-core` as the canonical convention and
conformance boundary. Checked point and real dual-sphere decomposition provides
the semantic boundary for consumers without exposing sparse blade positions;
checked direct- and compact-dual-plane decomposition provides normalized Hesse
parameters without relying on the lossy legacy plane dual projection.
Rendering, field simulation, and hardware backends are outside this crate.

## Layer library

`clifford-layers` owns framework-neutral CPU reference implementations of
Clifford linear, convolution, normalization, and optional Fourier layers. It
depends only on `clifford-core`, plus optional RustFFT support, and contains no
training runtime, model, networking, or accelerator integration.

## Mesh layer

`clifford-mesh` owns checked renderer-independent point, indexed-line, and
indexed-triangle buffers plus bounded Euclidean primitive tessellators. Its
default build is independent of graphics runtimes and algebra representations.
The optional `cga3d` feature consumes `clifford-geometry`'s checked point and
real-dual-sphere decomposition without decoding sparse blade positions itself.
A checked plane adapter remains a separate integration slice; CGA circle and
line adapters remain deferred until matching semantic decomposition APIs exist.

## Experiment layer

`experiment-core` owns typed descriptions of work and results, topology,
verification policy, checkpoint references, worker capabilities, and run
metadata. It does not choose an async runtime, wire format, database, or
identity system.

## Verification layer

`experiment-merkle` owns a versioned SHA-256 commitment format and
transport-neutral post-commitment spot-check messages. Roots bind experiment
context, ordered leaf count, leaf positions, exact bytes, deterministic
padding, and tree structure. It proves byte membership only; hosts retain
responsibility for unpredictable nonces, sampling policy, identity, and
independent scientific recomputation.

`experiment-accelerator` sits outside that commitment primitive. It owns
backend descriptors, deterministic partition plans, and differential
qualification reports that compare an optimized implementation with a
portable reference. Reports bind work content, worker identity, backend IDs,
implementation versions, precision, and a versioned policy with explicit
absolute and relative tolerances. Opaque reports authorize only the exact work,
worker, backend descriptor, concrete execution fingerprint, and current policy
checked in-process before and after execution. Separate serialize-only audit
snapshots expose digests and the policy/comparison as a commitment summary, not
the raw descriptor, worker, or work preimages, and provide no path back to
admission authority.
The default feature set contains no hardware API, and the crate contains no
network transport. Neither configuration turns untrusted serialized evidence
into an admission token.

Its default build remains hardware-neutral. The optional `cuda` feature is the
one concrete hardware boundary: a checked f64 vector-affine backend performs
real NVRTC compilation, device transfers, and kernel execution through cudarc,
then uses the same differential qualification path as external accelerators.
The committed NVIDIA GB10 record demonstrates that narrow affine contract for
one documented driver and policy; it includes raw recomputation inputs but is
unauthenticated until covered by signed release provenance and is not evidence
for unrelated kernels.
`ThreadedShardedBackend` is the concrete transport-neutral distributed
reference: contiguous work is assigned to in-process child workers, child
seeds are derived deterministically, failures retain shard ordinals, and output
is reassembled in source order. Network transports can preserve this contract
without being embedded in the crate.

## Lattice layer

`clifford-lattice` owns checked pure-f64 Cl⁺(6,0) elements and Spin(6)
rotors, generic periodic two-dimensional storage, explicit pair-interaction
and proposal policies, and a deterministic sequential Metropolis reference
sweep. Its only runtime dependency is `clifford-core`. Parallel checkerboard
sweeps, deterministic rotor flow, Langevin dynamics, accelerator kernels, and
experiment-specific parameter choices remain outside the crate until they
have one validated numerical contract.

## Harmonic dynamics

`harmonic-dynamics` owns dependency-free typed circle phases, complete-graph
Kuramoto dynamics, quaternion and sphere geometry, Lohe mean-field
synchronization, stable gated recurrences, and multi-head Helmholtz sequence
fibers. It is a portable numerical reference crate: neural-network frameworks,
model configuration, serialization, accelerator kernels, and distributed
execution remain outside its boundary.

## Research simulation crates

The research suite follows the same rule: each crate owns one scientific
contract and leaves large solvers or deployment concerns outside.

- `field-lyapunov` owns matrix-free tangent-flow and finite-time chaos
  diagnostics.
- `quantum-chaos` owns diagnostics over supplied spectra.
- `quantum-magnetism` owns finite spin-1/2 reference Hamiltonians and
  observables; `spin-lattice` owns classical magnetoelastic dynamics; and
  `phonon-transport` owns harmonic-chain and ballistic thermal references.
- `superconductivity` owns checked BdG assembly and observables;
  `superconducting-dynamics` owns gauge-covariant TDGL and Josephson dynamics;
  and `majorana-fermions` owns sparse Majorana operator algebra.
- `cluster-green` owns cluster Green-function validation and embedding;
  `cluster-embedding` builds solver-injected VCA/DMFT foundations on it;
  `keldysh-green` reuses its matrices for real-time nonequilibrium Green
  functions; and `quantum-transport` consumes the same convention for
  coherent NEGF.
- `quantum-light` owns bounded photonic Fock states and counting observables;
  `open-quantum-systems` owns dense Lindblad reference dynamics;
  `quantum-shadows` owns local-Pauli measurement reduction; and
  `quantum-tomography` owns state/PTM/Choi reconstruction diagnostics plus
  gauge-aware sequence models and deterministic small-system reconstruction of
  explicitly selected GST coordinates. Its reference optimizer is neither
  CPTP-constrained nor a turnkey production GST system.

`physics-conformance` is a workspace-only consumer of public APIs. It owns no
scientific implementation and is not released to crates.io; its purpose is to
detect convention drift across independently publishable packages.

See [`research-suite.md`](research-suite.md) for workflows and non-goals.

## Host adapters

A host may connect those layers to SIMD, Metal, CUDA, Accelerate, WebGPU,
HTTP, databases, identities, or volunteer-worker scheduling. Those
adapters remain outside the foundational crates so local experiments do not
inherit operational dependencies.

The allowed dependency direction for the advanced suite is:

```text
clifford-core
├── clifford-field / geometry / lattice / layers
├── harmonic-dynamics
├── majorana-fermions
└── clifford-geometry ──> clifford-mesh (optional feature)

cluster-green
├── cluster-embedding
├── keldysh-green
└── quantum-transport

open-quantum-systems ──> quantum-tomography
experiment-core ───────> experiment-merkle / experiment-accelerator

selected public APIs ──> physics-conformance (workspace-only)
```

Crates omitted from the arrows are independent at the package level. A
scientific conversion such as TDGL order parameters to BdG onsite gaps may be
kept dependency-neutral when a shared scalar/vector boundary is sufficient.

