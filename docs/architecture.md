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
The optional `cga3d` feature consumes `clifford-geometry`'s checked point,
real-dual-sphere, direct-plane, and compact-dual-plane decomposition without
decoding sparse blade positions itself. Plane tessellation size and resolution
remain caller policy. CGA circle and line adapters remain deferred until
matching semantic decomposition APIs exist.

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

## Host adapters

A host may connect those layers to SIMD, Metal, CUDA, Accelerate, WebGPU,
HTTP, databases, identities, or volunteer-worker scheduling. Those
adapters remain outside the foundational crates so local experiments do not
inherit operational dependencies.

