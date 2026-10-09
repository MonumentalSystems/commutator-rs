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
conformance boundary. Rendering, field simulation, and hardware backends are
outside this crate.

## Layer library

`clifford-layers` owns framework-neutral CPU reference implementations of
Clifford linear, convolution, normalization, and optional Fourier layers. It
depends only on `clifford-core`, plus optional RustFFT support, and contains no
training runtime, model, networking, or accelerator integration.

## Experiment layer

`experiment-core` owns typed descriptions of work and results, topology,
verification policy, checkpoint references, worker capabilities, and run
metadata. It does not choose an async runtime, wire format, database, or
identity system.

## Harmonic dynamics

`harmonic-dynamics` owns dependency-free quaternion and sphere geometry,
Lohe mean-field synchronization, stable gated recurrences, and multi-head
Helmholtz sequence fibers. It is a portable numerical reference crate: neural
network frameworks, model configuration, serialization, accelerator kernels,
and distributed execution remain outside its boundary.

## Host adapters

A host may connect those layers to SIMD, Metal, CUDA, Accelerate, WebGPU,
HTTP, databases, Merkle commitments, or volunteer-worker scheduling. Those
adapters remain outside the foundational crates so local experiments do not
inherit operational dependencies.

