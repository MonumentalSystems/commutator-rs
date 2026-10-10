# Changelog

All notable changes to the public crates will be documented here. Each crate
is versioned independently; release headings include the crate name.

## Unreleased

- Establish the public foundational workspace.
- Freeze cross-implementation Clifford conventions and golden vectors.
- Publish portable Cl(1,3) field operators, reference steppers, and analysis.
- Extract fixed-size Versor geometry into `clifford-geometry` while excluding
  its duplicate simulation and hardware-backend code.
- Extract checked Cl(6,0) rotors, periodic lattice storage, and a portable
  sequential Metropolis reference kernel into `clifford-lattice`.
- Extract framework-neutral Clifford neural layers into `clifford-layers`.
- Extract safe quaternion, spherical Lohe, gated-scan, and Helmholtz reference
  dynamics into the dependency-free `harmonic-dynamics` crate.
- Publish transport-neutral experiment and reproducibility contracts.
- Add context-bound Merkle commitments and deterministic post-commitment spot
  checks in `experiment-merkle`.
- Extract explicit mesh topology and corrected, bounded Euclidean tessellators
  into the renderer-independent `clifford-mesh` crate.
- Add checked CGA point and real dual-sphere decomposition to
  `clifford-geometry`, including homogeneous normalization when the scaled
  `f32` coefficients retain the represented semantics.
- Add optional checked CGA point-cloud and real-dual-sphere adapters to
  `clifford-mesh` while keeping its default build dependency-free.
- Add checked, transactional harmonic-wave, tanh-interface, and physical-core
  vortex initializers to `clifford-field` without coupling them to experiment
  presets or timestep selection. Initializers never select or change boundary
  policy; the lone-vortex initializer rejects periodic topology.
- Add typed `S¹` phases, circular order parameters, and checked all-to-all
  Kuramoto Euler dynamics to `harmonic-dynamics`, including natural-frequency
  drift for uncoupled and single-oscillator systems.

