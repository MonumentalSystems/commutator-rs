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

