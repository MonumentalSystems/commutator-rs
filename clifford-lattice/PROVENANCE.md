# Provenance

This crate was extracted and reworked by Monumental Systems from its
Commutator and HarmonicRust research code. The public implementation is
released under the workspace MIT license by the copyright holder.

Relevant source-history anchors:

- HarmonicRust origin: `d4589782dae81a6c40fa3dc70cf24fa68ef98d98`
- Commutator import: `4aa843af99caed0f4c7286d43692efef342a2182`
- f64 Rotor32 implementation:
  `d2b4c5591679bd36c29f2af3d4ecec62747a66cd`
- checkerboard implementation:
  `85b37bea7ce01d69b1f482fc2f6817dd60b2e8b5`
- fast-sweep integration and current Rotor32 lineage:
  `a478a9777a420978994930d07395cb8c780c3c9b`
- legacy rotor-simulation lineage reviewed for the portable CPU interaction:
  `c9924b3011e8c84262a68ef3e7625707c7053c92`

The extraction preserves the Cl(6,0) ShortLex basis order and the positive
`exp(B)` convention while changing the public contract in several ways:

- arbitrary even multivectors and checked unit rotors are separate types;
- the Cayley table and optimization internals are private;
- full-64 imports validate odd-grade coefficients;
- bivector exponentiation uses scaling and squaring instead of an unbounded
  six-term approximation;
- proposal directions and the symmetric pair interaction are explicit;
- the coupling is named `beta_coupling` to identify it as dimensionless.

No accelerator implementation was copied. Deterministic rotor flow and
Langevin dynamics were deliberately omitted: the source's purported tangent
projection does not project to grade two, and its CPU and GPU noise scalings
conflict. The CPU and GPU Metropolis paths also use different pair actions and
proposal bases, so this crate makes no backend-equivalence claim.
