# Provenance

This crate was extracted and reworked by Monumental Systems from its
Commutator and HarmonicRust research code. The public implementation is
released under the workspace MIT license by the copyright holder.

Relevant source-history anchors for the initial-condition formulas are:

- HarmonicRust snapshot: `d4589782dae81a6c40fa3dc70cf24fa68ef98d98`
- Commutator import: `4aa843af99caed0f4c7286d43692efef342a2182`
- imported `commutator-field/src/bivector_field/presets.rs` SHA-256 at the
  Commutator import above:
  `dd07e72d8eddd159acc347e747144edd5b8a35c32ca870a69f454f481ccc6d97`

The public API generalizes the reusable analytic pieces rather than exposing
the private experiment preset catalog. In particular, it does not carry over
astrophysical interpretation, regime labels, the real-valued chiral-sector
claim, implicit grid resizing, a nonzero nominal vortex core, or a lone vortex
with forced periodic boundaries. Coordinates and core widths are physical grid
coordinates, and the caller retains control of boundary conditions and the
time integrator.
