# Third-party notices

The quaternion, spherical-geodesic, Lohe synchronization, gated-scan, and
Helmholtz-fiber formulas in this crate were extracted and reworked from
HarmonicRust, commit
`c20fd04956f987f9a00d53c78728d8069f0a1589`.

The pinned HarmonicRust repository declares MIT licensing in its workspace
manifest and README. It does not carry a separate copyright attribution in
those files, so none is invented here. The crate's `LICENSE` contains the MIT
terms under which this derived work is distributed.

The gated-scan and Helmholtz-fiber source comments in HarmonicRust identify an
earlier HarmonicMLX implementation (`harmonic_mlx/attention.py` and
`harmonic_mlx/helmholtz.py`) as the formula lineage. No HarmonicMLX code or
framework integration is included in this crate.

This extraction replaces architecture-specific unsafe SIMD and unchecked slice
indexing with safe scalar Rust, typed quaternions, and validated dimensions.
Neural-network blocks, serialization configuration, and model diagnostics were
not carried over.
