# Provenance

`clifford-mesh` is extracted and redesigned from the renderer-agnostic
[`src/draw/`](https://github.com/DavinciDreams/versor-rs/tree/93a5a1334ab77131e5ef6b88105cdaac57d267e6/src/draw)
work in `DavinciDreams/versor-rs` at commit
`93a5a1334ab77131e5ef6b88105cdaac57d267e6`.

The public v0.1 API deliberately changes the source design:

- point clouds, indexed line segments, and triangle meshes are distinct types;
- generators validate finite geometry and bound tessellation sizes;
- triangle winding is checked against the intended outward normal;
- UV spheres use unique poles rather than degenerate pole quads;
- the source CGA circle, line, and plane decoders are not included.

The optional `cga3d` adapters are new composition over the checked point and
real-dual-sphere decomposition API in `clifford-geometry`; they do not preserve
the source module's tolerance-based classification, imaginary-radius absolute
value, or fabricated tiny spheres. Circle, line, and plane adapters remain
outside the current API; checked plane decomposition now exists in
`clifford-geometry`, while circle and line decomposition still require
conformance work. This crate does not claim parity with C++ Versor's
visualization layer.

The extracted and redesigned implementation retains the upstream BSD-2-Clause
license and attribution.
