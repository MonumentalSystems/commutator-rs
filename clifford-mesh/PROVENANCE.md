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

Those CGA adapters remain deferred pending conformance tests for translated
circle extraction, homogeneous line scaling, and direct-line layout in
`clifford-geometry`. This crate does not claim parity with C++ Versor's
visualization layer.

The extracted and redesigned implementation retains the upstream BSD-2-Clause
license and attribution.
