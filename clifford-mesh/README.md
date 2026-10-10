# clifford-mesh

`clifford-mesh` provides checked, renderer-independent geometry buffers and
Euclidean tessellators for geometric-algebra applications.

The crate distinguishes three topologies explicitly:

- `TriangleMesh`: vertices plus indexed triangle faces;
- `LineSegments`: vertices plus indexed independent edges;
- `PointCloud`: unconnected positions.

Its default build has no renderer, windowing, GPU, serialization, or algebra
dependency. Generated buffers can be adapted to wgpu, Bevy, OpenGL, Metal,
CUDA, or CPU visualization code without inheriting those ecosystems here. The
optional `cga3d` feature adds checked adapters from `clifford-geometry`.

```rust
use clifford_mesh::{Sphere3, UvSphereOptions, primitives::uv_sphere};

let sphere = Sphere3::new([1.0, 2.0, 3.0], 2.0)?;
let mesh = uv_sphere(&sphere, UvSphereOptions::uniform(24)?)?;
assert_eq!(mesh.triangle_count(), 2 * 24 * 23);
assert!(mesh.vertices().iter().all(|vertex| vertex.is_finite()));
# Ok::<(), clifford_mesh::MeshError>(())
```

## Scope

The initial release contains Euclidean descriptors and generators for UV and
icospheres, circle loops/discs/tubes, plane patches/grids, arrows, and open
cylinders. It validates finite inputs, positive dimensions, nonzero directions,
resolution bounds, index ranges, and buffer topology.

Coordinates are right-handed and use caller-defined units. Generated triangle
winding is counter-clockwise when viewed from the outward normal; meshes built
from caller-owned buffers retain the caller's winding. UV spheres use unique
poles; open cylinders are intentionally uncapped. Circular/grid resolutions
are limited to 1,024 and icosphere subdivision to 7 before allocation.

CGA circle and line adapters are intentionally deferred. The original
Versor-Rust drawing module's heuristic decoders do not preserve the required
translation and homogeneous-scaling semantics. Checked point-cloud and
real-dual-sphere adapters are available behind `cga3d`; a plane adapter remains
a separate follow-up now that checked direct- and compact-dual-plane
decomposition exists in `clifford-geometry`. Mesh adapters consume those
semantic APIs rather than inspecting sparse blade slots here.

```rust
# #[cfg(feature = "cga3d")]
# {
use clifford_geometry::cga3d::{point, Round};
use clifford_mesh::cga3d::real_dual_sphere;
use clifford_mesh::{primitives::icosphere, IcosphereOptions};

let value = Round::dls(&point(1.0, 2.0, 3.0), 2.0);
let sphere = real_dual_sphere(&value).unwrap();
let mesh = icosphere(&sphere, IcosphereOptions::new(2).unwrap()).unwrap();
assert_eq!(mesh.triangle_count(), 320);
# }
```

Enable it with `cargo add clifford-mesh --features cga3d`. Zero-radius point
spheres and imaginary spheres are not turned into arbitrary display geometry;
use a point adapter or an explicit application-owned marker policy instead.

See [`PROVENANCE.md`](PROVENANCE.md) for extraction details.

## License

BSD-2-Clause.
