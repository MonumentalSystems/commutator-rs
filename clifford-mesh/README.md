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
translation and homogeneous-scaling semantics. Checked point-cloud,
real-dual-sphere, direct-plane, and compact-dual-plane adapters are available
behind `cga3d`. Mesh adapters consume `clifford-geometry`'s semantic APIs
rather than inspecting sparse blade slots here. Plane adapters return the
closest point to the origin and preserve representative orientation: a
negative homogeneous scale reverses the normal and generated winding without
changing the plane locus.

```rust
# #[cfg(feature = "cga3d")]
# {
use clifford_geometry::cga3d::{point, Dlp, Round};
use clifford_geometry::mvec::Multivector;
use clifford_mesh::cga3d::{dual_plane, real_dual_sphere};
use clifford_mesh::{
    primitives::{icosphere, plane_patch},
    IcosphereOptions, PlanePatchOptions,
};

let value = Round::dls(&point(1.0, 2.0, 3.0), 2.0);
let sphere = real_dual_sphere(&value).unwrap();
let mesh = icosphere(&sphere, IcosphereOptions::new(2).unwrap()).unwrap();
assert_eq!(mesh.triangle_count(), 320);

let plane: Dlp = Multivector::new([0.0, 0.0, 1.0, -3.0]);
let descriptor = dual_plane(&plane).unwrap();
// `size` is the full side length; subdivisions are selected by the caller.
let patch = plane_patch(&descriptor, 4.0, PlanePatchOptions::new(4, 4).unwrap()).unwrap();
assert_eq!(patch.triangle_count(), 32);
# }
```

Enable it with `cargo add clifford-mesh --features cga3d`. Zero-radius point
spheres and imaginary spheres are not turned into arbitrary display geometry;
use a point adapter or an explicit application-owned marker policy instead.

See [`PROVENANCE.md`](PROVENANCE.md) for extraction details.

## License

BSD-2-Clause.
