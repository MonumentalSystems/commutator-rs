use clifford_mesh::primitives::{
    arrow, circle_disc, circle_loop, circle_tube, icosphere, line_segment, open_cylinder,
    plane_grid, plane_normal_indicator, plane_patch, uv_sphere,
};
use clifford_mesh::{
    Circle3, CircleOptions, CylinderOptions, IcosphereOptions, Line3, LineSegments, MeshError,
    Plane3, PlanePatchOptions, PointCloud, Sphere3, TriangleMesh, TubeOptions, UvSphereOptions,
    Vertex, MAX_ICOSPHERE_SUBDIVISIONS, MAX_RESOLUTION, MIN_CIRCULAR_SEGMENTS, MIN_PLANE_SEGMENTS,
    MIN_UV_SPHERE_STACKS,
};

type Vec3 = [f32; 3];

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: Vec3, b: Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(value: Vec3) -> f32 {
    dot(value, value).sqrt()
}

fn assert_valid_outward_mesh(mesh: &TriangleMesh) {
    assert!(mesh.vertices().iter().all(|vertex| vertex.is_finite()));
    for triangle in mesh.triangles() {
        let [a, b, c] = triangle.map(|index| index as usize);
        assert!(a < mesh.vertices().len());
        assert!(b < mesh.vertices().len());
        assert!(c < mesh.vertices().len());
        let va = mesh.vertices()[a];
        let vb = mesh.vertices()[b];
        let vc = mesh.vertices()[c];
        let geometric = cross(
            sub(vb.position(), va.position()),
            sub(vc.position(), va.position()),
        );
        assert!(norm(geometric) > 0.0, "zero-area face {triangle:?}");
        let desired = add(add(va.normal(), vb.normal()), vc.normal());
        assert!(
            dot(geometric, desired) > 0.0,
            "face winding opposes normals: {triangle:?}"
        );
    }
}

#[test]
fn descriptors_and_options_reject_invalid_inputs() {
    assert!(matches!(
        Sphere3::new([0.0; 3], 0.0),
        Err(MeshError::NonPositive("sphere radius"))
    ));
    assert!(matches!(
        Sphere3::new([f32::NAN, 0.0, 0.0], 1.0),
        Err(MeshError::NonFinite("sphere center"))
    ));
    assert!(matches!(
        Circle3::new([0.0; 3], [0.0; 3], 1.0),
        Err(MeshError::ZeroVector("circle normal"))
    ));
    assert!(Line3::new([0.0; 3], [f32::INFINITY, 0.0, 0.0]).is_err());
    assert!(Plane3::new([0.0; 3], [0.0; 3]).is_err());

    assert!(UvSphereOptions::new(MIN_UV_SPHERE_STACKS - 1, MIN_CIRCULAR_SEGMENTS).is_err());
    assert!(UvSphereOptions::uniform(MAX_RESOLUTION + 1).is_err());
    assert!(CircleOptions::new(2).is_err());
    assert!(TubeOptions::new(8, 2).is_err());
    assert!(PlanePatchOptions::new(MIN_PLANE_SEGMENTS - 1, 8).is_err());
    assert!(PlanePatchOptions::new(1, 1).is_ok());
    assert!(CylinderOptions::new(MAX_RESOLUTION + 1).is_err());
    assert!(IcosphereOptions::new(MAX_ICOSPHERE_SUBDIVISIONS + 1).is_err());
}

#[test]
fn extreme_finite_directions_normalize_without_panicking() {
    for direction in [
        [f32::MAX, 0.0, 0.0],
        [f32::MIN_POSITIVE, 0.0, 0.0],
        [f32::from_bits(1), 0.0, 0.0],
        [f32::MAX, -f32::MAX, f32::MAX / 2.0],
    ] {
        let line = Line3::new([0.0; 3], direction).unwrap();
        assert!((norm(line.direction()) - 1.0).abs() < 1e-6);
        assert_eq!(line_segment(&line, 1.0).unwrap().segment_count(), 1);

        let circle = Circle3::new([0.0; 3], direction, 1.0).unwrap();
        assert!((norm(circle.normal()) - 1.0).abs() < 1e-6);
        assert_eq!(
            circle_loop(&circle, CircleOptions::new(8).unwrap())
                .unwrap()
                .segment_count(),
            8
        );
    }
}

#[test]
fn checked_topology_rejects_non_finite_derived_geometry() {
    let vertices = vec![
        Vertex::new([f32::MAX, 0.0, 0.0], [0.0, 0.0, 1.0]).unwrap(),
        Vertex::new([0.0, f32::MAX, 0.0], [0.0, 0.0, 1.0]).unwrap(),
        Vertex::new([0.0, 0.0, f32::MAX], [0.0, 0.0, 1.0]).unwrap(),
    ];
    assert!(matches!(
        TriangleMesh::try_from_parts(vertices, vec![[0, 1, 2]]),
        Err(MeshError::NonFinite("triangle geometry"))
    ));
    assert!(matches!(
        LineSegments::try_from_parts(
            vec![[f32::MAX, 0.0, 0.0], [-f32::MAX, 0.0, 0.0]],
            vec![[0, 1]],
        ),
        Err(MeshError::NonFinite("line segment geometry"))
    ));

    let huge_sphere = Sphere3::new([0.0; 3], f32::MAX).unwrap();
    assert!(matches!(
        uv_sphere(&huge_sphere, UvSphereOptions::uniform(8).unwrap()),
        Err(MeshError::NonFinite(_))
    ));
}

#[test]
fn tube_radius_must_define_a_regular_embedded_tube() {
    let circle = Circle3::new([0.0; 3], [0.0, 0.0, 1.0], 1.0).unwrap();
    let options = TubeOptions::new(8, 7).unwrap();
    assert!(circle_tube(&circle, 0.999, options).is_ok());
    assert!(matches!(
        circle_tube(&circle, 1.0, options),
        Err(MeshError::InvalidRelation(_))
    ));
    assert!(matches!(
        circle_tube(&circle, 2.0, options),
        Err(MeshError::InvalidRelation(_))
    ));
}

#[test]
fn topology_containers_validate_indices_and_finiteness() {
    let vertices = vec![
        Vertex::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]).unwrap(),
        Vertex::new([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]).unwrap(),
        Vertex::new([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]).unwrap(),
    ];
    let mesh = TriangleMesh::try_from_parts(vertices.clone(), vec![[0, 1, 2]]).unwrap();
    assert_eq!(mesh.triangle_count(), 1);
    assert_eq!(mesh.to_flat_indices(), vec![0, 1, 2]);
    assert_eq!(mesh.to_interleaved_position_normal_f32().len(), 18);
    assert!(matches!(
        TriangleMesh::try_from_parts(vertices.clone(), vec![[0, 1, 3]]),
        Err(MeshError::IndexOutOfRange)
    ));
    assert!(matches!(
        TriangleMesh::try_from_parts(vertices, vec![[0, 0, 1]]),
        Err(MeshError::DegenerateTriangle)
    ));
    assert!(Vertex::new([0.0; 3], [0.0; 3]).is_err());
    assert!(LineSegments::try_from_parts(vec![[0.0; 3]], vec![[0, 1]]).is_err());
    assert!(matches!(
        LineSegments::try_from_parts(vec![[0.0; 3]], vec![[0, 0]]),
        Err(MeshError::DegenerateSegment)
    ));
    assert!(PointCloud::try_from_positions(vec![[0.0, f32::NAN, 0.0]]).is_err());
}

#[test]
fn redesigned_uv_sphere_has_no_degenerate_or_inward_caps() {
    let sphere = Sphere3::new([1.0, 2.0, 3.0], 2.5).unwrap();
    let options = UvSphereOptions::new(8, 8).unwrap();
    let mesh = uv_sphere(&sphere, options).unwrap();
    assert_eq!(mesh.vertices().len(), 2 + 7 * 8);
    assert_eq!(mesh.triangle_count(), 2 * 8 * 7);
    assert_valid_outward_mesh(&mesh);
    for vertex in mesh.vertices() {
        let offset = sub(vertex.position(), sphere.center());
        assert!((norm(offset) - sphere.radius()).abs() < 1e-4);
        assert!((norm(vertex.normal()) - 1.0).abs() < 1e-5);
    }
}

#[test]
fn icosphere_counts_and_geometry_are_stable() {
    let sphere = Sphere3::new([-2.0, 0.5, 4.0], 3.0).unwrap();
    for subdivisions in 0..=3 {
        let mesh = icosphere(&sphere, IcosphereOptions::new(subdivisions).unwrap()).unwrap();
        let factor = 4_usize.pow(subdivisions);
        assert_eq!(mesh.vertices().len(), 10 * factor + 2);
        assert_eq!(mesh.triangle_count(), 20 * factor);
        assert_valid_outward_mesh(&mesh);
        for vertex in mesh.vertices() {
            assert!((norm(sub(vertex.position(), sphere.center())) - 3.0).abs() < 2e-4);
        }
    }
}

#[test]
fn circle_topologies_are_explicit_and_geometrically_correct() {
    let circle = Circle3::new([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 2.0).unwrap();
    let loop_options = CircleOptions::new(17).unwrap();
    let line_loop = circle_loop(&circle, loop_options).unwrap();
    assert_eq!(line_loop.positions().len(), 17);
    assert_eq!(line_loop.segment_count(), 17);
    assert_eq!(line_loop.segments()[16], [16, 0]);
    for position in line_loop.positions() {
        let offset = sub(*position, circle.center());
        assert!((norm(offset) - circle.radius()).abs() < 2e-4);
        assert!(dot(offset, circle.normal()).abs() < 2e-4);
    }

    let disc = circle_disc(&circle, loop_options).unwrap();
    assert_eq!(disc.vertices().len(), 18);
    assert_eq!(disc.triangle_count(), 17);
    assert_valid_outward_mesh(&disc);

    let tube = circle_tube(&circle, 0.25, TubeOptions::new(19, 7).unwrap()).unwrap();
    assert_eq!(tube.vertices().len(), 19 * 7);
    assert_eq!(tube.triangle_count(), 2 * 19 * 7);
    assert_valid_outward_mesh(&tube);
}

#[test]
fn plane_patch_grid_and_indicator_preserve_explicit_topology() {
    let plane = Plane3::new([1.0, -2.0, 3.0], [1.0, 1.0, 2.0]).unwrap();
    let options = PlanePatchOptions::new(5, 7).unwrap();
    let patch = plane_patch(&plane, 4.0, options).unwrap();
    assert_eq!(patch.vertices().len(), 6 * 8);
    assert_eq!(patch.triangle_count(), 2 * 5 * 7);
    assert_valid_outward_mesh(&patch);
    for vertex in patch.vertices() {
        assert!(dot(sub(vertex.position(), plane.point()), plane.normal()).abs() < 2e-4);
    }

    let grid = plane_grid(&plane, 4.0, options).unwrap();
    assert_eq!(grid.segment_count(), 6 + 8);
    assert_eq!(grid.positions().len(), 2 * (6 + 8));
    let indicator = plane_normal_indicator(&plane, 2.0).unwrap();
    assert_eq!(indicator.segment_count(), 1);
}

#[test]
fn line_generators_are_finite_bounded_and_outward() {
    let line = Line3::new([2.0, 3.0, 4.0], [1.0, -2.0, 0.5]).unwrap();
    let segment = line_segment(&line, 2.0).unwrap();
    assert_eq!(segment.segment_count(), 1);
    let arrow_mesh = arrow(&line, 4.0, 1.0, 0.25).unwrap();
    assert_eq!(arrow_mesh.segment_count(), 5);
    assert!(arrow(&line, 1.0, 1.0, 0.25).is_err());

    let cylinder = open_cylinder(&line, 3.0, 0.5, CylinderOptions::new(11).unwrap()).unwrap();
    assert_eq!(cylinder.vertices().len(), 22);
    assert_eq!(cylinder.triangle_count(), 22);
    assert_valid_outward_mesh(&cylinder);
}

#[test]
fn renderer_buffer_helpers_have_exact_shapes() {
    let sphere = Sphere3::new([0.0; 3], 1.0).unwrap();
    let mesh = uv_sphere(&sphere, UvSphereOptions::uniform(4).unwrap()).unwrap();
    assert_eq!(
        mesh.to_interleaved_position_normal_f32().len(),
        mesh.vertices().len() * 6
    );
    assert_eq!(mesh.to_flat_indices().len(), mesh.triangle_count() * 3);

    let circle = Circle3::new([0.0; 3], [0.0, 0.0, 1.0], 1.0).unwrap();
    let lines = circle_loop(&circle, CircleOptions::new(5).unwrap()).unwrap();
    assert_eq!(lines.to_position_f32().len(), lines.positions().len() * 3);
    assert_eq!(lines.to_flat_indices().len(), lines.segment_count() * 2);
}
