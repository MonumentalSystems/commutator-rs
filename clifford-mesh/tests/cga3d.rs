#![cfg(feature = "cga3d")]

use std::error::Error;

use clifford_geometry::cga3d::decomposition::DecompositionError;
use clifford_geometry::cga3d::{point, Round};
use clifford_geometry::mvec::Multivector;
use clifford_mesh::cga3d::{direct_plane, dual_plane, point_cloud, real_dual_sphere, CgaMeshError};
use clifford_mesh::primitives::{
    icosphere, plane_grid, plane_normal_indicator, plane_patch, uv_sphere,
};
use clifford_mesh::{IcosphereOptions, MeshError, PlanePatchOptions, UvSphereOptions};

fn assert_vec3_close(actual: [f32; 3], expected: [f32; 3], tolerance: f32) {
    for axis in 0..3 {
        assert!(
            (actual[axis] - expected[axis]).abs() <= tolerance,
            "axis {axis}: expected {}, got {}",
            expected[axis],
            actual[axis]
        );
    }
}

fn subtract(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    core::array::from_fn(|axis| left[axis] - right[axis])
}

fn dot(left: [f32; 3], right: [f32; 3]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn cross(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

#[test]
fn direct_and_compact_dual_planes_produce_the_same_descriptor() {
    let direct = Multivector::new([-7.0, 6.0, 3.0, 2.0]);
    let dual = Multivector::new([2.0, -3.0, 6.0, -7.0]);
    let expected_normal = [2.0 / 7.0, -3.0 / 7.0, 6.0 / 7.0];

    let direct_descriptor = direct_plane(&direct).unwrap();
    let dual_descriptor = dual_plane(&dual).unwrap();
    assert_eq!(direct_descriptor, dual_descriptor);
    assert_vec3_close(direct_descriptor.point(), expected_normal, f32::EPSILON);
    assert_vec3_close(direct_descriptor.normal(), expected_normal, f32::EPSILON);
}

#[test]
fn direct_plane_preserves_translated_carrier_sign_and_composes_with_tessellators() {
    let descriptor = direct_plane(&Multivector::new([24.0, -8.0, 0.0, 0.0])).unwrap();
    assert_eq!(descriptor.point(), [0.0, 0.0, 3.0]);
    assert_eq!(descriptor.normal(), [0.0, 0.0, -1.0]);

    let options = PlanePatchOptions::new(5, 7).unwrap();
    let patch = plane_patch(&descriptor, 4.0, options).unwrap();
    assert_eq!(patch.vertices().len(), 48);
    assert_eq!(patch.triangle_count(), 70);
    for vertex in patch.vertices() {
        assert!(
            dot(
                subtract(vertex.position(), descriptor.point()),
                descriptor.normal()
            )
            .abs()
                < 1e-5
        );
        assert_eq!(vertex.normal(), descriptor.normal());
    }
    for triangle in patch.triangles() {
        let [a, b, c] = triangle.map(|index| patch.vertices()[index as usize].position());
        let geometric_normal = cross(subtract(b, a), subtract(c, a));
        assert!(dot(geometric_normal, descriptor.normal()) > 0.0);
    }

    let grid = plane_grid(&descriptor, 4.0, options).unwrap();
    assert_eq!(grid.segments().len(), 14);
    assert_eq!(grid.positions().len(), 28);
    for position in grid.positions() {
        assert!(dot(subtract(*position, descriptor.point()), descriptor.normal()).abs() < 1e-5);
    }

    let indicator = plane_normal_indicator(&descriptor, 2.0).unwrap();
    assert_eq!(indicator.segments(), &[[0, 1]]);
    assert_eq!(indicator.positions(), &[[0.0, 0.0, 3.0], [0.0, 0.0, 1.0]]);
}

#[test]
fn negative_plane_scale_preserves_locus_and_reverses_orientation() {
    let dual = Multivector::new([2.0, -3.0, 6.0, -7.0]);
    let direct = Multivector::new([-7.0, 6.0, 3.0, 2.0]);
    let forward = dual_plane(&dual).unwrap();
    let reversed = dual_plane(&(dual * -8.0)).unwrap();
    assert_eq!(dual_plane(&(dual * 65_536.0)).unwrap(), forward);
    assert_eq!(direct_plane(&(direct * 65_536.0)).unwrap(), forward);
    assert_eq!(direct_plane(&(direct * -8.0)).unwrap(), reversed);
    assert_eq!(reversed.point(), forward.point());
    assert_vec3_close(
        reversed.normal(),
        forward.normal().map(|value| -value),
        f32::EPSILON,
    );

    let options = PlanePatchOptions::new(3, 4).unwrap();
    for descriptor in [forward, reversed] {
        let patch = plane_patch(&descriptor, 2.0, options).unwrap();
        for vertex in patch.vertices() {
            assert!(
                dot(
                    subtract(vertex.position(), forward.point()),
                    forward.normal()
                )
                .abs()
                    < 2e-6
            );
            assert_eq!(vertex.normal(), descriptor.normal());
        }
        for triangle in patch.triangles() {
            let [a, b, c] = triangle.map(|index| patch.vertices()[index as usize].position());
            assert!(dot(cross(subtract(b, a), subtract(c, a)), descriptor.normal()) > 0.0);
        }
        let indicator = plane_normal_indicator(&descriptor, 2.0).unwrap();
        assert_vec3_close(
            subtract(indicator.positions()[1], indicator.positions()[0]),
            descriptor.normal().map(|value| value * 2.0),
            2e-6,
        );
    }
}

#[test]
fn malformed_planes_preserve_representation_specific_errors() {
    let subnormal = f32::from_bits(1);
    let cases = [
        (
            dual_plane(&Multivector::new([0.0, 0.0, 0.0, 1.0])),
            CgaMeshError::DualPlaneDecomposition(DecompositionError::DegeneratePlaneNormal),
        ),
        (
            direct_plane(&Multivector::new([1.0, 0.0, 0.0, 0.0])),
            CgaMeshError::DirectPlaneDecomposition(DecompositionError::DegeneratePlaneNormal),
        ),
        (
            dual_plane(&Multivector::new([f32::INFINITY, 0.0, 0.0, 1.0])),
            CgaMeshError::DualPlaneDecomposition(DecompositionError::NonFiniteInput),
        ),
        (
            direct_plane(&Multivector::new([f32::NAN, 0.0, 0.0, 1.0])),
            CgaMeshError::DirectPlaneDecomposition(DecompositionError::NonFiniteInput),
        ),
        (
            dual_plane(&Multivector::new([subnormal, 0.0, 0.0, f32::MAX])),
            CgaMeshError::DualPlaneDecomposition(DecompositionError::OutputOutOfRange),
        ),
        (
            direct_plane(&Multivector::new([f32::MAX, 0.0, 0.0, subnormal])),
            CgaMeshError::DirectPlaneDecomposition(DecompositionError::OutputOutOfRange),
        ),
    ];
    for (actual, expected) in cases {
        let error = actual.unwrap_err();
        assert_eq!(error, expected);
        assert_eq!(
            error.source().unwrap().to_string(),
            expected.source().unwrap().to_string()
        );
    }
}

#[test]
fn real_dual_sphere_composes_with_existing_tessellators() {
    let value = Round::dls(&point(1.5, -2.0, 0.25), 3.0);
    let sphere = real_dual_sphere(&value).unwrap();
    assert_eq!(sphere.center(), [1.5, -2.0, 0.25]);
    assert!((sphere.radius() - 3.0).abs() <= f32::EPSILON);

    let uv = uv_sphere(&sphere, UvSphereOptions::uniform(8).unwrap()).unwrap();
    assert_eq!(uv.vertices().len(), 58);
    assert_eq!(uv.triangle_count(), 112);
    let ico = icosphere(&sphere, IcosphereOptions::new(1).unwrap()).unwrap();
    assert_eq!(ico.vertices().len(), 42);
    assert_eq!(ico.triangle_count(), 80);

    for vertex in uv.vertices().iter().chain(ico.vertices()) {
        let offset =
            core::array::from_fn::<_, 3, _>(|axis| vertex.position()[axis] - sphere.center()[axis]);
        let radius = offset.iter().map(|value| value * value).sum::<f32>().sqrt();
        let radial_dot = offset
            .iter()
            .zip(vertex.normal())
            .map(|(component, normal)| component * normal)
            .sum::<f32>();
        assert!((radius - sphere.radius()).abs() < 2e-5);
        assert!(radial_dot > 0.0);
    }
}

#[test]
fn homogeneous_scale_does_not_change_descriptor_or_mesh_orientation() {
    let value = Round::dls(&point(-0.25, 4.0, 2.5), 1.5);
    let expected = real_dual_sphere(&value).unwrap();
    let options = IcosphereOptions::new(1).unwrap();
    let expected_mesh = icosphere(&expected, options).unwrap();

    for scale in [-65_536.0_f32, -1.0 / 4096.0, 1.0 / 4096.0, 65_536.0] {
        let actual = real_dual_sphere(&(value * scale)).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(icosphere(&actual, options).unwrap(), expected_mesh);
    }
}

#[test]
fn point_cloud_is_ordered_atomic_and_accepts_empty_input() {
    let values = [
        point(1.0, 2.0, 3.0),
        point(-4.0, 5.0, -6.0) * -8.0,
        point(0.25, -0.5, 0.75) * 4.0,
    ];
    let cloud = point_cloud(&values).unwrap();
    assert_eq!(
        cloud.positions(),
        &[[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0], [0.25, -0.5, 0.75]]
    );
    assert!(point_cloud(&[]).unwrap().positions().is_empty());

    let invalid = Round::dls(&point(0.0, 0.0, 0.0), 2.0);
    let error = point_cloud(&[values[0], invalid, values[2]]).unwrap_err();
    assert_eq!(
        error,
        CgaMeshError::PointDecomposition {
            index: 1,
            source: DecompositionError::NonNullPoint,
        }
    );
}

#[test]
fn point_spheres_and_imaginary_spheres_are_not_fabricated() {
    assert_eq!(
        real_dual_sphere(&point(1.0, 2.0, 3.0)),
        Err(CgaMeshError::Mesh(MeshError::NonPositive("sphere radius")))
    );

    let imaginary = Multivector::new([0.0, 0.0, 0.0, -0.5, 0.5 + f32::EPSILON]);
    assert_eq!(
        real_dual_sphere(&imaginary),
        Err(CgaMeshError::DualSphereDecomposition(
            DecompositionError::ImaginaryDualSphere
        ))
    );
}

#[test]
fn malformed_dual_spheres_preserve_decomposition_errors() {
    let cases = [
        (
            Multivector::new([1.0, 2.0, 3.0, 4.0, 4.0]),
            DecompositionError::DegenerateHomogeneousWeight,
        ),
        (
            Multivector::new([f32::INFINITY, 0.0, 0.0, -0.5, 0.5]),
            DecompositionError::NonFiniteInput,
        ),
        (
            Multivector::new([f32::MAX, 0.0, 0.0, 0.0, f32::from_bits(1)]),
            DecompositionError::OutputOutOfRange,
        ),
    ];
    for (value, source) in cases {
        let error = real_dual_sphere(&value).unwrap_err();
        assert_eq!(error, CgaMeshError::DualSphereDecomposition(source));
        assert_eq!(error.source().unwrap().to_string(), source.to_string());
    }
}
