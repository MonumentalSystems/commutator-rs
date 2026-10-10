#![cfg(feature = "cga3d")]

use std::error::Error;

use clifford_geometry::cga3d::decomposition::DecompositionError;
use clifford_geometry::cga3d::{point, Round};
use clifford_geometry::mvec::Multivector;
use clifford_mesh::cga3d::{point_cloud, real_dual_sphere, CgaMeshError};
use clifford_mesh::primitives::{icosphere, uv_sphere};
use clifford_mesh::{IcosphereOptions, MeshError, UvSphereOptions};

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
