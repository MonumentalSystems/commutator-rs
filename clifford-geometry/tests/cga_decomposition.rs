use clifford_geometry::cga3d::decomposition::{
    decompose_point, decompose_real_dual_sphere, DecompositionError,
};
use clifford_geometry::cga3d::{biv, point, spin_rot_pnt, translate, Gen, Round};
use clifford_geometry::mvec::Multivector;

fn assert_vec3_close(actual: [f32; 3], expected: [f32; 3], tolerance: f32) {
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} != {expected}"
        );
    }
}

#[test]
fn point_decomposition_matches_frozen_raw_layout() {
    let value = point(1.0, -2.0, 3.0);
    assert_eq!(value.data, [1.0, -2.0, 3.0, 6.5, 7.5]);
    let parameters = decompose_point(&value).unwrap();
    assert_eq!(parameters.position(), [1.0, -2.0, 3.0]);
}

#[test]
fn canonical_decimal_points_round_trip_without_spurious_radius() {
    let coordinates = [
        [0.1_f32, 0.2, 0.3],
        [1.1, 2.2, 3.3],
        [10.1, 0.0, 0.0],
        [100.1, 0.0, 0.0],
        [1000.1, 0.0, 0.0],
        [
            core::f32::consts::PI,
            core::f32::consts::E,
            core::f32::consts::SQRT_2,
        ],
    ];
    for expected in coordinates {
        let value = point(expected[0], expected[1], expected[2]);
        assert_eq!(decompose_point(&value).unwrap().position(), expected);
        assert_eq!(decompose_real_dual_sphere(&value).unwrap().radius(), 0.0);
    }
}

#[test]
fn point_decomposition_accepts_semantics_preserving_homogeneous_scaling() {
    let value = point(0.25, -4.0, 2.5);
    for scale in [7.0_f32, -3.5, f32::MIN_POSITIVE] {
        let scaled = value * scale;
        let parameters = decompose_point(&scaled).unwrap();
        assert_vec3_close(parameters.position(), [0.25, -4.0, 2.5], 2e-5);
    }
}

#[test]
fn lossy_homogeneous_scaling_is_not_silently_accepted_as_a_point() {
    let scaled = point(0.1, 0.2, 0.3) * 1.3;
    assert_eq!(
        decompose_point(&scaled),
        Err(DecompositionError::NonNullPoint)
    );
}

#[test]
fn point_decomposition_rejects_invalid_semantics() {
    let zero_weight = Multivector::new([1.0, 2.0, 3.0, 4.0, 4.0]);
    assert_eq!(
        decompose_point(&zero_weight),
        Err(DecompositionError::DegenerateHomogeneousWeight)
    );

    let non_null = Multivector::new([1.0, 0.0, 0.0, -0.5, 0.5]);
    assert_eq!(
        decompose_point(&non_null),
        Err(DecompositionError::NonNullPoint)
    );

    let non_finite = Multivector::new([f32::NAN, 0.0, 0.0, -0.5, 0.5]);
    assert_eq!(
        decompose_point(&non_finite),
        Err(DecompositionError::NonFiniteInput)
    );
}

#[test]
fn every_non_finite_slot_and_infinite_element_is_rejected() {
    for index in 0..5 {
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut coefficients = point(1.0, 2.0, 3.0).data;
            coefficients[index] = invalid;
            let value = Multivector::new(coefficients);
            assert_eq!(
                decompose_point(&value),
                Err(DecompositionError::NonFiniteInput)
            );
            assert_eq!(
                decompose_real_dual_sphere(&value),
                Err(DecompositionError::NonFiniteInput)
            );
        }
    }

    for value in [
        Multivector::new([0.0; 5]),
        Multivector::new([0.0, 0.0, 0.0, 1.0, 1.0]),
    ] {
        assert_eq!(
            decompose_point(&value),
            Err(DecompositionError::DegenerateHomogeneousWeight)
        );
        assert_eq!(
            decompose_real_dual_sphere(&value),
            Err(DecompositionError::DegenerateHomogeneousWeight)
        );
    }
}

#[test]
fn finite_inputs_with_unrepresentable_outputs_are_rejected() {
    let value = Multivector::new([f32::MAX, 0.0, 0.0, 0.0, f32::from_bits(1)]);
    assert_eq!(
        decompose_real_dual_sphere(&value),
        Err(DecompositionError::OutputOutOfRange)
    );
}

#[test]
fn maximum_finite_homogeneous_scale_remains_supported_when_representable() {
    let origin = point(0.0, 0.0, 0.0) * f32::MAX;
    assert_eq!(decompose_point(&origin).unwrap().position(), [0.0; 3]);

    let unit_sphere = Round::dls(&point(0.0, 0.0, 0.0), 1.0) * f32::MAX;
    let parameters = decompose_real_dual_sphere(&unit_sphere).unwrap();
    assert_eq!(parameters.center(), [0.0; 3]);
    assert!((parameters.radius() - 1.0).abs() <= f32::EPSILON);
}

#[test]
fn real_dual_sphere_decomposition_handles_scale_and_point_spheres() {
    let center = point(1.5, -2.0, 0.25);
    let sphere = Round::dls(&center, 3.0);
    assert_eq!(sphere.data, [1.5, -2.0, 0.25, -1.84375, -0.84375]);

    for scale in [1.0_f32, 8.0, -2.0, f32::MIN_POSITIVE] {
        let parameters = decompose_real_dual_sphere(&(sphere * scale)).unwrap();
        assert_vec3_close(parameters.center(), [1.5, -2.0, 0.25], 2e-5);
        assert!((parameters.radius() - 3.0).abs() <= 2e-5);
    }

    let point_sphere = decompose_real_dual_sphere(&center).unwrap();
    assert_eq!(point_sphere.radius(), 0.0);
}

#[test]
fn decomposition_is_stable_over_a_deterministic_parameter_grid() {
    let coordinates = [-2.0_f32, -1.0, 0.0, 0.25, 2.0];
    let radii = [0.0_f32, 0.5, 2.0, 10.0];
    // Powers of two preserve the represented f32 coefficients exactly while
    // still exercising tiny, large, and negative homogeneous weights.
    let scales = [-65_536.0_f32, -1.0 / 4096.0, 1.0 / 4096.0, 1.0, 65_536.0];

    for &x in &coordinates {
        for &y in &coordinates {
            let center = point(x, y, 0.75);
            for &radius in &radii {
                let sphere = Round::dls(&center, radius);
                for &scale in &scales {
                    let point_parameters = decompose_point(&(center * scale)).unwrap();
                    assert_vec3_close(point_parameters.position(), [x, y, 0.75], 2e-4);

                    let sphere_parameters = decompose_real_dual_sphere(&(sphere * scale)).unwrap();
                    assert_vec3_close(sphere_parameters.center(), [x, y, 0.75], 2e-4);
                    assert!(
                        (sphere_parameters.radius() - radius).abs() <= 5e-4,
                        "center=({x},{y},0.75) radius={radius} scale={scale}: got {}",
                        sphere_parameters.radius()
                    );
                }
            }
        }
    }
}

#[test]
fn represented_imaginary_offset_is_not_hidden_by_tolerance() {
    let near_null = Multivector::new([0.0, 0.0, 0.0, -0.5, 0.5 + f32::EPSILON]);
    assert_eq!(
        decompose_real_dual_sphere(&near_null),
        Err(DecompositionError::ImaginaryDualSphere)
    );
}

#[test]
fn canonical_reconstruction_preserves_distant_radii() {
    let center = point(100.0, 0.0, 0.0);
    let sphere = Round::dls(&center, 10.0);
    let parameters = decompose_real_dual_sphere(&sphere).unwrap();
    assert_eq!(parameters.center(), [100.0, 0.0, 0.0]);
    assert!((parameters.radius() - 10.0).abs() <= f32::EPSILON);
    assert_eq!(
        decompose_point(&sphere),
        Err(DecompositionError::NonNullPoint)
    );

    let imaginary = Multivector::new([100.0, 0.0, 0.0, center[3] + 50.0, center[4] + 50.0]);
    assert_eq!(
        decompose_real_dual_sphere(&imaginary),
        Err(DecompositionError::ImaginaryDualSphere)
    );

    let distant_center = point(1000.0, 0.0, 0.0);
    for radius in [0.0_f32, 1.0] {
        let distant_sphere = Round::dls(&distant_center, radius);
        for scale in [-1024.0_f32, 1.0 / 1024.0, 1.0, 1024.0] {
            let distant_parameters = decompose_real_dual_sphere(&(distant_sphere * scale)).unwrap();
            assert_eq!(distant_parameters.center(), [1000.0, 0.0, 0.0]);
            assert!((distant_parameters.radius() - radius).abs() <= f32::EPSILON);
        }
    }

    let smallest_distinct = Round::dls(&distant_center, 0.25);
    assert_eq!(
        decompose_point(&smallest_distinct),
        Err(DecompositionError::NonNullPoint)
    );
    assert_eq!(
        decompose_real_dual_sphere(&smallest_distinct)
            .unwrap()
            .radius(),
        0.25
    );

    let distant_imaginary = Multivector::new([
        1000.0,
        0.0,
        0.0,
        distant_center[3] + 0.5,
        distant_center[4] + 0.5,
    ]);
    assert_eq!(
        decompose_real_dual_sphere(&distant_imaginary),
        Err(DecompositionError::ImaginaryDualSphere)
    );
}

#[test]
fn imaginary_dual_spheres_are_rejected_without_absolute_value_fallback() {
    let imaginary = Multivector::new([0.0, 0.0, 0.0, 0.0, 1.0]);
    for scale in [1.0_f32, -3.0, 1.0e-20, 1.0e20] {
        assert_eq!(
            decompose_real_dual_sphere(&(imaginary * scale)),
            Err(DecompositionError::ImaginaryDualSphere)
        );
    }
}

#[test]
fn decomposition_is_equivariant_under_translation_and_rotation() {
    let point_value = point(1.0, 2.0, -0.5);
    let translated = translate(&point_value, 4.0, -1.0, 2.5);
    assert_vec3_close(
        decompose_point(&translated).unwrap().position(),
        [5.0, 1.0, 2.0],
        2e-5,
    );

    let sphere = Round::dls(&point_value, 1.25);
    let translated_sphere = translate(&sphere, 4.0, -1.0, 2.5);
    let translated_parameters = decompose_real_dual_sphere(&translated_sphere).unwrap();
    assert_vec3_close(translated_parameters.center(), [5.0, 1.0, 2.0], 2e-5);
    assert!((translated_parameters.radius() - 1.25).abs() <= 2e-5);

    let rotor = Gen::rot(&biv(core::f32::consts::FRAC_PI_2, 0.0, 0.0));
    let rotated_sphere = spin_rot_pnt(&rotor, &sphere);
    let rotated_parameters = decompose_real_dual_sphere(&rotated_sphere).unwrap();
    assert_vec3_close(rotated_parameters.center(), [-2.0, 1.0, -0.5], 2e-5);
    assert!((rotated_parameters.radius() - 1.25).abs() <= 2e-5);
}
