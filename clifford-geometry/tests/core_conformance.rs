use clifford_core::CliffordAlgebra;
use clifford_geometry::{adapter, cl3, cl6, ega3d, sta};

fn assert_close(left: &[f32], right: &[f32], tolerance: f32) {
    assert_eq!(left.len(), right.len());
    for (index, (&left, &right)) in left.iter().zip(right).enumerate() {
        assert!(
            (left - right).abs() <= tolerance,
            "coefficient {index}: {left} != {right}"
        );
    }
}

#[test]
fn legacy_cl3_product_maps_by_bitmap() {
    let algebra = CliffordAlgebra::cl3();
    let a = cl3::from_array([1.0, 2.0, -3.0, -5.0, 4.0, 6.0, -7.0, 8.0]);
    let b = cl3::from_array([-2.0, 3.0, 5.0, 11.0, -7.0, -13.0, 17.0, -19.0]);

    let dense_a = adapter::sparse_to_core(&algebra, &cl3::BASIS, &a).unwrap();
    let dense_b = adapter::sparse_to_core(&algebra, &cl3::BASIS, &b).unwrap();
    let expected = algebra.geometric_product(&dense_a, &dense_b);
    let actual =
        adapter::sparse_to_core(&algebra, &cl3::BASIS, &cl3::geometric_product(&a, &b)).unwrap();

    assert_close(&actual, &expected, 0.0);
}

#[test]
fn raw_commutator_scale_is_explicit() {
    let algebra = CliffordAlgebra::cl3();
    let a = cl3::bivector(1.0, -2.0, 3.0);
    let b = cl3::bivector(-4.0, 5.0, 6.0);
    let dense_a = adapter::sparse_to_core(&algebra, &cl3::BASIS, &a).unwrap();
    let dense_b = adapter::sparse_to_core(&algebra, &cl3::BASIS, &b).unwrap();

    let expected = adapter::core_raw_commutator(&algebra, &dense_a, &dense_b);
    let actual = adapter::sparse_to_core(&algebra, &cl3::BASIS, &cl3::commutator(&a, &b)).unwrap();

    assert_close(&actual, &expected, 0.0);
}

#[test]
fn sta_layout_round_trips_through_shortlex() {
    let algebra = CliffordAlgebra::sta();
    let value = sta::em_bivector(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    let dense = adapter::sparse_to_core(&algebra, &sta::BIV_BASIS, &value).unwrap();
    let round_trip = adapter::sparse_from_core(&algebra, &sta::BIV_BASIS, &dense).unwrap();

    assert_eq!(round_trip, value);
    assert_eq!(sta::BIV_BASIS, [3, 5, 9, 12, 10, 6]);
}

#[test]
fn cl6_bivector_product_matches_core() {
    let algebra = CliffordAlgebra::cl6();
    let a = cl6::biv_from_array(std::array::from_fn(|index| index as f32 - 7.0));
    let b = cl6::biv_from_array(std::array::from_fn(|index| 3.0 - index as f32));
    let dense_a = adapter::sparse_to_core(&algebra, &cl6::BIV_BASIS, &a).unwrap();
    let dense_b = adapter::sparse_to_core(&algebra, &cl6::BIV_BASIS, &b).unwrap();
    let expected = algebra.geometric_product(&dense_a, &dense_b);
    let actual =
        adapter::sparse_to_core(&algebra, &cl6::FULL_BASIS, &cl6::geometric_product(&a, &b))
            .unwrap();

    assert_close(&actual, &expected, 0.0);
}

#[test]
fn rotor_sign_adapter_matches_ega() {
    let algebra = CliffordAlgebra::cl3();
    let axis = ega3d::biv(1.0, 0.0, 0.0);
    let dense_axis = adapter::sparse_to_core(&algebra, &ega3d::BIV_BASIS, &axis).unwrap();
    let theta = 0.75;
    let expected = adapter::core_rotor_matching_versor(&algebra, &dense_axis, theta);
    let versor_rotor = ega3d::gen_rot(&(axis * theta));
    let actual = adapter::sparse_to_core(&algebra, &ega3d::ROT_BASIS, &versor_rotor).unwrap();

    assert_close(&actual, &expected, 2e-6);
}

#[test]
fn invalid_adapter_shapes_are_rejected() {
    let algebra = CliffordAlgebra::cl3();
    let error = adapter::sparse_from_core(&algebra, &ega3d::VEC_BASIS, &[0.0; 7]).unwrap_err();
    assert_eq!(
        error,
        adapter::AdapterError::DenseLength {
            expected: 8,
            actual: 7
        }
    );
}
