use clifford_field::{
    chiral_split, step_euler_reference, step_strang_reference, BivectorField, BoundaryCondition,
    StaBivector, StaBivector32, StaBivector64,
};

fn component_field(values: &[f64], boundary: BoundaryCondition) -> BivectorField<f64> {
    let mut field = BivectorField::new_1d(values.len(), 1.0, 1.0, boundary);
    for (point, &value) in field.data.iter_mut().zip(values) {
        point.components[0] = value;
    }
    field
}

#[test]
fn sta_layout_roundtrips_in_both_precisions() {
    let a = StaBivector32::new(1.0, -2.0, 3.0, -4.0, 5.0, -6.0);
    assert_eq!(
        StaBivector::from_sta_multivector(&a.to_sta_multivector()).components,
        a.components
    );

    let b = StaBivector64::new(0.25, -0.5, 0.75, -1.0, 1.25, -1.5);
    assert_eq!(
        StaBivector::from_sta_multivector(&b.to_sta_multivector()).components,
        b.components
    );
}

#[test]
fn sta_commutator_uses_the_frozen_raw_convention() {
    let e1 = StaBivector64::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let e2 = StaBivector64::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
    let e1_e2 = e1.commutator(&e2);
    let e2_e1 = e2.commutator(&e1);

    assert_eq!(e1_e2.components, [0.0, 0.0, 0.0, 0.0, 0.0, -2.0]);
    for component in 0..6 {
        assert_eq!(e1_e2.components[component], -e2_e1.components[component]);
    }
}

#[test]
fn indexing_and_pointwise_velocity_are_dimension_generic() {
    let mut field = BivectorField::new_3d(2, 3, 4, 0.5_f64, 2.0, BoundaryCondition::Periodic);
    assert_eq!(field.n_points(), 24);
    assert_eq!(field.index(&[1, 2, 3]), 23);
    assert_eq!(field.coords(23), [1, 2, 3]);
    assert_eq!(field.v2_at(7), 4.0);

    field.v_squared_per_point = Some((0..24).map(|point| point as f64).collect());
    assert_eq!(field.v2_at(7), 7.0);
}

#[test]
fn boundary_stencils_remain_distinct() {
    let periodic = component_field(&[0.0, 1.0, 4.0], BoundaryCondition::Periodic);
    let fixed = component_field(&[0.0, 1.0, 4.0], BoundaryCondition::Fixed);
    let free = component_field(&[0.0, 1.0, 4.0], BoundaryCondition::Free);

    let periodic_gradient = periodic.spatial_gradient();
    let fixed_gradient = fixed.spatial_gradient();
    let free_gradient = free.spatial_gradient();
    assert_eq!(periodic_gradient[0].components[0], -1.5);
    assert_eq!(fixed_gradient[0].components[0], 0.5);
    assert_eq!(free_gradient[2].components[0], 1.5);

    let periodic_laplacian = periodic.spatial_laplacian();
    let fixed_laplacian = fixed.spatial_laplacian();
    let free_laplacian = free.spatial_laplacian();
    assert_eq!(periodic_laplacian[0].components[0], 5.0);
    assert_eq!(fixed_laplacian[0].components[0], 1.0);
    assert_eq!(free_laplacian[2].components[0], -3.0);
}

#[test]
fn multidimensional_spatial_operators_have_stable_shapes() {
    let mut field = BivectorField::new_2d(3, 2, 1.0_f64, 1.0, BoundaryCondition::Periodic);
    for x in 0..3 {
        for y in 0..2 {
            let index = field.index(&[x, y]);
            field.data[index].components[0] = x as f64;
            field.data[index].components[1] = y as f64;
        }
    }

    assert_eq!(field.spatial_gradient().len(), field.n_points() * 2);
    assert_eq!(field.spatial_laplacian().len(), field.n_points());
    assert_eq!(field.commutator_term().len(), field.n_points());
}

#[test]
fn energy_and_chiral_observables_are_available_without_runtime_code() {
    let bivector = StaBivector64::new(1.0, 2.0, 3.0, 1.0, -2.0, 0.0);
    let (left, right) = chiral_split(&bivector);
    assert_eq!(left, [1.0, 0.0, 1.5]);
    assert_eq!(right, [0.0, 2.0, 1.5]);

    let mut field = BivectorField::new_1d(2, 1.0, 1.0, BoundaryCondition::Periodic);
    field.data[0] = bivector;
    field.data[1] = bivector;
    assert_eq!(field.total_energy(), 38.0);

    let chiral = field.chiral_decompose();
    assert_eq!(chiral.left, vec![left, left]);
    assert_eq!(chiral.right, vec![right, right]);
}

#[test]
fn boundary_condition_json_spelling_is_compatible() {
    assert_eq!(
        serde_json::to_string(&BoundaryCondition::Periodic).unwrap(),
        "\"Periodic\""
    );
    assert_eq!(
        serde_json::from_str::<BoundaryCondition>("\"Free\"").unwrap(),
        BoundaryCondition::Free
    );
}

#[test]
fn reference_steppers_preserve_the_three_pass_contract() {
    let initial = [0.0, 1.0, 0.0];

    let mut euler = component_field(&initial, BoundaryCondition::Periodic);
    step_euler_reference(&mut euler, 0.1);
    let euler_values: Vec<f64> = euler.data.iter().map(|point| point.components[0]).collect();
    assert_eq!(euler_values, [0.1, 0.8, 0.1]);

    let mut strang = component_field(&initial, BoundaryCondition::Periodic);
    step_strang_reference(&mut strang, 0.1);
    let strang_values: Vec<f64> = strang
        .data
        .iter()
        .map(|point| point.components[0])
        .collect();
    for (actual, expected) in strang_values.iter().zip([0.0925, 0.815, 0.0925]) {
        assert!((actual - expected).abs() < 1e-15);
    }

    let mut varying_speed = component_field(&initial, BoundaryCondition::Periodic);
    varying_speed.v_squared_per_point = Some(vec![1.0, 4.0, 1.0]);
    step_euler_reference(&mut varying_speed, 0.1);
    let varying_values: Vec<f64> = varying_speed
        .data
        .iter()
        .map(|point| point.components[0])
        .collect();
    for (actual, expected) in varying_values.iter().zip([0.1, 0.2, 0.1]) {
        assert!((actual - expected).abs() < 1e-15);
    }
}
