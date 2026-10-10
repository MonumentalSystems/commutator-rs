use spin_lattice::{
    Bond, LinearExchange, SpinLatticeError, SpinLatticeModel, SpinLatticeState, Vec3,
};

fn two_atom_model(
    spring: f64,
    reference_exchange: f64,
    exchange_derivative: f64,
) -> SpinLatticeModel {
    SpinLatticeModel::try_new(
        vec![1.0, 1.0],
        vec![Bond::new(
            0,
            1,
            Vec3::new(1.0, 0.0, 0.0),
            spring,
            LinearExchange::new(reference_exchange, exchange_derivative),
        )],
    )
    .unwrap()
}

fn parallel_state(model: &SpinLatticeModel, extension: f64) -> SpinLatticeState {
    SpinLatticeState::try_new(
        model,
        vec![Vec3::ZERO, Vec3::new(extension, 0.0, 0.0)],
        vec![Vec3::ZERO; 2],
        vec![Vec3::new(0.0, 0.0, 1.0); 2],
    )
    .unwrap()
}

#[test]
fn energy_components_follow_shared_hamiltonian() {
    let model = two_atom_model(2.0, 1.0, -0.5);
    let state = parallel_state(&model, 0.1);
    let energy = model.evaluate(&state).unwrap().energy();

    assert!((energy.elastic() - 0.01).abs() < 1e-14);
    assert!((energy.exchange() - -0.95).abs() < 1e-14);
    assert_eq!(energy.kinetic(), 0.0);
    assert!((energy.total() - -0.94).abs() < 1e-14);
}

#[test]
fn force_matches_energy_finite_difference() {
    let model = two_atom_model(3.0, 0.7, -0.4);
    let state = parallel_state(&model, 0.13);
    let analytic_force = model.evaluate(&state).unwrap().forces()[0].x;
    let epsilon = 1e-6;

    let mut plus = state.clone();
    plus.set_displacement(0, Vec3::new(epsilon, 0.0, 0.0))
        .unwrap();
    let mut minus = state;
    minus
        .set_displacement(0, Vec3::new(-epsilon, 0.0, 0.0))
        .unwrap();
    let plus_energy = model.evaluate(&plus).unwrap().energy().total();
    let minus_energy = model.evaluate(&minus).unwrap().energy().total();
    let finite_difference_force = -(plus_energy - minus_energy) / (2.0 * epsilon);

    assert!((analytic_force - finite_difference_force).abs() < 1e-9);
    let plus_evaluation = model.evaluate(&plus).unwrap();
    let forces = plus_evaluation.forces();
    assert!((forces[0] + forces[1]).norm() < 1e-14);
}

#[test]
fn velocity_verlet_has_small_energy_drift_for_harmonic_dimer() {
    let model = two_atom_model(5.0, 0.0, 0.0);
    let mut state = parallel_state(&model, 0.1);
    let initial = model.evaluate(&state).unwrap().energy().total();

    for _ in 0..10_000 {
        model.step_velocity_verlet(&mut state, 0.001).unwrap();
    }

    let final_energy = model.evaluate(&state).unwrap().energy().total();
    assert!((final_energy - initial).abs() / initial.abs() < 1e-5);
}

#[test]
fn spin_precession_preserves_norm_and_zero_field_spin() {
    let model = two_atom_model(0.0, 1.2, 0.0);
    let mut state = SpinLatticeState::stationary(
        &model,
        vec![Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)],
    )
    .unwrap();
    for _ in 0..1_000 {
        model.step_spin_precession(&mut state, 0.003, 1.5).unwrap();
    }
    assert!(model.max_spin_norm_error(&state).unwrap() < 1e-14);

    let isolated = SpinLatticeModel::try_new(vec![1.0], vec![]).unwrap();
    let spin = Vec3::new(0.0, 0.6, 0.8);
    let mut isolated_state = SpinLatticeState::stationary(&isolated, vec![spin]).unwrap();
    isolated
        .step_spin_precession(&mut isolated_state, 0.5, 2.0)
        .unwrap();
    assert_eq!(isolated_state.spins()[0], spin);
}

#[test]
fn effective_field_is_negative_spin_energy_gradient() {
    let model = two_atom_model(0.0, 1.1, -0.5);
    let extension = 0.2;
    let state = SpinLatticeState::try_new(
        &model,
        vec![Vec3::ZERO, Vec3::new(extension, 0.0, 0.0)],
        vec![Vec3::ZERO; 2],
        vec![Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)],
    )
    .unwrap();
    let expected_exchange = 1.1 - 0.5 * extension;
    let evaluation = model.evaluate(&state).unwrap();
    assert_eq!(
        evaluation.effective_fields()[0],
        Vec3::new(0.0, expected_exchange, 0.0)
    );
    assert_eq!(
        evaluation.effective_fields()[1],
        Vec3::new(expected_exchange, 0.0, 0.0)
    );

    // Rotate spin zero about +z. At theta=0, dE/dtheta must equal
    // -B_0 . (z cross s_0) = -J.
    let epsilon: f64 = 1e-6;
    let energy_at_angle = |angle: f64| {
        let mut varied = state.clone();
        varied
            .set_spin(0, Vec3::new(angle.cos(), angle.sin(), 0.0))
            .unwrap();
        model.evaluate(&varied).unwrap().energy().total()
    };
    let derivative = (energy_at_angle(epsilon) - energy_at_angle(-epsilon)) / (2.0 * epsilon);
    assert!((derivative + expected_exchange).abs() < 1e-10);
}

#[test]
fn magnetization_and_coupled_step_are_finite() {
    let model = two_atom_model(2.0, 0.8, -0.2);
    let mut state = SpinLatticeState::try_new(
        &model,
        vec![Vec3::ZERO, Vec3::new(0.03, 0.0, 0.0)],
        vec![Vec3::new(0.01, 0.0, 0.0), Vec3::new(-0.01, 0.0, 0.0)],
        vec![Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)],
    )
    .unwrap();
    let magnetization = model.magnetization(&state).unwrap();
    assert_eq!(magnetization, Vec3::new(0.5, 0.5, 0.0));
    model.step_coupled(&mut state, 1e-3, 1.0).unwrap();
    assert!(model.evaluate(&state).unwrap().energy().total().is_finite());
    assert!(model.max_spin_norm_error(&state).unwrap() < 1e-14);
}

#[test]
fn invalid_models_states_and_steps_are_rejected() {
    assert_eq!(
        SpinLatticeModel::try_new(vec![], vec![]).unwrap_err(),
        SpinLatticeError::EmptySystem
    );
    assert_eq!(
        SpinLatticeModel::try_new(vec![0.0], vec![]).unwrap_err(),
        SpinLatticeError::InvalidMass { atom: 0 }
    );

    let self_bond = Bond::new(
        0,
        0,
        Vec3::new(1.0, 0.0, 0.0),
        1.0,
        LinearExchange::new(0.0, 0.0),
    );
    assert!(matches!(
        SpinLatticeModel::try_new(vec![1.0], vec![self_bond]),
        Err(SpinLatticeError::SelfBond { .. })
    ));

    let model = two_atom_model(1.0, 0.0, 0.0);
    assert!(matches!(
        SpinLatticeState::stationary(
            &model,
            vec![Vec3::new(2.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)]
        ),
        Err(SpinLatticeError::InvalidSpin { atom: 0, .. })
    ));

    let mut state = parallel_state(&model, 0.0);
    assert_eq!(
        model.step_velocity_verlet(&mut state, 0.0),
        Err(SpinLatticeError::InvalidTimeStep)
    );
}

#[test]
fn failed_collapsed_bond_step_is_transactional() {
    let model = two_atom_model(1.0, 0.0, 0.0);
    let mut state = SpinLatticeState::try_new(
        &model,
        vec![Vec3::ZERO; 2],
        vec![Vec3::ZERO, Vec3::new(-1.0, 0.0, 0.0)],
        vec![Vec3::new(0.0, 0.0, 1.0); 2],
    )
    .unwrap();
    let before = state.clone();
    assert_eq!(
        model.step_velocity_verlet(&mut state, 1.0),
        Err(SpinLatticeError::DegenerateCurrentBond { bond: 0 })
    );
    assert_eq!(state, before);
}
