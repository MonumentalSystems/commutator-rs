use std::f64::consts::{FRAC_PI_2, PI};

use superconducting_dynamics::{
    Complex64, DynamicsError, GaugeLink, JosephsonJunction, LoopEdge, OrderParameter,
    SiteParameters, TdglIntegrator, TdglModel,
};

fn local(alpha: f64, beta: f64) -> SiteParameters {
    SiteParameters::try_new(alpha, beta, 1.0).unwrap()
}

#[test]
fn uniform_zero_field_state_has_analytic_energy_and_gradient() {
    let model = TdglModel::uniform_rectangular(3, 2, local(-2.0, 0.5), 3.0).unwrap();
    let psi = Complex64::from_polar(1.25, 0.37);
    let state = OrderParameter::uniform(&model, psi).unwrap();
    let rho = psi.norm_sqr();
    let expected_energy = 6.0 * (-2.0 * rho + 0.25 * rho * rho);
    assert!((model.free_energy(&state).unwrap() - expected_energy).abs() < 1e-12);

    let expected_gradient = (-2.0 + 0.5 * rho) * psi;
    for value in model.gradient(&state).unwrap() {
        assert!((value - expected_gradient).norm() < 1e-12);
    }
}

#[test]
fn analytic_gradient_matches_central_finite_differences() {
    let model = TdglModel::try_new(
        vec![local(-0.7, 1.2), local(0.3, 0.8), local(-0.1, 1.7)],
        vec![
            GaugeLink::try_new(0, 1, 0.9, 0.27).unwrap(),
            GaugeLink::try_new(1, 2, 1.3, -0.41).unwrap(),
            GaugeLink::try_new(2, 0, 0.4, 0.16).unwrap(),
        ],
    )
    .unwrap();
    let base = vec![
        Complex64::new(0.7, -0.2),
        Complex64::new(-0.1, 0.8),
        Complex64::new(0.4, 0.3),
    ];
    let state = OrderParameter::try_new(&model, base.clone()).unwrap();
    let gradient = model.gradient(&state).unwrap();
    let epsilon = 1e-6;

    for site in 0..base.len() {
        for imaginary in [false, true] {
            let mut plus = base.clone();
            let mut minus = base.clone();
            if imaginary {
                plus[site].im += epsilon;
                minus[site].im -= epsilon;
            } else {
                plus[site].re += epsilon;
                minus[site].re -= epsilon;
            }
            let plus = OrderParameter::try_new(&model, plus).unwrap();
            let minus = OrderParameter::try_new(&model, minus).unwrap();
            let numerical = (model.free_energy(&plus).unwrap()
                - model.free_energy(&minus).unwrap())
                / (2.0 * epsilon);
            let analytic = if imaginary {
                2.0 * gradient[site].im
            } else {
                2.0 * gradient[site].re
            };
            assert!(
                (numerical - analytic).abs() < 2e-9,
                "site={site} imaginary={imaginary}: numerical={numerical}, analytic={analytic}"
            );
        }
    }
}

#[test]
fn gauge_transform_preserves_energy_gradient_covariance_current_and_flux() {
    // Square: 0 -> 1 -> 3 -> 2 -> 0. Rectangular constructor stores
    // 0->1, 0->2, 1->3, 2->3 in that order.
    let model = TdglModel::try_new(
        vec![local(-1.0, 1.0); 4],
        vec![
            GaugeLink::try_new(0, 1, 0.8, 0.11).unwrap(),
            GaugeLink::try_new(0, 2, 0.9, -0.22).unwrap(),
            GaugeLink::try_new(1, 3, 1.1, 0.37).unwrap(),
            GaugeLink::try_new(2, 3, 1.2, -0.09).unwrap(),
        ],
    )
    .unwrap();
    let state = OrderParameter::try_new(
        &model,
        vec![
            Complex64::new(0.7, 0.1),
            Complex64::new(-0.2, 0.9),
            Complex64::new(0.4, -0.3),
            Complex64::new(-0.6, -0.2),
        ],
    )
    .unwrap();
    let path = [
        LoopEdge::forward(0),
        LoopEdge::forward(2),
        LoopEdge::reverse(3),
        LoopEdge::reverse(1),
    ];
    let chi = [0.4, -0.7, 1.2, 0.05];
    let energy = model.free_energy(&state).unwrap();
    let gradient = model.gradient(&state).unwrap();
    let flux = model.loop_phase(&path).unwrap();
    let currents = (0..model.link_count())
        .map(|link| model.link_current(&state, link).unwrap())
        .collect::<Vec<_>>();

    let (transformed_model, transformed_state) = model.gauge_transform(&state, &chi).unwrap();
    assert!((transformed_model.free_energy(&transformed_state).unwrap() - energy).abs() < 2e-14);
    assert!((transformed_model.loop_phase(&path).unwrap() - flux).abs() < 2e-15);
    for (link, &current) in currents.iter().enumerate() {
        assert!(
            (transformed_model
                .link_current(&transformed_state, link)
                .unwrap()
                - current)
                .abs()
                < 2e-14
        );
    }
    for (site, (&before, &after)) in gradient
        .iter()
        .zip(
            transformed_model
                .gradient(&transformed_state)
                .unwrap()
                .iter(),
        )
        .enumerate()
    {
        let expected = Complex64::from_polar(1.0, chi[site]) * before;
        assert!((after - expected).norm() < 3e-14);
    }
}

#[test]
fn link_current_has_expected_josephson_phase_relation() {
    let stiffness = 0.75;
    let phase = 0.2;
    let model = TdglModel::try_new(
        vec![local(0.0, 1.0); 2],
        vec![GaugeLink::try_new(0, 1, stiffness, phase).unwrap()],
    )
    .unwrap();
    let theta_from = -0.3;
    let theta_to = 0.8;
    let state = OrderParameter::try_new(
        &model,
        vec![
            Complex64::from_polar(2.0, theta_from),
            Complex64::from_polar(1.5, theta_to),
        ],
    )
    .unwrap();
    let expected = 2.0 * stiffness * 2.0 * 1.5 * (theta_to - theta_from - phase).sin();
    assert!((model.link_current(&state, 0).unwrap() - expected).abs() < 1e-14);

    // It also equals -dF/dA with order parameters held fixed.
    let epsilon = 1e-6;
    let energy_at = |link_phase| {
        let shifted = TdglModel::try_new(
            vec![local(0.0, 1.0); 2],
            vec![GaugeLink::try_new(0, 1, stiffness, link_phase).unwrap()],
        )
        .unwrap();
        let shifted_state = OrderParameter::try_new(&shifted, state.as_slice().to_vec()).unwrap();
        shifted.free_energy(&shifted_state).unwrap()
    };
    let minus_derivative =
        -(energy_at(phase + epsilon) - energy_at(phase - epsilon)) / (2.0 * epsilon);
    assert!((minus_derivative - expected).abs() < 2e-8);
}

#[test]
fn josephson_junction_energy_and_current_share_one_derivative() {
    let junction = JosephsonJunction::try_new(2.5, 3.0, PI).unwrap();
    assert!((junction.critical_current() - 7.5).abs() < 1e-15);
    assert!((junction.current(PI + FRAC_PI_2).unwrap() - 7.5).abs() < 1e-14);
    let phase = 0.73;
    let epsilon = 1e-6;
    let derivative = (junction.energy(phase + epsilon).unwrap()
        - junction.energy(phase - epsilon).unwrap())
        / (2.0 * epsilon);
    assert!((junction.current(phase).unwrap() - 3.0 * derivative).abs() < 1e-8);
}

#[test]
fn tdgl_backtracking_makes_energy_monotone_and_relaxes_uniform_state() {
    let model = TdglModel::uniform_rectangular(4, 4, local(-1.0, 1.0), 2.0).unwrap();
    let mut state = OrderParameter::uniform(&model, Complex64::new(0.1, 0.0)).unwrap();
    let integrator = TdglIntegrator::reference(100.0).unwrap();
    let mut previous = model.free_energy(&state).unwrap();
    let mut observed_backtracking = false;
    for _ in 0..80 {
        let report = integrator.step(&model, &mut state).unwrap();
        observed_backtracking |= report.backtracks > 0;
        assert!(report.energy_after <= previous + 1e-12);
        previous = report.energy_after;
    }
    assert!(observed_backtracking);
    // Uniform equilibrium amplitude is sqrt(-alpha / beta) = 1.
    assert!((state.as_slice()[0].norm() - 1.0).abs() < 1e-6);
}

#[test]
fn rejected_step_is_transactional() {
    let model = TdglModel::try_new(vec![local(-1.0, 1.0)], vec![]).unwrap();
    let original = OrderParameter::uniform(&model, Complex64::new(0.25, 0.0)).unwrap();
    let mut state = original.clone();
    let integrator = TdglIntegrator::try_new(1e308, 1e308, 0, 0.0).unwrap();
    let error = integrator.step(&model, &mut state).unwrap_err();
    assert!(matches!(error, DynamicsError::NoDescentStep { .. }));
    assert_eq!(state, original);
}

#[test]
fn loop_validation_rejects_disconnected_path() {
    let model = TdglModel::try_new(
        vec![local(-1.0, 1.0); 3],
        vec![
            GaugeLink::try_new(0, 1, 1.0, 0.0).unwrap(),
            GaugeLink::try_new(2, 0, 1.0, 0.0).unwrap(),
        ],
    )
    .unwrap();
    assert!(matches!(
        model.loop_phase(&[LoopEdge::forward(0), LoopEdge::forward(1)]),
        Err(DynamicsError::OpenLoop { position: 1 })
    ));
}

#[test]
fn snapshot_produces_scaled_bdg_pairing_gaps() {
    let model = TdglModel::try_new(vec![local(-1.0, 1.0); 2], vec![]).unwrap();
    let state = OrderParameter::try_new(
        &model,
        vec![Complex64::new(0.4, 0.1), Complex64::new(-0.2, 0.3)],
    )
    .unwrap();
    let gaps = model.scaled_pairing_gaps(&state, 1.5).unwrap();
    assert_eq!(gaps.len(), model.site_count());
    assert_eq!(gaps[0], 1.5 * state.as_slice()[0]);
    assert_eq!(gaps[1], 1.5 * state.as_slice()[1]);
}
