use core::f64::consts::{FRAC_1_SQRT_2, PI};

use quantum_magnetism::{Bond, Complex64, LanczosConfig, MagnetismError, SpinAxis, SpinModel};

fn singlet() -> Vec<Complex64> {
    vec![
        Complex64::ZERO,
        Complex64::from(FRAC_1_SQRT_2),
        Complex64::from(-FRAC_1_SQRT_2),
        Complex64::ZERO,
    ]
}

#[test]
fn heisenberg_dimer_has_analytic_singlet_energy() {
    let model = SpinModel::builder(2)
        .bond(Bond::heisenberg(0, 1, 1.0))
        .build()
        .unwrap();
    assert!((model.energy(&singlet()).unwrap() + 0.75).abs() < 1.0e-14);

    let triplet = vec![
        Complex64::ZERO,
        Complex64::ZERO,
        Complex64::ZERO,
        Complex64::ONE,
    ];
    assert!((model.energy(&triplet).unwrap() - 0.25).abs() < 1.0e-14);
}

#[test]
fn singlet_observables_match_closed_form() {
    let model = SpinModel::builder(2).build().unwrap();
    let state = singlet();
    let magnetization = model.total_magnetization(&state).unwrap();
    assert!(magnetization.iter().all(|value| value.abs() < 1.0e-14));

    for axis in [SpinAxis::X, SpinAxis::Y, SpinAxis::Z] {
        let correlation = model.spin_correlation(&state, 0, axis, 1, axis).unwrap();
        assert!((correlation.re + 0.25).abs() < 1.0e-14);
        assert!(correlation.im.abs() < 1.0e-14);
    }

    let positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let uniform = model
        .static_structure_factor(&state, &positions, [0.0; 3], SpinAxis::Z)
        .unwrap();
    let staggered = model
        .static_structure_factor(&state, &positions, [PI, 0.0, 0.0], SpinAxis::Z)
        .unwrap();
    assert!(uniform.abs() < 1.0e-14);
    assert!((staggered - 0.5).abs() < 1.0e-14);
}

#[test]
fn longitudinal_field_uses_minus_h_sz_convention() {
    let model = SpinModel::builder(1)
        .longitudinal_field(0, 2.0)
        .build()
        .unwrap();
    let down = [Complex64::ONE, Complex64::ZERO];
    let up = [Complex64::ZERO, Complex64::ONE];
    assert!((model.energy(&down).unwrap() - 1.0).abs() < 1.0e-14);
    assert!((model.energy(&up).unwrap() + 1.0).abs() < 1.0e-14);
}

#[test]
fn dm_hamiltonian_is_hermitian() {
    let model = SpinModel::builder(3)
        .bond(Bond::xyz(0, 1, 0.7, -0.2, 1.3).with_dm([0.3, -0.5, 0.9]))
        .bond(Bond::heisenberg(1, 2, -0.4).with_dm([-0.1, 0.2, 0.6]))
        .build()
        .unwrap();
    let left: Vec<_> = (0..8)
        .map(|i| Complex64::new(i as f64 - 2.0, 0.3 * i as f64))
        .collect();
    let right: Vec<_> = (0..8)
        .map(|i| Complex64::new(0.2 * i as f64 + 0.1, 1.0 - i as f64))
        .collect();
    let h_left = model.applied(&left).unwrap();
    let h_right = model.applied(&right).unwrap();
    let lhs: Complex64 = left.iter().zip(&h_right).map(|(a, b)| a.conj() * *b).sum();
    let rhs: Complex64 = h_left.iter().zip(&right).map(|(a, b)| a.conj() * *b).sum();
    assert!((lhs - rhs).norm() < 1.0e-12);
}

#[test]
fn lanczos_recovers_dimer_ground_state() {
    let model = SpinModel::builder(2)
        .bond(Bond::heisenberg(0, 1, 1.0))
        .build()
        .unwrap();
    let result = model
        .ground_state(LanczosConfig {
            max_iterations: 4,
            tolerance: 1.0e-13,
            seed: 7,
        })
        .unwrap();
    assert!((result.energy + 0.75).abs() < 1.0e-12);
    assert!(result.residual_norm < 1.0e-12);
    assert!(result.converged);
    assert!((model.energy(&result.state).unwrap() - result.energy).abs() < 1.0e-12);
}

#[test]
fn invalid_graph_state_and_solver_inputs_are_rejected() {
    assert_eq!(
        SpinModel::builder(0).build().unwrap_err(),
        MagnetismError::EmptyModel
    );
    assert!(matches!(
        SpinModel::builder(2)
            .bond(Bond::heisenberg(0, 2, 1.0))
            .build(),
        Err(MagnetismError::SiteOutOfBounds { site: 2, spins: 2 })
    ));
    assert!(matches!(
        SpinModel::builder(2)
            .bond(Bond::heisenberg(1, 1, 1.0))
            .build(),
        Err(MagnetismError::SelfBond { site: 1 })
    ));
    assert!(matches!(
        SpinModel::builder(2).longitudinal_field(9, 1.0).build(),
        Err(MagnetismError::SiteOutOfBounds { site: 9, spins: 2 })
    ));
    let model = SpinModel::builder(1).build().unwrap();
    assert!(matches!(
        model.energy(&[Complex64::ZERO, Complex64::ZERO]),
        Err(MagnetismError::ZeroNorm)
    ));
    assert!(matches!(
        model.ground_state(LanczosConfig {
            max_iterations: 0,
            ..LanczosConfig::default()
        }),
        Err(MagnetismError::InvalidSolverParameter(_))
    ));
}
