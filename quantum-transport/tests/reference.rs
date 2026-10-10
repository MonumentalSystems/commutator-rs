use cluster_green::{Complex64, DenseMatrix};
use quantum_transport::{
    broadening, fermi_function, landauer_current_ev, landauer_integral, retarded_device_green,
    two_terminal_transmission, Reservoir, TransportError, ELEMENTARY_CHARGE_COULOMBS,
    PLANCK_CONSTANT_JOULE_SECONDS,
};

#[test]
fn resonant_level_matches_analytic_green_and_transmission() {
    let level = 0.3;
    let energy = 0.1;
    let eta = 0.02;
    let gamma_left = 0.4;
    let gamma_right = 0.6;
    let h = DenseMatrix::from_scalar(level.into()).unwrap();
    let sigma_left = DenseMatrix::from_scalar(Complex64::new(0.0, -gamma_left / 2.0)).unwrap();
    let sigma_right = DenseMatrix::from_scalar(Complex64::new(0.0, -gamma_right / 2.0)).unwrap();

    let result =
        two_terminal_transmission(energy, eta, &h, &sigma_left, &sigma_right, 1.0e-12).unwrap();
    let expected_green = Complex64::new(1.0, 0.0)
        / Complex64::new(energy - level, eta + (gamma_left + gamma_right) / 2.0);
    let expected_transmission = gamma_left * gamma_right
        / ((energy - level).powi(2) + (eta + (gamma_left + gamma_right) / 2.0).powi(2));
    assert!((result.green_retarded.get(0, 0).unwrap() - expected_green).norm() < 1.0e-13);
    assert!((result.gamma_left.get(0, 0).unwrap().re - gamma_left).abs() < 1.0e-14);
    assert!((result.transmission - expected_transmission).abs() < 1.0e-13);
}

#[test]
fn matrix_broadening_and_two_site_green_are_checked() {
    let h = DenseMatrix::try_new(
        2,
        2,
        vec![0.0.into(), (-0.5).into(), (-0.5).into(), 0.0.into()],
    )
    .unwrap();
    let sigma = DenseMatrix::try_new(
        2,
        2,
        vec![
            Complex64::new(0.0, -0.2),
            Complex64::new(0.0, -0.1),
            Complex64::new(0.0, -0.1),
            Complex64::new(0.0, -0.2),
        ],
    )
    .unwrap();
    let gamma = broadening(&sigma, 1.0e-12).unwrap();
    assert!(gamma.is_hermitian(1.0e-12).unwrap());
    let green = retarded_device_green(0.0, 0.01, &h, &[&sigma], 1.0e-12).unwrap();
    let inverse = green.inverse().unwrap();
    let expected_inverse = DenseMatrix::try_new(
        2,
        2,
        vec![
            Complex64::new(0.0, 0.21),
            Complex64::new(0.5, 0.1),
            Complex64::new(0.5, 0.1),
            Complex64::new(0.0, 0.21),
        ],
    )
    .unwrap();
    for (actual, expected) in inverse.as_slice().iter().zip(expected_inverse.as_slice()) {
        assert!((*actual - *expected).norm() < 1.0e-12);
    }
}

#[test]
fn fermi_function_is_stable_at_zero_and_large_arguments() {
    assert_eq!(fermi_function(-1.0, 0.0, 0.0).unwrap(), 1.0);
    assert_eq!(fermi_function(0.0, 0.0, 0.0).unwrap(), 0.5);
    assert_eq!(fermi_function(1.0, 0.0, 0.0).unwrap(), 0.0);
    assert_eq!(fermi_function(1.0e6, 0.0, 0.01).unwrap(), 0.0);
    assert_eq!(fermi_function(-1.0e6, 0.0, 0.01).unwrap(), 1.0);
    assert!((fermi_function(0.0, 0.0, 0.1).unwrap() - 0.5).abs() < 1.0e-15);
}

#[test]
fn trapezoidal_landauer_integral_and_si_current_match() {
    let energies = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0];
    let transmissions = [1.0; 7];
    let left = Reservoir::try_new(1.25, 0.0).unwrap();
    let right = Reservoir::try_new(-0.75, 0.0).unwrap();
    let integral = landauer_integral(&energies, &transmissions, left, right).unwrap();
    assert!((integral - 2.0).abs() < 1.0e-15);
    let current = landauer_current_ev(&energies, &transmissions, left, right, 2.0).unwrap();
    let expected = 4.0 * ELEMENTARY_CHARGE_COULOMBS.powi(2) / PLANCK_CONSTANT_JOULE_SECONDS;
    assert!((current - expected).abs() < 1.0e-15 * expected);
}

#[test]
fn invalid_hamiltonians_self_energies_and_grids_are_rejected() {
    let nonhermitian =
        DenseMatrix::try_new(2, 2, vec![0.0.into(), 1.0.into(), 0.0.into(), 0.0.into()]).unwrap();
    assert!(matches!(
        retarded_device_green(0.0, 0.01, &nonhermitian, &[], 1.0e-12),
        Err(TransportError::NonHermitian("device Hamiltonian"))
    ));

    let advanced = DenseMatrix::from_scalar(Complex64::new(0.0, 0.5)).unwrap();
    assert!(matches!(
        broadening(&advanced, 1.0e-12),
        Err(TransportError::NonCausalSelfEnergy { .. })
    ));
    let tiny_advanced = DenseMatrix::from_scalar(Complex64::new(0.0, 1.0e-200)).unwrap();
    assert!(matches!(
        broadening(&tiny_advanced, 1.0e-12),
        Err(TransportError::NonCausalSelfEnergy { .. })
    ));
    let indefinite_zero_pivot = DenseMatrix::try_new(
        2,
        2,
        vec![
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, -0.5),
            Complex64::new(0.0, -0.5),
            Complex64::new(0.0, 0.0),
        ],
    )
    .unwrap();
    assert!(matches!(
        broadening(&indefinite_zero_pivot, 1.0e-12),
        Err(TransportError::NonCausalSelfEnergy { .. })
    ));
    let h = DenseMatrix::from_scalar(0.0.into()).unwrap();
    assert!(matches!(
        retarded_device_green(0.0, 0.0, &h, &[], 1.0e-12),
        Err(TransportError::NonPositiveRegulator)
    ));

    let reservoir = Reservoir::try_new(0.0, 0.0).unwrap();
    assert!(matches!(
        landauer_integral(&[0.0], &[1.0], reservoir, reservoir),
        Err(TransportError::EnergyGridTooShort)
    ));
    assert!(matches!(
        landauer_integral(&[0.0, -1.0], &[1.0, 1.0], reservoir, reservoir),
        Err(TransportError::UnorderedEnergyGrid { index: 1 })
    ));
    assert!(matches!(
        landauer_integral(&[0.0, 1.0], &[1.0, -0.1], reservoir, reservoir),
        Err(TransportError::NegativeTransmission { .. })
    ));
}
