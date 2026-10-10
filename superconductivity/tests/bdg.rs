use superconductivity::{
    orbital_index, spectrum_particle_hole_residual, BdGEigensystem, Complex64, Hopping,
    NormalHamiltonian, OnsiteSWaveModel, Spin, SuperconductivityError,
};

const TOLERANCE: f64 = 1e-12;

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < TOLERANCE,
        "{actual} differs from {expected}"
    );
}

#[test]
fn one_site_matrix_is_hermitian_and_particle_hole_symmetric() {
    let normal = NormalHamiltonian::spin_independent(&[0.7], &[]).unwrap();
    let model = OnsiteSWaveModel::try_new(normal, 0.2, vec![Complex64::new(0.3, -0.4)]).unwrap();
    let matrix = model.hamiltonian();

    assert_eq!(matrix.dimension(), 4);
    assert!(matrix.is_hermitian(TOLERANCE));
    assert!(matrix.particle_hole_residual() < TOLERANCE);
    assert_eq!(matrix.get(0, 3), Complex64::new(0.3, -0.4));
    assert_eq!(matrix.get(1, 2), Complex64::new(-0.3, 0.4));
}

#[test]
fn complex_hopping_constructs_hermitian_two_site_model() {
    let hopping = Hopping {
        from: 0,
        to: 1,
        amplitude: Complex64::new(-1.0, 0.25),
    };
    let normal = NormalHamiltonian::spin_independent(&[0.1, -0.2], &[hopping]).unwrap();
    let up_zero = orbital_index(0, Spin::Up);
    let up_one = orbital_index(1, Spin::Up);
    assert_eq!(normal.get(up_zero, up_one), hopping.amplitude);
    assert_eq!(normal.get(up_one, up_zero), hopping.amplitude.conj());

    let matrix = OnsiteSWaveModel::try_new(
        normal,
        0.0,
        vec![Complex64::new(0.2, 0.1), Complex64::new(0.4, -0.1)],
    )
    .unwrap()
    .hamiltonian();
    assert!(matrix.is_hermitian(TOLERANCE));
    assert!(matrix.particle_hole_residual() < TOLERANCE);
}

#[test]
fn one_site_analytic_eigensystem_and_observables() {
    let xi: f64 = 0.6;
    let gap: f64 = 0.8;
    let energy = xi.hypot(gap);
    let u = ((energy + xi) / (2.0 * energy)).sqrt();
    let v = ((energy - xi) / (2.0 * energy)).sqrt();

    let normal = NormalHamiltonian::spin_independent(&[xi], &[]).unwrap();
    let matrix = OnsiteSWaveModel::try_new(normal, 0.0, vec![gap.into()])
        .unwrap()
        .hamiltonian();

    // Column-major eigenvectors. The two negative-energy modes precede the
    // two positive-energy modes, but no ordering is required by the API.
    let eigenvectors = vec![
        (-v).into(),
        0.0.into(),
        0.0.into(),
        u.into(),
        0.0.into(),
        v.into(),
        u.into(),
        0.0.into(),
        u.into(),
        0.0.into(),
        0.0.into(),
        v.into(),
        0.0.into(),
        u.into(),
        (-v).into(),
        0.0.into(),
    ];
    let energies = vec![-energy, -energy, energy, energy];
    let eigensystem =
        BdGEigensystem::try_new(&matrix, energies.clone(), eigenvectors, 1e-11).unwrap();

    assert!(spectrum_particle_hole_residual(&energies) < TOLERANCE);
    assert_close(
        eigensystem.singlet_pair_amplitude(0, 0.0).unwrap().re,
        gap / (2.0 * energy),
    );
    let density_at_peak = eigensystem
        .local_density_of_states(0, energy, 0.05)
        .unwrap();
    assert!(density_at_peak > 1.0);
}

#[test]
fn particle_hole_map_reverses_analytic_eigenvalue() {
    let normal = NormalHamiltonian::spin_independent(&[0.0], &[]).unwrap();
    let matrix = OnsiteSWaveModel::try_new(normal, 0.0, vec![1.0.into()])
        .unwrap()
        .hamiltonian();
    let positive = vec![
        (0.5_f64).sqrt().into(),
        0.0.into(),
        0.0.into(),
        (0.5_f64).sqrt().into(),
    ];
    let negative = matrix.particle_hole_conjugate(&positive).unwrap();
    let product = {
        let mut output = vec![Complex64::ZERO; 4];
        for (row, output_entry) in output.iter_mut().enumerate() {
            for (column, &negative_entry) in negative.iter().enumerate() {
                *output_entry += matrix.get(row, column) * negative_entry;
            }
        }
        output
    };
    for component in 0..4 {
        assert!((product[component] + negative[component]).norm() < TOLERANCE);
    }
}

#[test]
fn invalid_inputs_return_specific_errors() {
    assert_eq!(
        NormalHamiltonian::spin_independent(&[], &[]).unwrap_err(),
        SuperconductivityError::EmptyLattice
    );

    let invalid_hopping = Hopping {
        from: 0,
        to: 2,
        amplitude: 1.0.into(),
    };
    assert_eq!(
        NormalHamiltonian::spin_independent(&[0.0, 0.0], &[invalid_hopping]).unwrap_err(),
        SuperconductivityError::SiteOutOfBounds {
            site: 2,
            site_count: 2
        }
    );

    let normal = NormalHamiltonian::spin_independent(&[0.0], &[]).unwrap();
    assert!(matches!(
        OnsiteSWaveModel::try_new(normal, 0.0, vec![]),
        Err(SuperconductivityError::DimensionMismatch { .. })
    ));

    let non_hermitian = vec![0.0.into(), 1.0.into(), 0.0.into(), 0.0.into()];
    assert!(matches!(
        NormalHamiltonian::try_from_dense(1, non_hermitian, 1e-12),
        Err(SuperconductivityError::NotHermitian { .. })
    ));

    assert_eq!(spectrum_particle_hole_residual(&[2.0]), 2.0);
}

#[test]
fn invalid_eigenbasis_is_rejected() {
    let normal = NormalHamiltonian::spin_independent(&[0.0], &[]).unwrap();
    let matrix = OnsiteSWaveModel::try_new(normal, 0.0, vec![1.0.into()])
        .unwrap()
        .hamiltonian();
    let repeated = vec![Complex64::ONE; 16];
    assert!(matches!(
        BdGEigensystem::try_new(&matrix, vec![-1.0, -1.0, 1.0, 1.0], repeated, 1e-10),
        Err(SuperconductivityError::NonOrthonormalEigenvectors { .. })
    ));
}
