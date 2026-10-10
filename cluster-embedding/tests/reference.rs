use cluster_embedding::{
    bethe_dmft, embed_self_energy, find_stationary_point, hybridization_from_weiss,
    stationarity_diagnostics, BetheLattice, ClusterGreenGrid, Complex64, DenseMatrix, DmftConfig,
    EmbeddingError, GreenPoint, HubbardModel, HybridizationGrid, ImpuritySolution,
    LogDetBranchPolicy, MatrixFrequencyGrid, MomentumPoint, OneBodyTerm, PotthoffFunctional,
    ReferenceSolution, ReferenceSystem, SearchConfig, WeissFieldGrid,
};

fn scalar(value: Complex64) -> DenseMatrix {
    DenseMatrix::from_scalar(value).expect("finite scalar")
}

fn green_grid(frequencies: &[Complex64], values: &[Complex64]) -> ClusterGreenGrid {
    ClusterGreenGrid::try_new(
        1,
        frequencies
            .iter()
            .zip(values)
            .map(|(frequency, value)| GreenPoint::new(*frequency, scalar(*value)))
            .collect(),
    )
    .expect("causal grid")
}

#[test]
fn one_site_noninteracting_embedding_is_analytic() {
    let frequencies = [Complex64::new(-0.4, 0.2), Complex64::new(0.7, 0.2)];
    let hopping = scalar(Complex64::new(0.3, 0.0));
    let sigma = vec![scalar(Complex64::new(0.0, 0.0)); frequencies.len()];
    let embedded = embed_self_energy(&frequencies, &sigma, &hopping, 0.1).unwrap();
    for (point, frequency) in embedded.points().iter().zip(frequencies) {
        let expected = Complex64::new(1.0, 0.0) / (frequency + 0.1 - 0.3);
        assert!((point.green.get(0, 0).unwrap() - expected).norm() < 1.0e-13);
    }
}

#[test]
fn identical_reference_and_lattice_leave_grand_potential_unchanged() {
    let frequencies = [Complex64::new(-1.0, 0.3), Complex64::new(0.0, 0.3)];
    let values: Vec<_> = frequencies
        .iter()
        .map(|z| Complex64::new(1.0, 0.0) / *z)
        .collect();
    let reference = ReferenceSolution::try_new(
        green_grid(&frequencies, &values),
        vec![scalar(0.0.into()); frequencies.len()],
        -0.75,
    )
    .unwrap();
    let functional = PotthoffFunctional::try_new(
        0.0,
        vec![MomentumPoint::try_new(scalar(0.0.into()), 1.0).unwrap()],
        vec![0.4, 0.6],
        LogDetBranchPolicy::ContinuousFrequency,
    )
    .unwrap();
    let result = functional.evaluate(&reference).unwrap();
    assert!((result.value + 0.75).abs() < 1.0e-13);
    assert!(result.imaginary_residual.abs() < 1.0e-13);
}

#[test]
fn stationarity_diagnostics_find_injected_solver_saddle() {
    let model = HubbardModel::try_new(scalar(0.0.into()), vec![2.0], 0.0).unwrap();
    let term = OneBodyTerm::try_new("bath shift", scalar(1.0.into()), 0.0, 1.0e-3).unwrap();
    let system = ReferenceSystem::try_new(model, vec![term]).unwrap();
    let frequencies = vec![Complex64::new(0.0, 0.5)];
    let functional = PotthoffFunctional::try_new(
        0.0,
        vec![MomentumPoint::try_new(scalar(0.0.into()), 1.0).unwrap()],
        vec![1.0],
        LogDetBranchPolicy::ContinuousFrequency,
    )
    .unwrap();
    let mut solver = |reference: &ReferenceSystem| {
        let value = reference.terms()[0].value();
        ReferenceSolution::try_new(
            green_grid(&frequencies, &[Complex64::new(0.0, -2.0)]),
            vec![scalar(0.0.into())],
            -(value - 0.3) * (value - 0.3),
        )
    };
    let initial = stationarity_diagnostics(&mut solver, &system, &functional, 1.0e-8).unwrap();
    assert!((initial.gradient[0] - 0.6).abs() < 1.0e-9);
    assert!(initial.diagonal_curvature[0] < 0.0);
    let (found, report, history) = find_stationary_point(
        &mut solver,
        &system,
        &functional,
        SearchConfig {
            trust_radius: 0.5,
            ..SearchConfig::default()
        },
    )
    .unwrap();
    assert!(report.stationary);
    assert!((found.terms()[0].value() - 0.3).abs() < 1.0e-8);
    assert!(!history.is_empty());
}

#[test]
fn parameter_and_quadrature_errors_are_checked() {
    assert!(matches!(
        HubbardModel::try_new(scalar(0.0.into()), vec![1.0, 2.0], 0.0),
        Err(EmbeddingError::Length { .. })
    ));
    let point = MomentumPoint::try_new(scalar(0.0.into()), 0.4).unwrap();
    assert!(matches!(
        PotthoffFunctional::try_new(0.0, vec![point], vec![1.0], LogDetBranchPolicy::Principal),
        Err(EmbeddingError::InvalidValue(_))
    ));
}

#[test]
fn positive_imaginary_self_energy_and_hybridization_are_rejected() {
    let frequency = Complex64::new(0.0, 0.1);
    let green = green_grid(&[frequency], &[Complex64::new(0.0, -1.0)]);
    assert!(matches!(
        ReferenceSolution::try_new(green, vec![scalar(Complex64::new(0.0, 0.2))], 0.0),
        Err(EmbeddingError::NonCausal { .. })
    ));
    let grid =
        MatrixFrequencyGrid::try_new(vec![frequency], vec![scalar(Complex64::new(0.0, 0.2))])
            .unwrap();
    assert!(matches!(
        HybridizationGrid::try_new(grid, 1.0e-12),
        Err(EmbeddingError::NonCausal { .. })
    ));
}

#[test]
fn weiss_to_hybridization_recovers_scalar_bath() {
    let frequency = Complex64::new(0.2, 0.4);
    let delta = Complex64::new(0.1, -0.3);
    let weiss_value = Complex64::new(1.0, 0.0) / (frequency - 0.25 - delta);
    let weiss = WeissFieldGrid::try_new(
        MatrixFrequencyGrid::try_new(vec![frequency], vec![scalar(weiss_value)]).unwrap(),
        1.0e-12,
    )
    .unwrap();
    let recovered = hybridization_from_weiss(&weiss, &scalar(0.25.into()), 0.0, 1.0e-12).unwrap();
    assert!((recovered.grid().matrices()[0].get(0, 0).unwrap() - delta).norm() < 1.0e-13);
}

#[test]
fn bethe_noninteracting_fixed_point_converges_in_one_iteration() {
    let frequency = Complex64::new(0.0, 1.0);
    let hopping = 0.5;
    // Retarded root of t^2 G^2 - z G + 1 = 0.
    let green_value = (frequency - (frequency * frequency - 4.0 * hopping * hopping).sqrt())
        / (2.0 * hopping * hopping);
    assert!(green_value.im < 0.0);
    let initial = WeissFieldGrid::try_new(
        MatrixFrequencyGrid::try_new(vec![frequency], vec![scalar(green_value)]).unwrap(),
        1.0e-12,
    )
    .unwrap();
    let model = HubbardModel::try_new(scalar(0.0.into()), vec![0.0], 0.0).unwrap();
    let mut solver = |problem: &cluster_embedding::ImpurityProblem| {
        let green = ClusterGreenGrid::try_new(
            1,
            vec![GreenPoint::new(
                frequency,
                problem.weiss.grid().matrices()[0].clone(),
            )],
        )?;
        ImpuritySolution::try_new(
            green,
            MatrixFrequencyGrid::try_new(vec![frequency], vec![scalar(0.0.into())])?,
            1.0e-12,
        )
    };
    let outcome = bethe_dmft(
        &mut solver,
        &model,
        BetheLattice::try_new(hopping).unwrap(),
        initial,
        DmftConfig::default(),
    )
    .unwrap();
    assert!(outcome.converged);
    assert_eq!(outcome.history.iterations().len(), 1);
    assert!(outcome.history.iterations()[0].residual < 1.0e-12);
}

#[test]
fn nonconverged_outcome_keeps_weiss_and_impurity_solution_consistent() {
    let frequency = Complex64::new(0.0, 1.0);
    let initial_value = Complex64::new(0.0, -0.4);
    let initial = WeissFieldGrid::try_new(
        MatrixFrequencyGrid::try_new(vec![frequency], vec![scalar(initial_value)]).unwrap(),
        1.0e-12,
    )
    .unwrap();
    let model = HubbardModel::try_new(scalar(0.0.into()), vec![0.0], 0.0).unwrap();
    let mut solver = |problem: &cluster_embedding::ImpurityProblem| {
        let green = ClusterGreenGrid::try_new(
            1,
            vec![GreenPoint::new(
                frequency,
                problem.weiss.grid().matrices()[0].clone(),
            )],
        )?;
        ImpuritySolution::try_new(
            green,
            MatrixFrequencyGrid::try_new(vec![frequency], vec![scalar(0.0.into())])?,
            1.0e-12,
        )
    };
    let outcome = bethe_dmft(
        &mut solver,
        &model,
        BetheLattice::try_new(0.5).unwrap(),
        initial,
        DmftConfig {
            maximum_iterations: 1,
            mixing: 0.5,
            convergence_tolerance: 1.0e-15,
            causality_tolerance: 1.0e-12,
        },
    )
    .unwrap();

    assert!(!outcome.converged);
    let returned_weiss = outcome.weiss.grid().matrices()[0].get(0, 0).unwrap();
    let solved_weiss = outcome.impurity.green().points()[0]
        .green
        .get(0, 0)
        .unwrap();
    assert!((returned_weiss - initial_value).norm() < 1.0e-14);
    assert!((solved_weiss - returned_weiss).norm() < 1.0e-14);
}
