use quantum_tomography::*;
fn close(a: Complex64, b: Complex64) {
    assert!((a - b).norm() < 1e-8, "{a} {b}")
}
#[test]
fn plan_and_zero() {
    let p = LocalPauliPlan::complete(2).unwrap();
    assert_eq!(p.settings().len(), 9);
    let r = linear_inversion(1, &[1.0, 0.0, 0.0, 1.0], 1e-12).unwrap();
    close(r.get(0, 0).unwrap(), 1.0.into());
    close(r.get(1, 1).unwrap(), 0.0.into());
}

#[cfg(target_pointer_width = "64")]
#[test]
fn enormous_measurement_plan_returns_error_without_panicking() {
    let result = std::panic::catch_unwind(|| LocalPauliPlan::complete(40));
    assert!(result.is_ok(), "checked construction must not panic");
    assert_eq!(result.unwrap(), Err(TomographyError::InvalidQubits));
}
#[test]
fn bell_state() {
    let e = [
        1., 0., 0., 0., 0., 1., 0., 0., 0., 0., -1., 0., 0., 0., 0., 1.,
    ];
    let r = linear_inversion(2, &e, 1e-12).unwrap();
    close(r.get(0, 0).unwrap(), 0.5.into());
    close(r.get(0, 3).unwrap(), 0.5.into());
    close(r.get(3, 0).unwrap(), 0.5.into());
    close(r.get(3, 3).unwrap(), 0.5.into());
}
#[test]
fn noisy_projection() {
    let a = Operator::try_new(2, vec![1.1.into(), 0.0.into(), 0.0.into(), (-0.1).into()]).unwrap();
    let p = project_density(&a).unwrap();
    close(p.get(0, 0).unwrap(), 1.0.into());
    close(p.get(1, 1).unwrap(), 0.0.into());
}

#[test]
fn projection_uses_trace_one_simplex_not_clipping_rescale() {
    let estimate = Operator::try_new(
        3,
        vec![
            0.8.into(),
            0.0.into(),
            0.0.into(),
            0.0.into(),
            0.3.into(),
            0.0.into(),
            0.0.into(),
            0.0.into(),
            (-0.1).into(),
        ],
    )
    .unwrap();
    let projected = project_density_with_tolerance(&estimate, 1e-12).unwrap();
    close(projected.get(0, 0).unwrap(), 0.75.into());
    close(projected.get(1, 1).unwrap(), 0.25.into());
    close(projected.get(2, 2).unwrap(), 0.0.into());
}

#[test]
fn accepted_identity_expectation_is_pinned_to_unit_trace() {
    let estimate = linear_inversion(1, &[1.0 + 5e-10, 0.0, 0.0, 1.0], 1e-9).unwrap();
    close(estimate.trace(), 1.0.into());
}
#[test]
fn identity_and_depolarizing_channels() {
    let id = PauliTransferMatrix::try_new(
        1,
        vec![
            1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
        ],
    )
    .unwrap();
    assert!(id.is_trace_preserving(1e-10).unwrap());
    assert!(id.is_unital(1e-10).unwrap());
    assert!(id.is_completely_positive(1e-9).unwrap());
    let c = id.normalized_choi().unwrap();
    close(c.get(0, 0).unwrap(), 0.5.into());
    close(c.get(0, 3).unwrap(), 0.5.into());
    let dep = PauliTransferMatrix::try_new(
        1,
        vec![
            1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ],
    )
    .unwrap();
    assert!(dep.is_trace_preserving(1e-10).unwrap());
    assert!(dep.is_unital(1e-10).unwrap());
    assert!(dep.is_completely_positive(1e-9).unwrap());
    for i in 0..4 {
        close(
            dep.normalized_choi().unwrap().get(i, i).unwrap(),
            0.25.into(),
        );
    }
}

#[test]
fn process_tomography_recovers_amplitude_damping() {
    let damping = 0.3_f64;
    let transverse = (1.0 - damping).sqrt();
    let expected = PauliTransferMatrix::try_new(
        1,
        vec![
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            transverse,
            0.0,
            0.0,
            0.0,
            0.0,
            transverse,
            0.0,
            damping,
            0.0,
            0.0,
            1.0 - damping,
        ],
    )
    .unwrap();
    let inputs = product_probe_expectations(1).unwrap();
    let outputs = inputs
        .iter()
        .map(|input| expected.apply_expectations(input).unwrap())
        .collect::<Vec<_>>();
    let reconstructed = process_linear_inversion(1, &inputs, &outputs, 1e-12).unwrap();
    assert_eq!(reconstructed.qubits(), 1);
    for (&observed, &target) in reconstructed.as_slice().iter().zip(expected.as_slice()) {
        assert!((observed - target).abs() < 1e-12, "{observed} != {target}");
    }
    assert!(reconstructed.is_trace_preserving(1e-12).unwrap());
    assert!(!reconstructed.is_unital(1e-12).unwrap());
    assert!(reconstructed.is_completely_positive(1e-10).unwrap());
}

#[test]
fn process_tomography_rejects_dependent_or_unnormalized_probes() {
    let inputs = vec![vec![1.0, 0.0, 0.0, 1.0]; 4];
    assert_eq!(
        process_linear_inversion(1, &inputs, &inputs, 1e-12),
        Err(TomographyError::SingularProcessDesign)
    );
    let inputs = product_probe_expectations(1).unwrap();
    let mut outputs = inputs.clone();
    outputs[0][0] = 0.9;
    assert_eq!(
        process_linear_inversion(1, &inputs, &outputs, 1e-12),
        Err(TomographyError::InvalidNormalization(0.9))
    );
}

#[test]
fn gst_probabilities_are_similarity_gauge_invariant() {
    let model = GateSetModel::try_new(
        vec![1.0, 0.3],
        vec![vec![0.5, 0.5], vec![0.5, -0.5]],
        vec![("g".into(), vec![1.0, 0.0, 0.0, -1.0])],
    )
    .unwrap();
    let before = model.probabilities(&["g", "g"]).unwrap();
    let gauged = model.gauge_transform(&[1.0, 0.2, 0.0, 1.3]).unwrap();
    let after = gauged.probabilities(&["g", "g"]).unwrap();
    for (a, b) in before.iter().zip(after) {
        assert!((a - b).abs() < 1e-12)
    }
    let metrics = model
        .fit_diagnostics(&[GstRecord {
            sequence: vec!["g".into()],
            counts: vec![35, 65],
        }])
        .unwrap();
    assert!(metrics.log_likelihood.is_finite() && metrics.deviance >= 0.0);
}

#[test]
fn gst_validates_probabilities_zeros_and_count_overflow() {
    let structural_zero =
        GateSetModel::try_new(vec![1.0], vec![vec![0.0], vec![1.0]], vec![]).unwrap();
    let diagnostics = structural_zero
        .fit_diagnostics(&[GstRecord {
            sequence: vec![],
            counts: vec![0, 10],
        }])
        .unwrap();
    assert_eq!(diagnostics.log_likelihood, 0.0);
    assert_eq!(diagnostics.deviance, 0.0);

    let unnormalized =
        GateSetModel::try_new(vec![1.0], vec![vec![0.8], vec![0.8]], vec![]).unwrap();
    assert_eq!(
        unnormalized.fit_diagnostics(&[GstRecord {
            sequence: vec![],
            counts: vec![5, 5],
        }]),
        Err(TomographyError::InvalidProbabilities)
    );

    let fair = GateSetModel::try_new(vec![1.0], vec![vec![0.5], vec![0.5]], vec![]).unwrap();
    assert_eq!(
        fair.fit_diagnostics(&[GstRecord {
            sequence: vec![],
            counts: vec![u64::MAX, 1],
        }]),
        Err(TomographyError::CountOverflow)
    );
}

#[test]
fn globally_small_invertible_gauge_is_scale_invariant() {
    let model = GateSetModel::try_new(
        vec![1.0, 0.2],
        vec![vec![0.5, 0.5], vec![0.5, -0.5]],
        vec![],
    )
    .unwrap();
    let gauged = model.gauge_transform(&[1e-20, 0.0, 0.0, 2e-20]).unwrap();
    let original = model.probabilities(&[]).unwrap();
    let transformed = gauged.probabilities(&[]).unwrap();
    for (left, right) in original.iter().zip(transformed) {
        assert!((left - right).abs() < 1e-12);
    }
}
