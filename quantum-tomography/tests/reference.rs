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
