use quantum_chaos::{
    ChaosError, Spectrum, UnfoldedSpectrum, UnfoldingMethod, UnfoldingPolicy,
    GOE_SURMISE_MEAN_GAP_RATIO, POISSON_MEAN_GAP_RATIO,
};

#[test]
fn analytic_reference_means_match_closed_forms() {
    assert!((POISSON_MEAN_GAP_RATIO - (2.0 * 2.0_f64.ln() - 1.0)).abs() < 1e-15);
    assert!((GOE_SURMISE_MEAN_GAP_RATIO - (4.0 - 2.0 * 3.0_f64.sqrt())).abs() < 1e-15);
}

#[test]
fn picket_fence_has_unit_gap_ratios() {
    let levels: Vec<f64> = (0..64).map(f64::from).collect();
    let spectrum = Spectrum::try_from_sorted(levels).unwrap();
    let statistics = spectrum.adjacent_gap_ratios(0.0).unwrap();
    assert!(statistics.ratios().iter().all(|ratio| *ratio == 1.0));
    assert_eq!(statistics.mean(), 1.0);
}

#[test]
fn deterministic_exponential_spacings_approach_poisson_ratio_mean() {
    let mut generator = 0x4d59_5df4_d0f3_3173_u64;
    let mut level = 0.0;
    let mut levels = Vec::with_capacity(100_001);
    levels.push(level);
    for _ in 0..100_000 {
        generator = generator
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let mantissa = generator >> 11;
        let uniform = (mantissa as f64 + 0.5) / ((1_u64 << 53) as f64);
        level += -uniform.ln();
        levels.push(level);
    }
    let mean = Spectrum::try_from_sorted(levels)
        .unwrap()
        .adjacent_gap_ratios(0.0)
        .unwrap()
        .mean();
    assert!((mean - POISSON_MEAN_GAP_RATIO).abs() < 0.004);
}

#[test]
fn degeneracy_is_preserved_but_rejected_by_gap_diagnostics() {
    let spectrum = Spectrum::try_from_sorted(vec![0.0, 1.0, 1.0, 2.0]).unwrap();
    assert!(matches!(
        spectrum.adjacent_gap_ratios(0.0),
        Err(ChaosError::GapTooSmall { index: 1, .. })
    ));
    assert_eq!(spectrum.spectral_form_factor(&[0.0]).unwrap(), vec![4.0]);
    assert!(matches!(
        spectrum.unfold(UnfoldingPolicy::Affine { minimum_gap: 0.0 }),
        Err(ChaosError::GapTooSmall { index: 1, .. })
    ));
}

#[test]
fn affine_and_local_unfolding_record_policy() {
    let spectrum = Spectrum::try_from_sorted(vec![10.0, 12.0, 14.0, 16.0]).unwrap();
    let affine = spectrum
        .unfold(UnfoldingPolicy::Affine { minimum_gap: 0.0 })
        .unwrap();
    assert_eq!(affine.levels(), &[0.0, 1.0, 2.0, 3.0]);
    assert_eq!(affine.method(), UnfoldingMethod::Affine);

    let local = spectrum
        .unfold(UnfoldingPolicy::LocalGap {
            half_window: 2,
            minimum_gap: 0.0,
        })
        .unwrap();
    assert_eq!(local.levels(), &[0.0, 1.0, 2.0, 3.0]);
    assert_eq!(local.method(), UnfoldingMethod::LocalGap { half_window: 2 });
    assert_eq!(
        spectrum.unfold(UnfoldingPolicy::LocalGap {
            half_window: 0,
            minimum_gap: 0.0,
        }),
        Err(ChaosError::InvalidLocalWindow)
    );
}

#[test]
fn picket_fence_form_factor_has_analytic_revivals() {
    let spectrum = Spectrum::try_from_sorted((0..10).map(f64::from).collect()).unwrap();
    let values = spectrum
        .spectral_form_factor(&[0.0, core::f64::consts::PI, core::f64::consts::TAU])
        .unwrap();
    assert!((values[0] - 10.0).abs() < 1e-14);
    assert!(values[1] < 1e-28);
    assert!((values[2] - 10.0).abs() < 1e-14);
}

#[test]
fn picket_fence_number_variance_is_zero_for_fixed_counts() {
    let unfolded =
        UnfoldedSpectrum::try_from_sorted((0..101).map(f64::from).collect(), 0.0).unwrap();
    let origins: Vec<f64> = (0..95).map(|index| index as f64 + 0.25).collect();
    let statistic = unfolded.number_variance(5.0, &origins).unwrap();
    assert_eq!(statistic.window_count(), origins.len());
    assert_eq!(statistic.mean_count(), 5.0);
    assert_eq!(statistic.variance(), 0.0);
    assert_eq!(statistic.mean_square_deviation_from_unit_density(), 0.0);
}

#[test]
fn validation_reports_order_times_and_window_edges() {
    assert_eq!(
        Spectrum::try_from_sorted(vec![0.0, 2.0, 1.0]),
        Err(ChaosError::LevelsNotSorted { index: 2 })
    );
    let spectrum = Spectrum::try_from_sorted(vec![0.0, 1.0]).unwrap();
    assert_eq!(
        spectrum.spectral_form_factor(&[f64::NAN]),
        Err(ChaosError::NonFiniteTime { index: 0 })
    );
    let unfolded = UnfoldedSpectrum::try_from_sorted(vec![0.0, 1.0, 2.0], 0.0).unwrap();
    assert_eq!(
        unfolded.number_variance(1.5, &[1.0]),
        Err(ChaosError::InvalidOrigin { index: 0 })
    );

    let extreme = Spectrum::try_from_sorted(vec![-f64::MAX, f64::MAX]).unwrap();
    assert_eq!(
        extreme.unfold(UnfoldingPolicy::Affine { minimum_gap: 0.0 }),
        Err(ChaosError::GapOverflow { index: 0 })
    );
    assert!(extreme.spectral_form_factor(&[1.0]).unwrap()[0].is_finite());
}
