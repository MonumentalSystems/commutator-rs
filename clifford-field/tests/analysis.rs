use clifford_field::analysis::{
    correlation_raw_f64, creutz_ratios, creutz_sigma_estimate, creutz_sigma_estimate_with_policy,
    extract_vortex_core, extract_xi_single, fit_exponential, magnetic_power_spectrum,
    measure_magnetic_spectrum, vortex_correlation_2d, vortex_density, wilson_matrix_index,
    CorrelationFit, SparseCreutzPolicy,
};

#[test]
fn periodic_correlation_wraps_beyond_one_row() {
    let field = [1.0, 2.0, 4.0, 8.0];
    let corr = correlation_raw_f64(&field, 4, 1, 8, 1);
    assert_eq!(corr.len(), 9);
    assert_eq!(corr[0], 1.0);
    assert_eq!(corr[4], 1.0);
    assert_eq!(corr[8], 1.0);
    assert_eq!(corr[1], corr[5]);
}

#[test]
fn correlation_shape_contract_is_explicit() {
    let panic = std::panic::catch_unwind(|| correlation_raw_f64(&[1.0, 2.0], 2, 2, 1, 1));
    assert!(panic.is_err());
}

#[test]
fn exponential_fit_and_single_point_length_agree() {
    let correlation: Vec<f64> = (0..100).map(|r| (-(r as f64) / 5.0).exp()).collect();
    let fit = fit_exponential(&correlation, 5, 80).unwrap();
    assert!((fit.xi - 5.0).abs() < 1e-10);
    assert!((fit.amplitude - 1.0).abs() < 1e-10);
    assert!(fit.r_squared > 0.999_999);
    assert!((extract_xi_single(&correlation, 5) - 5.0).abs() < 1e-10);
}

#[test]
fn vortex_core_crossing_is_interpolated() {
    let correlation: Vec<f64> = (0..40)
        .map(|r| {
            let tail = (-(r as f64) / 10.0).exp();
            let ratio = if r < 3 { 0.5 + r as f64 * 0.2 } else { 1.1 };
            tail * ratio
        })
        .collect();
    let fit = CorrelationFit {
        amplitude: 1.0,
        xi: 10.0,
        r_squared: 1.0,
    };
    let core = extract_vortex_core(&correlation, &fit).unwrap();
    assert!((core.a0 - 2.5).abs() < 1e-12, "a0 = {}", core.a0);
    assert_eq!(core.alpha_inv, core.xi_over_a0);
}

#[test]
fn winding_map_analysis_has_stable_normalization() {
    let mut map = vec![0; 64];
    map[0] = 1;
    assert_eq!(vortex_density(&map), 1.0 / 64.0);
    assert_eq!(vortex_density(&[]), 0.0);

    let correlation = vortex_correlation_2d(&map, 8, 8, 4);
    assert_eq!(correlation[0], 1.0 / 64.0);
    let spectrum = magnetic_power_spectrum(&[1.0, 0.5, 0.25, 0.125]);
    let sum = 1.0 + 0.5 + 0.25 + 0.125;
    assert!((spectrum[0].1 - sum * sum / 4.0).abs() < 1e-12);

    let result = measure_magnetic_spectrum(&map, 8, 8);
    assert_eq!(result.vortex_density, 1.0 / 64.0);
    assert_eq!(result.b_seed, result.vortex_density.sqrt());
}

#[test]
fn creutz_ratio_cancels_perimeter_terms() {
    let sigma = 0.7;
    let perimeter = 0.2;
    let max_r = 5;
    let t_values = [1, 2, 3, 4];
    let mut matrix = vec![0.0; max_r * t_values.len()];
    for r in 1..=max_r {
        for (t_index, &t) in t_values.iter().enumerate() {
            matrix[wilson_matrix_index(r - 1, t_index, t_values.len())] =
                (-sigma * (r * t) as f64 - perimeter * (2 * (r + t)) as f64).exp();
        }
    }

    let ratios = creutz_ratios(&matrix, max_r, &t_values);
    assert!(!ratios.is_empty());
    assert!(ratios
        .iter()
        .all(|(_, _, value)| (value - sigma).abs() < 1e-12));
    let (estimate, error) = creutz_sigma_estimate(&matrix, max_r, &t_values);
    assert!((estimate - sigma).abs() < 1e-12);
    assert!(error < 1e-12);
}

#[test]
fn creutz_sparse_policy_preserves_both_legacy_contracts() {
    let t_values = [1, 2];
    let matrix = [1.0, 1.0, 1.0, (-0.8f64).exp()];
    let robust = creutz_sigma_estimate(&matrix, 2, &t_values);
    let historical =
        creutz_sigma_estimate_with_policy(&matrix, 2, &t_values, SparseCreutzPolicy::Iqr);
    assert!((robust.0 - 0.8).abs() < 1e-12);
    assert_eq!(robust, historical);
}

#[test]
fn creutz_invalid_and_short_inputs_are_explicit() {
    let invalid = [1.0, 0.0, 1.0, 1.0];
    assert_eq!(creutz_sigma_estimate(&invalid, 2, &[1, 2]), (0.0, f64::MAX));
    let panic = std::panic::catch_unwind(|| creutz_ratios(&[1.0], 2, &[1, 2]));
    assert!(panic.is_err());
}
