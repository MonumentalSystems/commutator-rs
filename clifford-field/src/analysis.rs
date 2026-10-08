//! Portable post-processing for field-simulation output.
//!
//! These routines operate on flat numeric arrays. They do not construct a
//! Clifford field, advance simulation state, or depend on a transport runtime.

/// Compute the normalized spatial correlation `C(r) / C(0)` along rows.
///
/// `field` has row-major shape `(ny, nx, n_comp)`. Displacements wrap
/// periodically, including when `max_r` is larger than `nx`.
///
/// # Panics
///
/// Panics when the dimensions overflow, when any dimension is zero, or when
/// `field` does not have exactly `nx * ny * n_comp` entries.
pub fn correlation_raw_f64(
    field: &[f64],
    nx: usize,
    ny: usize,
    max_r: usize,
    n_comp: usize,
) -> Vec<f64> {
    assert!(nx > 0, "nx must be nonzero");
    assert!(ny > 0, "ny must be nonzero");
    assert!(n_comp > 0, "n_comp must be nonzero");
    let row_len = nx.checked_mul(n_comp).expect("row length overflow");
    let expected_len = ny.checked_mul(row_len).expect("field length overflow");
    assert_eq!(field.len(), expected_len, "field shape mismatch");

    let mut corr = vec![0.0; max_r + 1];
    for y in 0..ny {
        let row_start = y * row_len;
        for (r, value) in corr.iter_mut().enumerate() {
            let shift = r % nx;
            let mut dot = 0.0;
            for x in 0..nx {
                let xr = (x + shift) % nx;
                for k in 0..n_comp {
                    dot += field[row_start + x * n_comp + k] * field[row_start + xr * n_comp + k];
                }
            }
            *value += dot;
        }
    }

    let c0 = corr[0];
    if c0 > 1e-30 {
        corr.iter().map(|value| value / c0).collect()
    } else {
        corr
    }
}

/// Extract a screening length from `xi = -r / ln(C(r) / C(0))`.
pub fn extract_xi_single(corr: &[f64], r: usize) -> f64 {
    if corr.len() <= r || corr[0].abs() <= 1e-30 {
        return 0.0;
    }
    let ratio = corr[r] / corr[0];
    if ratio <= 1e-30 || ratio >= 1.0 {
        return 0.0;
    }
    -(r as f64) / ratio.ln()
}

/// Result of fitting `C(r) = A exp(-r / xi)` to a correlation tail.
#[derive(Clone, Debug)]
pub struct CorrelationFit {
    /// Extrapolated amplitude `A`.
    pub amplitude: f64,
    /// Screening length `xi`.
    pub xi: f64,
    /// Coefficient of determination in log space.
    pub r_squared: f64,
}

/// Fit `C(r) = A exp(-r / xi)` by linear regression in log space.
pub fn fit_exponential(corr: &[f64], r_min: usize, r_max: usize) -> Option<CorrelationFit> {
    let r_max = r_max.min(corr.len());
    if r_min >= r_max {
        return None;
    }

    let points: Vec<(f64, f64)> = (r_min..r_max)
        .filter(|&r| corr[r] > 1e-30)
        .map(|r| (r as f64, corr[r].ln()))
        .collect();
    if points.len() < 2 {
        return None;
    }

    let n = points.len() as f64;
    let sum_x: f64 = points.iter().map(|(x, _)| x).sum();
    let sum_y: f64 = points.iter().map(|(_, y)| y).sum();
    let sum_xx: f64 = points.iter().map(|(x, _)| x * x).sum();
    let sum_xy: f64 = points.iter().map(|(x, y)| x * y).sum();
    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-30 {
        return None;
    }

    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;
    let xi = if slope.abs() > 1e-30 {
        -1.0 / slope
    } else {
        f64::INFINITY
    };

    let y_mean = sum_y / n;
    let ss_tot: f64 = points.iter().map(|(_, y)| (y - y_mean).powi(2)).sum();
    let ss_res: f64 = points
        .iter()
        .map(|(x, y)| (y - (intercept + slope * x)).powi(2))
        .sum();

    Some(CorrelationFit {
        amplitude: intercept.exp(),
        xi,
        r_squared: if ss_tot > 1e-30 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        },
    })
}

/// Vortex-core measurement inferred from a correlation fit.
#[derive(Clone, Debug)]
pub struct VortexCoreResult {
    /// Screening length from the exponential tail fit.
    pub xi: f64,
    /// Estimated vortex-core radius in lattice units.
    pub a0: f64,
    /// Dimensionless ratio of screening length to core radius.
    pub xi_over_a0: f64,
    /// Inverse effective coupling, currently equal to `xi_over_a0`.
    pub alpha_inv: f64,
}

/// Locate the small-distance crossover from a fitted exponential tail.
pub fn extract_vortex_core(corr: &[f64], fit: &CorrelationFit) -> Option<VortexCoreResult> {
    if fit.xi <= 0.0 || !fit.xi.is_finite() || fit.amplitude <= 0.0 {
        return None;
    }

    let max_check = (fit.xi as usize).min(corr.len()).min(20);
    if max_check < 2 {
        return None;
    }

    let ratios: Vec<(f64, f64)> = (1..max_check)
        .filter(|&r| corr[r] > 0.0)
        .filter_map(|r| {
            let fitted = fit.amplitude * (-(r as f64) / fit.xi).exp();
            (fitted > 1e-30).then_some((r as f64, corr[r] / fitted))
        })
        .collect();
    if ratios.len() < 2 {
        return None;
    }

    let mut best_r = 1.0;
    let mut best_deviation = f64::MAX;
    for &(r, ratio) in &ratios {
        let deviation = (ratio - 1.0).abs();
        if deviation < best_deviation {
            best_deviation = deviation;
            best_r = r;
        }
    }

    for pair in ratios.windows(2) {
        let (r1, ratio1) = pair[0];
        let (r2, ratio2) = pair[1];
        if (ratio1 - 1.0) * (ratio2 - 1.0) <= 0.0 {
            let denominator = ratio2 - ratio1;
            if denominator.abs() > f64::EPSILON {
                best_r = r1 + (1.0 - ratio1) / denominator * (r2 - r1);
            }
            break;
        }
    }

    let a0 = best_r.max(0.5);
    let xi_over_a0 = fit.xi / a0;
    Some(VortexCoreResult {
        xi: fit.xi,
        a0,
        xi_over_a0,
        alpha_inv: xi_over_a0,
    })
}

/// Vortex density in a winding-number map.
pub fn vortex_density(map: &[i8]) -> f64 {
    if map.is_empty() {
        return 0.0;
    }
    map.iter().filter(|&&winding| winding != 0).count() as f64 / map.len() as f64
}

/// Periodic two-point correlation of a row-major `(nx, ny)` winding map.
///
/// # Panics
///
/// Panics when either dimension is zero, their product overflows, or the map
/// length does not match the declared shape.
pub fn vortex_correlation_2d(map: &[i8], nx: usize, ny: usize, max_r: usize) -> Vec<f64> {
    assert!(nx > 0, "nx must be nonzero");
    assert!(ny > 0, "ny must be nonzero");
    assert_eq!(
        map.len(),
        nx.checked_mul(ny).expect("map length overflow"),
        "map shape mismatch"
    );

    let max_r = max_r.min(nx / 2).min(ny / 2);
    let mut corr = vec![0.0; max_r + 1];
    let mut count = vec![0usize; max_r + 1];
    for x in 0..nx {
        for y in 0..ny {
            let winding = map[x * ny + y] as f64;
            for r in 0..=max_r {
                corr[r] += winding * map[((x + r) % nx) * ny + y] as f64;
                count[r] += 1;
                if r > 0 {
                    corr[r] += winding * map[x * ny + (y + r) % ny] as f64;
                    count[r] += 1;
                }
            }
        }
    }
    for (value, samples) in corr.iter_mut().zip(count) {
        if samples > 0 {
            *value /= samples as f64;
        }
    }
    corr
}

/// Real discrete Fourier power spectrum of a vortex correlation sequence.
pub fn magnetic_power_spectrum(correlation: &[f64]) -> Vec<(f64, f64)> {
    let n = correlation.len();
    if n == 0 {
        return Vec::new();
    }

    (0..=n / 2)
        .map(|frequency| {
            let k = std::f64::consts::TAU * frequency as f64 / n as f64;
            let (real, imaginary) =
                correlation
                    .iter()
                    .enumerate()
                    .fold((0.0, 0.0), |(real, imaginary), (r, value)| {
                        let phase = k * r as f64;
                        (real + value * phase.cos(), imaginary + value * phase.sin())
                    });
            (k, (real * real + imaginary * imaginary) / n as f64)
        })
        .collect()
}

/// Derived measurements from a winding-number map.
#[derive(Clone, Debug)]
pub struct MagneticSpectrumResult {
    /// Fraction of sites with nonzero winding.
    pub vortex_density: f64,
    /// Periodic winding-number correlation sequence.
    pub c_v: Vec<f64>,
    /// `(wave number, power)` samples from the real discrete spectrum.
    pub power_spectrum: Vec<(f64, f64)>,
    /// Log-log slope fitted to positive finite-frequency power samples.
    pub spectral_index: f64,
    /// Seed-field proxy defined as the square root of vortex density.
    pub b_seed: f64,
}

/// Measure density, correlation, spectrum, and a log-log spectral slope.
pub fn measure_magnetic_spectrum(map: &[i8], nx: usize, ny: usize) -> MagneticSpectrumResult {
    let density = vortex_density(map);
    let c_v = vortex_correlation_2d(map, nx, ny, (nx / 4).max(8));
    let power_spectrum = magnetic_power_spectrum(&c_v);
    let spectral_index = fit_spectral_index(&power_spectrum);
    MagneticSpectrumResult {
        vortex_density: density,
        c_v,
        power_spectrum,
        spectral_index,
        b_seed: density.sqrt(),
    }
}

fn fit_spectral_index(spectrum: &[(f64, f64)]) -> f64 {
    let points: Vec<(f64, f64)> = spectrum
        .iter()
        .filter(|(k, power)| *k > 1e-10 && *power > 1e-30)
        .map(|(k, power)| (k.ln(), power.ln()))
        .collect();
    if points.len() < 2 {
        return 0.0;
    }

    let n = points.len() as f64;
    let sum_x: f64 = points.iter().map(|(x, _)| x).sum();
    let sum_y: f64 = points.iter().map(|(_, y)| y).sum();
    let sum_xx: f64 = points.iter().map(|(x, _)| x * x).sum();
    let sum_xy: f64 = points.iter().map(|(x, y)| x * y).sum();
    let denominator = n * sum_xx - sum_x * sum_x;
    if denominator.abs() < 1e-30 {
        0.0
    } else {
        (n * sum_xy - sum_x * sum_y) / denominator
    }
}

/// Sparse-sample behavior for robust Creutz-ratio reduction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SparseCreutzPolicy {
    /// Use the median and mean absolute deviation below four samples.
    MedianMad,
    /// Apply the historical IQR estimator to every nonempty sample.
    Iqr,
}

/// Index a row-major Wilson matrix where `R` varies slowest.
#[inline]
pub fn wilson_matrix_index(r_index: usize, t_index: usize, n_t: usize) -> usize {
    r_index * n_t + t_index
}

/// Compute Creutz ratios from adjacent columns of a Wilson matrix.
///
/// Adjacent columns are interpreted as adjacent temporal extents, so callers
/// should provide strictly increasing consecutive `t_values`.
///
/// # Panics
///
/// Panics when the matrix is shorter than `max_r * t_values.len()` or that
/// product overflows.
pub fn creutz_ratios(
    w_matrix: &[f64],
    max_r: usize,
    t_values: &[usize],
) -> Vec<(usize, usize, f64)> {
    let required = max_r
        .checked_mul(t_values.len())
        .expect("Wilson matrix length overflow");
    assert!(w_matrix.len() >= required, "Wilson matrix is too short");

    let n_t = t_values.len();
    let mut ratios = Vec::new();
    for r in 2..=max_r {
        for t_index in 1..n_t {
            let w_rt = w_matrix[wilson_matrix_index(r - 1, t_index, n_t)];
            let w_r1_t1 = w_matrix[wilson_matrix_index(r - 2, t_index - 1, n_t)];
            let w_r_t1 = w_matrix[wilson_matrix_index(r - 1, t_index - 1, n_t)];
            let w_r1_t = w_matrix[wilson_matrix_index(r - 2, t_index, n_t)];
            if w_rt.abs() > 1e-30
                && w_r1_t1.abs() > 1e-30
                && w_r_t1.abs() > 1e-30
                && w_r1_t.abs() > 1e-30
            {
                let ratio = (w_rt * w_r1_t1) / (w_r_t1 * w_r1_t);
                if ratio > 0.0 {
                    let sigma = -ratio.ln();
                    if sigma.is_finite() {
                        ratios.push((r, t_values[t_index], sigma));
                    }
                }
            }
        }
    }
    ratios
}

/// Estimate string tension with the robust small-sample policy.
pub fn creutz_sigma_estimate(w_matrix: &[f64], max_r: usize, t_values: &[usize]) -> (f64, f64) {
    creutz_sigma_estimate_with_policy(w_matrix, max_r, t_values, SparseCreutzPolicy::MedianMad)
}

/// Estimate string tension using an explicit sparse-sample policy.
pub fn creutz_sigma_estimate_with_policy(
    w_matrix: &[f64],
    max_r: usize,
    t_values: &[usize],
    sparse_policy: SparseCreutzPolicy,
) -> (f64, f64) {
    let mut values: Vec<f64> = creutz_ratios(w_matrix, max_r, t_values)
        .into_iter()
        .map(|(_, _, sigma)| sigma)
        .filter(|sigma| *sigma > 0.0 && *sigma < 10.0)
        .collect();
    if values.is_empty() {
        return (0.0, f64::MAX);
    }
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite Creutz ratio"));

    let median = if values.len() % 2 == 0 {
        (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
    } else {
        values[values.len() / 2]
    };
    if values.len() < 4 && sparse_policy == SparseCreutzPolicy::MedianMad {
        let mad = values
            .iter()
            .map(|sigma| (sigma - median).abs())
            .sum::<f64>()
            / values.len() as f64;
        return (median, mad / (values.len() as f64).sqrt());
    }

    let q1 = values[values.len() / 4];
    let q3 = values[3 * values.len() / 4];
    let fence = 1.5 * (q3 - q1);
    let lower = q1 - fence;
    let upper = q3 + fence;
    let inliers: Vec<f64> = values
        .iter()
        .copied()
        .filter(|sigma| *sigma >= lower && *sigma <= upper)
        .collect();
    if inliers.is_empty() {
        let mad = values
            .iter()
            .map(|sigma| (sigma - median).abs())
            .sum::<f64>()
            / values.len() as f64;
        return (median, mad / (values.len() as f64).sqrt());
    }

    let mean = inliers.iter().sum::<f64>() / inliers.len() as f64;
    let variance = inliers
        .iter()
        .map(|sigma| (sigma - mean).powi(2))
        .sum::<f64>()
        / inliers.len() as f64;
    (mean, (variance / inliers.len() as f64).sqrt())
}
