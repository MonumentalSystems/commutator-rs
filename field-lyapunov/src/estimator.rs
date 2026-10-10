use crate::{LyapunovError, Result, TangentStepper};

/// Configuration for the Benettin Lyapunov-spectrum estimator.
#[derive(Clone, Debug, PartialEq)]
pub struct BenettinConfig {
    top_k: usize,
    step_size: f64,
    steps_per_window: usize,
    windows: usize,
    transient_windows: usize,
    seed: u64,
}

impl BenettinConfig {
    /// Creates a configuration for `top_k` exponents.
    ///
    /// Tangent vectors are reorthonormalized after every
    /// `steps_per_window` integration steps. `windows` controls the number of
    /// measured finite-time windows and must be non-zero.
    pub fn new(
        top_k: usize,
        step_size: f64,
        steps_per_window: usize,
        windows: usize,
    ) -> Result<Self> {
        let config = Self {
            top_k,
            step_size,
            steps_per_window,
            windows,
            transient_windows: 0,
            seed: 0x4c59_4150_554e_4f56,
        };
        config.validate_counts()?;
        Ok(config)
    }

    /// Sets the number of unmeasured warm-up windows.
    #[must_use]
    pub const fn with_transient_windows(mut self, windows: usize) -> Self {
        self.transient_windows = windows;
        self
    }

    /// Sets the deterministic tangent-basis seed.
    #[must_use]
    pub const fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Number of leading exponents to estimate.
    #[must_use]
    pub const fn top_k(&self) -> usize {
        self.top_k
    }

    /// Integrator step size.
    #[must_use]
    pub const fn step_size(&self) -> f64 {
        self.step_size
    }

    /// Number of integrator steps in each finite-time window.
    #[must_use]
    pub const fn steps_per_window(&self) -> usize {
        self.steps_per_window
    }

    /// Number of measured finite-time windows.
    #[must_use]
    pub const fn windows(&self) -> usize {
        self.windows
    }

    /// Number of warm-up windows excluded from the estimate.
    #[must_use]
    pub const fn transient_windows(&self) -> usize {
        self.transient_windows
    }

    /// Seed used for deterministic tangent-basis initialization.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    fn validate_counts(&self) -> Result<()> {
        if self.top_k == 0 {
            return Err(LyapunovError::ZeroCount { field: "top_k" });
        }
        if !self.step_size.is_finite() || self.step_size <= 0.0 {
            return Err(LyapunovError::InvalidParameter {
                field: "step_size",
                value: self.step_size,
            });
        }
        if self.steps_per_window == 0 {
            return Err(LyapunovError::ZeroCount {
                field: "steps_per_window",
            });
        }
        if self.windows == 0 {
            return Err(LyapunovError::ZeroCount { field: "windows" });
        }
        Ok(())
    }
}

/// Lyapunov rates measured over one finite-time window.
#[derive(Clone, Debug, PartialEq)]
pub struct FiniteTimeEstimate {
    end_time: f64,
    local_exponents: Vec<f64>,
    running_exponents: Vec<f64>,
}

impl FiniteTimeEstimate {
    /// Absolute simulation time at the end of this window.
    #[must_use]
    pub const fn end_time(&self) -> f64 {
        self.end_time
    }

    /// Exponents computed from this window alone.
    #[must_use]
    pub fn local_exponents(&self) -> &[f64] {
        &self.local_exponents
    }

    /// Cumulative exponents from the first measured window through this one.
    #[must_use]
    pub fn running_exponents(&self) -> &[f64] {
        &self.running_exponents
    }
}

/// Result of a top-`k` Benettin spectrum calculation.
#[derive(Clone, Debug, PartialEq)]
pub struct SpectrumEstimate {
    exponents: Vec<f64>,
    finite_time: Vec<FiniteTimeEstimate>,
    final_state: Vec<f64>,
    final_time: f64,
    measured_time: f64,
}

impl SpectrumEstimate {
    /// Estimated exponents in Benettin/QR order, asymptotically descending.
    #[must_use]
    pub fn exponents(&self) -> &[f64] {
        &self.exponents
    }

    /// Per-window local and cumulative estimates.
    #[must_use]
    pub fn finite_time_estimates(&self) -> &[FiniteTimeEstimate] {
        &self.finite_time
    }

    /// State at the end of warm-up and measurement.
    #[must_use]
    pub fn final_state(&self) -> &[f64] {
        &self.final_state
    }

    /// Absolute time at the end of the calculation.
    #[must_use]
    pub const fn final_time(&self) -> f64 {
        self.final_time
    }

    /// Total time included in the reported exponents.
    #[must_use]
    pub const fn measured_time(&self) -> f64 {
        self.measured_time
    }
}

/// Estimates the leading Lyapunov spectrum beginning at time zero.
pub fn estimate_spectrum<S: TangentStepper>(
    stepper: &mut S,
    initial_state: &[f64],
    config: &BenettinConfig,
) -> Result<SpectrumEstimate> {
    estimate_spectrum_at(stepper, 0.0, initial_state, config)
}

/// Estimates only the largest Lyapunov exponent beginning at time zero.
///
/// The returned result retains finite-time samples and the final state in the
/// same form as [`estimate_spectrum`].
pub fn estimate_largest<S: TangentStepper>(
    stepper: &mut S,
    initial_state: &[f64],
    step_size: f64,
    steps_per_window: usize,
    windows: usize,
) -> Result<SpectrumEstimate> {
    let config = BenettinConfig::new(1, step_size, steps_per_window, windows)?;
    estimate_spectrum(stepper, initial_state, &config)
}

/// Estimates the leading Lyapunov spectrum from an explicit start time.
///
/// Warm-up windows evolve both the trajectory and tangent basis, but their
/// stretching is excluded from the reported rates. Each measured window ends
/// with twice-applied modified Gram-Schmidt reorthogonalization.
pub fn estimate_spectrum_at<S: TangentStepper>(
    stepper: &mut S,
    initial_time: f64,
    initial_state: &[f64],
    config: &BenettinConfig,
) -> Result<SpectrumEstimate> {
    config.validate_counts()?;
    if !initial_time.is_finite() {
        return Err(LyapunovError::InvalidParameter {
            field: "initial_time",
            value: initial_time,
        });
    }
    let dimension = stepper.dimension();
    if dimension == 0 {
        return Err(LyapunovError::ZeroDimension);
    }
    if initial_state.len() != dimension {
        return Err(LyapunovError::DimensionMismatch {
            context: "initial_state",
            expected: dimension,
            actual: initial_state.len(),
        });
    }
    ensure_finite(initial_state, "initial_state")?;
    if config.top_k > dimension {
        return Err(LyapunovError::InvalidExponentCount {
            requested: config.top_k,
            dimension,
        });
    }

    let mut state = initial_state.to_vec();
    let tangent_len =
        dimension
            .checked_mul(config.top_k)
            .ok_or(LyapunovError::InvalidExponentCount {
                requested: config.top_k,
                dimension,
            })?;
    let mut tangents = vec![0.0; tangent_len];
    initialize_basis(&mut tangents, dimension, config.seed)?;
    let mut time = initial_time;

    let window_duration = config.step_size * config.steps_per_window as f64;
    if !window_duration.is_finite() || window_duration <= 0.0 {
        return Err(LyapunovError::InvalidParameter {
            field: "window_duration",
            value: window_duration,
        });
    }

    for _ in 0..config.transient_windows {
        advance_window(stepper, &mut time, &mut state, &mut tangents, config)?;
        modified_gram_schmidt(&mut tangents, dimension)?;
    }

    let mut log_sums = vec![0.0; config.top_k];
    let mut finite_time = Vec::with_capacity(config.windows);

    for window in 0..config.windows {
        advance_window(stepper, &mut time, &mut state, &mut tangents, config)?;
        let log_growth = modified_gram_schmidt(&mut tangents, dimension)?;
        let elapsed = window_duration * (window + 1) as f64;
        let mut local_exponents = Vec::with_capacity(config.top_k);
        let mut running_exponents = Vec::with_capacity(config.top_k);
        for (sum, growth) in log_sums.iter_mut().zip(log_growth) {
            *sum += growth;
            local_exponents.push(growth / window_duration);
            running_exponents.push(*sum / elapsed);
        }
        finite_time.push(FiniteTimeEstimate {
            end_time: time,
            local_exponents,
            running_exponents,
        });
    }

    let measured_time = window_duration * config.windows as f64;
    let exponents = log_sums
        .into_iter()
        .map(|sum| sum / measured_time)
        .collect();
    Ok(SpectrumEstimate {
        exponents,
        finite_time,
        final_state: state,
        final_time: time,
        measured_time,
    })
}

fn advance_window<S: TangentStepper>(
    stepper: &mut S,
    time: &mut f64,
    state: &mut [f64],
    tangents: &mut [f64],
    config: &BenettinConfig,
) -> Result<()> {
    for _ in 0..config.steps_per_window {
        stepper.step(*time, state, tangents, config.top_k, config.step_size)?;
        let next_time = *time + config.step_size;
        if !next_time.is_finite() {
            return Err(LyapunovError::InvalidParameter {
                field: "time",
                value: next_time,
            });
        }
        *time = next_time;
        ensure_finite(state, "state")?;
        ensure_finite(tangents, "tangents")?;
    }
    Ok(())
}

fn ensure_finite(values: &[f64], context: &'static str) -> Result<()> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(LyapunovError::NonFiniteValue { context, index });
    }
    Ok(())
}

fn initialize_basis(tangents: &mut [f64], dimension: usize, seed: u64) -> Result<()> {
    let mut generator = SplitMix64::new(seed);
    for value in tangents.iter_mut() {
        *value = generator.symmetric_unit();
    }
    modified_gram_schmidt(tangents, dimension).map(|_| ())
}

fn modified_gram_schmidt(tangents: &mut [f64], dimension: usize) -> Result<Vec<f64>> {
    let count = tangents.len() / dimension;
    let mut log_norms = Vec::with_capacity(count);
    for vector in 0..count {
        let start = vector * dimension;
        for _ in 0..2 {
            for previous in 0..vector {
                let previous_start = previous * dimension;
                let projection = dot_disjoint(tangents, previous_start, start, dimension);
                subtract_projection(tangents, previous_start, start, dimension, projection);
            }
        }
        let norm_squared = tangents[start..start + dimension]
            .iter()
            .map(|value| value * value)
            .sum::<f64>();
        let norm = norm_squared.sqrt();
        if !norm.is_finite() || norm <= f64::MIN_POSITIVE {
            return Err(LyapunovError::DegenerateTangent { vector });
        }
        log_norms.push(norm.ln());
        for value in &mut tangents[start..start + dimension] {
            *value /= norm;
        }
    }
    Ok(log_norms)
}

fn dot_disjoint(values: &[f64], left_start: usize, right_start: usize, dimension: usize) -> f64 {
    values[left_start..left_start + dimension]
        .iter()
        .zip(&values[right_start..right_start + dimension])
        .map(|(left, right)| left * right)
        .sum()
}

fn subtract_projection(
    values: &mut [f64],
    basis_start: usize,
    target_start: usize,
    dimension: usize,
    projection: f64,
) {
    let (basis_prefix, target_suffix) = values.split_at_mut(target_start);
    let basis = &basis_prefix[basis_start..basis_start + dimension];
    let target = &mut target_suffix[..dimension];
    for (target, basis) in target.iter_mut().zip(basis) {
        *target -= projection * basis;
    }
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn symmetric_unit(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / ((1_u64 << 53) as f64);
        2.0 * ((self.next_u64() >> 11) as f64 * SCALE) - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ContinuousSystem, Rk4};

    #[derive(Clone)]
    struct Diagonal {
        rates: Vec<f64>,
    }

    impl ContinuousSystem for Diagonal {
        fn dimension(&self) -> usize {
            self.rates.len()
        }

        fn vector_field(&self, _time: f64, state: &[f64], output: &mut [f64]) -> Result<()> {
            for ((output, state), rate) in output.iter_mut().zip(state).zip(&self.rates) {
                *output = rate * state;
            }
            Ok(())
        }

        fn jacobian_vector_product(
            &self,
            _time: f64,
            _state: &[f64],
            direction: &[f64],
            output: &mut [f64],
        ) -> Result<()> {
            for ((output, direction), rate) in output.iter_mut().zip(direction).zip(&self.rates) {
                *output = rate * direction;
            }
            Ok(())
        }
    }

    #[test]
    fn diagonal_linear_flow_recovers_analytic_spectrum() {
        let mut stepper = Rk4::new(Diagonal {
            rates: vec![0.35, 0.05, -0.4],
        });
        let config = BenettinConfig::new(3, 0.02, 10, 1_000)
            .unwrap()
            .with_transient_windows(50)
            .with_seed(7);
        let estimate = estimate_spectrum(&mut stepper, &[0.0, 0.0, 0.0], &config).unwrap();
        for (actual, expected) in estimate.exponents().iter().zip([0.35, 0.05, -0.4]) {
            assert!((actual - expected).abs() < 2.0e-3, "{actual} != {expected}");
        }
        assert_eq!(estimate.finite_time_estimates().len(), 1_000);
        assert!((estimate.measured_time() - 200.0).abs() < 1.0e-10);
    }

    #[test]
    fn largest_exponent_helper_recovers_growth_rate() {
        let mut stepper = Rk4::new(Diagonal {
            rates: vec![-0.3, 0.2, -0.1],
        });
        let estimate = estimate_largest(&mut stepper, &[1.0, 1.0, 1.0], 0.01, 20, 1_000).unwrap();
        assert!((estimate.exponents()[0] - 0.2).abs() < 0.01);
    }

    #[test]
    fn seeded_calculation_is_reproducible() {
        let system = Diagonal {
            rates: vec![0.2, -0.1],
        };
        let config = BenettinConfig::new(2, 0.01, 5, 20).unwrap().with_seed(1234);
        let first =
            estimate_spectrum(&mut Rk4::new(system.clone()), &[0.4, -0.5], &config).unwrap();
        let second = estimate_spectrum(&mut Rk4::new(system), &[0.4, -0.5], &config).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn invalid_inputs_return_errors() {
        let mut stepper = Rk4::new(Diagonal {
            rates: vec![0.1, 0.2],
        });
        let config = BenettinConfig::new(2, 0.01, 2, 4).unwrap();
        assert!(matches!(
            estimate_spectrum(&mut stepper, &[1.0], &config),
            Err(LyapunovError::DimensionMismatch { .. })
        ));

        let too_many = BenettinConfig::new(3, 0.01, 2, 4).unwrap();
        assert!(matches!(
            estimate_spectrum(&mut stepper, &[1.0, 1.0], &too_many),
            Err(LyapunovError::InvalidExponentCount { .. })
        ));
        assert!(BenettinConfig::new(1, 0.0, 2, 4).is_err());
        assert!(BenettinConfig::new(1, 0.01, 0, 4).is_err());
        assert!(BenettinConfig::new(1, 0.01, 2, 0).is_err());
        assert!(matches!(
            estimate_spectrum(&mut stepper, &[f64::NAN, 1.0], &config),
            Err(LyapunovError::NonFiniteValue {
                context: "initial_state",
                index: 0
            })
        ));
    }

    #[test]
    fn collapsed_tangent_returns_a_specific_error() {
        struct CollapsingStepper;

        impl TangentStepper for CollapsingStepper {
            fn dimension(&self) -> usize {
                2
            }

            fn step(
                &mut self,
                _time: f64,
                _state: &mut [f64],
                tangents: &mut [f64],
                _tangent_count: usize,
                _step_size: f64,
            ) -> Result<()> {
                tangents.fill(0.0);
                Ok(())
            }
        }

        let config = BenettinConfig::new(1, 0.1, 1, 1).unwrap();
        let error = estimate_spectrum(&mut CollapsingStepper, &[0.0, 0.0], &config)
            .expect_err("a zero tangent must not produce a finite exponent");
        assert_eq!(error, LyapunovError::DegenerateTangent { vector: 0 });
    }

    #[test]
    fn matrix_free_jvp_is_used_for_each_tangent() {
        use core::cell::Cell;

        struct Counted {
            jvp_calls: Cell<usize>,
        }

        impl ContinuousSystem for Counted {
            fn dimension(&self) -> usize {
                4
            }

            fn vector_field(&self, _time: f64, _state: &[f64], output: &mut [f64]) -> Result<()> {
                output.fill(0.0);
                Ok(())
            }

            fn jacobian_vector_product(
                &self,
                _time: f64,
                _state: &[f64],
                direction: &[f64],
                output: &mut [f64],
            ) -> Result<()> {
                self.jvp_calls.set(self.jvp_calls.get() + 1);
                output.copy_from_slice(direction);
                Ok(())
            }
        }

        let mut stepper = Rk4::new(Counted {
            jvp_calls: Cell::new(0),
        });
        let config = BenettinConfig::new(2, 0.01, 3, 2).unwrap();
        estimate_spectrum(&mut stepper, &[0.0; 4], &config).unwrap();
        assert_eq!(stepper.system().jvp_calls.get(), 4 * 2 * 3 * 2);
    }
}
