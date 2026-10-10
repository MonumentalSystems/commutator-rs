//! Explicit finite-window level-count statistics.

use crate::{ChaosError, Result, UnfoldedSpectrum};

/// Number-count statistics for windows of one fixed length.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NumberVariance {
    interval_length: f64,
    window_count: usize,
    mean_count: f64,
    variance: f64,
    mean_square_deviation_from_unit_density: f64,
}

impl NumberVariance {
    /// Returns the unfolded interval length `L`.
    pub const fn interval_length(self) -> f64 {
        self.interval_length
    }

    /// Returns the number of sampled windows.
    pub const fn window_count(self) -> usize {
        self.window_count
    }

    /// Returns the empirical mean number of levels per window.
    pub const fn mean_count(self) -> f64 {
        self.mean_count
    }

    /// Returns the population variance around the empirical mean count.
    pub const fn variance(self) -> f64 {
        self.variance
    }

    /// Returns the mean of `(N(origin, L) - L)^2`.
    ///
    /// This exposes deviations from the ideal unit-density expectation in
    /// addition to the variance around the finite sample's own mean.
    pub const fn mean_square_deviation_from_unit_density(self) -> f64 {
        self.mean_square_deviation_from_unit_density
    }
}

impl UnfoldedSpectrum {
    /// Computes level-count variance over explicit half-open windows.
    ///
    /// Every `[origin, origin + interval_length)` must lie within the closed
    /// span from the first to last supplied level. This rejects edge-truncated
    /// windows instead of silently biasing counts.
    pub fn number_variance(&self, interval_length: f64, origins: &[f64]) -> Result<NumberVariance> {
        if !interval_length.is_finite() || interval_length <= 0.0 {
            return Err(ChaosError::InvalidIntervalLength);
        }
        if origins.is_empty() {
            return Err(ChaosError::EmptyOrigins);
        }
        let levels = self.levels();
        let first = levels[0];
        let last = levels[levels.len() - 1];
        let mut counts = Vec::with_capacity(origins.len());
        for (index, &origin) in origins.iter().enumerate() {
            let upper = origin + interval_length;
            if !origin.is_finite() || !upper.is_finite() || origin < first || upper > last {
                return Err(ChaosError::InvalidOrigin { index });
            }
            let lower_index = levels.partition_point(|level| *level < origin);
            let upper_index = levels.partition_point(|level| *level < upper);
            counts.push((upper_index - lower_index) as f64);
        }

        let mean_count = counts.iter().sum::<f64>() / counts.len() as f64;
        let variance = counts
            .iter()
            .map(|count| (count - mean_count).powi(2))
            .sum::<f64>()
            / counts.len() as f64;
        let mean_square_deviation_from_unit_density = counts
            .iter()
            .map(|count| (count - interval_length).powi(2))
            .sum::<f64>()
            / counts.len() as f64;
        Ok(NumberVariance {
            interval_length,
            window_count: counts.len(),
            mean_count,
            variance,
            mean_square_deviation_from_unit_density,
        })
    }
}
