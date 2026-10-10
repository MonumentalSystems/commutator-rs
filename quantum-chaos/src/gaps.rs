//! Adjacent-gap ratio statistics and reference means.

use crate::{validate_strict_gaps, ChaosError, Result, Spectrum};

/// Poisson mean adjacent-gap ratio, `2 ln(2) - 1`.
pub const POISSON_MEAN_GAP_RATIO: f64 = 0.386_294_361_119_890_6;

/// GOE ratio-distribution surmise mean, `4 - 2 sqrt(3)`.
///
/// This is the small-matrix surmise, not the distinct large-matrix numerical
/// GOE mean sometimes quoted in the literature.
pub const GOE_SURMISE_MEAN_GAP_RATIO: f64 = 0.535_898_384_862_245_4;

/// Absolute deviations of a sample mean from two common reference means.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeanReferenceComparison {
    poisson_absolute_deviation: f64,
    goe_surmise_absolute_deviation: f64,
}

impl MeanReferenceComparison {
    /// Returns absolute deviation from the Poisson reference mean.
    pub const fn poisson_absolute_deviation(self) -> f64 {
        self.poisson_absolute_deviation
    }

    /// Returns absolute deviation from the GOE-surmise reference mean.
    pub const fn goe_surmise_absolute_deviation(self) -> f64 {
        self.goe_surmise_absolute_deviation
    }
}

/// Adjacent-gap ratios and their arithmetic mean.
#[derive(Clone, Debug, PartialEq)]
pub struct GapRatioStatistics {
    ratios: Vec<f64>,
    mean: f64,
}

impl GapRatioStatistics {
    /// Returns each `min(s_n, s_(n+1)) / max(s_n, s_(n+1))` ratio.
    pub fn ratios(&self) -> &[f64] {
        &self.ratios
    }

    /// Returns the arithmetic mean of all adjacent-gap ratios.
    pub const fn mean(&self) -> f64 {
        self.mean
    }

    /// Compares the sample mean to Poisson and GOE-surmise means.
    ///
    /// The result is descriptive and is not a statistical classification.
    pub fn reference_comparison(&self) -> MeanReferenceComparison {
        MeanReferenceComparison {
            poisson_absolute_deviation: (self.mean - POISSON_MEAN_GAP_RATIO).abs(),
            goe_surmise_absolute_deviation: (self.mean - GOE_SURMISE_MEAN_GAP_RATIO).abs(),
        }
    }
}

impl Spectrum {
    /// Computes adjacent-gap ratios after strict gap validation.
    ///
    /// `minimum_gap` is an absolute degeneracy threshold. Every gap must be
    /// strictly larger, so exact or near degeneracies are reported rather
    /// than silently removed or mixed across symmetry sectors.
    pub fn adjacent_gap_ratios(&self, minimum_gap: f64) -> Result<GapRatioStatistics> {
        if self.levels().len() < 3 {
            return Err(ChaosError::TooFewLevels {
                required: 3,
                actual: self.levels().len(),
            });
        }
        let gaps = validate_strict_gaps(self.levels(), minimum_gap)?;
        let ratios: Vec<f64> = gaps
            .windows(2)
            .map(|pair| pair[0].min(pair[1]) / pair[0].max(pair[1]))
            .collect();
        let mean = ratios.iter().sum::<f64>() / ratios.len() as f64;
        Ok(GapRatioStatistics { ratios, mean })
    }
}
