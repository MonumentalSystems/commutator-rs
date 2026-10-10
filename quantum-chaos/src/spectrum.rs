//! Sorted spectra and explicit unfolding policies.

use crate::{ChaosError, Result};

/// An explicit finite-spectrum unfolding policy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum UnfoldingPolicy {
    /// Divide all spacings by the sequence-wide arithmetic mean gap.
    Affine {
        /// Absolute threshold below which a gap is treated as degenerate.
        minimum_gap: f64,
    },
    /// Divide each gap by a centered moving average of nearby raw gaps.
    ///
    /// The normalized gaps are cumulatively summed from zero. Near edges the
    /// window is truncated. A half-window of zero is rejected because it
    /// would map every gap to exactly one and erase all local information.
    LocalGap {
        /// Number of neighboring gaps included on each side.
        half_window: usize,
        /// Absolute threshold below which a gap is treated as degenerate.
        minimum_gap: f64,
    },
}

/// Metadata recording how an unfolded sequence was obtained.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UnfoldingMethod {
    /// The caller supplied a sequence already interpreted as unfolded.
    CallerSupplied,
    /// A single global mean spacing was removed.
    Affine,
    /// A moving mean of raw gaps was removed.
    LocalGap {
        /// Selected number of neighboring gaps on each side.
        half_window: usize,
    },
}

/// A finite sorted sequence of raw energy levels.
///
/// Exact degeneracies are allowed at construction because some diagnostics,
/// including the spectral form factor, remain defined. Gap-based diagnostics
/// perform their own explicit degeneracy checks.
#[derive(Clone, Debug, PartialEq)]
pub struct Spectrum {
    levels: Vec<f64>,
}

impl Spectrum {
    /// Validates a nondecreasing sequence of finite levels.
    pub fn try_from_sorted(levels: Vec<f64>) -> Result<Self> {
        if levels.is_empty() {
            return Err(ChaosError::TooFewLevels {
                required: 1,
                actual: 0,
            });
        }
        if let Some(index) = levels.iter().position(|level| !level.is_finite()) {
            return Err(ChaosError::NonFiniteLevel { index });
        }
        for index in 1..levels.len() {
            if levels[index] < levels[index - 1] {
                return Err(ChaosError::LevelsNotSorted { index });
            }
        }
        Ok(Self { levels })
    }

    /// Sorts finite levels into ascending order and validates them.
    pub fn try_from_unsorted(mut levels: Vec<f64>) -> Result<Self> {
        if let Some(index) = levels.iter().position(|level| !level.is_finite()) {
            return Err(ChaosError::NonFiniteLevel { index });
        }
        levels.sort_by(f64::total_cmp);
        Self::try_from_sorted(levels)
    }

    /// Returns the sorted levels.
    pub fn levels(&self) -> &[f64] {
        &self.levels
    }

    /// Returns the number of levels.
    pub fn len(&self) -> usize {
        self.levels.len()
    }

    /// Returns true only for an empty sequence.
    ///
    /// Valid spectra are never empty, but this method accompanies [`Self::len`]
    /// for conventional collection-style APIs.
    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }

    /// Unfolds the spectrum using an explicit finite-sequence policy.
    pub fn unfold(&self, policy: UnfoldingPolicy) -> Result<UnfoldedSpectrum> {
        match policy {
            UnfoldingPolicy::Affine { minimum_gap } => {
                let gaps = validate_strict_gaps(&self.levels, minimum_gap)?;
                let mean_gap = stable_positive_mean(&gaps);
                let mut levels = Vec::with_capacity(self.levels.len());
                levels.push(0.0);
                let mut cumulative = 0.0;
                for gap in gaps {
                    cumulative += gap / mean_gap;
                    levels.push(cumulative);
                }
                Ok(UnfoldedSpectrum {
                    spectrum: Spectrum { levels },
                    method: UnfoldingMethod::Affine,
                })
            }
            UnfoldingPolicy::LocalGap {
                half_window,
                minimum_gap,
            } => {
                if half_window == 0 {
                    return Err(ChaosError::InvalidLocalWindow);
                }
                let gaps = validate_strict_gaps(&self.levels, minimum_gap)?;
                let mut levels = Vec::with_capacity(self.levels.len());
                levels.push(0.0);
                let mut cumulative = 0.0;
                for (index, &gap) in gaps.iter().enumerate() {
                    let start = index.saturating_sub(half_window);
                    let end = index
                        .saturating_add(half_window)
                        .saturating_add(1)
                        .min(gaps.len());
                    let local_mean = stable_positive_mean(&gaps[start..end]);
                    cumulative += gap / local_mean;
                    levels.push(cumulative);
                }
                Ok(UnfoldedSpectrum {
                    spectrum: Spectrum { levels },
                    method: UnfoldingMethod::LocalGap { half_window },
                })
            }
        }
    }
}

/// A sorted sequence interpreted as having unit mean density.
#[derive(Clone, Debug, PartialEq)]
pub struct UnfoldedSpectrum {
    spectrum: Spectrum,
    method: UnfoldingMethod,
}

impl UnfoldedSpectrum {
    /// Validates a caller-supplied unfolded sequence and its strict gaps.
    ///
    /// The caller is responsible for the scientific unfolding procedure; the
    /// crate checks only order, finiteness, and the requested degeneracy bound.
    pub fn try_from_sorted(levels: Vec<f64>, minimum_gap: f64) -> Result<Self> {
        let spectrum = Spectrum::try_from_sorted(levels)?;
        validate_strict_gaps(spectrum.levels(), minimum_gap)?;
        Ok(Self {
            spectrum,
            method: UnfoldingMethod::CallerSupplied,
        })
    }

    /// Returns unfolded levels.
    pub fn levels(&self) -> &[f64] {
        self.spectrum.levels()
    }

    /// Returns metadata describing the unfolding policy.
    pub const fn method(&self) -> UnfoldingMethod {
        self.method
    }

    /// Computes adjacent-gap ratios from the unfolded levels.
    pub fn adjacent_gap_ratios(&self, minimum_gap: f64) -> Result<crate::GapRatioStatistics> {
        self.spectrum.adjacent_gap_ratios(minimum_gap)
    }
}

pub(crate) fn validate_strict_gaps(levels: &[f64], minimum_gap: f64) -> Result<Vec<f64>> {
    if !minimum_gap.is_finite() || minimum_gap < 0.0 {
        return Err(ChaosError::InvalidMinimumGap);
    }
    if levels.len() < 2 {
        return Err(ChaosError::TooFewLevels {
            required: 2,
            actual: levels.len(),
        });
    }
    let mut gaps = Vec::with_capacity(levels.len() - 1);
    for (index, pair) in levels.windows(2).enumerate() {
        let gap = pair[1] - pair[0];
        if !gap.is_finite() {
            return Err(ChaosError::GapOverflow { index });
        }
        if gap <= minimum_gap {
            return Err(ChaosError::GapTooSmall {
                index,
                gap,
                minimum: minimum_gap,
            });
        }
        gaps.push(gap);
    }
    Ok(gaps)
}

fn stable_positive_mean(values: &[f64]) -> f64 {
    let mut mean = values[0];
    for (index, &value) in values.iter().enumerate().skip(1) {
        mean += (value - mean) / (index + 1) as f64;
    }
    mean
}
