//! Spectral form factor.

use num_complex::Complex64;

use crate::{ChaosError, Result, Spectrum, UnfoldedSpectrum};

fn spectral_form_factor(levels: &[f64], times: &[f64]) -> Result<Vec<f64>> {
    let normalization = levels.len() as f64;
    times
        .iter()
        .enumerate()
        .map(|(time_index, &time)| {
            if !time.is_finite() {
                return Err(ChaosError::NonFiniteTime { index: time_index });
            }
            let mut amplitude = Complex64::new(0.0, 0.0);
            for (level_index, &level) in levels.iter().enumerate() {
                let phase = level * time;
                if !phase.is_finite() {
                    return Err(ChaosError::PhaseOverflow {
                        time_index,
                        level_index,
                    });
                }
                amplitude += Complex64::new(phase.cos(), -phase.sin());
            }
            Ok(amplitude.norm_sqr() / normalization)
        })
        .collect()
}

impl Spectrum {
    /// Evaluates `|sum_n exp(-i E_n t)|^2 / N` at each supplied time.
    ///
    /// No connected subtraction, ensemble averaging, time rescaling, or
    /// unfolding is applied implicitly.
    pub fn spectral_form_factor(&self, times: &[f64]) -> Result<Vec<f64>> {
        spectral_form_factor(self.levels(), times)
    }
}

impl UnfoldedSpectrum {
    /// Evaluates the normalized spectral form factor of unfolded levels.
    pub fn spectral_form_factor(&self, times: &[f64]) -> Result<Vec<f64>> {
        spectral_form_factor(self.levels(), times)
    }
}
