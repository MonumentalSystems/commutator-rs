//! Checked spectral diagnostics for finite quantum systems.
//!
//! `quantum-chaos` consumes eigenvalues computed elsewhere. It keeps
//! degeneracy thresholds, unfolding policies, normalization, and finite-window
//! sampling explicit, and does not classify a spectrum from one statistic.
//!
//! ```
//! use quantum_chaos::{Spectrum, UnfoldingPolicy};
//!
//! let spectrum = Spectrum::try_from_unsorted(vec![3.2, 0.1, 2.0, 1.0])?;
//! let ratios = spectrum.adjacent_gap_ratios(1e-12)?;
//! assert_eq!(ratios.ratios().len(), 2);
//! let unfolded = spectrum.unfold(UnfoldingPolicy::Affine { minimum_gap: 1e-12 })?;
//! assert_eq!(unfolded.spectral_form_factor(&[0.0])?, vec![4.0]);
//! # Ok::<(), quantum_chaos::ChaosError>(())
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod error;
mod form_factor;
mod gaps;
mod number_variance;
mod spectrum;

pub use error::ChaosError;
pub use gaps::{
    GapRatioStatistics, MeanReferenceComparison, GOE_SURMISE_MEAN_GAP_RATIO, POISSON_MEAN_GAP_RATIO,
};
pub use number_variance::NumberVariance;
pub use spectrum::{Spectrum, UnfoldedSpectrum, UnfoldingMethod, UnfoldingPolicy};

pub(crate) use spectrum::validate_strict_gaps;

/// Result type for checked spectral diagnostics.
pub type Result<T> = core::result::Result<T, ChaosError>;
