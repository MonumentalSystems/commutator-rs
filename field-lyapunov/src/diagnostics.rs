use crate::{LyapunovError, Result};

/// Computes the Kaplan-Yorke (Lyapunov) dimension from a spectrum.
///
/// The input may be in any order; a sorted copy is used internally. The
/// result is zero when the largest exponent is negative and equals the number
/// of supplied exponents when every cumulative sum is non-negative. A partial
/// spectrum can only provide a dimension relative to the supplied exponents.
pub fn kaplan_yorke_dimension(exponents: &[f64]) -> Result<f64> {
    validate_exponents(exponents)?;
    let mut sorted = exponents.to_vec();
    sorted.sort_by(|left, right| right.total_cmp(left));
    let mut cumulative = 0.0;
    for (index, exponent) in sorted.iter().copied().enumerate() {
        let next = cumulative + exponent;
        if next < 0.0 {
            if index == 0 {
                return Ok(0.0);
            }
            return Ok(index as f64 + cumulative / exponent.abs());
        }
        cumulative = next;
    }
    Ok(sorted.len() as f64)
}

/// Sums positive Lyapunov exponents as a Kolmogorov-Sinai entropy-rate estimate.
///
/// Equality with metric entropy requires the hypotheses of Pesin's identity;
/// this helper merely computes its Lyapunov-spectrum side. Rates use natural
/// logarithms and therefore have units of nats per unit time.
pub fn kolmogorov_sinai_entropy(exponents: &[f64]) -> Result<f64> {
    validate_exponents(exponents)?;
    Ok(exponents
        .iter()
        .copied()
        .filter(|exponent| *exponent > 0.0)
        .sum())
}

fn validate_exponents(exponents: &[f64]) -> Result<()> {
    if exponents.is_empty() {
        return Err(LyapunovError::ZeroCount { field: "exponents" });
    }
    if let Some(index) = exponents.iter().position(|value| !value.is_finite()) {
        return Err(LyapunovError::NonFiniteValue {
            context: "exponents",
            index,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kaplan_yorke_sorts_and_interpolates() {
        let dimension = kaplan_yorke_dimension(&[-2.0, 1.0, 0.0]).unwrap();
        assert!((dimension - 2.5).abs() < f64::EPSILON);
        assert_eq!(kaplan_yorke_dimension(&[-2.0, -1.0]).unwrap(), 0.0);
        assert_eq!(kaplan_yorke_dimension(&[0.2, 0.1]).unwrap(), 2.0);
    }

    #[test]
    fn entropy_sums_only_positive_rates() {
        let entropy = kolmogorov_sinai_entropy(&[-3.0, 0.4, 0.0, 0.2]).unwrap();
        assert!((entropy - 0.6).abs() < 1.0e-12);
    }

    #[test]
    fn diagnostics_reject_invalid_spectra() {
        assert!(kaplan_yorke_dimension(&[]).is_err());
        assert!(kolmogorov_sinai_entropy(&[f64::NAN]).is_err());
    }
}
