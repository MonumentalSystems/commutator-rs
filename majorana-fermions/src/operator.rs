use crate::{Complex64, FermionError, MajoranaMonomial, Result};
use std::collections::BTreeMap;

/// Deterministic sparse complex linear combination of Majorana monomials.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FermionOperator {
    terms: BTreeMap<MajoranaMonomial, Complex64>,
}

impl FermionOperator {
    /// Creates the zero operator.
    #[must_use]
    pub const fn zero() -> Self {
        Self {
            terms: BTreeMap::new(),
        }
    }

    /// Creates the identity operator.
    #[must_use]
    pub fn identity() -> Self {
        let mut terms = BTreeMap::new();
        terms.insert(MajoranaMonomial::scalar(), Complex64::new(1.0, 0.0));
        Self { terms }
    }

    /// Creates a one-term operator.
    pub fn from_term(monomial: MajoranaMonomial, coefficient: Complex64) -> Result<Self> {
        let mut operator = Self::zero();
        operator.add_term(monomial, coefficient)?;
        Ok(operator)
    }

    /// Returns the number of stored nonzero terms.
    #[must_use]
    pub fn term_count(&self) -> usize {
        self.terms.len()
    }

    /// Returns whether this is exactly the zero operator.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Iterates over terms in canonical monomial order.
    pub fn terms(
        &self,
    ) -> impl ExactSizeIterator<Item = (&MajoranaMonomial, &Complex64)> + DoubleEndedIterator {
        self.terms.iter()
    }

    /// Returns a monomial coefficient, or zero when the term is absent.
    #[must_use]
    pub fn coefficient(&self, monomial: &MajoranaMonomial) -> Complex64 {
        self.terms.get(monomial).copied().unwrap_or_default()
    }

    /// Adds a checked coefficient to a monomial, combining like terms.
    ///
    /// Terms that cancel exactly are removed from the sparse representation.
    pub fn add_term(&mut self, monomial: MajoranaMonomial, coefficient: Complex64) -> Result<()> {
        validate_complex(coefficient)?;
        if is_zero(coefficient) {
            return Ok(());
        }
        let updated = self.coefficient(&monomial) + coefficient;
        validate_complex(updated)?;
        if is_zero(updated) {
            self.terms.remove(&monomial);
        } else {
            self.terms.insert(monomial, updated);
        }
        Ok(())
    }

    /// Returns the checked sum of two sparse operators.
    pub fn add(&self, right: &Self) -> Result<Self> {
        let mut sum = self.clone();
        for (monomial, coefficient) in right.terms() {
            sum.add_term(monomial.clone(), *coefficient)?;
        }
        Ok(sum)
    }

    /// Returns the checked difference of two sparse operators.
    pub fn subtract(&self, right: &Self) -> Result<Self> {
        let mut difference = self.clone();
        for (monomial, coefficient) in right.terms() {
            difference.add_term(monomial.clone(), -*coefficient)?;
        }
        Ok(difference)
    }

    /// Multiplies every coefficient by a checked scalar.
    pub fn scaled(&self, factor: Complex64) -> Result<Self> {
        validate_complex(factor)?;
        let mut scaled = Self::zero();
        for (monomial, coefficient) in self.terms() {
            scaled.add_term(monomial.clone(), *coefficient * factor)?;
        }
        Ok(scaled)
    }

    /// Returns the geometric/operator product.
    pub fn multiply(&self, right: &Self) -> Result<Self> {
        let mut product = Self::zero();
        for (left_monomial, left_coefficient) in self.terms() {
            for (right_monomial, right_coefficient) in right.terms() {
                let signed = left_monomial.multiply(right_monomial);
                let sign = f64::from(signed.sign());
                let coefficient = *left_coefficient * *right_coefficient * sign;
                product.add_term(signed.into_parts().1, coefficient)?;
            }
        }
        Ok(product)
    }

    /// Returns the Hermitian adjoint.
    #[must_use]
    pub fn adjoint(&self) -> Self {
        let terms = self
            .terms()
            .map(|(monomial, coefficient)| {
                let sign = f64::from(monomial.adjoint_sign());
                (monomial.clone(), coefficient.conj() * sign)
            })
            .collect();
        Self { terms }
    }

    /// Returns `[self, right] = self right - right self`.
    pub fn commutator(&self, right: &Self) -> Result<Self> {
        self.multiply(right)?.subtract(&right.multiply(self)?)
    }

    /// Returns `{self, right} = self right + right self`.
    pub fn anticommutator(&self, right: &Self) -> Result<Self> {
        self.multiply(right)?.add(&right.multiply(self)?)
    }

    /// Tests equality coefficient-by-coefficient within an absolute tolerance.
    pub fn approx_eq(&self, right: &Self, tolerance: f64) -> Result<bool> {
        validate_tolerance(tolerance)?;
        for monomial in self.terms.keys().chain(right.terms.keys()) {
            if (self.coefficient(monomial) - right.coefficient(monomial)).norm() > tolerance {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Tests Hermiticity within an absolute coefficient tolerance.
    pub fn is_hermitian(&self, tolerance: f64) -> Result<bool> {
        self.approx_eq(&self.adjoint(), tolerance)
    }
}

/// Returns a single Majorana generator as an operator.
#[must_use]
pub fn majorana(index: usize) -> FermionOperator {
    FermionOperator::from_term(MajoranaMonomial::generator(index), Complex64::new(1.0, 0.0))
        .expect("unit coefficient is finite")
}

/// Returns the canonical annihilation operator for one fermionic mode.
///
/// With `a_j = gamma_(2j)` and `b_j = gamma_(2j+1)`, this uses
/// `c_j = (a_j + i b_j) / 2`.
pub fn annihilation(mode: usize) -> Result<FermionOperator> {
    let (even, odd) = mode_indices(mode)?;
    let mut operator = FermionOperator::zero();
    operator.add_term(MajoranaMonomial::generator(even), Complex64::new(0.5, 0.0))?;
    operator.add_term(MajoranaMonomial::generator(odd), Complex64::new(0.0, 0.5))?;
    Ok(operator)
}

/// Returns the canonical creation operator for one fermionic mode.
///
/// This is the Hermitian adjoint of [`annihilation`].
pub fn creation(mode: usize) -> Result<FermionOperator> {
    Ok(annihilation(mode)?.adjoint())
}

/// Returns `n_j = c_j^dagger c_j`.
pub fn number_operator(mode: usize) -> Result<FermionOperator> {
    creation(mode)?.multiply(&annihilation(mode)?)
}

/// Returns local fermion parity `(-1)^n = 1 - 2 n` for one mode.
pub fn mode_parity(mode: usize) -> Result<FermionOperator> {
    let number = number_operator(mode)?;
    FermionOperator::identity().subtract(&number.scaled(Complex64::new(2.0, 0.0))?)
}

/// Returns total fermion parity for modes `0..mode_count`.
pub fn fermion_parity(mode_count: usize) -> Result<FermionOperator> {
    let mut parity = FermionOperator::identity();
    for mode in 0..mode_count {
        parity = parity.multiply(&mode_parity(mode)?)?;
    }
    Ok(parity)
}

pub(crate) fn mode_indices(mode: usize) -> Result<(usize, usize)> {
    let even = mode
        .checked_mul(2)
        .ok_or(FermionError::ModeIndexOverflow { mode })?;
    let odd = even
        .checked_add(1)
        .ok_or(FermionError::ModeIndexOverflow { mode })?;
    Ok((even, odd))
}

fn validate_complex(value: Complex64) -> Result<()> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(FermionError::NonFiniteCoefficient)
    }
}

fn validate_tolerance(tolerance: f64) -> Result<()> {
    if tolerance.is_finite() && tolerance >= 0.0 {
        Ok(())
    } else {
        Err(FermionError::InvalidTolerance { tolerance })
    }
}

fn is_zero(value: Complex64) -> bool {
    value.re == 0.0 && value.im == 0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_anticommutation_relations_hold() {
        for left in 0..3 {
            for right in 0..3 {
                let c_left = annihilation(left).unwrap();
                let c_right = annihilation(right).unwrap();
                assert!(c_left.anticommutator(&c_right).unwrap().is_zero());

                let c_right_dagger = creation(right).unwrap();
                let car = c_left.anticommutator(&c_right_dagger).unwrap();
                if left == right {
                    assert_eq!(car, FermionOperator::identity());
                } else {
                    assert!(car.is_zero());
                }
            }
        }
    }

    #[test]
    fn number_and_parity_have_expected_algebra() {
        let number = number_operator(4).unwrap();
        assert_eq!(number.multiply(&number).unwrap(), number);
        assert!(number.is_hermitian(0.0).unwrap());

        let parity = mode_parity(4).unwrap();
        assert_eq!(
            parity.multiply(&parity).unwrap(),
            FermionOperator::identity()
        );
        assert!(parity.is_hermitian(0.0).unwrap());
        assert!(parity
            .anticommutator(&annihilation(4).unwrap())
            .unwrap()
            .is_zero());
    }

    #[test]
    fn total_parity_commutes_with_number_conserving_terms() {
        let parity = fermion_parity(5).unwrap();
        let hopping = creation(1)
            .unwrap()
            .multiply(&annihilation(3).unwrap())
            .unwrap();
        assert!(parity.commutator(&hopping).unwrap().is_zero());
    }

    #[test]
    fn adjoint_reverses_bivector_sign() {
        let bivector = FermionOperator::from_term(
            MajoranaMonomial::new(vec![2, 9]).unwrap(),
            Complex64::new(1.0, 0.0),
        )
        .unwrap();
        assert_eq!(
            bivector.adjoint(),
            bivector.scaled(Complex64::new(-1.0, 0.0)).unwrap()
        );
        let hermitian = bivector.scaled(Complex64::new(0.0, 1.0)).unwrap();
        assert!(hermitian.is_hermitian(0.0).unwrap());
    }

    #[test]
    fn rejects_non_finite_coefficients_and_tolerances() {
        assert!(FermionOperator::from_term(
            MajoranaMonomial::scalar(),
            Complex64::new(f64::NAN, 0.0)
        )
        .is_err());
        assert!(FermionOperator::zero().is_hermitian(-1.0).is_err());
    }
}
