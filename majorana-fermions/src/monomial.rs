use crate::{FermionError, Result};
use core::fmt;

/// Canonically ordered product of distinct Majorana generators.
///
/// The stored indices `i_1 < ... < i_k` represent
/// `gamma_(i_1) ... gamma_(i_k)`. Generator indices are not restricted by a
/// machine-word bitmap, so sparse high-dimensional models do not inherit a
/// small fixed rank limit.
#[derive(Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MajoranaMonomial {
    indices: Vec<usize>,
}

impl MajoranaMonomial {
    /// Creates the scalar monomial.
    #[must_use]
    pub const fn scalar() -> Self {
        Self {
            indices: Vec::new(),
        }
    }

    /// Creates one Majorana generator.
    #[must_use]
    pub fn generator(index: usize) -> Self {
        Self {
            indices: vec![index],
        }
    }

    /// Creates a monomial from strictly increasing generator indices.
    pub fn new(indices: Vec<usize>) -> Result<Self> {
        for (offset, pair) in indices.windows(2).enumerate() {
            if pair[0] >= pair[1] {
                return Err(FermionError::NonCanonicalIndices {
                    position: offset + 1,
                    previous: pair[0],
                    current: pair[1],
                });
            }
        }
        Ok(Self { indices })
    }

    /// Returns the ordered generator indices.
    #[must_use]
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Returns the monomial grade.
    #[must_use]
    pub fn grade(&self) -> usize {
        self.indices.len()
    }

    /// Returns whether this is the scalar monomial.
    #[must_use]
    pub fn is_scalar(&self) -> bool {
        self.indices.is_empty()
    }

    /// Multiplies two monomials with the exact Clifford sign.
    ///
    /// All generators square to `+1`. The output is canonical and contains the
    /// symmetric difference of the two index sets.
    #[must_use]
    pub fn multiply(&self, right: &Self) -> SignedMonomial {
        let mut odd_crossings = false;
        let mut right_below = 0;
        for left_index in &self.indices {
            while right_below < right.indices.len() && right.indices[right_below] < *left_index {
                right_below += 1;
            }
            odd_crossings ^= right_below % 2 == 1;
        }

        let mut indices = Vec::with_capacity(self.grade() + right.grade());
        let (mut left, mut right_index) = (0, 0);
        while left < self.indices.len() || right_index < right.indices.len() {
            match (self.indices.get(left), right.indices.get(right_index)) {
                (Some(a), Some(b)) if a < b => {
                    indices.push(*a);
                    left += 1;
                }
                (Some(a), Some(b)) if b < a => {
                    indices.push(*b);
                    right_index += 1;
                }
                (Some(_), Some(_)) => {
                    left += 1;
                    right_index += 1;
                }
                (Some(a), None) => {
                    indices.push(*a);
                    left += 1;
                }
                (None, Some(b)) => {
                    indices.push(*b);
                    right_index += 1;
                }
                (None, None) => break,
            }
        }

        SignedMonomial {
            sign: if odd_crossings { -1 } else { 1 },
            monomial: Self { indices },
        }
    }

    /// Sign acquired when taking the Hermitian adjoint of this monomial.
    #[must_use]
    pub fn adjoint_sign(&self) -> i8 {
        match self.indices.len() % 4 {
            0 | 1 => 1,
            _ => -1,
        }
    }
}

impl fmt::Display for MajoranaMonomial {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_scalar() {
            return formatter.write_str("1");
        }
        for index in &self.indices {
            write!(formatter, "γ{index}")?;
        }
        Ok(())
    }
}

/// Canonical monomial together with its exact product sign.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SignedMonomial {
    sign: i8,
    monomial: MajoranaMonomial,
}

impl SignedMonomial {
    /// Returns `+1` or `-1`.
    #[must_use]
    pub const fn sign(&self) -> i8 {
        self.sign
    }

    /// Borrows the canonical output monomial.
    #[must_use]
    pub const fn monomial(&self) -> &MajoranaMonomial {
        &self.monomial
    }

    /// Splits the product into its sign and canonical monomial.
    #[must_use]
    pub fn into_parts(self) -> (i8, MajoranaMonomial) {
        (self.sign, self.monomial)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generators_square_and_anticommute() {
        let left = MajoranaMonomial::generator(2);
        let right = MajoranaMonomial::generator(7);
        let square = left.multiply(&left);
        assert_eq!(square.sign(), 1);
        assert!(square.monomial().is_scalar());

        let forward = left.multiply(&right);
        let backward = right.multiply(&left);
        assert_eq!(forward.monomial(), backward.monomial());
        assert_eq!(forward.sign(), -backward.sign());
    }

    #[test]
    fn overlapping_blades_have_correct_sign() {
        let left = MajoranaMonomial::new(vec![0, 1]).unwrap();
        let right = MajoranaMonomial::generator(0);
        let product = left.multiply(&right);
        assert_eq!(product.sign(), -1);
        assert_eq!(product.monomial().indices(), &[1]);
    }

    #[test]
    fn rejects_unsorted_or_repeated_indices() {
        assert!(MajoranaMonomial::new(vec![1, 0]).is_err());
        assert!(MajoranaMonomial::new(vec![1, 1]).is_err());
    }
}
