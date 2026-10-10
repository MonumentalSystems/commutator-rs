use crate::operator::mode_indices;
use crate::{Complex64, FermionError, FermionOperator, MajoranaMonomial, Result};
use std::collections::BTreeMap;

/// Sparse Hermitian Hamiltonian quadratic in Majorana generators.
///
/// The represented operator is
/// `H = constant + i sum_(j<k) K_jk gamma_j gamma_k`, with real `K_jk`.
/// This is equivalent to the conventional `H = (i/4) gamma^T A gamma`
/// for the antisymmetric matrix `A_jk = 2 K_jk` when `j < k`.
#[derive(Clone, Debug, PartialEq)]
pub struct QuadraticMajoranaHamiltonian {
    majorana_count: usize,
    constant: f64,
    couplings: BTreeMap<(usize, usize), f64>,
}

impl QuadraticMajoranaHamiltonian {
    /// Creates a zero Hamiltonian with a declared number of generators.
    #[must_use]
    pub const fn new(majorana_count: usize) -> Self {
        Self {
            majorana_count,
            constant: 0.0,
            couplings: BTreeMap::new(),
        }
    }

    /// Returns the declared number of Majorana generators.
    #[must_use]
    pub const fn majorana_count(&self) -> usize {
        self.majorana_count
    }

    /// Returns the scalar energy offset.
    #[must_use]
    pub const fn constant(&self) -> f64 {
        self.constant
    }

    /// Sets the scalar energy offset.
    pub fn set_constant(&mut self, constant: f64) -> Result<()> {
        validate_real(constant)?;
        self.constant = constant;
        Ok(())
    }

    /// Adds a scalar energy offset.
    pub fn add_constant(&mut self, offset: f64) -> Result<()> {
        validate_real(offset)?;
        let updated = self.constant + offset;
        validate_real(updated)?;
        self.constant = updated;
        Ok(())
    }

    /// Adds `i coefficient gamma_left gamma_right` to the Hamiltonian.
    ///
    /// Reversing `left` and `right` reverses the effective coefficient. Equal
    /// indices are rejected because they produce a scalar rather than a
    /// bilinear term.
    pub fn add_bilinear(&mut self, left: usize, right: usize, coefficient: f64) -> Result<()> {
        validate_real(coefficient)?;
        self.validate_index(left)?;
        self.validate_index(right)?;
        if left == right {
            return Err(FermionError::RepeatedBilinearIndex { index: left });
        }
        let (pair, oriented) = if left < right {
            ((left, right), coefficient)
        } else {
            ((right, left), -coefficient)
        };
        let updated = self.couplings.get(&pair).copied().unwrap_or(0.0) + oriented;
        validate_real(updated)?;
        if updated == 0.0 {
            self.couplings.remove(&pair);
        } else {
            self.couplings.insert(pair, updated);
        }
        Ok(())
    }

    /// Returns the oriented coefficient multiplying `i gamma_left gamma_right`.
    pub fn bilinear_coefficient(&self, left: usize, right: usize) -> Result<f64> {
        self.validate_index(left)?;
        self.validate_index(right)?;
        if left == right {
            return Ok(0.0);
        }
        if left < right {
            Ok(self.couplings.get(&(left, right)).copied().unwrap_or(0.0))
        } else {
            Ok(-self.couplings.get(&(right, left)).copied().unwrap_or(0.0))
        }
    }

    /// Iterates over nonzero canonical `(j, k, K_jk)` couplings with `j < k`.
    pub fn couplings(&self) -> impl ExactSizeIterator<Item = (usize, usize, f64)> + '_ {
        self.couplings
            .iter()
            .map(|(&(left, right), &coefficient)| (left, right, coefficient))
    }

    /// Expands the sparse quadratic Hamiltonian into a fermionic operator.
    pub fn to_operator(&self) -> Result<FermionOperator> {
        let mut operator = FermionOperator::from_term(
            MajoranaMonomial::scalar(),
            Complex64::new(self.constant, 0.0),
        )?;
        for (left, right, coefficient) in self.couplings() {
            operator.add_term(
                MajoranaMonomial::new(vec![left, right])?,
                Complex64::new(0.0, coefficient),
            )?;
        }
        Ok(operator)
    }

    fn validate_index(&self, index: usize) -> Result<()> {
        if index < self.majorana_count {
            Ok(())
        } else {
            Err(FermionError::MajoranaIndexOutOfBounds {
                index,
                majorana_count: self.majorana_count,
            })
        }
    }
}

/// Constructs an open, real-parameter Kitaev chain.
///
/// For sites `j = 0..L`, the convention is
///
/// `H = -mu sum_j (n_j - 1/2)
///      - t sum_j (c_j^dagger c_(j+1) + h.c.)
///      + delta sum_j (c_j c_(j+1) + h.c.)`.
///
/// At the topological special point `mu = 0` and `delta = t`, only neighboring
/// `b_j`--`a_(j+1)` Majoranas are paired, leaving `a_0` and `b_(L-1)` absent.
pub fn kitaev_chain(
    sites: usize,
    hopping: f64,
    pairing: f64,
    chemical_potential: f64,
) -> Result<QuadraticMajoranaHamiltonian> {
    if sites == 0 {
        return Err(FermionError::ZeroSites);
    }
    validate_real(hopping)?;
    validate_real(pairing)?;
    validate_real(chemical_potential)?;
    let majorana_count = sites
        .checked_mul(2)
        .ok_or(FermionError::ModeIndexOverflow { mode: sites })?;
    let mut hamiltonian = QuadraticMajoranaHamiltonian::new(majorana_count);

    for site in 0..sites {
        let (a, b) = mode_indices(site)?;
        hamiltonian.add_bilinear(a, b, -0.5 * chemical_potential)?;
    }
    for site in 0..sites - 1 {
        let (a_left, b_left) = mode_indices(site)?;
        let (a_right, b_right) = mode_indices(site + 1)?;
        hamiltonian.add_bilinear(b_left, a_right, 0.5 * (hopping + pairing))?;
        hamiltonian.add_bilinear(a_left, b_right, 0.5 * (-hopping + pairing))?;
    }
    Ok(hamiltonian)
}

fn validate_real(value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(FermionError::NonFiniteCoefficient)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{annihilation, creation, number_operator};

    #[test]
    fn quadratic_form_is_hermitian() {
        let mut hamiltonian = QuadraticMajoranaHamiltonian::new(8);
        hamiltonian.set_constant(-0.25).unwrap();
        hamiltonian.add_bilinear(0, 7, 1.5).unwrap();
        hamiltonian.add_bilinear(5, 2, -0.75).unwrap();
        assert!(hamiltonian
            .to_operator()
            .unwrap()
            .is_hermitian(0.0)
            .unwrap());
        assert_eq!(hamiltonian.bilinear_coefficient(7, 0).unwrap(), -1.5);
    }

    #[test]
    fn kitaev_special_point_leaves_edge_majoranas_unpaired() {
        let chain = kitaev_chain(5, 1.25, 1.25, 0.0).unwrap();
        assert_eq!(chain.couplings().count(), 4);
        for site in 0..4 {
            assert_eq!(
                chain
                    .bilinear_coefficient(2 * site + 1, 2 * (site + 1))
                    .unwrap(),
                1.25
            );
        }
        assert!(chain
            .couplings()
            .all(|(left, right, _)| left != 0 && right != 9));
        assert!(chain.to_operator().unwrap().is_hermitian(0.0).unwrap());
    }

    #[test]
    fn kitaev_majorana_form_matches_complex_fermion_definition() {
        let (sites, hopping, pairing, chemical_potential) = (3, 0.7, -0.2, 0.35);
        let actual = kitaev_chain(sites, hopping, pairing, chemical_potential)
            .unwrap()
            .to_operator()
            .unwrap();
        let mut expected = FermionOperator::zero();
        for site in 0..sites {
            let centered_number = number_operator(site)
                .unwrap()
                .subtract(
                    &FermionOperator::identity()
                        .scaled(Complex64::new(0.5, 0.0))
                        .unwrap(),
                )
                .unwrap();
            expected = expected
                .add(
                    &centered_number
                        .scaled(Complex64::new(-chemical_potential, 0.0))
                        .unwrap(),
                )
                .unwrap();
        }
        for site in 0..sites - 1 {
            let hopping_term = creation(site)
                .unwrap()
                .multiply(&annihilation(site + 1).unwrap())
                .unwrap()
                .add(
                    &creation(site + 1)
                        .unwrap()
                        .multiply(&annihilation(site).unwrap())
                        .unwrap(),
                )
                .unwrap()
                .scaled(Complex64::new(-hopping, 0.0))
                .unwrap();
            expected = expected.add(&hopping_term).unwrap();

            let pairing_term = annihilation(site)
                .unwrap()
                .multiply(&annihilation(site + 1).unwrap())
                .unwrap()
                .add(
                    &creation(site + 1)
                        .unwrap()
                        .multiply(&creation(site).unwrap())
                        .unwrap(),
                )
                .unwrap()
                .scaled(Complex64::new(pairing, 0.0))
                .unwrap();
            expected = expected.add(&pairing_term).unwrap();
        }
        assert!(actual.approx_eq(&expected, 1.0e-14).unwrap());
    }

    #[test]
    fn invalid_quadratic_inputs_are_rejected() {
        let mut hamiltonian = QuadraticMajoranaHamiltonian::new(4);
        assert!(hamiltonian.add_bilinear(0, 0, 1.0).is_err());
        assert!(hamiltonian.add_bilinear(0, 4, 1.0).is_err());
        assert!(hamiltonian.add_bilinear(0, 1, f64::NAN).is_err());
        assert!(kitaev_chain(0, 1.0, 1.0, 0.0).is_err());
    }
}
