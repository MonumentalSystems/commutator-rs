//! Pure-f64 Cl⁺(6,0) elements, checked Spin(6) rotors, and proposals.

use std::sync::OnceLock;

use clifford_core::CliffordAlgebra;

use crate::error::{finite_scalar, finite_slice};
use crate::metropolis::{Proposal, SymmetricPairInteraction};
use crate::rng::RandomSource;
use crate::{LatticeError, Result};

/// Number of components in the even subalgebra Cl⁺(6,0).
pub const EVEN_COMPONENTS: usize = 32;
/// Number of grade-2 components in Cl(6,0).
pub const BIVECTOR_COMPONENTS: usize = 15;
/// Number of components in the full Cl(6,0) algebra.
pub const FULL_COMPONENTS: usize = 64;
/// Fixed tolerance for public checked rotor imports.
pub const DEFAULT_ROTOR_TOLERANCE: f64 = 1.0e-10;

/// Even-subalgebra index to full ShortLex Cl(6,0) index.
///
/// The order is scalar, 15 bivectors, 15 grade-4 blades, pseudoscalar.
pub const EVEN_TO_FULL: [usize; EVEN_COMPONENTS] = [
    0, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 42, 43, 44, 45, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 63,
];

const BIVECTOR_START: usize = 1;
const BIVECTOR_END: usize = BIVECTOR_START + BIVECTOR_COMPONENTS;
const MIN_NORM_SQUARED: f64 = 1.0e-30;

#[derive(Debug)]
struct EvenTable {
    cayley_out: [u8; EVEN_COMPONENTS * EVEN_COMPONENTS],
    cayley_sign: [f64; EVEN_COMPONENTS * EVEN_COMPONENTS],
    reverse_sign: [f64; EVEN_COMPONENTS],
    full_to_even: [u8; FULL_COMPONENTS],
}

fn even_table() -> &'static EvenTable {
    static TABLE: OnceLock<EvenTable> = OnceLock::new();
    TABLE.get_or_init(|| {
        let algebra = CliffordAlgebra::cl6();
        let mut full_to_even = [u8::MAX; FULL_COMPONENTS];
        let mut reverse_sign = [0.0; EVEN_COMPONENTS];
        for (even_index, full_index) in EVEN_TO_FULL.into_iter().enumerate() {
            assert_eq!(algebra.grades[full_index] % 2, 0);
            full_to_even[full_index] = even_index as u8;
            let grade = algebra.grades[full_index];
            reverse_sign[even_index] = if (grade * grade.saturating_sub(1) / 2) % 2 == 0 {
                1.0
            } else {
                -1.0
            };
        }

        let mut cayley_out = [0_u8; EVEN_COMPONENTS * EVEN_COMPONENTS];
        let mut cayley_sign = [0.0; EVEN_COMPONENTS * EVEN_COMPONENTS];
        for (left_even, left_full) in EVEN_TO_FULL.into_iter().enumerate() {
            for (right_even, right_full) in EVEN_TO_FULL.into_iter().enumerate() {
                let full_flat = left_full * FULL_COMPONENTS + right_full;
                let output_full = algebra.cayley_index[full_flat];
                let output_even = full_to_even[output_full];
                assert_ne!(output_even, u8::MAX);
                let even_flat = left_even * EVEN_COMPONENTS + right_even;
                cayley_out[even_flat] = output_even;
                cayley_sign[even_flat] = algebra.cayley_sign[full_flat] as f64;
            }
        }
        EvenTable {
            cayley_out,
            cayley_sign,
            reverse_sign,
            full_to_even,
        }
    })
}

/// Arbitrary even multivector in Cl⁺(6,0), stored as 32 f64 coefficients.
///
/// This type does not claim the unit-versor invariant. Use [`Spin6Rotor`] for
/// lattice sites and group composition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvenMultivector32 {
    coefficients: [f64; EVEN_COMPONENTS],
}

impl EvenMultivector32 {
    /// Returns the additive identity.
    pub const fn zero() -> Self {
        Self {
            coefficients: [0.0; EVEN_COMPONENTS],
        }
    }

    /// Returns the multiplicative identity.
    pub const fn one() -> Self {
        let mut coefficients = [0.0; EVEN_COMPONENTS];
        coefficients[0] = 1.0;
        Self { coefficients }
    }

    /// Constructs an even multivector after validating finite coefficients.
    pub fn from_coefficients(coefficients: [f64; EVEN_COMPONENTS]) -> Result<Self> {
        finite_slice("Cl(6,0) even coefficients", &coefficients)?;
        Ok(Self { coefficients })
    }

    /// Returns the scalar-first even-subalgebra coefficients.
    pub const fn coefficients(&self) -> &[f64; EVEN_COMPONENTS] {
        &self.coefficients
    }

    /// Returns the owned scalar-first coefficient array.
    pub const fn into_coefficients(self) -> [f64; EVEN_COMPONENTS] {
        self.coefficients
    }

    /// Returns the geometric product of two even multivectors.
    pub fn geometric_product(self, right: Self) -> Result<Self> {
        let table = even_table();
        let mut output = [0.0; EVEN_COMPONENTS];
        for left_index in 0..EVEN_COMPONENTS {
            let left_value = self.coefficients[left_index];
            if left_value == 0.0 {
                continue;
            }
            for right_index in 0..EVEN_COMPONENTS {
                let right_value = right.coefficients[right_index];
                if right_value == 0.0 {
                    continue;
                }
                let flat = left_index * EVEN_COMPONENTS + right_index;
                let output_index = table.cayley_out[flat] as usize;
                output[output_index] += left_value * right_value * table.cayley_sign[flat];
            }
        }
        Self::from_coefficients(output)
    }

    /// Returns the Clifford reverse.
    pub fn reversed(self) -> Self {
        let table = even_table();
        let mut output = self.coefficients;
        for (value, sign) in output.iter_mut().zip(table.reverse_sign) {
            *value *= sign;
        }
        Self {
            coefficients: output,
        }
    }

    /// Returns `⟨self reverse(right)⟩₀`.
    pub fn scalar_product(self, right: Self) -> Result<f64> {
        let product = self.geometric_product(right.reversed())?;
        Ok(product.coefficients[0])
    }

    /// Returns the positive coefficient norm squared for Cl⁺(6,0).
    pub fn norm_squared(self) -> Result<f64> {
        self.scalar_product(self)
    }

    /// Returns the fifteen grade-2 coefficients.
    pub fn grade2_components(self) -> [f64; BIVECTOR_COMPONENTS] {
        let mut output = [0.0; BIVECTOR_COMPONENTS];
        output.copy_from_slice(&self.coefficients[BIVECTOR_START..BIVECTOR_END]);
        output
    }

    /// Embeds the even element into the full 64-component ShortLex layout.
    pub fn to_full_64(self) -> [f64; FULL_COMPONENTS] {
        let mut full = [0.0; FULL_COMPONENTS];
        for (even_index, full_index) in EVEN_TO_FULL.into_iter().enumerate() {
            full[full_index] = self.coefficients[even_index];
        }
        full
    }

    /// Imports a full Cl(6,0) multivector if all odd-grade terms are exactly zero.
    pub fn try_from_full_64(full: [f64; FULL_COMPONENTS]) -> Result<Self> {
        finite_slice("full Cl(6,0) coefficients", &full)?;
        let table = even_table();
        let mut coefficients = [0.0; EVEN_COMPONENTS];
        for (full_index, value) in full.into_iter().enumerate() {
            let even_index = table.full_to_even[full_index];
            if even_index == u8::MAX {
                if value != 0.0 {
                    return Err(LatticeError::OddGradeComponent);
                }
            } else {
                coefficients[even_index as usize] = value;
            }
        }
        Self::from_coefficients(coefficients)
    }

    fn scaled(self, scale: f64) -> Result<Self> {
        finite_scalar("even multivector scale", scale)?;
        Self::from_coefficients(self.coefficients.map(|value| value * scale))
    }

    fn add_assign(&mut self, right: Self) -> Result<()> {
        for (left, right) in self.coefficients.iter_mut().zip(right.coefficients) {
            *left += right;
        }
        finite_slice("Cl(6,0) even sum", &self.coefficients)
    }

    fn max_abs(self) -> f64 {
        self.coefficients
            .into_iter()
            .map(f64::abs)
            .fold(0.0, f64::max)
    }
}

/// Checked unit rotor in Spin(6), backed by an f64 Cl⁺(6,0) element.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spin6Rotor(EvenMultivector32);

/// Storage-oriented compatibility name for [`Spin6Rotor`].
pub type Rotor32 = Spin6Rotor;

impl Spin6Rotor {
    /// Returns the identity rotor.
    pub const fn identity() -> Self {
        Self(EvenMultivector32::one())
    }

    /// Constructs a checked rotor from even coefficients.
    ///
    /// A nonzero scalar multiple of a valid rotor is normalized. General even
    /// multivectors whose product with their reverse has nonscalar residue are
    /// rejected.
    pub fn try_from_coefficients(coefficients: [f64; EVEN_COMPONENTS]) -> Result<Self> {
        Self::try_from_even_strict(EvenMultivector32::from_coefficients(coefficients)?)
    }

    /// Imports a checked rotor from the full 64-component ShortLex layout.
    pub fn try_from_full_64(full: [f64; FULL_COMPONENTS]) -> Result<Self> {
        Self::try_from_even_strict(EvenMultivector32::try_from_full_64(full)?)
    }

    /// Returns the scalar-first even-subalgebra coefficients.
    pub const fn coefficients(&self) -> &[f64; EVEN_COMPONENTS] {
        self.0.coefficients()
    }

    /// Embeds this rotor into the full 64-component ShortLex layout.
    pub fn to_full_64(self) -> [f64; FULL_COMPONENTS] {
        self.0.to_full_64()
    }

    /// Returns the grade-2 part of the rotor itself.
    ///
    /// This is not a general logarithm map; it is only a small-angle proxy for
    /// Lie-algebra coordinates.
    pub fn grade2_components(self) -> [f64; BIVECTOR_COMPONENTS] {
        self.0.grade2_components()
    }

    /// Returns the inverse rotor, equal to the Clifford reverse for unit rotors.
    pub fn reversed(self) -> Self {
        Self(self.0.reversed())
    }

    /// Composes two checked rotors.
    pub fn compose(self, right: Self) -> Result<Self> {
        Self::try_from_even_strict(self.0.geometric_product(right.0)?)
    }

    /// Returns the scalar similarity `⟨self reverse(right)⟩₀`.
    pub fn scalar_product(self, right: Self) -> Result<f64> {
        self.0.scalar_product(right.0)
    }

    /// Returns the scalar component of `R reverse(R)`.
    pub fn norm_squared(self) -> Result<f64> {
        self.0.norm_squared()
    }

    /// Returns the largest nonscalar component of `R reverse(R)`.
    pub fn versor_defect(self) -> Result<f64> {
        let product = self.0.geometric_product(self.0.reversed())?;
        Ok(product.coefficients[1..]
            .iter()
            .copied()
            .map(f64::abs)
            .fold(0.0, f64::max))
    }

    /// Computes `exp(B)` for a finite 15-component bivector.
    ///
    /// The sign and scale are literal: callers wanting the common geometric
    /// rotor convention `exp(-B/2)` must supply `-B/2`. Scaling and squaring
    /// with a converged Taylor series avoids the source implementation's
    /// unbounded fixed-order proposal approximation.
    pub fn exp_bivector(bivector: [f64; BIVECTOR_COMPONENTS]) -> Result<Self> {
        finite_slice("Cl(6,0) bivector", &bivector)?;
        let magnitude = bivector
            .iter()
            .fold(0.0_f64, |norm, value| norm.hypot(*value));
        finite_scalar("Cl(6,0) bivector magnitude", magnitude)?;
        if magnitude == 0.0 {
            return Ok(Self::identity());
        }

        let squarings = if magnitude <= 0.5 {
            0
        } else {
            (magnitude / 0.5).log2().ceil() as u32
        };
        if squarings > 60 {
            return Err(LatticeError::InvalidDomain(
                "Cl(6,0) bivector magnitude is too large",
            ));
        }
        let divisor = 2.0_f64.powi(squarings as i32);
        let mut input = EvenMultivector32::zero();
        input.coefficients[BIVECTOR_START..BIVECTOR_END]
            .copy_from_slice(&bivector.map(|value| value / divisor));

        let mut result = EvenMultivector32::one();
        let mut term = EvenMultivector32::one();
        for order in 1..=32_u32 {
            term = term.geometric_product(input)?.scaled(1.0 / order as f64)?;
            result.add_assign(term)?;
            if term.max_abs() <= 1.0e-17 {
                break;
            }
        }
        for _ in 0..squarings {
            result = result.geometric_product(result)?;
        }
        Self::try_from_even_strict_with_tolerance(result, 1.0e-9)
    }

    fn try_from_even_strict(element: EvenMultivector32) -> Result<Self> {
        Self::try_from_even_strict_with_tolerance(element, DEFAULT_ROTOR_TOLERANCE)
    }

    fn try_from_even_strict_with_tolerance(
        element: EvenMultivector32,
        tolerance: f64,
    ) -> Result<Self> {
        let rotor = Self::from_known_spin(element, tolerance)?;
        validate_spin_action(rotor.0, tolerance)?;
        Ok(rotor)
    }

    fn from_known_spin(element: EvenMultivector32, tolerance: f64) -> Result<Self> {
        let product = element.geometric_product(element.reversed())?;
        let norm_squared = product.coefficients[0];
        finite_scalar("rotor norm squared", norm_squared)?;
        if norm_squared <= MIN_NORM_SQUARED {
            return Err(LatticeError::ZeroNorm("Spin(6) rotor"));
        }
        let scale = norm_squared.abs().max(1.0);
        let defect = product.coefficients[1..]
            .iter()
            .copied()
            .map(f64::abs)
            .fold(0.0, f64::max);
        if defect > tolerance * scale {
            return Err(LatticeError::NotUnitRotor);
        }
        let normalized = element.scaled(norm_squared.sqrt().recip())?;
        let check = normalized.geometric_product(normalized.reversed())?;
        let scalar_error = (check.coefficients[0] - 1.0).abs();
        let residual = check.coefficients[1..]
            .iter()
            .copied()
            .map(f64::abs)
            .fold(0.0, f64::max);
        if scalar_error > tolerance * 4.0 || residual > tolerance * 4.0 {
            return Err(LatticeError::NotUnitRotor);
        }
        Ok(Self(normalized))
    }
}

fn validate_spin_action(element: EvenMultivector32, tolerance: f64) -> Result<()> {
    let algebra = CliffordAlgebra::cl6();
    let rotor = element.to_full_64();
    let reverse = element.reversed().to_full_64();
    let vector_indices = algebra
        .grades
        .iter()
        .enumerate()
        .filter_map(|(index, grade)| (*grade == 1).then_some(index))
        .collect::<Vec<_>>();
    debug_assert_eq!(vector_indices.len(), 6);

    let mut rotation = [[0.0; 6]; 6];
    for (input_column, input_index) in vector_indices.iter().copied().enumerate() {
        let transformed = algebra.geometric_product(
            &algebra.geometric_product(&rotor, &algebra.basis::<f64>(input_index)),
            &reverse,
        );
        finite_slice("rotor sandwich action", &transformed)?;
        for (output_index, value) in transformed.iter().copied().enumerate() {
            if algebra.grades[output_index] != 1 && value.abs() > tolerance * 8.0 {
                return Err(LatticeError::NotUnitRotor);
            }
        }
        for (output_row, output_index) in vector_indices.iter().copied().enumerate() {
            rotation[output_row][input_column] = transformed[output_index];
        }
    }

    for left in 0..6 {
        for right in 0..6 {
            let dot = (0..6)
                .map(|row| rotation[row][left] * rotation[row][right])
                .sum::<f64>();
            let expected = if left == right { 1.0 } else { 0.0 };
            if (dot - expected).abs() > tolerance * 32.0 {
                return Err(LatticeError::NotUnitRotor);
            }
        }
    }

    let determinant = determinant_6(rotation);
    if !determinant.is_finite() || (determinant - 1.0).abs() > tolerance * 64.0 {
        return Err(LatticeError::NotUnitRotor);
    }
    Ok(())
}

fn determinant_6(mut matrix: [[f64; 6]; 6]) -> f64 {
    let mut determinant = 1.0;
    for column in 0..6 {
        let pivot = (column..6)
            .max_by(|left, right| {
                matrix[*left][column]
                    .abs()
                    .total_cmp(&matrix[*right][column].abs())
            })
            .unwrap_or(column);
        if matrix[pivot][column] == 0.0 {
            return 0.0;
        }
        if pivot != column {
            matrix.swap(pivot, column);
            determinant = -determinant;
        }
        let pivot_value = matrix[column][column];
        determinant *= pivot_value;
        let pivot_row = matrix[column];
        for row in matrix.iter_mut().skip(column + 1) {
            let factor = row[column] / pivot_value;
            for (value, pivot_value) in row[(column + 1)..]
                .iter_mut()
                .zip(&pivot_row[(column + 1)..])
            {
                *value -= factor * pivot_value;
            }
        }
    }
    determinant
}

/// Full-rotor nearest-neighbor score `⟨Rᵢ reverse(Rⱼ)⟩₀`.
#[derive(Debug, Clone, Copy, Default)]
pub struct RotorScalarInteraction;

impl SymmetricPairInteraction<Spin6Rotor> for RotorScalarInteraction {
    fn score(&self, left: &Spin6Rotor, right: &Spin6Rotor) -> f64 {
        left.scalar_product(*right).unwrap_or(f64::NAN)
    }
}

/// Gaussian left-multiplication proposal over caller-supplied bivector directions.
///
/// Direction norms are preserved. This makes the proposal covariance explicit
/// and avoids silently treating the source CPU and GPU policies as equivalent.
#[derive(Debug, Clone, PartialEq)]
pub struct GaussianBivectorProposal {
    scale: f64,
    directions: Vec<[f64; BIVECTOR_COMPONENTS]>,
}

impl GaussianBivectorProposal {
    /// Constructs a proposal from one or more finite bivector directions.
    pub fn new(scale: f64, directions: Vec<[f64; BIVECTOR_COMPONENTS]>) -> Result<Self> {
        finite_scalar("proposal scale", scale)?;
        if scale < 0.0 {
            return Err(LatticeError::InvalidDomain("proposal scale"));
        }
        if directions.is_empty() {
            return Err(LatticeError::Empty("proposal directions"));
        }
        for direction in &directions {
            finite_slice("proposal direction", direction)?;
            if direction.iter().all(|value| *value == 0.0) {
                return Err(LatticeError::ZeroNorm("proposal direction"));
            }
        }
        Ok(Self { scale, directions })
    }

    /// Constructs an isotropic proposal in the canonical fifteen bivector coordinates.
    pub fn canonical(scale: f64) -> Result<Self> {
        let mut directions = Vec::with_capacity(BIVECTOR_COMPONENTS);
        for component in 0..BIVECTOR_COMPONENTS {
            let mut direction = [0.0; BIVECTOR_COMPONENTS];
            direction[component] = 1.0;
            directions.push(direction);
        }
        Self::new(scale, directions)
    }

    /// Returns the Gaussian coefficient scale.
    pub const fn scale(&self) -> f64 {
        self.scale
    }

    /// Returns the ordered proposal directions.
    pub fn directions(&self) -> &[[f64; BIVECTOR_COMPONENTS]] {
        &self.directions
    }
}

impl<R: RandomSource> Proposal<Spin6Rotor, R> for GaussianBivectorProposal {
    fn propose(&self, current: &Spin6Rotor, _: usize, random: &mut R) -> Result<Spin6Rotor> {
        let mut bivector = [0.0; BIVECTOR_COMPONENTS];
        for direction in &self.directions {
            let coefficient = self.scale * random.standard_normal_f64();
            for (value, direction_value) in bivector.iter_mut().zip(direction) {
                *value += coefficient * direction_value;
            }
        }
        Spin6Rotor::exp_bivector(bivector)?.compose(*current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(left: f64, right: f64, tolerance: f64) -> bool {
        (left - right).abs() <= tolerance
    }

    fn full_exp_reference(bivector: [f64; BIVECTOR_COMPONENTS]) -> Vec<f64> {
        let algebra = CliffordAlgebra::cl6();
        let mut input = algebra.zero::<f64>();
        for (component, value) in bivector.into_iter().enumerate() {
            input[EVEN_TO_FULL[BIVECTOR_START + component]] = value / 16.0;
        }
        let mut result = algebra.scalar(1.0_f64);
        let mut term = algebra.scalar(1.0_f64);
        for order in 1..=64 {
            term = algebra.geometric_product(&term, &input);
            for value in &mut term {
                *value /= order as f64;
            }
            for (sum, value) in result.iter_mut().zip(&term) {
                *sum += value;
            }
        }
        for _ in 0..4 {
            result = algebra.geometric_product(&result, &result);
        }
        result
    }

    #[test]
    fn even_layout_is_the_complete_core_even_sequence() {
        let algebra = CliffordAlgebra::cl6();
        let expected = algebra
            .grades
            .iter()
            .enumerate()
            .filter_map(|(index, grade)| (grade % 2 == 0).then_some(index))
            .collect::<Vec<_>>();
        assert_eq!(EVEN_TO_FULL.as_slice(), expected.as_slice());
    }

    #[test]
    fn all_even_basis_products_match_clifford_core() {
        let algebra = CliffordAlgebra::cl6();
        for left in 0..EVEN_COMPONENTS {
            for right in 0..EVEN_COMPONENTS {
                let mut left_coefficients = [0.0; EVEN_COMPONENTS];
                let mut right_coefficients = [0.0; EVEN_COMPONENTS];
                left_coefficients[left] = 1.0;
                right_coefficients[right] = 1.0;
                let actual = EvenMultivector32::from_coefficients(left_coefficients)
                    .unwrap()
                    .geometric_product(
                        EvenMultivector32::from_coefficients(right_coefficients).unwrap(),
                    )
                    .unwrap()
                    .to_full_64();
                let mut left_full = [0.0; FULL_COMPONENTS];
                let mut right_full = [0.0; FULL_COMPONENTS];
                left_full[EVEN_TO_FULL[left]] = 1.0;
                right_full[EVEN_TO_FULL[right]] = 1.0;
                let expected = algebra.geometric_product(&left_full, &right_full);
                assert_eq!(actual.as_slice(), expected.as_slice());
            }
        }
    }

    #[test]
    fn dense_product_scalar_product_and_reverse_match_core() {
        let left = EvenMultivector32::from_coefficients(std::array::from_fn(|index| {
            ((index * 17 + 3) as f64).sin()
        }))
        .unwrap();
        let right = EvenMultivector32::from_coefficients(std::array::from_fn(|index| {
            ((index * 11 + 5) as f64).cos()
        }))
        .unwrap();
        let algebra = CliffordAlgebra::cl6();
        let expected = algebra.geometric_product(&left.to_full_64(), &right.to_full_64());
        let actual = left.geometric_product(right).unwrap().to_full_64();
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!(close(actual, expected, 1.0e-12));
        }

        let scalar_expected =
            algebra.geometric_product(&left.to_full_64(), &algebra.reverse(&right.to_full_64()))[0];
        assert!(close(
            left.scalar_product(right).unwrap(),
            scalar_expected,
            1.0e-12
        ));
        assert_eq!(left.reversed().reversed(), left);
        let reversed_product = left.geometric_product(right).unwrap().reversed();
        let product_of_reverses = right.reversed().geometric_product(left.reversed()).unwrap();
        for (left, right) in reversed_product
            .coefficients()
            .iter()
            .zip(product_of_reverses.coefficients())
        {
            assert!(close(*left, *right, 2.0e-14));
        }
    }

    #[test]
    fn full_layout_round_trip_and_odd_rejection() {
        let mut coefficients = [0.0; EVEN_COMPONENTS];
        for (index, value) in coefficients.iter_mut().enumerate() {
            *value = index as f64 - 8.0;
        }
        let even = EvenMultivector32::from_coefficients(coefficients).unwrap();
        assert_eq!(
            EvenMultivector32::try_from_full_64(even.to_full_64()).unwrap(),
            even
        );
        let mut invalid = even.to_full_64();
        invalid[1] = 1.0;
        assert!(matches!(
            EvenMultivector32::try_from_full_64(invalid),
            Err(LatticeError::OddGradeComponent)
        ));
    }

    #[test]
    fn simple_plane_exponential_matches_closed_form() {
        let angle = 0.37;
        let mut bivector = [0.0; BIVECTOR_COMPONENTS];
        bivector[0] = angle;
        let rotor = Spin6Rotor::exp_bivector(bivector).unwrap();
        assert!(close(rotor.coefficients()[0], angle.cos(), 1.0e-13));
        assert!(close(rotor.coefficients()[1], angle.sin(), 1.0e-13));
        assert!(close(rotor.norm_squared().unwrap(), 1.0, 1.0e-12));
        assert!(rotor.versor_defect().unwrap() < 1.0e-12);
    }

    #[test]
    fn commuting_planes_generate_grade_four_term() {
        let first = 0.2;
        let second = -0.3;
        let mut bivector = [0.0; BIVECTOR_COMPONENTS];
        bivector[0] = first; // e01
        bivector[5] = second; // e23
        let rotor = Spin6Rotor::exp_bivector(bivector).unwrap();
        assert!(close(
            rotor.coefficients()[0],
            first.cos() * second.cos(),
            1.0e-13
        ));
        assert!(close(
            rotor.coefficients()[16],
            first.sin() * second.sin(),
            1.0e-13
        ));
    }

    #[test]
    fn dense_bivector_exponential_matches_full_reference_and_inverts() {
        let bivector = std::array::from_fn(|index| {
            let signed = index as f64 - 7.0;
            0.023 * signed + 0.007 * (index as f64).sin()
        });
        let rotor = Spin6Rotor::exp_bivector(bivector).unwrap();
        let reference = full_exp_reference(bivector);
        for (actual, expected) in rotor.to_full_64().into_iter().zip(reference) {
            assert!(close(actual, expected, 2.0e-12));
        }

        let inverse = Spin6Rotor::exp_bivector(bivector.map(|value| -value)).unwrap();
        let identity = rotor.compose(inverse).unwrap();
        assert!(close(identity.coefficients()[0], 1.0, 2.0e-12));
        assert!(identity.coefficients()[1..]
            .iter()
            .all(|value| value.abs() < 2.0e-12));
    }

    #[test]
    fn tiny_nonzero_bivector_does_not_collapse_to_identity() {
        let mut bivector = [0.0; BIVECTOR_COMPONENTS];
        bivector[3] = 1.0e-200;
        let rotor = Spin6Rotor::exp_bivector(bivector).unwrap();
        assert_eq!(rotor.coefficients()[4], 1.0e-200);
        assert_eq!(rotor.coefficients()[0], 1.0);
    }

    #[test]
    fn checked_rotor_rejects_zero_and_general_even_elements() {
        assert!(matches!(
            Spin6Rotor::try_from_coefficients([0.0; EVEN_COMPONENTS]),
            Err(LatticeError::ZeroNorm(_))
        ));
        let mut invalid = [0.0; EVEN_COMPONENTS];
        invalid[0] = 1.0;
        invalid[16] = 0.5;
        assert!(matches!(
            Spin6Rotor::try_from_coefficients(invalid),
            Err(LatticeError::NotUnitRotor)
        ));
    }

    #[test]
    fn scalar_pseudoscalar_unit_is_not_a_spin_rotor() {
        let angle = 0.37_f64;
        let mut coefficients = [0.0; EVEN_COMPONENTS];
        coefficients[0] = angle.cos();
        coefficients[EVEN_COMPONENTS - 1] = angle.sin();
        let even = EvenMultivector32::from_coefficients(coefficients).unwrap();
        assert!(close(even.norm_squared().unwrap(), 1.0, 1.0e-12));
        assert!(matches!(
            Spin6Rotor::try_from_coefficients(coefficients),
            Err(LatticeError::NotUnitRotor)
        ));
    }

    #[test]
    fn composition_revalidates_near_tolerance_inputs() {
        let angle = 3.0e-10_f64;
        let mut coefficients = [0.0; EVEN_COMPONENTS];
        coefficients[0] = angle.cos();
        coefficients[EVEN_COMPONENTS - 1] = angle.sin();
        let near_boundary = Spin6Rotor::try_from_coefficients(coefficients).unwrap();
        assert!(matches!(
            near_boundary.compose(near_boundary),
            Err(LatticeError::NotUnitRotor)
        ));
    }

    #[test]
    fn proposal_validation_is_explicit() {
        assert!(GaussianBivectorProposal::new(0.1, Vec::new()).is_err());
        assert!(GaussianBivectorProposal::new(-0.1, vec![[1.0; 15]]).is_err());
        assert!(GaussianBivectorProposal::new(0.1, vec![[0.0; 15]]).is_err());
    }
}
