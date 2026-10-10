//! Checked finite-Fock-space reference kernels for quantum photonics.
//!
//! This crate starts at the quantum-state boundary: classical field solvers and
//! nonlinear-material models may supply modes and parameters, while
//! [`StateVector`] represents photons in those modes. Basis ordering and cutoff
//! behavior are explicit, and operations never silently truncate probability.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use core::fmt;

pub use num_complex::Complex64;

const MAX_STATE_DIMENSION: usize = 1 << 20;
const MAX_OCCUPATION: usize = 64;

/// Errors returned by checked quantum-light kernels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum QuantumLightError {
    /// A Fock space must contain at least one mode.
    ZeroModes,
    /// The requested local occupation cutoff is unsupported.
    OccupationLimit {
        /// Requested inclusive occupation limit.
        requested: usize,
        /// Largest supported inclusive occupation limit.
        maximum: usize,
    },
    /// The tensor-product state would be too large for this reference crate.
    StateSpaceTooLarge,
    /// An occupation vector has the wrong number of modes.
    OccupationShape {
        /// Required number of modes.
        expected: usize,
        /// Supplied number of occupations.
        actual: usize,
    },
    /// An occupation exceeds the configured inclusive maximum.
    OccupationOutOfRange {
        /// Mode containing the invalid occupation.
        mode: usize,
        /// Supplied occupation.
        value: usize,
        /// Inclusive maximum occupation.
        maximum: usize,
    },
    /// A mode index is outside the configured space.
    ModeOutOfRange {
        /// Supplied mode index.
        mode: usize,
        /// Number of modes in the space.
        modes: usize,
    },
    /// Two-mode operations require distinct mode indices.
    DuplicateModes,
    /// An amplitude vector has the wrong dimension.
    AmplitudeShape {
        /// Required dimension.
        expected: usize,
        /// Supplied dimension or first invalid index plus one.
        actual: usize,
    },
    /// Input contains NaN or infinity.
    NonFiniteInput,
    /// A zero vector cannot be normalized or used for normalized observables.
    ZeroNorm,
    /// The configured cutoff cannot represent every output of an operation.
    CutoffTooSmall {
        /// Minimum occupation required to represent the result.
        required: usize,
        /// Configured inclusive maximum occupation.
        maximum: usize,
    },
    /// A squeezing parameter must satisfy `abs(lambda) < 1`.
    InvalidSqueezing,
}

impl fmt::Display for QuantumLightError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for QuantumLightError {}

/// Result type for checked quantum-light operations.
pub type Result<T> = core::result::Result<T, QuantumLightError>;

/// A bounded tensor-product Fock basis.
///
/// Mode zero is the least-significant mixed-radix digit. The maximum
/// occupation is inclusive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FockSpace {
    modes: usize,
    max_occupation: usize,
    dimension: usize,
}

impl FockSpace {
    /// Constructs a checked space with the same occupation cutoff on each mode.
    pub fn try_new(modes: usize, max_occupation: usize) -> Result<Self> {
        if modes == 0 {
            return Err(QuantumLightError::ZeroModes);
        }
        if max_occupation > MAX_OCCUPATION {
            return Err(QuantumLightError::OccupationLimit {
                requested: max_occupation,
                maximum: MAX_OCCUPATION,
            });
        }
        let local_dimension = max_occupation + 1;
        let dimension = local_dimension
            .checked_pow(u32::try_from(modes).map_err(|_| QuantumLightError::StateSpaceTooLarge)?)
            .ok_or(QuantumLightError::StateSpaceTooLarge)?;
        if dimension > MAX_STATE_DIMENSION {
            return Err(QuantumLightError::StateSpaceTooLarge);
        }
        Ok(Self {
            modes,
            max_occupation,
            dimension,
        })
    }

    /// Returns the number of optical modes.
    #[must_use]
    pub const fn modes(self) -> usize {
        self.modes
    }

    /// Returns the inclusive maximum occupation of each mode.
    #[must_use]
    pub const fn max_occupation(self) -> usize {
        self.max_occupation
    }

    /// Returns the total Hilbert-space dimension.
    #[must_use]
    pub const fn dimension(self) -> usize {
        self.dimension
    }

    /// Converts occupations to the stable basis index.
    pub fn basis_index(self, occupations: &[usize]) -> Result<usize> {
        if occupations.len() != self.modes {
            return Err(QuantumLightError::OccupationShape {
                expected: self.modes,
                actual: occupations.len(),
            });
        }
        let radix = self.max_occupation + 1;
        let mut stride = 1;
        let mut index = 0;
        for (mode, &occupation) in occupations.iter().enumerate() {
            if occupation > self.max_occupation {
                return Err(QuantumLightError::OccupationOutOfRange {
                    mode,
                    value: occupation,
                    maximum: self.max_occupation,
                });
            }
            index += occupation * stride;
            stride *= radix;
        }
        Ok(index)
    }

    /// Converts a basis index to occupations.
    pub fn occupations(self, mut index: usize) -> Result<Vec<usize>> {
        if index >= self.dimension {
            return Err(QuantumLightError::AmplitudeShape {
                expected: self.dimension,
                actual: index.saturating_add(1),
            });
        }
        let radix = self.max_occupation + 1;
        let mut occupations = Vec::with_capacity(self.modes);
        for _ in 0..self.modes {
            occupations.push(index % radix);
            index /= radix;
        }
        Ok(occupations)
    }

    fn check_mode(self, mode: usize) -> Result<()> {
        if mode >= self.modes {
            Err(QuantumLightError::ModeOutOfRange {
                mode,
                modes: self.modes,
            })
        } else {
            Ok(())
        }
    }
}

/// A complex vector in a bounded multimode Fock basis.
///
/// Operator application may produce an unnormalized or zero vector. Methods
/// that interpret amplitudes as probabilities normalize by the stored norm and
/// reject the zero vector.
#[derive(Clone, Debug, PartialEq)]
pub struct StateVector {
    space: FockSpace,
    amplitudes: Vec<Complex64>,
}

impl StateVector {
    /// Constructs a checked state vector without changing its normalization.
    pub fn try_from_amplitudes(space: FockSpace, amplitudes: Vec<Complex64>) -> Result<Self> {
        if amplitudes.len() != space.dimension {
            return Err(QuantumLightError::AmplitudeShape {
                expected: space.dimension,
                actual: amplitudes.len(),
            });
        }
        if amplitudes
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(QuantumLightError::NonFiniteInput);
        }
        Ok(Self { space, amplitudes })
    }

    /// Constructs one occupation-number basis state.
    pub fn basis(space: FockSpace, occupations: &[usize]) -> Result<Self> {
        let index = space.basis_index(occupations)?;
        let mut amplitudes = vec![Complex64::new(0.0, 0.0); space.dimension];
        amplitudes[index] = Complex64::new(1.0, 0.0);
        Ok(Self { space, amplitudes })
    }

    /// Constructs the multimode vacuum.
    #[must_use]
    pub fn vacuum(space: FockSpace) -> Self {
        let mut amplitudes = vec![Complex64::new(0.0, 0.0); space.dimension];
        amplitudes[0] = Complex64::new(1.0, 0.0);
        Self { space, amplitudes }
    }

    /// Constructs a normalized, cutoff two-mode squeezed-vacuum state.
    pub fn two_mode_squeezed(
        space: FockSpace,
        mode_a: usize,
        mode_b: usize,
        lambda: f64,
    ) -> Result<Self> {
        validate_two_modes(space, mode_a, mode_b)?;
        if !lambda.is_finite() {
            return Err(QuantumLightError::NonFiniteInput);
        }
        if lambda.abs() >= 1.0 {
            return Err(QuantumLightError::InvalidSqueezing);
        }
        let mut amplitudes = vec![Complex64::new(0.0, 0.0); space.dimension];
        let mut occupations = vec![0; space.modes];
        let mut coefficient = 1.0;
        for occupation in 0..=space.max_occupation {
            occupations[mode_a] = occupation;
            occupations[mode_b] = occupation;
            amplitudes[space.basis_index(&occupations)?] = Complex64::new(coefficient, 0.0);
            coefficient *= lambda;
        }
        Self::try_from_amplitudes(space, amplitudes)?.normalized()
    }

    /// Returns this vector's Fock space.
    #[must_use]
    pub const fn space(&self) -> FockSpace {
        self.space
    }

    /// Returns all amplitudes in stable basis order.
    #[must_use]
    pub fn amplitudes(&self) -> &[Complex64] {
        &self.amplitudes
    }

    /// Returns the squared vector norm.
    #[must_use]
    pub fn norm_squared(&self) -> f64 {
        self.amplitudes.iter().map(|value| value.norm_sqr()).sum()
    }

    /// Returns a normalized copy, rejecting the zero vector.
    pub fn normalized(&self) -> Result<Self> {
        let norm_squared = self.norm_squared();
        if !norm_squared.is_finite() {
            return Err(QuantumLightError::NonFiniteInput);
        }
        if norm_squared == 0.0 {
            return Err(QuantumLightError::ZeroNorm);
        }
        let inverse_norm = norm_squared.sqrt().recip();
        let amplitudes = self
            .amplitudes
            .iter()
            .map(|value| *value * inverse_norm)
            .collect();
        Ok(Self {
            space: self.space,
            amplitudes,
        })
    }

    /// Returns the normalized probability of an occupation vector.
    pub fn probability(&self, occupations: &[usize]) -> Result<f64> {
        let norm = self.checked_norm()?;
        Ok(self.amplitudes[self.space.basis_index(occupations)?].norm_sqr() / norm)
    }

    /// Applies a bosonic annihilation operator to one mode.
    pub fn annihilate(&self, mode: usize) -> Result<Self> {
        self.space.check_mode(mode)?;
        let mut output = vec![Complex64::new(0.0, 0.0); self.space.dimension];
        for (index, &amplitude) in self.amplitudes.iter().enumerate() {
            let mut occupations = self.space.occupations(index)?;
            let occupation = occupations[mode];
            if occupation > 0 {
                occupations[mode] -= 1;
                let destination = self.space.basis_index(&occupations)?;
                output[destination] += amplitude * (occupation as f64).sqrt();
            }
        }
        Self::try_from_amplitudes(self.space, output)
    }

    /// Applies a bosonic creation operator to one mode.
    ///
    /// Returns [`QuantumLightError::CutoffTooSmall`] if a nonzero component at
    /// the top occupation would be truncated.
    pub fn create(&self, mode: usize) -> Result<Self> {
        self.space.check_mode(mode)?;
        if self.creation_would_truncate(mode)? {
            return Err(QuantumLightError::CutoffTooSmall {
                required: self.space.max_occupation + 1,
                maximum: self.space.max_occupation,
            });
        }
        let mut output = vec![Complex64::new(0.0, 0.0); self.space.dimension];
        for (index, &amplitude) in self.amplitudes.iter().enumerate() {
            let mut occupations = self.space.occupations(index)?;
            let occupation = occupations[mode];
            if occupation < self.space.max_occupation {
                occupations[mode] += 1;
                let destination = self.space.basis_index(&occupations)?;
                output[destination] += amplitude * ((occupation + 1) as f64).sqrt();
            }
        }
        Self::try_from_amplitudes(self.space, output)
    }

    /// Reports whether creation on a mode would discard nonzero amplitude.
    pub fn creation_would_truncate(&self, mode: usize) -> Result<bool> {
        self.space.check_mode(mode)?;
        for (index, amplitude) in self.amplitudes.iter().enumerate() {
            if amplitude.norm_sqr() != 0.0
                && self.space.occupations(index)?[mode] == self.space.max_occupation
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Applies `exp(i phase * n)` to one mode.
    pub fn phase_shift(&self, mode: usize, phase: f64) -> Result<Self> {
        self.space.check_mode(mode)?;
        if !phase.is_finite() {
            return Err(QuantumLightError::NonFiniteInput);
        }
        let mut output = self.clone();
        for (index, amplitude) in output.amplitudes.iter_mut().enumerate() {
            let occupation = self.space.occupations(index)?[mode];
            let angle = phase * occupation as f64;
            *amplitude *= Complex64::new(angle.cos(), angle.sin());
        }
        Ok(output)
    }

    /// Applies an exact lossless real two-mode beam splitter.
    ///
    /// The creation operators transform as
    /// `a† -> cos(theta) a† + sin(theta) b†` and
    /// `b† -> -sin(theta) a† + cos(theta) b†`.
    pub fn beam_splitter(&self, mode_a: usize, mode_b: usize, theta: f64) -> Result<Self> {
        validate_two_modes(self.space, mode_a, mode_b)?;
        if !theta.is_finite() {
            return Err(QuantumLightError::NonFiniteInput);
        }
        for (index, amplitude) in self.amplitudes.iter().enumerate() {
            if amplitude.norm_sqr() == 0.0 {
                continue;
            }
            let occupations = self.space.occupations(index)?;
            let required = occupations[mode_a] + occupations[mode_b];
            if required > self.space.max_occupation {
                return Err(QuantumLightError::CutoffTooSmall {
                    required,
                    maximum: self.space.max_occupation,
                });
            }
        }

        let cosine = theta.cos();
        let sine = theta.sin();
        let mut output = vec![Complex64::new(0.0, 0.0); self.space.dimension];
        for (index, &amplitude) in self.amplitudes.iter().enumerate() {
            if amplitude.norm_sqr() == 0.0 {
                continue;
            }
            let mut occupations = self.space.occupations(index)?;
            let n = occupations[mode_a];
            let m = occupations[mode_b];
            let total = n + m;
            for from_a_to_a in 0..=n {
                for from_b_to_a in 0..=m {
                    let output_a = from_a_to_a + from_b_to_a;
                    let output_b = total - output_a;
                    occupations[mode_a] = output_a;
                    occupations[mode_b] = output_b;
                    let destination = self.space.basis_index(&occupations)?;
                    let coefficient =
                        beam_splitter_coefficient(n, m, from_a_to_a, from_b_to_a, cosine, sine);
                    output[destination] += amplitude * coefficient;
                }
            }
        }
        Self::try_from_amplitudes(self.space, output)
    }

    /// Returns the normalized photon-number distribution of one mode.
    pub fn photon_distribution(&self, mode: usize) -> Result<Vec<f64>> {
        self.space.check_mode(mode)?;
        let norm = self.checked_norm()?;
        let mut distribution = vec![0.0; self.space.max_occupation + 1];
        for (index, amplitude) in self.amplitudes.iter().enumerate() {
            let occupation = self.space.occupations(index)?[mode];
            distribution[occupation] += amplitude.norm_sqr() / norm;
        }
        Ok(distribution)
    }

    /// Returns the normalized mean photon number of one mode.
    pub fn mean_photon_number(&self, mode: usize) -> Result<f64> {
        let distribution = self.photon_distribution(mode)?;
        Ok(distribution
            .iter()
            .enumerate()
            .map(|(occupation, probability)| occupation as f64 * probability)
            .sum())
    }

    /// Returns `g²(0) = <n(n-1)> / <n>²` for one mode.
    ///
    /// Vacuum and other states with zero mean photon number return `None`.
    pub fn second_order_coherence(&self, mode: usize) -> Result<Option<f64>> {
        let distribution = self.photon_distribution(mode)?;
        let mean: f64 = distribution
            .iter()
            .enumerate()
            .map(|(n, probability)| n as f64 * probability)
            .sum();
        if mean == 0.0 {
            return Ok(None);
        }
        let factorial_moment: f64 = distribution
            .iter()
            .enumerate()
            .map(|(n, probability)| (n * n.saturating_sub(1)) as f64 * probability)
            .sum();
        Ok(Some(factorial_moment / (mean * mean)))
    }

    /// Returns the normalized coincidence moment `<n_a n_b>`.
    pub fn coincidence(&self, mode_a: usize, mode_b: usize) -> Result<f64> {
        validate_two_modes(self.space, mode_a, mode_b)?;
        let norm = self.checked_norm()?;
        let mut value = 0.0;
        for (index, amplitude) in self.amplitudes.iter().enumerate() {
            let occupations = self.space.occupations(index)?;
            value += occupations[mode_a] as f64 * occupations[mode_b] as f64 * amplitude.norm_sqr()
                / norm;
        }
        Ok(value)
    }

    /// Forms the pure-state density matrix `|psi><psi|` after normalization.
    pub fn density_matrix(&self) -> Result<DensityMatrix> {
        let normalized = self.normalized()?;
        let dimension = self.space.dimension;
        let mut elements = vec![Complex64::new(0.0, 0.0); dimension * dimension];
        for row in 0..dimension {
            for column in 0..dimension {
                elements[row * dimension + column] =
                    normalized.amplitudes[row] * normalized.amplitudes[column].conj();
            }
        }
        Ok(DensityMatrix {
            space: self.space,
            dimension,
            elements,
        })
    }

    fn checked_norm(&self) -> Result<f64> {
        let norm = self.norm_squared();
        if !norm.is_finite() {
            Err(QuantumLightError::NonFiniteInput)
        } else if norm == 0.0 {
            Err(QuantumLightError::ZeroNorm)
        } else {
            Ok(norm)
        }
    }
}

/// A density matrix associated with a [`FockSpace`].
#[derive(Clone, Debug, PartialEq)]
pub struct DensityMatrix {
    space: FockSpace,
    dimension: usize,
    elements: Vec<Complex64>,
}

impl DensityMatrix {
    /// Returns a matrix element in row-major order.
    pub fn element(&self, row: usize, column: usize) -> Result<Complex64> {
        if row >= self.dimension || column >= self.dimension {
            return Err(QuantumLightError::AmplitudeShape {
                expected: self.dimension,
                actual: row.max(column).saturating_add(1),
            });
        }
        Ok(self.elements[row * self.dimension + column])
    }

    /// Returns the real trace. Pure-state matrices produced by this crate have
    /// trace one to floating-point precision.
    #[must_use]
    pub fn trace(&self) -> f64 {
        (0..self.dimension)
            .map(|index| self.elements[index * self.dimension + index].re)
            .sum()
    }

    /// Returns `Tr(rho²)` for this Hermitian matrix.
    #[must_use]
    pub fn purity(&self) -> f64 {
        self.elements.iter().map(|value| value.norm_sqr()).sum()
    }

    /// Traces out every mode except one and returns its local density matrix.
    pub fn reduced_mode(&self, mode: usize) -> Result<ReducedDensityMatrix> {
        self.space.check_mode(mode)?;
        let local_dimension = self.space.max_occupation + 1;
        let mut elements = vec![Complex64::new(0.0, 0.0); local_dimension * local_dimension];
        for row in 0..self.dimension {
            let mut occupations = self.space.occupations(row)?;
            let local_row = occupations[mode];
            for local_column in 0..local_dimension {
                occupations[mode] = local_column;
                let column = self.space.basis_index(&occupations)?;
                elements[local_row * local_dimension + local_column] +=
                    self.elements[row * self.dimension + column];
            }
        }
        Ok(ReducedDensityMatrix {
            dimension: local_dimension,
            elements,
        })
    }
}

/// A one-mode reduced density matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct ReducedDensityMatrix {
    dimension: usize,
    elements: Vec<Complex64>,
}

impl ReducedDensityMatrix {
    /// Returns the local Hilbert-space dimension.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns a checked matrix element.
    pub fn element(&self, row: usize, column: usize) -> Result<Complex64> {
        if row >= self.dimension || column >= self.dimension {
            return Err(QuantumLightError::AmplitudeShape {
                expected: self.dimension,
                actual: row.max(column).saturating_add(1),
            });
        }
        Ok(self.elements[row * self.dimension + column])
    }

    /// Returns the real trace.
    #[must_use]
    pub fn trace(&self) -> f64 {
        (0..self.dimension)
            .map(|index| self.elements[index * self.dimension + index].re)
            .sum()
    }

    /// Returns `Tr(rho²)`.
    #[must_use]
    pub fn purity(&self) -> f64 {
        self.elements.iter().map(|value| value.norm_sqr()).sum()
    }
}

fn validate_two_modes(space: FockSpace, mode_a: usize, mode_b: usize) -> Result<()> {
    space.check_mode(mode_a)?;
    space.check_mode(mode_b)?;
    if mode_a == mode_b {
        Err(QuantumLightError::DuplicateModes)
    } else {
        Ok(())
    }
}

fn ln_factorial(value: usize) -> f64 {
    (2..=value).map(|factor| (factor as f64).ln()).sum()
}

fn ln_binomial(n: usize, k: usize) -> f64 {
    ln_factorial(n) - ln_factorial(k) - ln_factorial(n - k)
}

fn beam_splitter_coefficient(
    n: usize,
    m: usize,
    from_a_to_a: usize,
    from_b_to_a: usize,
    cosine: f64,
    sine: f64,
) -> f64 {
    let output_a = from_a_to_a + from_b_to_a;
    let output_b = n + m - output_a;
    let expansion_log = ln_binomial(n, from_a_to_a) + ln_binomial(m, from_b_to_a);
    let normalization_log =
        ln_factorial(output_a) + ln_factorial(output_b) - ln_factorial(n) - ln_factorial(m);
    let combinatorial = (expansion_log + 0.5 * normalization_log).exp();
    let cosine_power = from_a_to_a + m - from_b_to_a;
    let sine_power = n - from_a_to_a + from_b_to_a;
    let sign = if from_b_to_a % 2 == 0 { 1.0 } else { -1.0 };
    sign * combinatorial * cosine.powi(cosine_power as i32) * sine.powi(sine_power as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-11,
            "{actual} != {expected}"
        );
    }

    #[test]
    fn basis_order_round_trips() {
        let space = FockSpace::try_new(3, 2).unwrap();
        assert_eq!(space.dimension(), 27);
        assert_eq!(space.basis_index(&[2, 1, 0]).unwrap(), 5);
        for index in 0..space.dimension() {
            let occupations = space.occupations(index).unwrap();
            assert_eq!(space.basis_index(&occupations).unwrap(), index);
        }
    }

    #[test]
    fn ladder_operators_have_bosonic_weights() {
        let space = FockSpace::try_new(1, 3).unwrap();
        let state = StateVector::basis(space, &[2]).unwrap();
        let lowered = state.annihilate(0).unwrap();
        close(lowered.amplitudes()[1].re, 2.0_f64.sqrt());
        let restored = lowered.create(0).unwrap();
        close(restored.amplitudes()[2].re, 2.0);
        assert!(!state.creation_would_truncate(0).unwrap());
        assert!(StateVector::basis(space, &[3])
            .unwrap()
            .creation_would_truncate(0)
            .unwrap());
    }

    #[test]
    fn hong_ou_mandel_interference_suppresses_coincidences() {
        let space = FockSpace::try_new(2, 2).unwrap();
        let input = StateVector::basis(space, &[1, 1]).unwrap();
        let output = input
            .beam_splitter(0, 1, core::f64::consts::FRAC_PI_4)
            .unwrap();
        close(output.norm_squared(), 1.0);
        close(output.probability(&[1, 1]).unwrap(), 0.0);
        close(output.probability(&[2, 0]).unwrap(), 0.5);
        close(output.probability(&[0, 2]).unwrap(), 0.5);
        close(output.coincidence(0, 1).unwrap(), 0.0);
    }

    #[test]
    fn beam_splitter_rejects_unrepresentable_output() {
        let space = FockSpace::try_new(2, 1).unwrap();
        let input = StateVector::basis(space, &[1, 1]).unwrap();
        assert_eq!(
            input.beam_splitter(0, 1, 0.2),
            Err(QuantumLightError::CutoffTooSmall {
                required: 2,
                maximum: 1,
            })
        );
    }

    #[test]
    fn number_state_statistics_are_exact() {
        let space = FockSpace::try_new(2, 4).unwrap();
        let state = StateVector::basis(space, &[2, 3]).unwrap();
        close(state.mean_photon_number(0).unwrap(), 2.0);
        close(state.second_order_coherence(0).unwrap().unwrap(), 0.5);
        close(state.coincidence(0, 1).unwrap(), 6.0);
        assert_eq!(
            StateVector::vacuum(space)
                .second_order_coherence(0)
                .unwrap(),
            None
        );
    }

    #[test]
    fn squeezed_pair_has_thermal_marginal_and_entanglement() {
        let space = FockSpace::try_new(2, 4).unwrap();
        let state = StateVector::two_mode_squeezed(space, 0, 1, 0.4).unwrap();
        close(state.norm_squared(), 1.0);
        let reduced = state.density_matrix().unwrap().reduced_mode(0).unwrap();
        close(reduced.trace(), 1.0);
        assert!(reduced.purity() < 1.0);
        for n in 0..=4 {
            close(
                state.probability(&[n, n]).unwrap(),
                reduced.element(n, n).unwrap().re,
            );
        }
    }

    #[test]
    fn phase_shift_preserves_all_probabilities() {
        let space = FockSpace::try_new(2, 2).unwrap();
        let input = StateVector::basis(space, &[1, 1])
            .unwrap()
            .beam_splitter(0, 1, 0.31)
            .unwrap();
        let output = input.phase_shift(0, 1.7).unwrap();
        for index in 0..space.dimension() {
            let occupations = space.occupations(index).unwrap();
            close(
                input.probability(&occupations).unwrap(),
                output.probability(&occupations).unwrap(),
            );
        }
    }

    #[test]
    fn beam_splitter_is_unitary_on_every_representable_basis_sector() {
        let space = FockSpace::try_new(2, 5).unwrap();
        for n in 0..=5 {
            for m in 0..=5 - n {
                let input = StateVector::basis(space, &[n, m]).unwrap();
                let output = input.beam_splitter(0, 1, 0.37).unwrap();
                close(output.norm_squared(), 1.0);
                let recovered = output.beam_splitter(0, 1, -0.37).unwrap();
                for index in 0..space.dimension() {
                    let occupations = space.occupations(index).unwrap();
                    close(
                        recovered.probability(&occupations).unwrap(),
                        input.probability(&occupations).unwrap(),
                    );
                }
            }
        }
    }

    #[test]
    fn validation_rejects_bad_shapes_domains_and_zero_norm() {
        assert_eq!(FockSpace::try_new(0, 1), Err(QuantumLightError::ZeroModes));
        let space = FockSpace::try_new(2, 1).unwrap();
        assert!(matches!(
            StateVector::try_from_amplitudes(space, vec![]),
            Err(QuantumLightError::AmplitudeShape { .. })
        ));
        let zero = StateVector::try_from_amplitudes(
            space,
            vec![Complex64::new(0.0, 0.0); space.dimension()],
        )
        .unwrap();
        assert_eq!(zero.normalized(), Err(QuantumLightError::ZeroNorm));
        assert_eq!(
            StateVector::two_mode_squeezed(space, 0, 1, 1.0),
            Err(QuantumLightError::InvalidSqueezing)
        );
    }

    #[test]
    fn small_nonzero_amplitudes_are_not_silently_discarded() {
        let space = FockSpace::try_new(1, 1).unwrap();
        let state = StateVector::try_from_amplitudes(
            space,
            vec![Complex64::new(0.0, 0.0), Complex64::new(1.0e-100, 0.0)],
        )
        .unwrap();
        let normalized = state.normalized().unwrap();
        close(normalized.probability(&[1]).unwrap(), 1.0);
        assert!(state.creation_would_truncate(0).unwrap());
    }
}
