//! Harmonic-chain phonons and ballistic thermal-transport reference kernels.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use core::fmt;

use num_complex::Complex64;

/// Exact Boltzmann constant in joules per kelvin.
pub const BOLTZMANN_JOULES_PER_KELVIN: f64 = 1.380_649e-23;
/// Reduced Planck constant in joule-seconds.
pub const HBAR_JOULE_SECONDS: f64 = 1.054_571_817e-34;
/// Exact Planck constant in joule-seconds.
pub const PLANCK_JOULE_SECONDS: f64 = 6.626_070_15e-34;

/// Errors returned by checked phonon calculations.
#[derive(Clone, Debug, PartialEq)]
pub enum PhononError {
    /// The unit cell has no degrees of freedom.
    EmptyCell,
    /// A scalar parameter was NaN, infinite, or outside its physical range.
    InvalidParameter(&'static str),
    /// A bond references an atom outside the unit cell.
    AtomIndex {
        /// Number of atoms in the cell.
        atoms: usize,
        /// Invalid atom index.
        index: usize,
    },
    /// A checked allocation size overflowed.
    SizeOverflow,
    /// A dynamical matrix has a negative eigenvalue beyond tolerance.
    DynamicalInstability {
        /// Most negative squared frequency.
        omega_squared: f64,
    },
    /// The Hermitian Jacobi iteration did not converge.
    EigensolverDidNotConverge,
    /// Group velocity is not uniquely defined at a detected mode degeneracy.
    DegenerateModes {
        /// First branch in the degenerate pair.
        first: usize,
        /// Second branch in the degenerate pair.
        second: usize,
    },
    /// At least two reciprocal-space samples are required.
    TooFewSamples,
}

impl fmt::Display for PhononError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for PhononError {}

/// Result type for checked phonon calculations.
pub type Result<T> = core::result::Result<T, PhononError>;

fn finite_positive(value: f64, name: &'static str) -> Result<f64> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(PhononError::InvalidParameter(name))
    }
}

/// A scalar harmonic spring connecting two unit-cell degrees of freedom.
///
/// The bond connects `atom_a` in a reference cell to `atom_b` in the cell at
/// `cell_offset`. Each physical bond should be listed once.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HarmonicBond {
    atom_a: usize,
    atom_b: usize,
    cell_offset: i32,
    stiffness: f64,
}

impl HarmonicBond {
    /// Constructs a bond with strictly positive stiffness.
    pub fn try_new(atom_a: usize, atom_b: usize, cell_offset: i32, stiffness: f64) -> Result<Self> {
        finite_positive(stiffness, "bond stiffness")?;
        if atom_a == atom_b && cell_offset == 0 {
            return Err(PhononError::InvalidParameter("bond endpoints"));
        }
        Ok(Self {
            atom_a,
            atom_b,
            cell_offset,
            stiffness,
        })
    }

    /// Returns the atom in the reference cell.
    #[must_use]
    pub const fn atom_a(&self) -> usize {
        self.atom_a
    }

    /// Returns the atom in the displaced cell.
    #[must_use]
    pub const fn atom_b(&self) -> usize {
        self.atom_b
    }

    /// Returns the integer cell displacement of the second endpoint.
    #[must_use]
    pub const fn cell_offset(&self) -> i32 {
        self.cell_offset
    }

    /// Returns the spring stiffness.
    #[must_use]
    pub const fn stiffness(&self) -> f64 {
        self.stiffness
    }
}

/// A one-dimensional periodic harmonic lattice with scalar displacements.
#[derive(Clone, Debug, PartialEq)]
pub struct HarmonicChain {
    lattice_spacing: f64,
    masses: Vec<f64>,
    bonds: Vec<HarmonicBond>,
}

impl HarmonicChain {
    /// Constructs and validates a periodic unit-cell model.
    pub fn try_new(
        lattice_spacing: f64,
        masses: Vec<f64>,
        bonds: Vec<HarmonicBond>,
    ) -> Result<Self> {
        finite_positive(lattice_spacing, "lattice spacing")?;
        if masses.is_empty() {
            return Err(PhononError::EmptyCell);
        }
        for &mass in &masses {
            finite_positive(mass, "mass")?;
        }
        for bond in &bonds {
            for index in [bond.atom_a, bond.atom_b] {
                if index >= masses.len() {
                    return Err(PhononError::AtomIndex {
                        atoms: masses.len(),
                        index,
                    });
                }
            }
        }
        Ok(Self {
            lattice_spacing,
            masses,
            bonds,
        })
    }

    /// Returns the lattice spacing.
    #[must_use]
    pub const fn lattice_spacing(&self) -> f64 {
        self.lattice_spacing
    }

    /// Returns masses in stable atom order.
    #[must_use]
    pub fn masses(&self) -> &[f64] {
        &self.masses
    }

    /// Returns bonds in stable input order.
    #[must_use]
    pub fn bonds(&self) -> &[HarmonicBond] {
        &self.bonds
    }

    /// Returns the first Brillouin-zone boundary `pi / a`.
    #[must_use]
    pub fn brillouin_boundary(&self) -> f64 {
        core::f64::consts::PI / self.lattice_spacing
    }

    /// Builds the mass-weighted row-major dynamical matrix at wavevector `k`.
    pub fn dynamical_matrix(&self, wavevector: f64) -> Result<Vec<Complex64>> {
        if !wavevector.is_finite() {
            return Err(PhononError::InvalidParameter("wavevector"));
        }
        let n = self.masses.len();
        let length = n.checked_mul(n).ok_or(PhononError::SizeOverflow)?;
        let mut matrix = vec![Complex64::new(0.0, 0.0); length];
        for bond in &self.bonds {
            let i = bond.atom_a;
            let j = bond.atom_b;
            let stiffness = bond.stiffness;
            matrix[i * n + i] += stiffness / self.masses[i];
            matrix[j * n + j] += stiffness / self.masses[j];

            let phase = wavevector * self.lattice_spacing * f64::from(bond.cell_offset);
            let phase_factor = Complex64::new(phase.cos(), phase.sin());
            let coupling = -stiffness / (self.masses[i] * self.masses[j]).sqrt();
            matrix[i * n + j] += coupling * phase_factor;
            matrix[j * n + i] += coupling * phase_factor.conj();
        }
        Ok(matrix)
    }

    /// Returns sorted nonnegative angular frequencies at wavevector `k`.
    ///
    /// `tolerance` is a relative threshold for both eigensolver convergence
    /// and small negative squared frequencies caused by roundoff.
    pub fn frequencies(&self, wavevector: f64, tolerance: f64) -> Result<Vec<f64>> {
        finite_positive(tolerance, "eigensolver tolerance")?;
        Ok(self.mode_system(wavevector, tolerance)?.frequencies)
    }

    /// Samples all branches from `-pi/a` through `+pi/a`, inclusive.
    pub fn dispersion(&self, samples: usize, tolerance: f64) -> Result<Vec<DispersionPoint>> {
        if samples < 2 {
            return Err(PhononError::TooFewSamples);
        }
        let boundary = self.brillouin_boundary();
        (0..samples)
            .map(|index| {
                let fraction = index as f64 / (samples - 1) as f64;
                let wavevector = -boundary + 2.0 * boundary * fraction;
                Ok(DispersionPoint {
                    wavevector,
                    frequencies: self.frequencies(wavevector, tolerance)?,
                })
            })
            .collect()
    }

    /// Estimates branch group velocities `d omega / d k` by central difference.
    ///
    /// Eigenvector-overlap assignment follows physical branches through
    /// ordinary crossings. An exact degeneracy at the requested wavevector is
    /// rejected because individual velocities then depend on a basis choice
    /// inside the degenerate subspace.
    pub fn group_velocities(
        &self,
        wavevector: f64,
        wavevector_step: f64,
        tolerance: f64,
    ) -> Result<Vec<f64>> {
        finite_positive(wavevector_step, "wavevector step")?;
        finite_positive(tolerance, "eigensolver tolerance")?;
        let center = self.mode_system(wavevector, tolerance)?;
        for index in 1..center.omega_squared.len() {
            if (center.omega_squared[index] - center.omega_squared[index - 1]).abs()
                <= 8.0 * tolerance * center.scale.max(f64::MIN_POSITIVE)
            {
                return Err(PhononError::DegenerateModes {
                    first: index - 1,
                    second: index,
                });
            }
        }
        let lower = self.mode_system(wavevector - wavevector_step, tolerance)?;
        let upper = self.mode_system(wavevector + wavevector_step, tolerance)?;
        let lower_assignment =
            match_eigenvectors(&center.eigenvectors, &lower.eigenvectors, self.masses.len());
        let upper_assignment =
            match_eigenvectors(&center.eigenvectors, &upper.eigenvectors, self.masses.len());
        Ok((0..self.masses.len())
            .map(|branch| {
                (upper.frequencies[upper_assignment[branch]]
                    - lower.frequencies[lower_assignment[branch]])
                    / (2.0 * wavevector_step)
            })
            .collect())
    }

    /// Returns harmonic constant-volume heat capacity per unit cell.
    ///
    /// The Brillouin-zone integral uses midpoint sampling. Frequencies must be
    /// in radians per second and temperature in kelvin for an SI result.
    pub fn heat_capacity(&self, temperature: f64, samples: usize, tolerance: f64) -> Result<f64> {
        finite_positive(temperature, "temperature")?;
        if samples < 2 {
            return Err(PhononError::TooFewSamples);
        }
        let boundary = self.brillouin_boundary();
        let mut sum = 0.0;
        for index in 0..samples {
            let wavevector = -boundary + 2.0 * boundary * (index as f64 + 0.5) / samples as f64;
            for omega in self.frequencies(wavevector, tolerance)? {
                sum += mode_heat_capacity(omega, temperature)?;
            }
        }
        Ok(sum / samples as f64)
    }

    /// Returns perfect-transmission ballistic thermal conductance.
    ///
    /// The full Brillouin zone is midpoint sampled. The factor `|v| / 2`
    /// avoids double-counting opposite propagation directions.
    pub fn ballistic_thermal_conductance(
        &self,
        temperature: f64,
        samples: usize,
        tolerance: f64,
    ) -> Result<f64> {
        finite_positive(temperature, "temperature")?;
        if samples < 2 {
            return Err(PhononError::TooFewSamples);
        }
        let boundary = self.brillouin_boundary();
        let delta_k = 2.0 * boundary / samples as f64;
        let derivative_step = delta_k * 0.25;
        let mut integral = 0.0;
        for index in 0..samples {
            let wavevector = -boundary + delta_k * (index as f64 + 0.5);
            let frequencies = self.frequencies(wavevector, tolerance)?;
            let velocities = self.group_velocities(wavevector, derivative_step, tolerance)?;
            for (omega, velocity) in frequencies.into_iter().zip(velocities) {
                integral +=
                    0.5 * velocity.abs() * mode_energy_temperature_derivative(omega, temperature)?;
            }
        }
        Ok(integral * delta_k / (2.0 * core::f64::consts::PI))
    }

    fn mode_system(&self, wavevector: f64, tolerance: f64) -> Result<ModeSystem> {
        let matrix = self.dynamical_matrix(wavevector)?;
        let scale = matrix
            .iter()
            .fold(0.0_f64, |scale, value| scale.max(value.norm()));
        let (omega_squared, eigenvectors) =
            hermitian_eigensystem(&matrix, self.masses.len(), tolerance)?;
        let stability_scale = scale.max(f64::MIN_POSITIVE);
        let frequencies = omega_squared
            .iter()
            .map(|&value| {
                if value < -tolerance * stability_scale {
                    Err(PhononError::DynamicalInstability {
                        omega_squared: value,
                    })
                } else {
                    let stable_value = if value.abs() <= tolerance * stability_scale {
                        0.0
                    } else {
                        value
                    };
                    Ok(stable_value.max(0.0).sqrt())
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(ModeSystem {
            omega_squared,
            frequencies,
            eigenvectors,
            scale,
        })
    }
}

struct ModeSystem {
    omega_squared: Vec<f64>,
    frequencies: Vec<f64>,
    eigenvectors: Vec<Complex64>,
    scale: f64,
}

/// One sampled point on a multi-branch phonon dispersion.
#[derive(Clone, Debug, PartialEq)]
pub struct DispersionPoint {
    /// Wavevector.
    pub wavevector: f64,
    /// Sorted angular frequencies.
    pub frequencies: Vec<f64>,
}

/// Returns the universal thermal conductance quantum for one ballistic mode.
pub fn thermal_conductance_quantum(temperature: f64) -> Result<f64> {
    finite_positive(temperature, "temperature")?;
    Ok(
        core::f64::consts::PI.powi(2) * BOLTZMANN_JOULES_PER_KELVIN.powi(2) * temperature
            / (3.0 * PLANCK_JOULE_SECONDS),
    )
}

/// Returns the Bose–Einstein occupation of an angular-frequency mode.
pub fn bose_einstein_occupation(omega: f64, temperature: f64) -> Result<f64> {
    if !omega.is_finite() || omega < 0.0 {
        return Err(PhononError::InvalidParameter("angular frequency"));
    }
    if omega == 0.0 {
        return Ok(f64::INFINITY);
    }
    let x = HBAR_JOULE_SECONDS * omega / (BOLTZMANN_JOULES_PER_KELVIN * temperature);
    if x > 710.0 {
        Ok(0.0)
    } else {
        Ok(x.exp_m1().recip())
    }
}

fn mode_heat_capacity(omega: f64, temperature: f64) -> Result<f64> {
    if omega == 0.0 {
        return Ok(BOLTZMANN_JOULES_PER_KELVIN);
    }
    let x = HBAR_JOULE_SECONDS * omega / (BOLTZMANN_JOULES_PER_KELVIN * temperature);
    if x.is_nan() || x < 0.0 {
        return Err(PhononError::InvalidParameter("mode energy"));
    }
    if x.is_infinite() || x > 350.0 {
        return Ok(0.0);
    }
    if x.abs() < 1.0e-4 {
        return Ok(BOLTZMANN_JOULES_PER_KELVIN * (1.0 - x * x / 12.0));
    }
    let exponential = x.exp();
    Ok(BOLTZMANN_JOULES_PER_KELVIN * x * x * exponential / x.exp_m1().powi(2))
}

fn mode_energy_temperature_derivative(omega: f64, temperature: f64) -> Result<f64> {
    mode_heat_capacity(omega, temperature)
}

fn hermitian_eigensystem(
    values: &[Complex64],
    dimension: usize,
    tolerance: f64,
) -> Result<(Vec<f64>, Vec<Complex64>)> {
    let expected = dimension
        .checked_mul(dimension)
        .ok_or(PhononError::SizeOverflow)?;
    if dimension == 0 || values.len() != expected {
        return Err(PhononError::InvalidParameter("dynamical matrix shape"));
    }
    let mut matrix = values.to_vec();
    let scale = matrix
        .iter()
        .fold(0.0_f64, |scale, value| scale.max(value.norm()))
        .max(f64::MIN_POSITIVE);
    let maximum_rotations = 64usize
        .checked_mul(dimension)
        .and_then(|count| count.checked_mul(dimension))
        .ok_or(PhononError::SizeOverflow)?;
    let mut eigenvectors = vec![Complex64::new(0.0, 0.0); expected];
    for index in 0..dimension {
        eigenvectors[index * dimension + index] = Complex64::new(1.0, 0.0);
    }

    for _ in 0..maximum_rotations {
        let mut pivot = (0, 0);
        let mut maximum = 0.0_f64;
        for row in 0..dimension {
            for column in row + 1..dimension {
                let magnitude = matrix[row * dimension + column].norm();
                if magnitude > maximum {
                    maximum = magnitude;
                    pivot = (row, column);
                }
            }
        }
        if maximum <= tolerance * scale {
            let eigenvalues: Vec<_> = (0..dimension)
                .map(|index| matrix[index * dimension + index].re)
                .collect();
            let mut order: Vec<_> = (0..dimension).collect();
            order.sort_by(|&left, &right| eigenvalues[left].total_cmp(&eigenvalues[right]));
            let sorted_values = order.iter().map(|&index| eigenvalues[index]).collect();
            let mut sorted_vectors = vec![Complex64::new(0.0, 0.0); expected];
            for (new_column, &old_column) in order.iter().enumerate() {
                for row in 0..dimension {
                    sorted_vectors[row * dimension + new_column] =
                        eigenvectors[row * dimension + old_column];
                }
            }
            return Ok((sorted_values, sorted_vectors));
        }

        let (p, q) = pivot;
        let app = matrix[p * dimension + p].re;
        let aqq = matrix[q * dimension + q].re;
        let apq = matrix[p * dimension + q];
        let radius = apq.norm();
        let tau = (aqq - app) / (2.0 * radius);
        let tangent = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let cosine = 1.0 / (1.0 + tangent * tangent).sqrt();
        let sine = tangent * cosine;
        let phase = apq / radius;

        for index in 0..dimension {
            if index == p || index == q {
                continue;
            }
            let aip = matrix[index * dimension + p];
            let aiq = matrix[index * dimension + q];
            let new_ip = aip * phase * cosine - aiq * sine;
            let new_iq = aip * phase * sine + aiq * cosine;
            matrix[index * dimension + p] = new_ip;
            matrix[p * dimension + index] = new_ip.conj();
            matrix[index * dimension + q] = new_iq;
            matrix[q * dimension + index] = new_iq.conj();
        }
        for row in 0..dimension {
            let vip = eigenvectors[row * dimension + p];
            let viq = eigenvectors[row * dimension + q];
            eigenvectors[row * dimension + p] = vip * phase * cosine - viq * sine;
            eigenvectors[row * dimension + q] = vip * phase * sine + viq * cosine;
        }
        matrix[p * dimension + p] = Complex64::new(
            cosine * cosine * app + sine * sine * aqq - 2.0 * cosine * sine * radius,
            0.0,
        );
        matrix[q * dimension + q] = Complex64::new(
            sine * sine * app + cosine * cosine * aqq + 2.0 * cosine * sine * radius,
            0.0,
        );
        matrix[p * dimension + q] = Complex64::new(0.0, 0.0);
        matrix[q * dimension + p] = Complex64::new(0.0, 0.0);
    }
    Err(PhononError::EigensolverDidNotConverge)
}

fn match_eigenvectors(reference: &[Complex64], candidate: &[Complex64], n: usize) -> Vec<usize> {
    let mut costs = vec![vec![0.0_f64; n]; n];
    for reference_column in 0..n {
        for candidate_column in 0..n {
            let overlap: Complex64 = (0..n)
                .map(|row| {
                    reference[row * n + reference_column].conj()
                        * candidate[row * n + candidate_column]
                })
                .sum();
            costs[reference_column][candidate_column] = 1.0 - overlap.norm_sqr().min(1.0);
        }
    }
    minimum_cost_assignment(&costs)
}

fn minimum_cost_assignment(costs: &[Vec<f64>]) -> Vec<usize> {
    let n = costs.len();
    let mut row_potential = vec![0.0_f64; n + 1];
    let mut column_potential = vec![0.0_f64; n + 1];
    let mut matched_row = vec![0usize; n + 1];
    let mut predecessor = vec![0usize; n + 1];

    for row in 1..=n {
        matched_row[0] = row;
        let mut column = 0usize;
        let mut minimum = vec![f64::INFINITY; n + 1];
        let mut used = vec![false; n + 1];
        loop {
            used[column] = true;
            let active_row = matched_row[column];
            let mut delta = f64::INFINITY;
            let mut next_column = 0usize;
            for candidate_column in 1..=n {
                if used[candidate_column] {
                    continue;
                }
                let reduced = costs[active_row - 1][candidate_column - 1]
                    - row_potential[active_row]
                    - column_potential[candidate_column];
                if reduced < minimum[candidate_column] {
                    minimum[candidate_column] = reduced;
                    predecessor[candidate_column] = column;
                }
                if minimum[candidate_column] < delta {
                    delta = minimum[candidate_column];
                    next_column = candidate_column;
                }
            }
            for candidate_column in 0..=n {
                if used[candidate_column] {
                    row_potential[matched_row[candidate_column]] += delta;
                    column_potential[candidate_column] -= delta;
                } else {
                    minimum[candidate_column] -= delta;
                }
            }
            column = next_column;
            if matched_row[column] == 0 {
                break;
            }
        }
        loop {
            let previous = predecessor[column];
            matched_row[column] = matched_row[previous];
            column = previous;
            if column == 0 {
                break;
            }
        }
    }

    let mut assignment = vec![0usize; n];
    for column in 1..=n {
        assignment[matched_row[column] - 1] = column - 1;
    }
    assignment
}

#[cfg(test)]
mod tests {
    use super::*;

    fn monoatomic_chain() -> HarmonicChain {
        monoatomic_chain_with_stiffness(1.0)
    }

    fn monoatomic_chain_with_stiffness(stiffness: f64) -> HarmonicChain {
        HarmonicChain::try_new(
            1.0,
            vec![1.0],
            vec![HarmonicBond::try_new(0, 0, 1, stiffness).unwrap()],
        )
        .unwrap()
    }

    #[test]
    fn monoatomic_dispersion_matches_closed_form() {
        let chain = monoatomic_chain();
        for wavevector in [0.0, 0.3, 1.7, core::f64::consts::PI] {
            let actual = chain.frequencies(wavevector, 1e-12).unwrap()[0];
            let expected = 2.0 * (0.5 * wavevector).sin().abs();
            assert!((actual - expected).abs() < 1e-10, "{actual} {expected}");
        }
    }

    #[test]
    fn diatomic_chain_has_acoustic_and_optical_modes() {
        let chain = HarmonicChain::try_new(
            1.0,
            vec![1.0, 2.0],
            vec![
                HarmonicBond::try_new(0, 1, 0, 1.0).unwrap(),
                HarmonicBond::try_new(1, 0, 1, 1.0).unwrap(),
            ],
        )
        .unwrap();
        let frequencies = chain.frequencies(0.0, 1e-12).unwrap();
        assert!(frequencies[0] < 1e-8);
        assert!((frequencies[1] - 3.0_f64.sqrt()).abs() < 1e-10);
        let matrix = chain.dynamical_matrix(0.7).unwrap();
        assert!((matrix[1] - matrix[2].conj()).norm() < 1e-14);
    }

    #[test]
    fn group_velocity_matches_acoustic_derivative() {
        let velocity = monoatomic_chain()
            .group_velocities(0.7, 1e-5, 1e-12)
            .unwrap()[0];
        let expected = (0.35_f64).cos();
        assert!((velocity - expected).abs() < 1e-9);
    }

    #[test]
    fn classical_high_temperature_heat_capacity_is_one_kb_per_mode() {
        let chain = HarmonicChain::try_new(
            1.0,
            vec![1.0, 2.0],
            vec![
                HarmonicBond::try_new(0, 1, 0, 1.0).unwrap(),
                HarmonicBond::try_new(1, 0, 1, 1.0).unwrap(),
            ],
        )
        .unwrap();
        let capacity = chain.heat_capacity(300.0, 128, 1e-12).unwrap();
        assert!((capacity / BOLTZMANN_JOULES_PER_KELVIN - 2.0).abs() < 1e-10);
    }

    #[test]
    fn conductance_quantum_is_positive_and_linear_in_temperature() {
        let low = thermal_conductance_quantum(2.0).unwrap();
        let high = thermal_conductance_quantum(6.0).unwrap();
        assert!(low > 0.0);
        assert!((high / low - 3.0).abs() < 1e-14);
    }

    #[test]
    fn acoustic_stability_threshold_is_scale_invariant() {
        for stiffness in [1.0e-20, 1.0, 1.0e20] {
            let frequency = monoatomic_chain_with_stiffness(stiffness)
                .frequencies(core::f64::consts::PI, 1e-12)
                .unwrap()[0];
            let expected = 2.0 * stiffness.sqrt();
            assert!(((frequency - expected) / expected).abs() < 1e-12);
        }
    }

    #[test]
    fn exact_mode_degeneracy_is_reported_for_velocity() {
        let chain = HarmonicChain::try_new(
            1.0,
            vec![1.0, 1.0],
            vec![
                HarmonicBond::try_new(0, 0, 1, 1.0).unwrap(),
                HarmonicBond::try_new(1, 1, 1, 1.0).unwrap(),
            ],
        )
        .unwrap();
        assert!(matches!(
            chain.group_velocities(0.7, 1e-5, 1e-12),
            Err(PhononError::DegenerateModes { .. })
        ));
    }

    #[test]
    fn finite_extreme_low_temperature_has_zero_heat_capacity_limit() {
        let capacity = monoatomic_chain()
            .heat_capacity(f64::MIN_POSITIVE, 64, 1e-12)
            .unwrap();
        assert_eq!(capacity, 0.0);
    }
}
