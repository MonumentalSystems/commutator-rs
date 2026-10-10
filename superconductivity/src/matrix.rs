//! Dense matrix representation and symmetry diagnostics.

use crate::Complex64;

/// A dense BdG Hamiltonian in particle-first Nambu ordering.
#[derive(Clone, Debug, PartialEq)]
pub struct BdGMatrix {
    pub(crate) site_count: usize,
    pub(crate) entries: Vec<Complex64>,
}

impl BdGMatrix {
    /// Returns the number of physical lattice sites.
    pub fn site_count(&self) -> usize {
        self.site_count
    }

    /// Returns the number of spin orbitals, excluding Nambu duplication.
    pub fn orbital_count(&self) -> usize {
        self.site_count * 2
    }

    /// Returns the full Nambu matrix dimension.
    pub fn dimension(&self) -> usize {
        self.orbital_count() * 2
    }

    /// Returns a matrix element by row and column.
    ///
    /// # Panics
    ///
    /// Panics when either index is at least [`Self::dimension`].
    pub fn get(&self, row: usize, column: usize) -> Complex64 {
        self.entries[row * self.dimension() + column]
    }

    /// Returns all row-major matrix entries.
    pub fn as_slice(&self) -> &[Complex64] {
        &self.entries
    }

    /// Returns the maximum residual of `H = H^dagger`.
    pub fn hermiticity_residual(&self) -> f64 {
        let n = self.dimension();
        let mut residual: f64 = 0.0;
        for row in 0..n {
            for column in 0..n {
                residual =
                    residual.max((self.get(row, column) - self.get(column, row).conj()).norm());
            }
        }
        residual
    }

    /// Tests Hermiticity at an absolute element-wise tolerance.
    pub fn is_hermitian(&self, tolerance: f64) -> bool {
        tolerance.is_finite() && tolerance >= 0.0 && self.hermiticity_residual() <= tolerance
    }

    /// Returns the maximum residual of `tau_x H* tau_x = -H`.
    ///
    /// This is the particle-hole symmetry relation in the crate's Nambu
    /// convention, where `tau_x` exchanges particle and hole blocks.
    pub fn particle_hole_residual(&self) -> f64 {
        let orbitals = self.orbital_count();
        let n = self.dimension();
        let mut residual: f64 = 0.0;
        for row in 0..n {
            let transformed_row = if row < orbitals {
                row + orbitals
            } else {
                row - orbitals
            };
            for column in 0..n {
                let transformed_column = if column < orbitals {
                    column + orbitals
                } else {
                    column - orbitals
                };
                let lhs = self.get(transformed_row, transformed_column).conj();
                residual = residual.max((lhs + self.get(row, column)).norm());
            }
        }
        residual
    }

    /// Applies the antiunitary particle-hole map `tau_x K` to a Nambu vector.
    ///
    /// Returns `None` when `state` does not have [`Self::dimension`] entries.
    pub fn particle_hole_conjugate(&self, state: &[Complex64]) -> Option<Vec<Complex64>> {
        let orbitals = self.orbital_count();
        if state.len() != orbitals * 2 {
            return None;
        }
        let mut result = vec![Complex64::ZERO; state.len()];
        for index in 0..orbitals {
            result[index] = state[index + orbitals].conj();
            result[index + orbitals] = state[index].conj();
        }
        Some(result)
    }

    pub(crate) fn multiply(&self, vector: &[Complex64]) -> Vec<Complex64> {
        let n = self.dimension();
        let mut result = vec![Complex64::ZERO; n];
        for (row, result_entry) in result.iter_mut().enumerate() {
            let mut sum = Complex64::ZERO;
            for (column, &vector_entry) in vector.iter().enumerate() {
                sum += self.get(row, column) * vector_entry;
            }
            *result_entry = sum;
        }
        result
    }
}

/// Returns the maximum `E_i + E_(N-1-i)` residual after sorting a spectrum.
///
/// A particle-hole-symmetric spectrum is paired around zero. For an odd
/// number of values, the central sorted value must therefore be zero. An
/// empty input returns zero, while any non-finite energy returns infinity.
pub fn spectrum_particle_hole_residual(energies: &[f64]) -> f64 {
    if energies.iter().any(|energy| !energy.is_finite()) {
        return f64::INFINITY;
    }
    let mut sorted = energies.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mut residual: f64 = 0.0;
    for index in 0..(sorted.len() / 2) {
        residual = residual.max((sorted[index] + sorted[sorted.len() - 1 - index]).abs());
    }
    if sorted.len() % 2 == 1 {
        residual = residual.max(sorted[sorted.len() / 2].abs());
    }
    residual
}
