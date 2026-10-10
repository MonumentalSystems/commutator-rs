use crate::{Complex64, MagnetismError, Result, SpinModel};

/// Configuration for the deterministic Lanczos ground-state solver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LanczosConfig {
    /// Maximum Krylov-subspace dimension.
    pub max_iterations: usize,
    /// Absolute Ritz-residual convergence tolerance.
    pub tolerance: f64,
    /// Seed for the deterministic xorshift starting vector.
    pub seed: u64,
}

impl Default for LanczosConfig {
    fn default() -> Self {
        Self {
            max_iterations: 128,
            tolerance: 1.0e-12,
            seed: 0x5eed_5eed_cafe_f00d,
        }
    }
}

/// Lowest Ritz pair returned by the Lanczos solver.
#[derive(Debug, Clone, PartialEq)]
pub struct GroundState {
    /// Approximate lowest eigenvalue.
    pub energy: f64,
    /// Normalized approximate eigenvector in computational-basis order.
    pub state: Vec<Complex64>,
    /// Norm of the Ritz residual estimated from the Lanczos recurrence.
    pub residual_norm: f64,
    /// Number of Hamiltonian applications performed.
    pub iterations: usize,
    /// Whether `residual_norm <= tolerance` or an invariant subspace closed.
    pub converged: bool,
}

fn dot(left: &[Complex64], right: &[Complex64]) -> Complex64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left.conj() * *right)
        .sum()
}

fn vector_norm(vector: &[Complex64]) -> f64 {
    vector
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
}

fn deterministic_start(length: usize, mut state: u64) -> Vec<Complex64> {
    if state == 0 {
        state = 0x9e37_79b9_7f4a_7c15;
    }
    let mut vector = Vec::with_capacity(length);
    for _ in 0..length {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let real = (state as f64 / u64::MAX as f64) - 0.5;
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let imaginary = (state as f64 / u64::MAX as f64) - 0.5;
        vector.push(Complex64::new(real, imaginary));
    }
    let norm = vector_norm(&vector);
    vector.iter_mut().for_each(|value| *value = *value / norm);
    vector
}

fn smallest_symmetric_eigenpair(diagonal: &[f64], off_diagonal: &[f64]) -> Result<(f64, Vec<f64>)> {
    let size = diagonal.len();
    let mut matrix = vec![0.0; size * size];
    let mut vectors = vec![0.0; size * size];
    for index in 0..size {
        matrix[index * size + index] = diagonal[index];
        vectors[index * size + index] = 1.0;
        if index + 1 < size {
            matrix[index * size + index + 1] = off_diagonal[index];
            matrix[(index + 1) * size + index] = off_diagonal[index];
        }
    }

    let scale = diagonal
        .iter()
        .chain(off_diagonal)
        .fold(1.0_f64, |scale, value| scale.max(value.abs()));
    let threshold = 32.0 * f64::EPSILON * scale;
    let sweeps = 64usize.saturating_mul(size.max(1));
    for _ in 0..sweeps {
        let mut p = 0;
        let mut q = 0;
        let mut largest = 0.0;
        for row in 0..size {
            for column in row + 1..size {
                let value = matrix[row * size + column].abs();
                if value > largest {
                    largest = value;
                    p = row;
                    q = column;
                }
            }
        }
        if largest <= threshold {
            let (minimum, energy) = diagonal_index_min(&matrix, size);
            let eigenvector = (0..size).map(|row| vectors[row * size + minimum]).collect();
            return Ok((energy, eigenvector));
        }

        let app = matrix[p * size + p];
        let aqq = matrix[q * size + q];
        let apq = matrix[p * size + q];
        let angle = 0.5 * (2.0 * apq).atan2(aqq - app);
        let (sine, cosine) = angle.sin_cos();

        for index in 0..size {
            if index != p && index != q {
                let aip = matrix[index * size + p];
                let aiq = matrix[index * size + q];
                let new_ip = cosine * aip - sine * aiq;
                let new_iq = sine * aip + cosine * aiq;
                matrix[index * size + p] = new_ip;
                matrix[p * size + index] = new_ip;
                matrix[index * size + q] = new_iq;
                matrix[q * size + index] = new_iq;
            }
        }
        matrix[p * size + p] =
            cosine * cosine * app - 2.0 * sine * cosine * apq + sine * sine * aqq;
        matrix[q * size + q] =
            sine * sine * app + 2.0 * sine * cosine * apq + cosine * cosine * aqq;
        matrix[p * size + q] = 0.0;
        matrix[q * size + p] = 0.0;

        for row in 0..size {
            let vip = vectors[row * size + p];
            let viq = vectors[row * size + q];
            vectors[row * size + p] = cosine * vip - sine * viq;
            vectors[row * size + q] = sine * vip + cosine * viq;
        }
    }
    Err(MagnetismError::NumericalFailure(
        "Jacobi diagonalization did not converge",
    ))
}

fn diagonal_index_min(matrix: &[f64], size: usize) -> (usize, f64) {
    let mut index = 0;
    let mut value = matrix[0];
    for candidate in 1..size {
        let candidate_value = matrix[candidate * size + candidate];
        if candidate_value < value {
            index = candidate;
            value = candidate_value;
        }
    }
    (index, value)
}

impl SpinModel {
    /// Approximate the lowest eigenpair with fully reorthogonalized Lanczos.
    ///
    /// This reference implementation stores every Krylov vector. Work scales
    /// as `O(iterations × bonds × 2^N)` and memory as
    /// `O(iterations × 2^N)`, so it is deliberately intended for small
    /// clusters and validation.
    pub fn ground_state(&self, config: LanczosConfig) -> Result<GroundState> {
        if config.max_iterations == 0 {
            return Err(MagnetismError::InvalidSolverParameter(
                "max_iterations must be positive",
            ));
        }
        if !config.tolerance.is_finite() || config.tolerance <= 0.0 {
            return Err(MagnetismError::InvalidSolverParameter(
                "tolerance must be finite and positive",
            ));
        }
        let limit = config.max_iterations.min(self.hilbert_dimension());
        let mut basis = vec![deterministic_start(self.hilbert_dimension(), config.seed)];
        let mut diagonal = Vec::with_capacity(limit);
        let mut off_diagonal = Vec::with_capacity(limit.saturating_sub(1));
        let mut tail_beta = 0.0;
        let mut closed = false;

        for iteration in 0..limit {
            let current = &basis[iteration];
            let mut work = self.applied(current)?;
            if iteration > 0 {
                let beta = off_diagonal[iteration - 1];
                for (value, previous) in work.iter_mut().zip(&basis[iteration - 1]) {
                    *value -= *previous * beta;
                }
            }
            let alpha_complex = dot(current, &work);
            if alpha_complex.im.abs() > 1.0e-10 * (1.0 + alpha_complex.re.abs()) {
                return Err(MagnetismError::NumericalFailure(
                    "Lanczos diagonal was not real",
                ));
            }
            let alpha = alpha_complex.re;
            for (value, current) in work.iter_mut().zip(current) {
                *value -= *current * alpha;
            }

            // Two modified Gram-Schmidt passes make this a reliable small
            // reference solver even for clustered or degenerate spectra.
            for _ in 0..2 {
                for vector in &basis {
                    let projection = dot(vector, &work);
                    for (value, basis_value) in work.iter_mut().zip(vector) {
                        *value -= *basis_value * projection;
                    }
                }
            }
            diagonal.push(alpha);
            tail_beta = vector_norm(&work);
            if !tail_beta.is_finite() {
                return Err(MagnetismError::NumericalFailure(
                    "Lanczos recurrence overflowed",
                ));
            }
            if tail_beta <= config.tolerance {
                tail_beta = 0.0;
                closed = true;
                break;
            }
            if iteration + 1 == limit {
                break;
            }
            off_diagonal.push(tail_beta);
            work.iter_mut()
                .for_each(|value| *value = *value / tail_beta);
            basis.push(work);
        }

        let (energy, coefficients) = smallest_symmetric_eigenpair(&diagonal, &off_diagonal)?;
        let mut state = vec![Complex64::ZERO; self.hilbert_dimension()];
        for (coefficient, vector) in coefficients.iter().zip(&basis) {
            for (state, basis_value) in state.iter_mut().zip(vector) {
                *state += *basis_value * *coefficient;
            }
        }
        let norm = vector_norm(&state);
        if !norm.is_finite() || norm <= 0.0 {
            return Err(MagnetismError::NumericalFailure(
                "Ritz vector had invalid norm",
            ));
        }
        state.iter_mut().for_each(|value| *value = *value / norm);
        let residual_norm = tail_beta * coefficients.last().copied().unwrap_or(0.0).abs();
        Ok(GroundState {
            energy,
            state,
            residual_norm,
            iterations: diagonal.len(),
            converged: closed || residual_norm <= config.tolerance,
        })
    }
}
