use cluster_green::{Complex64, DenseMatrix};

use crate::{Result, TransportError};

/// One evaluated two-terminal coherent-transport point.
#[derive(Debug, Clone, PartialEq)]
pub struct TransmissionPoint {
    /// Energy used to evaluate the Green function and self-energies.
    pub energy: f64,
    /// Retarded device Green function.
    pub green_retarded: DenseMatrix,
    /// Left lead broadening matrix.
    pub gamma_left: DenseMatrix,
    /// Right lead broadening matrix.
    pub gamma_right: DenseMatrix,
    /// Dimensionless Caroli transmission.
    pub transmission: f64,
}

fn component_max(value: Complex64) -> f64 {
    value.re.abs().max(value.im.abs())
}

fn validate_tolerance(tolerance: f64) -> Result<()> {
    if tolerance.is_finite() && tolerance > 0.0 {
        Ok(())
    } else {
        Err(TransportError::InvalidTolerance)
    }
}

fn require_square(matrix: &DenseMatrix, name: &'static str) -> Result<usize> {
    if matrix.rows() == matrix.columns() {
        Ok(matrix.rows())
    } else {
        Err(TransportError::Dimensions {
            name,
            expected: (matrix.rows(), matrix.rows()),
            actual: (matrix.rows(), matrix.columns()),
        })
    }
}

fn require_size(matrix: &DenseMatrix, size: usize, name: &'static str) -> Result<()> {
    if matrix.rows() == size && matrix.columns() == size {
        Ok(())
    } else {
        Err(TransportError::Dimensions {
            name,
            expected: (size, size),
            actual: (matrix.rows(), matrix.columns()),
        })
    }
}

fn psd_minimum_pivot(matrix: &DenseMatrix, tolerance: f64) -> Result<f64> {
    validate_tolerance(tolerance)?;
    let size = require_square(matrix, "positive-semidefinite matrix")?;
    if !matrix.is_hermitian(tolerance)? {
        return Err(TransportError::NonHermitian("broadening matrix"));
    }
    let scale = matrix
        .as_slice()
        .iter()
        .fold(0.0_f64, |scale, value| scale.max(component_max(*value)));
    let threshold = tolerance * scale;
    let mut lower = vec![Complex64::new(0.0, 0.0); size * size];
    let mut diagonal = vec![0.0; size];
    let mut minimum = f64::INFINITY;
    for column in 0..size {
        let correction: f64 = (0..column)
            .map(|k| lower[column * size + k].norm_sqr() * diagonal[k])
            .sum();
        let value =
            matrix.get(column, column).expect("checked dimensions") - Complex64::from(correction);
        if value.im.abs() > threshold {
            return Err(TransportError::NonHermitian("broadening matrix"));
        }
        let pivot = value.re;
        minimum = minimum.min(pivot);
        if pivot < -threshold {
            return Ok(minimum);
        }
        if pivot.abs() <= threshold {
            diagonal[column] = 0.0;
            for row in column + 1..size {
                let correction: Complex64 = (0..column)
                    .map(|k| lower[row * size + k] * lower[column * size + k].conj() * diagonal[k])
                    .sum();
                let residual = matrix.get(row, column).expect("checked dimensions") - correction;
                if component_max(residual) > threshold {
                    return Ok(minimum.min(-2.0 * threshold));
                }
            }
        } else {
            diagonal[column] = pivot;
            lower[column * size + column] = Complex64::new(1.0, 0.0);
            for row in column + 1..size {
                let correction: Complex64 = (0..column)
                    .map(|k| lower[row * size + k] * lower[column * size + k].conj() * diagonal[k])
                    .sum();
                lower[row * size + column] =
                    (matrix.get(row, column).expect("checked dimensions") - correction) / pivot;
            }
        }
    }
    Ok(minimum)
}

fn add_scaled_identity_minus_terms(
    energy: f64,
    eta: f64,
    hamiltonian: &DenseMatrix,
    self_energies: &[&DenseMatrix],
) -> Result<DenseMatrix> {
    let size = hamiltonian.rows();
    let mut values = hamiltonian
        .as_slice()
        .iter()
        .map(|value| -*value)
        .collect::<Vec<_>>();
    for index in 0..size {
        values[index * size + index] += Complex64::new(energy, eta);
    }
    for self_energy in self_energies {
        require_size(self_energy, size, "lead self-energy")?;
        for (value, contribution) in values.iter_mut().zip(self_energy.as_slice()) {
            *value -= *contribution;
        }
    }
    Ok(DenseMatrix::try_new(size, size, values)?)
}

/// Compute `Gamma = i(Sigma^R - Sigma^{R†})` and validate causality.
pub fn broadening(self_energy: &DenseMatrix, tolerance: f64) -> Result<DenseMatrix> {
    validate_tolerance(tolerance)?;
    let size = require_square(self_energy, "lead self-energy")?;
    let adjoint = self_energy.adjoint();
    let values = self_energy
        .as_slice()
        .iter()
        .zip(adjoint.as_slice())
        .map(|(sigma, sigma_adjoint)| Complex64::new(0.0, 1.0) * (*sigma - *sigma_adjoint))
        .collect();
    let gamma = DenseMatrix::try_new(size, size, values)?;
    let minimum = psd_minimum_pivot(&gamma, tolerance)?;
    let scale = gamma
        .as_slice()
        .iter()
        .fold(0.0_f64, |scale, value| scale.max(component_max(*value)));
    if minimum < -tolerance * scale {
        Err(TransportError::NonCausalSelfEnergy {
            minimum_pivot: minimum,
        })
    } else {
        Ok(gamma)
    }
}

/// Compute a checked retarded device Green function.
///
/// The result is `[(E+i eta)I - H - sum_l Sigma_l^R]^-1`. Each self-energy
/// is checked for retarded causality through its broadening matrix.
pub fn retarded_device_green(
    energy: f64,
    eta: f64,
    hamiltonian: &DenseMatrix,
    self_energies: &[&DenseMatrix],
    tolerance: f64,
) -> Result<DenseMatrix> {
    validate_tolerance(tolerance)?;
    if !energy.is_finite() {
        return Err(TransportError::NonFinite("energy"));
    }
    if !eta.is_finite() || eta <= 0.0 {
        return Err(TransportError::NonPositiveRegulator);
    }
    require_square(hamiltonian, "device Hamiltonian")?;
    if !hamiltonian.is_hermitian(tolerance)? {
        return Err(TransportError::NonHermitian("device Hamiltonian"));
    }
    for self_energy in self_energies {
        require_size(self_energy, hamiltonian.rows(), "lead self-energy")?;
        broadening(self_energy, tolerance)?;
    }
    Ok(add_scaled_identity_minus_terms(energy, eta, hamiltonian, self_energies)?.inverse()?)
}

/// Evaluate the Caroli transmission trace.
///
/// Computes `Tr[Gamma_L G^R Gamma_R G^A]`. Both broadenings must be
/// Hermitian positive semidefinite. A real result slightly below zero is
/// clamped to zero only when it lies within the scale-aware tolerance.
pub fn caroli_transmission(
    gamma_left: &DenseMatrix,
    gamma_right: &DenseMatrix,
    green_retarded: &DenseMatrix,
    tolerance: f64,
) -> Result<f64> {
    validate_tolerance(tolerance)?;
    let size = require_square(green_retarded, "retarded Green function")?;
    require_size(gamma_left, size, "left broadening")?;
    require_size(gamma_right, size, "right broadening")?;
    for gamma in [gamma_left, gamma_right] {
        let minimum = psd_minimum_pivot(gamma, tolerance)?;
        let scale = gamma
            .as_slice()
            .iter()
            .fold(0.0_f64, |scale, value| scale.max(component_max(*value)));
        if minimum < -tolerance * scale {
            return Err(TransportError::NonCausalSelfEnergy {
                minimum_pivot: minimum,
            });
        }
    }
    let product = gamma_left
        .multiply(green_retarded)?
        .multiply(gamma_right)?
        .multiply(&green_retarded.adjoint())?;
    let trace = product.trace()?;
    let scale = 1.0 + trace.re.abs();
    if trace.im.abs() > tolerance * scale {
        return Err(TransportError::ComplexTransmission {
            imaginary: trace.im,
        });
    }
    if trace.re < -tolerance * scale {
        return Err(TransportError::NegativeTransmission { value: trace.re });
    }
    Ok(trace.re.max(0.0))
}

/// Evaluate a complete two-terminal coherent-transport point.
pub fn two_terminal_transmission(
    energy: f64,
    eta: f64,
    hamiltonian: &DenseMatrix,
    left_self_energy: &DenseMatrix,
    right_self_energy: &DenseMatrix,
    tolerance: f64,
) -> Result<TransmissionPoint> {
    let gamma_left = broadening(left_self_energy, tolerance)?;
    let gamma_right = broadening(right_self_energy, tolerance)?;
    let green_retarded = retarded_device_green(
        energy,
        eta,
        hamiltonian,
        &[left_self_energy, right_self_energy],
        tolerance,
    )?;
    let transmission = caroli_transmission(&gamma_left, &gamma_right, &green_retarded, tolerance)?;
    Ok(TransmissionPoint {
        energy,
        green_retarded,
        gamma_left,
        gamma_right,
        transmission,
    })
}
