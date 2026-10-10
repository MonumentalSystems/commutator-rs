use core::f64::consts::PI;

use crate::{Complex64, DenseMatrix, EmbeddingError, Result};

pub(crate) fn check_square(matrix: &DenseMatrix, size: usize, name: &'static str) -> Result<()> {
    if matrix.rows() == size && matrix.columns() == size {
        Ok(())
    } else {
        Err(EmbeddingError::Dimensions {
            name,
            expected: (size, size),
            actual: (matrix.rows(), matrix.columns()),
        })
    }
}

pub(crate) fn matrix_add(left: &DenseMatrix, right: &DenseMatrix) -> Result<DenseMatrix> {
    check_square(right, left.rows(), "matrix sum")?;
    if left.rows() != left.columns() {
        return Err(EmbeddingError::Dimensions {
            name: "matrix sum",
            expected: (left.rows(), left.rows()),
            actual: (left.rows(), left.columns()),
        });
    }
    DenseMatrix::try_new(
        left.rows(),
        left.columns(),
        left.as_slice()
            .iter()
            .zip(right.as_slice())
            .map(|(a, b)| *a + *b)
            .collect(),
    )
    .map_err(Into::into)
}

pub(crate) fn matrix_scale(matrix: &DenseMatrix, factor: f64) -> Result<DenseMatrix> {
    if !factor.is_finite() {
        return Err(EmbeddingError::NonFinite("matrix scale"));
    }
    DenseMatrix::try_new(
        matrix.rows(),
        matrix.columns(),
        matrix
            .as_slice()
            .iter()
            .map(|value| *value * factor)
            .collect(),
    )
    .map_err(Into::into)
}

pub(crate) fn matrix_max_difference(left: &DenseMatrix, right: &DenseMatrix) -> Result<f64> {
    if left.rows() != right.rows() || left.columns() != right.columns() {
        return Err(EmbeddingError::Dimensions {
            name: "matrix difference",
            expected: (left.rows(), left.columns()),
            actual: (right.rows(), right.columns()),
        });
    }
    Ok(left
        .as_slice()
        .iter()
        .zip(right.as_slice())
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0_f64, f64::max))
}

pub(crate) fn shifted_identity_minus(
    frequency: Complex64,
    chemical_potential: f64,
    terms: &[&DenseMatrix],
) -> Result<DenseMatrix> {
    if !frequency.re.is_finite() || !frequency.im.is_finite() || !chemical_potential.is_finite() {
        return Err(EmbeddingError::NonFinite(
            "Dyson frequency or chemical potential",
        ));
    }
    let size = terms
        .first()
        .ok_or(EmbeddingError::Empty("Dyson matrix terms"))?
        .rows();
    let mut values = vec![Complex64::new(0.0, 0.0); size * size];
    for row in 0..size {
        values[row * size + row] = frequency + chemical_potential;
    }
    for term in terms {
        check_square(term, size, "Dyson matrix term")?;
        for (out, value) in values.iter_mut().zip(term.as_slice()) {
            *out -= *value;
        }
    }
    DenseMatrix::try_new(size, size, values).map_err(Into::into)
}

pub(crate) fn validate_frequencies(frequencies: &[Complex64]) -> Result<()> {
    if frequencies.is_empty() {
        return Err(EmbeddingError::Empty("frequency grid"));
    }
    let mut previous = None;
    for frequency in frequencies {
        if !frequency.re.is_finite()
            || !frequency.im.is_finite()
            || frequency.im <= 0.0
            || previous.is_some_and(|value| frequency.re <= value)
        {
            return Err(EmbeddingError::InvalidFrequencyGrid);
        }
        previous = Some(frequency.re);
    }
    Ok(())
}

/// Validate `-Im_H matrix >= 0` by Hermitian principal-minor LDL elimination.
pub(crate) fn validate_retarded_causality(
    matrices: &[DenseMatrix],
    name: &'static str,
    tolerance: f64,
) -> Result<()> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(EmbeddingError::InvalidValue("causality tolerance"));
    }
    for (index, matrix) in matrices.iter().enumerate() {
        let size = matrix.rows();
        check_square(matrix, size, name)?;
        let mut a = vec![Complex64::new(0.0, 0.0); size * size];
        for row in 0..size {
            for column in 0..size {
                let value = matrix.get(row, column).expect("checked shape");
                let adjoint = matrix.get(column, row).expect("checked shape").conj();
                a[row * size + column] = (adjoint - value) / Complex64::new(0.0, 2.0);
            }
        }
        let scale = a.iter().map(|value| value.norm()).fold(0.0_f64, f64::max);
        let threshold = tolerance * scale.max(1.0);
        let mut lower = vec![Complex64::new(0.0, 0.0); size * size];
        let mut diagonal = vec![0.0; size];
        for column in 0..size {
            let correction: f64 = (0..column)
                .map(|k| lower[column * size + k].norm_sqr() * diagonal[k])
                .sum();
            let pivot = a[column * size + column] - correction;
            if pivot.im.abs() > threshold || pivot.re < -threshold {
                return Err(EmbeddingError::NonCausal {
                    name,
                    index,
                    violation: (-pivot.re).max(pivot.im.abs()),
                });
            }
            diagonal[column] = pivot.re.max(0.0);
            for row in column + 1..size {
                let correction: Complex64 = (0..column)
                    .map(|k| lower[row * size + k] * lower[column * size + k].conj() * diagonal[k])
                    .sum();
                let residual = a[row * size + column] - correction;
                if diagonal[column] <= threshold {
                    if residual.norm() > threshold {
                        return Err(EmbeddingError::NonCausal {
                            name,
                            index,
                            violation: residual.norm(),
                        });
                    }
                } else {
                    lower[row * size + column] = residual / diagonal[column];
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn negated_log_determinant(matrix: &DenseMatrix) -> Result<Complex64> {
    let size = matrix.rows();
    check_square(matrix, size, "log determinant")?;
    let mut values: Vec<_> = matrix.as_slice().iter().map(|value| -*value).collect();
    let mut phase = 0.0;
    let mut log_norm = 0.0;
    let scale = values
        .iter()
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return Err(EmbeddingError::SingularMatrix);
    }
    let threshold = 64.0 * f64::EPSILON * size as f64 * scale;
    for column in 0..size {
        let (pivot_row, pivot_norm) = (column..size)
            .map(|row| (row, values[row * size + column].norm()))
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .expect("nonempty pivot range");
        if pivot_norm <= threshold {
            return Err(EmbeddingError::SingularMatrix);
        }
        if pivot_row != column {
            for entry in 0..size {
                values.swap(column * size + entry, pivot_row * size + entry);
            }
            phase += PI;
        }
        let pivot = values[column * size + column];
        log_norm += pivot.norm().ln();
        phase += pivot.arg();
        for row in column + 1..size {
            let factor = values[row * size + column] / pivot;
            for entry in column + 1..size {
                let upper = values[column * size + entry];
                values[row * size + entry] -= factor * upper;
            }
        }
    }
    Ok(Complex64::new(log_norm, principal_phase(phase)))
}

pub(crate) fn unwrap_phase(principal: f64, previous: Option<f64>) -> f64 {
    let Some(previous) = previous else {
        return principal;
    };
    let turns = ((previous - principal) / (2.0 * PI)).round();
    principal + turns * 2.0 * PI
}

fn principal_phase(phase: f64) -> f64 {
    (phase + PI).rem_euclid(2.0 * PI) - PI
}
