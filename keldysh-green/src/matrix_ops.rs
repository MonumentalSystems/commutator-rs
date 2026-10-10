use crate::{Complex64, DenseMatrix, KeldyshError, Result};

pub(crate) fn zero(size: usize) -> Result<DenseMatrix> {
    let length = size.checked_mul(size).ok_or(KeldyshError::SizeOverflow)?;
    Ok(DenseMatrix::try_new(
        size,
        size,
        vec![Complex64::new(0.0, 0.0); length],
    )?)
}

pub(crate) fn validate_square(matrix: &DenseMatrix, size: usize, name: &'static str) -> Result<()> {
    if matrix.rows() == size && matrix.columns() == size {
        Ok(())
    } else {
        Err(KeldyshError::Dimensions {
            name,
            expected: (size, size),
            actual: (matrix.rows(), matrix.columns()),
        })
    }
}

pub(crate) fn add(left: &DenseMatrix, right: &DenseMatrix) -> Result<DenseMatrix> {
    validate_same(left, right)?;
    Ok(DenseMatrix::try_new(
        left.rows(),
        left.columns(),
        left.as_slice()
            .iter()
            .zip(right.as_slice())
            .map(|(a, b)| *a + *b)
            .collect(),
    )?)
}

pub(crate) fn scale(matrix: &DenseMatrix, factor: Complex64) -> Result<DenseMatrix> {
    Ok(DenseMatrix::try_new(
        matrix.rows(),
        matrix.columns(),
        matrix
            .as_slice()
            .iter()
            .map(|value| *value * factor)
            .collect(),
    )?)
}

pub(crate) fn max_difference(left: &DenseMatrix, right: &DenseMatrix) -> Result<f64> {
    validate_same(left, right)?;
    Ok(left
        .as_slice()
        .iter()
        .zip(right.as_slice())
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0_f64, f64::max))
}

fn validate_same(left: &DenseMatrix, right: &DenseMatrix) -> Result<()> {
    if left.rows() == right.rows() && left.columns() == right.columns() {
        Ok(())
    } else {
        Err(KeldyshError::Dimensions {
            name: "matrix operand",
            expected: (left.rows(), left.columns()),
            actual: (right.rows(), right.columns()),
        })
    }
}
