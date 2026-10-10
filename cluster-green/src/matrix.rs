use crate::{Complex64, GreenError, Result};

/// A small row-major dense complex matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseMatrix {
    rows: usize,
    columns: usize,
    values: Vec<Complex64>,
}

impl DenseMatrix {
    /// Construct a matrix after checking shape and finiteness.
    pub fn try_new(rows: usize, columns: usize, values: Vec<Complex64>) -> Result<Self> {
        if rows == 0 || columns == 0 {
            return Err(GreenError::Empty("matrix dimensions"));
        }
        let expected = rows.checked_mul(columns).ok_or(GreenError::SizeOverflow)?;
        if values.len() != expected {
            return Err(GreenError::Shape {
                name: "matrix values",
                expected,
                actual: values.len(),
            });
        }
        if !values
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
        {
            return Err(GreenError::NonFinite("matrix values"));
        }
        Ok(Self {
            rows,
            columns,
            values,
        })
    }

    /// Construct a checked `1 × 1` matrix.
    pub fn from_scalar(value: Complex64) -> Result<Self> {
        Self::try_new(1, 1, vec![value])
    }

    /// Construct an `N × N` identity matrix.
    pub fn identity(size: usize) -> Result<Self> {
        if size == 0 {
            return Err(GreenError::Empty("identity matrix"));
        }
        let length = size.checked_mul(size).ok_or(GreenError::SizeOverflow)?;
        let mut values = vec![Complex64::new(0.0, 0.0); length];
        for index in 0..size {
            values[index * size + index] = Complex64::new(1.0, 0.0);
        }
        Self::try_new(size, size, values)
    }

    /// Return the row count.
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Return the column count.
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// Return the row-major values.
    pub fn as_slice(&self) -> &[Complex64] {
        &self.values
    }

    /// Return an element, or `None` when either index is out of bounds.
    pub fn get(&self, row: usize, column: usize) -> Option<Complex64> {
        if row < self.rows && column < self.columns {
            Some(self.values[row * self.columns + column])
        } else {
            None
        }
    }

    /// Return the matrix trace.
    pub fn trace(&self) -> Result<Complex64> {
        self.require_square("trace input")?;
        Ok((0..self.rows)
            .map(|index| self.values[index * self.columns + index])
            .sum())
    }

    /// Return the conjugate transpose.
    pub fn adjoint(&self) -> Self {
        let mut values = vec![Complex64::new(0.0, 0.0); self.values.len()];
        for row in 0..self.rows {
            for column in 0..self.columns {
                values[column * self.rows + row] = self.values[row * self.columns + column].conj();
            }
        }
        Self {
            rows: self.columns,
            columns: self.rows,
            values,
        }
    }

    /// Return `self - rhs` after checking shape.
    pub fn subtract(&self, rhs: &Self) -> Result<Self> {
        self.require_same_shape(rhs, "matrix subtraction")?;
        Self::try_new(
            self.rows,
            self.columns,
            self.values
                .iter()
                .zip(&rhs.values)
                .map(|(left, right)| *left - *right)
                .collect(),
        )
    }

    /// Return the matrix product `self * rhs`.
    pub fn multiply(&self, rhs: &Self) -> Result<Self> {
        if self.columns != rhs.rows {
            return Err(GreenError::Shape {
                name: "matrix-product inner dimension",
                expected: self.columns,
                actual: rhs.rows,
            });
        }
        let length = self
            .rows
            .checked_mul(rhs.columns)
            .ok_or(GreenError::SizeOverflow)?;
        let mut values = vec![Complex64::new(0.0, 0.0); length];
        for row in 0..self.rows {
            for column in 0..rhs.columns {
                values[row * rhs.columns + column] = (0..self.columns)
                    .map(|inner| {
                        self.values[row * self.columns + inner]
                            * rhs.values[inner * rhs.columns + column]
                    })
                    .sum();
            }
        }
        Self::try_new(self.rows, rhs.columns, values)
    }

    /// Test Hermiticity with an absolute, scale-aware tolerance.
    pub fn is_hermitian(&self, tolerance: f64) -> Result<bool> {
        validate_tolerance(tolerance)?;
        if self.rows != self.columns {
            return Ok(false);
        }
        let scale = self
            .values
            .iter()
            .fold(1.0_f64, |scale, value| scale.max(value.norm()));
        let threshold = tolerance * scale;
        for row in 0..self.rows {
            for column in row..self.columns {
                let difference = self.values[row * self.columns + column]
                    - self.values[column * self.columns + row].conj();
                if difference.norm() > threshold {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Invert a square matrix with Gauss–Jordan elimination and partial pivoting.
    pub fn inverse(&self) -> Result<Self> {
        self.require_square("matrix inverse")?;
        let size = self.rows;
        let mut left = self.values.clone();
        let mut right = Self::identity(size)?.values;
        let scale = left
            .iter()
            .fold(0.0_f64, |scale, value| scale.max(value.norm()));
        let threshold = 64.0 * f64::EPSILON * scale.max(1.0) * size as f64;

        for pivot in 0..size {
            let mut pivot_row = pivot;
            let mut pivot_norm = left[pivot * size + pivot].norm();
            for row in pivot + 1..size {
                let candidate = left[row * size + pivot].norm();
                if candidate > pivot_norm {
                    pivot_norm = candidate;
                    pivot_row = row;
                }
            }
            if pivot_norm <= threshold {
                return Err(GreenError::SingularMatrix { pivot });
            }
            if pivot_row != pivot {
                for column in 0..size {
                    left.swap(pivot * size + column, pivot_row * size + column);
                    right.swap(pivot * size + column, pivot_row * size + column);
                }
            }
            let pivot_value = left[pivot * size + pivot];
            for column in 0..size {
                left[pivot * size + column] /= pivot_value;
                right[pivot * size + column] /= pivot_value;
            }
            for row in 0..size {
                if row == pivot {
                    continue;
                }
                let factor = left[row * size + pivot];
                if factor == Complex64::new(0.0, 0.0) {
                    continue;
                }
                for column in 0..size {
                    let pivot_left = left[pivot * size + column];
                    let pivot_right = right[pivot * size + column];
                    left[row * size + column] -= factor * pivot_left;
                    right[row * size + column] -= factor * pivot_right;
                }
            }
        }
        Self::try_new(size, size, right)
    }

    pub(crate) fn require_square(&self, name: &'static str) -> Result<()> {
        if self.rows == self.columns {
            Ok(())
        } else {
            Err(GreenError::Dimensions {
                name,
                expected: (self.rows, self.rows),
                actual: (self.rows, self.columns),
            })
        }
    }

    pub(crate) fn require_dimensions(
        &self,
        rows: usize,
        columns: usize,
        name: &'static str,
    ) -> Result<()> {
        if self.rows == rows && self.columns == columns {
            Ok(())
        } else {
            Err(GreenError::Dimensions {
                name,
                expected: (rows, columns),
                actual: (self.rows, self.columns),
            })
        }
    }

    fn require_same_shape(&self, rhs: &Self, name: &'static str) -> Result<()> {
        if self.rows == rhs.rows && self.columns == rhs.columns {
            Ok(())
        } else {
            Err(GreenError::Dimensions {
                name,
                expected: (self.rows, self.columns),
                actual: (rhs.rows, rhs.columns),
            })
        }
    }
}

pub(crate) fn validate_tolerance(tolerance: f64) -> Result<()> {
    if tolerance.is_finite() && tolerance > 0.0 {
        Ok(())
    } else {
        Err(GreenError::InvalidTolerance)
    }
}
