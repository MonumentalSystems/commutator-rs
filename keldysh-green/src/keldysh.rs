use crate::matrix_ops::{add, max_difference, scale, validate_square, zero};
use crate::{DenseMatrix, KeldyshError, RealTimeGrid, Result};

/// A square matrix-valued function of two discrete time arguments.
#[derive(Debug, Clone, PartialEq)]
pub struct TwoTimeMatrix {
    time_points: usize,
    orbitals: usize,
    values: Vec<DenseMatrix>,
}

impl TwoTimeMatrix {
    /// Construct a checked two-time function in row-major `(t, t')` order.
    pub fn try_new(time_points: usize, orbitals: usize, values: Vec<DenseMatrix>) -> Result<Self> {
        if time_points < 2 {
            return Err(KeldyshError::Empty("two-time grid"));
        }
        if orbitals == 0 {
            return Err(KeldyshError::Empty("orbital space"));
        }
        let expected = time_points
            .checked_mul(time_points)
            .ok_or(KeldyshError::SizeOverflow)?;
        if values.len() != expected {
            return Err(KeldyshError::Shape {
                name: "two-time matrix values",
                expected,
                actual: values.len(),
            });
        }
        for matrix in &values {
            validate_square(matrix, orbitals, "two-time matrix value")?;
        }
        Ok(Self {
            time_points,
            orbitals,
            values,
        })
    }

    /// Construct a two-time function by evaluating a fallible callback.
    pub fn try_from_fn<F>(time_points: usize, orbitals: usize, mut function: F) -> Result<Self>
    where
        F: FnMut(usize, usize) -> Result<DenseMatrix>,
    {
        let length = time_points
            .checked_mul(time_points)
            .ok_or(KeldyshError::SizeOverflow)?;
        let mut values = Vec::with_capacity(length);
        for first in 0..time_points {
            for second in 0..time_points {
                values.push(function(first, second)?);
            }
        }
        Self::try_new(time_points, orbitals, values)
    }

    /// Return the number of time points per argument.
    pub const fn time_points(&self) -> usize {
        self.time_points
    }

    /// Return the one-particle orbital dimension.
    pub const fn orbitals(&self) -> usize {
        self.orbitals
    }

    /// Return the matrix at `(t_i, t_j)`.
    pub fn get(&self, first: usize, second: usize) -> Result<&DenseMatrix> {
        if first >= self.time_points {
            return Err(KeldyshError::IndexOutOfBounds {
                name: "first time argument",
                index: first,
                length: self.time_points,
            });
        }
        if second >= self.time_points {
            return Err(KeldyshError::IndexOutOfBounds {
                name: "second time argument",
                index: second,
                length: self.time_points,
            });
        }
        Ok(&self.values[first * self.time_points + second])
    }

    fn compatible(&self, other: &Self) -> bool {
        self.time_points == other.time_points && self.orbitals == other.orbitals
    }
}

/// Maximum residuals of the fermionic Keldysh consistency identities.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConsistencyReport {
    /// Largest violation of retarded support (`t < t'`).
    pub retarded_causality: f64,
    /// Largest violation of advanced support (`t > t'`).
    pub advanced_causality: f64,
    /// Largest residual of `G^A(t,t') = G^R(t',t)†`.
    pub advanced_adjoint: f64,
    /// Largest residual of `G^<(t,t') = -G^<(t',t)†`.
    pub lesser_antihermiticity: f64,
    /// Largest residual of `G^>(t,t') = -G^>(t',t)†`.
    pub greater_antihermiticity: f64,
    /// Largest residual of `G^R-G^A = G^>-G^<`.
    pub spectral_identity: f64,
}

impl ConsistencyReport {
    /// Return whether all residuals are at most an absolute tolerance.
    pub fn is_consistent(&self, tolerance: f64) -> Result<bool> {
        validate_tolerance(tolerance)?;
        Ok([
            self.retarded_causality,
            self.advanced_causality,
            self.advanced_adjoint,
            self.lesser_antihermiticity,
            self.greater_antihermiticity,
            self.spectral_identity,
        ]
        .into_iter()
        .all(|residual| residual <= tolerance))
    }
}

/// Retarded, advanced, lesser, and greater Green-function components.
#[derive(Debug, Clone, PartialEq)]
pub struct KeldyshGreen {
    grid: RealTimeGrid,
    retarded: TwoTimeMatrix,
    advanced: TwoTimeMatrix,
    lesser: TwoTimeMatrix,
    greater: TwoTimeMatrix,
}

impl KeldyshGreen {
    /// Construct checked, shape-compatible components.
    ///
    /// Algebraic identities are intentionally inspected separately with
    /// [`Self::consistency_report`], allowing partially converged solvers to be
    /// represented and diagnosed.
    pub fn try_new(
        grid: RealTimeGrid,
        retarded: TwoTimeMatrix,
        advanced: TwoTimeMatrix,
        lesser: TwoTimeMatrix,
        greater: TwoTimeMatrix,
    ) -> Result<Self> {
        if retarded.time_points != grid.len() {
            return Err(KeldyshError::Incompatible("time grid"));
        }
        if !retarded.compatible(&advanced)
            || !retarded.compatible(&lesser)
            || !retarded.compatible(&greater)
        {
            return Err(KeldyshError::Incompatible("Keldysh components"));
        }
        Ok(Self {
            grid,
            retarded,
            advanced,
            lesser,
            greater,
        })
    }

    /// Return the real-time grid.
    pub const fn grid(&self) -> &RealTimeGrid {
        &self.grid
    }

    /// Return the retarded component.
    pub const fn retarded(&self) -> &TwoTimeMatrix {
        &self.retarded
    }

    /// Return the advanced component.
    pub const fn advanced(&self) -> &TwoTimeMatrix {
        &self.advanced
    }

    /// Return the lesser component.
    pub const fn lesser(&self) -> &TwoTimeMatrix {
        &self.lesser
    }

    /// Return the greater component.
    pub const fn greater(&self) -> &TwoTimeMatrix {
        &self.greater
    }

    /// Measure causality, adjoint, anti-Hermiticity, and spectral identities.
    pub fn consistency_report(&self) -> Result<ConsistencyReport> {
        let mut report = ConsistencyReport {
            retarded_causality: 0.0,
            advanced_causality: 0.0,
            advanced_adjoint: 0.0,
            lesser_antihermiticity: 0.0,
            greater_antihermiticity: 0.0,
            spectral_identity: 0.0,
        };
        for first in 0..self.grid.len() {
            for second in 0..self.grid.len() {
                let retarded = self.retarded.get(first, second)?;
                let advanced = self.advanced.get(first, second)?;
                let lesser = self.lesser.get(first, second)?;
                let greater = self.greater.get(first, second)?;
                if first < second {
                    report.retarded_causality =
                        report.retarded_causality.max(matrix_norm_max(retarded));
                }
                if first > second {
                    report.advanced_causality =
                        report.advanced_causality.max(matrix_norm_max(advanced));
                }
                report.advanced_adjoint = report.advanced_adjoint.max(max_difference(
                    advanced,
                    &self.retarded.get(second, first)?.adjoint(),
                )?);
                report.lesser_antihermiticity = report.lesser_antihermiticity.max(max_difference(
                    lesser,
                    &scale(&self.lesser.get(second, first)?.adjoint(), (-1.0).into())?,
                )?);
                report.greater_antihermiticity =
                    report.greater_antihermiticity.max(max_difference(
                        greater,
                        &scale(&self.greater.get(second, first)?.adjoint(), (-1.0).into())?,
                    )?);
                let causal_difference = retarded.subtract(advanced)?;
                let correlation_difference = greater.subtract(lesser)?;
                report.spectral_identity = report
                    .spectral_identity
                    .max(max_difference(&causal_difference, &correlation_difference)?);
            }
        }
        Ok(report)
    }
}

/// Convolve two two-time matrices over the complete stored time interval.
///
/// Computes `(A ∘ B)(t,t') = integral ds A(t,s) B(s,t')` with the grid's
/// composite-trapezoid weights. Complexity is `O(N_t^3 n^3)` for `n` orbitals.
pub fn time_convolution(
    grid: &RealTimeGrid,
    left: &TwoTimeMatrix,
    right: &TwoTimeMatrix,
) -> Result<TwoTimeMatrix> {
    validate_inputs(grid, left, right)?;
    TwoTimeMatrix::try_from_fn(grid.len(), left.orbitals, |first, second| {
        weighted_product_sum(grid, left, right, first, second, 0, grid.len() - 1, false)
    })
}

/// Causal Volterra convolution over `t' <= s <= t`.
///
/// Entries with `t < t'` are exactly zero. Truncated intervals use local
/// composite-trapezoid endpoint weights, including on nonuniform grids.
pub fn causal_volterra_convolution(
    grid: &RealTimeGrid,
    left: &TwoTimeMatrix,
    right: &TwoTimeMatrix,
) -> Result<TwoTimeMatrix> {
    validate_inputs(grid, left, right)?;
    TwoTimeMatrix::try_from_fn(grid.len(), left.orbitals, |first, second| {
        if first < second {
            zero(left.orbitals)
        } else {
            weighted_product_sum(grid, left, right, first, second, second, first, true)
        }
    })
}

/// Apply the real-time Langreth rules to a contour product `C = A ∘ B`.
///
/// The rules used are `C^R=A^R∘B^R`, `C^A=A^A∘B^A`, and
/// `C^{</>}=A^R∘B^{</>}+A^{</>}∘B^A`. Initial-correlation terms from an
/// imaginary contour spur are outside this real-time-only reference layer.
pub fn langreth_product(left: &KeldyshGreen, right: &KeldyshGreen) -> Result<KeldyshGreen> {
    if left.grid != right.grid || !left.retarded.compatible(&right.retarded) {
        return Err(KeldyshError::Incompatible("Langreth operands"));
    }
    let retarded = causal_volterra_convolution(&left.grid, &left.retarded, &right.retarded)?;
    let advanced = anti_causal_convolution(&left.grid, &left.advanced, &right.advanced)?;
    let lesser = add_functions(
        &time_convolution(&left.grid, &left.retarded, &right.lesser)?,
        &time_convolution(&left.grid, &left.lesser, &right.advanced)?,
    )?;
    let greater = add_functions(
        &time_convolution(&left.grid, &left.retarded, &right.greater)?,
        &time_convolution(&left.grid, &left.greater, &right.advanced)?,
    )?;
    KeldyshGreen::try_new(left.grid.clone(), retarded, advanced, lesser, greater)
}

/// Solve the retarded Dyson equation on a real-time grid.
///
/// Solves `G^R = G0^R + G0^R ∘ Sigma^R ∘ G^R` by first forming the causal
/// kernel `K=G0^R∘Sigma^R`, then forward-substituting the Volterra equation.
/// Composite-trapezoid quadrature is used on every truncated interval. The
/// dense reference algorithm scales as `O(N_t^3 n^3)` and stores `O(N_t^2 n^2)`
/// complex numbers.
pub fn retarded_dyson(
    grid: &RealTimeGrid,
    free: &TwoTimeMatrix,
    self_energy: &TwoTimeMatrix,
) -> Result<TwoTimeMatrix> {
    validate_inputs(grid, free, self_energy)?;
    let kernel = causal_volterra_convolution(grid, free, self_energy)?;
    let mut rows: Vec<Vec<DenseMatrix>> = (0..grid.len())
        .map(|_| (0..grid.len()).map(|_| zero(free.orbitals)).collect())
        .collect::<Result<_>>()?;

    for first in 0..grid.len() {
        let (previous_rows, current_and_later) = rows.split_at_mut(first);
        let current_row = &mut current_and_later[0];
        for (second, slot) in current_row.iter_mut().enumerate().take(first + 1) {
            let mut value = free.get(first, second)?.clone();
            if first > second {
                for (intermediate, previous_row) in previous_rows.iter().enumerate().skip(second) {
                    let weight = grid.interval_weight(intermediate, second, first);
                    let product = kernel
                        .get(first, intermediate)?
                        .multiply(&previous_row[second])?;
                    value = add(&value, &scale(&product, weight.into())?)?;
                }
            }
            *slot = value;
        }
    }
    TwoTimeMatrix::try_from_fn(grid.len(), free.orbitals, |first, second| {
        Ok(rows[first][second].clone())
    })
}

fn anti_causal_convolution(
    grid: &RealTimeGrid,
    left: &TwoTimeMatrix,
    right: &TwoTimeMatrix,
) -> Result<TwoTimeMatrix> {
    validate_inputs(grid, left, right)?;
    TwoTimeMatrix::try_from_fn(grid.len(), left.orbitals, |first, second| {
        if first > second {
            zero(left.orbitals)
        } else {
            weighted_product_sum(grid, left, right, first, second, first, second, true)
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn weighted_product_sum(
    grid: &RealTimeGrid,
    left: &TwoTimeMatrix,
    right: &TwoTimeMatrix,
    first: usize,
    second: usize,
    start: usize,
    end: usize,
    local_weights: bool,
) -> Result<DenseMatrix> {
    let mut result = zero(left.orbitals)?;
    for intermediate in start..=end {
        let weight = if local_weights {
            grid.interval_weight(intermediate, start, end)
        } else {
            grid.weights()[intermediate]
        };
        let product = left
            .get(first, intermediate)?
            .multiply(right.get(intermediate, second)?)?;
        result = add(&result, &scale(&product, weight.into())?)?;
    }
    Ok(result)
}

fn add_functions(left: &TwoTimeMatrix, right: &TwoTimeMatrix) -> Result<TwoTimeMatrix> {
    if !left.compatible(right) {
        return Err(KeldyshError::Incompatible("two-time functions"));
    }
    TwoTimeMatrix::try_from_fn(left.time_points, left.orbitals, |first, second| {
        add(left.get(first, second)?, right.get(first, second)?)
    })
}

fn validate_inputs(grid: &RealTimeGrid, left: &TwoTimeMatrix, right: &TwoTimeMatrix) -> Result<()> {
    if !left.compatible(right) || left.time_points != grid.len() {
        Err(KeldyshError::Incompatible("convolution operands"))
    } else {
        Ok(())
    }
}

fn matrix_norm_max(matrix: &DenseMatrix) -> f64 {
    matrix
        .as_slice()
        .iter()
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max)
}

fn validate_tolerance(tolerance: f64) -> Result<()> {
    if tolerance.is_finite() && tolerance > 0.0 {
        Ok(())
    } else {
        Err(KeldyshError::InvalidTolerance)
    }
}
