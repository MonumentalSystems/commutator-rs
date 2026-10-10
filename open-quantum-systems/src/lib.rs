//! Checked dense Lindblad master-equation reference dynamics.
//!
//! The implementation favors explicit invariants and auditable small-system
//! numerics over large-Hilbert-space performance.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use core::fmt;

pub use num_complex::Complex64;

/// Errors returned by checked open-system operations.
#[derive(Clone, Debug, PartialEq)]
pub enum QuantumError {
    /// A Hilbert-space dimension was zero or overflowed when squared.
    InvalidDimension,
    /// A row-major matrix has the wrong number of entries.
    Shape {
        /// Expected number of entries.
        expected: usize,
        /// Actual number of entries.
        actual: usize,
    },
    /// An input contained NaN or infinity.
    NonFinite(&'static str),
    /// A tolerance must be finite and strictly positive.
    InvalidTolerance,
    /// An operator expected to be Hermitian was not.
    NotHermitian {
        /// Largest absolute component of `A - A†`.
        residual: f64,
    },
    /// A density matrix did not have unit trace.
    TraceNotOne {
        /// Real trace component.
        real: f64,
        /// Imaginary trace component.
        imaginary: f64,
    },
    /// A density matrix failed the positive-semidefinite check.
    NotPositiveSemidefinite {
        /// Smallest Hermitian eigenvalue encountered.
        minimum_eigenvalue: f64,
    },
    /// Two operators have different Hilbert-space dimensions.
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },
    /// A basis-state index lies outside the Hilbert space.
    BasisIndex {
        /// Hilbert-space dimension.
        dimension: usize,
        /// Requested index.
        index: usize,
    },
    /// A collapse rate was negative or non-finite.
    InvalidRate,
    /// A time step was not finite and strictly positive.
    InvalidTimeStep,
    /// At least one integration step is required.
    ZeroSteps,
    /// Normalization failed because a numerical state had zero trace.
    ZeroTrace,
}

impl fmt::Display for QuantumError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for QuantumError {}

/// Result type for checked open-system operations.
pub type Result<T> = core::result::Result<T, QuantumError>;

fn validate_tolerance(tolerance: f64) -> Result<()> {
    if tolerance.is_finite() && tolerance > 0.0 {
        Ok(())
    } else {
        Err(QuantumError::InvalidTolerance)
    }
}

fn component_norm(value: Complex64) -> f64 {
    value.re.abs().max(value.im.abs())
}

fn zeroed_operator_values(length: usize) -> Result<Vec<Complex64>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(length)
        .map_err(|_| QuantumError::InvalidDimension)?;
    values.resize(length, Complex64::new(0.0, 0.0));
    Ok(values)
}

/// A checked square row-major complex operator.
#[derive(Clone, Debug, PartialEq)]
pub struct Operator {
    dimension: usize,
    values: Vec<Complex64>,
}

impl Operator {
    /// Constructs a square operator after checking shape and finiteness.
    pub fn try_new(dimension: usize, values: Vec<Complex64>) -> Result<Self> {
        if dimension == 0 {
            return Err(QuantumError::InvalidDimension);
        }
        let expected = dimension
            .checked_mul(dimension)
            .ok_or(QuantumError::InvalidDimension)?;
        if values.len() != expected {
            return Err(QuantumError::Shape {
                expected,
                actual: values.len(),
            });
        }
        if values
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(QuantumError::NonFinite("operator values"));
        }
        Ok(Self { dimension, values })
    }

    /// Constructs the identity operator.
    pub fn identity(dimension: usize) -> Result<Self> {
        let length = dimension
            .checked_mul(dimension)
            .ok_or(QuantumError::InvalidDimension)?;
        if dimension == 0 {
            return Err(QuantumError::InvalidDimension);
        }
        let mut values = zeroed_operator_values(length)?;
        for index in 0..dimension {
            values[index * dimension + index] = Complex64::new(1.0, 0.0);
        }
        Self::try_new(dimension, values)
    }

    /// Returns the Hilbert-space dimension.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the row-major values.
    #[must_use]
    pub fn as_slice(&self) -> &[Complex64] {
        &self.values
    }

    /// Returns one matrix element, or `None` for an out-of-range index.
    #[must_use]
    pub fn get(&self, row: usize, column: usize) -> Option<Complex64> {
        (row < self.dimension && column < self.dimension)
            .then(|| self.values[row * self.dimension + column])
    }

    /// Returns the operator trace.
    #[must_use]
    pub fn trace(&self) -> Complex64 {
        (0..self.dimension)
            .map(|index| self.values[index * self.dimension + index])
            .sum()
    }

    /// Returns the conjugate transpose.
    #[must_use]
    pub fn adjoint(&self) -> Self {
        let mut values = vec![Complex64::new(0.0, 0.0); self.values.len()];
        for row in 0..self.dimension {
            for column in 0..self.dimension {
                values[column * self.dimension + row] =
                    self.values[row * self.dimension + column].conj();
            }
        }
        Self {
            dimension: self.dimension,
            values,
        }
    }

    /// Returns the product `self * rhs`.
    pub fn multiply(&self, rhs: &Self) -> Result<Self> {
        self.require_dimension(rhs.dimension)?;
        let n = self.dimension;
        let mut values = vec![Complex64::new(0.0, 0.0); n * n];
        for row in 0..n {
            for column in 0..n {
                values[row * n + column] = (0..n)
                    .map(|inner| self.values[row * n + inner] * rhs.values[inner * n + column])
                    .sum();
            }
        }
        Self::try_new(n, values)
    }

    /// Returns the commutator `[self, rhs]`.
    pub fn commutator(&self, rhs: &Self) -> Result<Self> {
        let left = self.multiply(rhs)?;
        let right = rhs.multiply(self)?;
        left.add_scaled(&right, Complex64::new(-1.0, 0.0))
    }

    /// Tests Hermiticity with a scale-aware absolute tolerance.
    pub fn is_hermitian(&self, tolerance: f64) -> Result<bool> {
        validate_tolerance(tolerance)?;
        Ok(self.hermiticity_residual() <= tolerance * self.scale().max(1.0))
    }

    fn scale(&self) -> f64 {
        self.values
            .iter()
            .fold(0.0_f64, |scale, value| scale.max(component_norm(*value)))
    }

    fn hermiticity_residual(&self) -> f64 {
        let mut residual = 0.0_f64;
        for row in 0..self.dimension {
            for column in row..self.dimension {
                residual = residual.max(component_norm(
                    self.values[row * self.dimension + column]
                        - self.values[column * self.dimension + row].conj(),
                ));
            }
        }
        residual
    }

    fn require_dimension(&self, actual: usize) -> Result<()> {
        if self.dimension == actual {
            Ok(())
        } else {
            Err(QuantumError::DimensionMismatch {
                expected: self.dimension,
                actual,
            })
        }
    }

    fn add_scaled(&self, rhs: &Self, scale: Complex64) -> Result<Self> {
        self.require_dimension(rhs.dimension)?;
        Self::try_new(
            self.dimension,
            self.values
                .iter()
                .zip(&rhs.values)
                .map(|(left, right)| *left + scale * *right)
                .collect(),
        )
    }

    fn scaled(&self, scale: Complex64) -> Result<Self> {
        Self::try_new(
            self.dimension,
            self.values.iter().map(|value| scale * *value).collect(),
        )
    }

    fn hermitized_normalized(&self) -> Result<Self> {
        let adjoint = self.adjoint();
        let mut result = self.add_scaled(&adjoint, Complex64::new(1.0, 0.0))?;
        result = result.scaled(Complex64::new(0.5, 0.0))?;
        let trace = result.trace().re;
        if !trace.is_finite() {
            return Err(QuantumError::NonFinite("density-matrix trace"));
        }
        if trace.abs() <= f64::MIN_POSITIVE {
            return Err(QuantumError::ZeroTrace);
        }
        result.scaled(Complex64::new(trace.recip(), 0.0))
    }
}

/// Diagnostics for a candidate density matrix.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DensityDiagnostics {
    /// Complex trace.
    pub trace: Complex64,
    /// `Tr(rho²)`.
    pub purity: f64,
    /// Largest absolute component of `rho - rho†`.
    pub hermiticity_residual: f64,
    /// Smallest eigenvalue from a complex Hermitian Jacobi solve.
    pub minimum_eigenvalue: f64,
}

/// A Hermitian, unit-trace, positive-semidefinite density matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct DensityMatrix(Operator);

impl DensityMatrix {
    /// Constructs a density matrix and checks all physical invariants.
    pub fn try_new(operator: Operator, tolerance: f64) -> Result<Self> {
        validate_tolerance(tolerance)?;
        let scale = operator.scale().max(1.0);
        let hermiticity_residual = operator.hermiticity_residual();
        if hermiticity_residual > tolerance * scale {
            return Err(QuantumError::NotHermitian {
                residual: hermiticity_residual,
            });
        }
        let trace = operator.trace();
        if (trace.re - 1.0).abs() > tolerance || trace.im.abs() > tolerance {
            return Err(QuantumError::TraceNotOne {
                real: trace.re,
                imaginary: trace.im,
            });
        }
        let minimum_eigenvalue = minimum_hermitian_eigenvalue(&operator, tolerance)?;
        if minimum_eigenvalue < -tolerance * scale {
            return Err(QuantumError::NotPositiveSemidefinite { minimum_eigenvalue });
        }
        Ok(Self(operator))
    }

    /// Constructs the pure computational-basis state `|index><index|`.
    pub fn basis(dimension: usize, index: usize) -> Result<Self> {
        if dimension == 0 {
            return Err(QuantumError::InvalidDimension);
        }
        if index >= dimension {
            return Err(QuantumError::BasisIndex { dimension, index });
        }
        let length = dimension
            .checked_mul(dimension)
            .ok_or(QuantumError::InvalidDimension)?;
        let mut values = zeroed_operator_values(length)?;
        values[index * dimension + index] = Complex64::new(1.0, 0.0);
        Self::try_new(Operator::try_new(dimension, values)?, 64.0 * f64::EPSILON)
    }

    /// Constructs the maximally mixed state `I / dimension`.
    pub fn maximally_mixed(dimension: usize) -> Result<Self> {
        let operator = Operator::identity(dimension)?;
        Self::try_new(
            operator.scaled(Complex64::new((dimension as f64).recip(), 0.0))?,
            64.0 * f64::EPSILON,
        )
    }

    /// Returns the Hilbert-space dimension.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.0.dimension
    }

    /// Returns the checked operator representation.
    #[must_use]
    pub const fn as_operator(&self) -> &Operator {
        &self.0
    }

    /// Returns one density-matrix element.
    #[must_use]
    pub fn get(&self, row: usize, column: usize) -> Option<Complex64> {
        self.0.get(row, column)
    }

    /// Returns `Tr(rho²)`.
    pub fn purity(&self) -> Result<f64> {
        Ok(self.0.multiply(&self.0)?.trace().re)
    }

    /// Returns invariant diagnostics.
    pub fn diagnostics(&self, tolerance: f64) -> Result<DensityDiagnostics> {
        validate_tolerance(tolerance)?;
        Ok(DensityDiagnostics {
            trace: self.0.trace(),
            purity: self.purity()?,
            hermiticity_residual: self.0.hermiticity_residual(),
            minimum_eigenvalue: minimum_hermitian_eigenvalue(&self.0, tolerance)?,
        })
    }

    /// Returns the expectation value `Tr(rho observable)`.
    pub fn expectation(&self, observable: &Operator) -> Result<Complex64> {
        self.0.require_dimension(observable.dimension)?;
        Ok(self.0.multiply(observable)?.trace())
    }
}

fn minimum_hermitian_eigenvalue(matrix: &Operator, tolerance: f64) -> Result<f64> {
    let n = matrix.dimension;
    let scale = matrix.scale().max(1.0);
    let mut values = matrix.values.clone();
    let maximum_rotations = 64usize
        .checked_mul(n)
        .and_then(|count| count.checked_mul(n))
        .ok_or(QuantumError::InvalidDimension)?;

    for _ in 0..maximum_rotations {
        let mut maximum = 0.0_f64;
        let mut pivot = (0, 0);
        for row in 0..n {
            for column in row + 1..n {
                let magnitude = values[row * n + column].norm();
                if magnitude > maximum {
                    maximum = magnitude;
                    pivot = (row, column);
                }
            }
        }
        if maximum <= tolerance * scale {
            return Ok((0..n)
                .map(|index| values[index * n + index].re)
                .fold(f64::INFINITY, f64::min));
        }

        let (p, q) = pivot;
        let app = values[p * n + p].re;
        let aqq = values[q * n + q].re;
        let apq = values[p * n + q];
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

        for index in 0..n {
            if index == p || index == q {
                continue;
            }
            let aip = values[index * n + p];
            let aiq = values[index * n + q];
            let new_ip = aip * phase * cosine - aiq * sine;
            let new_iq = aip * phase * sine + aiq * cosine;
            values[index * n + p] = new_ip;
            values[p * n + index] = new_ip.conj();
            values[index * n + q] = new_iq;
            values[q * n + index] = new_iq.conj();
        }
        values[p * n + p] = Complex64::new(
            cosine * cosine * app + sine * sine * aqq - 2.0 * cosine * sine * radius,
            0.0,
        );
        values[q * n + q] = Complex64::new(
            sine * sine * app + cosine * cosine * aqq + 2.0 * cosine * sine * radius,
            0.0,
        );
        values[p * n + q] = Complex64::new(0.0, 0.0);
        values[q * n + p] = Complex64::new(0.0, 0.0);
    }
    Err(QuantumError::NonFinite(
        "Hermitian eigensolver did not converge",
    ))
}

/// One Lindblad collapse channel with a nonnegative rate.
#[derive(Clone, Debug, PartialEq)]
pub struct CollapseOperator {
    operator: Operator,
    rate: f64,
}

impl CollapseOperator {
    /// Constructs a checked collapse channel.
    pub fn try_new(operator: Operator, rate: f64) -> Result<Self> {
        if !rate.is_finite() || rate < 0.0 {
            return Err(QuantumError::InvalidRate);
        }
        Ok(Self { operator, rate })
    }

    /// Returns the collapse operator.
    #[must_use]
    pub const fn operator(&self) -> &Operator {
        &self.operator
    }

    /// Returns the channel rate.
    #[must_use]
    pub const fn rate(&self) -> f64 {
        self.rate
    }
}

/// A finite-dimensional time-independent GKSL/Lindblad model.
#[derive(Clone, Debug, PartialEq)]
pub struct LindbladModel {
    hamiltonian: Operator,
    collapse_operators: Vec<CollapseOperator>,
}

impl LindbladModel {
    /// Constructs a model after checking Hermiticity, dimensions, and rates.
    pub fn try_new(
        hamiltonian: Operator,
        collapse_operators: Vec<CollapseOperator>,
        tolerance: f64,
    ) -> Result<Self> {
        validate_tolerance(tolerance)?;
        let residual = hamiltonian.hermiticity_residual();
        if !hamiltonian.is_hermitian(tolerance)? {
            return Err(QuantumError::NotHermitian { residual });
        }
        for collapse in &collapse_operators {
            hamiltonian.require_dimension(collapse.operator.dimension)?;
        }
        Ok(Self {
            hamiltonian,
            collapse_operators,
        })
    }

    /// Returns the Hamiltonian.
    #[must_use]
    pub const fn hamiltonian(&self) -> &Operator {
        &self.hamiltonian
    }

    /// Returns collapse channels in stable input order.
    #[must_use]
    pub fn collapse_operators(&self) -> &[CollapseOperator] {
        &self.collapse_operators
    }

    /// Evaluates the Lindblad right-hand side at `state`.
    pub fn derivative(&self, state: &DensityMatrix) -> Result<Operator> {
        self.hamiltonian.require_dimension(state.dimension())?;
        self.derivative_operator(state.as_operator())
    }

    fn derivative_operator(&self, rho: &Operator) -> Result<Operator> {
        self.hamiltonian.require_dimension(rho.dimension())?;
        let commutator = self.hamiltonian.commutator(rho)?;
        let mut derivative = commutator.scaled(Complex64::new(0.0, -1.0))?;

        for collapse in &self.collapse_operators {
            if collapse.rate == 0.0 {
                continue;
            }
            let jump = &collapse.operator;
            let jump_adjoint = jump.adjoint();
            let jump_norm = jump_adjoint.multiply(jump)?;
            let gain = jump.multiply(rho)?.multiply(&jump_adjoint)?;
            let loss_left = jump_norm.multiply(rho)?;
            let loss_right = rho.multiply(&jump_norm)?;
            derivative = derivative.add_scaled(&gain, Complex64::new(collapse.rate, 0.0))?;
            derivative =
                derivative.add_scaled(&loss_left, Complex64::new(-0.5 * collapse.rate, 0.0))?;
            derivative =
                derivative.add_scaled(&loss_right, Complex64::new(-0.5 * collapse.rate, 0.0))?;
        }
        Ok(derivative)
    }

    /// Evolves a state with fixed-step classical fourth-order Runge–Kutta.
    ///
    /// Every accepted step is re-Hermitized, normalized, and checked for
    /// positive semidefiniteness. Choose a smaller `time_step` if positivity
    /// fails: explicit RK4 is not unconditionally completely positive.
    pub fn evolve_rk4(
        &self,
        initial: &DensityMatrix,
        time_step: f64,
        steps: usize,
        tolerance: f64,
    ) -> Result<DensityMatrix> {
        if !time_step.is_finite() || time_step <= 0.0 {
            return Err(QuantumError::InvalidTimeStep);
        }
        if steps == 0 {
            return Err(QuantumError::ZeroSteps);
        }
        validate_tolerance(tolerance)?;
        self.hamiltonian.require_dimension(initial.dimension())?;

        let mut state = initial.clone();
        for _ in 0..steps {
            state = self.rk4_step(&state, time_step, tolerance)?;
        }
        Ok(state)
    }

    fn rk4_step(
        &self,
        state: &DensityMatrix,
        time_step: f64,
        tolerance: f64,
    ) -> Result<DensityMatrix> {
        let rho = state.as_operator();
        let k1 = self.derivative(state)?;
        let intermediate = rho.add_scaled(&k1, Complex64::new(0.5 * time_step, 0.0))?;
        let k2 = self.derivative_operator(&intermediate)?;

        let intermediate = rho.add_scaled(&k2, Complex64::new(0.5 * time_step, 0.0))?;
        let k3 = self.derivative_operator(&intermediate)?;

        let intermediate = rho.add_scaled(&k3, Complex64::new(time_step, 0.0))?;
        let k4 = self.derivative_operator(&intermediate)?;

        let mut next = rho.add_scaled(&k1, Complex64::new(time_step / 6.0, 0.0))?;
        next = next.add_scaled(&k2, Complex64::new(time_step / 3.0, 0.0))?;
        next = next.add_scaled(&k3, Complex64::new(time_step / 3.0, 0.0))?;
        next = next.add_scaled(&k4, Complex64::new(time_step / 6.0, 0.0))?;
        DensityMatrix::try_new(
            next.hermitized_normalized()?,
            tolerance.max(128.0 * f64::EPSILON),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_hamiltonian() -> Operator {
        Operator::try_new(2, vec![Complex64::new(0.0, 0.0); 4]).unwrap()
    }

    #[test]
    fn density_validation_rejects_negative_eigenvalue() {
        let operator = Operator::try_new(
            2,
            vec![
                Complex64::new(1.1, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-0.1, 0.0),
            ],
        )
        .unwrap();
        assert!(matches!(
            DensityMatrix::try_new(operator, 1e-12),
            Err(QuantumError::NotPositiveSemidefinite { .. })
        ));
    }

    #[test]
    fn rank_one_state_is_accepted_independent_of_basis_order() {
        let epsilon = 1.0e-8_f64;
        let large = (1.0 - epsilon * epsilon).sqrt();
        for amplitudes in [[epsilon, large], [large, epsilon]] {
            let values = vec![
                Complex64::new(amplitudes[0] * amplitudes[0], 0.0),
                Complex64::new(amplitudes[0] * amplitudes[1], 0.0),
                Complex64::new(amplitudes[1] * amplitudes[0], 0.0),
                Complex64::new(amplitudes[1] * amplitudes[1], 0.0),
            ];
            let state = DensityMatrix::try_new(Operator::try_new(2, values).unwrap(), 1e-12);
            assert!(state.is_ok(), "{state:?}");
        }
    }

    #[test]
    fn maximally_mixed_purity_is_inverse_dimension() {
        let state = DensityMatrix::maximally_mixed(4).unwrap();
        assert!((state.purity().unwrap() - 0.25).abs() < 1e-15);
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn enormous_zero_initialized_operators_return_errors_without_panicking() {
        let dimension = 1usize << 30;
        let identity = std::panic::catch_unwind(|| Operator::identity(dimension));
        assert!(
            identity.is_ok(),
            "checked identity construction must not panic"
        );
        assert_eq!(identity.unwrap(), Err(QuantumError::InvalidDimension));

        let basis = std::panic::catch_unwind(|| DensityMatrix::basis(dimension, 0));
        assert!(basis.is_ok(), "checked basis construction must not panic");
        assert_eq!(basis.unwrap(), Err(QuantumError::InvalidDimension));
    }

    #[test]
    fn lindblad_derivative_preserves_trace() {
        let lowering = Operator::try_new(
            2,
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .unwrap();
        let model = LindbladModel::try_new(
            zero_hamiltonian(),
            vec![CollapseOperator::try_new(lowering, 0.7).unwrap()],
            1e-12,
        )
        .unwrap();
        let derivative = model
            .derivative(&DensityMatrix::basis(2, 1).unwrap())
            .unwrap();
        assert!(derivative.trace().norm() < 1e-14);
    }

    #[test]
    fn amplitude_damping_matches_exponential_population() {
        let lowering = Operator::try_new(
            2,
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .unwrap();
        let model = LindbladModel::try_new(
            zero_hamiltonian(),
            vec![CollapseOperator::try_new(lowering, 1.0).unwrap()],
            1e-12,
        )
        .unwrap();
        let evolved = model
            .evolve_rk4(&DensityMatrix::basis(2, 1).unwrap(), 0.001, 1_000, 1e-10)
            .unwrap();
        let excited_population = evolved.get(1, 1).unwrap().re;
        assert!((excited_population - (-1.0_f64).exp()).abs() < 2e-10);
        assert!((evolved.as_operator().trace().re - 1.0).abs() < 1e-13);
    }

    #[test]
    fn coherent_dynamics_rotates_basis_state() {
        let hamiltonian = Operator::try_new(
            2,
            vec![
                Complex64::new(0.0, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        )
        .unwrap();
        let model = LindbladModel::try_new(hamiltonian, vec![], 1e-12).unwrap();
        let evolved = model
            .evolve_rk4(
                &DensityMatrix::basis(2, 0).unwrap(),
                core::f64::consts::PI / 2_000.0,
                1_000,
                1e-10,
            )
            .unwrap();
        assert!((evolved.get(1, 1).unwrap().re - 0.5).abs() < 2e-10);
        assert!((evolved.purity().unwrap() - 1.0).abs() < 2e-10);
    }
}
