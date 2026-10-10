//! Checked deterministic initial conditions for bivector fields.
//!
//! These initializers describe numerical fields, not validated physical
//! models. They write only [`BivectorField::data`]: grid geometry, boundary
//! conditions, propagation coefficients, units, solvers, and timesteps remain
//! the caller's responsibility. Formula evaluation occurs in the field's
//! scalar type. Results are deterministic for a given build and platform, but
//! transcendental functions are not promised to be bit-identical across
//! precisions, standard-library implementations, or hardware.

use crate::{BivectorField, BoundaryCondition, FieldPrecision, FieldScalar, StaBivector};
use std::fmt;

const ALGORITHM_VERSION: u32 = 1;

/// The analytic family used to initialize a field.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitialConditionKind {
    /// A one-dimensional sinusoidal bivector field.
    HarmonicWave1d,
    /// A one-dimensional smooth interface based on `tanh`.
    TanhInterface1d,
    /// A two-dimensional vortex with a finite physical core radius.
    CoredVortex2d,
}

/// Description of a completed initialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InitializationMetadata {
    kind: InitialConditionKind,
    algorithm_version: u32,
    precision: FieldPrecision,
    shape: Vec<usize>,
    points_written: usize,
}

impl InitializationMetadata {
    /// Return the analytic family that was applied.
    pub fn kind(&self) -> InitialConditionKind {
        self.kind
    }

    /// Return the formula version used by this crate.
    pub fn algorithm_version(&self) -> u32 {
        self.algorithm_version
    }

    /// Return the scalar precision used for evaluation.
    pub fn precision(&self) -> FieldPrecision {
        self.precision
    }

    /// Return the initialized grid shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Return the number of field points written.
    pub fn points_written(&self) -> usize {
        self.points_written
    }
}

/// Failure to construct or apply an analytic initial condition.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InitialConditionError {
    /// The field has the wrong spatial rank.
    Dimension {
        /// Required number of spatial dimensions.
        expected: usize,
        /// Number of dimensions present in the field.
        actual: usize,
    },
    /// A field axis has no points.
    EmptyAxis {
        /// Zero-length axis index.
        axis: usize,
    },
    /// An axis contains indices that the selected scalar cannot represent exactly.
    ExtentNotRepresentable {
        /// Axis index.
        axis: usize,
        /// Requested axis length.
        extent: usize,
        /// Scalar precision used for coordinates.
        precision: FieldPrecision,
    },
    /// Multiplying the shape extents overflowed `usize`.
    SizeOverflow,
    /// The data buffer length does not match the field shape.
    DataLength {
        /// Product of the shape extents.
        expected: usize,
        /// Actual data buffer length.
        actual: usize,
    },
    /// A per-point coefficient buffer has the wrong length.
    AuxiliaryLength {
        /// Number of grid points.
        expected: usize,
        /// Actual coefficient buffer length.
        actual: usize,
    },
    /// A named scalar or bivector component is not finite.
    NonFinite(&'static str),
    /// A named parameter is outside its permitted domain.
    InvalidDomain(&'static str),
    /// Evaluating the formula produced a non-finite component.
    NonFiniteOutput {
        /// Flat field index.
        point: usize,
        /// Bivector component index in `(E1, E2, E3, B1, B2, B3)` order.
        component: usize,
    },
    /// A derived coordinate or formula intermediate is not finite.
    NonFiniteEvaluation {
        /// Flat field index being evaluated.
        point: usize,
        /// Formula stage that produced the non-finite value.
        stage: &'static str,
    },
    /// Scratch storage for a transactional update could not be reserved.
    AllocationFailed {
        /// Number of points requested.
        points: usize,
    },
}

impl fmt::Display for InitialConditionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dimension { expected, actual } => {
                write!(f, "expected a {expected}D field, found {actual}D")
            }
            Self::EmptyAxis { axis } => write!(f, "field axis {axis} is empty"),
            Self::ExtentNotRepresentable {
                axis,
                extent,
                precision,
            } => write!(
                f,
                "field axis {axis} length {extent} has indices not exactly representable in {precision}"
            ),
            Self::SizeOverflow => write!(f, "field shape product overflows usize"),
            Self::DataLength { expected, actual } => write!(
                f,
                "field shape requires {expected} points, but data contains {actual}"
            ),
            Self::AuxiliaryLength { expected, actual } => write!(
                f,
                "field requires {expected} per-point coefficients, but contains {actual}"
            ),
            Self::NonFinite(name) => write!(f, "{name} must be finite"),
            Self::InvalidDomain(name) => write!(f, "{name} is outside its permitted domain"),
            Self::NonFiniteOutput { point, component } => write!(
                f,
                "initial condition produced a non-finite value at point {point}, component {component}"
            ),
            Self::NonFiniteEvaluation { point, stage } => write!(
                f,
                "initial condition produced a non-finite {stage} at point {point}"
            ),
            Self::AllocationFailed { points } => {
                write!(f, "could not reserve scratch storage for {points} points")
            }
        }
    }
}

impl std::error::Error for InitialConditionError {}

/// A one-dimensional field
/// `background + cosine*cos(k*x + phase) + sine*sin(k*x + phase)`.
#[derive(Debug, Clone, Copy)]
pub struct HarmonicWave1d<S: FieldScalar> {
    background: StaBivector<S>,
    cosine: StaBivector<S>,
    sine: StaBivector<S>,
    wave_number: S,
    phase: S,
    origin: S,
}

impl<S: FieldScalar> HarmonicWave1d<S> {
    /// Construct a checked harmonic wave.
    ///
    /// `wave_number` is in radians per coordinate unit. A zero wave number is
    /// valid and produces a spatially constant field.
    ///
    /// # Errors
    ///
    /// Returns [`InitialConditionError::NonFinite`] if any coefficient or
    /// scalar parameter is not finite.
    pub fn try_new(
        background: StaBivector<S>,
        cosine: StaBivector<S>,
        sine: StaBivector<S>,
        wave_number: S,
        phase: S,
        origin: S,
    ) -> Result<Self, InitialConditionError> {
        validate_bivector(&background, "background")?;
        validate_bivector(&cosine, "cosine coefficient")?;
        validate_bivector(&sine, "sine coefficient")?;
        validate_finite(wave_number, "wave number")?;
        validate_finite(phase, "phase")?;
        validate_finite(origin, "origin")?;
        Ok(Self {
            background,
            cosine,
            sine,
            wave_number,
            phase,
            origin,
        })
    }

    /// Return the constant background bivector.
    pub fn background(&self) -> StaBivector<S> {
        self.background
    }

    /// Return the cosine coefficient.
    pub fn cosine_coefficient(&self) -> StaBivector<S> {
        self.cosine
    }

    /// Return the sine coefficient.
    pub fn sine_coefficient(&self) -> StaBivector<S> {
        self.sine
    }

    /// Return the angular wave number.
    pub fn wave_number(&self) -> S {
        self.wave_number
    }

    /// Return the phase offset in radians.
    pub fn phase(&self) -> S {
        self.phase
    }

    /// Return the coordinate assigned to field index zero.
    pub fn origin(&self) -> S {
        self.origin
    }

    /// Apply this wave to a one-dimensional field.
    ///
    /// Evaluation uses `x = origin + i * field.dx`. On error, the field is
    /// unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error when the field is not structurally valid 1D storage,
    /// an axis index is not exactly representable in the scalar precision, its
    /// spacing is not finite and positive, scratch allocation fails, or a
    /// coordinate, intermediate, or output component is not finite.
    pub fn apply(
        &self,
        field: &mut BivectorField<S>,
    ) -> Result<InitializationMetadata, InitialConditionError> {
        let points = validate_field(field, 1)?;
        let dx = field.dx;
        transactional_fill(field, points, InitialConditionKind::HarmonicWave1d, |i| {
            let x = checked_coordinate(self.origin, i, dx, i, "x coordinate")?;
            let angle = self.wave_number * x + self.phase;
            validate_evaluation(angle, i, "wave phase")?;
            let (sin, cos) = angle.sin_cos();
            Ok(self
                .background
                .add(&self.cosine.scale(cos))
                .add(&self.sine.scale(sin)))
        })
    }
}

/// A one-dimensional smooth interface between two bivectors.
///
/// The interpolation is
/// `negative + (positive-negative) * (1 + tanh((x-center)/width)) / 2`.
#[derive(Debug, Clone, Copy)]
pub struct TanhInterface1d<S: FieldScalar> {
    negative: StaBivector<S>,
    positive: StaBivector<S>,
    center: S,
    width: S,
    origin: S,
}

impl<S: FieldScalar> TanhInterface1d<S> {
    /// Construct a checked smooth interface.
    ///
    /// # Errors
    ///
    /// Returns an error if a coefficient or scalar parameter is not finite,
    /// or if `width` is not positive.
    pub fn try_new(
        negative: StaBivector<S>,
        positive: StaBivector<S>,
        center: S,
        width: S,
        origin: S,
    ) -> Result<Self, InitialConditionError> {
        validate_bivector(&negative, "negative-side value")?;
        validate_bivector(&positive, "positive-side value")?;
        validate_finite(center, "interface center")?;
        validate_positive(width, "interface width")?;
        validate_finite(origin, "origin")?;
        Ok(Self {
            negative,
            positive,
            center,
            width,
            origin,
        })
    }

    /// Return the negative-coordinate limiting value.
    pub fn negative_value(&self) -> StaBivector<S> {
        self.negative
    }

    /// Return the positive-coordinate limiting value.
    pub fn positive_value(&self) -> StaBivector<S> {
        self.positive
    }

    /// Return the interface center coordinate.
    pub fn center(&self) -> S {
        self.center
    }

    /// Return the positive transition width.
    pub fn width(&self) -> S {
        self.width
    }

    /// Return the coordinate assigned to field index zero.
    pub fn origin(&self) -> S {
        self.origin
    }

    /// Apply this interface to a one-dimensional field.
    ///
    /// Evaluation uses `x = origin + i * field.dx`. On error, the field is
    /// unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error when the field is not structurally valid 1D storage,
    /// an axis index is not exactly representable in the scalar precision, its
    /// spacing is not finite and positive, scratch allocation fails, or a
    /// coordinate, normalized coordinate, or output component is not finite.
    pub fn apply(
        &self,
        field: &mut BivectorField<S>,
    ) -> Result<InitializationMetadata, InitialConditionError> {
        let points = validate_field(field, 1)?;
        let delta = self.positive.sub(&self.negative);
        let dx = field.dx;
        transactional_fill(field, points, InitialConditionKind::TanhInterface1d, |i| {
            let x = checked_coordinate(self.origin, i, dx, i, "x coordinate")?;
            let normalized = (x - self.center) / self.width;
            validate_evaluation(normalized, i, "normalized interface coordinate")?;
            let blend = (S::ONE + normalized.tanh()) * S::HALF;
            Ok(self.negative.add(&delta.scale(blend)))
        })
    }
}

/// A two-dimensional winding field with a finite physical core radius.
///
/// The angular part is `winding*atan2(y-cy, x-cx) + phase`, while the
/// envelope is a numerically stable evaluation of `1 - exp(-r/core_radius)`,
/// using a third-order series when `r/core_radius < 10^-3` and its limiting
/// value of one when that positive ratio exceeds the scalar range. The
/// envelope is exactly zero at the center. A lone nonzero winding is not a
/// smooth periodic-torus field, so
/// [`CoredVortex2d::apply`] rejects periodic boundaries. Sampling can alias
/// windings that are too high for a particular grid resolution.
#[derive(Debug, Clone, Copy)]
pub struct CoredVortex2d<S: FieldScalar> {
    background: StaBivector<S>,
    cosine: StaBivector<S>,
    sine: StaBivector<S>,
    center: [S; 2],
    core_radius: S,
    winding: i16,
    phase: S,
    origin: [S; 2],
}

impl<S: FieldScalar> CoredVortex2d<S> {
    /// Construct a checked cored vortex.
    ///
    /// The cosine and sine coefficients must span a nondegenerate bivector
    /// plane. Specifically, the normalized coefficient vectors must satisfy
    /// `abs(dot) < 1 - 10^-6`. Their lengths need not be equal, so elliptical
    /// winding profiles are supported.
    ///
    /// # Errors
    ///
    /// Returns an error if a coefficient or scalar parameter is not finite,
    /// the coefficient plane is degenerate, `core_radius` is not positive, or
    /// `winding` is zero.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        background: StaBivector<S>,
        cosine: StaBivector<S>,
        sine: StaBivector<S>,
        center: [S; 2],
        core_radius: S,
        winding: i16,
        phase: S,
        origin: [S; 2],
    ) -> Result<Self, InitialConditionError> {
        validate_bivector(&background, "background")?;
        validate_bivector(&cosine, "cosine coefficient")?;
        validate_bivector(&sine, "sine coefficient")?;
        validate_coefficient_plane(&cosine, &sine)?;
        validate_finite(center[0], "center x")?;
        validate_finite(center[1], "center y")?;
        validate_positive(core_radius, "core radius")?;
        if winding == 0 {
            return Err(InitialConditionError::InvalidDomain(
                "winding must be nonzero",
            ));
        }
        validate_finite(phase, "phase")?;
        validate_finite(origin[0], "origin x")?;
        validate_finite(origin[1], "origin y")?;
        Ok(Self {
            background,
            cosine,
            sine,
            center,
            core_radius,
            winding,
            phase,
            origin,
        })
    }

    /// Return the constant background bivector.
    pub fn background(&self) -> StaBivector<S> {
        self.background
    }

    /// Return the cosine coefficient.
    pub fn cosine_coefficient(&self) -> StaBivector<S> {
        self.cosine
    }

    /// Return the sine coefficient.
    pub fn sine_coefficient(&self) -> StaBivector<S> {
        self.sine
    }

    /// Return the physical center `[x, y]`.
    pub fn center(&self) -> [S; 2] {
        self.center
    }

    /// Return the positive physical core radius.
    pub fn core_radius(&self) -> S {
        self.core_radius
    }

    /// Return the signed integer winding number.
    pub fn winding(&self) -> i16 {
        self.winding
    }

    /// Return the angular phase offset.
    pub fn phase(&self) -> S {
        self.phase
    }

    /// Return the coordinate assigned to grid index `[0, 0]`.
    pub fn origin(&self) -> [S; 2] {
        self.origin
    }

    /// Apply this vortex to a two-dimensional field.
    ///
    /// Field shape is interpreted as `[x, y]`; the last (`y`) axis is
    /// contiguous, so flat index `i` maps to `[i / ny, i % ny]`. On error,
    /// the field is unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error for periodic boundaries, invalid 2D field storage,
    /// axis indices not exactly representable in the scalar precision,
    /// non-positive or non-finite spacing, allocation failure, or a non-finite
    /// coordinate, phase, radius, or output component. An overflowing positive
    /// `radius / core_radius` ratio is safely saturated to the envelope limit.
    pub fn apply(
        &self,
        field: &mut BivectorField<S>,
    ) -> Result<InitializationMetadata, InitialConditionError> {
        let points = validate_field(field, 2)?;
        if field.boundary == BoundaryCondition::Periodic {
            return Err(InitialConditionError::InvalidDomain(
                "a lone cored vortex requires non-periodic boundaries",
            ));
        }
        let ny = field.shape[1];
        let dx = field.dx;
        transactional_fill(field, points, InitialConditionKind::CoredVortex2d, |i| {
            let ix = i / ny;
            let iy = i % ny;
            let x = checked_coordinate(self.origin[0], ix, dx, i, "x coordinate")?;
            let y = checked_coordinate(self.origin[1], iy, dx, i, "y coordinate")?;
            let delta_x = x - self.center[0];
            let delta_y = y - self.center[1];
            validate_evaluation(delta_x, i, "x displacement")?;
            validate_evaluation(delta_y, i, "y displacement")?;
            let radius = scaled_hypot(delta_x, delta_y);
            validate_evaluation(radius, i, "radius")?;
            let envelope = stable_cored_envelope(radius / self.core_radius);
            let angle = S::from_i32(i32::from(self.winding)) * delta_y.atan2(delta_x) + self.phase;
            validate_evaluation(angle, i, "vortex phase")?;
            let (sin, cos) = angle.sin_cos();
            Ok(self
                .background
                .add(&self.cosine.scale(envelope * cos))
                .add(&self.sine.scale(envelope * sin)))
        })
    }
}

fn validate_field<S: FieldScalar>(
    field: &BivectorField<S>,
    expected_dimension: usize,
) -> Result<usize, InitialConditionError> {
    if field.shape.len() != expected_dimension {
        return Err(InitialConditionError::Dimension {
            expected: expected_dimension,
            actual: field.shape.len(),
        });
    }
    let mut points = 1usize;
    for (axis, &extent) in field.shape.iter().enumerate() {
        if extent == 0 {
            return Err(InitialConditionError::EmptyAxis { axis });
        }
        let maximum_exact_index = match S::PRECISION {
            FieldPrecision::F32 => 1_u64 << 24,
            FieldPrecision::F64 => 1_u64 << 53,
        };
        if u64::try_from(extent - 1).unwrap_or(u64::MAX) > maximum_exact_index {
            return Err(InitialConditionError::ExtentNotRepresentable {
                axis,
                extent,
                precision: S::PRECISION,
            });
        }
        points = points
            .checked_mul(extent)
            .ok_or(InitialConditionError::SizeOverflow)?;
    }
    if field.data.len() != points {
        return Err(InitialConditionError::DataLength {
            expected: points,
            actual: field.data.len(),
        });
    }
    if let Some(values) = &field.v_squared_per_point {
        if values.len() != points {
            return Err(InitialConditionError::AuxiliaryLength {
                expected: points,
                actual: values.len(),
            });
        }
    }
    validate_positive(field.dx, "field spacing")?;
    Ok(points)
}

fn transactional_fill<S, F>(
    field: &mut BivectorField<S>,
    points: usize,
    kind: InitialConditionKind,
    mut evaluate: F,
) -> Result<InitializationMetadata, InitialConditionError>
where
    S: FieldScalar,
    F: FnMut(usize) -> Result<StaBivector<S>, InitialConditionError>,
{
    let mut data = Vec::new();
    data.try_reserve_exact(points)
        .map_err(|_| InitialConditionError::AllocationFailed { points })?;
    for point in 0..points {
        let value = evaluate(point)?;
        if let Some(component) = value
            .components
            .iter()
            .position(|component| !component.is_finite())
        {
            return Err(InitialConditionError::NonFiniteOutput { point, component });
        }
        data.push(value);
    }
    field.data = data;
    Ok(InitializationMetadata {
        kind,
        algorithm_version: ALGORITHM_VERSION,
        precision: S::PRECISION,
        shape: field.shape.clone(),
        points_written: points,
    })
}

fn checked_coordinate<S: FieldScalar>(
    origin: S,
    index: usize,
    spacing: S,
    point: usize,
    stage: &'static str,
) -> Result<S, InitialConditionError> {
    let coordinate = origin + S::from_usize(index) * spacing;
    validate_evaluation(coordinate, point, stage)?;
    Ok(coordinate)
}

fn validate_evaluation<S: FieldScalar>(
    value: S,
    point: usize,
    stage: &'static str,
) -> Result<(), InitialConditionError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(InitialConditionError::NonFiniteEvaluation { point, stage })
    }
}

fn scaled_hypot<S: FieldScalar>(x: S, y: S) -> S {
    let scale = x.abs().max(y.abs());
    if scale == S::ZERO {
        S::ZERO
    } else {
        let x_scaled = x / scale;
        let y_scaled = y / scale;
        scale * (x_scaled * x_scaled + y_scaled * y_scaled).sqrt()
    }
}

fn stable_cored_envelope<S: FieldScalar>(ratio: S) -> S {
    if !ratio.is_finite() {
        return S::ONE;
    }
    let series_limit = S::from_f64(1e-3);
    if ratio < series_limit {
        ratio * (S::ONE - S::HALF * ratio + ratio * ratio / S::from_f64(6.0))
    } else {
        S::ONE - (-ratio).exp()
    }
}

fn validate_coefficient_plane<S: FieldScalar>(
    cosine: &StaBivector<S>,
    sine: &StaBivector<S>,
) -> Result<(), InitialConditionError> {
    let Some(cosine_direction) = normalized_components(cosine) else {
        return Err(InitialConditionError::InvalidDomain(
            "vortex coefficient plane must be nondegenerate",
        ));
    };
    let Some(sine_direction) = normalized_components(sine) else {
        return Err(InitialConditionError::InvalidDomain(
            "vortex coefficient plane must be nondegenerate",
        ));
    };
    let normalized_dot: S = cosine_direction
        .iter()
        .zip(sine_direction)
        .map(|(&left, right)| left * right)
        .sum();
    if S::ONE - normalized_dot.abs() <= S::from_f64(1e-6) {
        Err(InitialConditionError::InvalidDomain(
            "vortex coefficient plane must be nondegenerate",
        ))
    } else {
        Ok(())
    }
}

fn normalized_components<S: FieldScalar>(value: &StaBivector<S>) -> Option<[S; 6]> {
    let scale = value
        .components
        .iter()
        .fold(S::ZERO, |current, component| current.max(component.abs()));
    if scale == S::ZERO {
        return None;
    }
    let normalized_square: S = value
        .components
        .iter()
        .map(|component| {
            let normalized = *component / scale;
            normalized * normalized
        })
        .sum();
    let normalized_norm = normalized_square.sqrt();
    let mut direction = [S::ZERO; 6];
    for (output, &component) in direction.iter_mut().zip(&value.components) {
        *output = (component / scale) / normalized_norm;
    }
    Some(direction)
}

fn validate_bivector<S: FieldScalar>(
    value: &StaBivector<S>,
    name: &'static str,
) -> Result<(), InitialConditionError> {
    if value
        .components
        .iter()
        .all(|component| component.is_finite())
    {
        Ok(())
    } else {
        Err(InitialConditionError::NonFinite(name))
    }
}

fn validate_finite<S: FieldScalar>(
    value: S,
    name: &'static str,
) -> Result<(), InitialConditionError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(InitialConditionError::NonFinite(name))
    }
}

fn validate_positive<S: FieldScalar>(
    value: S,
    name: &'static str,
) -> Result<(), InitialConditionError> {
    validate_finite(value, name)?;
    if value > S::ZERO {
        Ok(())
    } else {
        Err(InitialConditionError::InvalidDomain(name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BoundaryCondition;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, found {actual}"
        );
    }

    fn basis(component: usize) -> StaBivector<f64> {
        let mut components = [0.0; 6];
        components[component] = 1.0;
        StaBivector { components }
    }

    #[test]
    fn harmonic_wave_has_expected_cardinal_values() {
        let mut field = BivectorField::new_1d(4, 0.25, 1.0, BoundaryCondition::Periodic);
        let wave = HarmonicWave1d::try_new(
            StaBivector::new(0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            basis(0),
            basis(1),
            std::f64::consts::TAU,
            0.0,
            0.0,
        )
        .unwrap();
        let metadata = wave.apply(&mut field).unwrap();

        assert_eq!(metadata.kind(), InitialConditionKind::HarmonicWave1d);
        assert_eq!(metadata.algorithm_version(), 1);
        assert_eq!(metadata.precision(), FieldPrecision::F64);
        assert_eq!(metadata.shape(), &[4]);
        assert_eq!(metadata.points_written(), 4);
        assert_close(field.data[0].components[0], 1.0);
        assert_close(field.data[0].components[1], 0.0);
        assert_close(field.data[0].components[5], 1.0);
        assert_close(field.data[1].components[0], 0.0);
        assert_close(field.data[1].components[1], 1.0);
        assert_close(field.data[2].components[0], -1.0);
        assert_close(field.data[3].components[1], -1.0);
    }

    #[test]
    fn harmonic_phase_and_origin_shift_coordinates() {
        let mut field = BivectorField::new_1d(2, 0.5, 1.0, BoundaryCondition::Free);
        let wave = HarmonicWave1d::try_new(
            StaBivector::zero(),
            basis(2),
            basis(3),
            std::f64::consts::PI,
            std::f64::consts::FRAC_PI_2,
            -0.5,
        )
        .unwrap();
        wave.apply(&mut field).unwrap();
        assert_close(field.data[0].components[2], 1.0);
        assert_close(field.data[1].components[3], 1.0);
    }

    #[test]
    fn tanh_interface_is_centered_and_oriented() {
        let mut field = BivectorField::new_1d(3, 1.0, 2.0, BoundaryCondition::Fixed);
        let interface =
            TanhInterface1d::try_new(basis(0).scale(-1.0), basis(0), 0.0, 0.25, -1.0).unwrap();
        interface.apply(&mut field).unwrap();
        assert!(field.data[0].components[0] < -0.99);
        assert_close(field.data[1].components[0], 0.0);
        assert!(field.data[2].components[0] > 0.99);
        assert_eq!(field.boundary, BoundaryCondition::Fixed);
        assert_eq!(field.v_squared, 4.0);
    }

    #[test]
    fn vortex_uses_physical_coordinates_and_has_zero_core() {
        let mut field = BivectorField::new_2d(3, 5, 0.5, 1.0, BoundaryCondition::Free);
        let vortex = CoredVortex2d::try_new(
            basis(5).scale(0.5),
            basis(0),
            basis(1),
            [0.5, 1.0],
            0.5,
            1,
            0.0,
            [0.0, 0.0],
        )
        .unwrap();
        vortex.apply(&mut field).unwrap();

        let center = field.index(&[1, 2]);
        assert_eq!(
            field.data[center].components,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
        );
        let right = field.index(&[2, 2]);
        assert!(field.data[right].components[0] > 0.63);
        assert_close(field.data[right].components[1], 0.0);
        let above = field.index(&[1, 3]);
        assert_close(field.data[above].components[0], 0.0);
        assert!(field.data[above].components[1] > 0.63);
    }

    #[test]
    fn vortex_winding_sign_reverses_the_y_component() {
        let make = |winding| {
            CoredVortex2d::try_new(
                StaBivector::zero(),
                basis(0),
                basis(1),
                [0.0, 0.0],
                1.0,
                winding,
                0.0,
                [0.0, 0.0],
            )
            .unwrap()
        };
        let mut positive = BivectorField::new_2d(2, 2, 1.0, 1.0, BoundaryCondition::Free);
        let mut negative = BivectorField::new_2d(2, 2, 1.0, 1.0, BoundaryCondition::Free);
        make(1).apply(&mut positive).unwrap();
        make(-1).apply(&mut negative).unwrap();
        assert_close(
            positive.data[3].components[0],
            negative.data[3].components[0],
        );
        assert_close(
            positive.data[3].components[1],
            -negative.data[3].components[1],
        );
    }

    #[test]
    fn vortex_radius_is_stable_at_large_and_small_scales() {
        let vortex64 = CoredVortex2d::try_new(
            StaBivector::zero(),
            basis(0),
            basis(1),
            [0.0, 0.0],
            1e308,
            1,
            0.0,
            [0.0, 0.0],
        )
        .unwrap();
        let mut large = BivectorField::new_2d(2, 1, 1e308, 1.0, BoundaryCondition::Free);
        vortex64.apply(&mut large).unwrap();
        assert_close(large.data[1].components[0], 1.0 - (-1.0_f64).exp());

        let vortex32 = CoredVortex2d::try_new(
            StaBivector::<f32>::zero(),
            StaBivector::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            StaBivector::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
            [0.0, 0.0],
            1.0,
            1,
            0.0,
            [0.0, 0.0],
        )
        .unwrap();
        let mut small = BivectorField::new_2d(2, 1, 1e-8, 1.0, BoundaryCondition::Free);
        vortex32.apply(&mut small).unwrap();
        assert!(small.data[1].components[0] > 0.0);
        assert!((small.data[1].components[0] - 1e-8).abs() < 1e-14);
    }

    #[test]
    fn f32_and_f64_evaluate_the_same_formula_with_precision_tolerance() {
        let mut field32 = BivectorField::new_1d(7, 0.125_f32, 1.0, BoundaryCondition::Periodic);
        let mut field64 = BivectorField::new_1d(7, 0.125_f64, 1.0, BoundaryCondition::Periodic);
        HarmonicWave1d::try_new(
            StaBivector::<f32>::zero(),
            StaBivector::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            StaBivector::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
            2.3,
            0.2,
            -0.1,
        )
        .unwrap()
        .apply(&mut field32)
        .unwrap();
        HarmonicWave1d::try_new(
            StaBivector::<f64>::zero(),
            basis(0),
            basis(1),
            2.3,
            0.2,
            -0.1,
        )
        .unwrap()
        .apply(&mut field64)
        .unwrap();
        for (left, right) in field32.data.iter().zip(&field64.data) {
            for component in 0..6 {
                assert!(
                    (left.components[component] as f64 - right.components[component]).abs() < 2e-6
                );
            }
        }
    }

    #[test]
    fn constructors_reject_invalid_parameters() {
        assert!(matches!(
            TanhInterface1d::try_new(basis(0), basis(1), 0.0, 0.0, 0.0),
            Err(InitialConditionError::InvalidDomain("interface width"))
        ));
        assert!(matches!(
            CoredVortex2d::try_new(
                StaBivector::zero(),
                basis(0),
                basis(1),
                [0.0, 0.0],
                1.0,
                0,
                0.0,
                [0.0, 0.0]
            ),
            Err(InitialConditionError::InvalidDomain(_))
        ));
        assert!(matches!(
            HarmonicWave1d::try_new(StaBivector::zero(), basis(0), basis(1), f64::NAN, 0.0, 0.0),
            Err(InitialConditionError::NonFinite("wave number"))
        ));
        for (cosine, sine) in [
            (StaBivector::zero(), basis(1)),
            (basis(0), basis(0)),
            (basis(0), basis(0).scale(-2.0)),
            (
                StaBivector::new(f64::MAX, f64::MAX, 0.0, 0.0, 0.0, 0.0),
                StaBivector::new(f64::MAX, f64::MAX, 0.0, 0.0, 0.0, 0.0),
            ),
        ] {
            assert!(matches!(
                CoredVortex2d::try_new(
                    StaBivector::zero(),
                    cosine,
                    sine,
                    [0.0, 0.0],
                    1.0,
                    1,
                    0.0,
                    [0.0, 0.0]
                ),
                Err(InitialConditionError::InvalidDomain(
                    "vortex coefficient plane must be nondegenerate"
                ))
            ));
        }
    }

    #[test]
    fn apply_checks_structure_and_is_transactional() {
        let wave = HarmonicWave1d::try_new(StaBivector::zero(), basis(0), basis(1), 1.0, 0.0, 0.0)
            .unwrap();
        let mut field = BivectorField::new_1d(3, 1.0, 1.0, BoundaryCondition::Periodic);
        field.data[0] = basis(5).scale(7.0);
        field.shape[0] = 4;
        let before = field.data[0].components;
        assert!(matches!(
            wave.apply(&mut field),
            Err(InitialConditionError::DataLength {
                expected: 4,
                actual: 3
            })
        ));
        assert_eq!(field.data[0].components, before);

        field.shape = vec![3];
        field.dx = f64::INFINITY;
        assert!(matches!(
            wave.apply(&mut field),
            Err(InitialConditionError::NonFinite("field spacing"))
        ));
        assert_eq!(field.data[0].components, before);
    }

    #[test]
    fn non_finite_evaluation_does_not_partially_update_the_field() {
        let wave =
            HarmonicWave1d::try_new(StaBivector::zero(), basis(0), basis(1), f64::MAX, 0.0, 0.0)
                .unwrap();
        let mut field = BivectorField::new_1d(2, f64::MAX, 1.0, BoundaryCondition::Periodic);
        field.data[0] = basis(4).scale(3.0);
        field.data[1] = basis(5).scale(4.0);
        let before: Vec<_> = field.data.iter().map(|value| value.components).collect();

        assert!(matches!(
            wave.apply(&mut field),
            Err(InitialConditionError::NonFiniteEvaluation { point: 1, .. })
        ));
        let after: Vec<_> = field.data.iter().map(|value| value.components).collect();
        assert_eq!(after, before);

        let interface =
            TanhInterface1d::try_new(basis(0), basis(1), f64::MAX, f64::MAX, 0.0).unwrap();
        let mut interface_field = BivectorField::new_1d(3, f64::MAX, 1.0, BoundaryCondition::Free);
        interface_field.data[0] = basis(4).scale(3.0);
        interface_field.data[1] = basis(5).scale(4.0);
        let interface_before: Vec<_> = interface_field
            .data
            .iter()
            .map(|value| value.components)
            .collect();
        assert!(matches!(
            interface.apply(&mut interface_field),
            Err(InitialConditionError::NonFiniteEvaluation { point: 2, .. })
        ));
        let interface_after: Vec<_> = interface_field
            .data
            .iter()
            .map(|value| value.components)
            .collect();
        assert_eq!(interface_after, interface_before);
    }

    #[test]
    fn apply_rejects_wrong_rank_empty_axes_and_auxiliary_lengths() {
        let interface = TanhInterface1d::try_new(basis(0), basis(1), 0.0, 1.0, 0.0).unwrap();
        let mut two_d = BivectorField::new_2d(2, 2, 1.0, 1.0, BoundaryCondition::Free);
        assert!(matches!(
            interface.apply(&mut two_d),
            Err(InitialConditionError::Dimension {
                expected: 1,
                actual: 2
            })
        ));

        let mut empty = BivectorField::new_1d(0, 1.0, 1.0, BoundaryCondition::Free);
        assert!(matches!(
            interface.apply(&mut empty),
            Err(InitialConditionError::EmptyAxis { axis: 0 })
        ));

        let mut malformed = BivectorField::new_1d(2, 1.0, 1.0, BoundaryCondition::Free);
        malformed.v_squared_per_point = Some(vec![1.0]);
        assert!(matches!(
            interface.apply(&mut malformed),
            Err(InitialConditionError::AuxiliaryLength {
                expected: 2,
                actual: 1
            })
        ));

        let vortex = CoredVortex2d::try_new(
            StaBivector::zero(),
            basis(0),
            basis(1),
            [0.0, 0.0],
            1.0,
            1,
            0.0,
            [0.0, 0.0],
        )
        .unwrap();
        if let Some(large_extent) = 1usize.checked_shl(53) {
            let mut overflow = BivectorField::new_2d(1, 1, 1.0, 1.0, BoundaryCondition::Free);
            overflow.shape = vec![large_extent, large_extent];
            assert!(matches!(
                vortex.apply(&mut overflow),
                Err(InitialConditionError::SizeOverflow)
            ));
        }

        let mut periodic = BivectorField::new_2d(2, 2, 1.0, 1.0, BoundaryCondition::Periodic);
        assert!(matches!(
            vortex.apply(&mut periodic),
            Err(InitialConditionError::InvalidDomain(
                "a lone cored vortex requires non-periodic boundaries"
            ))
        ));

        let wave32 = HarmonicWave1d::try_new(
            StaBivector::<f32>::zero(),
            StaBivector::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            StaBivector::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
            1.0,
            0.0,
            0.0,
        )
        .unwrap();
        let mut inexact = BivectorField::new_1d(1, 1.0_f32, 1.0, BoundaryCondition::Free);
        inexact.shape = vec![(1 << 24) + 2];
        assert!(matches!(
            wave32.apply(&mut inexact),
            Err(InitialConditionError::ExtentNotRepresentable {
                axis: 0,
                precision: FieldPrecision::F32,
                ..
            })
        ));
    }
}
