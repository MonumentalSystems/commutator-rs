use crate::{LyapunovError, Result};

/// A continuous-time system that can apply its Jacobian without constructing it.
///
/// The state obeys `dx/dt = f(t, x)`. A tangent direction `v` obeys
/// `dv/dt = J_f(t, x) v`. Implementations write into caller-owned output slices
/// and should not alias an input and output slice.
pub trait ContinuousSystem {
    /// Returns the number of scalars in one state.
    fn dimension(&self) -> usize;

    /// Evaluates the vector field `f(t, state)` into `output`.
    fn vector_field(&self, time: f64, state: &[f64], output: &mut [f64]) -> Result<()>;

    /// Applies the state Jacobian to `direction`, writing `J_f direction` to `output`.
    fn jacobian_vector_product(
        &self,
        time: f64,
        state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<()>;
}

/// Advances a base trajectory and a collection of tangent vectors together.
///
/// Tangents use vector-major layout: tangent vector `j` occupies
/// `tangents[j * dimension..(j + 1) * dimension]`. An optimized PDE solver may
/// implement this trait directly to share stencils, halo exchanges, or device
/// kernels between the state and tangent updates.
pub trait TangentStepper {
    /// Returns the number of scalars in one state.
    fn dimension(&self) -> usize;

    /// Advances the state and tangent vectors by `step_size`.
    fn step(
        &mut self,
        time: f64,
        state: &mut [f64],
        tangents: &mut [f64],
        tangent_count: usize,
        step_size: f64,
    ) -> Result<()>;
}

#[derive(Default)]
struct Rk4Scratch {
    state_derivatives: [Vec<f64>; 4],
    tangent_derivatives: [Vec<f64>; 4],
    temporary_state: Vec<f64>,
    temporary_tangents: Vec<f64>,
}

impl Rk4Scratch {
    fn resize(&mut self, dimension: usize, tangent_len: usize) {
        for derivative in &mut self.state_derivatives {
            derivative.resize(dimension, 0.0);
        }
        for derivative in &mut self.tangent_derivatives {
            derivative.resize(tangent_len, 0.0);
        }
        self.temporary_state.resize(dimension, 0.0);
        self.temporary_tangents.resize(tangent_len, 0.0);
    }
}

/// Classical fourth-order Runge-Kutta adapter for a [`ContinuousSystem`].
///
/// The adapter retains its work buffers between calls. Its memory requirement
/// is `5 * dimension + 5 * top_k * dimension` scalar values.
pub struct Rk4<S> {
    system: S,
    scratch: Rk4Scratch,
}

impl<S> Rk4<S> {
    /// Wraps a matrix-free continuous system in an RK4 tangent stepper.
    #[must_use]
    pub fn new(system: S) -> Self {
        Self {
            system,
            scratch: Rk4Scratch::default(),
        }
    }

    /// Borrows the wrapped system.
    #[must_use]
    pub const fn system(&self) -> &S {
        &self.system
    }

    /// Mutably borrows the wrapped system.
    #[must_use]
    pub fn system_mut(&mut self) -> &mut S {
        &mut self.system
    }

    /// Returns the wrapped system.
    #[must_use]
    pub fn into_inner(self) -> S {
        self.system
    }
}

impl<S: ContinuousSystem> TangentStepper for Rk4<S> {
    fn dimension(&self) -> usize {
        self.system.dimension()
    }

    fn step(
        &mut self,
        time: f64,
        state: &mut [f64],
        tangents: &mut [f64],
        tangent_count: usize,
        step_size: f64,
    ) -> Result<()> {
        let dimension = self.system.dimension();
        validate_step_inputs(dimension, state, tangents, tangent_count, time, step_size)?;
        let tangent_len =
            dimension
                .checked_mul(tangent_count)
                .ok_or(LyapunovError::DimensionMismatch {
                    context: "tangent storage",
                    expected: usize::MAX,
                    actual: tangents.len(),
                })?;
        self.scratch.resize(dimension, tangent_len);

        self.system
            .vector_field(time, state, &mut self.scratch.state_derivatives[0])?;
        evaluate_tangent_stage(
            &self.system,
            time,
            state,
            tangents,
            dimension,
            &mut self.scratch.tangent_derivatives[0],
        )?;

        combine_stage(
            state,
            &self.scratch.state_derivatives[0],
            0.5 * step_size,
            &mut self.scratch.temporary_state,
        );
        combine_stage(
            tangents,
            &self.scratch.tangent_derivatives[0],
            0.5 * step_size,
            &mut self.scratch.temporary_tangents,
        );
        self.system.vector_field(
            time + 0.5 * step_size,
            &self.scratch.temporary_state,
            &mut self.scratch.state_derivatives[1],
        )?;
        evaluate_tangent_stage(
            &self.system,
            time + 0.5 * step_size,
            &self.scratch.temporary_state,
            &self.scratch.temporary_tangents,
            dimension,
            &mut self.scratch.tangent_derivatives[1],
        )?;

        combine_stage(
            state,
            &self.scratch.state_derivatives[1],
            0.5 * step_size,
            &mut self.scratch.temporary_state,
        );
        combine_stage(
            tangents,
            &self.scratch.tangent_derivatives[1],
            0.5 * step_size,
            &mut self.scratch.temporary_tangents,
        );
        self.system.vector_field(
            time + 0.5 * step_size,
            &self.scratch.temporary_state,
            &mut self.scratch.state_derivatives[2],
        )?;
        evaluate_tangent_stage(
            &self.system,
            time + 0.5 * step_size,
            &self.scratch.temporary_state,
            &self.scratch.temporary_tangents,
            dimension,
            &mut self.scratch.tangent_derivatives[2],
        )?;

        combine_stage(
            state,
            &self.scratch.state_derivatives[2],
            step_size,
            &mut self.scratch.temporary_state,
        );
        combine_stage(
            tangents,
            &self.scratch.tangent_derivatives[2],
            step_size,
            &mut self.scratch.temporary_tangents,
        );
        self.system.vector_field(
            time + step_size,
            &self.scratch.temporary_state,
            &mut self.scratch.state_derivatives[3],
        )?;
        evaluate_tangent_stage(
            &self.system,
            time + step_size,
            &self.scratch.temporary_state,
            &self.scratch.temporary_tangents,
            dimension,
            &mut self.scratch.tangent_derivatives[3],
        )?;

        rk4_update(state, &self.scratch.state_derivatives, step_size);
        rk4_update(tangents, &self.scratch.tangent_derivatives, step_size);
        ensure_finite(state, "state")?;
        ensure_finite(tangents, "tangents")?;
        Ok(())
    }
}

fn validate_step_inputs(
    dimension: usize,
    state: &[f64],
    tangents: &[f64],
    tangent_count: usize,
    time: f64,
    step_size: f64,
) -> Result<()> {
    if dimension == 0 {
        return Err(LyapunovError::ZeroDimension);
    }
    if state.len() != dimension {
        return Err(LyapunovError::DimensionMismatch {
            context: "state",
            expected: dimension,
            actual: state.len(),
        });
    }
    let expected =
        dimension
            .checked_mul(tangent_count)
            .ok_or(LyapunovError::DimensionMismatch {
                context: "tangent storage",
                expected: usize::MAX,
                actual: tangents.len(),
            })?;
    if tangents.len() != expected {
        return Err(LyapunovError::DimensionMismatch {
            context: "tangent storage",
            expected,
            actual: tangents.len(),
        });
    }
    if !time.is_finite() {
        return Err(LyapunovError::InvalidParameter {
            field: "time",
            value: time,
        });
    }
    if !step_size.is_finite() || step_size <= 0.0 {
        return Err(LyapunovError::InvalidParameter {
            field: "step_size",
            value: step_size,
        });
    }
    Ok(())
}

fn evaluate_tangent_stage<S: ContinuousSystem>(
    system: &S,
    time: f64,
    state: &[f64],
    tangents: &[f64],
    dimension: usize,
    output: &mut [f64],
) -> Result<()> {
    for (direction, result) in tangents
        .chunks_exact(dimension)
        .zip(output.chunks_exact_mut(dimension))
    {
        system.jacobian_vector_product(time, state, direction, result)?;
    }
    Ok(())
}

fn combine_stage(base: &[f64], derivative: &[f64], scale: f64, output: &mut [f64]) {
    for ((out, base), derivative) in output.iter_mut().zip(base).zip(derivative) {
        *out = base + scale * derivative;
    }
}

fn rk4_update(values: &mut [f64], derivatives: &[Vec<f64>; 4], step_size: f64) {
    let scale = step_size / 6.0;
    for (index, value) in values.iter_mut().enumerate() {
        *value += scale
            * (derivatives[0][index]
                + 2.0 * derivatives[1][index]
                + 2.0 * derivatives[2][index]
                + derivatives[3][index]);
    }
}

fn ensure_finite(values: &[f64], context: &'static str) -> Result<()> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return Err(LyapunovError::NonFiniteValue { context, index });
    }
    Ok(())
}
