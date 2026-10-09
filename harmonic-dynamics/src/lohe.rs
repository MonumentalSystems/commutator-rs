//! Lohe mean-field synchronization on spheres and unit quaternions.

use crate::error::{checked_product, expect_len, finite_scalar, finite_slice};
use crate::quaternion::Quaternion;
use crate::{DynamicsError, Result};

const NORM_EPSILON: f32 = 1.0e-12;

/// Coupling data shared by the spherical Lohe integrators.
///
/// Strengths must be finite and nonnegative. Optional confidence values lie in
/// `[0, 1]`. An optional row-major adjacency matrix supplies one row of mean
/// weights per oscillator; otherwise a uniform `1 / groups` mean is used.
#[derive(Debug, Clone, Copy)]
pub struct Coupling<'a> {
    /// Per-group nonnegative coupling strengths.
    pub strengths: &'a [f32],
    /// Optional row-major `(groups, groups)` mean weights.
    pub adjacency: Option<&'a [f32]>,
    /// Optional per-group confidence gates in `[0, 1]`.
    pub confidence: Option<&'a [f32]>,
}

/// Dimensions of a row-major sequence of sphere-state frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceShape {
    /// Number of independent frames.
    pub timesteps: usize,
    /// Number of oscillators in each frame.
    pub groups: usize,
    /// Coordinates in each oscillator state.
    pub dimension: usize,
}

/// Coupling data for [`sync_sphere_sequence`].
#[derive(Debug, Clone, Copy)]
pub struct SequenceCoupling<'a> {
    /// Per-group nonnegative coupling strengths, shared across frames.
    pub strengths: &'a [f32],
    /// Optional row-major `(groups, groups)` adjacency, shared across frames.
    pub adjacency: Option<&'a [f32]>,
    /// Optional row-major `(timesteps, groups)` confidence gates.
    pub confidence: Option<&'a [f32]>,
}

fn validate_coupling(groups: usize, coupling: Coupling<'_>) -> Result<()> {
    expect_len("coupling strengths", coupling.strengths.len(), groups)?;
    finite_slice("coupling strengths", coupling.strengths)?;
    if coupling.strengths.iter().any(|value| *value < 0.0) {
        return Err(DynamicsError::InvalidDomain("coupling strengths"));
    }
    if let Some(adjacency) = coupling.adjacency {
        let expected = checked_product("Lohe adjacency", &[groups, groups])?;
        expect_len("Lohe adjacency", adjacency.len(), expected)?;
        finite_slice("Lohe adjacency", adjacency)?;
    }
    if let Some(confidence) = coupling.confidence {
        expect_len("Lohe confidence", confidence.len(), groups)?;
        finite_slice("Lohe confidence", confidence)?;
        if confidence.iter().any(|value| !(0.0..=1.0).contains(value)) {
            return Err(DynamicsError::InvalidDomain("Lohe confidence"));
        }
    }
    Ok(())
}

fn normalize_rows(states: &mut [f32], dimension: usize) -> Result<()> {
    for row in states.chunks_exact_mut(dimension) {
        let norm = row.iter().map(|value| value * value).sum::<f32>().sqrt();
        if !norm.is_finite() {
            return Err(DynamicsError::NonFinite("Lohe state norm"));
        }
        if norm <= NORM_EPSILON {
            return Err(DynamicsError::ZeroNorm("Lohe state"));
        }
        row.iter_mut().for_each(|value| *value /= norm);
    }
    Ok(())
}

fn mean_fields(
    states: &[f32],
    groups: usize,
    dimension: usize,
    adjacency: Option<&[f32]>,
) -> Vec<f32> {
    let mut means = vec![0.0f32; states.len()];
    for row in 0..groups {
        for column in 0..groups {
            let weight = adjacency
                .map(|matrix| matrix[row * groups + column])
                .unwrap_or(1.0 / groups as f32);
            for component in 0..dimension {
                means[row * dimension + component] +=
                    weight * states[column * dimension + component];
            }
        }
    }
    means
}

fn sphere_step(
    state: &[f32],
    target: &[f32],
    angular_scale: f32,
    output: &mut [f32],
) -> Result<()> {
    let dot: f32 = state
        .iter()
        .zip(target)
        .map(|(state, target)| state * target)
        .sum();
    let tangent: Vec<f32> = target
        .iter()
        .zip(state)
        .map(|(target, state)| target - dot * state)
        .collect();
    let speed = tangent
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if speed <= NORM_EPSILON || angular_scale == 0.0 {
        output.copy_from_slice(state);
        return Ok(());
    }
    let angle = angular_scale * speed;
    let tangent_scale = angle.sin() / speed;
    for ((output, state), tangent) in output.iter_mut().zip(state).zip(tangent) {
        *output = angle.cos() * state + tangent_scale * tangent;
    }
    normalize_rows(output, output.len())
}

/// Synchronizes row-major sphere states with an explicit vector mean field.
///
/// The mean magnitude is retained: incoherent groups therefore exert less
/// influence than coherent groups. Inputs are normalized once before the
/// first update, and the exponential-map update preserves unit norm.
pub fn sync_sphere(
    states: &mut [f32],
    groups: usize,
    dimension: usize,
    coupling: Coupling<'_>,
    step: f32,
    iterations: usize,
) -> Result<()> {
    validate_sync_inputs(states, groups, dimension, coupling, step)?;
    normalize_rows(states, dimension)?;
    let mut next = vec![0.0f32; states.len()];

    for _ in 0..iterations {
        let means = mean_fields(states, groups, dimension, coupling.adjacency);
        for group in 0..groups {
            let start = group * dimension;
            let confidence = coupling
                .confidence
                .map(|values| values[group])
                .unwrap_or(1.0);
            sphere_step(
                &states[start..start + dimension],
                &means[start..start + dimension],
                step * coupling.strengths[group] * confidence,
                &mut next[start..start + dimension],
            )?;
        }
        states.copy_from_slice(&next);
    }
    Ok(())
}

fn validate_sync_inputs(
    states: &[f32],
    groups: usize,
    dimension: usize,
    coupling: Coupling<'_>,
    step: f32,
) -> Result<()> {
    if groups == 0 {
        return Err(DynamicsError::Empty("Lohe groups"));
    }
    if dimension == 0 {
        return Err(DynamicsError::Empty("Lohe dimension"));
    }
    let expected = checked_product("Lohe states", &[groups, dimension])?;
    expect_len("Lohe states", states.len(), expected)?;
    finite_slice("Lohe states", states)?;
    finite_scalar("Lohe step", step)?;
    if step < 0.0 {
        return Err(DynamicsError::InvalidDomain("Lohe step"));
    }
    validate_coupling(groups, coupling)
}

/// Synchronizes a sequence of independent sphere-state frames.
///
/// `states` has shape `(timesteps, groups, dimension)`. If supplied,
/// `confidence` has shape `(timesteps, groups)` and is sliced per frame;
/// adjacency and strengths are shared across frames.
pub fn sync_sphere_sequence(
    states: &mut [f32],
    shape: SequenceShape,
    coupling: SequenceCoupling<'_>,
    step: f32,
    iterations: usize,
) -> Result<()> {
    let SequenceShape {
        timesteps,
        groups,
        dimension,
    } = shape;
    if groups == 0 {
        return Err(DynamicsError::Empty("Lohe groups"));
    }
    if dimension == 0 {
        return Err(DynamicsError::Empty("Lohe dimension"));
    }
    finite_scalar("Lohe step", step)?;
    if step < 0.0 {
        return Err(DynamicsError::InvalidDomain("Lohe step"));
    }
    validate_coupling(
        groups,
        Coupling {
            strengths: coupling.strengths,
            adjacency: coupling.adjacency,
            confidence: None,
        },
    )?;
    let frame_width = checked_product("Lohe frame", &[groups, dimension])?;
    let expected = checked_product("Lohe sequence", &[timesteps, frame_width])?;
    expect_len("Lohe sequence", states.len(), expected)?;
    if let Some(values) = coupling.confidence {
        let expected_confidence =
            checked_product("Lohe sequence confidence", &[timesteps, groups])?;
        expect_len(
            "Lohe sequence confidence",
            values.len(),
            expected_confidence,
        )?;
        finite_slice("Lohe sequence confidence", values)?;
        if values.iter().any(|value| !(0.0..=1.0).contains(value)) {
            return Err(DynamicsError::InvalidDomain("Lohe sequence confidence"));
        }
    }

    for (timestep, frame) in states.chunks_exact_mut(frame_width).enumerate() {
        let frame_confidence = coupling.confidence.map(|values| {
            let start = timestep * groups;
            &values[start..start + groups]
        });
        sync_sphere(
            frame,
            groups,
            dimension,
            Coupling {
                strengths: coupling.strengths,
                adjacency: coupling.adjacency,
                confidence: frame_confidence,
            },
            step,
            iterations,
        )?;
    }
    Ok(())
}

/// Synchronizes unit quaternions through the same sphere update used by
/// [`sync_sphere`].
pub fn sync_quaternions(
    states: &mut [Quaternion],
    coupling: Coupling<'_>,
    step: f32,
    iterations: usize,
) -> Result<()> {
    if states.is_empty() {
        return Err(DynamicsError::Empty("quaternion states"));
    }
    let flat_len = checked_product("quaternion states", &[states.len(), 4])?;
    let mut flat = Vec::with_capacity(flat_len);
    for state in states.iter() {
        flat.extend_from_slice(state.as_array());
    }
    sync_sphere(&mut flat, states.len(), 4, coupling, step, iterations)?;
    for (state, components) in states.iter_mut().zip(flat.chunks_exact(4)) {
        *state =
            Quaternion::from_array([components[0], components[1], components[2], components[3]])?;
    }
    Ok(())
}

/// Synchronizes sphere states after a per-group Givens rotation of each mean.
///
/// `phases` contains one finite angle per group. The rotation acts in the
/// first two coordinates, so `dimension >= 2` is required.
pub fn sync_phase_shifted_sphere(
    states: &mut [f32],
    groups: usize,
    dimension: usize,
    coupling: Coupling<'_>,
    phases: &[f32],
    step: f32,
    iterations: usize,
) -> Result<()> {
    if dimension < 2 {
        return Err(DynamicsError::InvalidDomain(
            "phase-shifted sphere dimension must be at least two",
        ));
    }
    validate_sync_inputs(states, groups, dimension, coupling, step)?;
    expect_len("Lohe phases", phases.len(), groups)?;
    finite_slice("Lohe phases", phases)?;
    normalize_rows(states, dimension)?;
    let mut next = vec![0.0f32; states.len()];

    for _ in 0..iterations {
        let mut means = mean_fields(states, groups, dimension, coupling.adjacency);
        for group in 0..groups {
            let start = group * dimension;
            let (sine, cosine) = phases[group].sin_cos();
            let first = means[start];
            let second = means[start + 1];
            means[start] = cosine * first - sine * second;
            means[start + 1] = sine * first + cosine * second;
            let confidence = coupling
                .confidence
                .map(|values| values[group])
                .unwrap_or(1.0);
            sphere_step(
                &states[start..start + dimension],
                &means[start..start + dimension],
                step * coupling.strengths[group] * confidence,
                &mut next[start..start + dimension],
            )?;
        }
        states.copy_from_slice(&next);
    }
    Ok(())
}

/// Synchronizes quaternions after left-multiplying each mean by a unit phase.
pub fn sync_phase_shifted_quaternions(
    states: &mut [Quaternion],
    coupling: Coupling<'_>,
    phases: &[Quaternion],
    step: f32,
    iterations: usize,
) -> Result<()> {
    if states.is_empty() {
        return Err(DynamicsError::Empty("quaternion states"));
    }
    expect_len("quaternion phases", phases.len(), states.len())?;
    validate_coupling(states.len(), coupling)?;
    finite_scalar("Lohe step", step)?;
    if step < 0.0 {
        return Err(DynamicsError::InvalidDomain("Lohe step"));
    }
    for state in states.iter_mut() {
        *state = state.normalized()?;
    }
    let mut next = states.to_vec();

    for _ in 0..iterations {
        let flat_len = checked_product("quaternion states", &[states.len(), 4])?;
        let mut flat = Vec::with_capacity(flat_len);
        for state in states.iter() {
            flat.extend_from_slice(state.as_array());
        }
        let means = mean_fields(&flat, states.len(), 4, coupling.adjacency);
        for group in 0..states.len() {
            let start = group * 4;
            let mean = Quaternion::from_array([
                means[start],
                means[start + 1],
                means[start + 2],
                means[start + 3],
            ])?;
            let rotated = phases[group].normalized()?.hamilton_product(mean)?;
            let confidence = coupling
                .confidence
                .map(|values| values[group])
                .unwrap_or(1.0);
            let mut output = [0.0f32; 4];
            sphere_step(
                states[group].as_array(),
                rotated.as_array(),
                step * coupling.strengths[group] * confidence,
                &mut output,
            )?;
            next[group] = Quaternion::from_array(output)?;
        }
        states.copy_from_slice(&next);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spherical_sync_converges_and_preserves_norm() {
        let mut states = [1.0, 0.0, 0.0, 1.0];
        let coupling = Coupling {
            strengths: &[1.0, 1.0],
            adjacency: None,
            confidence: None,
        };
        sync_sphere(&mut states, 2, 2, coupling, 0.1, 30).unwrap();
        let dot = states[0] * states[2] + states[1] * states[3];
        assert!(dot > 0.9);
        assert!((states[..2].iter().map(|v| v * v).sum::<f32>() - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn quaternion_sync_uses_shared_update() {
        let mut quaternions = [
            Quaternion::IDENTITY,
            Quaternion::new(0.0, 1.0, 0.0, 0.0).unwrap(),
        ];
        sync_quaternions(
            &mut quaternions,
            Coupling {
                strengths: &[1.0, 1.0],
                adjacency: None,
                confidence: None,
            },
            0.1,
            30,
        )
        .unwrap();
        assert!(quaternions[0].dot(quaternions[1]).unwrap() > 0.9);
    }

    #[test]
    fn sequence_confidence_is_sliced_per_timestep() {
        let mut sequence = [
            1.0, 0.0, 0.0, 1.0, // first frame
            1.0, 0.0, 0.0, 1.0, // second frame
        ];
        sync_sphere_sequence(
            &mut sequence,
            SequenceShape {
                timesteps: 2,
                groups: 2,
                dimension: 2,
            },
            SequenceCoupling {
                strengths: &[1.0, 1.0],
                adjacency: None,
                confidence: Some(&[0.0, 0.0, 1.0, 1.0]),
            },
            0.2,
            1,
        )
        .unwrap();
        assert_eq!(&sequence[..4], &[1.0, 0.0, 0.0, 1.0]);
        assert_ne!(&sequence[4..], &[1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn zero_mean_and_zero_tangent_leave_state_unchanged() {
        let original = [1.0, 0.0, -1.0, 0.0];
        let mut states = original;
        sync_sphere(
            &mut states,
            2,
            2,
            Coupling {
                strengths: &[1.0, 1.0],
                adjacency: None,
                confidence: None,
            },
            0.5,
            2,
        )
        .unwrap();
        assert_eq!(states, original);
    }

    #[test]
    fn phase_shift_requires_two_dimensions() {
        let mut states = [1.0];
        let result = sync_phase_shifted_sphere(
            &mut states,
            1,
            1,
            Coupling {
                strengths: &[1.0],
                adjacency: None,
                confidence: None,
            },
            &[0.0],
            0.1,
            1,
        );
        assert!(matches!(result, Err(DynamicsError::InvalidDomain(_))));
    }

    #[test]
    fn invalid_topology_and_negative_coupling_are_rejected() {
        let mut states = [1.0, 0.0, 0.0, 1.0];
        assert!(sync_sphere(
            &mut states,
            2,
            2,
            Coupling {
                strengths: &[1.0, -1.0],
                adjacency: None,
                confidence: None,
            },
            0.1,
            1,
        )
        .is_err());
        assert!(sync_sphere(
            &mut states,
            2,
            2,
            Coupling {
                strengths: &[1.0, 1.0],
                adjacency: Some(&[1.0]),
                confidence: None,
            },
            0.1,
            1,
        )
        .is_err());
    }

    #[test]
    fn empty_sequence_still_validates_dimensions_and_coupling() {
        let mut states = [];
        let valid_empty = sync_sphere_sequence(
            &mut states,
            SequenceShape {
                timesteps: 0,
                groups: 1,
                dimension: 2,
            },
            SequenceCoupling {
                strengths: &[1.0],
                adjacency: None,
                confidence: Some(&[]),
            },
            0.1,
            1,
        );
        assert!(valid_empty.is_ok());

        let zero_groups = sync_sphere_sequence(
            &mut states,
            SequenceShape {
                timesteps: 0,
                groups: 0,
                dimension: 2,
            },
            SequenceCoupling {
                strengths: &[],
                adjacency: None,
                confidence: None,
            },
            0.1,
            1,
        );
        assert!(matches!(zero_groups, Err(DynamicsError::Empty(_))));

        let zero_dimension = sync_sphere_sequence(
            &mut states,
            SequenceShape {
                timesteps: 0,
                groups: 1,
                dimension: 0,
            },
            SequenceCoupling {
                strengths: &[1.0],
                adjacency: None,
                confidence: None,
            },
            0.1,
            1,
        );
        assert!(matches!(zero_dimension, Err(DynamicsError::Empty(_))));
    }
}
