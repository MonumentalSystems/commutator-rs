//! Stable recurrent scans and multi-head Helmholtz sequence fibers.

use core::f32::consts::PI;

use crate::error::{checked_product, expect_len, finite_slice};
use crate::{DynamicsError, Result};

fn validate_decay(name: &'static str, decay: &[f32]) -> Result<()> {
    finite_slice(name, decay)?;
    if decay.iter().all(|value| (0.0..=1.0).contains(value)) {
        Ok(())
    } else {
        Err(DynamicsError::InvalidDomain(name))
    }
}

fn ensure_finite_output(name: &'static str, output: &[f32]) -> Result<()> {
    finite_slice(name, output)
}

/// Computes `state[t] = decay * state[t - 1] + input[t]` per feature.
///
/// The direct recurrence avoids the overflowing inverse powers used by the
/// original closed-form scan. `input` is row-major with shape
/// `(timesteps, features)`, and `decay` has length `features` with values in
/// the closed interval `[0, 1]`.
pub fn gated_scan(
    input: &[f32],
    timesteps: usize,
    features: usize,
    decay: &[f32],
) -> Result<Vec<f32>> {
    if features == 0 {
        return Err(DynamicsError::Empty("gated-scan features"));
    }
    let expected = checked_product("gated-scan input", &[timesteps, features])?;
    expect_len("gated-scan input", input.len(), expected)?;
    expect_len("gated-scan decay", decay.len(), features)?;
    finite_slice("gated-scan input", input)?;
    validate_decay("gated-scan decay", decay)?;

    let mut output = vec![0.0f32; expected];
    let mut state = vec![0.0f32; features];
    for (input_row, output_row) in input
        .chunks_exact(features)
        .zip(output.chunks_exact_mut(features))
    {
        for (((state, input), output), decay) in
            state.iter_mut().zip(input_row).zip(output_row).zip(decay)
        {
            *state = *decay * *state + *input;
            *output = *state;
        }
    }
    ensure_finite_output("gated-scan output", &output)?;
    Ok(output)
}

/// Computes reverse-mode derivatives for [`gated_scan`].
///
/// Returns `(input_gradient, decay_gradient)`. The decay derivative uses the
/// reverse accumulated state adjoint, so losses attached to later timesteps
/// correctly contribute through the recurrence.
pub fn gated_scan_backward(
    output_gradient: &[f32],
    input: &[f32],
    timesteps: usize,
    features: usize,
    decay: &[f32],
) -> Result<(Vec<f32>, Vec<f32>)> {
    if features == 0 {
        return Err(DynamicsError::Empty("gated-scan features"));
    }
    let expected = checked_product("gated-scan input", &[timesteps, features])?;
    expect_len(
        "gated-scan output gradient",
        output_gradient.len(),
        expected,
    )?;
    expect_len("gated-scan input", input.len(), expected)?;
    expect_len("gated-scan decay", decay.len(), features)?;
    finite_slice("gated-scan output gradient", output_gradient)?;
    finite_slice("gated-scan input", input)?;
    validate_decay("gated-scan decay", decay)?;

    let states = gated_scan(input, timesteps, features, decay)?;
    let mut input_gradient = vec![0.0f32; expected];
    let mut decay_gradient = vec![0.0f32; features];

    for feature in 0..features {
        let mut adjoint = 0.0f32;
        for timestep in (0..timesteps).rev() {
            let index = timestep * features + feature;
            adjoint += output_gradient[index];
            input_gradient[index] = adjoint;
            let previous_state = if timestep == 0 {
                0.0
            } else {
                states[(timestep - 1) * features + feature]
            };
            decay_gradient[feature] += adjoint * previous_state;
            adjoint *= decay[feature];
        }
    }
    ensure_finite_output("gated-scan input gradient", &input_gradient)?;
    ensure_finite_output("gated-scan decay gradient", &decay_gradient)?;
    Ok((input_gradient, decay_gradient))
}

/// Applies a fixed-decay multi-head Helmholtz fiber.
///
/// `input` has shape `(timesteps, groups, harmonics)`. `decay` has shape
/// `(heads, harmonics)` and contains magnitudes in `[0, 1]`. Head `k` uses the
/// complex pole `decay[k] * exp(i * pi * k / heads)`; real outputs are averaged
/// across heads.
pub fn helmholtz_fiber(
    input: &[f32],
    timesteps: usize,
    groups: usize,
    harmonics: usize,
    decay: &[f32],
    heads: usize,
) -> Result<Vec<f32>> {
    validate_fiber(input, timesteps, groups, harmonics, decay, heads)?;
    let features = checked_product("Helmholtz features", &[groups, harmonics])?;
    let mut output = vec![0.0f32; input.len()];

    for head in 0..heads {
        let phase = head as f32 * PI / heads as f32;
        let (cosine, sine) = (phase.cos(), phase.sin());
        let mut real = vec![0.0f32; features];
        let mut imaginary = vec![0.0f32; features];
        for (input_row, output_row) in input
            .chunks_exact(features)
            .zip(output.chunks_exact_mut(features))
        {
            for feature in 0..features {
                let magnitude = decay[head * harmonics + feature % harmonics];
                let previous_real = real[feature];
                let previous_imaginary = imaginary[feature];
                real[feature] = magnitude * (cosine * previous_real - sine * previous_imaginary)
                    + input_row[feature];
                imaginary[feature] =
                    magnitude * (sine * previous_real + cosine * previous_imaginary);
                output_row[feature] += real[feature];
            }
        }
    }

    let inverse_heads = 1.0 / heads as f32;
    output.iter_mut().for_each(|value| *value *= inverse_heads);
    ensure_finite_output("Helmholtz output", &output)?;
    Ok(output)
}

/// Applies a content-adaptive multi-head Helmholtz fiber.
///
/// This accepts the same post-transform decay magnitudes as
/// [`helmholtz_fiber`]. At each timestep and group, the pole magnitude is
/// multiplied by the nonnegative cosine similarity to the previous input.
/// Keeping both APIs in the same domain avoids a hidden logits-vs-rates
/// dispatch contract.
pub fn adaptive_helmholtz_fiber(
    input: &[f32],
    timesteps: usize,
    groups: usize,
    harmonics: usize,
    decay: &[f32],
    heads: usize,
) -> Result<Vec<f32>> {
    validate_fiber(input, timesteps, groups, harmonics, decay, heads)?;
    let features = checked_product("adaptive Helmholtz features", &[groups, harmonics])?;
    let mut similarity = vec![1.0f32; checked_product("similarity", &[timesteps, groups])?];

    for timestep in 1..timesteps {
        for group in 0..groups {
            let current_start = timestep * features + group * harmonics;
            let previous_start = (timestep - 1) * features + group * harmonics;
            let current = &input[current_start..current_start + harmonics];
            let previous = &input[previous_start..previous_start + harmonics];
            let current_norm = current
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt();
            let previous_norm = previous
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt();
            let value = if current_norm > 1.0e-12 && previous_norm > 1.0e-12 {
                current
                    .iter()
                    .zip(previous)
                    .map(|(current, previous)| current * previous)
                    .sum::<f32>()
                    / (current_norm * previous_norm)
            } else {
                0.0
            };
            similarity[timestep * groups + group] = value.clamp(0.0, 1.0);
        }
    }

    let mut output = vec![0.0f32; input.len()];
    for head in 0..heads {
        let phase = head as f32 * PI / heads as f32;
        let (cosine, sine) = (phase.cos(), phase.sin());
        let mut real = vec![0.0f32; features];
        let mut imaginary = vec![0.0f32; features];
        for timestep in 0..timesteps {
            let row_start = timestep * features;
            for group in 0..groups {
                let adaptive = similarity[timestep * groups + group];
                for harmonic in 0..harmonics {
                    let feature = group * harmonics + harmonic;
                    let magnitude = decay[head * harmonics + harmonic] * adaptive;
                    let previous_real = real[feature];
                    let previous_imaginary = imaginary[feature];
                    real[feature] = magnitude
                        * (cosine * previous_real - sine * previous_imaginary)
                        + input[row_start + feature];
                    imaginary[feature] =
                        magnitude * (sine * previous_real + cosine * previous_imaginary);
                    output[row_start + feature] += real[feature];
                }
            }
        }
    }

    let inverse_heads = 1.0 / heads as f32;
    output.iter_mut().for_each(|value| *value *= inverse_heads);
    ensure_finite_output("adaptive Helmholtz output", &output)?;
    Ok(output)
}

fn validate_fiber(
    input: &[f32],
    timesteps: usize,
    groups: usize,
    harmonics: usize,
    decay: &[f32],
    heads: usize,
) -> Result<()> {
    if groups == 0 {
        return Err(DynamicsError::Empty("Helmholtz groups"));
    }
    if harmonics == 0 {
        return Err(DynamicsError::Empty("Helmholtz harmonics"));
    }
    if heads == 0 {
        return Err(DynamicsError::Empty("Helmholtz heads"));
    }
    let expected = checked_product("Helmholtz input", &[timesteps, groups, harmonics])?;
    expect_len("Helmholtz input", input.len(), expected)?;
    let expected_decay = checked_product("Helmholtz decay", &[heads, harmonics])?;
    expect_len("Helmholtz decay", decay.len(), expected_decay)?;
    finite_slice("Helmholtz input", input)?;
    validate_decay("Helmholtz decay", decay)
}

/// Applies a complex-pole gated recurrence and returns its real component.
///
/// `input` has shape `(timesteps, groups, harmonics, head_width)`,
/// `magnitude` has shape `(groups, harmonics)`, and `phase` has shape
/// `(timesteps, groups, harmonics)`. The recurrence is
/// `state[t] = magnitude * exp(i * phase[t]) * state[t - 1] + input[t]`.
pub fn complex_gated_scan(
    input: &[f32],
    timesteps: usize,
    groups: usize,
    harmonics: usize,
    head_width: usize,
    magnitude: &[f32],
    phase: &[f32],
) -> Result<Vec<f32>> {
    if groups == 0 || harmonics == 0 || head_width == 0 {
        return Err(DynamicsError::Empty("complex gated-scan dimensions"));
    }
    let features = checked_product("complex gated-scan features", &[groups, harmonics])?;
    let row_width = checked_product("complex gated-scan row", &[features, head_width])?;
    let expected = checked_product("complex gated-scan input", &[timesteps, row_width])?;
    expect_len("complex gated-scan input", input.len(), expected)?;
    expect_len("complex gated-scan magnitude", magnitude.len(), features)?;
    let expected_phase = checked_product("complex gated-scan phase", &[timesteps, features])?;
    expect_len("complex gated-scan phase", phase.len(), expected_phase)?;
    finite_slice("complex gated-scan input", input)?;
    validate_decay("complex gated-scan magnitude", magnitude)?;
    finite_slice("complex gated-scan phase", phase)?;

    let mut real = vec![0.0f32; row_width];
    let mut imaginary = vec![0.0f32; row_width];
    let mut output = vec![0.0f32; expected];
    for timestep in 0..timesteps {
        for feature in 0..features {
            let angle = phase[timestep * features + feature];
            let (sine, cosine) = angle.sin_cos();
            for column in 0..head_width {
                let within_row = feature * head_width + column;
                let index = timestep * row_width + within_row;
                let previous_real = real[within_row];
                let previous_imaginary = imaginary[within_row];
                real[within_row] = magnitude[feature]
                    * (cosine * previous_real - sine * previous_imaginary)
                    + input[index];
                imaginary[within_row] =
                    magnitude[feature] * (sine * previous_real + cosine * previous_imaginary);
                output[index] = real[within_row];
            }
        }
    }
    ensure_finite_output("complex gated-scan output", &output)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(left: &[f32], right: &[f32], tolerance: f32) -> bool {
        left.iter()
            .zip(right)
            .all(|(left, right)| (left - right).abs() <= tolerance)
    }

    #[test]
    fn gated_scan_matches_recurrence() {
        let result = gated_scan(&[1.0, 2.0, 3.0], 3, 1, &[0.5]).unwrap();
        assert!(close(&result, &[1.0, 2.5, 4.25], 1.0e-6));
    }

    #[test]
    fn gated_scan_boundaries_are_defined() {
        assert_eq!(
            gated_scan(&[1.0, 2.0], 2, 1, &[0.0]).unwrap(),
            vec![1.0, 2.0]
        );
        assert_eq!(
            gated_scan(&[1.0, 2.0], 2, 1, &[1.0]).unwrap(),
            vec![1.0, 3.0]
        );
    }

    #[test]
    fn backward_decay_matches_finite_difference() {
        let input = [0.4, -0.2, 0.7];
        let upstream = [0.3, -0.5, 1.2];
        let decay = 0.6;
        let (_, gradient) = gated_scan_backward(&upstream, &input, 3, 1, &[decay]).unwrap();
        let loss = |rate: f32| {
            gated_scan(&input, 3, 1, &[rate])
                .unwrap()
                .iter()
                .zip(upstream)
                .map(|(state, weight)| state * weight)
                .sum::<f32>()
        };
        let epsilon = 1.0e-3;
        let numerical = (loss(decay + epsilon) - loss(decay - epsilon)) / (2.0 * epsilon);
        assert!((gradient[0] - numerical).abs() < 1.0e-3);
    }

    #[test]
    fn backward_input_matches_finite_difference() {
        let input = [0.4, -0.2, 0.7];
        let upstream = [0.3, -0.5, 1.2];
        let decay = [0.6];
        let (gradient, _) = gated_scan_backward(&upstream, &input, 3, 1, &decay).unwrap();
        let epsilon = 1.0e-3;
        for index in 0..input.len() {
            let mut plus = input;
            let mut minus = input;
            plus[index] += epsilon;
            minus[index] -= epsilon;
            let loss = |values: &[f32]| {
                gated_scan(values, 3, 1, &decay)
                    .unwrap()
                    .iter()
                    .zip(upstream)
                    .map(|(state, weight)| state * weight)
                    .sum::<f32>()
            };
            let numerical = (loss(&plus) - loss(&minus)) / (2.0 * epsilon);
            assert!((gradient[index] - numerical).abs() < 1.0e-3);
        }
    }

    #[test]
    fn long_scan_reaches_geometric_steady_state() {
        for decay in [0.1, 0.5] {
            let input = vec![1.0; 2_000];
            let output = gated_scan(&input, input.len(), 1, &[decay]).unwrap();
            let expected = 1.0 / (1.0 - decay);
            assert!((output[output.len() - 1] - expected).abs() < 1.0e-5);
        }
    }

    #[test]
    fn one_head_fiber_is_gated_scan() {
        let input = [1.0, 2.0, 3.0];
        let fiber = helmholtz_fiber(&input, 3, 1, 1, &[0.5], 1).unwrap();
        let scan = gated_scan(&input, 3, 1, &[0.5]).unwrap();
        assert!(close(&fiber, &scan, 1.0e-6));
    }

    #[test]
    fn adaptive_fiber_resets_on_orthogonal_change() {
        let input = [1.0, 0.0, 0.0, 1.0];
        let fixed = helmholtz_fiber(&input, 2, 1, 2, &[0.9, 0.9], 1).unwrap();
        let adaptive = adaptive_helmholtz_fiber(&input, 2, 1, 2, &[0.9, 0.9], 1).unwrap();
        assert!(fixed[2] > adaptive[2]);
        assert_eq!(&adaptive[2..], &[0.0, 1.0]);
    }

    #[test]
    fn zero_phase_complex_scan_matches_real_scan() {
        let input = [1.0, 2.0, 3.0];
        let complex = complex_gated_scan(&input, 3, 1, 1, 1, &[0.5], &[0.0; 3]).unwrap();
        let real = gated_scan(&input, 3, 1, &[0.5]).unwrap();
        assert!(close(&complex, &real, 1.0e-6));
    }

    #[test]
    fn complex_scan_matches_direct_impulse_response() {
        let magnitude = 0.7f32;
        let phase = 0.4f32;
        let input = [1.0, 0.0, 0.0, 0.0];
        let output =
            complex_gated_scan(&input, input.len(), 1, 1, 1, &[magnitude], &[phase; 4]).unwrap();
        let expected: Vec<f32> = (0..input.len())
            .map(|power| magnitude.powi(power as i32) * (phase * power as f32).cos())
            .collect();
        assert!(close(&output, &expected, 1.0e-6));
    }

    #[test]
    fn invalid_shapes_rates_and_heads_are_rejected() {
        assert!(gated_scan(&[1.0], 2, 1, &[0.5]).is_err());
        assert!(gated_scan(&[1.0], 1, 1, &[1.1]).is_err());
        assert!(helmholtz_fiber(&[1.0], 1, 1, 1, &[], 0).is_err());
        assert!(complex_gated_scan(&[], 0, 1, 1, 0, &[0.5], &[]).is_err());
    }
}
