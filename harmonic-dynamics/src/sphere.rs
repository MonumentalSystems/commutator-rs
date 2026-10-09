//! Pure geometry on unit spheres of arbitrary finite dimension.

use crate::error::{checked_product, expect_len, finite_scalar, finite_slice};
use crate::{DynamicsError, Result};

const NORM_EPSILON: f32 = 1.0e-12;
const TANGENT_TOLERANCE: f32 = 1.0e-4;

fn norm(values: &[f32]) -> f32 {
    values.iter().map(|value| value * value).sum::<f32>().sqrt()
}

fn validate_pair<'a>(
    left_name: &'static str,
    left: &'a [f32],
    right_name: &'static str,
    right: &'a [f32],
) -> Result<()> {
    if left.is_empty() {
        return Err(DynamicsError::Empty(left_name));
    }
    expect_len(right_name, right.len(), left.len())?;
    finite_slice(left_name, left)?;
    finite_slice(right_name, right)
}

/// Returns a normalized copy of a nonzero finite vector.
pub fn normalize(values: &[f32]) -> Result<Vec<f32>> {
    if values.is_empty() {
        return Err(DynamicsError::Empty("sphere vector"));
    }
    finite_slice("sphere vector", values)?;
    let length = norm(values);
    if !length.is_finite() {
        return Err(DynamicsError::NonFinite("sphere vector norm"));
    }
    if length <= NORM_EPSILON {
        return Err(DynamicsError::ZeroNorm("sphere vector"));
    }
    Ok(values.iter().map(|value| value / length).collect())
}

/// Performs spherical linear interpolation between two nonzero vectors.
///
/// Inputs must already have unit norm. Antipodal points are rejected because
/// their shortest great-circle arc is not unique.
pub fn slerp(left: &[f32], right: &[f32], fraction: f32) -> Result<Vec<f32>> {
    validate_pair("left sphere point", left, "right sphere point", right)?;
    finite_scalar("interpolation fraction", fraction)?;
    for (name, point) in [("left sphere point", left), ("right sphere point", right)] {
        let point_norm = norm(point);
        if (point_norm - 1.0).abs() > 1.0e-5 {
            return Err(DynamicsError::InvalidDomain(name));
        }
    }
    let cosine: f32 = left
        .iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum::<f32>()
        .clamp(-1.0, 1.0);

    if cosine < -1.0 + 1.0e-6 {
        return Err(DynamicsError::InvalidDomain("antipodal slerp endpoints"));
    }
    if cosine > 1.0 - 1.0e-6 {
        let blended: Vec<f32> = left
            .iter()
            .zip(right.iter())
            .map(|(left, right)| (1.0 - fraction) * left + fraction * right)
            .collect();
        return normalize(&blended);
    }

    let angle = cosine.acos();
    let sine = angle.sin();
    let left_scale = ((1.0 - fraction) * angle).sin() / sine;
    let right_scale = (fraction * angle).sin() / sine;
    let result: Vec<f32> = left
        .iter()
        .zip(right.iter())
        .map(|(left, right)| left_scale * left + right_scale * right)
        .collect();
    normalize(&result)
}

/// Projects `vector` into the tangent space at `point`.
///
/// `point` may be non-unit but must be nonzero; it is normalized internally.
pub fn tangent_project(vector: &[f32], point: &[f32]) -> Result<Vec<f32>> {
    validate_pair("vector", vector, "sphere point", point)?;
    let point = normalize(point)?;
    let dot: f32 = vector
        .iter()
        .zip(&point)
        .map(|(vector, point)| vector * point)
        .sum();
    let result: Vec<f32> = vector
        .iter()
        .zip(point)
        .map(|(vector, point)| vector - dot * point)
        .collect();
    finite_slice("tangent projection", &result)?;
    Ok(result)
}

/// Applies the exact unit-sphere exponential map at `point`.
///
/// `tangent` must be orthogonal to the normalized base point. A zero tangent
/// returns the normalized base point.
pub fn exp_map(point: &[f32], tangent: &[f32], step: f32) -> Result<Vec<f32>> {
    validate_pair("sphere point", point, "tangent", tangent)?;
    finite_scalar("exponential-map step", step)?;
    let point = normalize(point)?;
    let tangent_norm = norm(tangent);
    if !tangent_norm.is_finite() {
        return Err(DynamicsError::NonFinite("tangent norm"));
    }
    let dot: f32 = point
        .iter()
        .zip(tangent)
        .map(|(point, tangent)| point * tangent)
        .sum();
    if dot.abs() > TANGENT_TOLERANCE * tangent_norm.max(1.0) {
        return Err(DynamicsError::InvalidDomain(
            "tangent must be orthogonal to the sphere point",
        ));
    }
    if tangent_norm <= NORM_EPSILON || step == 0.0 {
        return Ok(point);
    }
    let angle = step * tangent_norm;
    let result: Vec<f32> = point
        .iter()
        .zip(tangent)
        .map(|(point, tangent)| angle.cos() * point + angle.sin() * tangent / tangent_norm)
        .collect();
    normalize(&result)
}

/// Evaluates the linear coupling kernel `slope * cos(theta) + intercept`.
pub fn linear_coupling(cosine: f32, slope: f32, intercept: f32) -> Result<f32> {
    finite_scalar("coupling cosine", cosine)?;
    finite_scalar("coupling slope", slope)?;
    finite_scalar("coupling intercept", intercept)?;
    if !(-1.0..=1.0).contains(&cosine) {
        return Err(DynamicsError::InvalidDomain("coupling cosine"));
    }
    let result = slope * cosine + intercept;
    finite_scalar("linear coupling result", result)?;
    Ok(result)
}

/// Computes the mean-field order parameter for row-major sphere points.
///
/// `points` contains `count` vectors of length `dimension`. Each row is
/// normalized before averaging, so nonzero finite inputs need not be unit.
pub fn order_parameter(points: &[f32], count: usize, dimension: usize) -> Result<f32> {
    if count == 0 {
        return Err(DynamicsError::Empty("sphere point set"));
    }
    if dimension == 0 {
        return Err(DynamicsError::Empty("sphere point dimension"));
    }
    let expected = checked_product("sphere points", &[count, dimension])?;
    expect_len("sphere points", points.len(), expected)?;
    finite_slice("sphere points", points)?;

    let mut mean = vec![0.0f32; dimension];
    for point in points.chunks_exact(dimension) {
        let point = normalize(point)?;
        for (mean, component) in mean.iter_mut().zip(point) {
            *mean += component / count as f32;
        }
    }
    let result = norm(&mean);
    finite_scalar("order parameter", result)?;
    Ok(result.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slerp_follows_great_circle() {
        let midpoint = slerp(&[1.0, 0.0], &[0.0, 1.0], 0.5).unwrap();
        let expected = core::f32::consts::FRAC_1_SQRT_2;
        assert!((midpoint[0] - expected).abs() < 1.0e-6);
        assert!((midpoint[1] - expected).abs() < 1.0e-6);
    }

    #[test]
    fn antipodal_slerp_is_rejected() {
        assert!(matches!(
            slerp(&[1.0, 0.0], &[-1.0, 0.0], 0.5),
            Err(DynamicsError::InvalidDomain(_))
        ));
    }

    #[test]
    fn slerp_requires_unit_endpoints() {
        assert!(matches!(
            slerp(&[2.0, 0.0], &[0.0, 1.0], 0.5),
            Err(DynamicsError::InvalidDomain(_))
        ));
    }

    #[test]
    fn projection_and_exp_map_stay_on_sphere() {
        let tangent = tangent_project(&[1.0, 1.0, 0.0], &[1.0, 0.0, 0.0]).unwrap();
        assert_eq!(tangent, vec![0.0, 1.0, 0.0]);
        let next = exp_map(&[1.0, 0.0, 0.0], &tangent, 0.25).unwrap();
        assert!((norm(&next) - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn exp_map_rejects_nontangent_vector() {
        assert!(matches!(
            exp_map(&[1.0, 0.0], &[1.0, 0.0], 0.1),
            Err(DynamicsError::InvalidDomain(_))
        ));
    }

    #[test]
    fn order_parameter_measures_coherence() {
        assert!((order_parameter(&[1.0, 0.0, 1.0, 0.0], 2, 2).unwrap() - 1.0).abs() < 1.0e-6);
        assert!(order_parameter(&[1.0, 0.0, -1.0, 0.0], 2, 2).unwrap() < 1.0e-6);
    }

    #[test]
    fn dimensions_and_domains_are_checked() {
        assert!(matches!(
            tangent_project(&[1.0], &[1.0, 0.0]),
            Err(DynamicsError::Shape { .. })
        ));
        assert!(linear_coupling(1.1, 1.0, 0.0).is_err());
        assert!(order_parameter(&[], 0, 2).is_err());
    }
}
