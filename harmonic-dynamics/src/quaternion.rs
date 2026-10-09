//! Typed quaternion algebra and maps on the unit three-sphere.

use crate::error::{expect_len, finite_scalar, finite_slice};
use crate::{DynamicsError, Result};

const NORM_EPSILON: f32 = 1.0e-12;

/// A finite quaternion in scalar-first `[w, x, y, z]` order.
///
/// Construction validates finiteness. Operations that require an element of
/// `S^3` normalize explicitly rather than assuming unit length.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(transparent)]
pub struct Quaternion([f32; 4]);

impl Quaternion {
    /// Multiplicative identity.
    pub const IDENTITY: Self = Self([1.0, 0.0, 0.0, 0.0]);

    /// Constructs a finite quaternion from scalar-first components.
    pub fn new(w: f32, x: f32, y: f32, z: f32) -> Result<Self> {
        Self::from_array([w, x, y, z])
    }

    /// Constructs a finite quaternion from `[w, x, y, z]`.
    pub fn from_array(components: [f32; 4]) -> Result<Self> {
        finite_slice("quaternion components", &components)?;
        Ok(Self(components))
    }

    /// Returns a shared reference to `[w, x, y, z]`.
    pub const fn as_array(&self) -> &[f32; 4] {
        &self.0
    }

    /// Returns the scalar-first component array.
    pub const fn into_array(self) -> [f32; 4] {
        self.0
    }

    /// Returns the squared Euclidean norm.
    pub fn norm_squared(self) -> f32 {
        self.0.iter().map(|value| value * value).sum()
    }

    /// Returns the Euclidean norm.
    pub fn norm(self) -> f32 {
        self.norm_squared().sqrt()
    }

    /// Projects this quaternion to `S^3`.
    pub fn normalized(self) -> Result<Self> {
        let norm = self.norm();
        if !norm.is_finite() {
            return Err(DynamicsError::NonFinite("quaternion norm"));
        }
        if norm <= NORM_EPSILON {
            return Err(DynamicsError::ZeroNorm("quaternion"));
        }
        let inverse = norm.recip();
        Self::from_array(self.0.map(|value| value * inverse))
    }

    /// Returns the Hamilton product `self * rhs`.
    pub fn hamilton_product(self, rhs: Self) -> Result<Self> {
        let [w1, x1, y1, z1] = self.0;
        let [w2, x2, y2, z2] = rhs.0;
        Self::from_array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])
    }

    /// Returns the quaternion conjugate `[w, -x, -y, -z]`.
    pub fn conjugate(self) -> Self {
        Self([self.0[0], -self.0[1], -self.0[2], -self.0[3]])
    }

    /// Returns the Euclidean inner product with `rhs`.
    pub fn dot(self, rhs: Self) -> Result<f32> {
        let value = self
            .0
            .iter()
            .zip(rhs.0)
            .map(|(left, right)| left * right)
            .sum();
        finite_scalar("quaternion inner product", value)?;
        Ok(value)
    }

    /// Projects a four-vector onto the tangent space at this quaternion.
    ///
    /// The base quaternion is normalized before projection.
    pub fn tangent_projection(self, vector: [f32; 4]) -> Result<[f32; 4]> {
        finite_slice("tangent vector", &vector)?;
        let point = self.normalized()?;
        let dot: f32 = vector
            .iter()
            .zip(point.0)
            .map(|(left, right)| left * right)
            .sum();
        let projected = core::array::from_fn(|index| vector[index] - dot * point.0[index]);
        finite_slice("projected tangent", &projected)?;
        Ok(projected)
    }

    /// Applies the logarithmic map at the identity, returning an `R^3` vector.
    ///
    /// The antipodal identity `[-1, 0, 0, 0]` has no unique logarithm and is
    /// rejected.
    pub fn log(self) -> Result<[f32; 3]> {
        let unit = self.normalized()?;
        let [w, x, y, z] = unit.0;
        let vector_norm = (x * x + y * y + z * z).sqrt();
        if vector_norm == 0.0 {
            if w < 0.0 {
                return Err(DynamicsError::InvalidDomain(
                    "logarithm at the antipodal identity",
                ));
            }
            return Ok([0.0; 3]);
        }
        // atan2 retains small angles when `w` has rounded to exactly one in
        // f32, unlike acos(w).
        let angle = vector_norm.atan2(w);
        let scale = angle / vector_norm;
        let result = [x * scale, y * scale, z * scale];
        finite_slice("quaternion logarithm", &result)?;
        Ok(result)
    }

    /// Applies the exponential map from `R^3` at the identity.
    pub fn exp(vector: [f32; 3]) -> Result<Self> {
        finite_slice("quaternion exponential input", &vector)?;
        let theta = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
        if !theta.is_finite() {
            return Err(DynamicsError::NonFinite("quaternion exponential norm"));
        }
        if theta == 0.0 {
            return Ok(Self::IDENTITY);
        }
        let scale = theta.sin() / theta;
        Self::from_array([
            theta.cos(),
            vector[0] * scale,
            vector[1] * scale,
            vector[2] * scale,
        ])?
        .normalized()
    }

    /// Constructs a unit quaternion from an angle and a unit axis.
    pub fn from_axis_angle(angle: f32, axis: [f32; 3]) -> Result<Self> {
        finite_scalar("rotation angle", angle)?;
        finite_slice("rotation axis", &axis)?;
        let norm = axis.iter().map(|value| value * value).sum::<f32>().sqrt();
        if !norm.is_finite() {
            return Err(DynamicsError::NonFinite("rotation axis norm"));
        }
        if norm <= NORM_EPSILON {
            return Err(DynamicsError::ZeroNorm("rotation axis"));
        }
        if (norm - 1.0).abs() > 1.0e-5 {
            return Err(DynamicsError::InvalidDomain(
                "rotation axis must have unit norm",
            ));
        }
        let half = angle * 0.5;
        let scale = half.sin();
        Self::from_array([
            half.cos(),
            axis[0] * scale,
            axis[1] * scale,
            axis[2] * scale,
        ])?
        .normalized()
    }

    /// Takes the source-compatible first-order geodesic step toward `target`.
    pub fn geodesic_step(self, target: Self, step: f32) -> Result<Self> {
        finite_scalar("geodesic step", step)?;
        let point = self.normalized()?;
        let target = target.normalized()?;
        let difference = core::array::from_fn(|index| target.0[index] - point.0[index]);
        let tangent = point.tangent_projection(difference)?;
        let next = core::array::from_fn(|index| point.0[index] + step * tangent[index]);
        Self::from_array(next)?.normalized()
    }
}

impl TryFrom<[f32; 4]> for Quaternion {
    type Error = DynamicsError;

    fn try_from(value: [f32; 4]) -> Result<Self> {
        Self::from_array(value)
    }
}

impl From<Quaternion> for [f32; 4] {
    fn from(value: Quaternion) -> Self {
        value.into_array()
    }
}

/// Computes a normalized mean field for each quaternion state.
///
/// Without `adjacency`, every output is the global uniform mean. With an
/// adjacency matrix, row `i` supplies the weights used for output `i`.
pub fn mean_field(states: &[Quaternion], adjacency: Option<&[f32]>) -> Result<Vec<Quaternion>> {
    if states.is_empty() {
        return Err(DynamicsError::Empty("quaternion states"));
    }
    let count = states.len();
    if let Some(weights) = adjacency {
        let expected = count
            .checked_mul(count)
            .ok_or(DynamicsError::SizeOverflow("adjacency"))?;
        expect_len("adjacency", weights.len(), expected)?;
        finite_slice("adjacency", weights)?;
    }

    let normalized = states
        .iter()
        .copied()
        .map(Quaternion::normalized)
        .collect::<Result<Vec<_>>>()?;
    let mut output = Vec::with_capacity(count);
    for row in 0..count {
        let mut sum = [0.0f32; 4];
        for (column, state) in normalized.iter().enumerate() {
            let weight = adjacency
                .map(|weights| weights[row * count + column])
                .unwrap_or(1.0 / count as f32);
            for (sum_component, state_component) in sum.iter_mut().zip(state.0) {
                *sum_component += weight * state_component;
            }
        }
        output.push(Quaternion::from_array(sum)?.normalized()?);
    }
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
    fn hamilton_basis_product() {
        let i = Quaternion::new(0.0, 1.0, 0.0, 0.0).unwrap();
        let j = Quaternion::new(0.0, 0.0, 1.0, 0.0).unwrap();
        assert_eq!(
            i.hamilton_product(j).unwrap().into_array(),
            [0.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn identity_and_conjugate() {
        let q = Quaternion::new(0.5, 0.5, 0.5, 0.5).unwrap();
        assert_eq!(Quaternion::IDENTITY.hamilton_product(q).unwrap(), q);
        assert_eq!(q.conjugate().into_array(), [0.5, -0.5, -0.5, -0.5]);
    }

    #[test]
    fn exp_log_round_trip() {
        let input = [0.1, 0.2, 0.3];
        let recovered = Quaternion::exp(input).unwrap().log().unwrap();
        assert!(close(&input, &recovered, 1.0e-5));
    }

    #[test]
    fn exp_log_small_angles_are_not_epsilon_distorted() {
        for magnitude in [0.0, 1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2] {
            let input = [magnitude, 0.0, 0.0];
            let quaternion = Quaternion::exp(input).unwrap();
            if magnitude == 0.0 {
                assert_eq!(quaternion, Quaternion::IDENTITY);
            } else {
                let recovered = quaternion.log().unwrap();
                assert!((recovered[0] - magnitude).abs() <= 1.0e-6 * magnitude.max(1.0));
            }
        }
    }

    #[test]
    fn tangent_is_orthogonal() {
        let point = Quaternion::new(1.0, 1.0, 0.0, 0.0)
            .unwrap()
            .normalized()
            .unwrap();
        let tangent = point.tangent_projection([1.0, 0.0, 1.0, 0.0]).unwrap();
        let dot: f32 = point
            .as_array()
            .iter()
            .zip(tangent)
            .map(|(left, right)| left * right)
            .sum();
        assert!(dot.abs() < 1.0e-6);
    }

    #[test]
    fn unit_axis_is_validated() {
        let q = Quaternion::from_axis_angle(core::f32::consts::PI, [0.0, 0.0, 1.0]).unwrap();
        assert!(close(q.as_array(), &[0.0, 0.0, 0.0, 1.0], 1.0e-6));
        assert!(matches!(
            Quaternion::from_axis_angle(1.0, [0.0; 3]),
            Err(DynamicsError::ZeroNorm(_))
        ));
        assert!(matches!(
            Quaternion::from_axis_angle(1.0, [0.0, 0.0, 2.0]),
            Err(DynamicsError::InvalidDomain(_))
        ));
    }

    #[test]
    fn mean_field_validates_shape_and_zero_mean() {
        let states = [
            Quaternion::IDENTITY,
            Quaternion::new(0.0, 1.0, 0.0, 0.0).unwrap(),
        ];
        let field = mean_field(&states, None).unwrap();
        assert_eq!(field.len(), 2);
        assert!(field
            .iter()
            .all(|value| (value.norm() - 1.0).abs() < 1.0e-6));
        assert!(matches!(
            mean_field(&states, Some(&[1.0, 0.0])),
            Err(DynamicsError::Shape { .. })
        ));
        assert!(matches!(
            mean_field(
                &[
                    Quaternion::IDENTITY,
                    Quaternion::new(-1.0, 0.0, 0.0, 0.0).unwrap()
                ],
                None
            ),
            Err(DynamicsError::ZeroNorm(_))
        ));
    }

    #[test]
    fn nonfinite_and_zero_quaternions_are_rejected() {
        assert!(matches!(
            Quaternion::new(f32::NAN, 0.0, 0.0, 0.0),
            Err(DynamicsError::NonFinite(_))
        ));
        assert!(matches!(
            Quaternion::new(0.0, 0.0, 0.0, 0.0).unwrap().normalized(),
            Err(DynamicsError::ZeroNorm(_))
        ));
    }
}
