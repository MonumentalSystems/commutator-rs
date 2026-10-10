//! Checked conversion from sparse CGA vectors to Euclidean parameters.
//!
//! [`Pnt`] and [`Dls`] are representation aliases, so these functions require
//! the caller to supply the semantic form named by the function. They accept
//! finite, representable nonzero homogeneous scaling, including negative
//! scaling, when the scaled `f32` coefficients retain the element's semantics.
//! The sparse basis is `[e1, e2, e3, e4, e5]`, with metric
//! `[+1, +1, +1, +1, -1]` and homogeneous weight `w = e5 - e4`.
//! Input coefficients are widened to `f64` before subtraction, products, and
//! division; public results retain the crate's established `f32` precision.

use core::fmt;

use super::{point, Dls, Pnt};

/// Failure to decompose a conformal vector into finite Euclidean parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecompositionError {
    /// At least one input coefficient was NaN or infinity.
    NonFiniteInput,
    /// The conformal vector had zero homogeneous weight.
    DegenerateHomogeneousWeight,
    /// Decomposition produced a value outside finite `f32` range.
    OutputOutOfRange,
    /// A value supplied as a conformal point did not match canonical point form.
    NonNullPoint,
    /// A dual sphere had negative squared radius.
    ImaginaryDualSphere,
}

impl fmt::Display for DecompositionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteInput => {
                formatter.write_str("CGA coefficients must contain only finite values")
            }
            Self::DegenerateHomogeneousWeight => {
                formatter.write_str("CGA vector has zero homogeneous weight")
            }
            Self::OutputOutOfRange => {
                formatter.write_str("CGA decomposition is outside finite f32 range")
            }
            Self::NonNullPoint => {
                formatter.write_str("CGA point does not have canonical null-point form")
            }
            Self::ImaginaryDualSphere => {
                formatter.write_str("CGA dual sphere has imaginary radius")
            }
        }
    }
}

impl std::error::Error for DecompositionError {}

/// Finite Euclidean coordinates extracted from a conformal point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointDecomposition {
    position: [f32; 3],
}

impl PointDecomposition {
    /// Returns `[x, y, z]` in the caller's Euclidean units.
    pub const fn position(self) -> [f32; 3] {
        self.position
    }
}

/// Finite Euclidean parameters extracted from a real dual sphere.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DualSphereDecomposition {
    center: [f32; 3],
    radius: f32,
}

impl DualSphereDecomposition {
    /// Sphere center in the caller's Euclidean units.
    pub const fn center(self) -> [f32; 3] {
        self.center
    }

    /// Nonnegative sphere radius in the caller's Euclidean units.
    pub const fn radius(self) -> f32 {
        self.radius
    }
}

/// Decomposes a semantically conformal point into finite Euclidean coordinates.
///
/// The input must be a null grade-one CGA vector with nonzero homogeneous
/// weight. After removing homogeneous scale, the function reconstructs the
/// point through this crate's canonical [`point`] constructor and requires the
/// represented coefficients to match. This accepts ordinary decimal values
/// produced by the public constructor while rejecting every sphere offset that
/// remains distinguishable in `f32`. The operation is allocation-free and
/// leaves the original sparse value unchanged.
pub fn decompose_point(value: &Pnt) -> Result<PointDecomposition, DecompositionError> {
    let coefficients = finite_coefficients(value)?;
    let weight = homogeneous_weight(&coefficients)?;
    let normalized = normalized_coefficients(&coefficients, weight)?;
    let position = [normalized[0], normalized[1], normalized[2]];
    if point(position[0], position[1], position[2]).data != normalized {
        return Err(DecompositionError::NonNullPoint);
    }
    Ok(PointDecomposition { position })
}

/// Decomposes a semantically real dual sphere into center and radius.
///
/// A null dual sphere is accepted as a radius-zero point sphere. Radius squared
/// is measured as the stored `e4 + e5` offset from the canonical point at the
/// decomposed center. This removes the constructor's own `f32` quantization
/// residual without hiding any representably distinct sphere offset. A
/// negative offset is rejected as [`DecompositionError::ImaginaryDualSphere`]
/// rather than being converted with an absolute value. Infinite elements with
/// zero homogeneous weight are outside this API. The operation is
/// allocation-free.
pub fn decompose_real_dual_sphere(
    sphere: &Dls,
) -> Result<DualSphereDecomposition, DecompositionError> {
    let coefficients = finite_coefficients(sphere)?;
    let weight = homogeneous_weight(&coefficients)?;
    let normalized = normalized_coefficients(&coefficients, weight)?;
    let center = [normalized[0], normalized[1], normalized[2]];
    let canonical = point(center[0], center[1], center[2]);
    let radius_squared = (f64::from(canonical[3]) - f64::from(normalized[3]))
        + (f64::from(canonical[4]) - f64::from(normalized[4]));
    if radius_squared < 0.0 {
        return Err(DecompositionError::ImaginaryDualSphere);
    }
    let radius = radius_squared.sqrt() as f32;
    if !radius.is_finite() {
        return Err(DecompositionError::OutputOutOfRange);
    }
    Ok(DualSphereDecomposition { center, radius })
}

fn finite_coefficients(value: &Pnt) -> Result<[f64; 5], DecompositionError> {
    if value
        .data
        .iter()
        .any(|coefficient| !coefficient.is_finite())
    {
        return Err(DecompositionError::NonFiniteInput);
    }
    Ok(core::array::from_fn(|index| f64::from(value[index])))
}

fn homogeneous_weight(coefficients: &[f64; 5]) -> Result<f64, DecompositionError> {
    let weight = coefficients[4] - coefficients[3];
    if weight == 0.0 {
        Err(DecompositionError::DegenerateHomogeneousWeight)
    } else {
        Ok(weight)
    }
}

fn normalized_coefficients(
    coefficients: &[f64; 5],
    weight: f64,
) -> Result<[f32; 5], DecompositionError> {
    let normalized = core::array::from_fn(|index| (coefficients[index] / weight) as f32);
    if normalized
        .iter()
        .any(|coefficient| !coefficient.is_finite())
    {
        Err(DecompositionError::OutputOutOfRange)
    } else {
        Ok(normalized)
    }
}
