//! Checked conversion from sparse CGA vectors to Euclidean parameters.
//!
//! [`Pnt`] / [`Dls`] and [`Dlp`] / [`Pln`] are representation aliases, so these
//! functions require the caller to supply the semantic form and layout named
//! by the function. They accept finite, representable nonzero homogeneous
//! scaling when the scaled `f32` coefficients retain the element's semantics.
//! Point and sphere decomposition uses sparse basis `[e1, e2, e3, e4, e5]`,
//! metric `[+1, +1, +1, +1, -1]`, and homogeneous weight `w = e5 - e4`.
//! Plane decomposition produces normalized Hesse form `normal · x + d = 0`.
//! Input coefficients are widened to `f64` before subtraction, products, and
//! division; public results retain the crate's established `f32` precision.

use core::fmt;

use super::{point, Dlp, Dls, Pln, Pnt};

/// Failure to decompose a conformal vector into finite Euclidean parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DecompositionError {
    /// At least one input coefficient was NaN or infinity.
    NonFiniteInput,
    /// The conformal vector had zero homogeneous weight.
    DegenerateHomogeneousWeight,
    /// Decomposition produced a value unsupported by the finite `f32` output contract.
    OutputOutOfRange,
    /// A value supplied as a conformal point did not match canonical point form.
    NonNullPoint,
    /// A dual sphere had negative squared radius.
    ImaginaryDualSphere,
    /// A plane had no finite nonzero Euclidean normal.
    DegeneratePlaneNormal,
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
                formatter.write_str("CGA decomposition cannot be represented by its f32 output")
            }
            Self::NonNullPoint => {
                formatter.write_str("CGA point does not have canonical null-point form")
            }
            Self::ImaginaryDualSphere => {
                formatter.write_str("CGA dual sphere has imaginary radius")
            }
            Self::DegeneratePlaneNormal => {
                formatter.write_str("CGA plane must have a nonzero Euclidean normal")
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

/// Finite normalized Euclidean parameters extracted from a CGA plane.
///
/// The returned values use Hesse form `normal · x + d = 0`. The normal has
/// unit length and `d` is the signed distance obtained by evaluating that
/// equation at the origin. Negating the source's homogeneous scale preserves
/// the plane locus and closest point while reversing the normal and `d`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlaneDecomposition {
    closest_point: [f32; 3],
    normal: [f32; 3],
    signed_distance_from_origin: f32,
}

impl PlaneDecomposition {
    /// Point on the plane nearest the Euclidean origin.
    pub const fn closest_point(self) -> [f32; 3] {
        self.closest_point
    }

    /// Representative-oriented unit normal.
    pub const fn normal(self) -> [f32; 3] {
        self.normal
    }

    /// Signed distance term `d` in `normal · x + d = 0`.
    pub const fn signed_distance_from_origin(self) -> f32 {
        self.signed_distance_from_origin
    }
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

/// Decomposes a compact semantic dual plane into normalized Euclidean form.
///
/// The input layout is `[n_x, n_y, n_z, q]`, representing the equation
/// `n · x + q = 0`. This compact public layout is decoded directly; it must not
/// be preprocessed with [`super::undual_dlp`] because the legacy sparse dual
/// mapping drops its semantic offset. A finite exactly-zero normal is rejected
/// without an epsilon fallback. A nonzero normalized offset too small or large
/// for the public `f32` result is rejected rather than rounded onto a different
/// plane locus.
pub fn decompose_dual_plane(value: &Dlp) -> Result<PlaneDecomposition, DecompositionError> {
    let coefficients = finite_plane_coefficients(value)?;
    decompose_plane_coefficients(
        [coefficients[0], coefficients[1], coefficients[2]],
        coefficients[3],
    )
}

/// Decomposes a direct CGA plane into normalized Euclidean form.
///
/// The direct layout is `[e1235, e1245, e1345, e2345]`. Its raw Euclidean
/// equation has `normal = [e2345, -e1345, e1245]` and offset `e1235`.
/// Reading this layout directly preserves translated-plane offsets that the
/// legacy [`super::dual_pln`] projection drops. A finite exactly-zero normal
/// is rejected without an epsilon fallback. A nonzero normalized offset too
/// small or large for the public `f32` result is rejected rather than rounded
/// onto a different plane locus.
pub fn decompose_direct_plane(value: &Pln) -> Result<PlaneDecomposition, DecompositionError> {
    let coefficients = finite_plane_coefficients(value)?;
    decompose_plane_coefficients(
        [coefficients[3], -coefficients[2], coefficients[1]],
        coefficients[0],
    )
}

fn decompose_plane_coefficients(
    normal: [f64; 3],
    offset: f64,
) -> Result<PlaneDecomposition, DecompositionError> {
    let scale = normal
        .iter()
        .map(|component| component.abs())
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return Err(DecompositionError::DegeneratePlaneNormal);
    }
    let scaled_normal = normal.map(|component| component / scale);
    let scaled_length = scaled_normal
        .iter()
        .map(|component| component * component)
        .sum::<f64>()
        .sqrt();
    let unit_normal_f64 = scaled_normal.map(|component| component / scaled_length);
    let signed_distance_f64 = (offset / scale) / scaled_length;
    let closest_point_f64 = unit_normal_f64.map(|component| -signed_distance_f64 * component);

    let normal = finite_f32_array(unit_normal_f64)?;
    let closest_point = finite_f32_array(closest_point_f64)?;
    let signed_distance_from_origin = finite_f32_scalar(signed_distance_f64)?;
    Ok(PlaneDecomposition {
        closest_point,
        normal,
        signed_distance_from_origin,
    })
}

fn finite_f32_scalar(value: f64) -> Result<f32, DecompositionError> {
    let narrowed = value as f32;
    if !narrowed.is_finite() || (value != 0.0 && narrowed == 0.0) {
        Err(DecompositionError::OutputOutOfRange)
    } else {
        Ok(narrowed)
    }
}

fn finite_f32_array<const N: usize>(values: [f64; N]) -> Result<[f32; N], DecompositionError> {
    let values = values.map(|value| value as f32);
    if values.iter().any(|value| !value.is_finite()) {
        Err(DecompositionError::OutputOutOfRange)
    } else {
        Ok(values)
    }
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

fn finite_plane_coefficients(value: &Pln) -> Result<[f64; 4], DecompositionError> {
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
