//! Checked adapters from three-dimensional conformal geometry.
//!
//! This module deliberately delegates semantic decoding to
//! [`clifford_geometry::cga3d::decomposition`]. It never inspects sparse blade
//! slots or guesses whether the shared [`clifford_geometry::cga3d::Pnt`] /
//! [`clifford_geometry::cga3d::Dls`] representation means a point or a sphere.
//! Plane callers must likewise select the entry point matching the semantic
//! direct or compact-dual layout because both layouts share the same Rust type.

use core::fmt;

use clifford_geometry::cga3d::decomposition::{
    decompose_direct_plane, decompose_dual_plane, decompose_point, decompose_real_dual_sphere,
    DecompositionError,
};
use clifford_geometry::cga3d::{Dlp, Dls, Pln, Pnt};

use crate::{MeshError, Plane3, PointCloud, Sphere3};

/// Failure to adapt a checked CGA value into a mesh-domain value.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CgaMeshError {
    /// A value supplied as a direct plane could not be decomposed.
    DirectPlaneDecomposition(DecompositionError),
    /// A value supplied as a compact dual plane could not be decomposed.
    DualPlaneDecomposition(DecompositionError),
    /// A value supplied as a real dual sphere could not be decomposed.
    DualSphereDecomposition(DecompositionError),
    /// A point in a batch could not be decomposed.
    PointDecomposition {
        /// Zero-based index of the invalid point.
        index: usize,
        /// Semantic decomposition failure for that point.
        source: DecompositionError,
    },
    /// The decoded Euclidean value violates a mesh-domain invariant.
    Mesh(MeshError),
}

impl fmt::Display for CgaMeshError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DirectPlaneDecomposition(source) => {
                write!(formatter, "could not decompose direct plane: {source}")
            }
            Self::DualPlaneDecomposition(source) => {
                write!(
                    formatter,
                    "could not decompose compact dual plane: {source}"
                )
            }
            Self::DualSphereDecomposition(source) => {
                write!(formatter, "could not decompose real dual sphere: {source}")
            }
            Self::PointDecomposition { index, source } => {
                write!(
                    formatter,
                    "could not decompose point at index {index}: {source}"
                )
            }
            Self::Mesh(source) => write!(formatter, "invalid mesh-domain geometry: {source}"),
        }
    }
}

impl std::error::Error for CgaMeshError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::DirectPlaneDecomposition(source)
            | Self::DualPlaneDecomposition(source)
            | Self::DualSphereDecomposition(source)
            | Self::PointDecomposition { source, .. } => Some(source),
            Self::Mesh(source) => Some(source),
        }
    }
}

impl From<MeshError> for CgaMeshError {
    fn from(error: MeshError) -> Self {
        Self::Mesh(error)
    }
}

/// Converts a direct CGA plane to an oriented Euclidean descriptor.
///
/// The descriptor point is the point on the plane closest to the Euclidean
/// origin. Positive homogeneous scaling preserves the descriptor; negative
/// scaling preserves the locus and closest point but reverses the normal.
/// Existing plane tessellators consequently reverse their vertex normals and
/// winding. This adapter uses checked direct-plane decomposition and does not
/// pass through the legacy sparse dual projection, so translated offsets are
/// preserved.
///
/// [`Pln`] and [`Dlp`] are aliases of the same Rust type. Callers must select
/// this entry point only for the direct `[e1235, e1245, e1345, e2345]` layout.
pub fn direct_plane(value: &Pln) -> Result<Plane3, CgaMeshError> {
    let decomposition =
        decompose_direct_plane(value).map_err(CgaMeshError::DirectPlaneDecomposition)?;
    Plane3::new(decomposition.closest_point(), decomposition.normal()).map_err(Into::into)
}

/// Converts a compact dual CGA plane to an oriented Euclidean descriptor.
///
/// The compact layout is `[n_x, n_y, n_z, q]` for `n · x + q = 0`. The
/// descriptor point is the point on the plane closest to the Euclidean origin.
/// Positive homogeneous scaling preserves the descriptor; negative scaling
/// preserves the locus and closest point but reverses the normal. Existing
/// plane tessellators consequently reverse their vertex normals and winding.
/// This adapter decodes the compact layout directly and does not use the
/// legacy sparse undual projection, so translated offsets are preserved.
///
/// [`Dlp`] and [`Pln`] are aliases of the same Rust type. Callers must select
/// this entry point only for the compact dual layout.
pub fn dual_plane(value: &Dlp) -> Result<Plane3, CgaMeshError> {
    let decomposition =
        decompose_dual_plane(value).map_err(CgaMeshError::DualPlaneDecomposition)?;
    Plane3::new(decomposition.closest_point(), decomposition.normal()).map_err(Into::into)
}

/// Converts a semantically real CGA dual sphere to a Euclidean descriptor.
///
/// The result describes an unoriented spherical locus: changing the sign of
/// the source's homogeneous scale does not reverse generated mesh normals.
/// Imaginary spheres remain decomposition errors. A radius-zero point sphere
/// is valid CGA, but has no triangle surface, so it returns
/// `CgaMeshError::Mesh(MeshError::NonPositive("sphere radius"))`.
pub fn real_dual_sphere(value: &Dls) -> Result<Sphere3, CgaMeshError> {
    let decomposition =
        decompose_real_dual_sphere(value).map_err(CgaMeshError::DualSphereDecomposition)?;
    Sphere3::new(decomposition.center(), decomposition.radius()).map_err(Into::into)
}

/// Converts semantic CGA points into an ordered Euclidean point cloud.
///
/// Conversion is atomic: the first invalid input returns its stable, zero-based
/// index and no partial cloud. An empty input produces an empty point cloud.
pub fn point_cloud(values: &[Pnt]) -> Result<PointCloud, CgaMeshError> {
    let positions = values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            decompose_point(value)
                .map(|decomposition| decomposition.position())
                .map_err(|source| CgaMeshError::PointDecomposition { index, source })
        })
        .collect::<Result<Vec<_>, _>>()?;
    PointCloud::try_from_positions(positions).map_err(Into::into)
}
