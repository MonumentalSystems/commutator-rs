#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod descriptors;
mod error;
mod math;
mod mesh;
mod options;

/// Checked adapters from conformal geometric-algebra values to mesh inputs.
#[cfg(feature = "cga3d")]
pub mod cga3d;

/// Checked Euclidean tessellation functions.
pub mod primitives;

pub use descriptors::{Circle3, Line3, Plane3, Sphere3};
pub use error::MeshError;
pub use mesh::{LineSegments, PointCloud, TriangleMesh, Vertex};
pub use options::{
    CircleOptions, CylinderOptions, IcosphereOptions, PlanePatchOptions, TubeOptions,
    UvSphereOptions,
};

/// Result type returned by checked mesh operations.
pub type Result<T> = core::result::Result<T, MeshError>;

/// Minimum supported segment count for closed circular geometry.
pub const MIN_CIRCULAR_SEGMENTS: u32 = 3;
/// Minimum supported stack count for a UV sphere.
pub const MIN_UV_SPHERE_STACKS: u32 = 2;
/// Minimum supported subdivision count along either plane-patch axis.
pub const MIN_PLANE_SEGMENTS: u32 = 1;
/// Maximum supported circular or grid resolution.
pub const MAX_RESOLUTION: u32 = 1_024;
/// Maximum supported icosphere subdivision count.
pub const MAX_ICOSPHERE_SUBDIVISIONS: u32 = 7;
