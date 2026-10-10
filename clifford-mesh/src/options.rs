use crate::{
    MeshError, Result, MAX_ICOSPHERE_SUBDIVISIONS, MAX_RESOLUTION, MIN_CIRCULAR_SEGMENTS,
    MIN_PLANE_SEGMENTS, MIN_UV_SPHERE_STACKS,
};

fn resolution(value: u32, minimum: u32) -> Result<u32> {
    if (minimum..=MAX_RESOLUTION).contains(&value) {
        Ok(value)
    } else {
        Err(MeshError::ResolutionOutOfRange {
            requested: value,
            minimum,
            maximum: MAX_RESOLUTION,
        })
    }
}

/// Latitude and longitude resolution for a UV sphere.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UvSphereOptions {
    stacks: u32,
    slices: u32,
}

impl UvSphereOptions {
    /// Constructs checked stack and slice counts.
    pub fn new(stacks: u32, slices: u32) -> Result<Self> {
        Ok(Self {
            stacks: resolution(stacks, MIN_UV_SPHERE_STACKS)?,
            slices: resolution(slices, MIN_CIRCULAR_SEGMENTS)?,
        })
    }

    /// Uses the same resolution for stacks and slices.
    pub fn uniform(resolution_value: u32) -> Result<Self> {
        Self::new(resolution_value, resolution_value)
    }

    /// Number of latitude intervals.
    pub const fn stacks(self) -> u32 {
        self.stacks
    }

    /// Number of longitude intervals.
    pub const fn slices(self) -> u32 {
        self.slices
    }
}

impl Default for UvSphereOptions {
    fn default() -> Self {
        Self {
            stacks: 16,
            slices: 32,
        }
    }
}

/// Subdivision count for an icosphere.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IcosphereOptions {
    subdivisions: u32,
}

impl IcosphereOptions {
    /// Constructs a checked subdivision count.
    pub fn new(subdivisions: u32) -> Result<Self> {
        if subdivisions > MAX_ICOSPHERE_SUBDIVISIONS {
            Err(MeshError::SubdivisionOutOfRange {
                requested: subdivisions,
                maximum: MAX_ICOSPHERE_SUBDIVISIONS,
            })
        } else {
            Ok(Self { subdivisions })
        }
    }

    /// Number of recursive face subdivisions.
    pub const fn subdivisions(self) -> u32 {
        self.subdivisions
    }
}

impl Default for IcosphereOptions {
    fn default() -> Self {
        Self { subdivisions: 2 }
    }
}

/// Segment count for a circle loop or disc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CircleOptions {
    segments: u32,
}

impl CircleOptions {
    /// Constructs a checked segment count.
    pub fn new(segments: u32) -> Result<Self> {
        Ok(Self {
            segments: resolution(segments, MIN_CIRCULAR_SEGMENTS)?,
        })
    }

    /// Number of segments around the circle.
    pub const fn segments(self) -> u32 {
        self.segments
    }
}

impl Default for CircleOptions {
    fn default() -> Self {
        Self { segments: 32 }
    }
}

/// Main-ring and cross-section resolution for a tube.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TubeOptions {
    ring_segments: u32,
    cross_segments: u32,
}

impl TubeOptions {
    /// Constructs checked ring and cross-section segment counts.
    pub fn new(ring_segments: u32, cross_segments: u32) -> Result<Self> {
        Ok(Self {
            ring_segments: resolution(ring_segments, MIN_CIRCULAR_SEGMENTS)?,
            cross_segments: resolution(cross_segments, MIN_CIRCULAR_SEGMENTS)?,
        })
    }

    /// Segments around the main ring.
    pub const fn ring_segments(self) -> u32 {
        self.ring_segments
    }

    /// Segments around the tube cross-section.
    pub const fn cross_segments(self) -> u32 {
        self.cross_segments
    }
}

impl Default for TubeOptions {
    fn default() -> Self {
        Self {
            ring_segments: 32,
            cross_segments: 12,
        }
    }
}

/// Two-axis subdivision counts for a plane patch or grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanePatchOptions {
    u_segments: u32,
    v_segments: u32,
}

impl PlanePatchOptions {
    /// Constructs checked two-axis subdivision counts.
    pub fn new(u_segments: u32, v_segments: u32) -> Result<Self> {
        Ok(Self {
            u_segments: resolution(u_segments, MIN_PLANE_SEGMENTS)?,
            v_segments: resolution(v_segments, MIN_PLANE_SEGMENTS)?,
        })
    }

    /// Number of intervals along the local u axis.
    pub const fn u_segments(self) -> u32 {
        self.u_segments
    }

    /// Number of intervals along the local v axis.
    pub const fn v_segments(self) -> u32 {
        self.v_segments
    }
}

impl Default for PlanePatchOptions {
    fn default() -> Self {
        Self {
            u_segments: 8,
            v_segments: 8,
        }
    }
}

/// Segment count for an open cylinder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CylinderOptions {
    radial_segments: u32,
}

impl CylinderOptions {
    /// Constructs a checked radial segment count.
    pub fn new(radial_segments: u32) -> Result<Self> {
        Ok(Self {
            radial_segments: resolution(radial_segments, MIN_CIRCULAR_SEGMENTS)?,
        })
    }

    /// Segments around the cylinder circumference.
    pub const fn radial_segments(self) -> u32 {
        self.radial_segments
    }
}

impl Default for CylinderOptions {
    fn default() -> Self {
        Self {
            radial_segments: 24,
        }
    }
}
