use crate::math::{ensure_finite, ensure_positive, normalize, Vec3};
use crate::Result;

/// A Euclidean sphere with a finite center and positive radius.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sphere3 {
    center: Vec3,
    radius: f32,
}

impl Sphere3 {
    /// Constructs a checked sphere.
    pub fn new(center: Vec3, radius: f32) -> Result<Self> {
        Ok(Self {
            center: ensure_finite("sphere center", center)?,
            radius: ensure_positive("sphere radius", radius)?,
        })
    }

    /// Sphere center.
    pub const fn center(self) -> Vec3 {
        self.center
    }

    /// Sphere radius.
    pub const fn radius(self) -> f32 {
        self.radius
    }
}

/// A Euclidean circle with a finite center, unit normal, and positive radius.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Circle3 {
    center: Vec3,
    normal: Vec3,
    radius: f32,
}

impl Circle3 {
    /// Constructs a checked circle and normalizes its normal.
    pub fn new(center: Vec3, normal: Vec3, radius: f32) -> Result<Self> {
        Ok(Self {
            center: ensure_finite("circle center", center)?,
            normal: normalize("circle normal", normal)?,
            radius: ensure_positive("circle radius", radius)?,
        })
    }

    /// Circle center.
    pub const fn center(self) -> Vec3 {
        self.center
    }

    /// Unit normal to the circle plane.
    pub const fn normal(self) -> Vec3 {
        self.normal
    }

    /// Circle radius.
    pub const fn radius(self) -> f32 {
        self.radius
    }
}

/// A Euclidean plane represented by a point and unit normal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plane3 {
    point: Vec3,
    normal: Vec3,
}

impl Plane3 {
    /// Constructs a checked plane and normalizes its normal.
    pub fn new(point: Vec3, normal: Vec3) -> Result<Self> {
        Ok(Self {
            point: ensure_finite("plane point", point)?,
            normal: normalize("plane normal", normal)?,
        })
    }

    /// A point on the plane.
    pub const fn point(self) -> Vec3 {
        self.point
    }

    /// Unit plane normal.
    pub const fn normal(self) -> Vec3 {
        self.normal
    }
}

/// An oriented Euclidean line represented by a point and unit direction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Line3 {
    point: Vec3,
    direction: Vec3,
}

impl Line3 {
    /// Constructs a checked line and normalizes its direction.
    pub fn new(point: Vec3, direction: Vec3) -> Result<Self> {
        Ok(Self {
            point: ensure_finite("line point", point)?,
            direction: normalize("line direction", direction)?,
        })
    }

    /// A point on the line.
    pub const fn point(self) -> Vec3 {
        self.point
    }

    /// Unit line direction.
    pub const fn direction(self) -> Vec3 {
        self.direction
    }
}
