//! Twist / screw geometry: dual line + pitch.
//!
//! Port of `vsr_twist.h`. A Twist decomposes a dual line into its rotational
//! (bivector) and translational (direction vector) components, parameterized
//! by period and pitch.

use crate::cga3d::*;
use crate::mvec::Multivector;

/// A twist (screw motion) defined by a dual line axis.
///
/// The dual line encodes both the rotation axis (Euclidean bivector part)
/// and the translation moment (direction vector part). Period controls
/// rotation speed, pitch controls translation-per-rotation ratio.
#[derive(Clone, Copy, Debug)]
pub struct Twist {
    /// The dual line axis `[e12, e13, e23, e15, e25, e35]`.
    pub axis: Dll,
    /// Extrapolation parameter (multiplier for the twist).
    pub ext: f32,
}

impl Default for Twist {
    fn default() -> Self {
        Self::new()
    }
}

impl Twist {
    /// Default twist: rotation in XZ plane with translation along Y.
    pub fn new() -> Self {
        Self {
            axis: dll(0.0, 1.0, 0.0, 0.0, -1.0, 0.0),
            ext: 1.0,
        }
    }

    /// Create a twist from a direction vector (x, y, z).
    ///
    /// The rotation plane is the dual of the direction, and the translation
    /// is along the direction.
    pub fn from_direction(x: f32, y: f32, z: f32) -> Self {
        // Dual of (x,y,z) in Cl(3,0): e1->e23, e2->-e13, e3->e12
        // So dual of direction = bivector
        let b = biv(z, -y, x); // dual mapping
        let d = Multivector::new([-x, -y, -z]); // negative direction
        Self {
            axis: dll(b[0], b[1], b[2], d[0], d[1], d[2]),
            ext: 1.0,
        }
    }

    /// Create a twist along a given dual line with specified period and pitch.
    ///
    /// - `line`: the axis (dual line)
    /// - `period`: rotation amount (radians worth of rotation)
    /// - `pitch`: translation per rotation
    pub fn along(line: &Dll, period: f32, pitch: f32) -> Dll {
        let norm = (line[0] * line[0] + line[1] * line[1] + line[2] * line[2]).sqrt();
        if norm < 1e-10 {
            return *line;
        }
        let inv = 1.0 / norm;
        // Normalized dual line
        let b0 = line[0] * inv * period;
        let b1 = line[1] * inv * period;
        let b2 = line[2] * inv * period;
        let m0 = line[3] * inv * period;
        let m1 = line[4] * inv * period;
        let m2 = line[5] * inv * period;

        // Direction (dual of the bivector part): biv -> vec
        let dir_x = b2;
        let dir_y = -b1;
        let dir_z = b0;

        // Modify moment by pitch contribution
        let drv0 = m0 + dir_x * pitch;
        let drv1 = m1 + dir_y * pitch;
        let drv2 = m2 + dir_z * pitch;

        dll(b0, b1, b2, drv0, drv1, drv2)
    }

    /// Get the bivector (rotation) part of the twist.
    pub fn bivector(&self) -> Biv {
        Flat::direction(&self.axis)
    }

    /// Get the direction (translation) part of the twist.
    pub fn direction(&self) -> Drv {
        Flat::moment(&self.axis)
    }

    /// Period (rotation magnitude) of the twist.
    pub fn period(&self) -> f32 {
        let b = self.bivector();
        b.norm()
    }

    /// Pitch (translation magnitude relative to rotation) of the twist.
    pub fn pitch(&self) -> f32 {
        let d = self.direction();
        d.norm()
    }

    /// Get the motor at parameter t along the twist.
    pub fn motor_at(&self, t: f32) -> Mot {
        let scaled = self.axis * (t * self.ext);
        Gen::mot(&scaled)
    }

    /// Get the motor for the full twist (t=1).
    pub fn motor(&self) -> Mot {
        self.motor_at(1.0)
    }

    /// Set the twist axis from a motor (take its logarithm).
    pub fn set_from_motor(&mut self, m: &Mot) {
        // Approximate log: for a normalized motor, the bivector part
        // is approximately the dual line (for small angles).
        // Full log_mot is complex; use the bivector extraction for now.
        self.axis = dll(m[1], m[2], m[3], m[4], m[5], m[6]);
    }

    /// Scale the dual line by `t`.
    pub fn scaled(&self, t: f32) -> Dll {
        self.axis * t
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const EPS: f32 = 1e-3;
    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn test_twist_default() {
        let tw = Twist::new();
        assert!(
            tw.period() > 0.0,
            "Default twist should have nonzero period"
        );
    }

    #[test]
    fn test_twist_from_direction() {
        let tw = Twist::from_direction(0.0, 1.0, 0.0);
        assert!(tw.period() > 0.0);
    }

    #[test]
    fn test_twist_motor_at_zero_is_identity() {
        let tw = Twist::new();
        let m = tw.motor_at(0.0);
        // Motor at t=0 should be identity: [1, 0, 0, 0, 0, 0, 0, 0]
        assert!(approx(m[0], 1.0), "m[0]={}", m[0]);
        for i in 1..8 {
            assert!(approx(m[i], 0.0), "m[{}]={}", i, m[i]);
        }
    }

    #[test]
    fn test_twist_along() {
        let line = dll(0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        let result = Twist::along(&line, PI, 0.5);
        // Should produce a scaled dual line with modified moment
        assert!(result.norm() > EPS);
    }

    #[test]
    fn test_twist_motor_applies() {
        let tw = Twist::from_direction(0.0, 0.0, 1.0);
        let m = tw.motor_at(0.1);
        let p = point(1.0, 0.0, 0.0);
        let p2 = spin_mot_pnt(&m, &p);
        // The point should have moved
        let d = distance(&p, &p2);
        assert!(d > 0.0, "Point should move under twist motor");
    }
}
