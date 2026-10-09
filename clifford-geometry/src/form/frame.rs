//! CGA3D Frame: position + orientation as a motor.
//!
//! Port of `vsr_cga3D_frame.h`. A Frame is the fundamental building block
//! for articulated bodies — it stores a conformal point (position) and a
//! rotor (orientation), and exposes local coordinate axes, planes, lines,
//! circles, and velocities.

use crate::cga3d::*;
use crate::mvec::Multivector;

/// Orthonormal frame: position (conformal point) + orientation (rotor).
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    /// Position as a CGA null point.
    pub pos: Pnt,
    /// Orientation as a Euclidean rotor.
    pub rot: Rot,
    /// Rotational velocity (bivector).
    pub d_biv: Biv,
    /// Translational velocity (Euclidean vector).
    pub d_vec: Vec3,
    /// Scale factor.
    pub scale: f32,
}

impl Default for Frame {
    fn default() -> Self {
        Self::new()
    }
}

impl Frame {
    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// Identity frame at the origin.
    pub fn new() -> Self {
        Self {
            pos: origin(),
            rot: Multivector::new([1.0, 0.0, 0.0, 0.0]),
            d_biv: Multivector::zero(),
            d_vec: Multivector::zero(),
            scale: 1.0,
        }
    }

    /// Frame at Euclidean coordinates (x, y, z) with identity rotation.
    pub fn at(x: f32, y: f32, z: f32) -> Self {
        Self {
            pos: point(x, y, z),
            ..Self::new()
        }
    }

    /// Frame from a position point and a rotor.
    pub fn from_point_rotor(p: Pnt, r: Rot) -> Self {
        Self {
            pos: p,
            rot: r,
            ..Self::new()
        }
    }

    /// Frame from a motor (position + orientation relative to origin).
    pub fn from_motor(m: &Mot) -> Self {
        let p = spin_mot_pnt(m, &origin());
        // Extract rotation part: the first 4 components of the motor
        // (s, e12, e13, e23) form the rotor.
        let r = Multivector::new([m[0], m[1], m[2], m[3]]);
        let r_norm = r.norm();
        let r = if r_norm > 1e-10 {
            r * (1.0 / r_norm)
        } else {
            Multivector::new([1.0, 0.0, 0.0, 0.0])
        };
        Self {
            pos: p,
            rot: r,
            ..Self::new()
        }
    }

    /// Reset to origin with identity rotation.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    // ------------------------------------------------------------------
    // Local axes (spin basis vectors by the rotor)
    // ------------------------------------------------------------------

    /// Local x-axis direction.
    pub fn x(&self) -> Vec3 {
        spin_rot_vec(&self.rot, &vec3(1.0, 0.0, 0.0))
    }

    /// Local y-axis direction.
    pub fn y(&self) -> Vec3 {
        spin_rot_vec(&self.rot, &vec3(0.0, 1.0, 0.0))
    }

    /// Local z-axis direction.
    pub fn z(&self) -> Vec3 {
        spin_rot_vec(&self.rot, &vec3(0.0, 0.0, 1.0))
    }

    /// Convenience aliases matching common conventions.
    pub fn right(&self) -> Vec3 {
        self.x()
    }
    /// Local up direction, equivalent to [`Self::y`].
    pub fn up(&self) -> Vec3 {
        self.y()
    }
    /// Local forward direction, defined as the negative local z-axis.
    pub fn forward(&self) -> Vec3 {
        let z = self.z();
        -z
    }

    // ------------------------------------------------------------------
    // Local bivector planes
    // ------------------------------------------------------------------

    /// Local xy bivector.
    pub fn xy(&self) -> Biv {
        spin_rot_biv(&self.rot, &biv(1.0, 0.0, 0.0))
    }

    /// Local xz bivector.
    pub fn xz(&self) -> Biv {
        spin_rot_biv(&self.rot, &biv(0.0, 1.0, 0.0))
    }

    /// Local yz bivector.
    pub fn yz(&self) -> Biv {
        spin_rot_biv(&self.rot, &biv(0.0, 0.0, 1.0))
    }

    // ------------------------------------------------------------------
    // Euclidean position extraction
    // ------------------------------------------------------------------

    /// Extract (x, y, z) Euclidean coordinates from position.
    pub fn location(&self) -> (f32, f32, f32) {
        Round::location(&self.pos)
    }

    // ------------------------------------------------------------------
    // Motor representation
    // ------------------------------------------------------------------

    /// Get the absolute motor relative to origin (translation * rotation).
    pub fn motor(&self) -> Mot {
        let (x, y, z) = self.location();
        let _t = Gen::trs(x, y, z);
        let r_mot = Gen::rot_as_mot(&self.rot);
        // Motor = Translator * Rotor. We need to compose them.
        // Since Translator is a Bst and Rotor is a Mot, we build
        // the motor directly: M = T * R where T = 1 - d*ni/2.
        // For a motor stored as [s, e12, e13, e23, e15, e25, e35, e1235]:
        //   T has s=1, e15=-dx/2, e25=-dy/2, e35=-dz/2 in the Mot basis
        //   R has s, e12, e13, e23 from the rotor
        // T * R for a pure translator times pure rotor:
        //   scalar part = T_s * R_s = R_s
        //   e12 part = R_e12
        //   e13 part = R_e13
        //   e23 part = R_e23
        //   e15 part = T_e15 * R_s + ...
        //   etc.
        // Actually the simplest approach: use gp_mot_mot.
        let t_mot = mot(1.0, 0.0, 0.0, 0.0, -0.5 * x, -0.5 * y, -0.5 * z, 0.0);
        gp_mot_mot(&t_mot, &r_mot)
    }

    /// Get the dual line representation (log of the motor).
    pub fn dual_line(&self) -> Dll {
        let m = self.motor();
        // For a normalized motor, the dual line is approximately the bivector part.
        // Dll = [e12, e13, e23, e15, e25, e35]
        dll(m[1], m[2], m[3], m[4], m[5], m[6])
    }

    // ------------------------------------------------------------------
    // Transformations (mutating)
    // ------------------------------------------------------------------

    /// Translate by (dx, dy, dz), modifying the frame in place.
    pub fn translate(&mut self, dx: f32, dy: f32, dz: f32) -> &mut Self {
        self.pos = translate(&self.pos, dx, dy, dz);
        self
    }

    /// Rotate by a bivector generator, modifying the frame in place.
    pub fn rotate_by(&mut self, bivector: &Biv) -> &mut Self {
        let r = Gen::rot(bivector);
        self.rot = gp_rot_rot(&r, &self.rot);
        self
    }

    /// Rotate around the local XY plane by `amt` radians.
    pub fn rot_xy(&self, amt: f32) -> Frame {
        let local_biv = self.xy() * amt;
        let r = Gen::rot(&local_biv);
        Frame {
            pos: self.pos,
            rot: gp_rot_rot(&r, &self.rot),
            ..*self
        }
    }

    /// Move along the local Z axis by `amt`.
    pub fn move_z(&self, amt: f32) -> Frame {
        let z = self.z();
        let mut f = *self;
        f.pos = translate(&self.pos, z[0] * amt, z[1] * amt, z[2] * amt);
        f
    }

    /// Move along the local X axis by `amt`.
    pub fn move_x(&self, amt: f32) -> Frame {
        let x_dir = self.x();
        let mut f = *self;
        f.pos = translate(&self.pos, x_dir[0] * amt, x_dir[1] * amt, x_dir[2] * amt);
        f
    }

    /// Move along the local Y axis by `amt`.
    pub fn move_y(&self, amt: f32) -> Frame {
        let y_dir = self.y();
        let mut f = *self;
        f.pos = translate(&self.pos, y_dir[0] * amt, y_dir[1] * amt, y_dir[2] * amt);
        f
    }

    /// Integration step: apply velocity to update position and orientation.
    pub fn step(&mut self) -> &mut Self {
        // Rotation: apply bivector velocity
        if self.d_biv.norm() > 1e-10 {
            let r = Gen::rot(&self.d_biv);
            self.rot = gp_rot_rot(&r, &self.rot);
        }
        // Translation: apply vector velocity
        let v = &self.d_vec;
        if v.norm() > 1e-10 {
            self.pos = translate(&self.pos, v[0], v[1], v[2]);
        }
        self
    }

    /// Dual sphere bounding shell at this frame's position and scale.
    pub fn bound(&self) -> Dls {
        Round::dls(&self.pos, self.scale)
    }
}

// ============================================================================
// Helper: spin a Vec3 by a Rot in CGA context
// ============================================================================

/// Spin a Euclidean vector by a CGA rotor: v' = R v ~R.
/// Uses the EGA sandwich product since the vector has no null components.
fn spin_rot_vec(r: &Rot, v: &Vec3) -> Vec3 {
    // Rotor sandwich: v' = R * v * ~R
    // Rot = [s, e12, e13, e23], Vec = [e1, e2, e3]
    let s = r[0];
    let b12 = r[1];
    let b13 = r[2];
    let b23 = r[3];

    let x = v[0];
    let y = v[1];
    let z = v[2];

    // Direct formula for R * v * ~R in Cl(3,0):
    let s2 = s * s;
    let b12_2 = b12 * b12;
    let b13_2 = b13 * b13;
    let b23_2 = b23 * b23;

    let rx = x * (s2 + b12_2 + b13_2 - b23_2)
        - 2.0 * (x * b23_2 - y * (s * b12 + b13 * b23) - z * (s * b13 - b12 * b23))
        + x * b23_2
        - x * b23_2; // simplify below

    // Actually, let me use the well-known quaternion rotation formula.
    // Rotor [s, e12, e13, e23] maps to quaternion [w, -k, j, -i] in some convention.
    // Simpler: compute directly.
    //
    // For a rotor R = s + b12*e12 + b13*e13 + b23*e23 and vector v = x*e1 + y*e2 + z*e3:
    // R*v = (s*x - b12*y - b13*z)*e1 + (s*y + b12*x - b23*z)*e2 + (s*z + b13*x + b23*y)*e3
    //     + (b12*z - b13*y + b23*x)*e123
    // Wait, that's the half product. For the full sandwich we need R*v*~R.
    //
    // Use the explicit quaternion-style rotation matrix instead:
    let out_x = (s2 + b12_2 - b13_2 - b23_2) * x
        + 2.0 * (b12 * b23 - s * b13) * z
        + 2.0 * (s * b12 + b13 * b23) * y
        - 2.0 * b13 * b23 * y
        - 2.0 * b12 * b23 * z
        + 2.0 * b13 * b23 * y
        + 2.0 * b12 * b23 * z;
    // This is getting messy. Let me use the existing CGA spin machinery instead.
    //
    // Promote Vec3 to Pnt at origin, spin, extract.
    // Actually simplest: use the point-based spin and extract Euclidean part.
    let _ = (rx, out_x); // suppress unused

    // Use the point spin: create a point at v, spin it, extract direction.
    // A direction vector in CGA is encoded differently. Let's just compute
    // using the CGA point infrastructure: put a point at v and at origin,
    // spin both, take the difference.
    let p = point(x, y, z);
    let o = origin();
    let p2 = spin_rot_pnt(r, &p);
    let o2 = spin_rot_pnt(r, &o);
    let (px, py, pz) = Round::location(&p2);
    let (ox, oy, oz) = Round::location(&o2);
    vec3(px - ox, py - oy, pz - oz)
}

/// Spin a Euclidean bivector by a CGA rotor.
fn spin_rot_biv(r: &Rot, b: &Biv) -> Biv {
    // For a bivector B in the e12, e13, e23 basis, R B ~R produces another bivector.
    // We can compute this by rotating two vectors and taking the outer product.
    // For e12 = e1^e2: rotate e1 and e2, wedge them.
    // For a general bivector, decompose into basis and recombine.
    //
    // Alternatively, since Rot*Biv*~Rot is grade-preserving, we can compute it
    // by using the rotor composition: R*B*~R where B is treated as a bivector.
    // The rotor-bivector sandwich in Cl(3,0) can be computed via:
    // (R * B * ~R) where B = b12*e12 + b13*e13 + b23*e23.
    //
    // Use dual: biv -> vec -> spin -> vec -> biv
    let v = vec3(b[2], -b[1], b[0]); // dual of biv in Cl(3,0)
    let rv = spin_rot_vec(r, &v);
    // undual: vec -> biv
    biv(rv[2], -rv[1], rv[0])
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const EPS: f32 = 1e-4;
    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    #[test]
    fn test_frame_default_at_origin() {
        let f = Frame::new();
        let (x, y, z) = f.location();
        assert!(approx(x, 0.0));
        assert!(approx(y, 0.0));
        assert!(approx(z, 0.0));
    }

    #[test]
    fn test_frame_at() {
        let f = Frame::at(1.0, 2.0, 3.0);
        let (x, y, z) = f.location();
        assert!(approx(x, 1.0));
        assert!(approx(y, 2.0));
        assert!(approx(z, 3.0));
    }

    #[test]
    fn test_frame_local_axes_identity() {
        let f = Frame::new();
        let x = f.x();
        let y = f.y();
        let z = f.z();
        assert!(approx(x[0], 1.0) && approx(x[1], 0.0) && approx(x[2], 0.0));
        assert!(approx(y[0], 0.0) && approx(y[1], 1.0) && approx(y[2], 0.0));
        assert!(approx(z[0], 0.0) && approx(z[1], 0.0) && approx(z[2], 1.0));
    }

    #[test]
    fn test_frame_rotated_axes() {
        let r = Gen::rot(&biv(PI / 2.0, 0.0, 0.0)); // 90 deg in e12 plane
        let f = Frame::from_point_rotor(origin(), r);
        let x = f.x();
        // After 90 deg rotation in e12, x-axis (e1) -> e2
        assert!(approx(x[0], 0.0), "x[0]={}", x[0]);
        assert!(approx(x[1], 1.0), "x[1]={}", x[1]);
        assert!(approx(x[2], 0.0), "x[2]={}", x[2]);
    }

    #[test]
    fn test_frame_translate() {
        let mut f = Frame::new();
        f.translate(5.0, 0.0, 0.0);
        let (x, y, z) = f.location();
        assert!(approx(x, 5.0));
        assert!(approx(y, 0.0));
        assert!(approx(z, 0.0));
    }

    #[test]
    fn test_frame_move_z() {
        let f = Frame::at(0.0, 0.0, 0.0);
        let f2 = f.move_z(3.0);
        let (x, y, z) = f2.location();
        assert!(approx(x, 0.0));
        assert!(approx(y, 0.0));
        assert!(approx(z, 3.0));
    }

    #[test]
    fn test_frame_step_translation() {
        let mut f = Frame::new();
        f.d_vec = vec3(1.0, 0.0, 0.0);
        f.step();
        let (x, _y, _z) = f.location();
        assert!(approx(x, 1.0));
    }

    #[test]
    fn test_frame_bound() {
        let mut f = Frame::at(1.0, 2.0, 3.0);
        f.scale = 5.0;
        let dls = f.bound();
        let r2 = Round::radius_squared(&dls);
        assert!(approx(r2, 25.0), "r2={}", r2);
    }
}
