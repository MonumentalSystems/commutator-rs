//! Interpolation of geometric primitives.
//!
//! Port of `vsr_interp.h`. Provides linear, quadratic, cubic interpolation
//! for scalars, points, and motors. Also provides surface (bilinear) and
//! volume (trilinear) interpolation for fields.

use crate::cga3d::*;
use crate::mvec::Multivector;

/// Linear interpolation between two values: a*(1-t) + b*t.
pub fn lerp_f32(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}

/// Linear interpolation between two CGA points.
pub fn lerp_pnt(a: &Pnt, b: &Pnt, t: f32) -> Pnt {
    let (ax, ay, az) = Round::location(a);
    let (bx, by, bz) = Round::location(b);
    point(
        lerp_f32(ax, bx, t),
        lerp_f32(ay, by, t),
        lerp_f32(az, bz, t),
    )
}

/// Linear interpolation through an array of points.
///
/// `t` ranges from 0.0 to 1.0 over the full array.
pub fn lerp_pnt_array(pts: &[Pnt], t: f32) -> Pnt {
    let n = pts.len();
    if n == 0 {
        return origin();
    }
    if n == 1 {
        return pts[0];
    }
    let t_clamped = t.clamp(0.0, 1.0);
    let fw = t_clamped * (n - 1) as f32;
    let i = (fw as usize).min(n - 2);
    let frac = fw - i as f32;
    lerp_pnt(&pts[i], &pts[i + 1], frac)
}

/// Quadratic interpolation (de Casteljau) through three points.
///
/// `t` ranges from 0.0 to 1.0.
pub fn quadratic_pnt(a: &Pnt, b: &Pnt, c: &Pnt, t: f32) -> Pnt {
    let ab = lerp_pnt(a, b, t);
    let bc = lerp_pnt(b, c, t);
    lerp_pnt(&ab, &bc, t)
}

/// Cubic interpolation (de Casteljau) through four points.
///
/// `t` ranges from 0.0 to 1.0.
pub fn cubic_pnt(a: &Pnt, b: &Pnt, c: &Pnt, d: &Pnt, t: f32) -> Pnt {
    let ab = lerp_pnt(a, b, t);
    let bc = lerp_pnt(b, c, t);
    let cd = lerp_pnt(c, d, t);

    let abc = lerp_pnt(&ab, &bc, t);
    let bcd = lerp_pnt(&bc, &cd, t);

    lerp_pnt(&abc, &bcd, t)
}

/// Bilinear surface interpolation from four corner points.
///
/// Corners ordered: bottom-left, bottom-right, top-right, top-left.
/// u and v range from 0.0 to 1.0.
pub fn surface_pnt(bl: &Pnt, br: &Pnt, tr: &Pnt, tl: &Pnt, u: f32, v: f32) -> Pnt {
    let bottom = lerp_pnt(bl, br, u);
    let top = lerp_pnt(tl, tr, u);
    lerp_pnt(&bottom, &top, v)
}

/// Trilinear volume interpolation from eight corner points.
///
/// Corners ordered as a cube:
///   `[0]` = (0,0,0), `[1]` = (1,0,0), `[2]` = (0,1,0), `[3]` = (1,1,0),
///   `[4]` = (0,0,1), `[5]` = (1,0,1), `[6]` = (0,1,1), `[7]` = (1,1,1)
///
/// u, v, w range from 0.0 to 1.0.
pub fn volume_pnt(corners: &[Pnt; 8], u: f32, v: f32, w: f32) -> Pnt {
    // Interpolate front face (w=0)
    let front = surface_pnt(&corners[0], &corners[1], &corners[3], &corners[2], u, v);
    // Interpolate back face (w=1)
    let back = surface_pnt(&corners[4], &corners[5], &corners[7], &corners[6], u, v);
    // Interpolate between front and back
    lerp_pnt(&front, &back, w)
}

/// Motor interpolation: linear interpolation in the motor algebra.
///
/// For small relative rotations, this is a reasonable approximation
/// to geodesic interpolation. For large rotations, use `slerp_motor`.
pub fn lerp_motor(a: &Mot, b: &Mot, t: f32) -> Mot {
    let mut result = [0.0f32; 8];
    for i in 0..8 {
        result[i] = a[i] * (1.0 - t) + b[i] * t;
    }
    // Normalize the result
    let m = Multivector::new(result);
    let n = m.norm();
    if n > 1e-10 {
        m * (1.0 / n)
    } else {
        *a
    }
}

/// Spherical linear interpolation for rotors (quaternion slerp).
///
/// Interpolates along the geodesic (great arc) between two rotors.
pub fn slerp_rot(a: &Rot, b: &Rot, t: f32) -> Rot {
    // Compute cosine of angle between rotors
    let mut dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];

    // If dot is negative, negate one rotor (equivalent orientation, shorter path)
    let b_adj = if dot < 0.0 {
        dot = -dot;
        -*b
    } else {
        *b
    };

    // If very close, fall back to linear interpolation
    if dot > 0.9995 {
        let mut result = [0.0f32; 4];
        for i in 0..4 {
            result[i] = a[i] * (1.0 - t) + b_adj[i] * t;
        }
        let r = Multivector::new(result);
        let n = r.norm();
        return if n > 1e-10 { r * (1.0 / n) } else { *a };
    }

    let theta = dot.acos();
    let sin_theta = theta.sin();
    let wa = ((1.0 - t) * theta).sin() / sin_theta;
    let wb = (t * theta).sin() / sin_theta;

    let mut result = [0.0f32; 4];
    for i in 0..4 {
        result[i] = a[i] * wa + b_adj[i] * wb;
    }
    Multivector::new(result)
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
    fn test_lerp_f32() {
        assert!(approx(lerp_f32(0.0, 10.0, 0.5), 5.0));
        assert!(approx(lerp_f32(0.0, 10.0, 0.0), 0.0));
        assert!(approx(lerp_f32(0.0, 10.0, 1.0), 10.0));
    }

    #[test]
    fn test_lerp_pnt_midpoint() {
        let a = point(0.0, 0.0, 0.0);
        let b = point(10.0, 0.0, 0.0);
        let mid = lerp_pnt(&a, &b, 0.5);
        let (x, y, z) = Round::location(&mid);
        assert!(approx(x, 5.0), "x={}", x);
        assert!(approx(y, 0.0));
        assert!(approx(z, 0.0));
    }

    #[test]
    fn test_lerp_pnt_endpoints() {
        let a = point(1.0, 2.0, 3.0);
        let b = point(4.0, 5.0, 6.0);
        let p0 = lerp_pnt(&a, &b, 0.0);
        let p1 = lerp_pnt(&a, &b, 1.0);
        let (x0, y0, z0) = Round::location(&p0);
        let (x1, y1, z1) = Round::location(&p1);
        assert!(approx(x0, 1.0) && approx(y0, 2.0) && approx(z0, 3.0));
        assert!(approx(x1, 4.0) && approx(y1, 5.0) && approx(z1, 6.0));
    }

    #[test]
    fn test_lerp_pnt_array() {
        let pts = vec![
            point(0.0, 0.0, 0.0),
            point(1.0, 0.0, 0.0),
            point(2.0, 0.0, 0.0),
        ];
        let mid = lerp_pnt_array(&pts, 0.5);
        let (x, _, _) = Round::location(&mid);
        assert!(approx(x, 1.0), "x={}", x);
    }

    #[test]
    fn test_quadratic_pnt() {
        let a = point(0.0, 0.0, 0.0);
        let b = point(1.0, 1.0, 0.0);
        let c = point(2.0, 0.0, 0.0);
        let mid = quadratic_pnt(&a, &b, &c, 0.5);
        let (x, y, _) = Round::location(&mid);
        // At t=0.5, should be near (1, 0.5, 0) for a parabolic arc
        assert!(approx(x, 1.0), "x={}", x);
        assert!(y > 0.0 && y < 1.0, "y={}", y);
    }

    #[test]
    fn test_cubic_pnt_endpoints() {
        let a = point(0.0, 0.0, 0.0);
        let b = point(1.0, 1.0, 0.0);
        let c = point(2.0, 1.0, 0.0);
        let d = point(3.0, 0.0, 0.0);
        let p0 = cubic_pnt(&a, &b, &c, &d, 0.0);
        let p1 = cubic_pnt(&a, &b, &c, &d, 1.0);
        let (x0, y0, _) = Round::location(&p0);
        let (x1, y1, _) = Round::location(&p1);
        assert!(approx(x0, 0.0) && approx(y0, 0.0));
        assert!(approx(x1, 3.0) && approx(y1, 0.0));
    }

    #[test]
    fn test_surface_pnt_center() {
        let bl = point(0.0, 0.0, 0.0);
        let br = point(2.0, 0.0, 0.0);
        let tr = point(2.0, 2.0, 0.0);
        let tl = point(0.0, 2.0, 0.0);
        let center = surface_pnt(&bl, &br, &tr, &tl, 0.5, 0.5);
        let (x, y, z) = Round::location(&center);
        assert!(approx(x, 1.0), "x={}", x);
        assert!(approx(y, 1.0), "y={}", y);
        assert!(approx(z, 0.0));
    }

    #[test]
    fn test_volume_pnt_center() {
        let corners = [
            point(0.0, 0.0, 0.0),
            point(2.0, 0.0, 0.0),
            point(0.0, 2.0, 0.0),
            point(2.0, 2.0, 0.0),
            point(0.0, 0.0, 2.0),
            point(2.0, 0.0, 2.0),
            point(0.0, 2.0, 2.0),
            point(2.0, 2.0, 2.0),
        ];
        let center = volume_pnt(&corners, 0.5, 0.5, 0.5);
        let (x, y, z) = Round::location(&center);
        assert!(approx(x, 1.0), "x={}", x);
        assert!(approx(y, 1.0), "y={}", y);
        assert!(approx(z, 1.0), "z={}", z);
    }

    #[test]
    fn test_slerp_rot_endpoints() {
        let a = Gen::rot(&biv(0.0, 0.0, 0.0)); // identity
        let b = Gen::rot(&biv(PI / 2.0, 0.0, 0.0)); // 90 deg

        let r0 = slerp_rot(&a, &b, 0.0);
        let r1 = slerp_rot(&a, &b, 1.0);

        for i in 0..4 {
            assert!(
                approx(r0[i], a[i]),
                "r0[{}]={} vs a[{}]={}",
                i,
                r0[i],
                i,
                a[i]
            );
            assert!(
                approx(r1[i], b[i]),
                "r1[{}]={} vs b[{}]={}",
                i,
                r1[i],
                i,
                b[i]
            );
        }
    }

    #[test]
    fn test_slerp_rot_midpoint() {
        let a = Gen::rot(&biv(0.0, 0.0, 0.0));
        let b = Gen::rot(&biv(PI / 2.0, 0.0, 0.0));
        let mid = slerp_rot(&a, &b, 0.5);
        // Should be equivalent to a PI/4 rotation
        let expected = Gen::rot(&biv(PI / 4.0, 0.0, 0.0));
        for i in 0..4 {
            assert!(
                approx(mid[i], expected[i]),
                "mid[{}]={} vs expected[{}]={}",
                i,
                mid[i],
                i,
                expected[i]
            );
        }
    }

    #[test]
    fn test_lerp_motor_identity() {
        let a = mot(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let b = mot(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let m = lerp_motor(&a, &b, 0.5);
        assert!(approx(m[0], 1.0));
    }
}
