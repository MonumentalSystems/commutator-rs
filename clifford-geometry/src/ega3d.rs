//! Euclidean 3D Geometric Algebra — Cl(3,0).
//!
//! Port of Versor's EGA3D types. Basis: e1, e2, e3.
//! 2^3 = 8 total basis blades, but each type only stores its own blades.
//!
//! Types and their storage:
//! - Sca (scalar): 1 component  `[s]`
//! - Vec (vector): 3 components `[e1, e2, e3]`
//! - Biv (bivector): 3 components `[e12, e13, e23]`
//! - Tri (trivector/pseudoscalar): 1 component `[e123]`
//! - Rot (rotor = scalar + bivector): 4 components `[s, e12, e13, e23]`
//! - Mot (full multivector): 8 components `[s, e1, e2, e3, e12, e13, e23, e123]`

use crate::mvec::Multivector;
use crate::products::{self, DenseTable};

// ============================================================================
// Metric
// ============================================================================

/// Cl(3,0) Euclidean metric: all basis vectors square to +1.
pub const METRIC: [i32; 3] = [1, 1, 1];

// ============================================================================
// Blade layouts (which basis blades map to which index)
// ============================================================================

/// Scalar basis: `[1]`
pub const SCALAR_BASIS: [u32; 1] = [0b000];

/// Vector basis: `[e1, e2, e3]`
pub const VEC_BASIS: [u32; 3] = [0b001, 0b010, 0b100];

/// Bivector basis: `[e12, e13, e23]`
pub const BIV_BASIS: [u32; 3] = [0b011, 0b101, 0b110];

/// Trivector basis: `[e123]`
pub const TRI_BASIS: [u32; 1] = [0b111];

/// Rotor basis: `[s, e12, e13, e23]`
pub const ROT_BASIS: [u32; 4] = [0b000, 0b011, 0b101, 0b110];

/// Full multivector basis: `[s, e1, e2, e3, e12, e13, e23, e123]`
pub const FULL_BASIS: [u32; 8] = [0b000, 0b001, 0b010, 0b100, 0b011, 0b101, 0b110, 0b111];

// ============================================================================
// Type aliases — sparse multivectors
// ============================================================================

/// Scalar: 1 component.
pub type Sca = Multivector<1>;
/// Vector: 3 components `[e1, e2, e3]`.
pub type Vec3 = Multivector<3>;
/// Bivector: 3 components `[e12, e13, e23]`.
pub type Biv = Multivector<3>;
/// Trivector (pseudoscalar): 1 component `[e123]`.
pub type Tri = Multivector<1>;
/// Rotor (scalar + bivector): 4 components `[s, e12, e13, e23]`.
pub type Rot = Multivector<4>;
/// Full multivector: 8 components.
pub type Mot = Multivector<8>;

// ============================================================================
// Constructors
// ============================================================================

/// Create a vector from x, y, z components.
#[inline]
pub fn vec3(x: f32, y: f32, z: f32) -> Vec3 {
    Multivector::new([x, y, z])
}

/// Create a bivector from e12, e13, e23 components.
#[inline]
pub fn biv(e12: f32, e13: f32, e23: f32) -> Biv {
    Multivector::new([e12, e13, e23])
}

/// Create a scalar.
#[inline]
pub fn sca(s: f32) -> Sca {
    Multivector::new([s])
}

/// Create a rotor from scalar and bivector parts.
#[inline]
pub fn rot(s: f32, e12: f32, e13: f32, e23: f32) -> Rot {
    Multivector::new([s, e12, e13, e23])
}

// ============================================================================
// Product tables (lazily initialized)
// ============================================================================

use std::sync::OnceLock;

/// Vec * Vec geometric product table → Rot (scalar + bivector)
fn vec_vec_gp_table() -> &'static DenseTable<3, 3, 4> {
    static TABLE: OnceLock<DenseTable<3, 3, 4>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&VEC_BASIS, &VEC_BASIS, &ROT_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Rot * Rot geometric product table → Rot
fn rot_rot_gp_table() -> &'static DenseTable<4, 4, 4> {
    static TABLE: OnceLock<DenseTable<4, 4, 4>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&ROT_BASIS, &ROT_BASIS, &ROT_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Rot * Vec geometric product → full multivector projected to Vec
/// (sandwich product intermediate: R * v gives mixed grades, but R * v * R~ gives pure vector)
fn rot_vec_gp_table() -> &'static DenseTable<4, 3, 8> {
    static TABLE: OnceLock<DenseTable<4, 3, 8>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&ROT_BASIS, &VEC_BASIS, &FULL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Full * Rot geometric product → Full (for sandwich product second half)
fn full_rot_gp_table() -> &'static DenseTable<8, 4, 8> {
    static TABLE: OnceLock<DenseTable<8, 4, 8>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&FULL_BASIS, &ROT_BASIS, &FULL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Vec * Vec outer product → Biv
fn vec_vec_op_table() -> &'static DenseTable<3, 3, 3> {
    static TABLE: OnceLock<DenseTable<3, 3, 3>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_outer_product(&VEC_BASIS, &VEC_BASIS, &BIV_BASIS);
        DenseTable::from_instructions(&insts)
    })
}

/// Vec * Vec scalar product (dot product) → Sca
fn vec_vec_sp_table() -> &'static DenseTable<3, 3, 1> {
    static TABLE: OnceLock<DenseTable<3, 3, 1>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_scalar_product(&VEC_BASIS, &VEC_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

// ============================================================================
// Product operations
// ============================================================================

/// Geometric product of two vectors → rotor (scalar + bivector).
#[inline]
pub fn gp_vec_vec(a: &Vec3, b: &Vec3) -> Rot {
    Multivector::new(vec_vec_gp_table().execute(&a.data, &b.data))
}

/// Geometric product of two rotors → rotor.
#[inline]
pub fn gp_rot_rot(a: &Rot, b: &Rot) -> Rot {
    Multivector::new(rot_rot_gp_table().execute(&a.data, &b.data))
}

/// Outer product of two vectors → bivector.
#[inline]
pub fn op_vec_vec(a: &Vec3, b: &Vec3) -> Biv {
    Multivector::new(vec_vec_op_table().execute(&a.data, &b.data))
}

/// Dot product of two vectors → scalar.
#[inline]
pub fn dot(a: &Vec3, b: &Vec3) -> f32 {
    vec_vec_sp_table().execute(&a.data, &b.data)[0]
}

/// Cross product of two vectors (= dual of outer product in 3D).
#[inline]
pub fn cross(a: &Vec3, b: &Vec3) -> Vec3 {
    let biv = op_vec_vec(a, b);
    // In Cl(3,0), dual of e12 = e3, dual of e13 = -e2, dual of e23 = e1
    // Biv layout: [e12, e13, e23]
    // Vec layout: [e1, e2, e3]
    Multivector::new([biv[2], -biv[1], biv[0]])
}

// ============================================================================
// Unary operations
// ============================================================================

/// Reverse of a vector (unchanged, grade 1).
#[inline]
pub fn reverse_vec(v: &Vec3) -> Vec3 {
    *v
}

/// Reverse of a bivector (negated, grade 2).
#[inline]
pub fn reverse_biv(b: &Biv) -> Biv {
    -*b
}

/// Reverse of a rotor: negate the bivector part.
#[inline]
pub fn reverse_rot(r: &Rot) -> Rot {
    Multivector::new([r[0], -r[1], -r[2], -r[3]])
}

/// Dual of a vector in Cl(3,0): Vec → Biv.
/// e1 → e23, e2 → -e13, e3 → e12
#[inline]
pub fn dual_vec(v: &Vec3) -> Biv {
    Multivector::new([v[2], -v[1], v[0]])
}

/// Dual of a bivector in Cl(3,0): Biv → Vec.
/// e12 → e3, e13 → -e2, e23 → e1
#[inline]
pub fn dual_biv(b: &Biv) -> Vec3 {
    Multivector::new([b[2], -b[1], b[0]])
}

// ============================================================================
// Generators (exponential maps)
// ============================================================================

/// Generate a rotor from a bivector: R = exp(-B/2) = cos(|B|/2) - sin(|B|/2) * B/|B|
///
/// This is the fundamental rotation generator in geometric algebra.
/// The bivector defines the rotation plane and the angle (|B| = angle).
pub fn gen_rot(bivector: &Biv) -> Rot {
    let angle = bivector.norm();
    if angle < 1e-10 {
        return rot(1.0, 0.0, 0.0, 0.0); // identity rotor
    }
    let half = angle / 2.0;
    let cos_half = half.cos();
    let sin_half = -half.sin() / angle; // negative for exp(-B/2) convention
    rot(
        cos_half,
        sin_half * bivector[0],
        sin_half * bivector[1],
        sin_half * bivector[2],
    )
}

/// Apply a rotor to a vector: v' = R * v * R~
///
/// This is the sandwich product that performs rotation.
pub fn spin(rotor: &Rot, v: &Vec3) -> Vec3 {
    // R * v
    let rv = rot_vec_gp_table().execute(&rotor.data, &v.data);
    // (R * v) * R~
    let r_rev = reverse_rot(rotor);
    let result = full_rot_gp_table().execute(&rv, &r_rev.data);
    // Extract vector part (indices 1, 2, 3 in FULL_BASIS = e1, e2, e3)
    Multivector::new([result[1], result[2], result[3]])
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_vec_creation() {
        let v = vec3(1.0, 2.0, 3.0);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 2.0);
        assert_eq!(v[2], 3.0);
    }

    #[test]
    fn test_dot_product() {
        let a = vec3(1.0, 0.0, 0.0);
        let b = vec3(0.0, 1.0, 0.0);
        assert_eq!(dot(&a, &b), 0.0); // orthogonal

        let c = vec3(1.0, 2.0, 3.0);
        let d = vec3(4.0, 5.0, 6.0);
        assert_eq!(dot(&c, &d), 32.0); // 4 + 10 + 18
    }

    #[test]
    fn test_outer_product() {
        let a = vec3(1.0, 0.0, 0.0); // e1
        let b = vec3(0.0, 1.0, 0.0); // e2
        let ab = op_vec_vec(&a, &b);
        // e1 ^ e2 = e12
        assert_eq!(ab[0], 1.0); // e12 component
        assert_eq!(ab[1], 0.0); // e13 component
        assert_eq!(ab[2], 0.0); // e23 component
    }

    #[test]
    fn test_cross_product() {
        let a = vec3(1.0, 0.0, 0.0);
        let b = vec3(0.0, 1.0, 0.0);
        let c = cross(&a, &b);
        // e1 × e2 = e3
        assert!((c[0]).abs() < 1e-6);
        assert!((c[1]).abs() < 1e-6);
        assert!((c[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_geometric_product_vec_vec() {
        let a = vec3(1.0, 0.0, 0.0); // e1
        let b = vec3(1.0, 0.0, 0.0); // e1
        let ab = gp_vec_vec(&a, &b);
        // e1 * e1 = +1 (scalar)
        assert_eq!(ab[0], 1.0); // scalar
        assert_eq!(ab[1], 0.0); // e12
        assert_eq!(ab[2], 0.0); // e13
        assert_eq!(ab[3], 0.0); // e23
    }

    #[test]
    fn test_rotor_from_bivector() {
        // Rotation of PI/2 in the e12 plane
        let b = biv(PI / 2.0, 0.0, 0.0);
        let r = gen_rot(&b);
        // R = cos(pi/4) - sin(pi/4) * e12
        let c = (PI / 4.0).cos();
        let s = (PI / 4.0).sin();
        assert!((r[0] - c).abs() < 1e-6);
        assert!((r[1] + s).abs() < 1e-6); // negative because exp(-B/2)
        assert!((r[2]).abs() < 1e-6);
        assert!((r[3]).abs() < 1e-6);
    }

    #[test]
    fn test_rotor_rotation() {
        // Rotate e1 by PI/2 in e12 plane → should give e2
        let b = biv(PI / 2.0, 0.0, 0.0);
        let r = gen_rot(&b);
        let v = vec3(1.0, 0.0, 0.0);
        let rotated = spin(&r, &v);

        assert!((rotated[0]).abs() < 1e-5, "e1 = {}", rotated[0]);
        assert!((rotated[1] - 1.0).abs() < 1e-5, "e2 = {}", rotated[1]);
        assert!((rotated[2]).abs() < 1e-5, "e3 = {}", rotated[2]);
    }

    #[test]
    fn test_rotor_preserves_norm() {
        let b = biv(1.0, 0.5, 0.3);
        let r = gen_rot(&b);
        let v = vec3(1.0, 2.0, 3.0);
        let rotated = spin(&r, &v);

        let orig_norm = v.norm();
        let rot_norm = rotated.norm();
        assert!((orig_norm - rot_norm).abs() < 1e-5);
    }

    #[test]
    fn test_rotor_composition() {
        // Two rotations should compose via geometric product
        let b1 = biv(PI / 4.0, 0.0, 0.0);
        let b2 = biv(PI / 4.0, 0.0, 0.0);
        let r1 = gen_rot(&b1);
        let r2 = gen_rot(&b2);
        let r12 = gp_rot_rot(&r1, &r2);

        // Composed rotation should equal single PI/2 rotation
        let r_full = gen_rot(&biv(PI / 2.0, 0.0, 0.0));

        for i in 0..4 {
            assert!(
                (r12[i] - r_full[i]).abs() < 1e-5,
                "idx {} mismatch: {} vs {}",
                i,
                r12[i],
                r_full[i]
            );
        }
    }

    #[test]
    fn test_reverse_rotor() {
        let r = rot(0.7071, -0.7071, 0.0, 0.0);
        let r_rev = reverse_rot(&r);
        // Reverse negates bivector part
        assert_eq!(r_rev[0], r[0]);
        assert_eq!(r_rev[1], -r[1]);
        assert_eq!(r_rev[2], -r[2]);
        assert_eq!(r_rev[3], -r[3]);
    }

    #[test]
    fn test_rotor_identity() {
        let identity = rot(1.0, 0.0, 0.0, 0.0);
        let v = vec3(1.0, 2.0, 3.0);
        let result = spin(&identity, &v);
        for i in 0..3 {
            assert!((result[i] - v[i]).abs() < 1e-6);
        }
    }
}
