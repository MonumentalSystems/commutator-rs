//! Projective Geometric Algebra (PGA) 3D -- Cl(3,0,1).
//!
//! Port of Versor's PGA3D types. Basis: e1, e2, e3 (Euclidean), e4 (degenerate/null).
//! 2^4 = 16 total basis blades. Metric: `[1, 1, 1, 0]`.
//!
//! In PGA:
//! - Planes are vectors (grade 1): a*e1 + b*e2 + c*e3 + d*e4
//! - Lines are bivectors (grade 2): 6 components
//! - Points are trivectors (grade 3): 4 components
//! - The pseudoscalar e1234 squares to 0 (degenerate)
//!
//! The regressive product (meet) is the dual of the outer product of duals.

use crate::mvec::Multivector;
use crate::products::{self, DenseTable};
use std::sync::OnceLock;

// ============================================================================
// Metric
// ============================================================================

/// Cl(3,0,1) projective metric: e1^2=+1, e2^2=+1, e3^2=+1, e4^2=0.
pub const METRIC: [i32; 4] = [1, 1, 1, 0];

// ============================================================================
// Blade bitmasks
// ============================================================================

// Grade 0: scalar
const S: u32 = 0b0000;

// Grade 1: vectors (planes in PGA)
const E1: u32 = 0b0001;
const E2: u32 = 0b0010;
const E3: u32 = 0b0100;
const E4: u32 = 0b1000;

// Grade 2: bivectors (lines in PGA)
const E12: u32 = 0b0011;
const E13: u32 = 0b0101;
const E23: u32 = 0b0110;
const E14: u32 = 0b1001;
const E24: u32 = 0b1010;
const E34: u32 = 0b1100;

// Grade 3: trivectors (points in PGA)
const E123: u32 = 0b0111;
const E124: u32 = 0b1011;
const E134: u32 = 0b1101;
const E234: u32 = 0b1110;

// Grade 4: pseudoscalar
const E1234: u32 = 0b1111;

// ============================================================================
// Blade layouts (basis arrays for each type)
// ============================================================================

/// Scalar: `[s]`
pub const SCA_BASIS: [u32; 1] = [S];

/// Plane (grade-1 vector): `[e1, e2, e3, e4]`
/// Plane equation: a*e1 + b*e2 + c*e3 + d*e4
/// where (a,b,c) is the normal and d is the distance from origin.
pub const PLANE_BASIS: [u32; 4] = [E1, E2, E3, E4];

/// Line (grade-2 bivector): `[e12, e13, e23, e14, e24, e34]`
/// A line in PGA has 6 components split into:
///   - Euclidean part (direction): e12, e13, e23
///   - Ideal part (moment): e14, e24, e34
pub const LINE_BASIS: [u32; 6] = [E12, E13, E23, E14, E24, E34];

/// Point (grade-3 trivector): `[e123, e124, e134, e234]`
/// A point P = x*e032 + y*e013 + z*e021 + w*e123 in some conventions,
/// but we use the bitmask ordering: `[e123, e124, e134, e234]`.
/// A Euclidean point at (x,y,z) is: -e234*x + e134*y - e124*z + e123
pub const POINT_BASIS: [u32; 4] = [E123, E124, E134, E234];

/// Pseudoscalar: `[e1234]`
pub const PSS_BASIS: [u32; 1] = [E1234];

/// Motor (even-grade versor): `[s, e12, e13, e23, e14, e24, e34, e1234]`
/// M = R + d, where R is a rotor (scalar + euclidean bivector)
/// and d is the translation part (ideal bivector + pseudoscalar).
pub const MOTOR_BASIS: [u32; 8] = [S, E12, E13, E23, E14, E24, E34, E1234];

/// Rotor (pure rotation, no translation): `[s, e12, e13, e23]`
pub const ROTOR_BASIS: [u32; 4] = [S, E12, E13, E23];

/// Translator: `[s, e14, e24, e34]`
pub const TRANSLATOR_BASIS: [u32; 4] = [S, E14, E24, E34];

/// Full multivector: all 16 blades
/// Ordered by grade, then by bitmask within each grade.
pub const FULL_BASIS: [u32; 16] = [
    S, // grade 0
    E1, E2, E3, E4, // grade 1
    E12, E13, E23, E14, E24, E34, // grade 2
    E123, E124, E134, E234,  // grade 3
    E1234, // grade 4
];

// ============================================================================
// Type aliases -- sparse multivectors
// ============================================================================

/// Scalar: 1 component.
pub type Sca = Multivector<1>;
/// Plane (grade 1): 4 components `[e1, e2, e3, e4]`.
pub type Plane = Multivector<4>;
/// Line (grade 2): 6 components `[e12, e13, e23, e14, e24, e34]`.
pub type Line = Multivector<6>;
/// Point (grade 3): 4 components `[e123, e124, e134, e234]`.
pub type Point = Multivector<4>;
/// Pseudoscalar: 1 component `[e1234]`.
pub type Pss = Multivector<1>;
/// Motor (even subalgebra): 8 components `[s, e12, e13, e23, e14, e24, e34, e1234]`.
pub type Motor = Multivector<8>;
/// Rotor (pure rotation): 4 components `[s, e12, e13, e23]`.
pub type Rotor = Multivector<4>;
/// Translator: 4 components `[s, e14, e24, e34]`.
pub type Translator = Multivector<4>;

// ============================================================================
// Constructors
// ============================================================================

/// Create a plane from normal (a,b,c) and distance d.
/// Plane: a*e1 + b*e2 + c*e3 + d*e4
#[inline]
pub fn plane(a: f32, b: f32, c: f32, d: f32) -> Plane {
    Multivector::new([a, b, c, d])
}

/// Create a normalized Euclidean point at (x, y, z).
/// In PGA, a point is a grade-3 element (trivector).
/// Point = e123 - z*e124 + y*e134 - x*e234
/// (signs come from the wedge product of three planes)
#[inline]
pub fn point(x: f32, y: f32, z: f32) -> Point {
    // POINT_BASIS = [e123, e124, e134, e234]
    Multivector::new([1.0, -z, y, -x])
}

/// Create a line from Plucker coordinates.
/// Direction part: (d12, d13, d23) on e12, e13, e23
/// Moment part: (m14, m24, m34) on e14, e24, e34
#[inline]
pub fn line(d12: f32, d13: f32, d23: f32, m14: f32, m24: f32, m34: f32) -> Line {
    Multivector::new([d12, d13, d23, m14, m24, m34])
}

/// Create a motor from components.
#[inline]
pub fn motor(
    s: f32,
    e12: f32,
    e13: f32,
    e23: f32,
    e14: f32,
    e24: f32,
    e34: f32,
    e1234: f32,
) -> Motor {
    Multivector::new([s, e12, e13, e23, e14, e24, e34, e1234])
}

/// Create a rotor (pure rotation) from components.
#[inline]
pub fn rotor(s: f32, e12: f32, e13: f32, e23: f32) -> Rotor {
    Multivector::new([s, e12, e13, e23])
}

/// Create a translator from a direction (tx, ty, tz).
/// T = 1 + (tx*e14 + ty*e24 + tz*e34) / 2
/// (the factor of 1/2 is built in so that sandwich gives full translation)
#[inline]
pub fn translator(tx: f32, ty: f32, tz: f32) -> Translator {
    Multivector::new([1.0, tx / 2.0, ty / 2.0, tz / 2.0])
}

/// Create a line from two points (join = outer product of two trivectors
/// projected via regressive product, i.e., dual of outer product of duals).
/// In PGA: line = p1 & p2 (regressive product).
pub fn line_from_points(p1: &Point, p2: &Point) -> Line {
    regressive_pnt_pnt(p1, p2)
}

/// Create a line from two planes (meet = outer product of two planes).
/// In PGA: line = pi1 ^ pi2 (outer product).
pub fn line_from_planes(pi1: &Plane, pi2: &Plane) -> Line {
    op_pln_pln(pi1, pi2)
}

/// Create a point from three planes (meet = outer product of three planes).
/// point = pi1 ^ pi2 ^ pi3
pub fn point_from_planes(pi1: &Plane, pi2: &Plane, pi3: &Plane) -> Point {
    let l = op_pln_pln(pi1, pi2);
    op_lin_pln(&l, pi3)
}

// ============================================================================
// Product tables (lazily initialized via OnceLock)
// ============================================================================

// --- Geometric products ---

/// Motor * Motor -> Motor
fn mot_mot_gp() -> &'static DenseTable<8, 8, 8> {
    static TABLE: OnceLock<DenseTable<8, 8, 8>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts =
            products::gen_geometric_product(&MOTOR_BASIS, &MOTOR_BASIS, &MOTOR_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Motor * Point -> Full (intermediate for sandwich)
fn mot_pnt_gp() -> &'static DenseTable<8, 4, 16> {
    static TABLE: OnceLock<DenseTable<8, 4, 16>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts =
            products::gen_geometric_product(&MOTOR_BASIS, &POINT_BASIS, &FULL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Full * Motor -> Full (second half of sandwich for points)
fn full_mot_gp() -> &'static DenseTable<16, 8, 16> {
    static TABLE: OnceLock<DenseTable<16, 8, 16>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts =
            products::gen_geometric_product(&FULL_BASIS, &MOTOR_BASIS, &FULL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Motor * Line -> Full (intermediate for sandwich)
fn mot_lin_gp() -> &'static DenseTable<8, 6, 16> {
    static TABLE: OnceLock<DenseTable<8, 6, 16>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts =
            products::gen_geometric_product(&MOTOR_BASIS, &LINE_BASIS, &FULL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Motor * Plane -> Full (intermediate for sandwich)
fn mot_pln_gp() -> &'static DenseTable<8, 4, 16> {
    static TABLE: OnceLock<DenseTable<8, 4, 16>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts =
            products::gen_geometric_product(&MOTOR_BASIS, &PLANE_BASIS, &FULL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Plane * Plane -> Motor (GP of two planes gives motor components)
fn pln_pln_gp() -> &'static DenseTable<4, 4, 8> {
    static TABLE: OnceLock<DenseTable<4, 4, 8>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts =
            products::gen_geometric_product(&PLANE_BASIS, &PLANE_BASIS, &MOTOR_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Line * Line -> Motor
fn lin_lin_gp() -> &'static DenseTable<6, 6, 8> {
    static TABLE: OnceLock<DenseTable<6, 6, 8>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts =
            products::gen_geometric_product(&LINE_BASIS, &LINE_BASIS, &MOTOR_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

// --- Outer products ---

/// Plane ^ Plane -> Line
fn pln_pln_op() -> &'static DenseTable<4, 4, 6> {
    static TABLE: OnceLock<DenseTable<4, 4, 6>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_outer_product(&PLANE_BASIS, &PLANE_BASIS, &LINE_BASIS);
        DenseTable::from_instructions(&insts)
    })
}

/// Line ^ Plane -> Point
fn lin_pln_op() -> &'static DenseTable<6, 4, 4> {
    static TABLE: OnceLock<DenseTable<6, 4, 4>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_outer_product(&LINE_BASIS, &PLANE_BASIS, &POINT_BASIS);
        DenseTable::from_instructions(&insts)
    })
}

/// Plane ^ Line -> Point
fn pln_lin_op() -> &'static DenseTable<4, 6, 4> {
    static TABLE: OnceLock<DenseTable<4, 6, 4>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_outer_product(&PLANE_BASIS, &LINE_BASIS, &POINT_BASIS);
        DenseTable::from_instructions(&insts)
    })
}

/// Point ^ Point -> (should be 0 if same point, or grade-6 which doesn't exist in 4D)
/// Actually Point ^ Point is not useful in PGA since trivector ^ trivector = grade 6 > dim.
/// We use the regressive product for join instead.

// --- Inner products ---

/// Plane . Plane -> Scalar (inner product)
fn pln_pln_ip() -> &'static DenseTable<4, 4, 1> {
    static TABLE: OnceLock<DenseTable<4, 4, 1>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_inner_product(&PLANE_BASIS, &PLANE_BASIS, &SCA_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Line . Line -> Scalar
fn lin_lin_ip() -> &'static DenseTable<6, 6, 1> {
    static TABLE: OnceLock<DenseTable<6, 6, 1>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_inner_product(&LINE_BASIS, &LINE_BASIS, &SCA_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Plane . Line -> Plane (left contraction)
fn pln_lin_ip() -> &'static DenseTable<4, 6, 4> {
    static TABLE: OnceLock<DenseTable<4, 6, 4>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_inner_product(&PLANE_BASIS, &LINE_BASIS, &PLANE_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

// --- Regressive product tables ---
// The regressive product a & b = dual(dual(a) ^ dual(b))
// We compute this explicitly.

/// Dual of a Point -> Plane (in 4D, dual of grade 3 is grade 1)
/// Dual: blade -> blade ^ pseudoscalar complement
/// For each blade b, dual(b) = b* such that b ^ b* = pseudoscalar
/// In our convention: dual(b) = b XOR 0b1111, then we need the sign.
#[allow(dead_code)]
fn dual_sign(blade: u32) -> f32 {
    // Sign of (blade * dual(blade)) to get pseudoscalar in canonical order
    let d = blade ^ 0b1111;
    if crate::basis::sign_flip(blade, d) {
        -1.0
    } else {
        1.0
    }
}

/// Compute the dual of a point (trivector -> vector/plane)
/// POINT_BASIS  = `[e123, e124, e134, e234]`
/// PLANE_BASIS  = `[e1,   e2,   e3,   e4]`
/// dual(e123) = e4,  dual(e124) = e3,  dual(e134) = e2,  dual(e234) = e1
/// but with signs from reordering.
pub fn dual_point(p: &Point) -> Plane {
    // dual(e123=0b0111) = 0b1000=e4, sign: sign_flip(0b0111, 0b1000) = false -> +1
    // dual(e124=0b1011) = 0b0100=e3, sign: sign_flip(0b1011, 0b0100) -> ?
    // dual(e134=0b1101) = 0b0010=e2, sign: sign_flip(0b1101, 0b0010) -> ?
    // dual(e234=0b1110) = 0b0001=e1, sign: sign_flip(0b1110, 0b0001) -> ?
    //
    // Let's compute each sign carefully:
    // We need: blade * dual_blade = sign * pseudoscalar
    // e123 * e4 = e1234 -> sign_flip(0b0111, 0b1000): bits of a=0b0111, b=0b1000
    //   bit 0 of a: b bits below 0 = 0 swaps
    //   bit 1 of a: b bits below 1 = 0 swaps
    //   bit 2 of a: b bits below 2 = 0 swaps
    //   total 0 swaps -> +1
    // e124 * e3 = ? -> blade product e124*e3 = 0b1011 ^ 0b0100 = 0b1111 = e1234
    //   sign_flip(0b1011, 0b0100):
    //   bit 0 of a: b bits below 0 = 0
    //   bit 1 of a: b bits below 1 = 0
    //   bit 3 of a: b bits below 3 = 1 (bit 2)
    //   total 1 swap -> -1
    // e134 * e2 = 0b1101 ^ 0b0010 = 0b1111
    //   sign_flip(0b1101, 0b0010):
    //   bit 0 of a: b bits below 0 = 0
    //   bit 2 of a: b bits below 2 = 1 (bit 1)
    //   bit 3 of a: b bits below 3 = 1 (bit 1)
    //   total 2 swaps -> +1
    // e234 * e1 = 0b1110 ^ 0b0001 = 0b1111
    //   sign_flip(0b1110, 0b0001):
    //   bit 1 of a: b bits below 1 = 1 (bit 0)
    //   bit 2 of a: b bits below 2 = 1 (bit 0)
    //   bit 3 of a: b bits below 3 = 1 (bit 0)
    //   total 3 swaps -> -1

    // So: dual(e123) = +e4, dual(e124) = -e3, dual(e134) = +e2, dual(e234) = -e1
    // In PLANE_BASIS order [e1, e2, e3, e4]:
    //   e1 slot <- -p[3] (from -e234)
    //   e2 slot <- +p[2] (from +e134)
    //   e3 slot <- -p[1] (from -e124)
    //   e4 slot <- +p[0] (from +e123)
    Multivector::new([-p[3], p[2], -p[1], p[0]])
}

/// Compute the dual of a plane (vector -> trivector/point)
/// PLANE_BASIS = `[e1, e2, e3, e4]`
/// dual(e1) = e234, dual(e2) = e134, dual(e3) = e124, dual(e4) = e123
/// with signs:
///   e1 * e234 = 0b0001 ^ 0b1110 = 0b1111
///     sign_flip(0b0001, 0b1110): bit 0: b bits below 0 = 0 -> +1
///   e2 * e134 = 0b0010 ^ 0b1101 = 0b1111
///     sign_flip(0b0010, 0b1101): bit 1: b bits below 1 = 1 (bit 0) -> -1
///   e3 * e124 = 0b0100 ^ 0b1011 = 0b1111
///     sign_flip(0b0100, 0b1011): bit 2: b bits below 2 = 2 (bits 0,1) -> +1
///   e4 * e123 = 0b1000 ^ 0b0111 = 0b1111
///     sign_flip(0b1000, 0b0111): bit 3: b bits below 3 = 3 (bits 0,1,2) -> -1
///
/// So: dual(e1)=+e234, dual(e2)=-e134, dual(e3)=+e124, dual(e4)=-e123
/// In POINT_BASIS order `[e123, e124, e134, e234]`:
///   e123 slot <- -pi`[3]` (from -e4)
///   e124 slot <- +pi`[2]` (from +e3)
///   e134 slot <- -pi`[1]` (from -e2)
///   e234 slot <- +pi`[0]` (from +e1)
pub fn dual_plane(pi: &Plane) -> Point {
    Multivector::new([-pi[3], pi[2], -pi[1], pi[0]])
}

/// Compute the dual of a line (bivector -> bivector, self-dual in 4D)
/// LINE_BASIS = `[e12, e13, e23, e14, e24, e34]`
/// dual(e12)=e34, dual(e13)=e24, dual(e23)=e14, dual(e14)=e23, dual(e24)=e13, dual(e34)=e12
/// with signs:
///   e12*e34: sign_flip(0b0011, 0b1100) -> bits of a: bit0 (b below 0=0), bit1 (b below 1=0) -> 0 swaps -> +1
///   e13*e24: sign_flip(0b0101, 0b1010) -> bit0 (0), bit2 (1 swap for bit1) -> 1 swap -> -1
///   e23*e14: sign_flip(0b0110, 0b1001) -> bit1 (1 swap for bit0), bit2 (1 swap for bit0) -> 2 swaps -> +1
///   e14*e23: sign_flip(0b1001, 0b0110) -> bit0 (0), bit3 (2 swaps for bits1,2) -> 2 swaps -> +1
///   e24*e13: sign_flip(0b1010, 0b0101) -> bit1 (1 swap for bit0), bit3 (2 swaps for bits 0,2) -> 3 swaps -> -1
///   e34*e12: sign_flip(0b1100, 0b0011) -> bit2 (2 swaps for bits 0,1), bit3 (2 swaps for bits 0,1) -> 4 swaps -> +1
///
/// So: dual(e12)=+e34, dual(e13)=-e24, dual(e23)=+e14, dual(e14)=+e23, dual(e24)=-e13, dual(e34)=+e12
/// In LINE_BASIS order `[e12, e13, e23, e14, e24, e34]`:
///   e12 slot <- +l`[5]` (from +e34)
///   e13 slot <- -l`[4]` (from -e24)
///   e23 slot <- +l`[3]` (from +e14)
///   e14 slot <- +l`[2]` (from +e23)
///   e24 slot <- -l`[1]` (from -e13)
///   e34 slot <- +l`[0]` (from +e12)
pub fn dual_line(l: &Line) -> Line {
    Multivector::new([l[5], -l[4], l[3], l[2], -l[1], l[0]])
}

/// Undual (reverse dual). In PGA, undual(X) = dual(X) since dual^2 = +/- identity.
/// For even dimension (4D), dual^2 on a k-blade gives (-1)^(k*(n-k)) * product_of_metric.
/// Since e4^2=0, the pseudoscalar is degenerate and we need to be careful.
/// In practice, undual uses the same formula as dual for PGA with appropriate signs.
/// undual(X) = X * I (right complement) vs dual(X) = I * X (left complement).
/// For the regressive product, we use: a & b = undual(dual(a) ^ dual(b)).
/// In PGA Cl(3,0,1), the J-map (Poincare duality) satisfies J^2 = identity,
/// so undual = dual.
pub fn undual_point(p: &Point) -> Plane {
    dual_point(p)
}

/// Apply the inverse Poincare duality map to a plane.
pub fn undual_plane(pi: &Plane) -> Point {
    dual_plane(pi)
}

/// Apply the inverse Poincare duality map to a line.
pub fn undual_line(l: &Line) -> Line {
    dual_line(l)
}

// ============================================================================
// Product operations
// ============================================================================

/// Outer product of two planes -> Line (meet of planes).
#[inline]
pub fn op_pln_pln(a: &Plane, b: &Plane) -> Line {
    Multivector::new(pln_pln_op().execute(&a.data, &b.data))
}

/// Outer product of line ^ plane -> Point (meet of line and plane).
#[inline]
pub fn op_lin_pln(l: &Line, pi: &Plane) -> Point {
    Multivector::new(lin_pln_op().execute(&l.data, &pi.data))
}

/// Outer product of plane ^ line -> Point.
#[inline]
pub fn op_pln_lin(pi: &Plane, l: &Line) -> Point {
    Multivector::new(pln_lin_op().execute(&pi.data, &l.data))
}

/// Geometric product of two motors -> Motor.
#[inline]
pub fn gp_mot_mot(a: &Motor, b: &Motor) -> Motor {
    Multivector::new(mot_mot_gp().execute(&a.data, &b.data))
}

/// Geometric product of two planes -> Motor.
#[inline]
pub fn gp_pln_pln(a: &Plane, b: &Plane) -> Motor {
    Multivector::new(pln_pln_gp().execute(&a.data, &b.data))
}

/// Geometric product of two lines -> Motor.
#[inline]
pub fn gp_lin_lin(a: &Line, b: &Line) -> Motor {
    Multivector::new(lin_lin_gp().execute(&a.data, &b.data))
}

/// Inner product of two planes -> scalar (measures angle between planes).
#[inline]
pub fn ip_pln_pln(a: &Plane, b: &Plane) -> f32 {
    pln_pln_ip().execute(&a.data, &b.data)[0]
}

/// Inner product of two lines -> scalar.
#[inline]
pub fn ip_lin_lin(a: &Line, b: &Line) -> f32 {
    lin_lin_ip().execute(&a.data, &b.data)[0]
}

/// Inner product (left contraction) of plane into line -> plane.
#[inline]
pub fn ip_pln_lin(pi: &Plane, l: &Line) -> Plane {
    Multivector::new(pln_lin_ip().execute(&pi.data, &l.data))
}

// ============================================================================
// Regressive product (join/meet)
// ============================================================================

/// Regressive product of two points -> Line (join of two points).
/// a & b = dual(dual(a) ^ dual(b))
pub fn regressive_pnt_pnt(a: &Point, b: &Point) -> Line {
    let da = dual_point(a);
    let db = dual_point(b);
    let outer = op_pln_pln(&da, &db);
    dual_line(&outer)
}

/// Regressive product of point & line -> Plane (join of point and line).
/// a & b = dual(dual(a) ^ dual(b))
pub fn regressive_pnt_lin(p: &Point, l: &Line) -> Plane {
    let dp = dual_point(p);
    let dl = dual_line(l);
    // dual(point) = plane, dual(line) = line
    // plane ^ line -> point
    let outer = op_pln_lin(&dp, &dl);
    dual_point(&outer)
}

/// Regressive product of three points -> Plane (join of three points = plane through them).
pub fn regressive_pnt_pnt_pnt(a: &Point, b: &Point, c: &Point) -> Plane {
    let l = regressive_pnt_pnt(a, b);
    regressive_pnt_lin(c, &l)
}

// ============================================================================
// Unary operations
// ============================================================================

/// Reverse of a motor: negate the grade-2 and grade-4 parts.
/// Motor basis = `[s, e12, e13, e23, e14, e24, e34, e1234]`
/// Grade 0 (s): +1, Grade 2 (bivectors): -1, Grade 4 (e1234): +1
/// reverse_sign(grade k) = (-1)^(k*(k-1)/2)
/// grade 0: +1, grade 2: -1, grade 4: +1
#[inline]
pub fn reverse_motor(m: &Motor) -> Motor {
    Multivector::new([
        m[0],  // scalar: +
        -m[1], // e12: -
        -m[2], // e13: -
        -m[3], // e23: -
        -m[4], // e14: -
        -m[5], // e24: -
        -m[6], // e34: -
        m[7],  // e1234: +
    ])
}

/// Reverse of a line (grade 2 -> negated).
#[inline]
pub fn reverse_line(l: &Line) -> Line {
    -*l
}

/// Reverse of a plane (grade 1 -> unchanged).
#[inline]
pub fn reverse_plane(p: &Plane) -> Plane {
    *p
}

/// Reverse of a point (grade 3 -> negated).
#[inline]
pub fn reverse_point(p: &Point) -> Point {
    -*p
}

// ============================================================================
// Sandwich product (motor acting on elements)
// ============================================================================

/// Apply a motor to a point: p' = M * p * ~M
pub fn sandwich_point(m: &Motor, p: &Point) -> Point {
    let mp = mot_pnt_gp().execute(&m.data, &p.data);
    let m_rev = reverse_motor(m);
    let result = full_mot_gp().execute(&mp, &m_rev.data);
    // Extract point (trivector) part from FULL_BASIS
    // FULL_BASIS indices: grade-3 elements are at [11, 12, 13, 14] = e123, e124, e134, e234
    Multivector::new([result[11], result[12], result[13], result[14]])
}

/// Apply a motor to a line: l' = M * l * ~M
pub fn sandwich_line(m: &Motor, l: &Line) -> Line {
    let ml = mot_lin_gp().execute(&m.data, &l.data);
    let m_rev = reverse_motor(m);
    let result = full_mot_gp().execute(&ml, &m_rev.data);
    // Extract line (bivector) part from FULL_BASIS
    // FULL_BASIS indices: grade-2 elements are at [5, 6, 7, 8, 9, 10] = e12, e13, e23, e14, e24, e34
    Multivector::new([
        result[5], result[6], result[7], result[8], result[9], result[10],
    ])
}

/// Apply a motor to a plane: pi' = M * pi * ~M
pub fn sandwich_plane(m: &Motor, pi: &Plane) -> Plane {
    let mpi = mot_pln_gp().execute(&m.data, &pi.data);
    let m_rev = reverse_motor(m);
    let result = full_mot_gp().execute(&mpi, &m_rev.data);
    // Extract plane (vector) part from FULL_BASIS
    // FULL_BASIS indices: grade-1 elements are at [1, 2, 3, 4] = e1, e2, e3, e4
    Multivector::new([result[1], result[2], result[3], result[4]])
}

// ============================================================================
// Motor generation
// ============================================================================

/// Generate a rotor from a Euclidean bivector (rotation in the given plane).
/// R = exp(-B/2) = cos(|B|/2) - sin(|B|/2) * B/|B|
/// The bivector should have components only on e12, e13, e23 (Euclidean part).
pub fn gen_rotor(biv: &Line) -> Motor {
    // Extract only the Euclidean part: e12, e13, e23
    let b12 = biv[0];
    let b13 = biv[1];
    let b23 = biv[2];
    let angle = (b12 * b12 + b13 * b13 + b23 * b23).sqrt();

    if angle < 1e-10 {
        return motor(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    let half = angle / 2.0;
    let c = half.cos();
    let s = -half.sin() / angle;

    motor(c, s * b12, s * b13, s * b23, 0.0, 0.0, 0.0, 0.0)
}

/// Generate a translator motor from a direction vector (tx, ty, tz).
/// T = 1 + (d/2) where d = tx*e14 + ty*e24 + tz*e34
pub fn gen_translator(tx: f32, ty: f32, tz: f32) -> Motor {
    motor(1.0, 0.0, 0.0, 0.0, tx / 2.0, ty / 2.0, tz / 2.0, 0.0)
}

/// Generate a motor from a line bivector (general rigid motion = rotation + translation).
/// This is the exponential of a general bivector in PGA.
/// M = exp(-L/2) where L is a line (bivector).
///
/// For a line with Euclidean part d = (l12, l13, l23) and ideal part m = (l14, l24, l34):
///   |d| = angle of rotation
///   If |d| = 0: pure translation, M = 1 + m/2
///   Otherwise: M = cos(a/2) - sin(a/2)*d_hat + (m.d)/(2|d|) * (cos(a/2)*I - sin(a/2)*d_hat*I)
///     simplified via the PGA exponential formula.
pub fn gen_motor(l: &Line) -> Motor {
    let d12 = l[0];
    let d13 = l[1];
    let d23 = l[2];
    let m14 = l[3];
    let m24 = l[4];
    let m34 = l[5];

    let d_sq = d12 * d12 + d13 * d13 + d23 * d23;
    let d_norm = d_sq.sqrt();

    if d_norm < 1e-10 {
        // Pure translation
        return motor(1.0, 0.0, 0.0, 0.0, m14 / 2.0, m24 / 2.0, m34 / 2.0, 0.0);
    }

    let half = d_norm / 2.0;
    let cos_h = half.cos();
    let sin_h = half.sin();
    let sin_h_over_d = sin_h / d_norm;

    // Scalar part: cos(|d|/2)
    let _s = cos_h;

    // Euclidean bivector part: -sin(|d|/2) * d/|d|
    let _b12 = -sin_h_over_d * d12;
    let _b13 = -sin_h_over_d * d13;
    let _b23 = -sin_h_over_d * d23;

    // The ideal part involves the "moment" projection
    // m . d (the pseudo-scalar of the Plucker line = d12*m34 - d13*m24 + d23*m14)
    // Actually: the inner product of d and m in the Plucker sense
    let d_dot_m = d12 * m34 - d13 * m24 + d23 * m14;

    // Ideal bivector part: -sin(|d|/2)/|d| * m + (d.m)/(2*|d|^2) * (cos(|d|/2) - sin(|d|/2)/|d|) * d
    // Actually, using the standard PGA motor exponential:
    // For a line L = d + epsilon*m (where epsilon is the dual part),
    // exp(-L/2) = cos(a/2) - sin(a/2)*d_hat - epsilon*(p/2 * sin(a/2)*d_hat + cos(a/2)*m_perp_hat)
    // where a = |d|, d_hat = d/|d|, p = -d.m/|d|^2 (pitch), m_perp = m + p*d_hat (perpendicular moment)

    let _p = -d_dot_m / d_sq; // pitch
                              // m_perp = m - (m.d/|d|^2) * d_dual = m + p * hodge(d)
                              // Actually, m_perp[e14] = m14 + p*d23, etc. following the Hodge on the Euclidean part
                              // But simpler: use the formula directly:
                              // ideal_biv = -(sin(a/2)/|d|) * m_hat - (p/2) * cos(a/2) * d_hat
                              // where m_hat is the moment components scaled

    // Actually let me use a cleaner formulation.
    // Let l = magnitude of Euclidean part, s = d_dot_m / l^2 (pitch)
    // Motor = cos(l/2) - sin(l/2)*d/l + (s/2)(sin(l/2)*d/l - cos(l/2)*I)
    //       + (-sin(l/2)/l) * m_perp
    // where m_perp = m - s*dual(d) and I = e1234

    // m_perp = m - (pitch component projected onto d via duality)
    // For lines: dual(e12)=e34, dual(e13)=-e24, dual(e23)=e14
    // So dual(d) acting on ideal: m_perp_14 = m14 - p*d23
    //                              m_perp_24 = m24 + p*d13
    //                              m_perp_34 = m34 - p*d12
    // Wait, let me reconsider. Using the standard decomposition:
    // If L = l + epsilon*m is the Plucker line, pitch p = (l . m) / |l|^2
    // m_perp = m - p * l (subtracted in the ideal sense)
    // But l and m live in different subspaces (Euclidean and ideal bivectors),
    // so the "projection" uses the Hodge dual mapping between them.

    // Simpler approach: just use the formula
    // M[e14] = -sin_h_over_d * m14 - (p/2) * cos_h * (d23/d_norm)  ... no wait

    // Let me use the explicit motor exponential:
    // For bivector B = d + m (Euclidean + ideal parts), exp(-B/2) gives:
    // scalar: cos(a/2)
    // e12: -sin(a/2) * d12/a
    // e13: -sin(a/2) * d13/a
    // e23: -sin(a/2) * d23/a
    // e14: -sin(a/2)/a * m14 + d_dot_m/(2*a^2) * (sin(a/2)/a - cos(a/2)/a * ...) hmm

    // Let me just compute it properly using the dual number approach:
    // exp(-(d + eps*m)/2) = exp(-d/2) * (1 - eps*m/2)  (since eps^2 = 0 for ideal part)
    // where exp(-d/2) = cos(a/2) - sin(a/2)*d/a

    // So: M = (cos(a/2) - sin(a/2)*d/a) * (1 - m/2)  ... but m is ideal bivector

    // GP of (cos(a/2) - sin(a/2)*d/a) with (1 - m/2):
    // = cos(a/2) - sin(a/2)*d/a - cos(a/2)*m/2 + sin(a/2)*d*m/(2a)
    // The term d*m involves products like e12*e14 = ... these produce grade-4 (pseudoscalar) terms.

    // Let's compute d*m (geometric product of Euclidean and ideal bivectors):
    // Only the e1234 component survives (grade 4 from grade 2 * grade 2):
    // e12*e34 = e1234 (+1), e13*e24 = -e1234, e23*e14 = +e1234
    // So (d*m)_e1234 = d12*m34 - d13*m24 + d23*m14 = d_dot_m

    let _b14 = -sin_h_over_d * m14;
    let _b24 = -sin_h_over_d * m24;
    let _b34 = -sin_h_over_d * m34;
    let _b1234 = sin_h_over_d * d_dot_m / (2.0 * d_norm);
    // Wait: the term is sin(a/2)/(2a) * d_dot_m, but let me redo:
    // M = cos(a/2) - sin(a/2)/a * d - cos(a/2)/2 * m + sin(a/2)/(2a) * d_dot_m * e1234
    // Actually sin(a/2)/(2a) * d_dot_m for the e1234 component.

    // Hmm, but I also need: the ideal bivector part from -cos(a/2)*m/2
    // and from the Euclidean part acting on m... let me be more careful.

    // M_scalar = cos(a/2)
    // M_euclidean_biv = -sin(a/2)/a * d_i  (i = 12, 13, 23)
    // M_ideal_biv = -cos(a/2)/2 * m_i  (i = 14, 24, 34)
    //   Wait no -- the second factor is (1 - m/2) where m = m14*e14 + m24*e24 + m34*e34
    //   So -cos(a/2) * m_i/2 contributes to ideal bivector.
    // M_e1234 = +sin(a/2)/(2a) * (d12*m34 - d13*m24 + d23*m14)
    //         = sin(a/2)/(2a) * d_dot_m

    // But also: (-sin(a/2)/a * d) * (-m/2) = sin(a/2)/(2a) * d*m
    // d*m has e1234 component = d_dot_m, and potentially ideal bivector cross terms...
    // Actually no: e12*e14 = e1*e2*e1*e4 = -e2*e4 = -e24 (grade 2, not grade 4)
    // Wait: basis::product(E12, E14) = 0b0011 ^ 0b1001 = 0b1010 = E24. That's grade 2.
    // So d*m has both grade-2 AND grade-4 components!

    // Let me be really explicit. d*m with d = d12*e12 + d13*e13 + d23*e23, m = m14*e14 + m24*e24 + m34*e34:
    // Products (XOR for blade, need sign):
    // e12*e14: XOR=0b0011^0b1001=0b1010=e24, sign_flip(0b0011,0b1001)=bit0:(1 bit below=0),bit1:(1 bit below=0)=0->+1
    //   but metric: shared bits = 0b0001=e1, metric_sign(e1)=1. So sign=+1*1=+1. Hmm wait.
    //   Actually for GP: sign = reorder_sign * metric_sign
    //   metric_sign(0b0011, 0b1001) = metric_signs for shared bits (0b0011 & 0b1001 = 0b0001 = e1) = metric[0] = 1
    //   reorder_sign: sign_flip(0b0011, 0b1001):
    //     bit 0 of a (0b0011): b bits below 0 in 0b1001 = 0 -> 0 swaps
    //     bit 1 of a (0b0011): b bits below 1 in 0b1001 = 1 (bit 0) -> 1 swap
    //     total = 1 -> negative
    //   So e12*e14 = -e24 (with metric +1, so -1 * e24)

    // This is getting complex. Let me just compute this using the product table machinery.
    // Actually, the cleanest approach: construct the full line bivector, embed into motor basis, and use GP.
    // Or better: use a simpler formula.

    // CLEANER APPROACH: Use the known PGA motor exponential formula.
    // Given a normalized line L with direction d (unit) and moment m,
    // and we want rotation by angle theta about L plus translation by distance t along L:
    // The bivector is B = (theta * d + t * m) where d is Euclidean part, m is ideal part.
    // exp(-B/2) = cos(th/2) - sin(th/2)*d - (t/2)(cos(th/2)*m + sin(th/2)*...) ...
    // This is also getting complicated. Let me just use a different strategy.

    // SIMPLEST CORRECT APPROACH: Use the known closed-form:
    // M = cos(a/2) + sin(a/2)/a * (-d) + 1/2 * (-m) * (cos(a/2) + sin(a/2)/a * (-d))
    // where a = |d|. This uses the fact that ideal^2 = 0.
    // Expanding:
    // M = cos(a/2) - sin(a/2)/a * d - m/2 * cos(a/2) + m/2 * sin(a/2)/a * d
    // The last term m*d (ideal_biv * euclidean_biv GP) needs computation.

    // I'll precompute m*d's contributions to each grade.
    // Let me use the product tables for this.
    // Actually, let's just build the motor by composing a rotor and translator.
    // This avoids the complex exponential formula entirely.

    // For a line with Euclidean direction d and ideal moment m:
    // The rotation is about the axis d.
    // The translation along the axis is the pitch component.
    // M = T * R where R = exp(-d/2) and T = exp(-m_parallel/2)
    // But m_parallel involves projection... this is still complex.

    // OK, I'll just implement it numerically using the GP tables.
    // Build R = cos(a/2) - sin(a/2)/a * [d12*e12 + d13*e13 + d23*e23]
    // Build the ideal part as a separate motor and multiply.

    let r = motor(
        cos_h,
        -sin_h_over_d * d12,
        -sin_h_over_d * d13,
        -sin_h_over_d * d23,
        0.0,
        0.0,
        0.0,
        0.0,
    );
    let ideal_part = motor(1.0, 0.0, 0.0, 0.0, -m14 / 2.0, -m24 / 2.0, -m34 / 2.0, 0.0);
    gp_mot_mot(&ideal_part, &r)
}

// ============================================================================
// Geometric operations
// ============================================================================

/// Normalize a point (divide by the e123 component so it equals 1).
pub fn normalize_point(p: &Point) -> Point {
    let w = p[0]; // e123 component
    if w.abs() < 1e-10 {
        return *p; // ideal point, can't normalize
    }
    Multivector::new([1.0, p[1] / w, p[2] / w, p[3] / w])
}

/// Extract Euclidean coordinates from a normalized point.
/// Point = e123 - z*e124 + y*e134 - x*e234
/// So: x = -p`[3]`, y = p`[2]`, z = -p`[1]` (after normalization, p`[0]` = 1)
pub fn point_to_xyz(p: &Point) -> [f32; 3] {
    let pn = normalize_point(p);
    [-pn[3], pn[2], -pn[1]]
}

/// Distance between two normalized points.
/// In PGA, the distance between points P and Q is:
/// |P & Q| / (|P_e123| * |Q_e123|)
/// where & is the regressive product (join).
/// Actually: d(P,Q) = |normalize(P v Q)| where v is the regressive product
/// and the norm of the resulting line is computed from its Euclidean part.
pub fn distance_points(p: &Point, q: &Point) -> f32 {
    // Simpler: extract Euclidean coordinates and compute Euclidean distance
    let [px, py, pz] = point_to_xyz(p);
    let [qx, qy, qz] = point_to_xyz(q);
    let dx = px - qx;
    let dy = py - qy;
    let dz = pz - qz;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Angle between two planes (cosine of angle = inner product of unit normals).
pub fn angle_planes(a: &Plane, b: &Plane) -> f32 {
    // The Euclidean normal is (a[0], a[1], a[2]) for plane a*e1 + b*e2 + c*e3 + d*e4
    let na = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    let nb = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
    if na < 1e-10 || nb < 1e-10 {
        return 0.0;
    }
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let cos_angle = (dot / (na * nb)).clamp(-1.0, 1.0);
    cos_angle.acos()
}

/// Normalize a plane so that its Euclidean normal has unit length.
pub fn normalize_plane(pi: &Plane) -> Plane {
    let n = (pi[0] * pi[0] + pi[1] * pi[1] + pi[2] * pi[2]).sqrt();
    if n < 1e-10 {
        return *pi;
    }
    *pi * (1.0 / n)
}

/// Normalize a line so that its Euclidean (direction) part has unit length.
pub fn normalize_line(l: &Line) -> Line {
    let n = (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).sqrt();
    if n < 1e-10 {
        return *l;
    }
    *l * (1.0 / n)
}

/// Normalize a motor to unit magnitude.
pub fn normalize_motor(m: &Motor) -> Motor {
    // A motor M has scalar and bivector parts. The normalization condition is M * ~M = 1.
    // For a simple motor, this means s^2 + (euclidean bivector)^2 = 1.
    let s = m[0];
    let biv_sq = m[1] * m[1] + m[2] * m[2] + m[3] * m[3];
    let n = (s * s + biv_sq).sqrt();
    if n < 1e-10 {
        return *m;
    }
    *m * (1.0 / n)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const EPS: f32 = 1e-4;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    // ---- Basic construction ----

    #[test]
    fn test_point_creation() {
        let p = point(1.0, 2.0, 3.0);
        // POINT_BASIS = [e123, e124, e134, e234]
        // point(x,y,z) = e123 - z*e124 + y*e134 - x*e234
        assert_eq!(p[0], 1.0); // e123
        assert_eq!(p[1], -3.0); // e124 = -z
        assert_eq!(p[2], 2.0); // e134 = y
        assert_eq!(p[3], -1.0); // e234 = -x
    }

    #[test]
    fn test_plane_creation() {
        let pi = plane(1.0, 0.0, 0.0, 5.0);
        assert_eq!(pi[0], 1.0); // e1
        assert_eq!(pi[3], 5.0); // e4
    }

    #[test]
    fn test_point_roundtrip() {
        let p = point(3.0, -2.0, 7.0);
        let [x, y, z] = point_to_xyz(&p);
        assert!(approx_eq(x, 3.0));
        assert!(approx_eq(y, -2.0));
        assert!(approx_eq(z, 7.0));
    }

    // ---- Duality ----

    #[test]
    fn test_dual_point_plane_roundtrip() {
        // In 4D PGA, dual∘dual = -1 for points (grade 3) and planes (grade 1).
        let p = point(1.0, 2.0, 3.0);
        let pi = dual_point(&p);
        let p2 = dual_plane(&pi);
        for i in 0..4 {
            assert!(
                approx_eq(p[i], -p2[i]),
                "Dual roundtrip failed at {}: {} vs -{}",
                i,
                p[i],
                p2[i]
            );
        }
    }

    #[test]
    fn test_dual_line_roundtrip() {
        let l = line(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        let dl = dual_line(&l);
        let l2 = dual_line(&dl);
        for i in 0..6 {
            assert!(
                approx_eq(l[i], l2[i]),
                "Line dual roundtrip failed at {}: {} vs {}",
                i,
                l[i],
                l2[i]
            );
        }
    }

    // ---- Outer product (meet) ----

    #[test]
    fn test_two_planes_meet_line() {
        // XZ plane (y=0): e2
        let pi1 = plane(0.0, 1.0, 0.0, 0.0);
        // YZ plane (x=0): e1
        let pi2 = plane(1.0, 0.0, 0.0, 0.0);
        // Meet should be the z-axis
        let l = op_pln_pln(&pi1, &pi2);
        // e2 ^ e1 = -e12, so l should have e12 component = -1
        // The z-axis has direction e12 (or -e12) in PGA
        assert!(l[0].abs() > 0.5, "Expected nonzero e12: {:?}", l.data);
        // Ideal components should be zero (line through origin)
        assert!(approx_eq(l[3], 0.0));
        assert!(approx_eq(l[4], 0.0));
        assert!(approx_eq(l[5], 0.0));
    }

    #[test]
    fn test_three_planes_meet_point() {
        // x=1: e1 + e4
        let _pi1 = plane(1.0, 0.0, 0.0, -1.0);
        // y=2: e2 + 2*e4... wait, plane equation is a*e1 + b*e2 + c*e3 + d*e4
        // For the plane x=1: normal=(1,0,0), offset=-1 -> e1 - e4
        // Actually: a plane a*x + b*y + c*z + d = 0 in Euclidean space
        // maps to a*e1 + b*e2 + c*e3 + d*e4.
        // x = 1 -> x - 1 = 0 -> e1 - e4
        let pi1 = plane(1.0, 0.0, 0.0, -1.0);
        // y = 2 -> y - 2 = 0 -> e2 - 2*e4
        let pi2 = plane(0.0, 1.0, 0.0, -2.0);
        // z = 3 -> z - 3 = 0 -> e3 - 3*e4
        let pi3 = plane(0.0, 0.0, 1.0, -3.0);

        let p = point_from_planes(&pi1, &pi2, &pi3);
        let [x, y, z] = point_to_xyz(&p);
        assert!(approx_eq(x, 1.0), "x = {} expected 1.0", x);
        assert!(approx_eq(y, 2.0), "y = {} expected 2.0", y);
        assert!(approx_eq(z, 3.0), "z = {} expected 3.0", z);
    }

    // ---- Regressive product (join) ----

    #[test]
    fn test_join_two_points_gives_line() {
        let p1 = point(0.0, 0.0, 0.0); // origin
        let p2 = point(1.0, 0.0, 0.0); // x=1
        let l = line_from_points(&p1, &p2);
        // This should be a line along the x-axis
        // Normalize and check that it has a direction
        let n = (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).sqrt();
        assert!(
            n > EPS,
            "Line should have nonzero Euclidean direction: {:?}",
            l.data
        );
    }

    #[test]
    fn test_join_three_points_gives_plane() {
        let p1 = point(0.0, 0.0, 0.0);
        let p2 = point(1.0, 0.0, 0.0);
        let p3 = point(0.0, 1.0, 0.0);
        // Join of three points in the xy-plane should give z=0 plane (e3)
        let pi = regressive_pnt_pnt_pnt(&p1, &p2, &p3);
        // The plane should have its normal in the z-direction
        let n = normalize_plane(&pi);
        assert!(
            n[2].abs() > 0.5,
            "Expected plane normal along z: {:?}",
            n.data
        );
        // d (e4 component) should be ~0 since plane passes through origin
        assert!(
            approx_eq(n[3], 0.0),
            "Expected plane through origin, d = {}",
            n[3]
        );
    }

    // ---- Motor operations ----

    #[test]
    fn test_identity_motor() {
        let m = motor(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let p = point(1.0, 2.0, 3.0);
        let result = sandwich_point(&m, &p);
        let [x, y, z] = point_to_xyz(&result);
        assert!(approx_eq(x, 1.0), "x = {}", x);
        assert!(approx_eq(y, 2.0), "y = {}", y);
        assert!(approx_eq(z, 3.0), "z = {}", z);
    }

    #[test]
    fn test_translation() {
        // Translate by (5, 0, 0)
        let t = gen_translator(5.0, 0.0, 0.0);
        let p = point(1.0, 2.0, 3.0);
        let result = sandwich_point(&t, &p);
        let [x, y, z] = point_to_xyz(&result);
        assert!(approx_eq(x, 6.0), "x = {} expected 6.0", x);
        assert!(approx_eq(y, 2.0), "y = {} expected 2.0", y);
        assert!(approx_eq(z, 3.0), "z = {} expected 3.0", z);
    }

    #[test]
    fn test_translation_y() {
        let t = gen_translator(0.0, 3.0, 0.0);
        let p = point(0.0, 0.0, 0.0);
        let result = sandwich_point(&t, &p);
        let [x, y, z] = point_to_xyz(&result);
        assert!(approx_eq(x, 0.0), "x = {}", x);
        assert!(approx_eq(y, 3.0), "y = {} expected 3.0", y);
        assert!(approx_eq(z, 0.0), "z = {}", z);
    }

    #[test]
    fn test_rotation_90_xy() {
        // Rotation of PI/2 in the e12 plane (rotation in xy, i.e., about z-axis)
        let biv = line(PI / 2.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let r = gen_rotor(&biv);
        let p = point(1.0, 0.0, 0.0);
        let result = sandwich_point(&r, &p);
        let [x, y, z] = point_to_xyz(&result);
        assert!(approx_eq(x, 0.0), "x = {} expected 0.0", x);
        assert!(approx_eq(y, 1.0), "y = {} expected 1.0", y);
        assert!(approx_eq(z, 0.0), "z = {} expected 0.0", z);
    }

    #[test]
    fn test_rotation_preserves_distance() {
        let biv = line(1.0, 0.5, 0.3, 0.0, 0.0, 0.0);
        let r = gen_rotor(&biv);
        let p = point(1.0, 2.0, 3.0);
        let q = point(4.0, 5.0, 6.0);
        let rp = sandwich_point(&r, &p);
        let rq = sandwich_point(&r, &q);
        let d_orig = distance_points(&p, &q);
        let d_rot = distance_points(&rp, &rq);
        assert!(
            approx_eq(d_orig, d_rot),
            "Distance not preserved: {} vs {}",
            d_orig,
            d_rot
        );
    }

    #[test]
    fn test_motor_composition() {
        // Translate then rotate should compose as motor product
        let t = gen_translator(1.0, 0.0, 0.0);
        let biv = line(PI / 2.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let r = gen_rotor(&biv);
        let p = point(0.0, 0.0, 0.0);

        // Apply separately: first translate, then rotate
        let pt = sandwich_point(&t, &p);
        let ptr = sandwich_point(&r, &pt);

        // Apply as composed motor: M = R * T (right-to-left application)
        let m = gp_mot_mot(&r, &t);
        let pm = sandwich_point(&m, &p);

        let [x1, y1, z1] = point_to_xyz(&ptr);
        let [x2, y2, z2] = point_to_xyz(&pm);
        assert!(approx_eq(x1, x2), "x: {} vs {}", x1, x2);
        assert!(approx_eq(y1, y2), "y: {} vs {}", y1, y2);
        assert!(approx_eq(z1, z2), "z: {} vs {}", z1, z2);
    }

    #[test]
    fn test_motor_reversal() {
        let m = gen_translator(3.0, 4.0, 5.0);
        let m_rev = reverse_motor(&m);
        let identity = gp_mot_mot(&m, &m_rev);
        // Should be approximately the identity motor
        assert!(approx_eq(identity[0], 1.0), "scalar = {}", identity[0]);
        for i in 1..7 {
            assert!(
                approx_eq(identity[i], 0.0),
                "Motor[{}] = {} expected 0",
                i,
                identity[i]
            );
        }
    }

    // ---- Sandwich on lines and planes ----

    #[test]
    fn test_translate_line() {
        // Line along x-axis through origin
        let p1 = point(0.0, 0.0, 0.0);
        let p2 = point(1.0, 0.0, 0.0);
        let l = line_from_points(&p1, &p2);

        // Translate by (0, 1, 0)
        let t = gen_translator(0.0, 1.0, 0.0);
        let l2 = sandwich_line(&t, &l);

        // The translated line should pass through (0,1,0) and (1,1,0)
        // Verify by checking that (0,1,0) lies on the translated line
        // A point lies on a line if their outer product is zero
        // In PGA: l ^ p = 0 iff p is on l... but l is grade 2 and p is grade 3 -> l^p = grade 5 which is 0 in 4D
        // Actually we check using the inner product: p . l = 0 iff p is on l (in the dual sense)
        // Or just translate the endpoints and check
        let p1t = sandwich_point(&t, &p1);
        let p2t = sandwich_point(&t, &p2);
        let [x1, y1, _z1] = point_to_xyz(&p1t);
        let [x2, y2, _z2] = point_to_xyz(&p2t);
        assert!(approx_eq(y1, 1.0), "p1 y = {} expected 1.0", y1);
        assert!(approx_eq(x1, 0.0));
        assert!(approx_eq(y2, 1.0), "p2 y = {} expected 1.0", y2);
        assert!(approx_eq(x2, 1.0));

        // Also verify l2 is nonzero
        let n = (l2[0] * l2[0] + l2[1] * l2[1] + l2[2] * l2[2]).sqrt();
        assert!(n > EPS, "Translated line has zero direction");
    }

    #[test]
    fn test_translate_plane() {
        // z=0 plane: e3
        let pi = plane(0.0, 0.0, 1.0, 0.0);
        // Translate by (0, 0, 5)
        let t = gen_translator(0.0, 0.0, 5.0);
        let pi2 = sandwich_plane(&t, &pi);
        let pi2n = normalize_plane(&pi2);
        // Should be z=5 plane: e3 - 5*e4 (normalized)
        assert!(
            approx_eq(pi2n[2], 1.0) || approx_eq(pi2n[2], -1.0),
            "Normal should be along z: {:?}",
            pi2n.data
        );
        assert!(
            approx_eq(pi2n[3].abs(), 5.0),
            "Distance should be 5: d = {}",
            pi2n[3]
        );
    }

    // ---- Distance and angle ----

    #[test]
    fn test_distance_between_points() {
        let p1 = point(0.0, 0.0, 0.0);
        let p2 = point(3.0, 4.0, 0.0);
        let d = distance_points(&p1, &p2);
        assert!(approx_eq(d, 5.0), "Distance = {} expected 5.0", d);
    }

    #[test]
    fn test_angle_between_planes() {
        // XY plane (e3) and XZ plane (e2)
        let pi1 = plane(0.0, 0.0, 1.0, 0.0);
        let pi2 = plane(0.0, 1.0, 0.0, 0.0);
        let angle = angle_planes(&pi1, &pi2);
        assert!(
            approx_eq(angle, PI / 2.0),
            "Angle = {} expected {}",
            angle,
            PI / 2.0
        );
    }

    #[test]
    fn test_angle_parallel_planes() {
        let pi1 = plane(1.0, 0.0, 0.0, 0.0);
        let pi2 = plane(1.0, 0.0, 0.0, 5.0);
        let angle = angle_planes(&pi1, &pi2);
        assert!(approx_eq(angle, 0.0), "Parallel planes angle = {}", angle);
    }

    // ---- Line from planes / points ----

    #[test]
    fn test_line_from_planes() {
        let pi1 = plane(1.0, 0.0, 0.0, 0.0); // YZ plane
        let pi2 = plane(0.0, 1.0, 0.0, 0.0); // XZ plane
        let l = line_from_planes(&pi1, &pi2);
        // Intersection is the z-axis
        let n = (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).sqrt();
        assert!(n > EPS, "Line should have nonzero direction");
    }

    #[test]
    fn test_line_from_points_direction() {
        let p1 = point(0.0, 0.0, 0.0);
        let p2 = point(0.0, 0.0, 1.0);
        let l = line_from_points(&p1, &p2);
        let n = (l[0] * l[0] + l[1] * l[1] + l[2] * l[2]).sqrt();
        assert!(n > EPS, "Line direction should be nonzero");
    }

    // ---- General motor (rotation + translation) ----

    #[test]
    fn test_gen_motor_pure_rotation() {
        // Pure rotation: no ideal part
        let l = line(PI / 2.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let m = gen_motor(&l);
        let p = point(1.0, 0.0, 0.0);
        let result = sandwich_point(&m, &p);
        let [x, y, z] = point_to_xyz(&result);
        assert!(approx_eq(x, 0.0), "x = {} expected 0.0", x);
        assert!(approx_eq(y, 1.0), "y = {} expected 1.0", y);
        assert!(approx_eq(z, 0.0), "z = {} expected 0.0", z);
    }

    #[test]
    fn test_gen_motor_pure_translation() {
        // Pure translation: no Euclidean part
        let l = line(0.0, 0.0, 0.0, 0.0, 0.0, 3.0);
        let m = gen_motor(&l);
        let p = point(0.0, 0.0, 0.0);
        let result = sandwich_point(&m, &p);
        let [_x, _y, _z] = point_to_xyz(&result);
        // e34 component translates in some direction... let's just check it moved
        // Actually, e14 -> x translation, e24 -> y translation, e34 -> z translation
        // m34 = 3.0 -> translate z by some amount
        // For translator T = 1 - (d/2)*e_i4, sandwich gives translation by d.
        // Here motor = 1 + (-3.0/2)*e34, so translation in z by... let's check.
        // gen_motor with m34=3: motor(1, 0,0,0, 0,0,-1.5, 0)
        // sandwich_point with that should translate z.
        let d = distance_points(&p, &result);
        assert!(d > EPS, "Point should have moved, d = {}", d);
    }

    // ---- Inner product ----

    #[test]
    fn test_inner_product_planes() {
        let pi1 = plane(1.0, 0.0, 0.0, 0.0);
        let pi2 = plane(1.0, 0.0, 0.0, 0.0);
        let ip = ip_pln_pln(&pi1, &pi2);
        assert!(
            approx_eq(ip, 1.0),
            "Self inner product = {} expected 1.0",
            ip
        );
    }

    #[test]
    fn test_inner_product_orthogonal_planes() {
        let pi1 = plane(1.0, 0.0, 0.0, 0.0);
        let pi2 = plane(0.0, 1.0, 0.0, 0.0);
        let ip = ip_pln_pln(&pi1, &pi2);
        assert!(approx_eq(ip, 0.0), "Orthogonal planes IP = {}", ip);
    }

    // ---- Misc ----

    #[test]
    fn test_normalize_point() {
        let p = Multivector::new([2.0, -6.0, 4.0, -2.0]); // 2 * point(1,2,3)
        let pn = normalize_point(&p);
        assert!(approx_eq(pn[0], 1.0));
        let [x, y, z] = point_to_xyz(&pn);
        assert!(approx_eq(x, 1.0));
        assert!(approx_eq(y, 2.0));
        assert!(approx_eq(z, 3.0));
    }

    #[test]
    fn test_motor_normalize() {
        let m = motor(2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let mn = normalize_motor(&m);
        assert!(approx_eq(mn[0], 1.0));
    }

    #[test]
    fn test_geometric_product_planes() {
        // GP of two orthogonal planes should give a rotor component
        let pi1 = plane(1.0, 0.0, 0.0, 0.0); // e1
        let pi2 = plane(0.0, 1.0, 0.0, 0.0); // e2
        let m = gp_pln_pln(&pi1, &pi2);
        // e1 * e2 = e12, so motor should have e12 component
        assert!(m[1].abs() > 0.5, "Expected e12 component: {:?}", m.data);
        assert!(approx_eq(m[0], 0.0), "Expected zero scalar");
    }
}
