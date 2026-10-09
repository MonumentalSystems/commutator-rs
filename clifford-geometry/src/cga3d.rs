//! Conformal 3D Geometric Algebra — Cl(4,1).
//!
//! Port of Versor's CGA3D types. Basis: e1, e2, e3, e4 (e+), e5 (e-).
//! 2^5 = 32 total basis blades. Metric: `[1, 1, 1, 1, -1]`.
//!
//! Null vectors:
//!   origin o = 0.5*(e5 - e4)
//!   infinity inf = e4 + e5
//!
//! A conformal point at (x, y, z) is:
//!   P = o + x*e1 + y*e2 + z*e3 + 0.5*(x^2+y^2+z^2)*inf
//!
//! Types follow the original Versor library naming conventions.

use crate::mvec::Multivector;
use crate::products::{self, DenseTable};
use std::sync::OnceLock;

// ============================================================================
// Metric
// ============================================================================

/// Cl(4,1) conformal metric: e1^2=+1, e2^2=+1, e3^2=+1, e4^2=+1, e5^2=-1.
pub const METRIC: [i32; 5] = [1, 1, 1, 1, -1];

// ============================================================================
// Blade bitmasks
// ============================================================================

// Grade 0: scalar
const S: u32 = 0b00000;

// Grade 1: vectors
const E1: u32 = 0b00001;
const E2: u32 = 0b00010;
const E3: u32 = 0b00100;
const E4: u32 = 0b01000;
const E5: u32 = 0b10000;

// Grade 2: bivectors
const E12: u32 = 0b00011;
const E13: u32 = 0b00101;
const E23: u32 = 0b00110;
const E14: u32 = 0b01001;
const E24: u32 = 0b01010;
const E34: u32 = 0b01100;
const E15: u32 = 0b10001;
const E25: u32 = 0b10010;
const E35: u32 = 0b10100;
const E45: u32 = 0b11000;

// Grade 3: trivectors
const E123: u32 = 0b00111;
const E124: u32 = 0b01011;
const E134: u32 = 0b01101;
const E234: u32 = 0b01110;
const E125: u32 = 0b10011;
const E135: u32 = 0b10101;
const E235: u32 = 0b10110;
const E145: u32 = 0b11001;
const E245: u32 = 0b11010;
const E345: u32 = 0b11100;

// Grade 4: quadvectors
const E1234: u32 = 0b01111;
const E1235: u32 = 0b10111;
const E1245: u32 = 0b11011;
const E1345: u32 = 0b11101;
const E2345: u32 = 0b11110;

// Grade 5: pseudoscalar
const E12345: u32 = 0b11111;

// ============================================================================
// Blade layouts (basis arrays for each type)
// ============================================================================

/// Scalar: `[s]`
pub const SCA_BASIS: [u32; 1] = [S];

/// Euclidean vector: `[e1, e2, e3]`
pub const VEC_BASIS: [u32; 3] = [E1, E2, E3];

/// Euclidean bivector: `[e12, e13, e23]`
pub const BIV_BASIS: [u32; 3] = [E12, E13, E23];

/// Euclidean trivector: `[e123]`
pub const TRI_BASIS: [u32; 1] = [E123];

/// Euclidean rotor: `[s, e12, e13, e23]`
pub const ROT_BASIS: [u32; 4] = [S, E12, E13, E23];

/// Minkowski blade: `[e45]`
pub const MNK_BASIS: [u32; 1] = [E45];

/// Pseudoscalar: `[e12345]`
pub const PSS_BASIS: [u32; 1] = [E12345];

/// Point / DualSphere: `[e1, e2, e3, e4, e5]`  (grade-1 in 5D)
pub const PNT_BASIS: [u32; 5] = [E1, E2, E3, E4, E5];

/// Pair (PointPair): all grade-2 blades (10 total)
pub const PAR_BASIS: [u32; 10] = [E12, E13, E23, E14, E24, E34, E15, E25, E35, E45];

/// Circle: all grade-3 blades (10 total)
pub const CIR_BASIS: [u32; 10] = [E123, E124, E134, E234, E125, E135, E235, E145, E245, E345];

/// Sphere: all grade-4 blades (5 total)
pub const SPH_BASIS: [u32; 5] = [E1234, E1235, E1245, E1345, E2345];

/// DualSphere = same as Point
pub const DLS_BASIS: [u32; 5] = [E1, E2, E3, E4, E5];

/// Flat Point: `[e15, e25, e35, e45]`  (these are e_i ^ einf)
pub const FLP_BASIS: [u32; 4] = [E15, E25, E35, E45];

/// Dual Line: `[e12, e13, e23, e15, e25, e35]`  (bivector without e_i4 except via null)
/// Actually: grade-2 blades that form a line in CGA.
/// Dll = e12, e13, e23, e15, e25, e35
pub const DLL_BASIS: [u32; 6] = [E12, E13, E23, E15, E25, E35];

/// Line: `[e123, e145, e245, e345, e125, e135, e235]`
/// Wait -- Line in Versor is grade 3 with 6 components:
/// e145, e245, e345, e125, e135, e235
pub const LIN_BASIS: [u32; 6] = [E145, E245, E345, E125, E135, E235];

/// Dual Plane: `[e1, e2, e3, e5]`  (conformal vector minus e4 slot, but actually it's specific)
/// Dlp = grade-1 elements that form a plane: e1, e2, e3, e5
pub const DLP_BASIS: [u32; 4] = [E1, E2, E3, E5];

/// Plane: `[e1235, e1245, e1345, e2345]`
pub const PLN_BASIS: [u32; 4] = [E1235, E1245, E1345, E2345];

/// Direction vector: `[e14, e24, e34]` (or using null: e_i ^ einf projected)
/// Drv components: e15, e25, e35 -- actually in Versor Drv = e_i wedge ninf
/// which maps to e15+e14, e25+e24, e35+e34... Let me use the simpler form.
/// In Versor, Drv stores: e15, e25, e35  (direction = point at infinity contribution)
pub const DRV_BASIS: [u32; 3] = [E15, E25, E35];

/// Direction bivector: `[e125, e135, e235]`
pub const DRB_BASIS: [u32; 3] = [E125, E135, E235];

/// Tangent vector: `[e1, e2, e3, e4, e5]` (same layout as Pnt but used differently)
pub const TNV_BASIS: [u32; 5] = [E1, E2, E3, E4, E5];

/// Translator: `[s, e15, e25, e35]`  (T = 1 - d/2 * ninf)
/// In our basis, ninf = e4 + e5, so e_i * ninf components map to e_i4 + e_i5.
/// But for a translator, the bivector parts are: e_i ^ ninf.
/// e1^(e4+e5) = e14 + e15.  Hmm, that's two components each.
/// Actually in the Versor convention, Trs stores: `[s, e14+e15, e24+e25, e34+e35]`
/// which is NOT a standard basis decomposition. Let me think...
///
/// Actually, Versor's Trs IS stored as `[s, e15, e25, e35]` when using the null basis
/// representation where no = e5-e4, ni = e4+e5. The translator in that convention is
/// T = 1 - (d/2)*ni where d = a*e1+b*e2+c*e3. The bivector part is -(d^ni)/2.
///
/// In the standard basis:
///   e1 ^ ni = e1 ^ (e4 + e5) = e14 + e15
///
/// But the Versor library typically stores the translator in a compact form.
/// For simplicity, let's store the full motor instead.
/// Translator: `[s, e14, e24, e34, e15, e25, e35]`
/// Actually let's just do full Motor basis for Trs too. But that's wasteful.
///
/// Let me use a pragmatic approach: Store Trs in motor basis (8 components).
/// Trs basis = `[s, e12, e13, e23, e15, e25, e35, e1235]`
/// No wait -- a translator only has s and e_i^ninf parts.
///
/// For a pure translator T = 1 - (a*e15 + a*e14 + b*e25 + b*e24 + c*e35 + c*e34)/2
/// This doesn't decompose cleanly into a small number of standard basis blades.
///
/// The practical solution: use Motor (8-component) for everything that's a versor,
/// and provide constructor functions that build translators/rotors as special cases.

/// Motor: `[s, e12, e13, e23, e1245, e1345, e2345, e12345]`
/// This is the even-grade subalgebra elements that form motors.
/// Wait -- motors in CGA are: scalar + bivector + quadvector + pseudoscalar (even grades).
/// Actually a general motor = R * T where R is rotor (s + biv_eucl) and T is translator.
/// The full even subalgebra has 16 components. But practical motors use 8.
///
/// In Versor, Motor stores 8 components:
/// `[s, e12, e13, e23, e1245, e1345, e2345, e12345]`
/// But that doesn't seem right either. Let me think about this more carefully.
///
/// A motor M = T * R where T = 1 - t*ninf/2, R = cos(a/2) - sin(a/2)*B
/// Expanding: M = R - (t^ninf/2)*R
/// The scalar + bivector_eucl part comes from R.
/// The e_i^ninf part (mixed Eucl-null bivectors) comes from translation.
///
/// Motor in Versor (C++) stores: s, e12, e13, e23, e15, e25, e35, e1235
/// These are: scalar, Euclidean bivectors, null bivectors (e_i5 form), and e1235.
///
/// Hmm wait, actually looking at standard CGA motor decomposition:
/// A general motor M has components in grades 0, 2, 4:
///   grade 0: scalar
///   grade 2: e12, e13, e23, e15, e25, e35 (the "dual line" blades)
///   grade 4: e1235
/// That's 1 + 6 + 1 = 8 components. This matches Dll_basis + scalar + e1235.

pub const MOT_BASIS: [u32; 8] = [S, E12, E13, E23, E15, E25, E35, E1235];

/// Translator: `[s, e15, e25, e35]`
/// A pure translator T = 1 - (d^ninf)/2, and in our basis (metric `[1,1,1,1,-1]`),
/// ninf = e4 + e5. So e1^ninf = e14 + e15.
/// BUT: in the null basis representation used by Versor, the translator simplifies.
/// The key insight: Versor uses a specific internal representation where
/// Trs stores `[s, e15, e25, e35]` and the e14, e24, e34 components are derived.
///
/// For our implementation, let's store translator as a Motor with specific structure.
/// We'll provide Trs as a convenience alias and constructor.
pub const TRS_BASIS: [u32; 4] = [S, E15, E25, E35];

/// Dilator: `[s, e45]`
pub const DIL_BASIS: [u32; 2] = [S, E45];

/// Transversor: `[s, e14, e24, e34]`
pub const TRV_BASIS: [u32; 4] = [S, E14, E24, E34];

/// Boost (Point Pair exponent): `[s, e12, e13, e23, e14, e24, e34, e15, e25, e35, e45]`
pub const BST_BASIS: [u32; 11] = [S, E12, E13, E23, E14, E24, E34, E15, E25, E35, E45];

/// Origin: representation using e4 and e5. In our metric, o = 0.5*(e5 - e4).
/// We represent Ori as a Pnt-like type, but for explicit use we provide a constructor.
pub const ORI_BASIS: [u32; 2] = [E4, E5];

/// Infinity: representation. inf = e4 + e5.
pub const INF_BASIS: [u32; 2] = [E4, E5];

/// Full 32-component multivector (all blades of 5D algebra)
pub const FULL_BASIS: [u32; 32] = [
    // grade 0
    S, // grade 1
    E1, E2, E3, E4, E5, // grade 2
    E12, E13, E23, E14, E24, E34, E15, E25, E35, E45, // grade 3
    E123, E124, E134, E234, E125, E135, E235, E145, E245, E345, // grade 4
    E1234, E1235, E1245, E1345, E2345, // grade 5
    E12345,
];

// ============================================================================
// Type aliases
// ============================================================================

/// Scalar: 1 component.
pub type Sca = Multivector<1>;
/// Euclidean vector: 3 components `[e1, e2, e3]`.
pub type Vec3 = Multivector<3>;
/// Euclidean bivector: 3 components `[e12, e13, e23]`.
pub type Biv = Multivector<3>;
/// Euclidean trivector: 1 component `[e123]`.
pub type Tri = Multivector<1>;
/// Rotor: 4 components `[s, e12, e13, e23]`.
pub type Rot = Multivector<4>;
/// Minkowski blade: 1 component `[e45]`.
pub type Mnk = Multivector<1>;
/// Pseudoscalar: 1 component `[e12345]`.
pub type Pss = Multivector<1>;

/// Point / DualSphere: 5 components `[e1, e2, e3, e4, e5]`.
pub type Pnt = Multivector<5>;
/// DualSphere: alias for Pnt.
pub type Dls = Multivector<5>;
/// Point Pair: 10 components (all grade-2 blades).
pub type Par = Multivector<10>;
/// Circle: 10 components (all grade-3 blades).
pub type Cir = Multivector<10>;
/// Sphere: 5 components (all grade-4 blades).
pub type Sph = Multivector<5>;

/// Flat Point: 4 components `[e15, e25, e35, e45]`.
pub type Flp = Multivector<4>;
/// Dual Line: 6 components `[e12, e13, e23, e15, e25, e35]`.
pub type Dll = Multivector<6>;
/// Line: 6 components `[e145, e245, e345, e125, e135, e235]`.
pub type Lin = Multivector<6>;
/// Dual Plane: 4 components `[e1, e2, e3, e5]`.
pub type Dlp = Multivector<4>;
/// Plane: 4 components `[e1235, e1245, e1345, e2345]`.
pub type Pln = Multivector<4>;

/// Direction vector: 3 components `[e15, e25, e35]`.
pub type Drv = Multivector<3>;
/// Direction bivector: 3 components `[e125, e135, e235]`.
pub type Drb = Multivector<3>;

/// Tangent vector: 5 components (same as Pnt layout).
pub type Tnv = Multivector<5>;
/// Tangent bivector: 5 components.
pub type Tnb = Multivector<5>;

/// Motor: 8 components `[s, e12, e13, e23, e15, e25, e35, e1235]`.
pub type Mot = Multivector<8>;
/// Translator: 4 components `[s, e15, e25, e35]`.
pub type Trs = Multivector<4>;
/// Dilator: 2 components `[s, e45]`.
pub type Dil = Multivector<2>;
/// Transversor: 4 components `[s, e14, e24, e34]`.
pub type Trv = Multivector<4>;
/// Boost: 11 components.
pub type Bst = Multivector<11>;

// ============================================================================
// Constructors
// ============================================================================

/// Create a Euclidean vector.
#[inline]
pub fn vec3(x: f32, y: f32, z: f32) -> Vec3 {
    Multivector::new([x, y, z])
}

/// Create a Euclidean bivector.
#[inline]
pub fn biv(e12: f32, e13: f32, e23: f32) -> Biv {
    Multivector::new([e12, e13, e23])
}

/// Create a scalar.
#[inline]
pub fn sca(s: f32) -> Sca {
    Multivector::new([s])
}

/// Create a conformal point from Euclidean coordinates.
///
/// P = o + x*e1 + y*e2 + z*e3 + 0.5*(x^2+y^2+z^2)*inf
///
/// Where o = 0.5*(e5 - e4), inf = e4 + e5.
/// So: P = x*e1 + y*e2 + z*e3 + (-0.5 + 0.5*r2)*e4 + (0.5 + 0.5*r2)*e5
///       where r2 = x*x + y*y + z*z
#[inline]
pub fn point(x: f32, y: f32, z: f32) -> Pnt {
    let r2 = x * x + y * y + z * z;
    Multivector::new([
        x,               // e1
        y,               // e2
        z,               // e3
        -0.5 + 0.5 * r2, // e4  (origin contrib + inf contrib)
        0.5 + 0.5 * r2,  // e5
    ])
}

/// Origin point: o = 0.5*(e5 - e4) as a Pnt.
#[inline]
pub fn origin() -> Pnt {
    point(0.0, 0.0, 0.0)
}

/// Infinity: inf = e4 + e5 as a Pnt.
#[inline]
pub fn infinity() -> Pnt {
    Multivector::new([0.0, 0.0, 0.0, 1.0, 1.0])
}

/// Create a dual line from direction (a, b, c) and moment (d, e, f).
/// Dll = `[e12, e13, e23, e15, e25, e35]`
#[inline]
pub fn dll(e12: f32, e13: f32, e23: f32, e15: f32, e25: f32, e35: f32) -> Dll {
    Multivector::new([e12, e13, e23, e15, e25, e35])
}

/// Create a motor from components.
#[inline]
pub fn mot(s: f32, e12: f32, e13: f32, e23: f32, e15: f32, e25: f32, e35: f32, e1235: f32) -> Mot {
    Multivector::new([s, e12, e13, e23, e15, e25, e35, e1235])
}

// ============================================================================
// Product tables (lazily initialized)
// ============================================================================

/// Pnt * Pnt geometric product → Par (grade 0+2 result, but we store in Par=grade2)
/// Actually Pnt * Pnt produces scalar + bivector in CGA. Let's use a dedicated result.
/// For CGA, point * point = scalar + all grade-2 = scalar + Par.
/// We'll use a combined basis for the result.

/// Scalar + Par combined basis for Pnt*Pnt result (1 + 10 = 11 components).
const SCA_PAR_BASIS: [u32; 11] = [S, E12, E13, E23, E14, E24, E34, E15, E25, E35, E45];

fn pnt_pnt_gp_table() -> &'static DenseTable<5, 5, 11> {
    static TABLE: OnceLock<DenseTable<5, 5, 11>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts =
            products::gen_geometric_product(&PNT_BASIS, &PNT_BASIS, &SCA_PAR_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Pnt * Pnt inner product → Sca  (for computing distances)
fn pnt_pnt_ip_table() -> &'static DenseTable<5, 5, 1> {
    static TABLE: OnceLock<DenseTable<5, 5, 1>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_inner_product(&PNT_BASIS, &PNT_BASIS, &SCA_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Pnt * Pnt outer product → Par
fn pnt_pnt_op_table() -> &'static DenseTable<5, 5, 10> {
    static TABLE: OnceLock<DenseTable<5, 5, 10>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_outer_product(&PNT_BASIS, &PNT_BASIS, &PAR_BASIS);
        DenseTable::from_instructions(&insts)
    })
}

/// Pnt ^ Pnt ^ Pnt outer product: we compute (Pnt^Pnt)^Pnt = Par ^ Pnt → Cir
fn par_pnt_op_table() -> &'static DenseTable<10, 5, 10> {
    static TABLE: OnceLock<DenseTable<10, 5, 10>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_outer_product(&PAR_BASIS, &PNT_BASIS, &CIR_BASIS);
        DenseTable::from_instructions(&insts)
    })
}

/// Cir ^ Pnt → Sph
fn cir_pnt_op_table() -> &'static DenseTable<10, 5, 5> {
    static TABLE: OnceLock<DenseTable<10, 5, 5>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_outer_product(&CIR_BASIS, &PNT_BASIS, &SPH_BASIS);
        DenseTable::from_instructions(&insts)
    })
}

/// Pnt * Pnt scalar product → scalar (for distance)
fn pnt_pnt_sp_table() -> &'static DenseTable<5, 5, 1> {
    static TABLE: OnceLock<DenseTable<5, 5, 1>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_scalar_product(&PNT_BASIS, &PNT_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Rot * Rot → Rot
fn rot_rot_gp_table() -> &'static DenseTable<4, 4, 4> {
    static TABLE: OnceLock<DenseTable<4, 4, 4>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&ROT_BASIS, &ROT_BASIS, &ROT_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Rot * Pnt → full 32-component intermediate
fn rot_pnt_gp_table() -> &'static DenseTable<4, 5, 32> {
    static TABLE: OnceLock<DenseTable<4, 5, 32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&ROT_BASIS, &PNT_BASIS, &FULL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Full * Rot → Full (for sandwich second half)
fn full_rot_gp_table() -> &'static DenseTable<32, 4, 32> {
    static TABLE: OnceLock<DenseTable<32, 4, 32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&FULL_BASIS, &ROT_BASIS, &FULL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Mot * Pnt → Full (motor acting on point)
fn mot_pnt_gp_table() -> &'static DenseTable<8, 5, 32> {
    static TABLE: OnceLock<DenseTable<8, 5, 32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&MOT_BASIS, &PNT_BASIS, &FULL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Full * Mot → Full (for sandwich second half with motor)
fn full_mot_gp_table() -> &'static DenseTable<32, 8, 32> {
    static TABLE: OnceLock<DenseTable<32, 8, 32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&FULL_BASIS, &MOT_BASIS, &FULL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Mot * Mot → Mot
fn mot_mot_gp_table() -> &'static DenseTable<8, 8, 8> {
    static TABLE: OnceLock<DenseTable<8, 8, 8>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&MOT_BASIS, &MOT_BASIS, &MOT_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Pnt inner product with Pnt → Sca (already defined above as pnt_pnt_ip_table)

/// Dll * Dll geometric product (for motor generation / dual line products)
/// Dll * Dll → Mot  (dual line squared gives motor-like result)
fn dll_dll_gp_table() -> &'static DenseTable<6, 6, 8> {
    static TABLE: OnceLock<DenseTable<6, 6, 8>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&DLL_BASIS, &DLL_BASIS, &MOT_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

// ============================================================================
// Product operations
// ============================================================================

/// Outer product of two points → point pair.
#[inline]
pub fn op_pnt_pnt(a: &Pnt, b: &Pnt) -> Par {
    Multivector::new(pnt_pnt_op_table().execute(&a.data, &b.data))
}

/// Outer product: pair ^ point → circle.
#[inline]
pub fn op_par_pnt(par: &Par, p: &Pnt) -> Cir {
    Multivector::new(par_pnt_op_table().execute(&par.data, &p.data))
}

/// Outer product: circle ^ point → sphere.
#[inline]
pub fn op_cir_pnt(cir: &Cir, p: &Pnt) -> Sph {
    Multivector::new(cir_pnt_op_table().execute(&cir.data, &p.data))
}

/// Inner product of two points → scalar (related to distance).
/// ip(P, Q) = -0.5 * d^2  for normalized points.
#[inline]
pub fn ip_pnt_pnt(a: &Pnt, b: &Pnt) -> f32 {
    pnt_pnt_ip_table().execute(&a.data, &b.data)[0]
}

/// Geometric product of two rotors.
#[inline]
pub fn gp_rot_rot(a: &Rot, b: &Rot) -> Rot {
    Multivector::new(rot_rot_gp_table().execute(&a.data, &b.data))
}

/// Geometric product of two motors.
#[inline]
pub fn gp_mot_mot(a: &Mot, b: &Mot) -> Mot {
    Multivector::new(mot_mot_gp_table().execute(&a.data, &b.data))
}

// ============================================================================
// Unary operations
// ============================================================================

/// Reverse of a rotor: negate the bivector part.
#[inline]
pub fn reverse_rot(r: &Rot) -> Rot {
    Multivector::new([r[0], -r[1], -r[2], -r[3]])
}

/// Reverse of a motor: negate grade-2 part, keep grade-0 and grade-4 parts.
/// Mot = `[s, e12, e13, e23, e15, e25, e35, e1235]`
/// grade 0: s (keep), grade 2: e12..e35 (negate), grade 4: e1235 (keep, since 4*(4-1)/2=6 even)
#[inline]
pub fn reverse_mot(m: &Mot) -> Mot {
    // grade 0: s → keep
    // grade 2: e12, e13, e23, e15, e25, e35 → negate (2*1/2 = 1, odd → negate)
    // grade 4: e1235 → keep (4*3/2 = 6, even → keep)
    Multivector::new([m[0], -m[1], -m[2], -m[3], -m[4], -m[5], -m[6], m[7]])
}

/// Reverse of a point (grade 1 → unchanged).
#[inline]
pub fn reverse_pnt(p: &Pnt) -> Pnt {
    *p
}

/// Conjugate of a point (grade 1 → negate, since involute flips odd grades).
#[inline]
pub fn involute_pnt(p: &Pnt) -> Pnt {
    -*p
}

// ============================================================================
// Sandwich product (spin)
// ============================================================================

/// Extract Pnt components from a full 32-component multivector.
/// Full basis indices: e1=1, e2=2, e3=3, e4=4, e5=5
#[inline]
fn extract_pnt_from_full(full: &[f32; 32]) -> Pnt {
    Multivector::new([full[1], full[2], full[3], full[4], full[5]])
}

/// Apply a rotor to a point: P' = R * P * R~
pub fn spin_rot_pnt(rotor: &Rot, p: &Pnt) -> Pnt {
    let rv = rot_pnt_gp_table().execute(&rotor.data, &p.data);
    let r_rev = reverse_rot(rotor);
    let result = full_rot_gp_table().execute(&rv, &r_rev.data);
    extract_pnt_from_full(&result)
}

/// Apply a motor to a point: P' = M * P * M~
pub fn spin_mot_pnt(motor: &Mot, p: &Pnt) -> Pnt {
    let mv = mot_pnt_gp_table().execute(&motor.data, &p.data);
    let m_rev = reverse_mot(motor);
    let result = full_mot_gp_table().execute(&mv, &m_rev.data);
    extract_pnt_from_full(&result)
}

// ============================================================================
// Round element operations
// ============================================================================

/// Namespace for round element operations (points, circles, spheres).
pub struct Round;

impl Round {
    /// Create a null point (conformal point) at Euclidean position (x, y, z).
    #[inline]
    pub fn null(x: f32, y: f32, z: f32) -> Pnt {
        point(x, y, z)
    }

    /// Create a dual sphere from a center point and radius.
    /// Dls = P - 0.5 * r^2 * ninf
    /// Since ninf = e4 + e5, we subtract 0.5*r^2 from both e4 and e5 components.
    pub fn dls(center: &Pnt, radius: f32) -> Dls {
        let r2 = radius * radius;
        Multivector::new([
            center[0],            // e1
            center[1],            // e2
            center[2],            // e3
            center[3] - 0.5 * r2, // e4
            center[4] - 0.5 * r2, // e5
        ])
    }

    /// Extract the Euclidean center point from a dual sphere / point.
    /// Center = Dls * ninf * Dls  (sandwich of ninf by Dls, then normalize).
    /// But simpler: divide Euclidean components by the weight.
    ///
    /// For a normalized point P = o + p + 0.5*p^2*ni:
    ///   weight w related to the ni component.
    /// In our basis, ni = e4 + e5. The "weight" is (e5 - e4 coeff related).
    ///
    /// Actually: for a CGA point P = `[e1, e2, e3, e4, e5]`:
    ///   ni_inner_P = P`[3]` + P`[4]`  (since ni = e4 + e5, and inner with a vector picks the scalar)
    ///   Wait, that's the outer product. Let me think again.
    ///
    /// For a conformal point P = x*e1 + y*e2 + z*e3 + alpha*e4 + beta*e5:
    ///   where alpha = -0.5 + 0.5*r2, beta = 0.5 + 0.5*r2 for a null point at (x,y,z).
    ///   The "infinity component" is: beta + alpha = r2 (coeff of ninf = e4+e5... no)
    ///
    ///   Actually o = 0.5*(e5 - e4), ni = e4 + e5.
    ///   Decomposing: e4 = 0.5*(ni - 2*o_adj)... Let me just use the direct formula.
    ///
    ///   P = x*e1 + y*e2 + z*e3 + alpha*e4 + beta*e5
    ///   beta - alpha = 0.5 + 0.5*r2 - (-0.5 + 0.5*r2) = 1.0
    ///   beta + alpha = r2
    ///
    ///   The no component = 0.5*(P`[4]` - P`[3]`) = 0.5*(beta - alpha) = 0.5
    ///   The ni component = P`[3]` + P`[4]` = alpha + beta = r2
    ///
    /// For a general (possibly non-normalized) dual sphere:
    ///   weight = no component = 0.5*(P`[4]` - P`[3]`)
    ///   Euclidean coords = (P`[0]`, P`[1]`, P`[2]`) / weight
    ///
    /// For a null point, weight = 0.5, so center = (P`[0]`/0.5, ...) = 2*P`[0..2]`.
    /// But we want to handle non-unit weight, so:
    pub fn center(dls: &Dls) -> Pnt {
        // Weight = coefficient of origin (no) in the point.
        // Since no = 0.5*(e5 - e4), and P = w*no + ..., we have:
        //   e4 coeff = -w/2 + ..., e5 coeff = w/2 + ...
        // So w = e5_coeff - e4_coeff.
        let weight = dls[4] - dls[3];
        if weight.abs() < 1e-10 {
            return *dls; // degenerate
        }
        let inv_w = 1.0 / weight;
        // Return a normalized null point at the center location
        point(dls[0] * inv_w, dls[1] * inv_w, dls[2] * inv_w)
    }

    /// Squared radius of a dual sphere.
    pub fn radius_squared(dls: &Dls) -> f32 {
        let sq =
            dls[0] * dls[0] + dls[1] * dls[1] + dls[2] * dls[2] + dls[3] * dls[3] - dls[4] * dls[4];
        let weight = dls[4] - dls[3];
        let w2 = weight * weight;
        if w2.abs() < 1e-20 {
            return 0.0;
        }
        sq / w2
    }

    /// Extract Euclidean location from a point.
    pub fn location(p: &Pnt) -> (f32, f32, f32) {
        let weight = p[4] - p[3];
        if weight.abs() < 1e-10 {
            return (p[0], p[1], p[2]);
        }
        let inv_w = 1.0 / weight;
        (p[0] * inv_w, p[1] * inv_w, p[2] * inv_w)
    }
}

// ============================================================================
// Flat element operations
// ============================================================================

/// Namespace for flat element operations (lines, planes).
pub struct Flat;

impl Flat {
    /// Direction of a dual line.
    /// Dll = `[e12, e13, e23, e15, e25, e35]`
    /// The "direction" part is the Euclidean bivector: `[e12, e13, e23]`.
    pub fn direction(dll: &Dll) -> Biv {
        Multivector::new([dll[0], dll[1], dll[2]])
    }

    /// Moment of a dual line.
    /// The "moment" part: `[e15, e25, e35]`.
    pub fn moment(dll: &Dll) -> Drv {
        Multivector::new([dll[3], dll[4], dll[5]])
    }
}

// ============================================================================
// Generators (exponential maps)
// ============================================================================

/// Namespace for versor generators.
pub struct Gen;

impl Gen {
    /// Generate a rotor from a Euclidean bivector: R = exp(-B/2).
    /// Same formula as EGA but in the CGA context.
    pub fn rot(bivector: &Biv) -> Rot {
        let angle = bivector.norm();
        if angle < 1e-10 {
            return Multivector::new([1.0, 0.0, 0.0, 0.0]);
        }
        let half = angle / 2.0;
        let cos_half = half.cos();
        let sin_half = -half.sin() / angle;
        Multivector::new([
            cos_half,
            sin_half * bivector[0],
            sin_half * bivector[1],
            sin_half * bivector[2],
        ])
    }

    /// Generate a translator from a Euclidean direction vector.
    ///
    /// T = 1 - (d/2) * ninf
    ///
    /// where d = dx*e1 + dy*e2 + dz*e3, ninf = e4 + e5.
    /// d * ninf = dx*(e14 + e15) + dy*(e24 + e25) + dz*(e34 + e35)
    /// So T = 1 - 0.5*(dx*(e14+e15) + dy*(e24+e25) + dz*(e34+e35))
    ///
    /// This returns a Motor (since a translator is a special motor).
    /// Mot = `[s, e12, e13, e23, e15, e25, e35, e1235]`
    /// The translator only has s and e15, e25, e35 components in the Mot basis...
    /// Wait, it also has e14, e24, e34 components which are NOT in MOT_BASIS.
    ///
    /// This is the key issue: a translator T = 1 - 0.5*d^ni has components in
    /// e14, e24, e34, e15, e25, e35 (6 bivector components). Only e15, e25, e35
    /// are in our MOT_BASIS. The e14, e24, e34 are missing.
    ///
    /// In Versor, the motor basis includes all needed blades. Let me reconsider
    /// the motor basis to include the full set of blades needed.
    ///
    /// Actually, in many CGA implementations, the motor is stored with the FULL
    /// even-grade subalgebra. But for efficiency, Versor uses a specific subset.
    ///
    /// The standard approach: use the "study number" form where a motor in CGA is
    /// M = R + epsilon * T, with R a rotor and T the translation part, both in
    /// the {1, e12, e13, e23} algebra, and epsilon = e1234 or similar.
    ///
    /// For a PURE translator (no rotation), in the null basis:
    ///   T = 1 - (1/2)(t1*e1 + t2*e2 + t3*e3) ^ ni
    ///   = 1 - (1/2)(t1*(e14+e15) + t2*(e24+e25) + t3*(e34+e35))
    ///
    /// Rather than trying to fit this into the compact MOT_BASIS, let's use a
    /// direct approach: build the translator as a Bst (11-component) or just
    /// compute its action directly.
    ///
    /// Practical approach: translate a point directly without forming the versor.
    /// P' = P + t1*e1 + t2*e2 + t3*e3 + (t . p)*ni + 0.5*t^2*ni
    /// where p is the Euclidean part and t . p = t1*x + t2*y + t3*z.
    ///
    /// Actually the sandwich product T*P*~T for translator T = 1 - d*ni/2 gives:
    /// P' = P + d + (d . P_eucl) * ni + 0.5 * d^2 * ni
    /// Wait, let me be more careful. For a conformal point
    /// P = o + p + 0.5*p^2*ni:
    ///
    /// T*P*~T = P + d*ni*(-P_no) = ... This simplifies to:
    /// P' = (p + d)*e_eucl + o + 0.5*(p+d)^2 * ni
    ///    = point(x + dx, y + dy, z + dz)
    ///
    /// So for a pure translation we can just create a new point! But the versor
    /// form is needed for composition with rotations.
    ///
    /// Let me use a full even-grade versor for the motor to handle this properly.
    /// Actually the simplest correct solution: build translator as Bst (which
    /// has all grade-0 and grade-2 blades).
    pub fn trs(dx: f32, dy: f32, dz: f32) -> Bst {
        // T = 1 - 0.5 * (d ^ ni)
        // d ^ ni = dx*(e14 + e15) + dy*(e24 + e25) + dz*(e34 + e35)
        // Bst = [s, e12, e13, e23, e14, e24, e34, e15, e25, e35, e45]
        Multivector::new([
            1.0, // s
            0.0,
            0.0,
            0.0,       // e12, e13, e23
            -0.5 * dx, // e14
            -0.5 * dy, // e24
            -0.5 * dz, // e34
            -0.5 * dx, // e15
            -0.5 * dy, // e25
            -0.5 * dz, // e35
            0.0,       // e45
        ])
    }

    /// Generate a motor from a dual line: M = exp(-Dll/2).
    ///
    /// For a dual line L = d + epsilon*m where d is direction (bivector) and
    /// m is moment (related to position), the motor is:
    ///   M = cos(|d|/2) - sin(|d|/2)*d/|d| - (m . d)/(|d|) * (cos(|d|/2)*I_s/|d| ...)
    ///
    /// Simplified for a line through origin (m perpendicular to d):
    ///   M = cos(a/2) - sin(a/2)*d_hat + epsilon*(sin(a/2)*m/|d| ... )
    ///
    /// For now, implement the pure rotation case (moment=0 means rotation only)
    /// and pure translation case separately, then compose.
    pub fn mot(dll: &Dll) -> Mot {
        // Direction part (Euclidean bivector): determines rotation
        let d0 = dll[0]; // e12
        let d1 = dll[1]; // e13
        let d2 = dll[2]; // e23
        let dir_sq = d0 * d0 + d1 * d1 + d2 * d2;

        // Moment part: determines translation
        let m0 = dll[3]; // e15
        let m1 = dll[4]; // e25
        let m2 = dll[5]; // e35

        if dir_sq < 1e-10 {
            // Pure translation (direction is zero, only moment)
            // Motor = 1 + moment terms
            return Multivector::new([1.0, 0.0, 0.0, 0.0, -m0, -m1, -m2, 0.0]);
        }

        let dir_norm = dir_sq.sqrt();
        let half = dir_norm / 2.0;
        let cos_half = half.cos();
        let sin_half = half.sin();
        let sin_norm = sin_half / dir_norm;

        // d . m (pseudo-scalar part of the product)
        let dm = d0 * m0 + d1 * m1 + d2 * m2;
        let dm_term = -0.5 * dm / dir_sq;

        Multivector::new([
            cos_half,                                            // s
            -sin_norm * d0,                                      // e12
            -sin_norm * d1,                                      // e13
            -sin_norm * d2,                                      // e23
            -sin_norm * m0 + dm_term * sin_half * d0 / dir_norm, // e15
            -sin_norm * m1 + dm_term * sin_half * d1 / dir_norm, // e25
            -sin_norm * m2 + dm_term * sin_half * d2 / dir_norm, // e35
            dm_term * cos_half,                                  // e1235
        ])
    }

    /// Generate a dilator from a flat point and amount.
    /// D = cosh(a/2) + sinh(a/2) * e45 / |e45|
    /// Simplified: D = cosh(a/2) + sinh(a/2) * e45_normalized
    pub fn dil(amount: f32) -> Dil {
        let half = amount / 2.0;
        Multivector::new([half.cosh(), half.sinh()])
    }

    /// Rotor as a motor (embed Rot into Mot).
    pub fn rot_as_mot(r: &Rot) -> Mot {
        Multivector::new([r[0], r[1], r[2], r[3], 0.0, 0.0, 0.0, 0.0])
    }
}

// ============================================================================
// Boost sandwich product (Bst acts on Pnt)
// ============================================================================

/// Bst * Pnt geometric product table → Full
fn bst_pnt_gp_table() -> &'static DenseTable<11, 5, 32> {
    static TABLE: OnceLock<DenseTable<11, 5, 32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&BST_BASIS, &PNT_BASIS, &FULL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Full * Bst geometric product table → Full
fn full_bst_gp_table() -> &'static DenseTable<32, 11, 32> {
    static TABLE: OnceLock<DenseTable<32, 11, 32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&FULL_BASIS, &BST_BASIS, &FULL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Reverse of a Bst.
/// Bst = `[s, e12, e13, e23, e14, e24, e34, e15, e25, e35, e45]`
/// grade 0: s → keep; grade 2: all bivectors → negate
#[inline]
pub fn reverse_bst(b: &Bst) -> Bst {
    Multivector::new([
        b[0], -b[1], -b[2], -b[3], -b[4], -b[5], -b[6], -b[7], -b[8], -b[9], -b[10],
    ])
}

/// Apply a Bst (boost/translator) to a point: P' = B * P * ~B
pub fn spin_bst_pnt(bst: &Bst, p: &Pnt) -> Pnt {
    let bv = bst_pnt_gp_table().execute(&bst.data, &p.data);
    let b_rev = reverse_bst(bst);
    let result = full_bst_gp_table().execute(&bv, &b_rev.data);
    extract_pnt_from_full(&result)
}

/// Translate a point directly (convenience function).
/// Equivalent to applying translator T = Gen::trs(dx, dy, dz) to the point.
pub fn translate(p: &Pnt, dx: f32, dy: f32, dz: f32) -> Pnt {
    let t = Gen::trs(dx, dy, dz);
    spin_bst_pnt(&t, p)
}

// ============================================================================
// Utility: squared distance between two conformal points
// ============================================================================

/// Squared Euclidean distance between two conformal points.
/// d^2 = -2 * (P . Q) for normalized null points.
pub fn distance_sq(a: &Pnt, b: &Pnt) -> f32 {
    -2.0 * ip_pnt_pnt(a, b)
}

/// Euclidean distance between two conformal points.
pub fn distance(a: &Pnt, b: &Pnt) -> f32 {
    let d2 = distance_sq(a, b);
    if d2 < 0.0 {
        0.0
    } else {
        d2.sqrt()
    }
}

// ============================================================================
// Dual / Undual via geometric product with pseudoscalar
// ============================================================================

// The CGA pseudoscalar is I = e12345. Its inverse is I^{-1} = -I in Cl(4,1)
// because I*I = e12345*e12345 = (-1)^{5*4/2} * metric_sign = +1 * (1*1*1*1*(-1)) = -1
// So I^{-1} = -I (since I*(-I) = -I*I = 1).
//
// Dual:   A* = A * I^{-1} = A * (-I)   i.e. right-multiply by -pseudoscalar
// Undual: A = A* * I                     i.e. right-multiply by pseudoscalar
//
// We implement dual/undual using the full 32-component multivector as intermediate.

/// Generic dual: compute A * I^{-1} for any type.
/// We use the full geometric product with the pseudoscalar.
/// For CGA Cl(4,1): I^{-1} = -I, so dual(A) = A * (-e12345).
///
/// The sign: I * I = e12345 * e12345.
/// Reordering: move e5 past e1234 = 4 swaps, move e4 past e123 = 3 swaps,
/// move e3 past e12 = 2 swaps, move e2 past e1 = 1 swap. Total = 10 swaps = even → +1.
/// Metric: shared = e12345, metric product = 1*1*1*1*(-1) = -1.
/// So I*I = -1, and I^{-1} = -I.

/// Pnt (grade 1) → Sph (grade 4) dual
fn pnt_pss_gp_table() -> &'static DenseTable<5, 1, 5> {
    static TABLE: OnceLock<DenseTable<5, 1, 5>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&PNT_BASIS, &PSS_BASIS, &SPH_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Sph (grade 4) → Pnt (grade 1) undual
fn sph_pss_gp_table() -> &'static DenseTable<5, 1, 5> {
    static TABLE: OnceLock<DenseTable<5, 1, 5>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&SPH_BASIS, &PSS_BASIS, &PNT_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Par (grade 2) → Cir (grade 3) dual
fn par_pss_gp_table() -> &'static DenseTable<10, 1, 10> {
    static TABLE: OnceLock<DenseTable<10, 1, 10>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&PAR_BASIS, &PSS_BASIS, &CIR_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Cir (grade 3) → Par (grade 2) dual
fn cir_pss_gp_table() -> &'static DenseTable<10, 1, 10> {
    static TABLE: OnceLock<DenseTable<10, 1, 10>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&CIR_BASIS, &PSS_BASIS, &PAR_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Sph (grade 4) → Pnt (grade 1) dual (sphere → dual sphere)
fn sph_pss_gp_to_pnt_table() -> &'static DenseTable<5, 1, 5> {
    static TABLE: OnceLock<DenseTable<5, 1, 5>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&SPH_BASIS, &PSS_BASIS, &PNT_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Lin (grade 3) → Dll (grade 2) dual
fn lin_pss_gp_table() -> &'static DenseTable<6, 1, 6> {
    static TABLE: OnceLock<DenseTable<6, 1, 6>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&LIN_BASIS, &PSS_BASIS, &DLL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Dll (grade 2) → Lin (grade 3) undual
fn dll_pss_gp_table() -> &'static DenseTable<6, 1, 6> {
    static TABLE: OnceLock<DenseTable<6, 1, 6>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&DLL_BASIS, &PSS_BASIS, &LIN_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Pln (grade 4) → Dlp (grade 1) dual
fn pln_pss_gp_table() -> &'static DenseTable<4, 1, 4> {
    static TABLE: OnceLock<DenseTable<4, 1, 4>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&PLN_BASIS, &PSS_BASIS, &DLP_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Dlp (grade 1) → Pln (grade 4) undual
fn dlp_pss_gp_table() -> &'static DenseTable<4, 1, 4> {
    static TABLE: OnceLock<DenseTable<4, 1, 4>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&DLP_BASIS, &PSS_BASIS, &PLN_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Dual of a Point/DualSphere (grade 1 → grade 4 Sphere).
/// dual(A) = A * I^{-1} = A * (-I)
#[inline]
pub fn dual_pnt(p: &Pnt) -> Sph {
    let neg_pss = [-1.0f32];
    Multivector::new(pnt_pss_gp_table().execute(&p.data, &neg_pss))
}

/// Undual of a Sphere (grade 4 → grade 1 Point/DualSphere).
/// undual(A) = A * I
#[inline]
pub fn undual_sph(s: &Sph) -> Pnt {
    let pss = [1.0f32];
    Multivector::new(sph_pss_gp_table().execute(&s.data, &pss))
}

/// Dual of a Pair (grade 2 → grade 3 Circle).
#[inline]
pub fn dual_par(p: &Par) -> Cir {
    let neg_pss = [-1.0f32];
    Multivector::new(par_pss_gp_table().execute(&p.data, &neg_pss))
}

/// Undual of a Circle (grade 3 → grade 2 Pair).
#[inline]
pub fn undual_cir(c: &Cir) -> Par {
    let pss = [1.0f32];
    Multivector::new(cir_pss_gp_table().execute(&c.data, &pss))
}

/// Dual of a Circle (grade 3 → grade 2 Pair).
#[inline]
pub fn dual_cir(c: &Cir) -> Par {
    let neg_pss = [-1.0f32];
    Multivector::new(cir_pss_gp_table().execute(&c.data, &neg_pss))
}

/// Undual of a Pair (grade 2 → grade 3 Circle).
#[inline]
pub fn undual_par(p: &Par) -> Cir {
    let pss = [1.0f32];
    Multivector::new(par_pss_gp_table().execute(&p.data, &pss))
}

/// Dual of a Sphere (grade 4 → grade 1 DualSphere/Point).
#[inline]
pub fn dual_sph(s: &Sph) -> Dls {
    let neg_pss = [-1.0f32];
    Multivector::new(sph_pss_gp_to_pnt_table().execute(&s.data, &neg_pss))
}

/// Undual of a DualSphere/Point (grade 1 → grade 4 Sphere).
#[inline]
pub fn undual_pnt(p: &Pnt) -> Sph {
    let pss = [1.0f32];
    Multivector::new(pnt_pss_gp_table().execute(&p.data, &pss))
}

/// Dual of a Line (grade 3 → grade 2 DualLine).
#[inline]
pub fn dual_lin(l: &Lin) -> Dll {
    let neg_pss = [-1.0f32];
    Multivector::new(lin_pss_gp_table().execute(&l.data, &neg_pss))
}

/// Undual of a DualLine (grade 2 → grade 3 Line).
#[inline]
pub fn undual_dll(d: &Dll) -> Lin {
    let pss = [1.0f32];
    Multivector::new(dll_pss_gp_table().execute(&d.data, &pss))
}

/// Dual of a Plane (grade 4 → grade 1 DualPlane).
#[inline]
pub fn dual_pln(p: &Pln) -> Dlp {
    let neg_pss = [-1.0f32];
    Multivector::new(pln_pss_gp_table().execute(&p.data, &neg_pss))
}

/// Undual of a DualPlane (grade 1 → grade 4 Plane).
#[inline]
pub fn undual_dlp(d: &Dlp) -> Pln {
    let pss = [1.0f32];
    Multivector::new(dlp_pss_gp_table().execute(&d.data, &pss))
}

// ============================================================================
// Additional product tables for new operations
// ============================================================================

/// Pnt inner product with Par → Full (for split and related ops)
/// Inner product (left contraction): Pnt <= Par → Pnt (grade 2 - 1 = 1)
fn pnt_par_ip_table() -> &'static DenseTable<5, 10, 5> {
    static TABLE: OnceLock<DenseTable<5, 10, 5>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_inner_product(&PNT_BASIS, &PAR_BASIS, &PNT_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Par inner product with Par → Sca (for size computation: (pp <= pp)`[0]`)
fn par_par_ip_table() -> &'static DenseTable<10, 10, 1> {
    static TABLE: OnceLock<DenseTable<10, 10, 1>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_inner_product(&PAR_BASIS, &PAR_BASIS, &SCA_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Par ^ Pnt_inf → Pnt outer product (Par wedge infinity → line-like)
/// Actually: Par ^ Inf → Lin (for carrier of pair)
fn par_pnt_op_to_lin_table() -> &'static DenseTable<10, 5, 6> {
    static TABLE: OnceLock<DenseTable<10, 5, 6>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_outer_product(&PAR_BASIS, &PNT_BASIS, &LIN_BASIS);
        DenseTable::from_instructions(&insts)
    })
}

/// Cir ^ Pnt outer product → Pln (for carrier of circle: circle ^ inf = plane)
fn cir_pnt_op_to_pln_table() -> &'static DenseTable<10, 5, 4> {
    static TABLE: OnceLock<DenseTable<10, 5, 4>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_outer_product(&CIR_BASIS, &PNT_BASIS, &PLN_BASIS);
        DenseTable::from_instructions(&insts)
    })
}

/// Par GP Par → Full (for surround computation: par / (par ^ inf))
fn par_par_gp_table() -> &'static DenseTable<10, 10, 32> {
    static TABLE: OnceLock<DenseTable<10, 10, 32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&PAR_BASIS, &PAR_BASIS, &FULL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Cir GP Cir → Full
fn cir_cir_gp_table() -> &'static DenseTable<10, 10, 32> {
    static TABLE: OnceLock<DenseTable<10, 10, 32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&CIR_BASIS, &CIR_BASIS, &FULL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Pnt IP Cir → Dll (inner product: Pnt <= Cir → grade 3-1=2 = DualLine)
fn pnt_cir_ip_table() -> &'static DenseTable<5, 10, 6> {
    static TABLE: OnceLock<DenseTable<5, 10, 6>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_inner_product(&PNT_BASIS, &CIR_BASIS, &DLL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Pnt IP Sph → Cir-like (inner product: Pnt <= Sph → grade 4-1=3)
fn pnt_sph_ip_table() -> &'static DenseTable<5, 5, 10> {
    static TABLE: OnceLock<DenseTable<5, 5, 10>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_inner_product(&PNT_BASIS, &SPH_BASIS, &CIR_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Pnt IP Lin → Flp-like (inner product: Pnt <= Lin → grade 3-1=2, but specific blades)
/// Actually: Pnt <= Lin might give a Par (grade 2). Let's use Par basis.
fn pnt_lin_ip_table() -> &'static DenseTable<5, 6, 10> {
    static TABLE: OnceLock<DenseTable<5, 6, 10>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_inner_product(&PNT_BASIS, &LIN_BASIS, &PAR_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Pnt IP Pln → Cir-like (inner product: Pnt <= Pln → grade 4-1=3)
fn pnt_pln_ip_table() -> &'static DenseTable<5, 4, 10> {
    static TABLE: OnceLock<DenseTable<5, 4, 10>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_inner_product(&PNT_BASIS, &PLN_BASIS, &CIR_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Lin GP Lin → Full
fn lin_lin_gp_table() -> &'static DenseTable<6, 6, 32> {
    static TABLE: OnceLock<DenseTable<6, 6, 32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&LIN_BASIS, &LIN_BASIS, &FULL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Pln GP Pln → Full
fn pln_pln_gp_table() -> &'static DenseTable<4, 4, 32> {
    static TABLE: OnceLock<DenseTable<4, 4, 32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&PLN_BASIS, &PLN_BASIS, &FULL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

// --- Motor sandwich tables for various types ---

/// Mot * Par → Full
fn mot_par_gp_table() -> &'static DenseTable<8, 10, 32> {
    static TABLE: OnceLock<DenseTable<8, 10, 32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&MOT_BASIS, &PAR_BASIS, &FULL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Mot * Cir → Full
fn mot_cir_gp_table() -> &'static DenseTable<8, 10, 32> {
    static TABLE: OnceLock<DenseTable<8, 10, 32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&MOT_BASIS, &CIR_BASIS, &FULL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Mot * Sph → Full
fn mot_sph_gp_table() -> &'static DenseTable<8, 5, 32> {
    static TABLE: OnceLock<DenseTable<8, 5, 32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&MOT_BASIS, &SPH_BASIS, &FULL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Mot * Lin → Full
fn mot_lin_gp_table() -> &'static DenseTable<8, 6, 32> {
    static TABLE: OnceLock<DenseTable<8, 6, 32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&MOT_BASIS, &LIN_BASIS, &FULL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Mot * Dll → Full
fn mot_dll_gp_table() -> &'static DenseTable<8, 6, 32> {
    static TABLE: OnceLock<DenseTable<8, 6, 32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&MOT_BASIS, &DLL_BASIS, &FULL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Mot * Pln → Full
fn mot_pln_gp_table() -> &'static DenseTable<8, 4, 32> {
    static TABLE: OnceLock<DenseTable<8, 4, 32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&MOT_BASIS, &PLN_BASIS, &FULL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Mot * Dlp → Full
fn mot_dlp_gp_table() -> &'static DenseTable<8, 4, 32> {
    static TABLE: OnceLock<DenseTable<8, 4, 32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&MOT_BASIS, &DLP_BASIS, &FULL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Bst * Par → Full
fn bst_par_gp_table() -> &'static DenseTable<11, 10, 32> {
    static TABLE: OnceLock<DenseTable<11, 10, 32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&BST_BASIS, &PAR_BASIS, &FULL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Bst * Cir → Full
fn bst_cir_gp_table() -> &'static DenseTable<11, 10, 32> {
    static TABLE: OnceLock<DenseTable<11, 10, 32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&BST_BASIS, &CIR_BASIS, &FULL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Bst * Sph → Full
fn bst_sph_gp_table() -> &'static DenseTable<11, 5, 32> {
    static TABLE: OnceLock<DenseTable<11, 5, 32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&BST_BASIS, &SPH_BASIS, &FULL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Bst * Lin → Full
fn bst_lin_gp_table() -> &'static DenseTable<11, 6, 32> {
    static TABLE: OnceLock<DenseTable<11, 6, 32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&BST_BASIS, &LIN_BASIS, &FULL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Bst * Pln → Full
fn bst_pln_gp_table() -> &'static DenseTable<11, 4, 32> {
    static TABLE: OnceLock<DenseTable<11, 4, 32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&BST_BASIS, &PLN_BASIS, &FULL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Dlp GP Pnt → Full (for Flat location: (p <= f) / f needs GP)
fn dlp_dlp_gp_table() -> &'static DenseTable<4, 4, 32> {
    static TABLE: OnceLock<DenseTable<4, 4, 32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&DLP_BASIS, &DLP_BASIS, &FULL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Pnt IP Dlp → Sca+Par basis (left contraction Pnt <= Dlp → scalar)
fn pnt_dlp_ip_table() -> &'static DenseTable<5, 4, 1> {
    static TABLE: OnceLock<DenseTable<5, 4, 1>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_inner_product(&PNT_BASIS, &DLP_BASIS, &SCA_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Pnt OP Dlp → Par (outer product Pnt ^ Dlp)
fn pnt_dlp_op_table() -> &'static DenseTable<5, 4, 10> {
    static TABLE: OnceLock<DenseTable<5, 4, 10>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_outer_product(&PNT_BASIS, &DLP_BASIS, &PAR_BASIS);
        DenseTable::from_instructions(&insts)
    })
}

/// Full GP Pnt → Full (for Flat location computation: ratio needs full gp)
fn par_dlp_gp_table() -> &'static DenseTable<10, 4, 32> {
    static TABLE: OnceLock<DenseTable<10, 4, 32>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&PAR_BASIS, &DLP_BASIS, &FULL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Cir inner product → Sca
fn cir_cir_ip_table() -> &'static DenseTable<10, 10, 1> {
    static TABLE: OnceLock<DenseTable<10, 10, 1>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_inner_product(&CIR_BASIS, &CIR_BASIS, &SCA_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Sph inner product → Sca
fn sph_sph_ip_table() -> &'static DenseTable<5, 5, 1> {
    static TABLE: OnceLock<DenseTable<5, 5, 1>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_inner_product(&SPH_BASIS, &SPH_BASIS, &SCA_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    })
}

/// Drv OP Pnt → Par (direction vector outer product with infinity)
fn drv_pnt_op_table() -> &'static DenseTable<3, 5, 10> {
    static TABLE: OnceLock<DenseTable<3, 5, 10>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let insts = products::gen_outer_product(&DRV_BASIS, &PNT_BASIS, &PAR_BASIS);
        DenseTable::from_instructions(&insts)
    })
}

// ============================================================================
// Extraction helpers for various types from Full multivector
// ============================================================================

/// Extract Par (grade 2) from full 32-component multivector.
/// Full basis: grade-2 blades at indices 6..15 (E12=6, E13=7, E23=8, E14=9, E24=10, E34=11, E15=12, E25=13, E35=14, E45=15)
#[inline]
fn extract_par_from_full(full: &[f32; 32]) -> Par {
    Multivector::new([
        full[6], full[7], full[8], // e12, e13, e23
        full[9], full[10], full[11], // e14, e24, e34
        full[12], full[13], full[14], // e15, e25, e35
        full[15], // e45
    ])
}

/// Extract Cir (grade 3) from full 32-component multivector.
/// Grade-3 blades at indices 16..25
#[inline]
fn extract_cir_from_full(full: &[f32; 32]) -> Cir {
    Multivector::new([
        full[16], full[17], full[18], full[19], // e123, e124, e134, e234
        full[20], full[21], full[22], // e125, e135, e235
        full[23], full[24], full[25], // e145, e245, e345
    ])
}

/// Extract Sph (grade 4) from full 32-component multivector.
/// Grade-4 blades at indices 26..30
#[inline]
fn extract_sph_from_full(full: &[f32; 32]) -> Sph {
    Multivector::new([full[26], full[27], full[28], full[29], full[30]])
}

/// Extract Lin (grade 3, specific blades) from full multivector.
/// Lin = `[E145, E245, E345, E125, E135, E235]`
/// In FULL_BASIS: E125=20, E135=21, E235=22, E145=23, E245=24, E345=25
#[inline]
fn extract_lin_from_full(full: &[f32; 32]) -> Lin {
    Multivector::new([full[23], full[24], full[25], full[20], full[21], full[22]])
}

/// Extract Dll (grade 2, specific blades) from full multivector.
/// Dll = `[E12, E13, E23, E15, E25, E35]`
/// In FULL_BASIS: E12=6, E13=7, E23=8, E15=12, E25=13, E35=14
#[inline]
fn extract_dll_from_full(full: &[f32; 32]) -> Dll {
    Multivector::new([full[6], full[7], full[8], full[12], full[13], full[14]])
}

/// Extract Pln (grade 4) from full multivector.
/// Pln = `[E1235, E1245, E1345, E2345]`
/// In FULL_BASIS: E1235=27, E1245=28, E1345=29, E2345=30
#[inline]
fn extract_pln_from_full(full: &[f32; 32]) -> Pln {
    Multivector::new([full[27], full[28], full[29], full[30]])
}

/// Extract Dlp (grade 1, specific blades) from full multivector.
/// Dlp = `[E1, E2, E3, E5]`
/// In FULL_BASIS: E1=1, E2=2, E3=3, E5=5
#[inline]
fn extract_dlp_from_full(full: &[f32; 32]) -> Dlp {
    Multivector::new([full[1], full[2], full[3], full[5]])
}

/// Extract Drv (direction vector) from full multivector.
/// Drv = `[E15, E25, E35]`
/// In FULL_BASIS: E15=12, E25=13, E35=14
#[inline]
fn extract_drv_from_full(full: &[f32; 32]) -> Drv {
    Multivector::new([full[12], full[13], full[14]])
}

/// Extract Drb (direction bivector) from full multivector.
/// Drb = `[E125, E135, E235]`
/// In FULL_BASIS: E125=20, E135=21, E235=22
#[inline]
fn extract_drb_from_full(full: &[f32; 32]) -> Drb {
    Multivector::new([full[20], full[21], full[22]])
}

// Direction Trivector: [E1235]
// In FULL_BASIS: E1235=27
/// Direction trivector: 1 component `[e1235]`.
pub type Drt = Multivector<1>;
/// Direction trivector basis.
pub const DRT_BASIS: [u32; 1] = [E1235];

// ============================================================================
// Round operations (expanded)
// ============================================================================

impl Round {
    /// Direction of a Pair.
    /// direction(pp) = (inf(-1) <= pp) ^ inf(1)
    /// = (-inf <= pp) ^ inf
    /// where inf = e4 + e5
    pub fn direction_pair(pp: &Par) -> Drv {
        // Step 1: compute -inf <= pp (left contraction of -infinity with pair)
        // -inf = -(e4 + e5) as a Pnt: [0, 0, 0, -1, -1]
        let neg_inf = Multivector::<5>::new([0.0, 0.0, 0.0, -1.0, -1.0]);
        let contracted: Pnt = Multivector::new(pnt_par_ip_table().execute(&neg_inf.data, &pp.data));
        // Step 2: wedge result with inf
        let inf = infinity();
        // Pnt ^ Pnt → Par, then extract the Drv components
        let wedge_result = op_pnt_pnt(&contracted, &inf);
        // Direction vector components are the e15, e25, e35 parts of the pair
        // Par = [e12, e13, e23, e14, e24, e34, e15, e25, e35, e45]
        Multivector::new([wedge_result[6], wedge_result[7], wedge_result[8]])
    }

    /// Direction of a Circle.
    /// direction(c) = (inf(-1) <= c) ^ inf(1)
    pub fn direction_circle(c: &Cir) -> Drb {
        let neg_inf = Multivector::<5>::new([0.0, 0.0, 0.0, -1.0, -1.0]);
        let contracted: Dll = Multivector::new(pnt_cir_ip_table().execute(&neg_inf.data, &c.data));
        // Dll ^ inf → should give us Drb components
        // Dll = [e12, e13, e23, e15, e25, e35]
        // We need dll ^ inf. inf = e4+e5 as a Pnt.
        // Compute using a dedicated approach: dll[e12] ^ inf gives e124+e125, etc.
        // The direction bivector blades are e125, e135, e235.
        // Actually: (e12)^(e4+e5) = e124 + e125; only e125 is in Drb.
        // But direction = (contracted) ^ inf. Let's use the outer product properly.
        // Actually, direction is defined as full outer product. The result's Drb components
        // are the e_ij ^ e5 parts (since e_ij ^ e4 gives different blades).
        //
        // Simpler approach: direction for a circle is the e125, e135, e235 blades of the result.
        // From (Dll ^ Pnt): e12 ^ e5 = e125, e13 ^ e5 = e135, e23 ^ e5 = e235
        // e12 ^ e4 = e124, e13 ^ e4 = e134, e23 ^ e4 = e234 (these are NOT Drb)
        // e15 ^ e4 = e145, e25 ^ e4 = e245, e35 ^ e4 = e345 (NOT Drb)
        // So only: contracted[0]*e5_coeff for e12, etc.
        //
        // Actually the clean formula: the e_ij components of the contracted DualLine,
        // wedged with infinity, give us the direction bivector blades.
        // e12 ^ (e4+e5) = e124 + e125 → e125 blade for Drb
        // So Drb = [contracted[0], contracted[1], contracted[2]] (the euclidean bivector part)
        // (since the outer product with e5 gives e125, e135, e235)
        //
        // Wait, but these include wedge with both e4 and e5.
        // The Drb = [e125, e135, e235] only picks up the e5 wedge.
        // So the coefficient of e125 from e12^inf = e12^(e4+e5) is just the coefficient of e12.
        // Similarly for e135 and e235.
        Multivector::new([contracted[0], contracted[1], contracted[2]])
    }

    /// Direction of a Sphere.
    /// direction(s) = (inf(-1) <= s) ^ inf(1)
    pub fn direction_sphere(s: &Sph) -> Drt {
        let neg_inf = Multivector::<5>::new([0.0, 0.0, 0.0, -1.0, -1.0]);
        let contracted: Cir = Multivector::new(pnt_sph_ip_table().execute(&neg_inf.data, &s.data));
        // Circle contracted with sphere then wedge with inf.
        // The e123 component wedged with e5 gives e1235, which is Drt.
        Multivector::new([contracted[0]]) // e123 component
    }

    /// Carrier flat of a Pair: carrier(pp) = pp ^ infinity
    pub fn carrier_pair(pp: &Par) -> Lin {
        let inf = infinity();
        Multivector::new(par_pnt_op_to_lin_table().execute(&pp.data, &inf.data))
    }

    /// Carrier flat of a Circle: carrier(c) = c ^ infinity
    pub fn carrier_circle(c: &Cir) -> Pln {
        let inf = infinity();
        Multivector::new(cir_pnt_op_to_pln_table().execute(&c.data, &inf.data))
    }

    /// Dual Surround of a Pair: surround(pp) = pp / (pp ^ infinity)
    /// Returns DualSphere. The surround is the smallest sphere containing the round.
    pub fn surround_pair(pp: &Par) -> Dls {
        let carrier = Self::carrier_pair(pp);
        // We need pp / carrier. In GA, A/B = A * B^{-1} = A * reverse(B) / (B * reverse(B)).
        // For a line, B * ~B is a scalar.
        // Let's compute directly: surround = dual of (pp ^ inf)... no.
        // Actually surround(s) = s / (s ^ inf) as per the generic op.
        // s / carrier = s * carrier^{-1}
        // carrier^{-1} = ~carrier / (carrier * ~carrier)
        // For a direct line L, L * ~L is a scalar.
        let carrier_rev = reverse_lin(&carrier);
        let lin_sq = gp_lin_lin_scalar(&carrier, &carrier_rev);
        if lin_sq.abs() < 1e-20 {
            return Multivector::zero();
        }
        let inv_sq = 1.0 / lin_sq;
        // pp * carrier_rev / lin_sq → need Par * Lin → extract Pnt from result
        let pp_cr = gp_par_lin(pp, &carrier_rev);
        // The result should be a grade-1 (Pnt/Dls) element
        let mut result = Multivector::<5>::zero();
        for i in 0..5 {
            result.data[i] = pp_cr[1 + i] * inv_sq; // grade-1 at indices 1..5 in full
        }
        result
    }

    /// Dual Surround of a Circle.
    pub fn surround_circle(c: &Cir) -> Dls {
        let carrier = Self::carrier_circle(c);
        let carrier_rev = reverse_pln(&carrier);
        let pln_sq = gp_pln_pln_scalar(&carrier, &carrier_rev);
        if pln_sq.abs() < 1e-20 {
            return Multivector::zero();
        }
        let inv_sq = 1.0 / pln_sq;
        let c_cr = gp_cir_pln(c, &carrier_rev);
        let mut result = Multivector::<5>::zero();
        for i in 0..5 {
            result.data[i] = c_cr[1 + i] * inv_sq;
        }
        result
    }

    /// Squared size of a Pair (can be negative).
    /// size(pp) = (pp * pp_inv) / (inf_contracted * inf_contracted) * (-1 if dual)
    ///
    /// Simpler: size = (pp <= pp)`[0]`  gives the squared size directly for a normalized pair.
    /// From Versor: r = sqrt(|pp <= pp|), sign determined by positivity.
    pub fn size_pair(pp: &Par) -> f32 {
        par_par_ip_table().execute(&pp.data, &pp.data)[0]
    }

    /// Squared size of a Circle.
    pub fn size_circle(c: &Cir) -> f32 {
        cir_cir_ip_table().execute(&c.data, &c.data)[0]
    }

    /// Squared size of a Sphere.
    pub fn size_sphere(s: &Sph) -> f32 {
        sph_sph_ip_table().execute(&s.data, &s.data)[0]
    }

    /// Split a point pair into two points.
    /// From Versor: given pp, compute
    ///   r = sqrt(|pp <= pp|)
    ///   d = -inf <= pp  (left contraction)
    ///   pa = (pp + r) / d,  pb = (pp - r) / d
    /// Returns (point_a, point_b).
    pub fn split(pp: &Par) -> (Pnt, Pnt) {
        let wt = Self::size_pair(pp);
        let r = wt.abs().sqrt();

        // d = inf(-1) <= pp
        let neg_inf = Multivector::<5>::new([0.0, 0.0, 0.0, -1.0, -1.0]);
        let d: Pnt = Multivector::new(pnt_par_ip_table().execute(&neg_inf.data, &pp.data));

        // bst_a = pp + r (add r to scalar part -- but Par has no scalar!)
        // In Versor, pp is cast to Bst: bstA = pp; then bstA += r (adds to scalar).
        // Bst = [s, e12, e13, e23, e14, e24, e34, e15, e25, e35, e45]
        // Par = [e12, e13, e23, e14, e24, e34, e15, e25, e35, e45]
        let mut bst_a = Multivector::<11>::zero();
        bst_a.data[0] = r; // scalar
        for i in 0..10 {
            bst_a.data[i + 1] = pp.data[i];
        }

        let mut bst_b = Multivector::<11>::zero();
        bst_b.data[0] = -r; // scalar
        for i in 0..10 {
            bst_b.data[i + 1] = pp.data[i];
        }

        // pa = bst_a / d
        // In CGA, (Bst / Pnt) = Bst * Pnt^{-1} = Bst * reverse(Pnt) / (Pnt * reverse(Pnt))
        // For a null point, Pnt * ~Pnt = 0... so we need a different approach.
        // Actually from the C++ code: pair[0] = ( ( bstA ) / d ).cast<Pnt>();
        // The division A/B in Versor is A * B.inv() = A * ~B / (B * ~B)
        // For a vector d (grade 1), ~d = d (reverse of grade 1 is identity).
        // d * d = d . d (scalar part) in CGA metric.
        // Metric is [1,1,1,1,-1]. So d*d = d1^2 + d2^2 + d3^2 + d4^2 - d5^2.
        let d_sq_metric = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + d[3] * d[3] - d[4] * d[4];

        if d_sq_metric.abs() < 1e-20 {
            // Degenerate case
            return (point(0.0, 0.0, 0.0), point(0.0, 0.0, 0.0));
        }
        let inv_d_sq = 1.0 / d_sq_metric;

        // bst_a * d (GP) then divide by d_sq to get bst_a / d
        let result_a = gp_bst_pnt(&bst_a, &d);
        let result_b = gp_bst_pnt(&bst_b, &d);

        let pa = Multivector::<5>::new([
            result_a[1] * inv_d_sq,
            result_a[2] * inv_d_sq,
            result_a[3] * inv_d_sq,
            result_a[4] * inv_d_sq,
            result_a[5] * inv_d_sq,
        ]);
        let pb = Multivector::<5>::new([
            result_b[1] * inv_d_sq,
            result_b[2] * inv_d_sq,
            result_b[3] * inv_d_sq,
            result_b[4] * inv_d_sq,
            result_b[5] * inv_d_sq,
        ]);

        (pa, pb)
    }
}

// Helper GP functions for surround computation

/// Reverse of a Line (grade 3: sign = (-1)^{3*2/2} = (-1)^3 = -1).
#[inline]
pub fn reverse_lin(l: &Lin) -> Lin {
    Multivector::new([-l[0], -l[1], -l[2], -l[3], -l[4], -l[5]])
}

/// Reverse of a Plane (grade 4: sign = (-1)^{4*3/2} = (-1)^6 = +1).
#[inline]
pub fn reverse_pln(p: &Pln) -> Pln {
    *p // grade 4, reverse sign = +1
}

/// Reverse of a Pair (grade 2: sign = (-1)^{2*1/2} = -1).
#[inline]
pub fn reverse_par(p: &Par) -> Par {
    -*p
}

/// Reverse of a Circle (grade 3: sign = -1).
#[inline]
pub fn reverse_cir(c: &Cir) -> Cir {
    -*c
}

/// Reverse of a Sphere (grade 4: sign = +1).
#[inline]
pub fn reverse_sph(s: &Sph) -> Sph {
    *s
}

/// GP of Lin * Lin → scalar part only
fn gp_lin_lin_scalar(a: &Lin, b: &Lin) -> f32 {
    let full = lin_lin_gp_table().execute(&a.data, &b.data);
    full[0] // scalar
}

/// GP of Pln * Pln → scalar part only
fn gp_pln_pln_scalar(a: &Pln, b: &Pln) -> f32 {
    let full = pln_pln_gp_table().execute(&a.data, &b.data);
    full[0]
}

/// GP of Par * Lin → Full
fn gp_par_lin(pp: &Par, l: &Lin) -> [f32; 32] {
    // Par * Lin → need a table. Let's create it inline.
    static TABLE: OnceLock<DenseTable<10, 6, 32>> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&PAR_BASIS, &LIN_BASIS, &FULL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    });
    table.execute(&pp.data, &l.data)
}

/// GP of Cir * Pln → Full
fn gp_cir_pln(c: &Cir, p: &Pln) -> [f32; 32] {
    static TABLE: OnceLock<DenseTable<10, 4, 32>> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&CIR_BASIS, &PLN_BASIS, &FULL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    });
    table.execute(&c.data, &p.data)
}

/// GP of Bst * Pnt → Full (reuse of bst_pnt_gp_table but returning raw array)
fn gp_bst_pnt(b: &Bst, p: &Pnt) -> [f32; 32] {
    bst_pnt_gp_table().execute(&b.data, &p.data)
}

// ============================================================================
// Flat operations (expanded)
// ============================================================================

impl Flat {
    /// Location on a dual plane closest to a given point.
    /// From Versor: location(f, p, dual) = dual ? (p ^ f) / f : (p <= f) / f
    /// For a DualPlane (already dual), we use: (p ^ dlp) / dlp
    pub fn location_dlp(dlp: &Dlp, p: &Pnt) -> Pnt {
        // Compute p ^ dlp → Par
        let wedge: Par = Multivector::new(pnt_dlp_op_table().execute(&p.data, &dlp.data));
        // Compute wedge / dlp = wedge * dlp^{-1}
        // dlp^{-1} = dlp / (dlp * dlp) since dlp is grade 1
        let dlp_sq = dlp[0] * dlp[0] + dlp[1] * dlp[1] + dlp[2] * dlp[2] - dlp[3] * dlp[3];
        if dlp_sq.abs() < 1e-20 {
            return point(0.0, 0.0, 0.0);
        }
        let inv = 1.0 / dlp_sq;
        // wedge * dlp → Full, then extract Pnt, then multiply by inv
        let full = par_dlp_gp_table().execute(&wedge.data, &dlp.data);
        Multivector::new([
            full[1] * inv,
            full[2] * inv,
            full[3] * inv,
            full[4] * inv,
            full[5] * inv,
        ])
    }

    /// Direction of a Line (direct flat).
    /// For a direct line Lin = `[e145, e245, e345, e125, e135, e235]`,
    /// the direction is extracted from the e_i^e4^e5 components (indices 0,1,2),
    /// representing the e1, e2, e3 Euclidean directions.
    pub fn direction_lin(l: &Lin) -> Vec3 {
        // LIN_BASIS = [E145, E245, E345, E125, E135, E235]
        // The e_i45 components encode the direction of the line.
        Multivector::new([l[0], l[1], l[2]])
    }

    /// Direction of a Plane (direct flat).
    /// direction(pln) = -inf <= pln
    pub fn direction_pln(pln: &Pln) -> Drt {
        let neg_inf = Multivector::<5>::new([0.0, 0.0, 0.0, -1.0, -1.0]);
        // Pnt <= Pln → Cir (grade 4-1=3)
        let contracted: Cir =
            Multivector::new(pnt_pln_ip_table().execute(&neg_inf.data, &pln.data));
        // The direction trivector = e123 component
        Multivector::new([contracted[0]])
    }

    /// Location on a direct Line closest to origin.
    /// For a direct line L, location = (origin <= L) / L ... but this is complex.
    /// Simpler: dual the line, then use dual plane location approach.
    /// Actually: location(L, p, false) = (p <= L) / L
    pub fn location_lin(l: &Lin, p: &Pnt) -> Pnt {
        // p <= L → Par (grade 2)
        let contracted: Par = Multivector::new(pnt_lin_ip_table().execute(&p.data, &l.data));
        // contracted / L = contracted * L^{-1} = contracted * ~L / (L * ~L)
        let l_rev = reverse_lin(l);
        let l_sq = gp_lin_lin_scalar(l, &l_rev);
        if l_sq.abs() < 1e-20 {
            return point(0.0, 0.0, 0.0);
        }
        let inv = 1.0 / l_sq;
        let full = gp_par_lin(&contracted, &l_rev);
        Multivector::new([
            full[1] * inv,
            full[2] * inv,
            full[3] * inv,
            full[4] * inv,
            full[5] * inv,
        ])
    }
}

// ============================================================================
// Versor inverse
// ============================================================================

/// Inverse of a Rotor: R^{-1} = ~R / (R * ~R).
/// For a unit rotor, R^{-1} = ~R.
#[inline]
pub fn inverse_rot(r: &Rot) -> Rot {
    let rev = reverse_rot(r);
    let sq = gp_rot_rot(r, &rev);
    let norm = sq[0]; // scalar part of R * ~R
    if norm.abs() < 1e-20 {
        return Multivector::zero();
    }
    let inv = 1.0 / norm;
    Multivector::new([rev[0] * inv, rev[1] * inv, rev[2] * inv, rev[3] * inv])
}

/// Inverse of a Motor: M^{-1} = ~M / (M * ~M).
#[inline]
pub fn inverse_mot(m: &Mot) -> Mot {
    let rev = reverse_mot(m);
    let sq = gp_mot_mot(m, &rev);
    // M * ~M for a CGA motor yields s + p*e1235 (scalar + grade-4 part).
    // In Cl(4,1), e1235² = e1²*e2²*e3²*e5² * sign = 1*1*1*(-1) * (-1)^6 = -1.
    // So (s + p*j) with j²=-1. Inverse: (s - p*j) / (s² + p²).
    let s = sq[0];
    let p = sq[7]; // e1235 coefficient
    let denom = s * s + p * p;
    if denom.abs() < 1e-20 {
        return Multivector::zero();
    }
    let inv_d = 1.0 / denom;
    // M^{-1} = ~M * (s - p*e1235) / (s² + p²)
    let conj = Multivector::<8>::new([s, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -p]);
    let result = gp_mot_mot(&rev, &conj);
    Multivector::new([
        result[0] * inv_d,
        result[1] * inv_d,
        result[2] * inv_d,
        result[3] * inv_d,
        result[4] * inv_d,
        result[5] * inv_d,
        result[6] * inv_d,
        result[7] * inv_d,
    ])
}

/// Inverse of a Bst (translator/boost).
#[inline]
pub fn inverse_bst(b: &Bst) -> Bst {
    let rev = reverse_bst(b);
    // b * ~b → we need Bst * Bst GP. Use full computation.
    static TABLE: OnceLock<DenseTable<11, 11, 32>> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let insts = products::gen_geometric_product(&BST_BASIS, &BST_BASIS, &FULL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    });
    let sq = table.execute(&b.data, &rev.data);
    let norm = sq[0]; // scalar
    if norm.abs() < 1e-20 {
        return Multivector::zero();
    }
    let inv = 1.0 / norm;
    let mut result = rev;
    for i in 0..11 {
        result.data[i] *= inv;
    }
    result
}

// ============================================================================
// Reflection
// ============================================================================

/// Reflect a point through a dual plane: P' = dlp * P * dlp
/// (Note: reflection is NOT a sandwich with reverse, it's A * X * A for vectors)
pub fn reflect_pnt_dlp(p: &Pnt, dlp: &Dlp) -> Pnt {
    // Reflection: P' = -dlp * P * dlp^{-1}
    // For a unit dual plane (dlp * dlp = +-1), this simplifies to: -dlp * P * dlp
    // In CGA, reflection through a plane (grade-1 versor) is: P' = -v * P * v^{-1}
    // where v is the reflecting vector/plane.

    // dlp is grade 1 in CGA: [e1, e2, e3, e5]
    // We need dlp * P * dlp^{-1} (with sign flip for odd-grade versor)
    // dlp^{-1} = dlp / (dlp*dlp)
    let dlp_sq = dlp[0] * dlp[0] + dlp[1] * dlp[1] + dlp[2] * dlp[2] - dlp[3] * dlp[3];
    if dlp_sq.abs() < 1e-20 {
        return *p;
    }
    let inv = 1.0 / dlp_sq;

    // Compute dlp * P (using DLP as a specialized Pnt-like type)
    // DLP = [e1, e2, e3, e5], which is a sub-type of Pnt [e1, e2, e3, e4, e5] with e4=0
    let dlp_as_pnt = Multivector::<5>::new([dlp[0], dlp[1], dlp[2], 0.0, dlp[3]]);

    // dlp * P → full (Pnt * Pnt → SCA_PAR = 11 components)
    let first = pnt_pnt_gp_table().execute(&dlp_as_pnt.data, &p.data);
    // Now we have a scalar+bivector result (11 components: [s, e12..e45])
    // We need to multiply this by dlp again to get a vector (grade 1) result.
    // (scalar+bivector) * vector → vector + trivector
    // Embed the 11-component result into full 32, then multiply by dlp full
    let mut full_first = [0.0f32; 32];
    full_first[0] = first[0]; // scalar
                              // Par components map to full indices 6..15
    for i in 0..10 {
        full_first[6 + i] = first[1 + i];
    }

    // Now full_first * dlp_as_pnt → full, extract Pnt
    static FULL_PNT_GP: OnceLock<DenseTable<32, 5, 32>> = OnceLock::new();
    let table = FULL_PNT_GP.get_or_init(|| {
        let insts = products::gen_geometric_product(&FULL_BASIS, &PNT_BASIS, &FULL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    });
    let result = table.execute(&full_first, &dlp_as_pnt.data);

    // Extract Pnt, negate (odd-grade versor), and divide by dlp_sq
    let neg_inv = -inv;
    Multivector::new([
        result[1] * neg_inv,
        result[2] * neg_inv,
        result[3] * neg_inv,
        result[4] * neg_inv,
        result[5] * neg_inv,
    ])
}

/// Reflect a point through a dual sphere: P' = -S * P * S^{-1}
/// where S is the dual sphere (grade-1 versor in CGA).
pub fn reflect_pnt_dls(p: &Pnt, s: &Dls) -> Pnt {
    // Dual sphere is grade 1 in CGA, same formula as plane reflection
    let s_sq = s[0] * s[0] + s[1] * s[1] + s[2] * s[2] + s[3] * s[3] - s[4] * s[4];
    if s_sq.abs() < 1e-20 {
        return *p;
    }
    let inv = 1.0 / s_sq;

    let first = pnt_pnt_gp_table().execute(&s.data, &p.data);
    let mut full_first = [0.0f32; 32];
    full_first[0] = first[0];
    for i in 0..10 {
        full_first[6 + i] = first[1 + i];
    }

    static FULL_PNT_GP2: OnceLock<DenseTable<32, 5, 32>> = OnceLock::new();
    let table = FULL_PNT_GP2.get_or_init(|| {
        let insts = products::gen_geometric_product(&FULL_BASIS, &PNT_BASIS, &FULL_BASIS, &METRIC);
        DenseTable::from_instructions(&insts)
    });
    let result = table.execute(&full_first, &s.data);

    let neg_inv = -inv;
    Multivector::new([
        result[1] * neg_inv,
        result[2] * neg_inv,
        result[3] * neg_inv,
        result[4] * neg_inv,
        result[5] * neg_inv,
    ])
}

// ============================================================================
// Full motor sandwich operations for all types
// ============================================================================

/// Apply a motor to a Pair: Par' = M * Par * ~M
pub fn spin_mot_par(motor: &Mot, pp: &Par) -> Par {
    let mv = mot_par_gp_table().execute(&motor.data, &pp.data);
    let m_rev = reverse_mot(motor);
    let result = full_mot_gp_table().execute(&mv, &m_rev.data);
    extract_par_from_full(&result)
}

/// Apply a motor to a Circle: Cir' = M * Cir * ~M
pub fn spin_mot_cir(motor: &Mot, c: &Cir) -> Cir {
    let mv = mot_cir_gp_table().execute(&motor.data, &c.data);
    let m_rev = reverse_mot(motor);
    let result = full_mot_gp_table().execute(&mv, &m_rev.data);
    extract_cir_from_full(&result)
}

/// Apply a motor to a Sphere: Sph' = M * Sph * ~M
pub fn spin_mot_sph(motor: &Mot, s: &Sph) -> Sph {
    let mv = mot_sph_gp_table().execute(&motor.data, &s.data);
    let m_rev = reverse_mot(motor);
    let result = full_mot_gp_table().execute(&mv, &m_rev.data);
    extract_sph_from_full(&result)
}

/// Apply a motor to a Line: Lin' = M * Lin * ~M
pub fn spin_mot_lin(motor: &Mot, l: &Lin) -> Lin {
    let mv = mot_lin_gp_table().execute(&motor.data, &l.data);
    let m_rev = reverse_mot(motor);
    let result = full_mot_gp_table().execute(&mv, &m_rev.data);
    extract_lin_from_full(&result)
}

/// Apply a motor to a DualLine: Dll' = M * Dll * ~M
pub fn spin_mot_dll(motor: &Mot, d: &Dll) -> Dll {
    let mv = mot_dll_gp_table().execute(&motor.data, &d.data);
    let m_rev = reverse_mot(motor);
    let result = full_mot_gp_table().execute(&mv, &m_rev.data);
    extract_dll_from_full(&result)
}

/// Apply a motor to a Plane: Pln' = M * Pln * ~M
pub fn spin_mot_pln(motor: &Mot, p: &Pln) -> Pln {
    let mv = mot_pln_gp_table().execute(&motor.data, &p.data);
    let m_rev = reverse_mot(motor);
    let result = full_mot_gp_table().execute(&mv, &m_rev.data);
    extract_pln_from_full(&result)
}

/// Apply a motor to a DualPlane: Dlp' = M * Dlp * ~M
pub fn spin_mot_dlp(motor: &Mot, d: &Dlp) -> Dlp {
    let mv = mot_dlp_gp_table().execute(&motor.data, &d.data);
    let m_rev = reverse_mot(motor);
    let result = full_mot_gp_table().execute(&mv, &m_rev.data);
    extract_dlp_from_full(&result)
}

/// Apply a Bst (translator) to a Pair.
pub fn spin_bst_par(bst: &Bst, pp: &Par) -> Par {
    let bv = bst_par_gp_table().execute(&bst.data, &pp.data);
    let b_rev = reverse_bst(bst);
    let result = full_bst_gp_table().execute(&bv, &b_rev.data);
    extract_par_from_full(&result)
}

/// Apply a Bst (translator) to a Circle.
pub fn spin_bst_cir(bst: &Bst, c: &Cir) -> Cir {
    let bv = bst_cir_gp_table().execute(&bst.data, &c.data);
    let b_rev = reverse_bst(bst);
    let result = full_bst_gp_table().execute(&bv, &b_rev.data);
    extract_cir_from_full(&result)
}

/// Apply a Bst (translator) to a Sphere.
pub fn spin_bst_sph(bst: &Bst, s: &Sph) -> Sph {
    let bv = bst_sph_gp_table().execute(&bst.data, &s.data);
    let b_rev = reverse_bst(bst);
    let result = full_bst_gp_table().execute(&bv, &b_rev.data);
    extract_sph_from_full(&result)
}

/// Apply a Bst (translator) to a Line.
pub fn spin_bst_lin(bst: &Bst, l: &Lin) -> Lin {
    let bv = bst_lin_gp_table().execute(&bst.data, &l.data);
    let b_rev = reverse_bst(bst);
    let result = full_bst_gp_table().execute(&bv, &b_rev.data);
    extract_lin_from_full(&result)
}

/// Apply a Bst (translator) to a Plane.
pub fn spin_bst_pln(bst: &Bst, p: &Pln) -> Pln {
    let bv = bst_pln_gp_table().execute(&bst.data, &p.data);
    let b_rev = reverse_bst(bst);
    let result = full_bst_gp_table().execute(&bv, &b_rev.data);
    extract_pln_from_full(&result)
}

/// Translate a pair by (dx, dy, dz).
pub fn translate_par(pp: &Par, dx: f32, dy: f32, dz: f32) -> Par {
    let t = Gen::trs(dx, dy, dz);
    spin_bst_par(&t, pp)
}

/// Translate a circle by (dx, dy, dz).
pub fn translate_cir(c: &Cir, dx: f32, dy: f32, dz: f32) -> Cir {
    let t = Gen::trs(dx, dy, dz);
    spin_bst_cir(&t, c)
}

/// Translate a sphere by (dx, dy, dz).
pub fn translate_sph(s: &Sph, dx: f32, dy: f32, dz: f32) -> Sph {
    let t = Gen::trs(dx, dy, dz);
    spin_bst_sph(&t, s)
}

/// Translate a line by (dx, dy, dz).
pub fn translate_lin(l: &Lin, dx: f32, dy: f32, dz: f32) -> Lin {
    let t = Gen::trs(dx, dy, dz);
    spin_bst_lin(&t, l)
}

/// Translate a plane by (dx, dy, dz).
pub fn translate_pln(p: &Pln, dx: f32, dy: f32, dz: f32) -> Pln {
    let t = Gen::trs(dx, dy, dz);
    spin_bst_pln(&t, p)
}

// ============================================================================
// Meet and Join
// ============================================================================

/// Meet (intersection) of two geometric objects using the regressive product.
/// meet(A, B) = (A* ^ B*)* where * is the CGA dual.
///
/// We implement specific versions for common type combinations.

/// Outer product Sph ^ Sph → Pss (grade 4+4=8, but clamped to grade 5 in 5D)
/// Actually Sph ^ Sph → grade 8 which doesn't exist in 5D, so it's 0.
/// Meet of two spheres = dual of (dual(sph_a) ^ dual(sph_b))
/// dual(Sph) = Dls (grade 1). So dual(A) ^ dual(B) = Dls ^ Dls = Par (grade 2).
/// Then undual of Par = Cir (grade 3). So meet of two spheres is a Circle!

/// Meet of two Spheres → Circle (their intersection).
/// meet(A, B) = undual(dual(A) ^ dual(B))
pub fn meet_sph_sph(a: &Sph, b: &Sph) -> Cir {
    let da = dual_sph(a);
    let db = dual_sph(b);
    let wedge = op_pnt_pnt(&da, &db);
    undual_par(&wedge)
}

/// Meet of a Sphere and a Plane → Circle.
/// dual(Sph) = Dls (grade 1), dual(Pln) = Dlp (grade 1).
/// Dls ^ Dlp → Par (grade 2). Undual(Par) → Cir.
pub fn meet_sph_pln(s: &Sph, p: &Pln) -> Cir {
    let ds = dual_sph(s);
    let dp = dual_pln(p);
    // Dls ^ Dlp: both are grade-1, result is grade-2
    // Dls = [e1,e2,e3,e4,e5], Dlp = [e1,e2,e3,e5]
    // Use Pnt ^ Dlp → Par
    let wedge: Par = Multivector::new(pnt_dlp_op_table().execute(&ds.data, &dp.data));
    undual_par(&wedge)
}

/// Meet of two Planes → Line.
/// dual(Pln) = Dlp (grade 1). Dlp ^ Dlp → grade 2 (Par-like).
/// Undual(grade 2) → grade 3 (Lin/Cir).
pub fn meet_pln_pln(a: &Pln, b: &Pln) -> Lin {
    let da = dual_pln(a);
    let db = dual_pln(b);
    // Dlp is [e1,e2,e3,e5] which is a subtype of Pnt [e1,e2,e3,e4,e5]
    let da_pnt = Multivector::<5>::new([da[0], da[1], da[2], 0.0, da[3]]);
    let db_pnt = Multivector::<5>::new([db[0], db[1], db[2], 0.0, db[3]]);
    let wedge = op_pnt_pnt(&da_pnt, &db_pnt);
    // Undual Par → Cir, then extract Lin blades
    let result_cir = undual_par(&wedge);
    // Lin = [E145, E245, E345, E125, E135, E235]
    // Cir = [E123, E124, E134, E234, E125, E135, E235, E145, E245, E345]
    Multivector::new([
        result_cir[7],
        result_cir[8],
        result_cir[9], // E145, E245, E345
        result_cir[4],
        result_cir[5],
        result_cir[6], // E125, E135, E235
    ])
}

/// Join of two points → Pair (their span).
/// join(A, B) = A ^ B (outer product).
#[inline]
pub fn join_pnt_pnt(a: &Pnt, b: &Pnt) -> Par {
    op_pnt_pnt(a, b)
}

/// Join of a Pair and a Point → Circle.
#[inline]
pub fn join_par_pnt(pp: &Par, p: &Pnt) -> Cir {
    op_par_pnt(pp, p)
}

/// Join of a Circle and a Point → Sphere.
#[inline]
pub fn join_cir_pnt(c: &Cir, p: &Pnt) -> Sph {
    op_cir_pnt(c, p)
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

    #[test]
    fn test_point_creation() {
        let p = point(1.0, 2.0, 3.0);
        // e1 = 1, e2 = 2, e3 = 3
        assert_eq!(p[0], 1.0);
        assert_eq!(p[1], 2.0);
        assert_eq!(p[2], 3.0);
        // e4 = -0.5 + 0.5*14 = 6.5
        assert!(approx_eq(p[3], 6.5));
        // e5 = 0.5 + 0.5*14 = 7.5
        assert!(approx_eq(p[4], 7.5));
    }

    #[test]
    fn test_origin() {
        let o = origin();
        assert_eq!(o[0], 0.0);
        assert_eq!(o[1], 0.0);
        assert_eq!(o[2], 0.0);
        // e4 = -0.5, e5 = 0.5
        assert!(approx_eq(o[3], -0.5));
        assert!(approx_eq(o[4], 0.5));
    }

    #[test]
    fn test_null_point_is_null() {
        // A null point should satisfy P . P = 0 (in CGA metric)
        let p = point(3.0, 4.0, 5.0);
        let sq = p[0] * p[0] + p[1] * p[1] + p[2] * p[2] + p[3] * p[3] - p[4] * p[4];
        assert!(approx_eq(sq, 0.0), "Point should be null, got sq={}", sq);
    }

    #[test]
    fn test_location_extraction() {
        let p = point(3.0, -1.5, 7.0);
        let (x, y, z) = Round::location(&p);
        assert!(approx_eq(x, 3.0));
        assert!(approx_eq(y, -1.5));
        assert!(approx_eq(z, 7.0));
    }

    #[test]
    fn test_inner_product_distance() {
        let a = point(0.0, 0.0, 0.0);
        let b = point(3.0, 4.0, 0.0);
        let d2 = distance_sq(&a, &b);
        assert!(approx_eq(d2, 25.0), "Expected 25, got {}", d2);
    }

    #[test]
    fn test_distance() {
        let a = point(1.0, 0.0, 0.0);
        let b = point(4.0, 0.0, 0.0);
        let d = distance(&a, &b);
        assert!(approx_eq(d, 3.0), "Expected 3, got {}", d);
    }

    #[test]
    fn test_outer_product_pair() {
        let a = point(0.0, 0.0, 0.0);
        let b = point(1.0, 0.0, 0.0);
        let pair = op_pnt_pnt(&a, &b);
        // Should be non-zero (a point pair)
        assert!(pair.norm() > EPS, "Pair should be non-zero");
    }

    #[test]
    fn test_outer_product_circle() {
        let a = point(1.0, 0.0, 0.0);
        let b = point(0.0, 1.0, 0.0);
        let c = point(0.0, 0.0, 1.0);
        let pair = op_pnt_pnt(&a, &b);
        let circle = op_par_pnt(&pair, &c);
        assert!(circle.norm() > EPS, "Circle should be non-zero");
    }

    #[test]
    fn test_outer_product_sphere() {
        let a = point(1.0, 0.0, 0.0);
        let b = point(0.0, 1.0, 0.0);
        let c = point(0.0, 0.0, 1.0);
        let d = point(-1.0, 0.0, 0.0);
        let pair = op_pnt_pnt(&a, &b);
        let circle = op_par_pnt(&pair, &c);
        let sphere = op_cir_pnt(&circle, &d);
        assert!(sphere.norm() > EPS, "Sphere should be non-zero");
    }

    #[test]
    fn test_dual_sphere_radius() {
        let center = point(0.0, 0.0, 0.0);
        let dls = Round::dls(&center, 3.0);
        let r2 = Round::radius_squared(&dls);
        assert!(approx_eq(r2, 9.0), "Expected r^2=9, got {}", r2);
    }

    #[test]
    fn test_dual_sphere_center() {
        let center = point(2.0, 3.0, 4.0);
        let dls = Round::dls(&center, 5.0);
        let c = Round::center(&dls);
        let (x, y, z) = Round::location(&c);
        assert!(approx_eq(x, 2.0), "x={}", x);
        assert!(approx_eq(y, 3.0), "y={}", y);
        assert!(approx_eq(z, 4.0), "z={}", z);
    }

    #[test]
    fn test_rotor_rotation_point() {
        // Rotate point (1, 0, 0) by PI/2 in e12 plane → (0, 1, 0)
        let b = biv(PI / 2.0, 0.0, 0.0);
        let r = Gen::rot(&b);
        let p = point(1.0, 0.0, 0.0);
        let rotated = spin_rot_pnt(&r, &p);
        let (x, y, z) = Round::location(&rotated);
        assert!(approx_eq(x, 0.0), "x={}", x);
        assert!(approx_eq(y, 1.0), "y={}", y);
        assert!(approx_eq(z, 0.0), "z={}", z);
    }

    #[test]
    fn test_rotor_preserves_null() {
        // After rotation, point should still be null
        let b = biv(1.0, 0.5, 0.3);
        let r = Gen::rot(&b);
        let p = point(1.0, 2.0, 3.0);
        let rotated = spin_rot_pnt(&r, &p);
        let sq = rotated[0] * rotated[0]
            + rotated[1] * rotated[1]
            + rotated[2] * rotated[2]
            + rotated[3] * rotated[3]
            - rotated[4] * rotated[4];
        assert!(
            approx_eq(sq, 0.0),
            "Rotated point should be null, got sq={}",
            sq
        );
    }

    #[test]
    fn test_rotor_preserves_distance() {
        let b = biv(0.7, -0.3, 1.2);
        let r = Gen::rot(&b);
        let p = point(1.0, 2.0, 3.0);
        let q = point(4.0, 5.0, 6.0);
        let d_before = distance(&p, &q);
        let p2 = spin_rot_pnt(&r, &p);
        let q2 = spin_rot_pnt(&r, &q);
        let d_after = distance(&p2, &q2);
        assert!(
            approx_eq(d_before, d_after),
            "Distance should be preserved: {} vs {}",
            d_before,
            d_after
        );
    }

    #[test]
    fn test_translation() {
        let p = point(1.0, 2.0, 3.0);
        let translated = translate(&p, 10.0, 20.0, 30.0);
        let (x, y, z) = Round::location(&translated);
        assert!(approx_eq(x, 11.0), "x={}", x);
        assert!(approx_eq(y, 22.0), "y={}", y);
        assert!(approx_eq(z, 33.0), "z={}", z);
    }

    #[test]
    fn test_translation_preserves_null() {
        let p = point(1.0, 2.0, 3.0);
        let translated = translate(&p, 5.0, -3.0, 7.0);
        let sq = translated[0] * translated[0]
            + translated[1] * translated[1]
            + translated[2] * translated[2]
            + translated[3] * translated[3]
            - translated[4] * translated[4];
        assert!(
            approx_eq(sq, 0.0),
            "Translated point should be null, got sq={}",
            sq
        );
    }

    #[test]
    fn test_translation_preserves_distance() {
        let p = point(0.0, 0.0, 0.0);
        let q = point(3.0, 4.0, 0.0);
        let d_before = distance(&p, &q);

        let p2 = translate(&p, 1.0, 2.0, 3.0);
        let q2 = translate(&q, 1.0, 2.0, 3.0);
        let d_after = distance(&p2, &q2);

        assert!(
            approx_eq(d_before, d_after),
            "Distance should be preserved under translation: {} vs {}",
            d_before,
            d_after
        );
    }

    #[test]
    fn test_motor_rotation_only() {
        // A motor with zero moment = pure rotation
        let b = biv(PI / 2.0, 0.0, 0.0);
        let r = Gen::rot(&b);
        let m = Gen::rot_as_mot(&r);
        let p = point(1.0, 0.0, 0.0);
        let rotated = spin_mot_pnt(&m, &p);
        let (x, y, z) = Round::location(&rotated);
        assert!(approx_eq(x, 0.0), "x={}", x);
        assert!(approx_eq(y, 1.0), "y={}", y);
        assert!(approx_eq(z, 0.0), "z={}", z);
    }

    #[test]
    fn test_rotor_composition() {
        // Two PI/4 rotations = one PI/2 rotation
        let b1 = biv(PI / 4.0, 0.0, 0.0);
        let b2 = biv(PI / 4.0, 0.0, 0.0);
        let r1 = Gen::rot(&b1);
        let r2 = Gen::rot(&b2);
        let r_composed = gp_rot_rot(&r1, &r2);

        let r_full = Gen::rot(&biv(PI / 2.0, 0.0, 0.0));
        for i in 0..4 {
            assert!(
                approx_eq(r_composed[i], r_full[i]),
                "idx {} mismatch: {} vs {}",
                i,
                r_composed[i],
                r_full[i]
            );
        }
    }

    #[test]
    fn test_identity_rotor() {
        let r = Gen::rot(&biv(0.0, 0.0, 0.0));
        let p = point(1.0, 2.0, 3.0);
        let result = spin_rot_pnt(&r, &p);
        let (x, y, z) = Round::location(&result);
        assert!(approx_eq(x, 1.0));
        assert!(approx_eq(y, 2.0));
        assert!(approx_eq(z, 3.0));
    }

    #[test]
    fn test_dilator() {
        let d = Gen::dil(0.0);
        // dil(0) should be identity-like: [cosh(0), sinh(0)] = [1, 0]
        assert!(approx_eq(d[0], 1.0));
        assert!(approx_eq(d[1], 0.0));
    }

    // ================================================================
    // Dual / Undual tests
    // ================================================================

    #[test]
    fn test_dual_undual_point_roundtrip() {
        let p = point(1.0, 2.0, 3.0);
        let d = dual_pnt(&p);
        let p2 = undual_sph(&d);
        for i in 0..5 {
            assert!(
                approx_eq(p[i], p2[i]),
                "Pnt dual/undual roundtrip failed at {}: {} vs {}",
                i,
                p[i],
                p2[i]
            );
        }
    }

    #[test]
    fn test_dual_undual_pair_roundtrip() {
        let a = point(1.0, 0.0, 0.0);
        let b = point(0.0, 1.0, 0.0);
        let pp = op_pnt_pnt(&a, &b);
        let d = dual_par(&pp);
        let pp2 = undual_cir(&d);
        for i in 0..10 {
            assert!(
                approx_eq(pp[i], pp2[i]),
                "Par dual/undual roundtrip failed at {}: {} vs {}",
                i,
                pp[i],
                pp2[i]
            );
        }
    }

    #[test]
    fn test_dual_undual_circle_roundtrip() {
        let a = point(1.0, 0.0, 0.0);
        let b = point(0.0, 1.0, 0.0);
        let c = point(0.0, 0.0, 1.0);
        let pp = op_pnt_pnt(&a, &b);
        let cir = op_par_pnt(&pp, &c);
        let d = dual_cir(&cir);
        let cir2 = undual_par(&d);
        for i in 0..10 {
            assert!(
                approx_eq(cir[i], cir2[i]),
                "Cir dual/undual roundtrip failed at {}: {} vs {}",
                i,
                cir[i],
                cir2[i]
            );
        }
    }

    #[test]
    fn test_dual_undual_sphere_roundtrip() {
        let a = point(1.0, 0.0, 0.0);
        let b = point(0.0, 1.0, 0.0);
        let c = point(0.0, 0.0, 1.0);
        let d = point(-1.0, 0.0, 0.0);
        let pp = op_pnt_pnt(&a, &b);
        let cir = op_par_pnt(&pp, &c);
        let sph = op_cir_pnt(&cir, &d);
        let ds = dual_sph(&sph);
        let sph2 = undual_pnt(&ds);
        for i in 0..5 {
            assert!(
                approx_eq(sph[i], sph2[i]),
                "Sph dual/undual roundtrip failed at {}: {} vs {}",
                i,
                sph[i],
                sph2[i]
            );
        }
    }

    // ================================================================
    // Round operation tests
    // ================================================================

    #[test]
    fn test_split_pair() {
        let a = point(1.0, 0.0, 0.0);
        let b = point(-1.0, 0.0, 0.0);
        let pp = op_pnt_pnt(&a, &b);
        let (pa, pb) = Round::split(&pp);
        let (xa, _ya, _za) = Round::location(&pa);
        let (xb, _yb, _zb) = Round::location(&pb);
        // One should be near 1.0 and the other near -1.0 (or vice versa)
        let locs = if xa > xb { (xa, xb) } else { (xb, xa) };
        assert!(approx_eq(locs.0, 1.0), "Expected ~1.0, got {}", locs.0);
        assert!(approx_eq(locs.1, -1.0), "Expected ~-1.0, got {}", locs.1);
    }

    #[test]
    fn test_split_pair_off_origin() {
        let a = point(3.0, 4.0, 0.0);
        let b = point(3.0, -4.0, 0.0);
        let pp = op_pnt_pnt(&a, &b);
        let (pa, pb) = Round::split(&pp);
        let (xa, ya, _za) = Round::location(&pa);
        let (xb, yb, _zb) = Round::location(&pb);
        // Sort by y coordinate
        let (y_hi, y_lo) = if ya > yb { (ya, yb) } else { (yb, ya) };
        assert!(approx_eq(y_hi, 4.0), "Expected y~4.0, got {}", y_hi);
        assert!(approx_eq(y_lo, -4.0), "Expected y~-4.0, got {}", y_lo);
        // Both should have x = 3.0
        assert!(approx_eq(xa, 3.0), "Expected x~3.0, got {}", xa);
        assert!(approx_eq(xb, 3.0), "Expected x~3.0, got {}", xb);
    }

    #[test]
    fn test_size_pair() {
        let a = point(1.0, 0.0, 0.0);
        let b = point(-1.0, 0.0, 0.0);
        let pp = op_pnt_pnt(&a, &b);
        let sz = Round::size_pair(&pp);
        // For a pair from points at distance 2, size should be nonzero
        assert!(sz.abs() > EPS, "Size should be nonzero, got {}", sz);
    }

    #[test]
    fn test_direction_pair() {
        // Pair along x-axis
        let a = point(2.0, 0.0, 0.0);
        let b = point(-2.0, 0.0, 0.0);
        let pp = op_pnt_pnt(&a, &b);
        let dir = Round::direction_pair(&pp);
        // Direction should be along e15 (x-direction)
        let norm = dir.norm();
        assert!(norm > EPS, "Direction norm should be nonzero, got {}", norm);
        // The x-component (e15) should dominate
        let ratio = dir[0].abs() / norm;
        assert!(
            ratio > 0.9,
            "Direction should be along x, ratio = {}",
            ratio
        );
    }

    #[test]
    fn test_carrier_pair() {
        // Pair along x-axis → carrier should be a line along x
        let a = point(1.0, 0.0, 0.0);
        let b = point(-1.0, 0.0, 0.0);
        let pp = op_pnt_pnt(&a, &b);
        let carrier = Round::carrier_pair(&pp);
        assert!(carrier.norm() > EPS, "Carrier should be non-zero");
    }

    #[test]
    fn test_carrier_circle() {
        // Circle in the xy-plane → carrier should be a plane
        let a = point(1.0, 0.0, 0.0);
        let b = point(0.0, 1.0, 0.0);
        let c = point(-1.0, 0.0, 0.0);
        let pp = op_pnt_pnt(&a, &b);
        let cir = op_par_pnt(&pp, &c);
        let carrier = Round::carrier_circle(&cir);
        assert!(carrier.norm() > EPS, "Carrier plane should be non-zero");
    }

    #[test]
    fn test_surround_pair() {
        // Surround of a pair should be a dual sphere containing both points
        let a = point(1.0, 0.0, 0.0);
        let b = point(-1.0, 0.0, 0.0);
        let pp = op_pnt_pnt(&a, &b);
        let sur = Round::surround_pair(&pp);
        assert!(sur.norm() > EPS, "Surround should be non-zero");
        // The center should be at origin
        let (x, y, z) = Round::location(&Round::center(&sur));
        assert!(approx_eq(x, 0.0), "Center x = {}", x);
        assert!(approx_eq(y, 0.0), "Center y = {}", y);
        assert!(approx_eq(z, 0.0), "Center z = {}", z);
    }

    // ================================================================
    // Flat operation tests
    // ================================================================

    #[test]
    fn test_flat_direction_line() {
        // Build a line through two points along x-axis
        let a = point(1.0, 0.0, 0.0);
        let b = point(-1.0, 0.0, 0.0);
        let pp = op_pnt_pnt(&a, &b);
        let lin = Round::carrier_pair(&pp);
        let dir = Flat::direction_lin(&lin);
        // Should have some nonzero direction
        assert!(dir.norm() > EPS, "Line direction should be non-zero");
    }

    #[test]
    fn test_flat_direction_dll() {
        // Test existing Flat::direction on a dual line
        let d = dll(0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
        let dir = Flat::direction(&d);
        assert!(approx_eq(dir[2], 1.0), "Dll direction e23 = {}", dir[2]);
    }

    // ================================================================
    // Versor inverse tests
    // ================================================================

    #[test]
    fn test_inverse_rotor() {
        let b = biv(0.7, -0.3, 1.2);
        let r = Gen::rot(&b);
        let r_inv = inverse_rot(&r);
        let product = gp_rot_rot(&r, &r_inv);
        // Should be identity: [1, 0, 0, 0]
        assert!(approx_eq(product[0], 1.0), "Scalar = {}", product[0]);
        assert!(approx_eq(product[1], 0.0), "e12 = {}", product[1]);
        assert!(approx_eq(product[2], 0.0), "e13 = {}", product[2]);
        assert!(approx_eq(product[3], 0.0), "e23 = {}", product[3]);
    }

    #[test]
    fn test_inverse_motor() {
        let d = dll(0.0, 0.0, PI / 4.0, 1.0, 0.0, 0.0);
        let m = Gen::mot(&d);
        let m_inv = inverse_mot(&m);
        let product = gp_mot_mot(&m, &m_inv);
        assert!(approx_eq(product[0], 1.0), "Scalar = {}", product[0]);
        for i in 1..8 {
            assert!(
                approx_eq(product[i], 0.0),
                "Mot*Mot_inv component {} = {}",
                i,
                product[i]
            );
        }
    }

    #[test]
    fn test_inverse_translator() {
        let t = Gen::trs(3.0, 4.0, 5.0);
        let t_inv = inverse_bst(&t);
        // Apply t then t_inv to a point → should return to original
        let p = point(1.0, 2.0, 3.0);
        let p2 = spin_bst_pnt(&t, &p);
        let p3 = spin_bst_pnt(&t_inv, &p2);
        let (x, y, z) = Round::location(&p3);
        assert!(approx_eq(x, 1.0), "x = {}", x);
        assert!(approx_eq(y, 2.0), "y = {}", y);
        assert!(approx_eq(z, 3.0), "z = {}", z);
    }

    // ================================================================
    // Reflection tests
    // ================================================================

    #[test]
    fn test_reflect_point_through_yz_plane() {
        // Reflect (1, 2, 3) through the yz-plane (normal = e1)
        // Dlp for yz-plane: e1 = 1, e2 = 0, e3 = 0, e5 = 0
        let dlp = Multivector::<4>::new([1.0, 0.0, 0.0, 0.0]);
        let p = point(1.0, 2.0, 3.0);
        let r = reflect_pnt_dlp(&p, &dlp);
        let (x, y, z) = Round::location(&r);
        assert!(approx_eq(x, -1.0), "Reflected x = {}", x);
        assert!(approx_eq(y, 2.0), "Reflected y = {}", y);
        assert!(approx_eq(z, 3.0), "Reflected z = {}", z);
    }

    #[test]
    fn test_reflect_point_through_xz_plane() {
        // Reflect (1, 2, 3) through the xz-plane (normal = e2)
        let dlp = Multivector::<4>::new([0.0, 1.0, 0.0, 0.0]);
        let p = point(1.0, 2.0, 3.0);
        let r = reflect_pnt_dlp(&p, &dlp);
        let (x, y, z) = Round::location(&r);
        assert!(approx_eq(x, 1.0), "Reflected x = {}", x);
        assert!(approx_eq(y, -2.0), "Reflected y = {}", y);
        assert!(approx_eq(z, 3.0), "Reflected z = {}", z);
    }

    #[test]
    fn test_reflect_preserves_null() {
        let dlp = Multivector::<4>::new([1.0, 0.0, 0.0, 0.0]);
        let p = point(2.0, 3.0, 4.0);
        let r = reflect_pnt_dlp(&p, &dlp);
        let sq = r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3] - r[4] * r[4];
        assert!(
            approx_eq(sq, 0.0),
            "Reflected point should be null, sq = {}",
            sq
        );
    }

    #[test]
    fn test_reflect_through_sphere() {
        // Inversion through a unit sphere at origin
        let s = Round::dls(&point(0.0, 0.0, 0.0), 1.0);
        let p = point(2.0, 0.0, 0.0);
        let r = reflect_pnt_dls(&p, &s);
        let (x, y, z) = Round::location(&r);
        // Inversion through unit sphere: x → 1/x for a point along x-axis
        assert!(approx_eq(x, 0.5), "Inverted x = {} (expected 0.5)", x);
        assert!(approx_eq(y, 0.0), "y = {}", y);
        assert!(approx_eq(z, 0.0), "z = {}", z);
    }

    // ================================================================
    // Motor sandwich tests for various types
    // ================================================================

    #[test]
    fn test_motor_translate_pair() {
        let a = point(0.0, 0.0, 0.0);
        let b = point(1.0, 0.0, 0.0);
        let pp = op_pnt_pnt(&a, &b);
        let translated = translate_par(&pp, 5.0, 0.0, 0.0);
        // Split the translated pair and check locations
        let (pa, pb) = Round::split(&translated);
        let (xa, _ya, _za) = Round::location(&pa);
        let (xb, _yb, _zb) = Round::location(&pb);
        let (x_hi, x_lo) = if xa > xb { (xa, xb) } else { (xb, xa) };
        assert!(approx_eq(x_hi, 6.0), "Expected ~6.0, got {}", x_hi);
        assert!(approx_eq(x_lo, 5.0), "Expected ~5.0, got {}", x_lo);
    }

    #[test]
    fn test_motor_translate_circle() {
        let a = point(1.0, 0.0, 0.0);
        let b = point(0.0, 1.0, 0.0);
        let c = point(-1.0, 0.0, 0.0);
        let pp = op_pnt_pnt(&a, &b);
        let cir = op_par_pnt(&pp, &c);
        let translated = translate_cir(&cir, 10.0, 0.0, 0.0);
        assert!(
            translated.norm() > EPS,
            "Translated circle should be non-zero"
        );
    }

    #[test]
    fn test_motor_translate_sphere() {
        // Create sphere, translate it, check center moved
        let a = point(1.0, 0.0, 0.0);
        let b = point(-1.0, 0.0, 0.0);
        let c = point(0.0, 1.0, 0.0);
        let d = point(0.0, 0.0, 1.0);
        let pp = op_pnt_pnt(&a, &b);
        let cir = op_par_pnt(&pp, &c);
        let sph = op_cir_pnt(&cir, &d);
        let translated = translate_sph(&sph, 5.0, 0.0, 0.0);
        assert!(
            translated.norm() > EPS,
            "Translated sphere should be non-zero"
        );
    }

    #[test]
    fn test_motor_rotate_line() {
        // Create a line along x-axis, rotate PI/2 in e12 plane → should go along y
        let a = point(0.0, 0.0, 0.0);
        let b = point(1.0, 0.0, 0.0);
        let pp = op_pnt_pnt(&a, &b);
        let lin = Round::carrier_pair(&pp);
        let rot_biv = biv(PI / 2.0, 0.0, 0.0);
        let r = Gen::rot(&rot_biv);
        let m = Gen::rot_as_mot(&r);
        let rotated = spin_mot_lin(&m, &lin);
        assert!(rotated.norm() > EPS, "Rotated line should be non-zero");
    }

    #[test]
    fn test_motor_translate_line() {
        // Translate a line, then check it moved
        let a = point(0.0, 0.0, 0.0);
        let b = point(1.0, 0.0, 0.0);
        let pp = op_pnt_pnt(&a, &b);
        let lin = Round::carrier_pair(&pp);
        let translated = translate_lin(&lin, 0.0, 5.0, 0.0);
        assert!(
            translated.norm() > EPS,
            "Translated line should be non-zero"
        );
    }

    #[test]
    fn test_motor_translate_plane() {
        // Create plane, translate
        let a = point(1.0, 0.0, 0.0);
        let b = point(0.0, 1.0, 0.0);
        let c = point(-1.0, 0.0, 0.0);
        let pp = op_pnt_pnt(&a, &b);
        let cir = op_par_pnt(&pp, &c);
        let pln = Round::carrier_circle(&cir);
        let translated = translate_pln(&pln, 0.0, 0.0, 5.0);
        assert!(
            translated.norm() > EPS,
            "Translated plane should be non-zero"
        );
    }

    #[test]
    fn test_motor_preserves_pair_size() {
        let a = point(1.0, 0.0, 0.0);
        let b = point(-1.0, 0.0, 0.0);
        let pp = op_pnt_pnt(&a, &b);
        let size_before = Round::size_pair(&pp);

        let rot_biv = biv(0.5, 0.3, 0.7);
        let r = Gen::rot(&rot_biv);
        let m = Gen::rot_as_mot(&r);
        let rotated = spin_mot_par(&m, &pp);
        let size_after = Round::size_pair(&rotated);
        assert!(
            approx_eq(size_before, size_after),
            "Size should be preserved: {} vs {}",
            size_before,
            size_after
        );
    }

    #[test]
    fn test_motor_on_dual_line() {
        // Translate a dual line
        let d = dll(0.0, 0.0, 1.0, 0.0, 0.0, 0.0);
        let rot_biv = biv(PI / 2.0, 0.0, 0.0);
        let r = Gen::rot(&rot_biv);
        let m = Gen::rot_as_mot(&r);
        let rotated = spin_mot_dll(&m, &d);
        assert!(rotated.norm() > EPS, "Rotated dual line should be non-zero");
    }

    #[test]
    fn test_motor_on_dual_plane() {
        // Translate a dual plane
        let dlp = Multivector::<4>::new([0.0, 0.0, 1.0, 0.0]); // z-normal plane through origin
        let rot_biv = biv(PI / 2.0, 0.0, 0.0);
        let r = Gen::rot(&rot_biv);
        let m = Gen::rot_as_mot(&r);
        let rotated = spin_mot_dlp(&m, &dlp);
        assert!(
            rotated.norm() > EPS,
            "Rotated dual plane should be non-zero"
        );
    }

    // ================================================================
    // Meet and Join tests
    // ================================================================

    #[test]
    fn test_join_is_outer_product() {
        let a = point(1.0, 0.0, 0.0);
        let b = point(0.0, 1.0, 0.0);
        let join = join_pnt_pnt(&a, &b);
        let wedge = op_pnt_pnt(&a, &b);
        for i in 0..10 {
            assert!(
                approx_eq(join[i], wedge[i]),
                "Join should equal outer product at {}",
                i
            );
        }
    }

    #[test]
    fn test_meet_two_spheres() {
        // Two intersecting spheres → should give a circle
        let s1 = Round::dls(&point(0.0, 0.0, 0.0), 2.0);
        let s2 = Round::dls(&point(1.0, 0.0, 0.0), 2.0);
        let sph1 = undual_pnt(&s1);
        let sph2 = undual_pnt(&s2);
        let meet = meet_sph_sph(&sph1, &sph2);
        assert!(
            meet.norm() > EPS,
            "Meet of intersecting spheres should be non-zero"
        );
    }

    #[test]
    fn test_meet_planes_gives_line() {
        // Two planes intersecting → should give a line
        // Plane 1: xy-plane (z = 0), in CGA the plane through 3 points
        let a1 = point(1.0, 0.0, 0.0);
        let b1 = point(0.0, 1.0, 0.0);
        let c1 = point(-1.0, 0.0, 0.0);
        let pp1 = op_pnt_pnt(&a1, &b1);
        let cir1 = op_par_pnt(&pp1, &c1);
        let pln1 = Round::carrier_circle(&cir1);

        // Plane 2: xz-plane (y = 0)
        let a2 = point(1.0, 0.0, 0.0);
        let b2 = point(0.0, 0.0, 1.0);
        let c2 = point(-1.0, 0.0, 0.0);
        let pp2 = op_pnt_pnt(&a2, &b2);
        let cir2 = op_par_pnt(&pp2, &c2);
        let pln2 = Round::carrier_circle(&cir2);

        let meet = meet_pln_pln(&pln1, &pln2);
        assert!(
            meet.norm() > EPS,
            "Meet of two planes should give a non-zero line, norm = {}",
            meet.norm()
        );
    }

    // ================================================================
    // Dual Line / Line dual roundtrip
    // ================================================================

    #[test]
    fn test_dual_undual_line_roundtrip() {
        // Build a line, dual it, undual it, check roundtrip
        let a = point(0.0, 0.0, 0.0);
        let b = point(1.0, 0.0, 0.0);
        let pp = op_pnt_pnt(&a, &b);
        let lin = Round::carrier_pair(&pp);
        let d = dual_lin(&lin);
        let lin2 = undual_dll(&d);
        for i in 0..6 {
            assert!(
                approx_eq(lin[i], lin2[i]),
                "Lin dual/undual roundtrip failed at {}: {} vs {}",
                i,
                lin[i],
                lin2[i]
            );
        }
    }

    #[test]
    fn test_dual_undual_plane_roundtrip() {
        let a = point(1.0, 0.0, 0.0);
        let b = point(0.0, 1.0, 0.0);
        let c = point(-1.0, 0.0, 0.0);
        let pp = op_pnt_pnt(&a, &b);
        let cir = op_par_pnt(&pp, &c);
        let pln = Round::carrier_circle(&cir);
        let d = dual_pln(&pln);
        let pln2 = undual_dlp(&d);
        for i in 0..4 {
            assert!(
                approx_eq(pln[i], pln2[i]),
                "Pln dual/undual roundtrip failed at {}: {} vs {}",
                i,
                pln[i],
                pln2[i]
            );
        }
    }

    // ================================================================
    // Direction of Round elements
    // ================================================================

    #[test]
    fn test_direction_circle() {
        // Circle in the xy-plane → direction should be along e12 (i.e., z-like)
        let a = point(1.0, 0.0, 0.0);
        let b = point(0.0, 1.0, 0.0);
        let c = point(-1.0, 0.0, 0.0);
        let pp = op_pnt_pnt(&a, &b);
        let cir = op_par_pnt(&pp, &c);
        let dir = Round::direction_circle(&cir);
        assert!(dir.norm() > EPS, "Circle direction should be non-zero");
    }

    #[test]
    fn test_direction_sphere() {
        let a = point(1.0, 0.0, 0.0);
        let b = point(-1.0, 0.0, 0.0);
        let c = point(0.0, 1.0, 0.0);
        let d = point(0.0, 0.0, 1.0);
        let pp = op_pnt_pnt(&a, &b);
        let cir = op_par_pnt(&pp, &c);
        let sph = op_cir_pnt(&cir, &d);
        let dir = Round::direction_sphere(&sph);
        assert!(
            dir.norm() > EPS,
            "Sphere direction should be non-zero, got {}",
            dir.norm()
        );
    }

    // ================================================================
    // Size of Round elements
    // ================================================================

    #[test]
    fn test_size_circle() {
        let a = point(1.0, 0.0, 0.0);
        let b = point(0.0, 1.0, 0.0);
        let c = point(-1.0, 0.0, 0.0);
        let pp = op_pnt_pnt(&a, &b);
        let cir = op_par_pnt(&pp, &c);
        let sz = Round::size_circle(&cir);
        // Circle through (1,0,0), (0,1,0), (-1,0,0) should have nonzero size
        assert!(sz.abs() > EPS, "Circle size should be nonzero, got {}", sz);
    }

    #[test]
    fn test_size_sphere() {
        let a = point(1.0, 0.0, 0.0);
        let b = point(-1.0, 0.0, 0.0);
        let c = point(0.0, 1.0, 0.0);
        let d = point(0.0, 0.0, 1.0);
        let pp = op_pnt_pnt(&a, &b);
        let cir = op_par_pnt(&pp, &c);
        let sph = op_cir_pnt(&cir, &d);
        let sz = Round::size_sphere(&sph);
        assert!(sz.abs() > EPS, "Sphere size should be nonzero, got {}", sz);
    }

    // ================================================================
    // Flat location tests
    // ================================================================

    #[test]
    fn test_flat_location_dlp() {
        // Dual plane: z = 5 → dlp = [0, 0, 1, -5] i.e. e3 - 5*e_inf...
        // Actually Dlp = [e1, e2, e3, e5]. For a plane z = d,
        // the dual plane is e3 + d * (e4 + e5)/2... this is complex.
        // Simpler: take a plane through origin normal to z: dlp = [0, 0, 1, 0]
        // Location of any point on this plane should project onto z=0.
        let dlp = Multivector::<4>::new([0.0, 0.0, 1.0, 0.0]);
        let p = point(3.0, 4.0, 5.0);
        let loc = Flat::location_dlp(&dlp, &p);
        let (_x, _y, z) = Round::location(&loc);
        // The closest point on the plane z=0 to (3,4,5) is (3,4,0)
        assert!(approx_eq(z, 0.0), "z should be 0, got {}", z);
    }
}
