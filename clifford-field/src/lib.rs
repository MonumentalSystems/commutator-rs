//! Precision-generic Clifford field primitives and CPU spatial operators.
//!
//! This crate deliberately contains no networking or experiment runtime. It
//! supplies sealed f32/f64 scalar behavior, Cl(1,3) bivectors, field storage,
//! spatial derivatives, commutator terms, energy, chiral decomposition, and
//! transport-neutral numerical post-processing.

#![forbid(unsafe_code)]

use clifford_core::CliffordScalar;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};

mod bivector;
mod stepper;

pub mod analysis;

pub use bivector::{
    chiral_split, BivectorField, BivectorField32, BivectorField64, BoundaryCondition,
    ChiralDecomposition, StaBivector, StaBivector32, StaBivector64,
};
pub use stepper::{step_euler_reference, step_strang_reference};

const DEFAULT_BIVECTOR_RAYON_THRESHOLD: usize = 65_536;
static BIVECTOR_RAYON_THRESHOLD: AtomicUsize = AtomicUsize::new(DEFAULT_BIVECTOR_RAYON_THRESHOLD);

/// Minimum field size at which CPU spatial operators use Rayon.
///
/// Changing this affects dispatch and floating-point reduction order, not the
/// mathematical stencil. Host runtimes may mirror a calibrated threshold here.
pub fn bivector_rayon_threshold() -> usize {
    BIVECTOR_RAYON_THRESHOLD.load(Ordering::Relaxed)
}

/// Configure the Rayon crossover used by CPU field operators.
///
/// Returns the previous value so hosts can record or restore their dispatch
/// policy. A value of zero forces the parallel paths.
pub fn set_bivector_rayon_threshold(threshold: usize) -> usize {
    BIVECTOR_RAYON_THRESHOLD.swap(threshold, Ordering::Relaxed)
}

mod private {
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Runtime precision selector for protocol messages and configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FieldPrecision {
    /// IEEE-754 single precision.
    F32,
    /// IEEE-754 double precision.
    F64,
}

impl FieldPrecision {
    /// Bytes per scalar value.
    pub fn byte_size(self) -> usize {
        match self {
            FieldPrecision::F32 => 4,
            FieldPrecision::F64 => 8,
        }
    }
}

impl std::fmt::Display for FieldPrecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FieldPrecision::F32 => write!(f, "f32"),
            FieldPrecision::F64 => write!(f, "f64"),
        }
    }
}

impl std::str::FromStr for FieldPrecision {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "f32" | "fp32" | "float32" | "32" => Ok(FieldPrecision::F32),
            "f64" | "fp64" | "float64" | "64" => Ok(FieldPrecision::F64),
            _ => Err(format!("unknown precision '{}', expected f32 or f64", s)),
        }
    }
}

/// Trait for scalar types used in bivector field simulation.
///
/// Sealed and implemented only for `f32` and `f64`. All physics simulation structs
/// (`StaBivector<S>`, `BivectorField<S>`, etc.) are generic over this trait.
/// The monomorphized code paths are selected at runtime via `FieldPrecision`.
pub trait FieldScalar:
    CliffordScalar
    + private::Sealed
    + Copy
    + Clone
    + Default
    + Send
    + Sync
    + PartialEq
    + PartialOrd
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
    + std::ops::AddAssign
    + std::ops::SubAssign
    + std::ops::MulAssign
    + std::ops::DivAssign
    + std::fmt::Debug
    + std::fmt::Display
    + Serialize
    + for<'de> Deserialize<'de>
    + 'static
    + std::iter::Sum
{
    /// Scalar value two.
    const TWO: Self;
    /// Scalar value three.
    const THREE: Self;
    /// Ratio of a circle's circumference to its diameter.
    const PI: Self;
    /// One full turn in radians, equal to `2 * PI`.
    const TAU: Self;
    /// Small positive tolerance used by field algorithms.
    const EPSILON: Self;
    /// Negative infinity.
    const NEG_INFINITY: Self;
    /// Positive infinity.
    const INFINITY: Self;
    /// Largest finite value representable by the scalar type.
    const MAX: Self;
    /// Bytes per value (4 for f32, 8 for f64).
    const BYTE_SIZE: usize;
    /// Whether GPU acceleration is available for this precision.
    /// Currently true only for f32 (Metal, CUDA consumer GPUs).
    const GPU_SUPPORTED: bool;
    /// The corresponding runtime precision enum variant.
    const PRECISION: FieldPrecision;

    // --- Math functions ---
    /// Return the absolute value.
    fn abs(self) -> Self;
    /// Return the sine and cosine in one operation.
    fn sin_cos(self) -> (Self, Self);
    /// Return the inverse cosine in radians.
    fn acos(self) -> Self;
    /// Return the four-quadrant arctangent of `self / other`.
    fn atan2(self, other: Self) -> Self;
    /// Round to the nearest integer-valued scalar.
    fn round(self) -> Self;
    /// Round down to the nearest integer-valued scalar.
    fn floor(self) -> Self;
    /// Round up to the nearest integer-valued scalar.
    fn ceil(self) -> Self;
    /// Restrict this value to the inclusive interval `[min, max]`.
    fn clamp(self, min: Self, max: Self) -> Self;
    /// Raise this value to an integer power.
    fn powi(self, n: i32) -> Self;
    /// Raise this value to a scalar power.
    fn powf(self, n: Self) -> Self;
    /// Return the natural logarithm.
    fn ln(self) -> Self;
    /// Return the base-10 logarithm.
    fn log10(self) -> Self;
    /// Return the base-2 logarithm.
    fn log2(self) -> Self;
    /// Return `e` raised to this value.
    fn exp(self) -> Self;
    /// Return the hyperbolic tangent.
    fn tanh(self) -> Self;
    /// Return the lesser of `self` and `other`.
    fn min(self, other: Self) -> Self;
    /// Return whether this value is finite.
    fn is_finite(self) -> bool;
    /// Return whether this value is not a number.
    fn is_nan(self) -> bool;
    /// Return a value with magnitude one and this value's sign.
    fn signum(self) -> Self;
    /// Return this magnitude with `sign`'s sign bit.
    fn copysign(self, sign: Self) -> Self;
    /// Return the multiplicative inverse.
    fn recip(self) -> Self;

    // --- Conversion ---
    /// Convert to `f64`.
    fn to_f64(self) -> f64;
    /// Convert to `f32`.
    fn to_f32(self) -> f32;
    /// Convert a `usize` into this scalar type.
    fn from_usize(v: usize) -> Self;
    /// Convert an `i32` into this scalar type.
    fn from_i32(v: i32) -> Self;
    /// Convert a `u32` into this scalar type.
    fn from_u32(v: u32) -> Self;

    // --- Serialization ---
    /// Write little-endian bytes into the provided buffer.
    /// Buffer must be at least `BYTE_SIZE` bytes.
    fn write_le_bytes(self, buf: &mut [u8]);
    /// Read from little-endian bytes.
    fn read_le_bytes(buf: &[u8]) -> Self;
    /// Convert to raw bits for XOR delta encoding.
    fn to_bits_u64(self) -> u64;
    /// Convert from raw bits.
    fn from_bits_u64(bits: u64) -> Self;
}

// =============================================================================
// f32 implementation
// =============================================================================

impl FieldScalar for f32 {
    const TWO: f32 = 2.0;
    const THREE: f32 = 3.0;
    const PI: f32 = std::f32::consts::PI;
    const TAU: f32 = std::f32::consts::TAU;
    const EPSILON: f32 = 1e-10;
    const NEG_INFINITY: f32 = f32::NEG_INFINITY;
    const INFINITY: f32 = f32::INFINITY;
    const MAX: f32 = f32::MAX;
    const BYTE_SIZE: usize = 4;
    const GPU_SUPPORTED: bool = true;
    const PRECISION: FieldPrecision = FieldPrecision::F32;

    #[inline(always)]
    fn abs(self) -> f32 {
        f32::abs(self)
    }
    #[inline(always)]
    fn sin_cos(self) -> (f32, f32) {
        f32::sin_cos(self)
    }
    #[inline(always)]
    fn acos(self) -> f32 {
        f32::acos(self)
    }
    #[inline(always)]
    fn atan2(self, other: f32) -> f32 {
        f32::atan2(self, other)
    }
    #[inline(always)]
    fn round(self) -> f32 {
        f32::round(self)
    }
    #[inline(always)]
    fn floor(self) -> f32 {
        f32::floor(self)
    }
    #[inline(always)]
    fn ceil(self) -> f32 {
        f32::ceil(self)
    }
    #[inline(always)]
    fn clamp(self, min: f32, max: f32) -> f32 {
        f32::clamp(self, min, max)
    }
    #[inline(always)]
    fn powi(self, n: i32) -> f32 {
        f32::powi(self, n)
    }
    #[inline(always)]
    fn powf(self, n: f32) -> f32 {
        f32::powf(self, n)
    }
    #[inline(always)]
    fn ln(self) -> f32 {
        f32::ln(self)
    }
    #[inline(always)]
    fn log10(self) -> f32 {
        f32::log10(self)
    }
    #[inline(always)]
    fn log2(self) -> f32 {
        f32::log2(self)
    }
    #[inline(always)]
    fn exp(self) -> f32 {
        f32::exp(self)
    }
    #[inline(always)]
    fn tanh(self) -> f32 {
        f32::tanh(self)
    }
    #[inline(always)]
    fn min(self, other: f32) -> f32 {
        f32::min(self, other)
    }
    #[inline(always)]
    fn is_finite(self) -> bool {
        f32::is_finite(self)
    }
    #[inline(always)]
    fn is_nan(self) -> bool {
        f32::is_nan(self)
    }
    #[inline(always)]
    fn signum(self) -> f32 {
        f32::signum(self)
    }
    #[inline(always)]
    fn copysign(self, sign: f32) -> f32 {
        f32::copysign(self, sign)
    }
    #[inline(always)]
    fn recip(self) -> f32 {
        1.0 / self
    }

    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }
    #[inline(always)]
    fn from_usize(v: usize) -> f32 {
        v as f32
    }
    #[inline(always)]
    fn from_i32(v: i32) -> f32 {
        v as f32
    }
    #[inline(always)]
    fn from_u32(v: u32) -> f32 {
        v as f32
    }

    #[inline(always)]
    fn write_le_bytes(self, buf: &mut [u8]) {
        buf[..4].copy_from_slice(&self.to_le_bytes());
    }
    #[inline(always)]
    fn read_le_bytes(buf: &[u8]) -> f32 {
        f32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]])
    }
    #[inline(always)]
    fn to_bits_u64(self) -> u64 {
        self.to_bits() as u64
    }
    #[inline(always)]
    fn from_bits_u64(bits: u64) -> f32 {
        f32::from_bits(bits as u32)
    }
}

// =============================================================================
// f64 implementation
// =============================================================================

impl FieldScalar for f64 {
    const TWO: f64 = 2.0;
    const THREE: f64 = 3.0;
    const PI: f64 = std::f64::consts::PI;
    const TAU: f64 = std::f64::consts::TAU;
    const EPSILON: f64 = 1e-20;
    const NEG_INFINITY: f64 = f64::NEG_INFINITY;
    const INFINITY: f64 = f64::INFINITY;
    const MAX: f64 = f64::MAX;
    const BYTE_SIZE: usize = 8;
    const GPU_SUPPORTED: bool = false;
    const PRECISION: FieldPrecision = FieldPrecision::F64;

    #[inline(always)]
    fn abs(self) -> f64 {
        f64::abs(self)
    }
    #[inline(always)]
    fn sin_cos(self) -> (f64, f64) {
        f64::sin_cos(self)
    }
    #[inline(always)]
    fn acos(self) -> f64 {
        f64::acos(self)
    }
    #[inline(always)]
    fn atan2(self, other: f64) -> f64 {
        f64::atan2(self, other)
    }
    #[inline(always)]
    fn round(self) -> f64 {
        f64::round(self)
    }
    #[inline(always)]
    fn floor(self) -> f64 {
        f64::floor(self)
    }
    #[inline(always)]
    fn ceil(self) -> f64 {
        f64::ceil(self)
    }
    #[inline(always)]
    fn clamp(self, min: f64, max: f64) -> f64 {
        f64::clamp(self, min, max)
    }
    #[inline(always)]
    fn powi(self, n: i32) -> f64 {
        f64::powi(self, n)
    }
    #[inline(always)]
    fn powf(self, n: f64) -> f64 {
        f64::powf(self, n)
    }
    #[inline(always)]
    fn ln(self) -> f64 {
        f64::ln(self)
    }
    #[inline(always)]
    fn log10(self) -> f64 {
        f64::log10(self)
    }
    #[inline(always)]
    fn log2(self) -> f64 {
        f64::log2(self)
    }
    #[inline(always)]
    fn exp(self) -> f64 {
        f64::exp(self)
    }
    #[inline(always)]
    fn tanh(self) -> f64 {
        f64::tanh(self)
    }
    #[inline(always)]
    fn min(self, other: f64) -> f64 {
        f64::min(self, other)
    }
    #[inline(always)]
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }
    #[inline(always)]
    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }
    #[inline(always)]
    fn signum(self) -> f64 {
        f64::signum(self)
    }
    #[inline(always)]
    fn copysign(self, sign: f64) -> f64 {
        f64::copysign(self, sign)
    }
    #[inline(always)]
    fn recip(self) -> f64 {
        1.0 / self
    }

    #[inline(always)]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32
    }
    #[inline(always)]
    fn from_usize(v: usize) -> f64 {
        v as f64
    }
    #[inline(always)]
    fn from_i32(v: i32) -> f64 {
        v as f64
    }
    #[inline(always)]
    fn from_u32(v: u32) -> f64 {
        v as f64
    }

    #[inline(always)]
    fn write_le_bytes(self, buf: &mut [u8]) {
        buf[..8].copy_from_slice(&self.to_le_bytes());
    }
    #[inline(always)]
    fn read_le_bytes(buf: &[u8]) -> f64 {
        f64::from_le_bytes([
            buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7],
        ])
    }
    #[inline(always)]
    fn to_bits_u64(self) -> u64 {
        self.to_bits()
    }
    #[inline(always)]
    fn from_bits_u64(bits: u64) -> f64 {
        f64::from_bits(bits)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn check_scalar_basics<S: FieldScalar>() {
        assert_eq!(S::ZERO + S::ONE, S::ONE);
        assert_eq!(S::ONE + S::ONE, S::TWO);
        assert_eq!(S::TWO * S::HALF, S::ONE);
        assert!((S::PI.to_f64() - std::f64::consts::PI).abs() < 1e-6);
        assert!(S::EPSILON.to_f64() > 0.0);
        assert!(!S::ONE.is_nan());
        assert!(!S::ONE.is_finite() || S::ONE.is_finite()); // tautology, just exercises the method
        assert!(S::ONE.is_finite());
        assert!(!S::INFINITY.is_finite());
    }

    #[test]
    fn test_f32_basics() {
        check_scalar_basics::<f32>();
    }

    #[test]
    fn test_f64_basics() {
        check_scalar_basics::<f64>();
    }

    fn check_math_functions<S: FieldScalar>() {
        let x = S::from_f64(0.5);
        assert!((x.sin().to_f64() - 0.5_f64.sin()).abs() < 1e-6);
        assert!((x.cos().to_f64() - 0.5_f64.cos()).abs() < 1e-6);
        assert!((x.sqrt().to_f64() - 0.5_f64.sqrt()).abs() < 1e-6);
        assert!((x.exp().to_f64() - 0.5_f64.exp()).abs() < 1e-6);
        assert!((x.ln().to_f64() - 0.5_f64.ln()).abs() < 1e-6);

        let four = S::from_f64(4.0);
        assert_eq!(four.sqrt().to_f64(), 2.0);
    }

    #[test]
    fn test_f32_math() {
        check_math_functions::<f32>();
    }

    #[test]
    fn test_f64_math() {
        check_math_functions::<f64>();
    }

    fn check_serialization_roundtrip<S: FieldScalar>() {
        let vals = [
            S::ZERO,
            S::ONE,
            S::PI,
            S::from_f64(-1.23456789012345),
            S::EPSILON,
        ];
        for &v in &vals {
            let mut buf = vec![0u8; S::BYTE_SIZE];
            v.write_le_bytes(&mut buf);
            let v2 = S::read_le_bytes(&buf);
            assert_eq!(v, v2, "roundtrip failed for {:?}", v);
        }
    }

    #[test]
    fn test_f32_serialization() {
        check_serialization_roundtrip::<f32>();
    }

    #[test]
    fn test_f64_serialization() {
        check_serialization_roundtrip::<f64>();
    }

    fn check_bits_roundtrip<S: FieldScalar>() {
        let vals = [S::ZERO, S::ONE, S::PI, S::from_f64(-42.0)];
        for &v in &vals {
            let bits = v.to_bits_u64();
            let v2 = S::from_bits_u64(bits);
            assert_eq!(v, v2);
        }
    }

    #[test]
    fn test_f32_bits() {
        check_bits_roundtrip::<f32>();
    }

    #[test]
    fn test_f64_bits() {
        check_bits_roundtrip::<f64>();
    }

    #[test]
    fn test_precision_enum_display() {
        assert_eq!(format!("{}", FieldPrecision::F32), "f32");
        assert_eq!(format!("{}", FieldPrecision::F64), "f64");
    }

    #[test]
    fn test_precision_enum_parse() {
        assert_eq!(
            "f32".parse::<FieldPrecision>().unwrap(),
            FieldPrecision::F32
        );
        assert_eq!(
            "fp64".parse::<FieldPrecision>().unwrap(),
            FieldPrecision::F64
        );
        assert_eq!("64".parse::<FieldPrecision>().unwrap(), FieldPrecision::F64);
        assert!("f128".parse::<FieldPrecision>().is_err());
    }

    #[test]
    fn test_precision_byte_size() {
        assert_eq!(FieldPrecision::F32.byte_size(), 4);
        assert_eq!(FieldPrecision::F64.byte_size(), 8);
    }

    #[test]
    fn test_f64_has_more_precision() {
        // The whole point: f64 should preserve more digits than f32
        let v = 1.0000001_f64;
        let f32_roundtrip = (v as f32) as f64;
        let f64_roundtrip = v;
        assert!((f64_roundtrip - v).abs() < (f32_roundtrip - v).abs());
    }

    #[test]
    fn test_conversion_between_types() {
        let v = std::f64::consts::PI;
        let f32_val = f32::from_f64(v);
        let back = f32_val.to_f64();
        // f32 loses precision
        assert!((back - v).abs() > 1e-8);
        // f64 preserves it
        let f64_val = f64::from_f64(v);
        assert_eq!(f64_val, v);
    }
}
