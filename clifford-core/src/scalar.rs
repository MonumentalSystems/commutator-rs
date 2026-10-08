//! Minimal scalar contract for precision-generic Clifford algebra.

/// Scalar operations required by the Clifford algebra engine.
///
/// This trait is open to external implementations. It contains no assumptions
/// about memory layout, GPU support, serialization, or wire precision.
pub trait CliffordScalar:
    Copy
    + Default
    + PartialEq
    + PartialOrd
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::AddAssign
    + std::iter::Sum
{
    /// Additive identity.
    const ZERO: Self;
    /// Multiplicative identity.
    const ONE: Self;
    /// Exact representation of one half.
    const HALF: Self;

    /// Return the principal square root.
    fn sqrt(self) -> Self;
    /// Return the sine of this value in radians.
    fn sin(self) -> Self;
    /// Return the cosine of this value in radians.
    fn cos(self) -> Self;
    /// Return the greater of `self` and `other`.
    fn max(self, other: Self) -> Self;
    /// Convert an `f32` into this scalar type.
    fn from_f32(value: f32) -> Self;
    /// Convert an `f64` into this scalar type.
    fn from_f64(value: f64) -> Self;
}

impl CliffordScalar for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const HALF: Self = 0.5;

    fn sqrt(self) -> Self {
        self.sqrt()
    }
    fn sin(self) -> Self {
        self.sin()
    }
    fn cos(self) -> Self {
        self.cos()
    }
    fn max(self, other: Self) -> Self {
        self.max(other)
    }
    fn from_f32(value: f32) -> Self {
        value
    }
    fn from_f64(value: f64) -> Self {
        value as f32
    }
}

impl CliffordScalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const HALF: Self = 0.5;

    fn sqrt(self) -> Self {
        self.sqrt()
    }
    fn sin(self) -> Self {
        self.sin()
    }
    fn cos(self) -> Self {
        self.cos()
    }
    fn max(self, other: Self) -> Self {
        self.max(other)
    }
    fn from_f32(value: f32) -> Self {
        value as f64
    }
    fn from_f64(value: f64) -> Self {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::CliffordScalar;

    fn check_scalar<S: CliffordScalar + std::fmt::Debug>() {
        assert_eq!(S::ZERO + S::ONE, S::ONE);
        assert_eq!(S::ONE * S::HALF, S::HALF);
        assert_eq!(S::from_f32(4.0).sqrt(), S::from_f32(2.0));
        assert_eq!(S::from_f64(4.0).sqrt(), S::from_f64(2.0));
    }

    #[test]
    fn f32_implements_clifford_scalar() {
        check_scalar::<f32>();
    }

    #[test]
    fn f64_implements_clifford_scalar() {
        check_scalar::<f64>();
    }
}
