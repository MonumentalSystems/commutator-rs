//! Minimal complex scalar used by the model representation.

use core::fmt;
use core::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

/// A double-precision complex number.
///
/// The deliberately small type avoids imposing a linear-algebra or complex
/// number dependency on applications that only need BdG matrix assembly.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex64 {
    /// Real component.
    pub re: f64,
    /// Imaginary component.
    pub im: f64,
}

impl Complex64 {
    /// Additive identity.
    pub const ZERO: Self = Self::new(0.0, 0.0);

    /// Multiplicative identity.
    pub const ONE: Self = Self::new(1.0, 0.0);

    /// Creates a complex value from real and imaginary components.
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Returns `true` when both components are finite.
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    /// Returns the complex conjugate.
    pub fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    /// Returns the squared magnitude.
    pub fn norm_sqr(self) -> f64 {
        self.re.mul_add(self.re, self.im * self.im)
    }

    /// Returns the magnitude.
    pub fn norm(self) -> f64 {
        self.norm_sqr().sqrt()
    }
}

impl From<f64> for Complex64 {
    fn from(value: f64) -> Self {
        Self::new(value, 0.0)
    }
}

impl fmt::Display for Complex64 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} {:+}i", self.re, self.im)
    }
}

impl Add for Complex64 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl AddAssign for Complex64 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Complex64 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl SubAssign for Complex64 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Neg for Complex64 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::new(-self.re, -self.im)
    }
}

impl Mul for Complex64 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.re.mul_add(rhs.re, -(self.im * rhs.im)),
            self.re.mul_add(rhs.im, self.im * rhs.re),
        )
    }
}

impl Mul<f64> for Complex64 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

impl Mul<Complex64> for f64 {
    type Output = Complex64;

    fn mul(self, rhs: Complex64) -> Self::Output {
        rhs * self
    }
}

impl Div<f64> for Complex64 {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        Self::new(self.re / rhs, self.im / rhs)
    }
}
