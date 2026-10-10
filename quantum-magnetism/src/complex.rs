use core::iter::Sum;
use core::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// A minimal double-precision complex number used for spin amplitudes.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[repr(C)]
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
    /// Imaginary unit.
    pub const I: Self = Self::new(0.0, 1.0);

    /// Construct a complex number from Cartesian components.
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Construct a real complex number.
    pub const fn from_real(value: f64) -> Self {
        Self::new(value, 0.0)
    }

    /// Return the complex conjugate.
    pub const fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    /// Return the squared modulus.
    pub fn norm_sqr(self) -> f64 {
        self.re.mul_add(self.re, self.im * self.im)
    }

    /// Return the modulus.
    pub fn norm(self) -> f64 {
        self.norm_sqr().sqrt()
    }

    /// Return whether both components are finite.
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    /// Return `exp(i angle)`.
    pub fn cis(angle: f64) -> Self {
        let (sine, cosine) = angle.sin_cos();
        Self::new(cosine, sine)
    }
}

impl From<f64> for Complex64 {
    fn from(value: f64) -> Self {
        Self::from_real(value)
    }
}

impl Add for Complex64 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl AddAssign for Complex64 {
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl Sub for Complex64 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl SubAssign for Complex64 {
    fn sub_assign(&mut self, rhs: Self) {
        self.re -= rhs.re;
        self.im -= rhs.im;
    }
}

impl Neg for Complex64 {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.re, -self.im)
    }
}

impl Mul for Complex64 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::new(
            self.re.mul_add(rhs.re, -(self.im * rhs.im)),
            self.re.mul_add(rhs.im, self.im * rhs.re),
        )
    }
}

impl Mul<f64> for Complex64 {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self::new(self.re * rhs, self.im * rhs)
    }
}

impl Mul<Complex64> for f64 {
    type Output = Complex64;
    fn mul(self, rhs: Complex64) -> Complex64 {
        rhs * self
    }
}

impl MulAssign<f64> for Complex64 {
    fn mul_assign(&mut self, rhs: f64) {
        self.re *= rhs;
        self.im *= rhs;
    }
}

impl Div<f64> for Complex64 {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self::new(self.re / rhs, self.im / rhs)
    }
}

impl Sum for Complex64 {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |sum, value| sum + value)
    }
}
