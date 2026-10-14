//! Checked Berry-curvature-dipole response and measurement reference contracts.
//!
//! This crate starts with an already integrated dipole. It does not compute
//! electronic bands, integrate a Brillouin zone, or supply a material model.
//! See the crate README for the SI, Berry-sign, field-ordering and lead contracts.
//!
//! # Example
//!
//! ```
//! use berry_response::{BerryConvention, Complex64, Dipole3D, FieldOrdering, Relaxation};
//! let d = Dipole3D::new(
//!     [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -2.0]],
//!     BerryConvention::ConnectionI,
//! )?;
//! let chi = d.harmonic_response(Relaxation::new(1e-13, 1e3)?, FieldOrdering::Ordered)?;
//! let e = [Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)];
//! let j_peak = chi.current(e)?;
//! assert!(j_peak[1].re > 0.0);
//! # Ok::<(), berry_response::ResponseError>(())
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod dipole;
mod transport;

pub use dipole::{Dipole2D, Dipole3D, ResponseTensor};
pub use num_complex::Complex64;
pub use transport::{HallBar, LinearTransport};

/// Positive elementary charge in coulombs; electron charge is `-e`.
pub const ELEMENTARY_CHARGE_COULOMBS: f64 = 1.602_176_634e-19;
/// Reduced Planck constant in joule seconds, derived from exact SI Planck constant.
pub const HBAR_JOULE_SECONDS: f64 = 6.626_070_15e-34 / core::f64::consts::TAU;

/// Result type for checked response calculations.
pub type Result<T> = core::result::Result<T, ResponseError>;

/// Invalid inputs or unrepresentable numerical output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResponseError {
    /// A named scalar failed its physical domain or finiteness constraint.
    InvalidParameter(&'static str),
    /// An array contains a nonfinite component.
    NonFiniteData,
    /// Only two and three spatial dimensions are supported.
    InvalidDimension,
    /// An axis is outside the supported dimension.
    InvalidAxis,
    /// The linear transport matrix is not exactly symmetric positive definite.
    NotPositiveDefinite,
    /// An intermediate or output cannot be represented as a finite number.
    NumericalOverflow,
}

impl core::fmt::Display for ResponseError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidParameter(name) => write!(f, "invalid physical parameter: {name}"),
            Self::NonFiniteData => f.write_str("array contains nonfinite data"),
            Self::InvalidDimension => f.write_str("dimension must be 2 or 3"),
            Self::InvalidAxis => f.write_str("axis outside spatial dimension"),
            Self::NotPositiveDefinite => {
                f.write_str("transport must be symmetric positive definite")
            }
            Self::NumericalOverflow => f.write_str("nonfinite numerical intermediate or output"),
        }
    }
}

impl std::error::Error for ResponseError {}

/// Berry-connection convention supplied by an external band calculation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BerryConvention {
    /// `A = i <u|grad_k u>`, so `Omega_z = -2 Im <d_x u|d_y u>`.
    ConnectionI,
    /// `A = -i <u|grad_k u>`; imported dipoles are negated into [`Self::ConnectionI`].
    ConnectionMinusI,
}

impl BerryConvention {
    pub(crate) fn factor(self) -> f64 {
        match self {
            Self::ConnectionI => 1.0,
            Self::ConnectionMinusI => -1.0,
        }
    }
}

/// Representation of the two driving-field indices, following the current index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldOrdering {
    /// Store `chi[a,b,c] = -K sum_d epsilon[a,d,c] D[b,d]`.
    Ordered,
    /// Store `(chi[a,b,c] + chi[a,c,b])/2`, with identical full-field contraction.
    Symmetric,
}

/// Time-domain meaning of a response coefficient.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResponseKind {
    /// Peak second-harmonic output from peak fundamental fields; includes `1/2`.
    SecondHarmonic,
    /// True constant-field output from real constant fields; no harmonic `1/2`.
    Static,
}

/// Sinusoidal phasor amplitude convention for explicitly designated boundaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Amplitude {
    /// `x(t) = Re[x_peak exp(i omega t)]`.
    Peak,
    /// `x_rms = x_peak / sqrt(2)`, preserving the phasor phase.
    Rms,
}

impl Amplitude {
    /// Convert an input sinusoidal phasor to its peak representation.
    pub fn to_peak(self, value: Complex64) -> Result<Complex64> {
        finite_complex(value)?;
        checked_complex(value * self.peak_factor())
    }

    /// Convert a peak sinusoidal phasor to this output representation.
    pub fn from_peak(self, value: Complex64) -> Result<Complex64> {
        finite_complex(value)?;
        checked_complex(value / self.peak_factor())
    }

    fn peak_factor(self) -> f64 {
        match self {
            Self::Peak => 1.0,
            Self::Rms => core::f64::consts::SQRT_2,
        }
    }
}

/// Positive constant relaxation time and nonnegative angular drive frequency.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Relaxation {
    tau_seconds: f64,
    omega_radians_per_second: f64,
}

impl Relaxation {
    /// Check `tau > 0` seconds and `omega >= +0` radians/second, both finite.
    ///
    /// Zero frequency here denotes the limit of an AC harmonic response. A true
    /// constant field requires [`Dipole3D::static_response`] or its 2D counterpart.
    pub fn new(tau_seconds: f64, omega_radians_per_second: f64) -> Result<Self> {
        positive(tau_seconds, "relaxation time")?;
        nonnegative(omega_radians_per_second, "angular frequency")?;
        let result = Self {
            tau_seconds,
            omega_radians_per_second,
        };
        result.harmonic_prefactor()?;
        Ok(result)
    }

    /// Relaxation time in seconds.
    pub fn tau_seconds(self) -> f64 {
        self.tau_seconds
    }

    /// Angular drive frequency in radians per second.
    pub fn omega_radians_per_second(self) -> f64 {
        self.omega_radians_per_second
    }

    /// Intrinsic harmonic coefficient `e^3 tau / [2 hbar^2 (1+i omega tau)]`.
    ///
    /// Input field and output second-harmonic current phasors both use peak
    /// amplitudes with `exp(+i omega t)`. This does not include scattering terms.
    pub fn harmonic_prefactor(self) -> Result<Complex64> {
        let scale = 0.5 * static_prefactor(self.tau_seconds)?;
        let x = self.omega_radians_per_second * self.tau_seconds;
        if !x.is_finite() {
            return Err(ResponseError::NumericalOverflow);
        }
        // Rearranging for large x avoids squaring it in complex division.
        let inverse = if x <= 1.0 {
            Complex64::new(1.0, -x) / (1.0 + x * x)
        } else {
            let inv = 1.0 / x;
            Complex64::new(inv, -1.0) * (inv / (1.0 + inv * inv))
        };
        checked_complex(inverse * scale)
    }
}

pub(crate) fn static_prefactor(tau: f64) -> Result<f64> {
    positive(tau, "relaxation time")?;
    let e = ELEMENTARY_CHARGE_COULOMBS;
    let value = (e * e * e / (HBAR_JOULE_SECONDS * HBAR_JOULE_SECONDS)) * tau;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(ResponseError::NumericalOverflow)
    }
}

pub(crate) fn positive(value: f64, name: &'static str) -> Result<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(ResponseError::InvalidParameter(name))
    }
}

pub(crate) fn nonnegative(value: f64, name: &'static str) -> Result<()> {
    if value.is_finite() && !value.is_sign_negative() {
        Ok(())
    } else {
        Err(ResponseError::InvalidParameter(name))
    }
}

pub(crate) fn finite_complex(value: Complex64) -> Result<()> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(ResponseError::NonFiniteData)
    }
}

pub(crate) fn checked_complex(value: Complex64) -> Result<Complex64> {
    finite_complex(value).map_err(|_| ResponseError::NumericalOverflow)?;
    Ok(value)
}

pub(crate) fn epsilon(a: usize, b: usize, c: usize) -> f64 {
    match (a, b, c) {
        (0, 1, 2) | (1, 2, 0) | (2, 0, 1) => 1.0,
        (0, 2, 1) | (2, 1, 0) | (1, 0, 2) => -1.0,
        _ => 0.0,
    }
}
