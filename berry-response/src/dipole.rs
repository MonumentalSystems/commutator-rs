use crate::{
    checked_complex, epsilon, finite_complex, static_prefactor, BerryConvention, Complex64,
    FieldOrdering, Relaxation, ResponseError, ResponseKind, Result,
};

/// Integrated two-dimensional dipole `(D_xz,D_yz)` in metres.
///
/// The right-handed sheet lies in the `xy` plane with normal `+z`. The canonical
/// definition is `D_bz = sum_n integral d^2 k/(2 pi)^2 f_n d_b Omega_n,z`,
/// with wave vector in inverse metres and Berry connection `i<u|grad_k u>`.
/// This is an integrated dipole, never a momentum-resolved integrand.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dipole2D([f64; 2]);

impl Dipole2D {
    /// Check an integrated dipole in SI metres and normalize its Berry sign.
    pub fn new(metres: [f64; 2], convention: BerryConvention) -> Result<Self> {
        if metres.iter().any(|v| !v.is_finite()) {
            return Err(ResponseError::NonFiniteData);
        }
        Ok(Self(metres.map(|v| v * convention.factor())))
    }

    /// Canonical `(D_xz,D_yz)` components in metres, suitable for backend exports.
    pub fn metres(self) -> [f64; 2] {
        self.0
    }

    /// Intrinsic peak-field to peak-second-harmonic sheet-current response.
    ///
    /// Field units are V/m; current units are A/m. Zero angular frequency is an
    /// AC limit with the harmonic factor `1/2`, not a constant-field response.
    pub fn harmonic_response(
        self,
        relaxation: Relaxation,
        ordering: FieldOrdering,
    ) -> Result<ResponseTensor<2>> {
        self.with_prefactor(
            relaxation.harmonic_prefactor()?,
            ordering,
            ResponseKind::SecondHarmonic,
        )
    }

    /// Constant-field response, with coefficient `e^3 tau / hbar^2`.
    ///
    /// The field passed to the returned tensor is a real constant field in V/m,
    /// not an AC peak or RMS phasor. The resulting sheet current is in A/m.
    pub fn static_response(
        self,
        tau_seconds: f64,
        ordering: FieldOrdering,
    ) -> Result<ResponseTensor<2>> {
        self.with_prefactor(
            Complex64::new(static_prefactor(tau_seconds)?, 0.0),
            ordering,
            ResponseKind::Static,
        )
    }

    fn with_prefactor(
        self,
        k: Complex64,
        ordering: FieldOrdering,
        kind: ResponseKind,
    ) -> Result<ResponseTensor<2>> {
        let data = core::array::from_fn(|a| {
            core::array::from_fn(|b| core::array::from_fn(|c| -k * epsilon(a, 2, c) * self.0[b]))
        });
        ResponseTensor::checked(data, ordering, kind)
    }
}

/// Dimensionless integrated three-dimensional dipole in derivative/curvature order.
///
/// `D[b][d] = sum_n integral d^3 k/(2 pi)^3 f_n d_b Omega_n,d`, with a
/// right-handed Cartesian basis, wave vector in inverse metres, and canonical
/// Berry connection `i<u|grad_k u>`. A momentum-resolved distribution is not D.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dipole3D([[f64; 3]; 3]);

impl Dipole3D {
    /// Check a dimensionless integrated tensor and normalize its Berry sign.
    pub fn new(dimensionless: [[f64; 3]; 3], convention: BerryConvention) -> Result<Self> {
        if dimensionless.iter().flatten().any(|v| !v.is_finite()) {
            return Err(ResponseError::NonFiniteData);
        }
        Ok(Self(
            dimensionless.map(|row| row.map(|v| v * convention.factor())),
        ))
    }

    /// Canonical `D[derivative][curvature]` tensor, suitable for backend exports.
    pub fn dimensionless(self) -> [[f64; 3]; 3] {
        self.0
    }

    /// Intrinsic peak-field to peak-second-harmonic bulk-current response.
    ///
    /// Field units are V/m; current units are A/m². The frequency belongs to the
    /// fundamental drive, not to the detected second harmonic.
    pub fn harmonic_response(
        self,
        relaxation: Relaxation,
        ordering: FieldOrdering,
    ) -> Result<ResponseTensor<3>> {
        self.with_prefactor(
            relaxation.harmonic_prefactor()?,
            ordering,
            ResponseKind::SecondHarmonic,
        )
    }

    /// Constant-field response, with coefficient `e^3 tau / hbar^2`.
    ///
    /// The returned tensor expects real constant electric fields in V/m.
    pub fn static_response(
        self,
        tau_seconds: f64,
        ordering: FieldOrdering,
    ) -> Result<ResponseTensor<3>> {
        self.with_prefactor(
            Complex64::new(static_prefactor(tau_seconds)?, 0.0),
            ordering,
            ResponseKind::Static,
        )
    }

    fn with_prefactor(
        self,
        k: Complex64,
        ordering: FieldOrdering,
        kind: ResponseKind,
    ) -> Result<ResponseTensor<3>> {
        let data = core::array::from_fn(|a| {
            core::array::from_fn(|b| {
                core::array::from_fn(|c| {
                    let coefficient: f64 = (0..3).map(|d| -epsilon(a, d, c) * self.0[b][d]).sum();
                    k * coefficient
                })
            })
        });
        ResponseTensor::checked(data, ordering, kind)
    }
}

/// Current-first intrinsic response `j[a] = sum_bc chi[a][b][c] E[b] E[c]`.
///
/// Both field permutations are always included in [`Self::current`], even when
/// the stored tensor is ordered. This is not a general multi-frequency response
/// or a measured total response including skew scattering or side jump.
#[derive(Clone, Debug, PartialEq)]
pub struct ResponseTensor<const N: usize> {
    data: [[[Complex64; N]; N]; N],
    ordering: FieldOrdering,
    kind: ResponseKind,
}

impl<const N: usize> ResponseTensor<N> {
    fn checked(
        mut data: [[[Complex64; N]; N]; N],
        ordering: FieldOrdering,
        kind: ResponseKind,
    ) -> Result<Self> {
        for value in data.iter().flatten().flatten() {
            checked_complex(*value)?;
        }
        if ordering == FieldOrdering::Symmetric {
            let original = data;
            for (a, plane) in data.iter_mut().enumerate() {
                for (b, row) in plane.iter_mut().enumerate() {
                    for (c, value) in row.iter_mut().enumerate() {
                        *value = original[a][b][c] * 0.5 + original[a][c][b] * 0.5;
                        checked_complex(*value)?;
                    }
                }
            }
        }
        Ok(Self {
            data,
            ordering,
            kind,
        })
    }

    /// Return a copy in `[current][first field][second field]` order.
    pub fn components(&self) -> [[[Complex64; N]; N]; N] {
        self.data
    }

    /// The stored field-index convention.
    pub fn ordering(&self) -> FieldOrdering {
        self.ordering
    }

    /// Whether this tensor represents a constant field or a second harmonic.
    pub fn kind(&self) -> ResponseKind {
        self.kind
    }

    /// Contract both field indices with one electric-field vector in V/m.
    ///
    /// Harmonic tensors take complex peak phasors and return complex peak
    /// second-harmonic currents. Static tensors expect real constant fields.
    /// Bilinear contraction uses no complex conjugation; rectification of a
    /// complex AC field is a different operation and is not implemented here.
    pub fn current(&self, electric_field: [Complex64; N]) -> Result<[Complex64; N]> {
        for value in electric_field {
            finite_complex(value)?;
        }
        if self.kind == ResponseKind::Static && electric_field.iter().any(|v| v.im != 0.0) {
            return Err(ResponseError::InvalidParameter("static field must be real"));
        }
        let mut out = [Complex64::new(0.0, 0.0); N];
        for (a, value) in out.iter_mut().enumerate() {
            for (b, first_field) in electric_field.iter().enumerate() {
                for (c, second_field) in electric_field.iter().enumerate() {
                    *value += self.data[a][b][c] * first_field * second_field;
                    checked_complex(*value)?;
                }
            }
        }
        Ok(out)
    }
}
