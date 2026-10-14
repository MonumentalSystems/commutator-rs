use crate::{
    checked_complex, finite_complex, positive, Amplitude, Complex64, ResponseError, ResponseKind,
    ResponseTensor, Result,
};

/// Checked real, exactly symmetric positive-definite linear transport matrix.
///
/// In 2D this is sheet conductance in siemens; in 3D this is bulk conductivity
/// in S/m. The simple reference assumes the matrix is real and frequency
/// independent at both the drive and detection frequencies. It includes all
/// Cartesian components, and is not a scalar-resistivity approximation.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearTransport<const N: usize> {
    data: [[f64; N]; N],
    lower_scaled: [[f64; N]; N],
    scale: f64,
}

impl LinearTransport<2> {
    /// Check a sheet-conductance matrix in siemens, without thickness conversion.
    pub fn sheet_conductance_siemens(data: [[f64; 2]; 2]) -> Result<Self> {
        Self::checked(data)
    }
}

impl LinearTransport<3> {
    /// Check a bulk-conductivity matrix in S/m, without sheet conversion.
    pub fn bulk_conductivity_siemens_per_metre(data: [[f64; 3]; 3]) -> Result<Self> {
        Self::checked(data)
    }
}

impl<const N: usize> LinearTransport<N> {
    fn checked(data: [[f64; N]; N]) -> Result<Self> {
        if N != 2 && N != 3 {
            return Err(ResponseError::InvalidDimension);
        }
        if data.iter().flatten().any(|v| !v.is_finite()) {
            return Err(ResponseError::NonFiniteData);
        }
        let scale = data
            .iter()
            .flatten()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max);
        if scale <= 0.0 {
            return Err(ResponseError::NotPositiveDefinite);
        }
        let mut lower = [[0.0; N]; N];
        for (i, row) in data.iter().enumerate() {
            for (j, input) in row.iter().enumerate().take(i + 1) {
                if *input != data[j][i] {
                    return Err(ResponseError::NotPositiveDefinite);
                }
                let sum: f64 = (0..j).map(|k| lower[i][k] * lower[j][k]).sum();
                let residual = input / scale - sum;
                lower[i][j] = if i == j {
                    if !residual.is_finite() || residual <= 0.0 {
                        return Err(ResponseError::NotPositiveDefinite);
                    }
                    residual.sqrt()
                } else {
                    residual / lower[j][j]
                };
                if !lower[i][j].is_finite() {
                    return Err(ResponseError::NumericalOverflow);
                }
            }
        }
        Ok(Self {
            data,
            lower_scaled: lower,
            scale,
        })
    }

    /// Copy of the checked matrix, in S for 2D or S/m for 3D.
    pub fn components_si(&self) -> [[f64; N]; N] {
        self.data
    }

    /// Solve `linear_current = transport * electric_field` for the field in V/m.
    ///
    /// Input is A/m for 2D or A/m² for 3D. This permits anisotropic mixing.
    pub fn electric_field(&self, linear_current: [Complex64; N]) -> Result<[Complex64; N]> {
        for value in linear_current {
            finite_complex(value)?;
        }
        let mut y = [Complex64::new(0.0, 0.0); N];
        for (i, current) in linear_current.iter().enumerate() {
            let sum: Complex64 = (0..i).map(|k| y[k] * self.lower_scaled[i][k]).sum();
            y[i] = checked_complex((current / self.scale - sum) / self.lower_scaled[i][i])?;
        }
        let mut out = [Complex64::new(0.0, 0.0); N];
        for i in (0..N).rev() {
            let sum: Complex64 = ((i + 1)..N).map(|k| out[k] * self.lower_scaled[k][i]).sum();
            out[i] = checked_complex((y[i] - sum) / self.lower_scaled[i][i])?;
        }
        Ok(out)
    }

    /// Weak-response open-circuit field `E^(2w) = -transport^-1 j_source^(2w)`.
    ///
    /// Assumes no imposed second-harmonic terminal current in any direction.
    /// The source is A/m (2D) or A/m² (3D); output is peak V/m.
    pub fn open_circuit_field(&self, source_peak: [Complex64; N]) -> Result<[Complex64; N]> {
        Ok(self.electric_field(source_peak)?.map(|v| -v))
    }
}

/// Uniform rectangular-bar conversion with explicit current cross section and leads.
///
/// Current is injected along one Cartesian axis. The signed displacement vector
/// runs from the negative voltage lead to the positive voltage lead in metres;
/// voltage is `V_positive - V_negative = -E dot displacement`. No contact-size,
/// nonuniform-current, heating, capacitive or external-circuit model is included.
#[derive(Clone, Debug, PartialEq)]
pub struct HallBar<const N: usize> {
    drive_axis: usize,
    current_cross_section: f64,
    voltage_displacement_metres: [f64; N],
}

impl HallBar<2> {
    /// Sheet bar with positive transverse width in metres and signed lead vector.
    pub fn sheet(
        drive_axis: usize,
        width_metres: f64,
        voltage_displacement_metres: [f64; 2],
    ) -> Result<Self> {
        positive(width_metres, "sheet bar width")?;
        Self::checked(drive_axis, width_metres, voltage_displacement_metres)
    }
}

impl HallBar<3> {
    /// Bulk bar with transverse width and thickness in metres and signed leads.
    ///
    /// Both lengths span the current cross section perpendicular to `drive_axis`.
    pub fn bulk(
        drive_axis: usize,
        width_metres: f64,
        thickness_metres: f64,
        voltage_displacement_metres: [f64; 3],
    ) -> Result<Self> {
        positive(width_metres, "bulk bar width")?;
        positive(thickness_metres, "bulk bar thickness")?;
        let area = width_metres * thickness_metres;
        positive(area, "current cross-sectional area")?;
        Self::checked(drive_axis, area, voltage_displacement_metres)
    }
}

impl<const N: usize> HallBar<N> {
    fn checked(
        drive_axis: usize,
        current_cross_section: f64,
        voltage_displacement_metres: [f64; N],
    ) -> Result<Self> {
        if drive_axis >= N {
            return Err(ResponseError::InvalidAxis);
        }
        if voltage_displacement_metres.iter().any(|v| !v.is_finite()) {
            return Err(ResponseError::NonFiniteData);
        }
        if voltage_displacement_metres.iter().all(|v| *v == 0.0) {
            return Err(ResponseError::InvalidParameter("voltage lead separation"));
        }
        Ok(Self {
            drive_axis,
            current_cross_section,
            voltage_displacement_metres,
        })
    }

    /// Drive direction in the fixed Cartesian basis.
    pub fn drive_axis(&self) -> usize {
        self.drive_axis
    }

    /// Width in metres for 2D, or area in square metres for 3D.
    pub fn current_cross_section_si(&self) -> f64 {
        self.current_cross_section
    }

    /// Signed displacement from negative to positive voltage lead, in metres.
    pub fn voltage_displacement_metres(&self) -> [f64; N] {
        self.voltage_displacement_metres
    }

    /// Peak fundamental field from total applied current in amperes.
    ///
    /// The input amplitude convention is explicit. The full transport matrix
    /// is inverted, allowing transverse field components under anisotropy.
    pub fn drive_field(
        &self,
        transport: &LinearTransport<N>,
        current_amperes: Complex64,
        amplitude: Amplitude,
    ) -> Result<[Complex64; N]> {
        let mut density = [Complex64::new(0.0, 0.0); N];
        density[self.drive_axis] =
            checked_complex(amplitude.to_peak(current_amperes)? / self.current_cross_section)?;
        transport.electric_field(density)
    }

    /// Signed open-circuit second-harmonic voltage from a peak current source.
    ///
    /// Peak output is divided by `sqrt(2)` when RMS voltage is requested; this
    /// is separate from squaring the fundamental input's peak conversion.
    pub fn open_circuit_voltage(
        &self,
        transport: &LinearTransport<N>,
        source_peak: [Complex64; N],
        output: Amplitude,
    ) -> Result<Complex64> {
        let field = transport.open_circuit_field(source_peak)?;
        let mut voltage = Complex64::new(0.0, 0.0);
        for (component, displacement) in field.iter().zip(self.voltage_displacement_metres) {
            voltage -= component * displacement;
            checked_complex(voltage)?;
        }
        output.from_peak(voltage)
    }

    /// Complete weak-response current-to-voltage conversion with explicit amplitudes.
    ///
    /// `response` must be an intrinsic harmonic tensor with drive frequency and
    /// relaxation time appropriate to the sample. The matrix is used at both
    /// frequencies; total transport beyond this real constant-matrix model is
    /// outside the contract. Input current is in amperes, output voltage in volts.
    pub fn second_harmonic_voltage(
        &self,
        transport: &LinearTransport<N>,
        response: &ResponseTensor<N>,
        current_amperes: Complex64,
        input: Amplitude,
        output: Amplitude,
    ) -> Result<Complex64> {
        if response.kind() != ResponseKind::SecondHarmonic {
            return Err(ResponseError::InvalidParameter(
                "second harmonic requires harmonic tensor",
            ));
        }
        let field = self.drive_field(transport, current_amperes, input)?;
        self.open_circuit_voltage(transport, response.current(field)?, output)
    }
}
