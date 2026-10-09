//! Optional spectral convolution for one-dimensional multivector fields.

use clifford_core::CliffordAlgebra;

use rustfft::{num_complex::Complex32, FftPlanner};

/// Shape and spectral truncation for [`CliffordFourierConv1d`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CliffordFourierConvConfig {
    /// Number of input multivector channels.
    pub in_channels: usize,
    /// Number of output multivector channels.
    pub out_channels: usize,
    /// Number of one-sided low-frequency modes to retain, or all modes.
    pub n_modes: Option<usize>,
}

/// One-dimensional spectral convolution with Clifford-valued weights.
///
/// The layer transforms each blade independently, performs a Clifford
/// geometric product at each retained Fourier mode, restores conjugate
/// symmetry, and transforms the result back to a real spatial field.
#[derive(Debug, Clone)]
pub struct CliffordFourierConv1d {
    /// Layer shape and truncation settings.
    pub config: CliffordFourierConvConfig,
    /// Algebra and blade ordering used by the weights and input.
    pub algebra: CliffordAlgebra,
    /// Real weight coefficients in
    /// `[out_channels, in_channels, retained_modes, n_blades]` order.
    pub weights_re: Vec<f32>,
    /// Imaginary coefficients for strictly complex modes in
    /// `[out_channels, in_channels, complex_modes, n_blades]` order.
    ///
    /// DC and an even-length Nyquist mode are self-conjugate and therefore
    /// have no imaginary parameters.
    pub weights_im: Vec<f32>,
    spatial_size: usize,
    retained_modes: usize,
    complex_modes: usize,
}

impl CliffordFourierConv1d {
    /// Construct a spectral layer for `spatial_size` samples.
    ///
    /// # Panics
    ///
    /// Panics for zero channel counts, zero spatial size, or a requested mode
    /// count outside `1..=spatial_size / 2 + 1`.
    pub fn new(
        algebra: CliffordAlgebra,
        config: CliffordFourierConvConfig,
        spatial_size: usize,
    ) -> Self {
        assert!(config.in_channels > 0, "in_channels must be nonzero");
        assert!(config.out_channels > 0, "out_channels must be nonzero");
        assert!(spatial_size > 0, "spatial_size must be nonzero");
        let available_modes = spatial_size / 2 + 1;
        let retained_modes = config.n_modes.unwrap_or(available_modes);
        assert!(
            (1..=available_modes).contains(&retained_modes),
            "n_modes must fit the one-sided spectrum"
        );
        let n_real_weights =
            config.out_channels * config.in_channels * retained_modes * algebra.n_blades;
        let has_nyquist = spatial_size % 2 == 0 && retained_modes == available_modes;
        let complex_modes = retained_modes - 1 - usize::from(has_nyquist);
        let n_imaginary_weights =
            config.out_channels * config.in_channels * complex_modes * algebra.n_blades;
        let scale = 1.0 / (config.in_channels * config.out_channels) as f32;
        let scale = scale.sqrt();
        let weights_re = (0..n_real_weights)
            .map(|index| scale * (index as f32 * 0.1).cos())
            .collect();
        let weights_im = (0..n_imaginary_weights)
            .map(|index| scale * (index as f32 * 0.1).sin())
            .collect();
        Self {
            config,
            algebra,
            weights_re,
            weights_im,
            spatial_size,
            retained_modes,
            complex_modes,
        }
    }

    /// Return the spatial size fixed when the layer was constructed.
    pub fn spatial_size(&self) -> usize {
        self.spatial_size
    }

    /// Return the number of retained one-sided modes represented by weights.
    pub fn retained_modes(&self) -> usize {
        self.retained_modes
    }

    /// Return the number of retained modes with independent imaginary weights.
    pub fn complex_modes(&self) -> usize {
        self.complex_modes
    }

    /// Return the number of learnable scalar parameters.
    pub fn parameter_count(&self) -> usize {
        self.weights_re.len() + self.weights_im.len()
    }

    /// Backward-compatible alias for [`Self::parameter_count`].
    pub fn n_params(&self) -> usize {
        self.parameter_count()
    }

    /// Apply the spectral convolution.
    ///
    /// Input order is `[in_channels, spatial_size, n_blades]`; output order is
    /// `[out_channels, spatial_size, n_blades]`.
    ///
    /// # Panics
    ///
    /// Panics when the input, spatial size, or weight buffers are invalid.
    pub fn forward(&self, input: &[f32], spatial_size: usize) -> Vec<f32> {
        assert_eq!(
            spatial_size, self.spatial_size,
            "spatial_size must match the constructor"
        );
        let n_blades = self.algebra.n_blades;
        assert_eq!(
            input.len(),
            self.config.in_channels * spatial_size * n_blades,
            "input shape mismatch"
        );
        let expected_real_weights =
            self.config.out_channels * self.config.in_channels * self.retained_modes * n_blades;
        assert_eq!(
            self.weights_re.len(),
            expected_real_weights,
            "real weight shape mismatch"
        );
        let expected_imaginary_weights =
            self.config.out_channels * self.config.in_channels * self.complex_modes * n_blades;
        assert_eq!(
            self.weights_im.len(),
            expected_imaginary_weights,
            "imaginary weight shape mismatch"
        );

        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(spatial_size);
        let inverse_fft = planner.plan_fft_inverse(spatial_size);

        let mut input_spectrum = vec![
            vec![vec![Complex32::new(0.0, 0.0); spatial_size]; n_blades];
            self.config.in_channels
        ];
        for (channel, channel_spectrum) in input_spectrum.iter_mut().enumerate() {
            for (blade, spectrum) in channel_spectrum.iter_mut().enumerate() {
                for (position, value) in spectrum.iter_mut().enumerate() {
                    *value = Complex32::new(
                        input[(channel * spatial_size + position) * n_blades + blade],
                        0.0,
                    );
                }
                fft.process(spectrum);
            }
        }

        let mut output_spectrum = vec![
            vec![vec![Complex32::new(0.0, 0.0); spatial_size]; n_blades];
            self.config.out_channels
        ];
        for (output_channel, output_blades) in output_spectrum.iter_mut().enumerate() {
            for (input_channel, input_blades) in input_spectrum.iter().enumerate() {
                for mode in 0..self.retained_modes {
                    let real_weight_start = ((output_channel * self.config.in_channels
                        + input_channel)
                        * self.retained_modes
                        + mode)
                        * n_blades;
                    let imaginary_weight_start = (mode > 0
                        && !(spatial_size % 2 == 0 && mode == spatial_size / 2))
                        .then(|| {
                            ((output_channel * self.config.in_channels + input_channel)
                                * self.complex_modes
                                + (mode - 1))
                                * n_blades
                        });
                    for weight_blade in 0..n_blades {
                        let weight = Complex32::new(
                            self.weights_re[real_weight_start + weight_blade],
                            imaginary_weight_start
                                .map_or(0.0, |start| self.weights_im[start + weight_blade]),
                        );
                        for (input_blade, blade_spectrum) in input_blades.iter().enumerate() {
                            let table_index = weight_blade * n_blades + input_blade;
                            let output_blade = self.algebra.cayley_index[table_index];
                            let sign = self.algebra.cayley_sign[table_index];
                            output_blades[output_blade][mode] +=
                                weight * blade_spectrum[mode] * sign;
                        }
                    }
                }
            }

            // A real spatial signal requires X[N-k] = conj(X[k]). DC and an
            // even-length Nyquist bin are self-conjugate and therefore real.
            for spectrum in output_blades.iter_mut() {
                spectrum[0].im = 0.0;
                for mode in 1..self.retained_modes {
                    if spatial_size % 2 == 0 && mode == spatial_size / 2 {
                        spectrum[mode].im = 0.0;
                    } else {
                        spectrum[spatial_size - mode] = spectrum[mode].conj();
                    }
                }
            }
        }

        let inverse_scale = 1.0 / spatial_size as f32;
        let mut output = vec![0.0; self.config.out_channels * spatial_size * n_blades];
        for (channel, channel_spectrum) in output_spectrum.iter_mut().enumerate() {
            for (blade, spectrum) in channel_spectrum.iter_mut().enumerate() {
                inverse_fft.process(spectrum);
                for (position, value) in spectrum.iter().enumerate() {
                    output[(channel * spatial_size + position) * n_blades + blade] =
                        value.re * inverse_scale;
                }
            }
        }
        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_shape_matches_configuration() {
        let layer = CliffordFourierConv1d::new(
            CliffordAlgebra::cl3(),
            CliffordFourierConvConfig {
                in_channels: 2,
                out_channels: 4,
                n_modes: Some(8),
            },
            32,
        );
        assert_eq!(layer.weights_re.len(), 4 * 2 * 8 * 8);
        assert_eq!(layer.weights_im.len(), 4 * 2 * 7 * 8);
        assert_eq!(layer.parameter_count(), 4 * 2 * (8 + 7) * 8);
        assert_eq!(layer.spatial_size(), 32);
        assert_eq!(layer.complex_modes(), 7);
    }

    #[test]
    fn forward_shape_and_values_are_valid() {
        let layer = CliffordFourierConv1d::new(
            CliffordAlgebra::cl3(),
            CliffordFourierConvConfig {
                in_channels: 1,
                out_channels: 2,
                n_modes: Some(8),
            },
            16,
        );
        let output = layer.forward(&vec![0.1; 16 * 8], 16);
        assert_eq!(output.len(), 2 * 16 * 8);
        assert!(output.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn even_full_spectrum_omits_dc_and_nyquist_imaginary_parameters() {
        let layer = CliffordFourierConv1d::new(
            CliffordAlgebra::complex(),
            CliffordFourierConvConfig {
                in_channels: 1,
                out_channels: 1,
                n_modes: None,
            },
            8,
        );
        assert_eq!(layer.retained_modes(), 5);
        assert_eq!(layer.complex_modes(), 3);
        assert_eq!(layer.weights_re.len(), 5 * 2);
        assert_eq!(layer.weights_im.len(), 3 * 2);
        assert_eq!(layer.parameter_count(), 16);
    }

    #[test]
    #[should_panic(expected = "spatial_size must match the constructor")]
    fn forward_rejects_a_different_spatial_size() {
        let layer = CliffordFourierConv1d::new(
            CliffordAlgebra::complex(),
            CliffordFourierConvConfig {
                in_channels: 1,
                out_channels: 1,
                n_modes: Some(2),
            },
            8,
        );
        layer.forward(&[0.0; 16], 4);
    }

    #[test]
    fn truncated_spectrum_removes_unrepresented_frequency() {
        let spatial_size = 32;
        let mut layer = CliffordFourierConv1d::new(
            CliffordAlgebra::cl3(),
            CliffordFourierConvConfig {
                in_channels: 1,
                out_channels: 1,
                n_modes: Some(4),
            },
            spatial_size,
        );
        layer.weights_re.fill(0.0);
        layer.weights_im.fill(0.0);
        for mode in 0..layer.retained_modes() {
            layer.weights_re[mode * 8] = 1.0;
        }
        let mut input = vec![0.0; spatial_size * 8];
        for position in 0..spatial_size {
            input[position * 8] =
                (16.0 * std::f32::consts::PI * position as f32 / spatial_size as f32).sin();
        }
        let energy = layer
            .forward(&input, spatial_size)
            .into_iter()
            .map(|value| value * value)
            .sum::<f32>();
        assert!(energy < 1e-6, "truncated output energy was {energy}");
    }
}
