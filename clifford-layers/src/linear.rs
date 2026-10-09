//! Geometric-product linear layers.

use clifford_core::CliffordAlgebra;

/// A linear layer whose weights and channels are Clifford multivectors.
///
/// For output channel `o`, the layer computes
/// `sum_i weights[o, i] * input[i] + bias[o]`, where `*` is the
/// algebra's geometric product.
#[derive(Debug, Clone)]
pub struct CliffordLinear {
    /// Algebra and blade ordering used by this layer.
    pub algebra: CliffordAlgebra,
    /// Weights in `[out_channels, in_channels, n_blades]` order.
    pub weights: Vec<f32>,
    /// Optional bias in `[out_channels, n_blades]` order.
    pub bias: Option<Vec<f32>>,
    /// Number of input multivector channels.
    pub in_channels: usize,
    /// Number of output multivector channels.
    pub out_channels: usize,
}

impl CliffordLinear {
    /// Construct a layer with deterministic Kaiming-scale parameters.
    ///
    /// # Panics
    ///
    /// Panics when either channel count is zero.
    pub fn new(
        algebra: CliffordAlgebra,
        in_channels: usize,
        out_channels: usize,
        use_bias: bool,
    ) -> Self {
        assert!(in_channels > 0, "in_channels must be nonzero");
        assert!(out_channels > 0, "out_channels must be nonzero");
        let n = algebra.n_blades;
        let n_weights = out_channels * in_channels * n;
        let scale = 1.0 / ((in_channels * n).max(1) as f32).sqrt();
        let weights = (0..n_weights)
            .map(|i| {
                let value = ((i as f32 + 1.0) * 2.6534).fract() * 2.0 - 1.0;
                value * scale
            })
            .collect();
        let bias = use_bias.then(|| vec![0.0; out_channels * n]);
        Self {
            algebra,
            weights,
            bias,
            in_channels,
            out_channels,
        }
    }

    /// Apply the layer to one `[in_channels, n_blades]` sample.
    ///
    /// # Panics
    ///
    /// Panics when the input or parameter buffers do not match the declared
    /// channel counts and algebra.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let n = self.algebra.n_blades;
        assert_eq!(input.len(), self.in_channels * n, "input shape mismatch");
        assert_eq!(
            self.weights.len(),
            self.out_channels * self.in_channels * n,
            "weight shape mismatch"
        );
        if let Some(bias) = &self.bias {
            assert_eq!(bias.len(), self.out_channels * n, "bias shape mismatch");
        }

        let mut output = vec![0.0; self.out_channels * n];
        for output_channel in 0..self.out_channels {
            for input_channel in 0..self.in_channels {
                let weight_start = (output_channel * self.in_channels + input_channel) * n;
                let input_start = input_channel * n;
                let product = self.algebra.geometric_product(
                    &self.weights[weight_start..weight_start + n],
                    &input[input_start..input_start + n],
                );
                for (blade, value) in product.into_iter().enumerate() {
                    output[output_channel * n + blade] += value;
                }
            }
        }
        if let Some(bias) = &self.bias {
            for (value, bias) in output.iter_mut().zip(bias) {
                *value += bias;
            }
        }
        output
    }

    /// Apply the layer independently to a sequence of samples.
    ///
    /// Input order is `[sequence, in_channels, n_blades]`; output order is
    /// `[sequence, out_channels, n_blades]`.
    ///
    /// # Panics
    ///
    /// Panics when the input length does not match `sequence_len`.
    pub fn forward_sequence(&self, input: &[f32], sequence_len: usize) -> Vec<f32> {
        let input_size = self.in_channels * self.algebra.n_blades;
        assert_eq!(
            input.len(),
            sequence_len * input_size,
            "input shape mismatch"
        );
        input
            .chunks_exact(input_size)
            .flat_map(|sample| self.forward(sample))
            .collect()
    }

    /// Backward-compatible alias for [`Self::forward_sequence`].
    pub fn forward_seq(&self, input: &[f32], sequence_len: usize) -> Vec<f32> {
        self.forward_sequence(input, sequence_len)
    }

    /// Return the number of learnable scalar parameters.
    pub fn parameter_count(&self) -> usize {
        self.weights.len() + self.bias.as_ref().map_or(0, Vec::len)
    }

    /// Backward-compatible alias for [`Self::parameter_count`].
    pub fn n_params(&self) -> usize {
        self.parameter_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_shape_and_values_are_finite() {
        let layer = CliffordLinear::new(CliffordAlgebra::cl3(), 3, 5, true);
        let output = layer.forward(&[0.1; 3 * 8]);
        assert_eq!(output.len(), 5 * 8);
        assert!(output.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn scalar_identity_weight_preserves_input() {
        let mut layer = CliffordLinear::new(CliffordAlgebra::cl3(), 1, 1, false);
        layer.weights.fill(0.0);
        layer.weights[0] = 1.0;
        let input = vec![0.5, 0.1, -0.3, 0.7, 0.2, -0.1, 0.4, 0.05];
        assert_eq!(layer.forward(&input), input);
    }

    #[test]
    fn sequence_shape_is_preserved() {
        let layer = CliffordLinear::new(CliffordAlgebra::cl3(), 2, 4, true);
        let output = layer.forward_sequence(&vec![0.25; 7 * 2 * 8], 7);
        assert_eq!(output.len(), 7 * 4 * 8);
    }

    #[test]
    fn parameter_count_includes_bias() {
        let layer = CliffordLinear::new(CliffordAlgebra::cl3(), 2, 3, true);
        assert_eq!(layer.parameter_count(), 3 * 2 * 8 + 3 * 8);
    }

    #[test]
    #[should_panic(expected = "in_channels must be nonzero")]
    fn rejects_zero_input_channels() {
        CliffordLinear::new(CliffordAlgebra::cl3(), 0, 1, false);
    }

    #[test]
    #[should_panic(expected = "out_channels must be nonzero")]
    fn rejects_zero_output_channels() {
        CliffordLinear::new(CliffordAlgebra::cl3(), 1, 0, false);
    }

    #[test]
    #[should_panic(expected = "input shape mismatch")]
    fn rejects_invalid_input_shape() {
        CliffordLinear::new(CliffordAlgebra::cl3(), 1, 1, false).forward(&[0.0; 7]);
    }
}
