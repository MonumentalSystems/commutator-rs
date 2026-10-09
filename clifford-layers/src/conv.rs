//! Spatial Clifford convolutions and multivector activations.

use clifford_core::CliffordAlgebra;

/// Expand per-blade weights into a dense real block kernel.
///
/// Weights use `[weight_blade, out_channel, in_channel, spatial]` order.
/// The result uses `[output_blade * out_channel, input_blade * in_channel,
/// spatial]` order and implements left geometric multiplication.
///
/// # Panics
///
/// Panics when the weight length does not match the declared shape.
pub fn build_clifford_kernel(
    weights: &[f32],
    out_channels: usize,
    in_channels: usize,
    spatial_size: usize,
    algebra: &CliffordAlgebra,
) -> Vec<f32> {
    let n_blades = algebra.n_blades;
    let weight_stride = out_channels * in_channels * spatial_size;
    assert_eq!(
        weights.len(),
        n_blades * weight_stride,
        "weight shape mismatch"
    );
    let output_width = n_blades * out_channels;
    let input_width = n_blades * in_channels;
    let mut kernel = vec![0.0; output_width * input_width * spatial_size];

    for weight_blade in 0..n_blades {
        for input_blade in 0..n_blades {
            let table_index = weight_blade * n_blades + input_blade;
            let output_blade = algebra.cayley_index[table_index];
            let sign = algebra.cayley_sign[table_index];
            for output_channel in 0..out_channels {
                for input_channel in 0..in_channels {
                    for spatial in 0..spatial_size {
                        let weight_index = weight_blade * weight_stride
                            + (output_channel * in_channels + input_channel) * spatial_size
                            + spatial;
                        let output_row = output_blade * out_channels + output_channel;
                        let input_column = input_blade * in_channels + input_channel;
                        let kernel_index =
                            (output_row * input_width + input_column) * spatial_size + spatial;
                        kernel[kernel_index] += sign * weights[weight_index];
                    }
                }
            }
        }
    }
    kernel
}

/// Build the four-component rotation kernel used by rotation-mode 2D layers.
///
/// The six leading weight planes contain four quaternion coefficients, a
/// rotation scale, and the scalar-to-vector coefficient respectively.
///
/// # Panics
///
/// Panics when the weight length does not equal
/// `6 * out_channels * in_channels * spatial_size`.
pub fn build_rotation_kernel(
    weights: &[f32],
    out_channels: usize,
    in_channels: usize,
    spatial_size: usize,
) -> Vec<f32> {
    let weight_stride = out_channels * in_channels * spatial_size;
    assert_eq!(weights.len(), 6 * weight_stride, "weight shape mismatch");
    let output_width = 4 * out_channels;
    let input_width = 4 * in_channels;
    let mut kernel = vec![0.0; output_width * input_width * spatial_size];

    for output_channel in 0..out_channels {
        for input_channel in 0..in_channels {
            for spatial in 0..spatial_size {
                let base = (output_channel * in_channels + input_channel) * spatial_size + spatial;
                let quaternion = [
                    weights[base],
                    weights[weight_stride + base],
                    weights[2 * weight_stride + base],
                    weights[3 * weight_stride + base],
                ];
                let scale = weights[4 * weight_stride + base];
                let scalar_to_vector = weights[5 * weight_stride + base];
                let norm =
                    (quaternion.iter().map(|value| value * value).sum::<f32>() + 1e-4).sqrt();
                let q = quaternion.map(|value| value / norm);
                let [q0, q1, q2, q3] = q;
                let matrix = [
                    [
                        quaternion[0],
                        -quaternion[1],
                        -quaternion[2],
                        -quaternion[3],
                    ],
                    [
                        scalar_to_vector,
                        scale * (1.0 - 2.0 * (q2 * q2 + q3 * q3)),
                        scale * (2.0 * (q1 * q2 - q0 * q3)),
                        scale * (2.0 * (q1 * q3 + q0 * q2)),
                    ],
                    [
                        scalar_to_vector,
                        scale * (2.0 * (q1 * q2 + q0 * q3)),
                        scale * (1.0 - 2.0 * (q1 * q1 + q3 * q3)),
                        scale * (2.0 * (q2 * q3 - q0 * q1)),
                    ],
                    [
                        scalar_to_vector,
                        scale * (2.0 * (q1 * q3 - q0 * q2)),
                        scale * (2.0 * (q2 * q3 + q0 * q1)),
                        scale * (1.0 - 2.0 * (q1 * q1 + q2 * q2)),
                    ],
                ];
                for (output_blade, matrix_row) in matrix.iter().enumerate() {
                    for (input_blade, &coefficient) in matrix_row.iter().enumerate() {
                        let output_row = output_blade * out_channels + output_channel;
                        let input_column = input_blade * in_channels + input_channel;
                        kernel
                            [(output_row * input_width + input_column) * spatial_size + spatial] =
                            coefficient;
                    }
                }
            }
        }
    }
    kernel
}

/// Convert `[blade, channel, spatial]` storage into
/// `[channel, blade, spatial]` storage.
///
/// # Panics
///
/// Panics when the buffer length does not match the declared shape.
pub fn interleave_blades(
    data: &[f32],
    n_blades: usize,
    n_channels: usize,
    spatial_size: usize,
) -> Vec<f32> {
    assert_eq!(
        data.len(),
        n_blades * n_channels * spatial_size,
        "data shape mismatch"
    );
    let mut output = vec![0.0; data.len()];
    for blade in 0..n_blades {
        for channel in 0..n_channels {
            for spatial in 0..spatial_size {
                output[(channel * n_blades + blade) * spatial_size + spatial] =
                    data[(blade * n_channels + channel) * spatial_size + spatial];
            }
        }
    }
    output
}

/// Convert `[channel, blade, spatial]` storage back into
/// `[blade, channel, spatial]` storage.
///
/// # Panics
///
/// Panics when the buffer length does not match the declared shape.
pub fn deinterleave_blades(
    data: &[f32],
    n_blades: usize,
    n_channels: usize,
    spatial_size: usize,
) -> Vec<f32> {
    assert_eq!(
        data.len(),
        n_blades * n_channels * spatial_size,
        "data shape mismatch"
    );
    let mut output = vec![0.0; data.len()];
    for channel in 0..n_channels {
        for blade in 0..n_blades {
            for spatial in 0..spatial_size {
                output[(blade * n_channels + channel) * spatial_size + spatial] =
                    data[(channel * n_blades + blade) * spatial_size + spatial];
            }
        }
    }
    output
}

fn deterministic_weights(length: usize, fan_in: usize) -> Vec<f32> {
    let bound = 1.0 / fan_in.max(1) as f32;
    let bound = bound.sqrt();
    (0..length)
        .map(|index| (((index as f32 + 1.0) * 2.6534).fract() * 2.0 - 1.0) * bound)
        .collect()
}

fn output_extent(input: usize, kernel: usize, stride: usize, padding: usize) -> usize {
    assert!(kernel > 0, "kernel extent must be nonzero");
    assert!(stride > 0, "stride must be nonzero");
    let padded = input
        .checked_add(padding.checked_mul(2).expect("padding overflow"))
        .expect("padded extent overflow");
    assert!(padded >= kernel, "kernel exceeds padded input");
    (padded - kernel) / stride + 1
}

/// One-dimensional convolution over multivector-valued spatial fields.
#[derive(Debug, Clone)]
pub struct CliffordConv1d {
    /// Algebra and blade ordering used by the layer.
    pub algebra: CliffordAlgebra,
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels.
    pub out_channels: usize,
    /// Kernel width.
    pub kernel_size: usize,
    /// Spatial stride.
    pub stride: usize,
    /// Symmetric zero-padding width.
    pub padding: usize,
    /// Number of independent channel groups.
    pub groups: usize,
    /// Weights in `[blade, out_channel, in_channel_per_group, kernel]` order.
    pub weights: Vec<f32>,
    /// Optional `[out_channel, blade]` bias.
    pub bias: Option<Vec<f32>>,
}

impl CliffordConv1d {
    /// Construct a one-dimensional Clifford convolution.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        algebra: CliffordAlgebra,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        groups: usize,
        use_bias: bool,
    ) -> Self {
        assert!(groups > 0, "groups must be nonzero");
        assert!(
            in_channels > 0 && in_channels % groups == 0,
            "invalid input groups"
        );
        assert!(
            out_channels > 0 && out_channels % groups == 0,
            "invalid output groups"
        );
        assert!(kernel_size > 0, "kernel_size must be nonzero");
        assert!(stride > 0, "stride must be nonzero");
        let in_per_group = in_channels / groups;
        let weight_count = algebra.n_blades * out_channels * in_per_group * kernel_size;
        let weights =
            deterministic_weights(weight_count, kernel_size * in_channels * algebra.n_blades);
        let bias = use_bias.then(|| vec![0.0; out_channels * algebra.n_blades]);
        Self {
            algebra,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            weights,
            bias,
        }
    }

    /// Apply the convolution to `[batch, channel, spatial, blade]` input.
    ///
    /// # Panics
    ///
    /// Panics when the input or parameter buffers do not match the layer.
    pub fn forward(&self, input: &[f32], batch: usize, spatial_len: usize) -> Vec<f32> {
        let n_blades = self.algebra.n_blades;
        assert_eq!(
            input.len(),
            batch * self.in_channels * spatial_len * n_blades,
            "input shape mismatch"
        );
        let in_per_group = self.in_channels / self.groups;
        let out_per_group = self.out_channels / self.groups;
        let weight_stride = self.out_channels * in_per_group * self.kernel_size;
        assert_eq!(
            self.weights.len(),
            n_blades * weight_stride,
            "weight shape mismatch"
        );
        if let Some(bias) = &self.bias {
            assert_eq!(
                bias.len(),
                self.out_channels * n_blades,
                "bias shape mismatch"
            );
        }
        let output_len = output_extent(spatial_len, self.kernel_size, self.stride, self.padding);
        let mut output = vec![0.0; batch * self.out_channels * output_len * n_blades];

        for batch_index in 0..batch {
            for group in 0..self.groups {
                for output_local in 0..out_per_group {
                    let output_channel = group * out_per_group + output_local;
                    for output_position in 0..output_len {
                        let mut sum = vec![0.0; n_blades];
                        for input_local in 0..in_per_group {
                            let input_channel = group * in_per_group + input_local;
                            for kernel_position in 0..self.kernel_size {
                                let padded_position =
                                    output_position * self.stride + kernel_position;
                                if padded_position < self.padding
                                    || padded_position >= spatial_len + self.padding
                                {
                                    continue;
                                }
                                let input_position = padded_position - self.padding;
                                let input_start = ((batch_index * self.in_channels
                                    + input_channel)
                                    * spatial_len
                                    + input_position)
                                    * n_blades;
                                let mut weight = vec![0.0; n_blades];
                                for (weight_blade, coefficient) in weight.iter_mut().enumerate() {
                                    *coefficient = self.weights[weight_blade * weight_stride
                                        + (output_channel * in_per_group + input_local)
                                            * self.kernel_size
                                        + kernel_position];
                                }
                                let product = self.algebra.geometric_product(
                                    &weight,
                                    &input[input_start..input_start + n_blades],
                                );
                                for (target, value) in sum.iter_mut().zip(product) {
                                    *target += value;
                                }
                            }
                        }
                        let output_start = ((batch_index * self.out_channels + output_channel)
                            * output_len
                            + output_position)
                            * n_blades;
                        for blade in 0..n_blades {
                            output[output_start + blade] = sum[blade]
                                + self
                                    .bias
                                    .as_ref()
                                    .map_or(0.0, |bias| bias[output_channel * n_blades + blade]);
                        }
                    }
                }
            }
        }
        output
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

/// Two-dimensional convolution over multivector-valued spatial fields.
#[derive(Debug, Clone)]
pub struct CliffordConv2d {
    /// Algebra and blade ordering used by the layer.
    pub algebra: CliffordAlgebra,
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels.
    pub out_channels: usize,
    /// Kernel `(height, width)`.
    pub kernel_size: (usize, usize),
    /// Spatial `(row, column)` stride.
    pub stride: (usize, usize),
    /// Symmetric `(row, column)` zero padding.
    pub padding: (usize, usize),
    /// Number of independent channel groups.
    pub groups: usize,
    /// Whether weights encode the specialized four-component rotation kernel.
    pub rotation: bool,
    /// Per-blade weights, or six weight planes in rotation mode.
    pub weights: Vec<f32>,
    /// Optional `[out_channel, blade]` bias.
    pub bias: Option<Vec<f32>>,
}

impl CliffordConv2d {
    /// Construct a two-dimensional Clifford convolution.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        algebra: CliffordAlgebra,
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        groups: usize,
        rotation: bool,
        use_bias: bool,
    ) -> Self {
        assert!(groups > 0, "groups must be nonzero");
        assert!(
            in_channels > 0 && in_channels % groups == 0,
            "invalid input groups"
        );
        assert!(
            out_channels > 0 && out_channels % groups == 0,
            "invalid output groups"
        );
        assert!(
            kernel_size.0 > 0 && kernel_size.1 > 0,
            "kernel must be nonzero"
        );
        assert!(stride.0 > 0 && stride.1 > 0, "stride must be nonzero");
        if rotation {
            assert_eq!(algebra.n_blades, 4, "rotation mode requires four blades");
        }
        let in_per_group = in_channels / groups;
        let kernel_area = kernel_size.0 * kernel_size.1;
        let weight_planes = if rotation { 6 } else { algebra.n_blades };
        let weight_count = weight_planes * out_channels * in_per_group * kernel_area;
        let weights =
            deterministic_weights(weight_count, kernel_area * in_channels * algebra.n_blades);
        let bias = use_bias.then(|| vec![0.0; out_channels * algebra.n_blades]);
        Self {
            algebra,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            rotation,
            weights,
            bias,
        }
    }

    /// Apply the convolution to `[batch, channel, height, width, blade]` input.
    ///
    /// # Panics
    ///
    /// Panics when the input or parameter buffers do not match the layer.
    pub fn forward(&self, input: &[f32], batch: usize, height: usize, width: usize) -> Vec<f32> {
        let n_blades = self.algebra.n_blades;
        assert_eq!(
            input.len(),
            batch * self.in_channels * height * width * n_blades,
            "input shape mismatch"
        );
        let output_height =
            output_extent(height, self.kernel_size.0, self.stride.0, self.padding.0);
        let output_width = output_extent(width, self.kernel_size.1, self.stride.1, self.padding.1);
        let in_per_group = self.in_channels / self.groups;
        let out_per_group = self.out_channels / self.groups;
        let kernel_area = self.kernel_size.0 * self.kernel_size.1;
        let weight_planes = if self.rotation { 6 } else { n_blades };
        let weight_stride = self.out_channels * in_per_group * kernel_area;
        assert_eq!(
            self.weights.len(),
            weight_planes * weight_stride,
            "weight shape mismatch"
        );
        let rotation_kernel = self.rotation.then(|| {
            build_rotation_kernel(&self.weights, self.out_channels, in_per_group, kernel_area)
        });
        let mut output =
            vec![0.0; batch * self.out_channels * output_height * output_width * n_blades];

        for batch_index in 0..batch {
            for group in 0..self.groups {
                for output_local in 0..out_per_group {
                    let output_channel = group * out_per_group + output_local;
                    for output_row in 0..output_height {
                        for output_column in 0..output_width {
                            let mut sum = vec![0.0; n_blades];
                            for input_local in 0..in_per_group {
                                let input_channel = group * in_per_group + input_local;
                                for kernel_row in 0..self.kernel_size.0 {
                                    for kernel_column in 0..self.kernel_size.1 {
                                        let padded_row = output_row * self.stride.0 + kernel_row;
                                        let padded_column =
                                            output_column * self.stride.1 + kernel_column;
                                        if padded_row < self.padding.0
                                            || padded_row >= height + self.padding.0
                                            || padded_column < self.padding.1
                                            || padded_column >= width + self.padding.1
                                        {
                                            continue;
                                        }
                                        let input_row = padded_row - self.padding.0;
                                        let input_column = padded_column - self.padding.1;
                                        let input_start = (((batch_index * self.in_channels
                                            + input_channel)
                                            * height
                                            + input_row)
                                            * width
                                            + input_column)
                                            * n_blades;
                                        let kernel_position =
                                            kernel_row * self.kernel_size.1 + kernel_column;
                                        if let Some(kernel) = &rotation_kernel {
                                            let input_width = n_blades * in_per_group;
                                            for (output_blade, sum_value) in
                                                sum.iter_mut().enumerate()
                                            {
                                                let output_row_index = output_blade
                                                    * self.out_channels
                                                    + output_channel;
                                                for input_blade in 0..n_blades {
                                                    let input_column_index =
                                                        input_blade * in_per_group + input_local;
                                                    let kernel_index = (output_row_index
                                                        * input_width
                                                        + input_column_index)
                                                        * kernel_area
                                                        + kernel_position;
                                                    *sum_value += kernel[kernel_index]
                                                        * input[input_start + input_blade];
                                                }
                                            }
                                        } else {
                                            let mut weight = vec![0.0; n_blades];
                                            for (weight_blade, coefficient) in
                                                weight.iter_mut().enumerate()
                                            {
                                                *coefficient = self.weights[weight_blade
                                                    * weight_stride
                                                    + (output_channel * in_per_group
                                                        + input_local)
                                                        * kernel_area
                                                    + kernel_position];
                                            }
                                            let product = self.algebra.geometric_product(
                                                &weight,
                                                &input[input_start..input_start + n_blades],
                                            );
                                            for (target, value) in sum.iter_mut().zip(product) {
                                                *target += value;
                                            }
                                        }
                                    }
                                }
                            }
                            let output_start = (((batch_index * self.out_channels
                                + output_channel)
                                * output_height
                                + output_row)
                                * output_width
                                + output_column)
                                * n_blades;
                            for blade in 0..n_blades {
                                output[output_start + blade] = sum[blade]
                                    + self.bias.as_ref().map_or(0.0, |bias| {
                                        bias[output_channel * n_blades + blade]
                                    });
                            }
                        }
                    }
                }
            }
        }
        output
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

/// Aggregation used to derive the scalar gate in [`vector_silu`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorSiLUMode {
    /// Sum all blade coefficients.
    Sum,
    /// Average all blade coefficients.
    Mean,
    /// Use a caller-supplied linear combination of blade coefficients.
    Linear,
}

/// Apply one scalar SiLU-style gate to every blade of each multivector.
///
/// # Panics
///
/// Panics for zero blades, a non-divisible input, or missing/invalid weights
/// in [`VectorSiLUMode::Linear`] mode.
pub fn vector_silu(
    input: &[f32],
    n_blades: usize,
    mode: VectorSiLUMode,
    linear_weights: Option<&[f32]>,
) -> Vec<f32> {
    assert!(n_blades > 0, "n_blades must be nonzero");
    assert_eq!(input.len() % n_blades, 0, "input shape mismatch");
    input
        .chunks_exact(n_blades)
        .flat_map(|multivector| {
            let gate_input = match mode {
                VectorSiLUMode::Sum => multivector.iter().sum(),
                VectorSiLUMode::Mean => multivector.iter().sum::<f32>() / n_blades as f32,
                VectorSiLUMode::Linear => {
                    let weights = linear_weights.expect("linear mode requires weights");
                    assert_eq!(weights.len(), n_blades, "linear weight shape mismatch");
                    multivector
                        .iter()
                        .zip(weights)
                        .map(|(value, weight)| value * weight)
                        .sum()
                }
            };
            let gate = 1.0 / (1.0 + (-gate_input).exp());
            multivector
                .iter()
                .map(move |value| value * gate)
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Linear feature mixing preceded by a shared Clifford sandwich action.
///
/// Despite its origin in a projective-geometric-algebra layer, this type is
/// signature-independent: the caller supplies the algebra and action.
#[derive(Debug, Clone)]
pub struct CliffordConjugateLinear {
    /// Algebra and blade ordering used by the action.
    pub algebra: CliffordAlgebra,
    /// Number of input multivector features.
    pub in_features: usize,
    /// Number of output multivector features.
    pub out_features: usize,
    /// Learned versor-like action multivector.
    pub action: Vec<f32>,
    /// Scalar mixing weights in `[out_features, in_features]` order.
    pub weight: Vec<f32>,
}

impl CliffordConjugateLinear {
    /// Construct a layer whose action starts at the scalar identity.
    pub fn new(algebra: CliffordAlgebra, in_features: usize, out_features: usize) -> Self {
        assert!(in_features > 0, "in_features must be nonzero");
        assert!(out_features > 0, "out_features must be nonzero");
        let mut action = vec![0.0; algebra.n_blades];
        action[0] = 1.0;
        let weight = deterministic_weights(out_features * in_features, in_features);
        Self {
            algebra,
            in_features,
            out_features,
            action,
            weight,
        }
    }

    /// Apply only the shared sandwich action to one multivector.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        self.algebra.sandwich(&self.action, input)
    }

    /// Apply the action and then mix `[in_features, n_blades]` features.
    pub fn forward_features(&self, inputs: &[f32]) -> Vec<f32> {
        let n_blades = self.algebra.n_blades;
        assert_eq!(
            inputs.len(),
            self.in_features * n_blades,
            "input shape mismatch"
        );
        assert_eq!(self.action.len(), n_blades, "action shape mismatch");
        assert_eq!(
            self.weight.len(),
            self.out_features * self.in_features,
            "weight shape mismatch"
        );
        let transformed: Vec<Vec<f32>> = inputs
            .chunks_exact(n_blades)
            .map(|input| self.algebra.sandwich(&self.action, input))
            .collect();
        let mut output = vec![0.0; self.out_features * n_blades];
        for output_feature in 0..self.out_features {
            for (input_feature, transformed_input) in transformed.iter().enumerate() {
                let weight = self.weight[output_feature * self.in_features + input_feature];
                for blade in 0..n_blades {
                    output[output_feature * n_blades + blade] += weight * transformed_input[blade];
                }
            }
        }
        output
    }

    /// Return the number of learnable scalar parameters.
    pub fn parameter_count(&self) -> usize {
        self.action.len() + self.weight.len()
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
    fn scalar_kernel_is_identity() {
        let algebra = CliffordAlgebra::complex();
        let kernel = build_clifford_kernel(&[1.0, 0.0], 1, 1, 1, &algebra);
        assert_eq!(kernel, [1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn kernel_matches_direct_geometric_product() {
        let algebra = CliffordAlgebra::cl3();
        let weights: Vec<f32> = (0..8).map(|index| index as f32 * 0.1 - 0.2).collect();
        let input: Vec<f32> = (0..8).map(|index| (index as f32 * 0.3).sin()).collect();
        let kernel = build_clifford_kernel(&weights, 1, 1, 1, &algebra);
        let direct = algebra.geometric_product(&weights, &input);
        let mut expanded = [0.0; 8];
        for output_blade in 0..8 {
            for input_blade in 0..8 {
                expanded[output_blade] +=
                    kernel[output_blade * 8 + input_blade] * input[input_blade];
            }
        }
        for (actual, expected) in expanded.iter().zip(direct) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn interleave_roundtrip_is_identity() {
        let input: Vec<f32> = (0..24).map(|value| value as f32).collect();
        assert_eq!(
            deinterleave_blades(&interleave_blades(&input, 4, 3, 2), 4, 3, 2),
            input
        );
    }

    #[test]
    fn one_dimensional_identity_kernel_preserves_signal() {
        let mut layer = CliffordConv1d::new(CliffordAlgebra::cl3(), 1, 1, 1, 1, 0, 1, false);
        layer.weights.fill(0.0);
        layer.weights[0] = 1.0;
        let input: Vec<f32> = (0..40).map(|value| value as f32 * 0.01).collect();
        assert_eq!(layer.forward(&input, 1, 5), input);
    }

    #[test]
    fn grouped_convolution_uses_each_groups_weights() {
        let mut layer = CliffordConv1d::new(CliffordAlgebra::complex(), 2, 2, 1, 1, 0, 2, false);
        layer.weights.fill(0.0);
        layer.weights[0] = 1.0;
        layer.weights[1] = 2.0;
        let input = [1.0, 0.0, 3.0, 0.0];
        assert_eq!(layer.forward(&input, 1, 1), [1.0, 0.0, 6.0, 0.0]);
    }

    #[test]
    fn two_dimensional_output_shape_is_correct() {
        let layer = CliffordConv2d::new(
            CliffordAlgebra::complex(),
            2,
            4,
            (3, 3),
            (1, 1),
            (1, 1),
            1,
            false,
            true,
        );
        let output = layer.forward(&vec![0.1; 2 * 5 * 7 * 2], 1, 5, 7);
        assert_eq!(output.len(), 4 * 5 * 7 * 2);
        assert!(output.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn rotation_kernel_is_finite() {
        let kernel = build_rotation_kernel(&vec![0.1; 6 * 2 * 3 * 4], 2, 3, 4);
        assert_eq!(kernel.len(), 4 * 2 * 4 * 3 * 4);
        assert!(kernel.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn vector_silu_uses_one_gate_per_multivector() {
        let output = vector_silu(&[1.0, -1.0, 2.0, -2.0], 2, VectorSiLUMode::Sum, None);
        assert_eq!(output, [0.5, -0.5, 1.0, -1.0]);
    }

    #[test]
    fn conjugate_linear_identity_action_preserves_input() {
        let layer = CliffordConjugateLinear::new(CliffordAlgebra::cl3(), 1, 1);
        let input = [1.0, 0.5, -0.5, 0.25, 0.1, 0.2, 0.3, 0.4];
        assert_eq!(layer.forward(&input), input);
    }
}
