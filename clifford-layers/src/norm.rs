//! Blade-covariance normalization via Cholesky whitening.
//!
//! These layers operate in coefficient space and are intended for readout or
//! post-processing paths, not norm-preserving manifold dynamics.

fn cholesky(matrix: &[f32], dimension: usize) -> Option<Vec<f32>> {
    let mut lower = vec![0.0; dimension * dimension];
    for row in 0..dimension {
        for column in 0..=row {
            let product_sum = (0..column)
                .map(|inner| lower[row * dimension + inner] * lower[column * dimension + inner])
                .sum::<f32>();
            if row == column {
                let diagonal = matrix[row * dimension + row] - product_sum;
                if diagonal <= 0.0 || !diagonal.is_finite() {
                    return None;
                }
                lower[row * dimension + column] = diagonal.sqrt();
            } else {
                lower[row * dimension + column] = (matrix[row * dimension + column] - product_sum)
                    / lower[column * dimension + column];
            }
        }
    }
    Some(lower)
}

fn forward_solve(lower: &[f32], values: &[f32], dimension: usize) -> Vec<f32> {
    let mut solution = vec![0.0; dimension];
    for row in 0..dimension {
        let product_sum = (0..row)
            .map(|column| lower[row * dimension + column] * solution[column])
            .sum::<f32>();
        solution[row] = (values[row] - product_sum) / lower[row * dimension + row];
    }
    solution
}

fn moments(input: &[f32], n_blades: usize) -> (Vec<f32>, Vec<f32>) {
    let sample_count = input.len() / n_blades;
    let inverse_count = 1.0 / sample_count as f32;
    let mut mean = vec![0.0; n_blades];
    for sample in input.chunks_exact(n_blades) {
        for (target, value) in mean.iter_mut().zip(sample) {
            *target += value;
        }
    }
    for value in &mut mean {
        *value *= inverse_count;
    }

    let mut covariance = vec![0.0; n_blades * n_blades];
    for sample in input.chunks_exact(n_blades) {
        for row in 0..n_blades {
            let row_delta = sample[row] - mean[row];
            for column in 0..n_blades {
                covariance[row * n_blades + column] += row_delta * (sample[column] - mean[column]);
            }
        }
    }
    for value in &mut covariance {
        *value *= inverse_count;
    }
    (mean, covariance)
}

/// Whitens blade components using statistics from the current sample group.
#[derive(Debug, Clone)]
pub struct CliffordGroupNorm {
    /// Number of blade components in one multivector.
    pub n_blades: usize,
    /// Learnable scale for each blade.
    pub gamma: Vec<f32>,
    /// Learnable offset for each blade.
    pub beta: Vec<f32>,
    /// Positive diagonal regularizer for the covariance matrix.
    pub eps: f32,
}

impl CliffordGroupNorm {
    /// Construct a normalization layer.
    ///
    /// # Panics
    ///
    /// Panics when `n_blades` is zero or `eps` is not positive and finite.
    pub fn new(n_blades: usize, eps: f32) -> Self {
        assert!(n_blades > 0, "n_blades must be nonzero");
        assert!(
            eps.is_finite() && eps > 0.0,
            "eps must be positive and finite"
        );
        Self {
            n_blades,
            gamma: vec![1.0; n_blades],
            beta: vec![0.0; n_blades],
            eps,
        }
    }

    /// Normalize a `[samples, n_blades]` buffer.
    ///
    /// Groups with zero or one sample receive only the affine transform.
    ///
    /// # Panics
    ///
    /// Panics when the input or affine parameter lengths are invalid.
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len() % self.n_blades, 0, "input shape mismatch");
        assert_eq!(self.gamma.len(), self.n_blades, "gamma shape mismatch");
        assert_eq!(self.beta.len(), self.n_blades, "beta shape mismatch");
        let sample_count = input.len() / self.n_blades;
        if sample_count <= 1 {
            return input
                .iter()
                .enumerate()
                .map(|(index, value)| {
                    value * self.gamma[index % self.n_blades] + self.beta[index % self.n_blades]
                })
                .collect();
        }

        let (mean, mut covariance) = moments(input, self.n_blades);
        for blade in 0..self.n_blades {
            covariance[blade * self.n_blades + blade] += self.eps;
        }
        let Some(lower) = cholesky(&covariance, self.n_blades) else {
            return self.diagonal_fallback(input, &mean, &covariance);
        };

        let mut output = Vec::with_capacity(input.len());
        for sample in input.chunks_exact(self.n_blades) {
            let centered: Vec<f32> = sample
                .iter()
                .zip(&mean)
                .map(|(value, mean)| value - mean)
                .collect();
            let whitened = forward_solve(&lower, &centered, self.n_blades);
            output.extend(
                whitened
                    .iter()
                    .enumerate()
                    .map(|(blade, value)| value * self.gamma[blade] + self.beta[blade]),
            );
        }
        output
    }

    fn diagonal_fallback(&self, input: &[f32], mean: &[f32], covariance: &[f32]) -> Vec<f32> {
        input
            .chunks_exact(self.n_blades)
            .flat_map(|sample| {
                sample.iter().enumerate().map(|(blade, value)| {
                    let standard_deviation = covariance[blade * self.n_blades + blade]
                        .max(self.eps)
                        .sqrt();
                    (value - mean[blade]) / standard_deviation * self.gamma[blade]
                        + self.beta[blade]
                })
            })
            .collect()
    }

    /// Return the number of learnable affine parameters.
    pub fn parameter_count(&self) -> usize {
        self.gamma.len() + self.beta.len()
    }

    /// Backward-compatible alias for [`Self::parameter_count`].
    pub fn n_params(&self) -> usize {
        self.parameter_count()
    }
}

/// Blade-covariance normalization with running inference statistics.
#[derive(Debug, Clone)]
pub struct CliffordBatchNorm {
    /// Affine parameters and current-batch whitening behavior.
    pub group_norm: CliffordGroupNorm,
    /// Exponential moving average of each blade mean.
    pub running_mean: Vec<f32>,
    /// Exponential moving average of the blade covariance matrix.
    pub running_cov: Vec<f32>,
    /// Weight assigned to the newest batch statistics.
    pub momentum: f32,
    /// Whether a training batch has initialized the running statistics.
    pub has_stats: bool,
}

impl CliffordBatchNorm {
    /// Construct a batch-normalization layer.
    ///
    /// # Panics
    ///
    /// Panics when `momentum` is not finite or lies outside `[0, 1]`, or when
    /// the underlying group-normalization arguments are invalid.
    pub fn new(n_blades: usize, eps: f32, momentum: f32) -> Self {
        assert!(
            momentum.is_finite() && (0.0..=1.0).contains(&momentum),
            "momentum must lie in [0, 1]"
        );
        let mut running_cov = vec![0.0; n_blades * n_blades];
        for blade in 0..n_blades {
            running_cov[blade * n_blades + blade] = 1.0;
        }
        Self {
            group_norm: CliffordGroupNorm::new(n_blades, eps),
            running_mean: vec![0.0; n_blades],
            running_cov,
            momentum,
            has_stats: false,
        }
    }

    fn validate_public_shapes(&self) {
        let n_blades = self.group_norm.n_blades;
        assert!(n_blades > 0, "n_blades must be nonzero");
        assert_eq!(
            self.group_norm.gamma.len(),
            n_blades,
            "gamma shape mismatch"
        );
        assert_eq!(self.group_norm.beta.len(), n_blades, "beta shape mismatch");
        assert_eq!(
            self.running_mean.len(),
            n_blades,
            "running mean shape mismatch"
        );
        assert_eq!(
            self.running_cov.len(),
            n_blades * n_blades,
            "running covariance shape mismatch"
        );
    }

    /// Normalize with current-batch statistics and update running statistics.
    pub fn forward_train(&mut self, input: &[f32]) -> Vec<f32> {
        self.validate_public_shapes();
        let output = self.group_norm.forward(input);
        let n_blades = self.group_norm.n_blades;
        let sample_count = input.len() / n_blades;
        if sample_count > 1 {
            let (batch_mean, batch_covariance) = moments(input, n_blades);
            if self.has_stats {
                for (running, batch) in self.running_mean.iter_mut().zip(batch_mean) {
                    *running = (1.0 - self.momentum) * *running + self.momentum * batch;
                }
                for (running, batch) in self.running_cov.iter_mut().zip(batch_covariance) {
                    *running = (1.0 - self.momentum) * *running + self.momentum * batch;
                }
            } else {
                self.running_mean = batch_mean;
                self.running_cov = batch_covariance;
                self.has_stats = true;
            }
        }
        output
    }

    /// Normalize using only the stored running statistics.
    ///
    /// Before the first training batch, the initial zero mean and identity
    /// covariance are used. If the stored covariance is not positive definite,
    /// evaluation falls back to its stored diagonal rather than recomputing
    /// statistics from `input`.
    ///
    /// # Panics
    ///
    /// Panics when the input or any public running/affine buffer has an invalid
    /// shape.
    pub fn forward_eval(&self, input: &[f32]) -> Vec<f32> {
        self.validate_public_shapes();
        let n_blades = self.group_norm.n_blades;
        assert_eq!(input.len() % n_blades, 0, "input shape mismatch");
        let mut covariance = self.running_cov.clone();
        for blade in 0..n_blades {
            covariance[blade * n_blades + blade] += self.group_norm.eps;
        }
        let lower = cholesky(&covariance, n_blades);
        input
            .chunks_exact(n_blades)
            .flat_map(|sample| {
                let centered: Vec<f32> = sample
                    .iter()
                    .zip(&self.running_mean)
                    .map(|(value, mean)| value - mean)
                    .collect();
                let normalized = lower.as_ref().map_or_else(
                    || {
                        centered
                            .iter()
                            .enumerate()
                            .map(|(blade, value)| {
                                let variance = self.running_cov[blade * n_blades + blade].max(0.0)
                                    + self.group_norm.eps;
                                value / variance.sqrt()
                            })
                            .collect()
                    },
                    |lower| forward_solve(lower, &centered, n_blades),
                );
                normalized
                    .into_iter()
                    .enumerate()
                    .map(|(blade, value)| {
                        value * self.group_norm.gamma[blade] + self.group_norm.beta[blade]
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Return the number of learnable affine parameters.
    pub fn parameter_count(&self) -> usize {
        self.group_norm.parameter_count()
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
    fn cholesky_of_identity_is_identity() {
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert_eq!(cholesky(&identity, 3).unwrap(), identity);
    }

    #[test]
    fn group_norm_centers_each_blade() {
        let norm = CliffordGroupNorm::new(4, 1e-5);
        let input: Vec<f32> = (0..40).map(|i| (i as f32 * 0.3).sin() + 2.0).collect();
        let output = norm.forward(&input);
        for blade in 0..4 {
            let mean = output
                .chunks_exact(4)
                .map(|sample| sample[blade])
                .sum::<f32>()
                / 10.0;
            assert!(mean.abs() < 0.1, "blade {blade} mean was {mean}");
        }
    }

    #[test]
    fn degenerate_group_is_finite() {
        let output = CliffordGroupNorm::new(4, 1e-5).forward(&[0.0; 40]);
        assert!(output.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn batch_norm_accumulates_and_uses_statistics() {
        let mut norm = CliffordBatchNorm::new(4, 1e-5, 0.1);
        for batch in 0..5 {
            let input: Vec<f32> = (0..40)
                .map(|i| (i as f32 * 0.3 + batch as f32).sin())
                .collect();
            norm.forward_train(&input);
        }
        assert!(norm.has_stats);
        assert!(norm
            .forward_eval(&(0..40).map(|i| (i as f32 * 0.3).sin()).collect::<Vec<_>>())
            .iter()
            .all(|value| value.is_finite()));
    }

    #[test]
    fn batch_norm_eval_before_training_uses_initial_running_statistics() {
        let norm = CliffordBatchNorm::new(2, 1e-5, 0.1);
        let output = norm.forward_eval(&[5.0, 7.0, 5.0, 7.0]);
        assert!(output[0] > 4.9);
        assert!(output[1] > 6.9);
    }

    #[test]
    fn batch_norm_eval_fallback_uses_stored_diagonal() {
        let mut norm = CliffordBatchNorm::new(2, 1e-6, 0.1);
        norm.has_stats = true;
        norm.running_mean = vec![1.0, 2.0];
        norm.running_cov = vec![4.0, 100.0, 100.0, 9.0];
        let output = norm.forward_eval(&[5.0, 8.0]);
        assert!((output[0] - 2.0).abs() < 1e-4);
        assert!((output[1] - 2.0).abs() < 1e-4);
    }

    #[test]
    #[should_panic(expected = "running mean shape mismatch")]
    fn batch_norm_rejects_invalid_running_mean_shape() {
        let mut norm = CliffordBatchNorm::new(2, 1e-5, 0.1);
        norm.running_mean.clear();
        norm.forward_eval(&[0.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "running covariance shape mismatch")]
    fn batch_norm_rejects_invalid_running_covariance_shape() {
        let mut norm = CliffordBatchNorm::new(2, 1e-5, 0.1);
        norm.running_cov.clear();
        norm.forward_eval(&[0.0, 0.0]);
    }

    #[test]
    fn parameter_count_is_affine_size() {
        assert_eq!(CliffordGroupNorm::new(8, 1e-5).parameter_count(), 16);
    }
}
