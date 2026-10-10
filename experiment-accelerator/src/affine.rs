use std::collections::BTreeMap;

use serde::Serialize;

use crate::{
    AdapterError, BackendDescriptor, BackendKind, BackendOutput, ComputeBackend, Precision,
};

/// Checked f64 vector-affine input, `output[i] = scale * input[i] + bias`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AffineVectorWork {
    input: Vec<f64>,
    scale: f64,
    bias: f64,
}

impl AffineVectorWork {
    /// Constructs nonempty finite affine work.
    pub fn try_new(input: Vec<f64>, scale: f64, bias: f64) -> Result<Self, AffineError> {
        if input.is_empty() {
            return Err(AffineError::EmptyInput);
        }
        if let Some(index) = input.iter().position(|value| !value.is_finite()) {
            return Err(AffineError::NonFiniteInput { index });
        }
        if !scale.is_finite() {
            return Err(AffineError::NonFiniteParameter("scale"));
        }
        if !bias.is_finite() {
            return Err(AffineError::NonFiniteParameter("bias"));
        }
        Ok(Self { input, scale, bias })
    }

    /// Returns the input vector.
    #[must_use]
    pub fn input(&self) -> &[f64] {
        &self.input
    }

    /// Returns the multiplicative coefficient.
    #[must_use]
    pub const fn scale(&self) -> f64 {
        self.scale
    }

    /// Returns the additive coefficient.
    #[must_use]
    pub const fn bias(&self) -> f64 {
        self.bias
    }
}

/// Invalid affine input or a non-finite computed result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AffineError {
    /// At least one vector element is required.
    EmptyInput,
    /// An input element was NaN or infinite.
    NonFiniteInput {
        /// Index of the invalid element.
        index: usize,
    },
    /// A named scalar parameter was NaN or infinite.
    NonFiniteParameter(&'static str),
    /// A computed output element was NaN or infinite.
    NonFiniteOutput {
        /// Index of the invalid element.
        index: usize,
    },
}

impl std::fmt::Display for AffineError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyInput => write!(formatter, "affine input must not be empty"),
            Self::NonFiniteInput { index } => {
                write!(formatter, "affine input at index {index} is not finite")
            }
            Self::NonFiniteParameter(name) => {
                write!(formatter, "affine parameter {name} is not finite")
            }
            Self::NonFiniteOutput { index } => {
                write!(formatter, "affine output at index {index} is not finite")
            }
        }
    }
}

impl std::error::Error for AffineError {}

/// Portable f64 reference implementation for [`AffineVectorWork`].
#[derive(Clone, Debug)]
pub struct AffineCpuBackend {
    descriptor: BackendDescriptor,
}

impl AffineCpuBackend {
    /// Constructs a deterministic CPU-reference backend descriptor.
    pub fn try_new(
        id: impl Into<String>,
        implementation_version: impl Into<String>,
    ) -> Result<Self, AdapterError> {
        Ok(Self {
            descriptor: BackendDescriptor::try_new(
                id,
                implementation_version,
                BackendKind::CpuReference,
                Precision::F64,
                true,
                Vec::new(),
                BTreeMap::new(),
            )?,
        })
    }
}

impl ComputeBackend<AffineVectorWork, Vec<f64>> for AffineCpuBackend {
    type Error = AffineError;

    fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }

    fn execute(
        &mut self,
        payload: &AffineVectorWork,
        _seed: u64,
    ) -> Result<BackendOutput<Vec<f64>>, Self::Error> {
        let output: Vec<_> = payload
            .input
            .iter()
            .map(|value| payload.scale.mul_add(*value, payload.bias))
            .collect();
        if let Some(index) = output.iter().position(|value| !value.is_finite()) {
            return Err(AffineError::NonFiniteOutput { index });
        }
        BackendOutput::try_new(output, BTreeMap::new()).map_err(|_| AffineError::NonFiniteOutput {
            index: payload.input.len(),
        })
    }
}
