#![allow(unsafe_code)]

use std::collections::BTreeMap;
use std::sync::Arc;

use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

use crate::{
    AdapterError, AffineVectorWork, BackendDescriptor, BackendKind, BackendOutput, ComputeBackend,
    Precision, GPU_API_LABEL, GPU_AVAILABLE_CAPABILITY,
};

const MODULE_NAME: &str = "experiment_accelerator_affine_f64";
const FUNCTION_NAME: &str = "affine_f64";
const CUDA_SOURCE: &str = r#"
extern "C" __global__ void affine_f64(
    const double* input,
    double* output,
    double scale,
    double bias,
    unsigned long long length
) {
    unsigned long long index =
        (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length) {
        output[index] = fma(scale, input[index], bias);
    }
}
"#;

/// CUDA initialization, launch, transfer, or output validation failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CudaAffineError {
    /// CUDA driver initialization or device interaction failed.
    Driver(String),
    /// NVRTC rejected the embedded kernel.
    Compilation(String),
    /// The compiled kernel could not be retrieved.
    MissingKernel,
    /// The vector exceeds CUDA's checked one-dimensional launch bound.
    InputTooLarge(usize),
    /// CUDA returned a NaN or infinite output element.
    NonFiniteOutput {
        /// Index of the invalid output.
        index: usize,
    },
    /// Metrics violated the generic backend contract.
    InvalidMetrics(AdapterError),
    /// The CUDA backend descriptor violated the generic descriptor contract.
    InvalidDescriptor(AdapterError),
}

impl std::fmt::Display for CudaAffineError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Driver(message) => write!(formatter, "CUDA driver error: {message}"),
            Self::Compilation(message) => write!(formatter, "NVRTC compilation failed: {message}"),
            Self::MissingKernel => write!(formatter, "compiled CUDA affine kernel is missing"),
            Self::InputTooLarge(length) => {
                write!(
                    formatter,
                    "CUDA affine input length {length} exceeds u32::MAX"
                )
            }
            Self::NonFiniteOutput { index } => {
                write!(
                    formatter,
                    "CUDA affine output at index {index} is not finite"
                )
            }
            Self::InvalidMetrics(error) => write!(formatter, "invalid CUDA metrics: {error}"),
            Self::InvalidDescriptor(error) => {
                write!(formatter, "invalid CUDA backend descriptor: {error}")
            }
        }
    }
}

impl std::error::Error for CudaAffineError {}

/// Real f64 CUDA implementation of [`AffineVectorWork`].
///
/// Construction initializes a CUDA primary context and compiles the embedded
/// kernel with NVRTC. Execution transfers input to the selected device, launches
/// one checked kernel, synchronizes through the device-to-host copy, and rejects
/// non-finite output. The only unsafe operation is the cudarc kernel launch; its
/// safety invariants are documented at that call site.
pub struct CudaAffineBackend {
    descriptor: BackendDescriptor,
    device: Arc<CudaDevice>,
    device_name: String,
}

impl std::fmt::Debug for CudaAffineBackend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CudaAffineBackend")
            .field("descriptor", &self.descriptor)
            .field("device_ordinal", &self.device.ordinal())
            .field("device_name", &self.device_name)
            .finish_non_exhaustive()
    }
}

impl CudaAffineBackend {
    /// Initializes a device and compiles the f64 affine kernel.
    pub fn try_new(
        device_ordinal: usize,
        id: impl Into<String>,
        implementation_version: impl Into<String>,
    ) -> Result<Self, CudaAffineError> {
        let device = CudaDevice::new(device_ordinal)
            .map_err(|error| CudaAffineError::Driver(error.to_string()))?;
        let device_name = device
            .name()
            .map_err(|error| CudaAffineError::Driver(error.to_string()))?;
        let ptx = compile_ptx(CUDA_SOURCE)
            .map_err(|error| CudaAffineError::Compilation(error.to_string()))?;
        device
            .load_ptx(ptx, MODULE_NAME, &[FUNCTION_NAME])
            .map_err(|error| CudaAffineError::Driver(error.to_string()))?;
        let descriptor = BackendDescriptor::try_new(
            id,
            implementation_version,
            BackendKind::Gpu,
            Precision::F64,
            true,
            vec![crate::WorkerCapability {
                name: GPU_AVAILABLE_CAPABILITY.to_owned(),
                min_value: Some(1.0),
            }],
            BTreeMap::from([(GPU_API_LABEL.to_owned(), "cuda".to_owned())]),
        )
        .map_err(CudaAffineError::InvalidDescriptor)?;
        Ok(Self {
            descriptor,
            device,
            device_name,
        })
    }

    /// Returns the CUDA device ordinal.
    #[must_use]
    pub fn device_ordinal(&self) -> usize {
        self.device.ordinal()
    }

    /// Returns the CUDA driver-reported device name.
    #[must_use]
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Returns the CUDA compute capability `(major, minor)`.
    pub fn compute_capability(&self) -> Result<(i32, i32), CudaAffineError> {
        use cudarc::driver::sys::CUdevice_attribute_enum::{
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        };

        let major = self
            .device
            .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .map_err(|error| CudaAffineError::Driver(error.to_string()))?;
        let minor = self
            .device
            .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .map_err(|error| CudaAffineError::Driver(error.to_string()))?;
        Ok((major, minor))
    }
}

impl ComputeBackend<AffineVectorWork, Vec<f64>> for CudaAffineBackend {
    type Error = CudaAffineError;

    fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }

    fn execute(
        &mut self,
        payload: &AffineVectorWork,
        _seed: u64,
    ) -> Result<BackendOutput<Vec<f64>>, Self::Error> {
        let length = u32::try_from(payload.input().len())
            .map_err(|_| CudaAffineError::InputTooLarge(payload.input().len()))?;
        let input = self
            .device
            .htod_copy(payload.input().to_vec())
            .map_err(|error| CudaAffineError::Driver(error.to_string()))?;
        let mut output = self
            .device
            .alloc_zeros::<f64>(payload.input().len())
            .map_err(|error| CudaAffineError::Driver(error.to_string()))?;
        let function = self
            .device
            .get_func(MODULE_NAME, FUNCTION_NAME)
            .ok_or(CudaAffineError::MissingKernel)?;

        // SAFETY: `input` and `output` are live allocations from the same CUDA
        // context, both contain exactly `length` f64 elements, the embedded
        // kernel signature matches this tuple, and the launch grid is computed
        // from the same checked u32 length. The buffers outlive synchronization.
        unsafe {
            function.launch(
                LaunchConfig::for_num_elems(length),
                (
                    &input,
                    &mut output,
                    payload.scale(),
                    payload.bias(),
                    u64::from(length),
                ),
            )
        }
        .map_err(|error| CudaAffineError::Driver(error.to_string()))?;
        let output = self
            .device
            .dtoh_sync_copy(&output)
            .map_err(|error| CudaAffineError::Driver(error.to_string()))?;
        if let Some(index) = output.iter().position(|value| !value.is_finite()) {
            return Err(CudaAffineError::NonFiniteOutput { index });
        }
        BackendOutput::try_new(
            output,
            BTreeMap::from([
                ("element.count".to_owned(), f64::from(length)),
                ("device.ordinal".to_owned(), self.device.ordinal() as f64),
            ]),
        )
        .map_err(CudaAffineError::InvalidMetrics)
    }
}
