#![allow(unsafe_code)]

use std::collections::BTreeMap;
use std::sync::Arc;

use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::{
    canonical_digest, AdapterError, AffineVectorWork, BackendDescriptor, BackendIdentity,
    BackendKind, BackendOutput, ComputeBackend, Precision, GPU_API_LABEL, GPU_AVAILABLE_CAPABILITY,
};

const MODULE_NAME: &str = "experiment_accelerator_affine_f64";
const FUNCTION_NAME: &str = "affine_f64";
const ALGORITHM_VERSION: &str = "f64-affine-fma-v1";
const CUDARC_VERSION: &str = "0.13.9";
const CUDA_API_BINDINGS: &str = "12.8";
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

#[derive(Serialize)]
struct CudaExecutionIdentity<'a> {
    schema: &'static str,
    algorithm_version: &'static str,
    descriptor: &'a BackendDescriptor,
    module_name: &'static str,
    function_name: &'static str,
    kernel_source_sha256: [u8; 32],
    ptx_sha256: [u8; 32],
    cudarc_version: &'static str,
    cuda_api_bindings: &'static str,
    nvrtc_version: (i32, i32),
    driver_api_version: i32,
    device_ordinal: usize,
    device_name: &'a str,
    device_uuid: &'a str,
    compute_capability: (i32, i32),
}

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
    device_uuid: String,
    compute_capability: (i32, i32),
    driver_api_version: i32,
    nvrtc_version: (i32, i32),
    kernel_source_sha256: [u8; 32],
    ptx_sha256: [u8; 32],
}

impl std::fmt::Debug for CudaAffineBackend {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CudaAffineBackend")
            .field("descriptor", &self.descriptor)
            .field("device_ordinal", &self.device.ordinal())
            .field("device_name", &self.device_name)
            .field("device_uuid", &self.device_uuid)
            .field("compute_capability", &self.compute_capability)
            .field("driver_api_version", &self.driver_api_version)
            .field("nvrtc_version", &self.nvrtc_version)
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
        let device_uuid = device
            .uuid()
            .map(|uuid| {
                uuid.bytes
                    .iter()
                    .map(|byte| format!("{:02x}", byte.to_ne_bytes()[0]))
                    .collect::<String>()
            })
            .map_err(|error| CudaAffineError::Driver(error.to_string()))?;
        let compute_capability = compute_capability(&device)?;
        let driver_api_version = driver_api_version()?;
        let nvrtc_version = nvrtc_version()?;
        let kernel_source_sha256 = Sha256::digest(CUDA_SOURCE.as_bytes()).into();
        let ptx = compile_ptx(CUDA_SOURCE)
            .map_err(|error| CudaAffineError::Compilation(error.to_string()))?;
        let ptx_sha256 = Sha256::digest(ptx.to_src().as_bytes()).into();
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
            device_uuid,
            compute_capability,
            driver_api_version,
            nvrtc_version,
            kernel_source_sha256,
            ptx_sha256,
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

    /// Returns the CUDA driver-reported device UUID as lowercase hexadecimal.
    #[must_use]
    pub fn device_uuid(&self) -> &str {
        &self.device_uuid
    }

    /// Returns the CUDA compute capability `(major, minor)`.
    #[must_use]
    pub const fn compute_capability(&self) -> (i32, i32) {
        self.compute_capability
    }

    /// Returns the CUDA driver API version encoded as `1000 * major + 10 * minor`.
    #[must_use]
    pub const fn driver_api_version(&self) -> i32 {
        self.driver_api_version
    }

    /// Returns the runtime NVRTC `(major, minor)` version.
    #[must_use]
    pub const fn nvrtc_version(&self) -> (i32, i32) {
        self.nvrtc_version
    }

    /// Returns the SHA-256 digest of the embedded CUDA source bytes.
    #[must_use]
    pub const fn kernel_source_sha256(&self) -> [u8; 32] {
        self.kernel_source_sha256
    }

    /// Returns the SHA-256 digest of the NVRTC-generated PTX loaded by this backend.
    #[must_use]
    pub const fn ptx_sha256(&self) -> [u8; 32] {
        self.ptx_sha256
    }
}

impl BackendIdentity for CudaAffineBackend {
    fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }

    fn execution_fingerprint(&self) -> Result<[u8; 32], AdapterError> {
        canonical_digest(
            b"commutator.cuda-affine-execution.v1",
            &CudaExecutionIdentity {
                schema: "commutator.cuda-affine-execution.v1",
                algorithm_version: ALGORITHM_VERSION,
                descriptor: &self.descriptor,
                module_name: MODULE_NAME,
                function_name: FUNCTION_NAME,
                kernel_source_sha256: self.kernel_source_sha256,
                ptx_sha256: self.ptx_sha256,
                cudarc_version: CUDARC_VERSION,
                cuda_api_bindings: CUDA_API_BINDINGS,
                nvrtc_version: self.nvrtc_version,
                driver_api_version: self.driver_api_version,
                device_ordinal: self.device.ordinal(),
                device_name: &self.device_name,
                device_uuid: &self.device_uuid,
                compute_capability: self.compute_capability,
            },
        )
    }
}

impl ComputeBackend<AffineVectorWork, Vec<f64>> for CudaAffineBackend {
    type Error = CudaAffineError;

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

fn compute_capability(device: &CudaDevice) -> Result<(i32, i32), CudaAffineError> {
    use cudarc::driver::sys::CUdevice_attribute_enum::{
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
    };

    let major = device
        .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .map_err(|error| CudaAffineError::Driver(error.to_string()))?;
    let minor = device
        .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .map_err(|error| CudaAffineError::Driver(error.to_string()))?;
    Ok((major, minor))
}

fn driver_api_version() -> Result<i32, CudaAffineError> {
    let mut version = 0;
    // SAFETY: CUDA was initialized by `CudaDevice::new`; `version` is a valid,
    // writable `c_int` for the duration of this synchronous driver query.
    let status = unsafe { cudarc::driver::sys::lib().cuDriverGetVersion(&mut version) };
    if status == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        Ok(version)
    } else {
        Err(CudaAffineError::Driver(format!(
            "cuDriverGetVersion failed with {status:?}"
        )))
    }
}

fn nvrtc_version() -> Result<(i32, i32), CudaAffineError> {
    let mut major = 0;
    let mut minor = 0;
    // SAFETY: `major` and `minor` are valid writable `c_int` pointers for this
    // synchronous NVRTC query; loading NVRTC precedes kernel compilation.
    let status = unsafe { cudarc::nvrtc::sys::lib().nvrtcVersion(&mut major, &mut minor) };
    if status == cudarc::nvrtc::sys::nvrtcResult::NVRTC_SUCCESS {
        Ok((major, minor))
    } else {
        Err(CudaAffineError::Compilation(format!(
            "nvrtcVersion failed with {status:?}"
        )))
    }
}
