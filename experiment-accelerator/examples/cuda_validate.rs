use std::collections::BTreeMap;

use experiment_accelerator::{
    authorize_reference, differential_check, independent_replica_work, AdapterError,
    AffineCpuBackend, AffineVectorWork, BackendDescriptor, Comparison, ComputeBackend,
    CudaAffineBackend, QualificationPolicy, ResultComparator, WorkerContext, GPU_API_LABEL,
    GPU_AVAILABLE_CAPABILITY,
};
use serde_json::json;

struct F64Comparator;

impl ResultComparator<Vec<f64>> for F64Comparator {
    fn compare(
        &self,
        reference: &Vec<f64>,
        candidate: &Vec<f64>,
    ) -> Result<Comparison, AdapterError> {
        let mut absolute = 0.0_f64;
        let mut relative = 0.0_f64;
        for (&expected, &actual) in reference.iter().zip(candidate) {
            let difference = (actual - expected).abs();
            absolute = absolute.max(difference);
            relative = relative.max(difference / expected.abs().max(f64::MIN_POSITIVE));
        }
        Comparison::try_new(absolute, relative, reference.len() == candidate.len())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut reference = AffineCpuBackend::try_new("affine-cpu", "kernel-v1")?;
    let mut cuda = CudaAffineBackend::try_new(0, "affine-cuda", "kernel-v1")?;
    let mut worker = WorkerContext {
        worker_id: "cuda-hardware-validation".to_owned(),
        capabilities: BTreeMap::new(),
        labels: BTreeMap::new(),
    };
    worker
        .capabilities
        .insert(GPU_AVAILABLE_CAPABILITY.to_owned(), 1.0);
    worker
        .labels
        .insert(GPU_API_LABEL.to_owned(), "cuda".to_owned());
    let payload = AffineVectorWork::try_new(vec![-3.0, -0.5, 0.0, 0.25, 2.0, 1.0e6], 1.75, -0.125)?;
    let work =
        independent_replica_work("cuda-affine", "hardware", 0, 1, 0, [(17, payload)])?.remove(0);
    let authorization = authorize_reference(
        &|descriptor: &BackendDescriptor| descriptor.id() == "affine-cpu",
        reference.descriptor(),
    )?;
    let policy = QualificationPolicy::try_new("f64-affine", "v1", 1e-14, 1e-14)?;
    let report = differential_check(
        &mut reference,
        &authorization,
        &mut cuda,
        &worker,
        &work,
        &policy,
        &F64Comparator,
    )
    .map_err(|error| format!("differential validation failed: {error:?}"))?;
    let audit = report.audit_snapshot();
    let (compute_major, compute_minor) = cuda.compute_capability()?;
    let evidence = json!({
        "schema": "commutator.cuda-validation.v1",
        "cudarc_version": "0.13.9",
        "cuda_api_bindings": "12.8",
        "gpu_name": cuda.device_name(),
        "gpu_ordinal": cuda.device_ordinal(),
        "compute_capability": format!("{compute_major}.{compute_minor}"),
        "nvidia_driver": std::env::var("NVIDIA_DRIVER_VERSION").unwrap_or_else(|_| "not-recorded".to_owned()),
        "backend": cuda.descriptor(),
        "policy": policy,
        "qualification": audit,
    });
    println!("{}", serde_json::to_string_pretty(&evidence)?);
    if !report.comparison().accepted() {
        return Err("CUDA backend did not satisfy its qualification policy".into());
    }
    Ok(())
}
