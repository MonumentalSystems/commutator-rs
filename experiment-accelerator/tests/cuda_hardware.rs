#![cfg(feature = "cuda")]

use std::collections::BTreeMap;

use experiment_accelerator::{
    authorize_reference, differential_check, independent_replica_work, AdapterError,
    AffineCpuBackend, AffineVectorWork, BackendDescriptor, Comparison, ComputeBackend,
    CudaAffineBackend, QualificationPolicy, ResultComparator, WorkerContext, GPU_API_LABEL,
    GPU_AVAILABLE_CAPABILITY,
};

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

#[test]
#[ignore = "requires an NVIDIA CUDA device and NVRTC; run with --ignored --nocapture"]
fn real_cuda_affine_backend_is_differentially_qualified() {
    let mut reference = AffineCpuBackend::try_new("affine-cpu", "kernel-v1").unwrap();
    let mut cuda = CudaAffineBackend::try_new(0, "affine-cuda", "kernel-v1").unwrap();
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
    let payload =
        AffineVectorWork::try_new(vec![-3.0, -0.5, 0.0, 0.25, 2.0, 1.0e6], 1.75, -0.125).unwrap();
    let work = independent_replica_work("cuda-affine", "hardware", 0, 1, 0, [(17, payload)])
        .unwrap()
        .remove(0);
    let authorization = authorize_reference(
        &|descriptor: &BackendDescriptor| descriptor.id() == "affine-cpu",
        reference.descriptor(),
    )
    .unwrap();
    let policy = QualificationPolicy::try_new("f64-affine", "v1", 1e-14, 1e-14).unwrap();
    let report = differential_check(
        &mut reference,
        &authorization,
        &mut cuda,
        &worker,
        &work,
        &policy,
        &F64Comparator,
    )
    .unwrap();
    assert!(report.comparison().accepted(), "{report:?}");
    eprintln!("validated CUDA device: {}", cuda.device_name());
}
