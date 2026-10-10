use std::collections::BTreeMap;

use experiment_accelerator::{
    authorize_reference, differential_check, execute_work_unit, independent_replica_work,
    AdapterError, AffineCpuBackend, AffineVectorWork, BackendDescriptor, BackendIdentity,
    BackendKind, BackendOutput, Comparison, ComputeBackend, ExecutionAuthorization, ExecutionError,
    Precision, QualificationPolicy, ResultComparator, ShardedError, ThreadedShardedBackend,
    WorkerContext,
};

struct ExactComparator;

impl ResultComparator<Vec<i64>> for ExactComparator {
    fn compare(
        &self,
        reference: &Vec<i64>,
        candidate: &Vec<i64>,
    ) -> Result<Comparison, AdapterError> {
        let mismatch = reference != candidate;
        Comparison::try_new(f64::from(mismatch), f64::from(mismatch), !mismatch)
    }
}

#[derive(Debug)]
struct MapBackend {
    descriptor: BackendDescriptor,
    factor: i64,
    fail_on: Option<i64>,
    panic_on: Option<i64>,
    execution_tag: u8,
}

impl MapBackend {
    fn new(id: &str, kind: BackendKind, factor: i64) -> Self {
        Self {
            descriptor: BackendDescriptor::try_new(
                id,
                "test-v1",
                kind,
                Precision::F64,
                true,
                Vec::new(),
                BTreeMap::new(),
            )
            .unwrap(),
            factor,
            fail_on: None,
            panic_on: None,
            execution_tag: factor as u8,
        }
    }

    fn configured(
        id: &str,
        precision: Precision,
        deterministic: bool,
        labels: BTreeMap<String, String>,
    ) -> Self {
        Self {
            descriptor: BackendDescriptor::try_new(
                id,
                "test-v1",
                BackendKind::CpuAccelerated,
                precision,
                deterministic,
                Vec::new(),
                labels,
            )
            .unwrap(),
            factor: 1,
            fail_on: None,
            panic_on: None,
            execution_tag: 1,
        }
    }
}

impl BackendIdentity for MapBackend {
    fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }

    fn execution_fingerprint(&self) -> Result<[u8; 32], AdapterError> {
        Ok([self.execution_tag; 32])
    }
}

impl ComputeBackend<Vec<i64>, Vec<i64>> for MapBackend {
    type Error = &'static str;

    fn execute(
        &mut self,
        payload: &Vec<i64>,
        _seed: u64,
    ) -> Result<BackendOutput<Vec<i64>>, Self::Error> {
        if self
            .panic_on
            .is_some_and(|needle| payload.contains(&needle))
        {
            panic!("injected child panic");
        }
        if self.fail_on.is_some_and(|needle| payload.contains(&needle)) {
            return Err("injected child failure");
        }
        BackendOutput::try_new(
            payload.iter().map(|value| value * self.factor).collect(),
            BTreeMap::from([("child.items".to_owned(), payload.len() as f64)]),
        )
        .map_err(|_| "invalid output")
    }
}

#[test]
fn threaded_shards_drain_later_panics_after_an_earlier_error() {
    let mut children = vec![
        MapBackend::new("child-0", BackendKind::CpuAccelerated, 1),
        MapBackend::new("child-1", BackendKind::CpuAccelerated, 1),
    ];
    children[0].fail_on = Some(0);
    children[1].panic_on = Some(3);
    let mut backend = ThreadedShardedBackend::try_new("threaded-map", "test-v1", children).unwrap();
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        backend.execute(&(0..6).collect(), 1)
    }));
    assert!(caught.is_ok(), "a later child panic escaped the adapter");
    assert_eq!(
        caught.unwrap(),
        Err(ShardedError::Child {
            shard: 0,
            source: "injected child failure",
        })
    );
}

#[test]
fn cpu_affine_reference_is_checked() {
    let work = AffineVectorWork::try_new(vec![-2.0, 0.5, 4.0], 1.5, -0.25).unwrap();
    let mut backend = AffineCpuBackend::try_new("affine-reference", "test-v1").unwrap();
    let output = backend.execute(&work, 7).unwrap();
    assert_eq!(output.payload(), &[-3.25, 0.5, 5.75]);
    assert!(AffineVectorWork::try_new(vec![f64::NAN], 1.0, 0.0).is_err());
}

#[test]
fn committed_cuda_evidence_separates_raw_inputs_from_commitments() {
    let evidence: serde_json::Value =
        serde_json::from_str(include_str!("../evidence/cuda-gb10-driver-580.173.02.json")).unwrap();
    assert_eq!(evidence["schema"], "commutator.cuda-validation.v2");
    assert_eq!(evidence["provenance"]["authenticated"], false);
    assert_eq!(
        evidence["raw_recomputation_inputs"]["candidate_execution_fingerprint_sha256"],
        evidence["qualification_commitments"]["candidate_execution_fingerprint_sha256"]
    );
    assert_eq!(
        evidence["raw_recomputation_inputs"]["reference_execution_fingerprint_sha256"],
        evidence["qualification_commitments"]["reference_execution_fingerprint_sha256"]
    );
    assert!(evidence["raw_recomputation_inputs"]["work_unit"]["payload"]["input"].is_array());
    assert_eq!(
        evidence["qualification_commitments"]["schema"],
        "commutator.accelerator-qualification.v2"
    );
}

#[test]
fn threaded_shards_preserve_order_and_namespace_metrics() {
    let children = (0..3)
        .map(|index| MapBackend::new(&format!("child-{index}"), BackendKind::CpuAccelerated, 3))
        .collect();
    let mut backend = ThreadedShardedBackend::try_new("threaded-map", "test-v1", children).unwrap();
    let output = backend.execute(&(0..10).collect(), 91).unwrap();
    assert_eq!(
        output.payload(),
        &(0..10).map(|value| value * 3).collect::<Vec<_>>()
    );
    assert_eq!(output.metrics()["shard.count"], 3.0);
    assert_eq!(output.metrics()["shard.0.child.items"], 4.0);
    assert_eq!(output.metrics()["shard.2.child.items"], 2.0);
}

#[test]
fn sharded_descriptor_is_derived_from_children() {
    let children = vec![
        MapBackend::configured("f32-child", Precision::F32, true, BTreeMap::new()),
        MapBackend::configured("f64-child", Precision::F64, false, BTreeMap::new()),
    ];
    let backend: ThreadedShardedBackend<_, i64, i64> =
        ThreadedShardedBackend::try_new("mixed-proxy", "test-v1", children).unwrap();
    assert_eq!(backend.descriptor().precision(), Precision::Mixed);
    assert!(!backend.descriptor().deterministic());

    let conflicting = vec![
        MapBackend::configured(
            "linux-child",
            Precision::F64,
            true,
            BTreeMap::from([("host.os".to_owned(), "linux".to_owned())]),
        ),
        MapBackend::configured(
            "windows-child",
            Precision::F64,
            true,
            BTreeMap::from([("host.os".to_owned(), "windows".to_owned())]),
        ),
    ];
    assert!(matches!(
        ThreadedShardedBackend::<_, i64, i64>::try_new("impossible-proxy", "test-v1", conflicting,),
        Err(AdapterError::InvalidCapability)
    ));
}

#[test]
fn threaded_shards_report_the_stable_failing_ordinal() {
    let mut children = (0..3)
        .map(|index| MapBackend::new(&format!("child-{index}"), BackendKind::CpuAccelerated, 1))
        .collect::<Vec<_>>();
    children[1].fail_on = Some(5);
    let mut backend = ThreadedShardedBackend::try_new("threaded-map", "test-v1", children).unwrap();
    assert_eq!(
        backend.execute(&(0..9).collect(), 1),
        Err(ShardedError::Child {
            shard: 1,
            source: "injected child failure",
        })
    );
}

#[test]
fn distributed_adapter_is_differentially_qualifiable() {
    let mut reference = MapBackend::new("map-reference", BackendKind::CpuReference, 2);
    let children = (0..3)
        .map(|index| MapBackend::new(&format!("child-{index}"), BackendKind::CpuAccelerated, 2))
        .collect();
    let mut distributed =
        ThreadedShardedBackend::try_new("threaded-map", "test-v1", children).unwrap();
    let worker = WorkerContext {
        worker_id: "host-thread-pool".to_owned(),
        ..WorkerContext::default()
    };
    let work = independent_replica_work("sharded", "run", 0, 1, 0, [(77, (0..17).collect())])
        .unwrap()
        .remove(0);
    let reference_fingerprint = reference.execution_fingerprint().unwrap();
    let reference_authorization = authorize_reference(
        &|descriptor: &BackendDescriptor, fingerprint: &[u8; 32]| {
            descriptor.id() == "map-reference" && *fingerprint == reference_fingerprint
        },
        &reference,
    )
    .unwrap();
    let policy = QualificationPolicy::try_new("exact-integers", "v1", 0.0, 0.0).unwrap();
    let report = differential_check(
        &mut reference,
        &reference_authorization,
        &mut distributed,
        &worker,
        &work,
        &policy,
        &ExactComparator,
    )
    .unwrap();
    assert!(report.comparison().accepted());
    let result = execute_work_unit(
        &mut distributed,
        &worker,
        &work,
        1,
        ExecutionAuthorization::Differential {
            report: &report,
            policy: &policy,
        },
    )
    .unwrap();
    assert_eq!(
        result.payload,
        (0..17).map(|value| value * 2).collect::<Vec<_>>()
    );

    let mut changed_children = (0..3)
        .map(|index| MapBackend::new(&format!("child-{index}"), BackendKind::CpuAccelerated, 2))
        .collect::<Vec<_>>();
    changed_children[1].execution_tag = 99;
    let mut changed_distributed =
        ThreadedShardedBackend::try_new("threaded-map", "test-v1", changed_children).unwrap();
    assert_eq!(changed_distributed.descriptor(), distributed.descriptor());
    assert_ne!(
        changed_distributed.execution_fingerprint().unwrap(),
        distributed.execution_fingerprint().unwrap()
    );
    assert!(matches!(
        execute_work_unit(
            &mut changed_distributed,
            &worker,
            &work,
            1,
            ExecutionAuthorization::Differential {
                report: &report,
                policy: &policy,
            },
        ),
        Err(ExecutionError::UnqualifiedBackend)
    ));

    let changed_policy = QualificationPolicy::try_new("exact-integers", "v2", 0.0, 0.0).unwrap();
    assert!(matches!(
        execute_work_unit(
            &mut distributed,
            &worker,
            &work,
            1,
            ExecutionAuthorization::Differential {
                report: &report,
                policy: &changed_policy,
            },
        ),
        Err(ExecutionError::UnqualifiedBackend)
    ));
}
