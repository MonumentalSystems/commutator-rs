//! Validated accelerator and distributed-work adapters for `experiment-core`.

#![deny(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

mod affine;
#[cfg(feature = "cuda")]
mod cuda;
mod sharded;

use core::fmt;
use std::collections::BTreeMap;

pub use experiment_core::{RunMetadata, WorkResult, WorkUnit, WorkerCapability, WorkerContext};
use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};

pub use affine::{AffineCpuBackend, AffineError, AffineVectorWork};
#[cfg(feature = "cuda")]
pub use cuda::{CudaAffineBackend, CudaAffineError};
pub use sharded::{shard_seed, ShardedError, ThreadedShardedBackend};

/// Errors returned while constructing or applying experiment adapters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdapterError {
    /// A required identifier was empty.
    EmptyIdentifier(&'static str),
    /// A capability requirement contained an empty name or non-finite value.
    InvalidCapability,
    /// A reported metric was non-finite.
    NonFiniteMetric(String),
    /// A comparison tolerance was non-finite or negative.
    InvalidTolerance,
    /// A chunk size was zero.
    ZeroChunkSize,
    /// A requested allocation or replica ordinal was not representable.
    SizeOverflow,
    /// A GPU backend omitted the canonical availability capability or API label.
    MissingGpuRequirement,
    /// The host did not authorize a backend as a trusted CPU reference.
    UnauthorizedReference,
    /// A worker identity was empty.
    EmptyWorkerIdentity,
    /// Generated execution provenance would overwrite an existing attribute.
    ProvenanceConflict(String),
    /// A backend changed its descriptor or execution fingerprint during qualification.
    UnstableExecutionFingerprint,
}

impl fmt::Display for AdapterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyIdentifier(name) => write!(formatter, "{name} must not be empty"),
            Self::InvalidCapability => write!(formatter, "invalid backend capability requirement"),
            Self::NonFiniteMetric(name) => write!(formatter, "metric {name} is not finite"),
            Self::InvalidTolerance => write!(
                formatter,
                "comparison tolerance must be finite and nonnegative"
            ),
            Self::ZeroChunkSize => write!(formatter, "chunk size must be nonzero"),
            Self::SizeOverflow => write!(formatter, "requested size or ordinal overflows"),
            Self::MissingGpuRequirement => {
                write!(
                    formatter,
                    "GPU backends require gpu.available >= 1 and a gpu.api label"
                )
            }
            Self::UnauthorizedReference => {
                write!(
                    formatter,
                    "backend is not authorized as a trusted CPU reference"
                )
            }
            Self::EmptyWorkerIdentity => write!(formatter, "worker id must not be empty"),
            Self::ProvenanceConflict(name) => {
                write!(
                    formatter,
                    "execution provenance attribute {name} already exists"
                )
            }
            Self::UnstableExecutionFingerprint => {
                write!(
                    formatter,
                    "backend execution identity changed during qualification"
                )
            }
        }
    }
}

/// Canonical numeric capability proving that a worker has a usable GPU.
pub const GPU_AVAILABLE_CAPABILITY: &str = "gpu.available";

/// Canonical worker label selecting the GPU programming API (for example,
/// `cuda`, `metal`, or `webgpu`).
pub const GPU_API_LABEL: &str = "gpu.api";

impl std::error::Error for AdapterError {}

/// Broad execution family used for scheduling and provenance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    /// Portable scalar or reference CPU implementation.
    CpuReference,
    /// Parallel CPU or platform-accelerated implementation.
    CpuAccelerated,
    /// A discrete or integrated GPU implementation.
    Gpu,
    /// A distributed child runtime that may fan work out further.
    DistributedProxy,
}

/// Arithmetic precision used by a backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    /// IEEE binary32 arithmetic.
    F32,
    /// IEEE binary64 arithmetic.
    F64,
    /// Backend-defined mixture of binary32 and binary64 operations.
    Mixed,
}

impl Precision {
    fn tag(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::Mixed => "mixed",
        }
    }
}

/// Serializable backend identity and scheduling requirements.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BackendDescriptor {
    id: String,
    implementation_version: String,
    kind: BackendKind,
    precision: Precision,
    deterministic: bool,
    requirements: Vec<WorkerCapability>,
    required_labels: BTreeMap<String, String>,
}

impl BackendDescriptor {
    /// Constructs a checked descriptor.
    pub fn try_new(
        id: impl Into<String>,
        implementation_version: impl Into<String>,
        kind: BackendKind,
        precision: Precision,
        deterministic: bool,
        requirements: Vec<WorkerCapability>,
        required_labels: BTreeMap<String, String>,
    ) -> Result<Self, AdapterError> {
        let id = id.into();
        let implementation_version = implementation_version.into();
        if id.trim().is_empty() {
            return Err(AdapterError::EmptyIdentifier("backend id"));
        }
        if implementation_version.trim().is_empty() {
            return Err(AdapterError::EmptyIdentifier("implementation version"));
        }
        if requirements.iter().any(|requirement| {
            requirement.name.trim().is_empty()
                || requirement
                    .min_value
                    .is_some_and(|value| !value.is_finite())
        }) {
            return Err(AdapterError::InvalidCapability);
        }
        if required_labels
            .iter()
            .any(|(name, value)| name.trim().is_empty() || value.trim().is_empty())
        {
            return Err(AdapterError::InvalidCapability);
        }
        if kind == BackendKind::Gpu {
            let has_gpu = requirements.iter().any(|requirement| {
                requirement.name == GPU_AVAILABLE_CAPABILITY
                    && requirement.min_value.is_some_and(|minimum| minimum >= 1.0)
            });
            let has_api = required_labels
                .get(GPU_API_LABEL)
                .is_some_and(|api| !api.trim().is_empty());
            if !has_gpu || !has_api {
                return Err(AdapterError::MissingGpuRequirement);
            }
        }
        Ok(Self {
            id,
            implementation_version,
            kind,
            precision,
            deterministic,
            requirements,
            required_labels,
        })
    }

    /// Returns the stable backend identifier.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the backend implementation, kernel, or driver version.
    #[must_use]
    pub fn implementation_version(&self) -> &str {
        &self.implementation_version
    }

    /// Returns the broad backend family.
    #[must_use]
    pub const fn kind(&self) -> BackendKind {
        self.kind
    }

    /// Returns the arithmetic precision.
    #[must_use]
    pub const fn precision(&self) -> Precision {
        self.precision
    }

    /// Returns whether the implementation claims repeatability for identical
    /// input, seed, device, and implementation version.
    #[must_use]
    pub const fn deterministic(&self) -> bool {
        self.deterministic
    }

    /// Returns numeric scheduling requirements.
    #[must_use]
    pub fn requirements(&self) -> &[WorkerCapability] {
        &self.requirements
    }

    /// Returns exact categorical worker-label requirements.
    #[must_use]
    pub const fn required_labels(&self) -> &BTreeMap<String, String> {
        &self.required_labels
    }

    /// Returns whether a worker satisfies every numeric requirement.
    #[must_use]
    pub fn supports_worker(&self, worker: &WorkerContext) -> bool {
        self.requirements.iter().all(|requirement| {
            let Some(actual) = worker.capabilities.get(&requirement.name) else {
                return false;
            };
            actual.is_finite()
                && match requirement.min_value {
                    Some(minimum) => *actual >= minimum,
                    None => true,
                }
        }) && self
            .required_labels
            .iter()
            .all(|(name, value)| worker.labels.get(name) == Some(value))
    }
}

#[derive(Deserialize)]
struct RawBackendDescriptor {
    id: String,
    implementation_version: String,
    kind: BackendKind,
    precision: Precision,
    deterministic: bool,
    requirements: Vec<WorkerCapability>,
    required_labels: BTreeMap<String, String>,
}

impl<'de> Deserialize<'de> for BackendDescriptor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawBackendDescriptor::deserialize(deserializer)?;
        Self::try_new(
            raw.id,
            raw.implementation_version,
            raw.kind,
            raw.precision,
            raw.deterministic,
            raw.requirements,
            raw.required_labels,
        )
        .map_err(serde::de::Error::custom)
    }
}

/// Checked output returned by a compute backend.
#[derive(Clone, Debug, PartialEq)]
pub struct BackendOutput<R> {
    /// Scientific result payload.
    payload: R,
    /// Finite scalar measurements for observability or reduction.
    metrics: BTreeMap<String, f64>,
}

impl<R> BackendOutput<R> {
    /// Constructs output after rejecting empty metric names and non-finite values.
    pub fn try_new(payload: R, metrics: BTreeMap<String, f64>) -> Result<Self, AdapterError> {
        if let Some((name, _)) = metrics
            .iter()
            .find(|(name, value)| name.trim().is_empty() || !value.is_finite())
        {
            return Err(AdapterError::NonFiniteMetric(name.clone()));
        }
        Ok(Self { payload, metrics })
    }

    /// Returns the scientific result payload.
    #[must_use]
    pub const fn payload(&self) -> &R {
        &self.payload
    }

    /// Returns finite scalar metrics.
    #[must_use]
    pub const fn metrics(&self) -> &BTreeMap<String, f64> {
        &self.metrics
    }

    fn into_parts(self) -> (R, BTreeMap<String, f64>) {
        (self.payload, self.metrics)
    }
}

/// Synchronous scientific compute backend.
///
/// Network and async hosts may call this trait inside their own task model.
pub trait ComputeBackend<W, R> {
    /// Backend-specific execution error.
    type Error;

    /// Returns stable backend metadata.
    fn descriptor(&self) -> &BackendDescriptor;
    /// Returns a checked digest of the concrete executable implementation.
    ///
    /// The compatibility default binds the complete descriptor. Backends whose
    /// descriptor does not fully identify kernels, child composition, runtime,
    /// or device equivalence class must override this method. The value must
    /// remain stable before and after one execution.
    fn execution_fingerprint(&self) -> Result<[u8; 32], AdapterError> {
        canonical_digest(
            b"commutator.execution-fingerprint.descriptor-default.v1",
            self.descriptor(),
        )
    }
    /// Executes one payload with the work unit's deterministic seed.
    fn execute(&mut self, payload: &W, seed: u64) -> Result<BackendOutput<R>, Self::Error>;
}

/// Host-owned policy that identifies trusted portable reference backends.
///
/// Backend implementations do not authorize themselves. The application host
/// supplies this policy from its own allowlist or configuration boundary.
pub trait ReferenceAuthorizer {
    /// Returns whether `descriptor` is an approved scientific reference.
    fn authorize(&self, descriptor: &BackendDescriptor) -> bool;
}

impl<F> ReferenceAuthorizer for F
where
    F: Fn(&BackendDescriptor) -> bool,
{
    fn authorize(&self, descriptor: &BackendDescriptor) -> bool {
        self(descriptor)
    }
}

/// Opaque host authorization for one exact CPU-reference descriptor.
///
/// This capability is intentionally not deserializable. A host must recreate
/// it from its trusted [`ReferenceAuthorizer`] after every process boundary.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReferenceAuthorization {
    descriptor_sha256: [u8; 32],
}

/// Apply a host policy and authorize one exact reference descriptor.
pub fn authorize_reference<A: ReferenceAuthorizer + ?Sized>(
    authorizer: &A,
    descriptor: &BackendDescriptor,
) -> Result<ReferenceAuthorization, AdapterError> {
    if descriptor.kind != BackendKind::CpuReference || !authorizer.authorize(descriptor) {
        return Err(AdapterError::UnauthorizedReference);
    }
    Ok(ReferenceAuthorization {
        descriptor_sha256: canonical_digest(b"commutator.reference-descriptor.v1", descriptor)?,
    })
}

impl ReferenceAuthorization {
    fn authorizes(&self, descriptor: &BackendDescriptor) -> bool {
        descriptor.kind == BackendKind::CpuReference
            && canonical_digest(b"commutator.reference-descriptor.v1", descriptor)
                .is_ok_and(|digest| digest == self.descriptor_sha256)
    }
}

/// Trusted evidence required to execute a backend.
#[derive(Clone, Copy, Debug)]
pub enum ExecutionAuthorization<'a> {
    /// Host authorization for a portable reference implementation.
    Reference(&'a ReferenceAuthorization),
    /// Accepted differential qualification under the host's current policy.
    Differential {
        /// Opaque in-process qualification capability.
        report: &'a DifferentialReport,
        /// Exact checked policy that the host currently requires.
        policy: &'a QualificationPolicy,
    },
}

/// Failure returned by [`execute_work_unit`].
#[derive(Debug, PartialEq)]
pub enum ExecutionError<E> {
    /// The worker does not meet backend requirements.
    UnsupportedWorker,
    /// The backend rejected or failed the work.
    Backend(E),
    /// An optimized backend lacks a matching accepted qualification report.
    UnqualifiedBackend,
    /// Backend output violated the finite-metric contract.
    InvalidOutput(AdapterError),
    /// Worker identity or result provenance was invalid.
    InvalidProvenance(AdapterError),
}

/// Executes one typed work unit while preserving its identifiers and provenance.
pub fn execute_work_unit<W, R, B>(
    backend: &mut B,
    worker: &WorkerContext,
    work: &WorkUnit<W>,
    elapsed_ms: u64,
    authorization: ExecutionAuthorization<'_>,
) -> Result<WorkResult<R>, ExecutionError<B::Error>>
where
    B: ComputeBackend<W, R>,
    W: Serialize,
{
    if worker.worker_id.trim().is_empty() {
        return Err(ExecutionError::InvalidProvenance(
            AdapterError::EmptyWorkerIdentity,
        ));
    }
    if !backend.descriptor().supports_worker(worker) {
        return Err(ExecutionError::UnsupportedWorker);
    }
    let authorized = match authorization {
        ExecutionAuthorization::Reference(reference) => reference.authorizes(backend.descriptor()),
        ExecutionAuthorization::Differential { report, policy } => {
            backend.execution_fingerprint().is_ok_and(|fingerprint| {
                report.qualifies(backend.descriptor(), &fingerprint, worker, work, policy)
            })
        }
    };
    if !authorized {
        return Err(ExecutionError::UnqualifiedBackend);
    }
    let mut run = work.run.clone();
    let mut additions = Vec::new();
    if let Some(requested) = &run.backend {
        additions.push(("requested_backend".to_owned(), requested.clone()));
    }
    if let Some(requested) = &run.precision {
        additions.push(("requested_precision".to_owned(), requested.clone()));
    }
    if let Some(requested) = &run.implementation_version {
        additions.push((
            "requested_implementation_version".to_owned(),
            requested.clone(),
        ));
    }
    additions.push(("worker_id".to_owned(), worker.worker_id.clone()));
    additions.extend(
        worker
            .labels
            .iter()
            .map(|(name, value)| (format!("worker.label.{name}"), value.clone())),
    );
    if let Some((name, _)) = additions
        .iter()
        .find(|(name, _)| run.attributes.contains_key(name))
    {
        return Err(ExecutionError::InvalidProvenance(
            AdapterError::ProvenanceConflict(name.clone()),
        ));
    }
    run.backend = None;
    run.precision = None;
    run.implementation_version = None;
    run.attributes.extend(additions);
    run.backend = Some(backend.descriptor().id.clone());
    run.precision = Some(backend.descriptor().precision.tag().to_owned());
    run.implementation_version = Some(backend.descriptor().implementation_version.clone());
    let output = backend
        .execute(&work.payload, work.run.seed)
        .map_err(ExecutionError::Backend)?;
    if let ExecutionAuthorization::Differential { report, policy } = authorization {
        let still_authorized = backend.execution_fingerprint().is_ok_and(|fingerprint| {
            report.qualifies(backend.descriptor(), &fingerprint, worker, work, policy)
        });
        if !still_authorized {
            return Err(ExecutionError::UnqualifiedBackend);
        }
    }
    let (payload, metrics) = output.into_parts();
    if let Some((name, _)) = metrics
        .iter()
        .find(|(name, value)| name.trim().is_empty() || !value.is_finite())
    {
        return Err(ExecutionError::InvalidOutput(
            AdapterError::NonFiniteMetric(name.clone()),
        ));
    }
    Ok(WorkResult {
        experiment_id: work.experiment_id.clone(),
        instance_id: work.instance_id.clone(),
        unit_id: work.unit_id.clone(),
        generation: work.generation,
        run,
        elapsed_ms,
        metrics,
        payload,
    })
}

/// Finite comparison metrics and an acceptance decision.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Comparison {
    /// Largest absolute component error.
    max_absolute_error: f64,
    /// Largest relative component error under the comparator's scale policy.
    max_relative_error: f64,
    /// Whether all scientific invariants and tolerances passed.
    accepted: bool,
}

#[derive(Deserialize)]
struct RawComparison {
    max_absolute_error: f64,
    max_relative_error: f64,
    accepted: bool,
}

impl<'de> Deserialize<'de> for Comparison {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawComparison::deserialize(deserializer)?;
        Self::try_new(raw.max_absolute_error, raw.max_relative_error, raw.accepted)
            .map_err(serde::de::Error::custom)
    }
}

impl Comparison {
    /// Constructs checked comparison output.
    pub fn try_new(
        max_absolute_error: f64,
        max_relative_error: f64,
        accepted: bool,
    ) -> Result<Self, AdapterError> {
        if !max_absolute_error.is_finite()
            || max_absolute_error < 0.0
            || !max_relative_error.is_finite()
            || max_relative_error < 0.0
        {
            return Err(AdapterError::InvalidTolerance);
        }
        Ok(Self {
            max_absolute_error,
            max_relative_error,
            accepted,
        })
    }

    /// Returns the largest absolute component error.
    #[must_use]
    pub const fn max_absolute_error(&self) -> f64 {
        self.max_absolute_error
    }

    /// Returns the largest relative component error.
    #[must_use]
    pub const fn max_relative_error(&self) -> f64 {
        self.max_relative_error
    }

    /// Returns whether all comparator checks passed.
    #[must_use]
    pub const fn accepted(&self) -> bool {
        self.accepted
    }
}

/// Versioned numerical admission policy for differential qualification.
///
/// Both maximum-error limits are enforced as upper bounds. A comparator may
/// additionally reject a result through [`Comparison::accepted`] when a
/// scientific invariant fails.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct QualificationPolicy {
    id: String,
    version: String,
    absolute_tolerance: f64,
    relative_tolerance: f64,
}

impl QualificationPolicy {
    /// Constructs a checked, versioned policy with finite nonnegative limits.
    pub fn try_new(
        id: impl Into<String>,
        version: impl Into<String>,
        absolute_tolerance: f64,
        relative_tolerance: f64,
    ) -> Result<Self, AdapterError> {
        let id = id.into();
        let version = version.into();
        if id.trim().is_empty() {
            return Err(AdapterError::EmptyIdentifier("qualification policy id"));
        }
        if version.trim().is_empty() {
            return Err(AdapterError::EmptyIdentifier(
                "qualification policy version",
            ));
        }
        if !absolute_tolerance.is_finite()
            || absolute_tolerance < 0.0
            || !relative_tolerance.is_finite()
            || relative_tolerance < 0.0
        {
            return Err(AdapterError::InvalidTolerance);
        }
        Ok(Self {
            id,
            version,
            absolute_tolerance,
            relative_tolerance,
        })
    }

    /// Returns the stable policy identifier.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the policy implementation or configuration version.
    #[must_use]
    pub fn version(&self) -> &str {
        &self.version
    }

    /// Returns the maximum admitted absolute error.
    #[must_use]
    pub const fn absolute_tolerance(&self) -> f64 {
        self.absolute_tolerance
    }

    /// Returns the maximum admitted relative error.
    #[must_use]
    pub const fn relative_tolerance(&self) -> f64 {
        self.relative_tolerance
    }

    fn accepts(&self, comparison: &Comparison) -> bool {
        comparison.accepted
            && comparison.max_absolute_error <= self.absolute_tolerance
            && comparison.max_relative_error <= self.relative_tolerance
    }
}

#[derive(Deserialize)]
struct RawQualificationPolicy {
    id: String,
    version: String,
    absolute_tolerance: f64,
    relative_tolerance: f64,
}

impl<'de> Deserialize<'de> for QualificationPolicy {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawQualificationPolicy::deserialize(deserializer)?;
        Self::try_new(
            raw.id,
            raw.version,
            raw.absolute_tolerance,
            raw.relative_tolerance,
        )
        .map_err(serde::de::Error::custom)
    }
}

/// Scientific result comparison policy used by differential validation.
pub trait ResultComparator<R> {
    /// Compares a candidate result to the reference result.
    fn compare(&self, reference: &R, candidate: &R) -> Result<Comparison, AdapterError>;
}

/// Opaque result of executing reference and candidate backends on identical work.
///
/// Reports intentionally implement neither serialization nor deserialization:
/// safe code can obtain an admission capability only from [`differential_check`].
/// Use [`Self::audit_snapshot`] for a serialize-only commitment summary.
///
/// ```compile_fail
/// use experiment_accelerator::DifferentialReport;
/// let _: DifferentialReport = serde_json::from_str("{}").unwrap();
/// ```
///
/// ```compile_fail
/// use experiment_accelerator::DifferentialReport;
/// # fn opaque(report: &DifferentialReport) {
/// let _ = serde_json::to_string(report).unwrap();
/// # }
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct DifferentialReport {
    /// Reference backend identifier.
    reference_backend: String,
    /// Reference implementation version.
    reference_version: String,
    /// Canonical digest of the complete reference descriptor.
    reference_descriptor_sha256: [u8; 32],
    /// Digest of the concrete reference execution implementation.
    reference_execution_fingerprint_sha256: [u8; 32],
    /// Candidate backend identifier.
    candidate_backend: String,
    /// Candidate implementation version.
    candidate_version: String,
    /// Canonical digest of the complete candidate descriptor.
    candidate_descriptor_sha256: [u8; 32],
    /// Digest of the concrete candidate execution implementation.
    candidate_execution_fingerprint_sha256: [u8; 32],
    /// Worker identity used for qualification.
    worker_id: String,
    /// Canonical digest of the complete worker context.
    worker_sha256: [u8; 32],
    /// Experiment identifier.
    experiment_id: String,
    /// Run instance identifier.
    instance_id: String,
    /// Work-unit identifier.
    unit_id: String,
    /// Work generation.
    generation: u64,
    /// Payload schema version.
    schema_version: u32,
    /// SHA-256 of the serialized typed work unit.
    work_sha256: [u8; 32],
    /// Shared deterministic seed.
    seed: u64,
    /// Checked policy used to make the admission decision.
    policy: QualificationPolicy,
    /// Canonical digest of the exact checked policy.
    policy_sha256: [u8; 32],
    /// Numerical and invariant comparison.
    comparison: Comparison,
}

impl DifferentialReport {
    /// Returns the validated comparison.
    #[must_use]
    pub const fn comparison(&self) -> &Comparison {
        &self.comparison
    }

    /// Returns the candidate backend identifier.
    #[must_use]
    pub fn candidate_backend(&self) -> &str {
        &self.candidate_backend
    }

    /// Returns the shared deterministic seed.
    #[must_use]
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// Returns a serializable commitment summary that carries no admission
    /// authority and cannot be converted back into this report.
    #[must_use]
    pub fn audit_snapshot(&self) -> QualificationAudit {
        QualificationAudit {
            schema: "commutator.accelerator-qualification.v2",
            reference_backend: self.reference_backend.clone(),
            reference_version: self.reference_version.clone(),
            reference_descriptor_sha256: digest_hex(&self.reference_descriptor_sha256),
            reference_execution_fingerprint_sha256: digest_hex(
                &self.reference_execution_fingerprint_sha256,
            ),
            candidate_backend: self.candidate_backend.clone(),
            candidate_version: self.candidate_version.clone(),
            candidate_descriptor_sha256: digest_hex(&self.candidate_descriptor_sha256),
            candidate_execution_fingerprint_sha256: digest_hex(
                &self.candidate_execution_fingerprint_sha256,
            ),
            worker_id: self.worker_id.clone(),
            worker_sha256: digest_hex(&self.worker_sha256),
            experiment_id: self.experiment_id.clone(),
            instance_id: self.instance_id.clone(),
            unit_id: self.unit_id.clone(),
            generation: self.generation,
            schema_version: self.schema_version,
            work_sha256: digest_hex(&self.work_sha256),
            seed: self.seed,
            policy: self.policy.clone(),
            policy_sha256: digest_hex(&self.policy_sha256),
            comparison: self.comparison.clone(),
        }
    }

    fn qualifies<W: Serialize>(
        &self,
        descriptor: &BackendDescriptor,
        execution_fingerprint: &[u8; 32],
        worker: &WorkerContext,
        work: &WorkUnit<W>,
        policy: &QualificationPolicy,
    ) -> bool {
        self.policy == *policy
            && policy.accepts(&self.comparison)
            && canonical_digest(b"commutator.qualification-policy.v1", policy)
                .is_ok_and(|digest| digest == self.policy_sha256)
            && self.candidate_backend == descriptor.id
            && self.candidate_version == descriptor.implementation_version
            && canonical_digest(b"commutator.candidate-descriptor.v1", descriptor)
                .is_ok_and(|digest| digest == self.candidate_descriptor_sha256)
            && *execution_fingerprint == self.candidate_execution_fingerprint_sha256
            && self.worker_id == worker.worker_id
            && canonical_digest(b"commutator.worker-context.v1", worker)
                .is_ok_and(|digest| digest == self.worker_sha256)
            && self.experiment_id == work.experiment_id
            && self.instance_id == work.instance_id
            && self.unit_id == work.unit_id
            && self.generation == work.generation
            && self.schema_version == work.run.schema_version
            && self.seed == work.run.seed
            && canonical_digest(b"commutator.work-unit.v1", work)
                .is_ok_and(|hash| hash == self.work_sha256)
    }
}

/// Serializable, non-authoritative commitment summary from a differential check.
///
/// This snapshot deliberately implements `Serialize` but not `Deserialize`.
/// Even a snapshot obtained from a trusted log cannot be used with
/// [`ExecutionAuthorization`]; only the opaque originating
/// [`DifferentialReport`] carries in-process admission authority.
///
/// ```compile_fail
/// use experiment_accelerator::QualificationAudit;
/// let _: QualificationAudit = serde_json::from_str("{}").unwrap();
/// ```
///
/// ```compile_fail
/// use experiment_accelerator::{ExecutionAuthorization, QualificationAudit,
///     QualificationPolicy};
/// # fn separate(audit: &QualificationAudit, policy: &QualificationPolicy) {
/// let _ = ExecutionAuthorization::Differential { report: audit, policy };
/// # }
/// ```
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct QualificationAudit {
    schema: &'static str,
    reference_backend: String,
    reference_version: String,
    reference_descriptor_sha256: String,
    reference_execution_fingerprint_sha256: String,
    candidate_backend: String,
    candidate_version: String,
    candidate_descriptor_sha256: String,
    candidate_execution_fingerprint_sha256: String,
    worker_id: String,
    worker_sha256: String,
    experiment_id: String,
    instance_id: String,
    unit_id: String,
    generation: u64,
    schema_version: u32,
    work_sha256: String,
    seed: u64,
    policy: QualificationPolicy,
    policy_sha256: String,
    comparison: Comparison,
}

impl QualificationAudit {
    /// Returns the audit schema identifier.
    #[must_use]
    pub const fn schema(&self) -> &str {
        self.schema
    }

    /// Returns the exact checked policy recorded by the check.
    #[must_use]
    pub const fn policy(&self) -> &QualificationPolicy {
        &self.policy
    }

    /// Returns the final, policy-enforced comparison.
    #[must_use]
    pub const fn comparison(&self) -> &Comparison {
        &self.comparison
    }

    /// Returns the canonical digest of the complete reference descriptor.
    #[must_use]
    pub fn reference_descriptor_sha256(&self) -> &str {
        &self.reference_descriptor_sha256
    }

    /// Returns the canonical digest of the complete candidate descriptor.
    #[must_use]
    pub fn candidate_descriptor_sha256(&self) -> &str {
        &self.candidate_descriptor_sha256
    }

    /// Returns the concrete reference execution fingerprint.
    #[must_use]
    pub fn reference_execution_fingerprint_sha256(&self) -> &str {
        &self.reference_execution_fingerprint_sha256
    }

    /// Returns the concrete candidate execution fingerprint.
    #[must_use]
    pub fn candidate_execution_fingerprint_sha256(&self) -> &str {
        &self.candidate_execution_fingerprint_sha256
    }

    /// Returns the canonical digest of the complete worker context.
    #[must_use]
    pub fn worker_sha256(&self) -> &str {
        &self.worker_sha256
    }

    /// Returns the canonical digest of the typed work unit.
    #[must_use]
    pub fn work_sha256(&self) -> &str {
        &self.work_sha256
    }

    /// Returns the canonical digest of the checked policy.
    #[must_use]
    pub fn policy_sha256(&self) -> &str {
        &self.policy_sha256
    }
}

/// Failure returned by [`differential_check`].
#[derive(Debug, PartialEq)]
pub enum DifferentialError<ReferenceError, CandidateError> {
    /// Reference execution failed.
    Reference(ReferenceError),
    /// Candidate execution failed.
    Candidate(CandidateError),
    /// The comparison policy returned invalid metrics.
    Comparison(AdapterError),
    /// Descriptor roles, identities, or worker support were invalid.
    InvalidBoundary(AdapterError),
}

/// Runs portable and optimized backends on the same typed work unit.
///
/// The returned report is bound to the work serialization, worker identity,
/// complete backend descriptors, concrete execution fingerprints, and the
/// checked qualification policy. Fingerprints are sampled before and after the
/// check and must remain stable. Its final acceptance bit enforces both policy
/// tolerances in addition to the comparator's scientific-invariant decision.
/// An accepted report and the identical current policy are required by
/// [`execute_work_unit`] for an optimized backend. The reference backend must
/// carry a capability minted by a host-owned allowlist.
pub fn differential_check<W, R, Reference, Candidate, Comparator>(
    reference: &mut Reference,
    reference_authorization: &ReferenceAuthorization,
    candidate: &mut Candidate,
    worker: &WorkerContext,
    work: &WorkUnit<W>,
    policy: &QualificationPolicy,
    comparator: &Comparator,
) -> Result<DifferentialReport, DifferentialError<Reference::Error, Candidate::Error>>
where
    Reference: ComputeBackend<W, R>,
    Candidate: ComputeBackend<W, R>,
    Comparator: ResultComparator<R>,
    W: Serialize,
{
    if worker.worker_id.trim().is_empty() {
        return Err(DifferentialError::InvalidBoundary(
            AdapterError::EmptyWorkerIdentity,
        ));
    }
    if !reference_authorization.authorizes(reference.descriptor()) {
        return Err(DifferentialError::InvalidBoundary(
            AdapterError::UnauthorizedReference,
        ));
    }
    if reference.descriptor().id == candidate.descriptor().id {
        return Err(DifferentialError::InvalidBoundary(
            AdapterError::EmptyIdentifier("distinct candidate backend id"),
        ));
    }
    if !reference.descriptor().supports_worker(worker)
        || !candidate.descriptor().supports_worker(worker)
    {
        return Err(DifferentialError::InvalidBoundary(
            AdapterError::InvalidCapability,
        ));
    }
    let reference_descriptor_sha256 = canonical_digest(
        b"commutator.reference-descriptor.v1",
        reference.descriptor(),
    )
    .map_err(DifferentialError::InvalidBoundary)?;
    let candidate_descriptor_sha256 = canonical_digest(
        b"commutator.candidate-descriptor.v1",
        candidate.descriptor(),
    )
    .map_err(DifferentialError::InvalidBoundary)?;
    let reference_execution_fingerprint_sha256 = reference
        .execution_fingerprint()
        .map_err(DifferentialError::InvalidBoundary)?;
    let candidate_execution_fingerprint_sha256 = candidate
        .execution_fingerprint()
        .map_err(DifferentialError::InvalidBoundary)?;
    let reference_output = reference
        .execute(&work.payload, work.run.seed)
        .map_err(DifferentialError::Reference)?;
    let candidate_output = candidate
        .execute(&work.payload, work.run.seed)
        .map_err(DifferentialError::Candidate)?;
    if reference
        .execution_fingerprint()
        .map_err(DifferentialError::InvalidBoundary)?
        != reference_execution_fingerprint_sha256
        || candidate
            .execution_fingerprint()
            .map_err(DifferentialError::InvalidBoundary)?
            != candidate_execution_fingerprint_sha256
    {
        return Err(DifferentialError::InvalidBoundary(
            AdapterError::UnstableExecutionFingerprint,
        ));
    }
    if canonical_digest(
        b"commutator.reference-descriptor.v1",
        reference.descriptor(),
    )
    .map_err(DifferentialError::InvalidBoundary)?
        != reference_descriptor_sha256
        || canonical_digest(
            b"commutator.candidate-descriptor.v1",
            candidate.descriptor(),
        )
        .map_err(DifferentialError::InvalidBoundary)?
            != candidate_descriptor_sha256
    {
        return Err(DifferentialError::InvalidBoundary(
            AdapterError::UnstableExecutionFingerprint,
        ));
    }
    let mut comparison = comparator
        .compare(reference_output.payload(), candidate_output.payload())
        .map_err(DifferentialError::Comparison)?;
    comparison.accepted = policy.accepts(&comparison);
    let worker_sha256 = canonical_digest(b"commutator.worker-context.v1", worker)
        .map_err(DifferentialError::InvalidBoundary)?;
    let work_sha256 = canonical_digest(b"commutator.work-unit.v1", work)
        .map_err(DifferentialError::InvalidBoundary)?;
    let policy_sha256 = canonical_digest(b"commutator.qualification-policy.v1", policy)
        .map_err(DifferentialError::InvalidBoundary)?;
    Ok(DifferentialReport {
        reference_backend: reference.descriptor().id.clone(),
        reference_version: reference.descriptor().implementation_version.clone(),
        reference_descriptor_sha256,
        reference_execution_fingerprint_sha256,
        candidate_backend: candidate.descriptor().id.clone(),
        candidate_version: candidate.descriptor().implementation_version.clone(),
        candidate_descriptor_sha256,
        candidate_execution_fingerprint_sha256,
        worker_id: worker.worker_id.clone(),
        worker_sha256,
        experiment_id: work.experiment_id.clone(),
        instance_id: work.instance_id.clone(),
        unit_id: work.unit_id.clone(),
        generation: work.generation,
        schema_version: work.run.schema_version,
        work_sha256,
        seed: work.run.seed,
        policy: policy.clone(),
        policy_sha256,
        comparison,
    })
}

fn digest_hex(digest: &[u8; 32]) -> String {
    use core::fmt::Write as _;

    let mut encoded = String::with_capacity(64);
    for byte in digest {
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}

pub(crate) fn canonical_digest<T: Serialize + ?Sized>(
    domain: &[u8],
    value: &T,
) -> Result<[u8; 32], AdapterError> {
    let value = serde_json::to_value(value).map_err(|_| AdapterError::InvalidCapability)?;
    let canonical = canonicalize_json(value);
    let encoded = serde_json::to_vec(&canonical).map_err(|_| AdapterError::InvalidCapability)?;
    let mut digest = Sha256::new();
    digest.update(domain);
    digest.update([0]);
    digest.update(encoded);
    Ok(digest.finalize().into())
}

fn canonicalize_json(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Array(values) => {
            serde_json::Value::Array(values.into_iter().map(canonicalize_json).collect())
        }
        serde_json::Value::Object(values) => {
            let mut entries: Vec<_> = values.into_iter().collect();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            let mut sorted = serde_json::Map::new();
            for (name, value) in entries {
                sorted.insert(name, canonicalize_json(value));
            }
            serde_json::Value::Object(sorted)
        }
        scalar => scalar,
    }
}

/// Half-open contiguous index range for map or domain work.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct IndexChunk {
    /// Inclusive first index.
    start: usize,
    /// Exclusive end index.
    end: usize,
}

impl IndexChunk {
    /// Constructs a checked nonempty half-open range.
    pub fn try_new(start: usize, end: usize) -> Result<Self, AdapterError> {
        if start >= end {
            return Err(AdapterError::InvalidCapability);
        }
        Ok(Self { start, end })
    }

    /// Returns the inclusive first index.
    #[must_use]
    pub const fn start(self) -> usize {
        self.start
    }

    /// Returns the exclusive end index.
    #[must_use]
    pub const fn end(self) -> usize {
        self.end
    }

    /// Returns the number of indices in the chunk.
    #[must_use]
    pub const fn len(self) -> usize {
        self.end - self.start
    }

    /// Returns whether the chunk contains no indices.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.start == self.end
    }
}

#[derive(Deserialize)]
struct RawIndexChunk {
    start: usize,
    end: usize,
}

impl<'de> Deserialize<'de> for IndexChunk {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawIndexChunk::deserialize(deserializer)?;
        Self::try_new(raw.start, raw.end).map_err(serde::de::Error::custom)
    }
}

/// Partitions `[0, total)` into stable contiguous chunks.
pub fn partition_range(total: usize, chunk_size: usize) -> Result<Vec<IndexChunk>, AdapterError> {
    if chunk_size == 0 {
        return Err(AdapterError::ZeroChunkSize);
    }
    let chunk_count = total.div_ceil(chunk_size);
    let mut chunks = Vec::new();
    chunks
        .try_reserve_exact(chunk_count)
        .map_err(|_| AdapterError::SizeOverflow)?;
    let mut start = 0usize;
    while start < total {
        let end = start.saturating_add(chunk_size).min(total);
        chunks.push(IndexChunk::try_new(start, end)?);
        start = end;
    }
    Ok(chunks)
}

/// Builds deterministic independent-replica work units.
pub fn independent_replica_work<P>(
    experiment_id: &str,
    instance_id: &str,
    generation: u64,
    schema_version: u32,
    start_ordinal: u64,
    payloads: impl IntoIterator<Item = (u64, P)>,
) -> Result<Vec<WorkUnit<P>>, AdapterError> {
    if experiment_id.trim().is_empty() {
        return Err(AdapterError::EmptyIdentifier("experiment id"));
    }
    if instance_id.trim().is_empty() {
        return Err(AdapterError::EmptyIdentifier("instance id"));
    }
    payloads
        .into_iter()
        .enumerate()
        .map(|(index, (seed, payload))| {
            let offset = u64::try_from(index).map_err(|_| AdapterError::SizeOverflow)?;
            let ordinal = start_ordinal
                .checked_add(offset)
                .ok_or(AdapterError::SizeOverflow)?;
            Ok(WorkUnit {
                experiment_id: experiment_id.to_owned(),
                instance_id: instance_id.to_owned(),
                unit_id: format!("replica-{ordinal:016x}"),
                generation,
                run: RunMetadata {
                    seed,
                    schema_version,
                    ..RunMetadata::default()
                },
                payload,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ScaleBackend {
        descriptor: BackendDescriptor,
        scale: f64,
    }

    impl ComputeBackend<Vec<f64>, Vec<f64>> for ScaleBackend {
        type Error = ();

        fn descriptor(&self) -> &BackendDescriptor {
            &self.descriptor
        }

        fn execute(
            &mut self,
            payload: &Vec<f64>,
            _seed: u64,
        ) -> Result<BackendOutput<Vec<f64>>, Self::Error> {
            Ok(BackendOutput::try_new(
                payload.iter().map(|value| value * self.scale).collect(),
                BTreeMap::new(),
            )
            .unwrap())
        }
    }

    struct MutatingFingerprintBackend {
        descriptor: BackendDescriptor,
        fingerprint_tag: u8,
        mutate_on_execute: bool,
    }

    impl ComputeBackend<Vec<f64>, Vec<f64>> for MutatingFingerprintBackend {
        type Error = ();

        fn descriptor(&self) -> &BackendDescriptor {
            &self.descriptor
        }

        fn execution_fingerprint(&self) -> Result<[u8; 32], AdapterError> {
            Ok([self.fingerprint_tag; 32])
        }

        fn execute(
            &mut self,
            payload: &Vec<f64>,
            _seed: u64,
        ) -> Result<BackendOutput<Vec<f64>>, Self::Error> {
            if self.mutate_on_execute {
                self.fingerprint_tag = self.fingerprint_tag.wrapping_add(1);
            }
            Ok(BackendOutput::try_new(payload.clone(), BTreeMap::new()).unwrap())
        }
    }

    struct VectorComparator {
        tolerance: f64,
    }

    impl ResultComparator<Vec<f64>> for VectorComparator {
        fn compare(
            &self,
            reference: &Vec<f64>,
            candidate: &Vec<f64>,
        ) -> Result<Comparison, AdapterError> {
            let maximum = reference
                .iter()
                .zip(candidate)
                .map(|(left, right)| (left - right).abs())
                .fold(0.0_f64, f64::max);
            Comparison::try_new(maximum, maximum, maximum <= self.tolerance)
        }
    }

    fn descriptor(id: &str, kind: BackendKind) -> BackendDescriptor {
        let requirements = if kind == BackendKind::Gpu {
            vec![WorkerCapability {
                name: GPU_AVAILABLE_CAPABILITY.into(),
                min_value: Some(1.0),
            }]
        } else {
            vec![]
        };
        let labels = if kind == BackendKind::Gpu {
            BTreeMap::from([(GPU_API_LABEL.into(), "test".into())])
        } else {
            BTreeMap::new()
        };
        BackendDescriptor::try_new(
            id,
            "test-v1",
            kind,
            Precision::F64,
            true,
            requirements,
            labels,
        )
        .unwrap()
    }

    fn reference_authorization(descriptor: &BackendDescriptor) -> ReferenceAuthorization {
        let trusted_id = descriptor.id().to_owned();
        authorize_reference(
            &|candidate: &BackendDescriptor| candidate.id() == trusted_id,
            descriptor,
        )
        .unwrap()
    }

    fn policy(tolerance: f64) -> QualificationPolicy {
        QualificationPolicy::try_new("vector-max-error", "test-v1", tolerance, tolerance).unwrap()
    }

    #[test]
    fn range_partition_is_complete_and_disjoint() {
        let chunks = partition_range(10, 4).unwrap();
        assert_eq!(
            chunks,
            vec![
                IndexChunk::try_new(0, 4).unwrap(),
                IndexChunk::try_new(4, 8).unwrap(),
                IndexChunk::try_new(8, 10).unwrap()
            ]
        );
        assert_eq!(chunks.iter().map(|chunk| chunk.len()).sum::<usize>(), 10);
    }

    #[test]
    fn replica_plan_has_stable_ids_and_seeds() {
        let units =
            independent_replica_work("scan", "scan-7", 3, 1, 20, [(9, "cold"), (11, "hot")])
                .unwrap();
        assert_eq!(units[0].unit_id, "replica-0000000000000014");
        assert_eq!(units[1].run.seed, 11);
        assert_eq!(units[1].generation, 3);
    }

    #[test]
    fn worker_capabilities_gate_backend_execution() {
        let descriptor = BackendDescriptor::try_new(
            "gpu",
            "kernel-v2",
            BackendKind::Gpu,
            Precision::F32,
            false,
            vec![
                WorkerCapability {
                    name: GPU_AVAILABLE_CAPABILITY.into(),
                    min_value: Some(1.0),
                },
                WorkerCapability {
                    name: "gpu.memory_gib".into(),
                    min_value: Some(8.0),
                },
            ],
            BTreeMap::from([(GPU_API_LABEL.into(), "metal".into())]),
        )
        .unwrap();
        let mut worker = WorkerContext::default();
        worker
            .capabilities
            .insert(GPU_AVAILABLE_CAPABILITY.into(), 1.0);
        worker.capabilities.insert("gpu.memory_gib".into(), 4.0);
        worker.labels.insert(GPU_API_LABEL.into(), "metal".into());
        assert!(!descriptor.supports_worker(&worker));
        worker.capabilities.insert("gpu.memory_gib".into(), 12.0);
        assert!(descriptor.supports_worker(&worker));
    }

    #[test]
    fn execution_preserves_work_identity_and_sets_backend_provenance() {
        let mut work = independent_replica_work("demo", "run", 2, 4, 0, [(17, vec![2.0])])
            .unwrap()
            .remove(0);
        work.run.backend = Some("requested-backend".into());
        work.run.precision = Some("requested-precision".into());
        work.run.implementation_version = Some("requested-version".into());
        let mut backend = ScaleBackend {
            descriptor: descriptor("cpu-reference", BackendKind::CpuReference),
            scale: 2.0,
        };
        let mut worker = WorkerContext {
            worker_id: "worker-7".into(),
            ..WorkerContext::default()
        };
        worker.labels.insert("host".into(), "ci".into());
        let authorization = reference_authorization(&backend.descriptor);
        let result = execute_work_unit(
            &mut backend,
            &worker,
            &work,
            5,
            ExecutionAuthorization::Reference(&authorization),
        )
        .unwrap();
        assert_eq!(result.payload, vec![4.0]);
        assert_eq!(result.unit_id, work.unit_id);
        assert_eq!(result.run.backend.as_deref(), Some("cpu-reference"));
        assert_eq!(result.run.precision.as_deref(), Some("f64"));
        assert_eq!(
            result.run.implementation_version.as_deref(),
            Some("test-v1")
        );
        assert_eq!(
            result.run.attributes.get("worker_id").map(String::as_str),
            Some("worker-7")
        );
        assert_eq!(
            result
                .run
                .attributes
                .get("requested_backend")
                .map(String::as_str),
            Some("requested-backend")
        );
        assert_eq!(
            result
                .run
                .attributes
                .get("requested_precision")
                .map(String::as_str),
            Some("requested-precision")
        );
        assert_eq!(
            result
                .run
                .attributes
                .get("requested_implementation_version")
                .map(String::as_str),
            Some("requested-version")
        );
    }

    #[test]
    fn differential_check_detects_biased_candidate() {
        let mut reference = ScaleBackend {
            descriptor: descriptor("reference", BackendKind::CpuReference),
            scale: 1.0,
        };
        let mut candidate = ScaleBackend {
            descriptor: descriptor("gpu", BackendKind::Gpu),
            scale: 1.01,
        };
        let mut worker = WorkerContext {
            worker_id: "gpu-worker".into(),
            ..WorkerContext::default()
        };
        worker
            .capabilities
            .insert(GPU_AVAILABLE_CAPABILITY.into(), 1.0);
        worker.labels.insert(GPU_API_LABEL.into(), "test".into());
        let work = independent_replica_work("diff", "run", 0, 1, 0, [(99, vec![1.0, 2.0])])
            .unwrap()
            .remove(0);
        let authorization = reference_authorization(&reference.descriptor);
        let policy = policy(1e-3);
        let report = differential_check(
            &mut reference,
            &authorization,
            &mut candidate,
            &worker,
            &work,
            &policy,
            // The comparator accepts this numerical difference; the declared
            // qualification policy must still reject it.
            &VectorComparator { tolerance: 1.0 },
        )
        .unwrap();
        assert!(!report.comparison().accepted());
        assert!((report.comparison().max_absolute_error() - 0.02).abs() < 1e-14);
        assert_eq!(report.seed(), 99);
        assert!(matches!(
            execute_work_unit(
                &mut candidate,
                &worker,
                &work,
                1,
                ExecutionAuthorization::Differential {
                    report: &report,
                    policy: &policy,
                }
            ),
            Err(ExecutionError::UnqualifiedBackend)
        ));
    }

    #[test]
    fn checked_policy_rejects_invalid_deserialization() {
        assert!(serde_json::from_str::<QualificationPolicy>(
            r#"{"id":" ","version":"v1","absolute_tolerance":0.001,"relative_tolerance":0.001}"#
        )
        .is_err());
        assert!(serde_json::from_str::<QualificationPolicy>(
            r#"{"id":"strict","version":"v1","absolute_tolerance":-0.001,"relative_tolerance":0.001}"#
        )
        .is_err());
    }

    #[test]
    fn accepted_report_qualifies_only_the_bound_work() {
        let mut reference = ScaleBackend {
            descriptor: descriptor("reference", BackendKind::CpuReference),
            scale: 1.0,
        };
        let mut candidate = ScaleBackend {
            descriptor: descriptor("gpu", BackendKind::Gpu),
            scale: 1.0,
        };
        let mut worker = WorkerContext {
            worker_id: "gpu-worker".into(),
            ..WorkerContext::default()
        };
        worker
            .capabilities
            .insert(GPU_AVAILABLE_CAPABILITY.into(), 1.0);
        worker.labels.insert(GPU_API_LABEL.into(), "test".into());
        let work = independent_replica_work("diff", "run", 0, 1, 0, [(7, vec![3.0])])
            .unwrap()
            .remove(0);
        let authorization = reference_authorization(&reference.descriptor);
        let policy = policy(1e-12);
        let report = differential_check(
            &mut reference,
            &authorization,
            &mut candidate,
            &worker,
            &work,
            &policy,
            &VectorComparator { tolerance: 1e-12 },
        )
        .unwrap();
        assert!(report.comparison().accepted());
        assert!(execute_work_unit(
            &mut candidate,
            &worker,
            &work,
            1,
            ExecutionAuthorization::Differential {
                report: &report,
                policy: &policy,
            }
        )
        .is_ok());

        let changed_policy = QualificationPolicy::try_new(
            policy.id(),
            policy.version(),
            policy.absolute_tolerance() * 10.0,
            policy.relative_tolerance() * 10.0,
        )
        .unwrap();
        assert!(matches!(
            execute_work_unit(
                &mut candidate,
                &worker,
                &work,
                1,
                ExecutionAuthorization::Differential {
                    report: &report,
                    policy: &changed_policy,
                }
            ),
            Err(ExecutionError::UnqualifiedBackend)
        ));

        let mut changed = work.clone();
        changed.payload[0] = 4.0;
        assert!(matches!(
            execute_work_unit(
                &mut candidate,
                &worker,
                &changed,
                1,
                ExecutionAuthorization::Differential {
                    report: &report,
                    policy: &policy,
                }
            ),
            Err(ExecutionError::UnqualifiedBackend)
        ));
    }

    #[test]
    fn audit_snapshot_serializes_commitments_without_admission_authority() {
        let mut reference = ScaleBackend {
            descriptor: descriptor("reference", BackendKind::CpuReference),
            scale: 1.0,
        };
        let mut candidate = ScaleBackend {
            descriptor: descriptor("gpu", BackendKind::Gpu),
            scale: 1.0,
        };
        let mut worker = WorkerContext {
            worker_id: "gpu-worker".into(),
            ..WorkerContext::default()
        };
        worker
            .capabilities
            .insert(GPU_AVAILABLE_CAPABILITY.into(), 1.0);
        worker.labels.insert(GPU_API_LABEL.into(), "test".into());
        let work = independent_replica_work("diff", "run", 2, 3, 0, [(41, vec![2.0])])
            .unwrap()
            .remove(0);
        let authorization = reference_authorization(&reference.descriptor);
        let policy = policy(1e-12);
        let report = differential_check(
            &mut reference,
            &authorization,
            &mut candidate,
            &worker,
            &work,
            &policy,
            &VectorComparator { tolerance: 1e-12 },
        )
        .unwrap();

        let audit = report.audit_snapshot();
        let value = serde_json::to_value(&audit).unwrap();
        assert_eq!(value["schema"], "commutator.accelerator-qualification.v2");
        assert_eq!(value["reference_backend"], "reference");
        assert_eq!(value["candidate_backend"], "gpu");
        assert_eq!(value["worker_id"], "gpu-worker");
        assert_eq!(value["experiment_id"], "diff");
        assert_eq!(value["generation"], 2);
        assert_eq!(value["schema_version"], 3);
        assert_eq!(value["seed"], 41);
        assert_eq!(value["policy"]["id"], "vector-max-error");
        for field in [
            "reference_descriptor_sha256",
            "reference_execution_fingerprint_sha256",
            "candidate_descriptor_sha256",
            "candidate_execution_fingerprint_sha256",
            "worker_sha256",
            "work_sha256",
            "policy_sha256",
        ] {
            let digest = value[field].as_str().unwrap();
            assert_eq!(digest.len(), 64);
            assert!(digest.bytes().all(|byte| byte.is_ascii_hexdigit()));
        }
        assert!(audit.comparison().accepted());
    }

    #[test]
    fn admitted_execution_rechecks_a_mutating_fingerprint() {
        let mut reference = ScaleBackend {
            descriptor: descriptor("reference", BackendKind::CpuReference),
            scale: 1.0,
        };
        let mut candidate = MutatingFingerprintBackend {
            descriptor: descriptor("gpu", BackendKind::Gpu),
            fingerprint_tag: 7,
            mutate_on_execute: false,
        };
        let mut worker = WorkerContext {
            worker_id: "gpu-worker".into(),
            ..WorkerContext::default()
        };
        worker
            .capabilities
            .insert(GPU_AVAILABLE_CAPABILITY.into(), 1.0);
        worker.labels.insert(GPU_API_LABEL.into(), "test".into());
        let work = independent_replica_work("diff", "run", 0, 1, 0, [(17, vec![2.0])])
            .unwrap()
            .remove(0);
        let authorization = reference_authorization(&reference.descriptor);
        let policy = policy(1e-12);
        let report = differential_check(
            &mut reference,
            &authorization,
            &mut candidate,
            &worker,
            &work,
            &policy,
            &VectorComparator { tolerance: 1e-12 },
        )
        .unwrap();

        candidate.mutate_on_execute = true;
        assert!(matches!(
            execute_work_unit(
                &mut candidate,
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
    }

    #[test]
    fn malformed_serialized_boundaries_and_extreme_partitions_are_rejected() {
        assert!(serde_json::from_str::<IndexChunk>(r#"{"start":9,"end":2}"#).is_err());
        assert!(serde_json::from_str::<BackendDescriptor>(
            r#"{"id":" ","implementation_version":"v1","kind":"cpu_reference","precision":"f64","deterministic":true,"requirements":[],"required_labels":{}}"#
        )
        .is_err());
        assert_eq!(
            partition_range(usize::MAX, 1),
            Err(AdapterError::SizeOverflow)
        );
    }

    #[test]
    fn gpu_descriptors_require_canonical_availability_and_api() {
        let arbitrary = BackendDescriptor::try_new(
            "gpu",
            "v1",
            BackendKind::Gpu,
            Precision::F32,
            false,
            vec![WorkerCapability {
                name: "cpu.cores".into(),
                min_value: Some(1.0),
            }],
            BTreeMap::from([("host.os".into(), "linux".into())]),
        );
        assert_eq!(arbitrary, Err(AdapterError::MissingGpuRequirement));

        let missing_api = BackendDescriptor::try_new(
            "gpu",
            "v1",
            BackendKind::Gpu,
            Precision::F32,
            false,
            vec![WorkerCapability {
                name: GPU_AVAILABLE_CAPABILITY.into(),
                min_value: Some(1.0),
            }],
            BTreeMap::new(),
        );
        assert_eq!(missing_api, Err(AdapterError::MissingGpuRequirement));
    }

    #[test]
    fn host_authorization_cannot_be_reused_for_another_reference() {
        let allowed = descriptor("reference-a", BackendKind::CpuReference);
        let denied = descriptor("reference-b", BackendKind::CpuReference);
        let authorization = authorize_reference(
            &|candidate: &BackendDescriptor| candidate.id() == "reference-a",
            &allowed,
        )
        .unwrap();
        assert_eq!(
            authorize_reference(&|_: &BackendDescriptor| false, &denied),
            Err(AdapterError::UnauthorizedReference)
        );

        let mut backend = ScaleBackend {
            descriptor: denied,
            scale: 1.0,
        };
        let worker = WorkerContext {
            worker_id: "worker".into(),
            ..WorkerContext::default()
        };
        let work = independent_replica_work("demo", "run", 0, 1, 0, [(1, vec![1.0])])
            .unwrap()
            .remove(0);
        assert!(matches!(
            execute_work_unit(
                &mut backend,
                &worker,
                &work,
                1,
                ExecutionAuthorization::Reference(&authorization)
            ),
            Err(ExecutionError::UnqualifiedBackend)
        ));
    }

    #[test]
    fn qualification_binds_full_descriptor_and_worker_context() {
        let mut reference = ScaleBackend {
            descriptor: descriptor("reference", BackendKind::CpuReference),
            scale: 1.0,
        };
        let mut candidate = ScaleBackend {
            descriptor: descriptor("gpu", BackendKind::Gpu),
            scale: 1.0,
        };
        let mut worker = WorkerContext {
            worker_id: "gpu-worker".into(),
            ..WorkerContext::default()
        };
        worker
            .capabilities
            .insert(GPU_AVAILABLE_CAPABILITY.into(), 1.0);
        worker.labels.insert(GPU_API_LABEL.into(), "test".into());
        let work = independent_replica_work("diff", "run", 0, 1, 0, [(3, vec![2.0])])
            .unwrap()
            .remove(0);
        let authorization = reference_authorization(&reference.descriptor);
        let policy = policy(1e-12);
        let report = differential_check(
            &mut reference,
            &authorization,
            &mut candidate,
            &worker,
            &work,
            &policy,
            &VectorComparator { tolerance: 1e-12 },
        )
        .unwrap();

        candidate.descriptor = BackendDescriptor::try_new(
            "gpu",
            "test-v1",
            BackendKind::Gpu,
            Precision::F32,
            true,
            vec![WorkerCapability {
                name: GPU_AVAILABLE_CAPABILITY.into(),
                min_value: Some(1.0),
            }],
            BTreeMap::from([(GPU_API_LABEL.into(), "test".into())]),
        )
        .unwrap();
        assert!(matches!(
            execute_work_unit(
                &mut candidate,
                &worker,
                &work,
                1,
                ExecutionAuthorization::Differential {
                    report: &report,
                    policy: &policy,
                }
            ),
            Err(ExecutionError::UnqualifiedBackend)
        ));

        candidate.descriptor = descriptor("gpu", BackendKind::Gpu);
        worker.labels.insert("device.uuid".into(), "changed".into());
        assert!(matches!(
            execute_work_unit(
                &mut candidate,
                &worker,
                &work,
                1,
                ExecutionAuthorization::Differential {
                    report: &report,
                    policy: &policy,
                }
            ),
            Err(ExecutionError::UnqualifiedBackend)
        ));
    }

    #[test]
    fn canonical_work_digest_ignores_map_insertion_order() {
        use std::collections::HashMap;

        let mut left_payload = HashMap::new();
        left_payload.insert("alpha", 1_u32);
        left_payload.insert("beta", 2_u32);
        let mut right_payload = HashMap::new();
        right_payload.insert("beta", 2_u32);
        right_payload.insert("alpha", 1_u32);
        let make_work = |payload| WorkUnit {
            experiment_id: "canonical".into(),
            instance_id: "run".into(),
            unit_id: "unit".into(),
            generation: 0,
            run: RunMetadata::default(),
            payload,
        };
        let left = make_work(left_payload);
        let right = make_work(right_payload);
        assert_eq!(
            canonical_digest(b"commutator.work-unit.v1", &left).unwrap(),
            canonical_digest(b"commutator.work-unit.v1", &right).unwrap()
        );
    }

    #[test]
    fn provenance_collisions_and_empty_workers_are_rejected() {
        let mut backend = ScaleBackend {
            descriptor: descriptor("reference", BackendKind::CpuReference),
            scale: 1.0,
        };
        let authorization = reference_authorization(&backend.descriptor);
        let mut work = independent_replica_work("demo", "run", 0, 1, 0, [(1, vec![1.0])])
            .unwrap()
            .remove(0);
        let empty_worker = WorkerContext::default();
        assert!(matches!(
            execute_work_unit(
                &mut backend,
                &empty_worker,
                &work,
                1,
                ExecutionAuthorization::Reference(&authorization)
            ),
            Err(ExecutionError::InvalidProvenance(
                AdapterError::EmptyWorkerIdentity
            ))
        ));

        let worker = WorkerContext {
            worker_id: "worker".into(),
            ..WorkerContext::default()
        };
        work.run
            .attributes
            .insert("worker_id".into(), "original".into());
        assert!(matches!(
            execute_work_unit(
                &mut backend,
                &worker,
                &work,
                1,
                ExecutionAuthorization::Reference(&authorization)
            ),
            Err(ExecutionError::InvalidProvenance(
                AdapterError::ProvenanceConflict(_)
            ))
        ));
    }

    #[test]
    fn replica_ordinal_overflow_is_rejected() {
        assert!(matches!(
            independent_replica_work("scan", "run", 0, 1, u64::MAX, [(1, ()), (2, ())]),
            Err(AdapterError::SizeOverflow)
        ));
    }
}
