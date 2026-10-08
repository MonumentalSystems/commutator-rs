#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::error::Error;

/// How work is divided across executors.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum ExperimentTopology {
    /// Adjacent workers own parts of a shared domain.
    DomainDecomposition {
        /// Number of spatial dimensions in the decomposed domain.
        dimensions: u8,
        /// Whether neighboring tiles exchange overlapping boundary data.
        uses_ghost_cells: bool,
    },
    /// Every work unit is an independent full-system replica.
    IndependentReplica,
    /// Independent chunks are combined by a reducer.
    MapReduce,
}

impl ExperimentTopology {
    /// Return the stable snake-case identifier for this topology.
    pub fn tag(&self) -> &'static str {
        match self {
            Self::DomainDecomposition { .. } => "domain_decomposition",
            Self::IndependentReplica => "independent_replica",
            Self::MapReduce => "map_reduce",
        }
    }
}

/// Framework-level result verification policy.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum VerificationStrategy {
    /// Commit to a result and recompute selected points as challenges.
    MerkleSpotCheck {
        /// Number of result points challenged per verification round.
        challenge_points: usize,
    },
    /// Reject results beyond a statistical distance from an expected cohort.
    StatisticalOutlier {
        /// Rejection threshold measured in standard deviations.
        sigma_threshold: f64,
    },
    /// Execute each work unit independently on multiple workers.
    Redundant {
        /// Number of independently computed copies required.
        replication_factor: usize,
    },
    /// Accept results without framework-level verification.
    None,
}

/// A numeric capability advertised by an executor.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkerCapability {
    /// Runtime-defined capability name.
    pub name: String,
    /// Optional minimum numeric value required for the capability.
    pub min_value: Option<f64>,
}

/// Static multi-node capabilities advertised by an experiment runtime.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExperimentMultiNodeCapabilities {
    /// Version of the experiment's multi-node protocol.
    pub protocol_version: u32,
    /// Whether workers can exchange data directly.
    pub direct_p2p: bool,
    /// Whether a relay may carry data when direct exchange is unavailable.
    pub relay_fallback: bool,
    /// Whether a proxy can subdivide work among child workers.
    pub proxy_fanout: bool,
}

/// Runtime geometry for tiled execution.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExperimentTileConfig {
    /// Extent of the complete logical domain in each dimension.
    pub global_shape: Vec<usize>,
    /// Preferred work-tile extent in each dimension.
    pub tile_shape: Vec<usize>,
    /// Number of neighboring cells carried on each tile boundary.
    pub ghost_depth: usize,
    /// Whether workers may exchange tile dependencies directly.
    pub direct_p2p: bool,
    /// Whether dependency payloads may pass through a relay.
    pub relay_fallback: bool,
    /// Whether proxies may subdivide assigned tiles.
    pub proxy_fanout: bool,
    /// Version of the tile protocol used by this configuration.
    pub protocol_version: u32,
}

/// Transport-neutral category of a dependency between work tiles.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TileDependencyKind {
    /// Neighboring-domain boundary values.
    Halo,
    /// Persisted state required to resume or seed a tile.
    Checkpoint,
    /// Coordination dependency with no scientific payload.
    ControlBarrier,
}

/// Framework-neutral executor identity and capabilities.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct WorkerContext {
    /// Runtime-defined stable identifier. It need not be a public key.
    pub worker_id: String,
    /// Numeric capabilities keyed by runtime-defined names.
    pub capabilities: BTreeMap<String, f64>,
    /// Free-form categorical attributes used for scheduling or provenance.
    pub labels: BTreeMap<String, String>,
}

/// Reproducibility metadata carried by work and results.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct RunMetadata {
    /// Deterministic random seed for the work unit.
    pub seed: u64,
    /// Optional compute-backend identifier.
    pub backend: Option<String>,
    /// Optional numeric-precision identifier.
    pub precision: Option<String>,
    /// Optional version of the implementation that produced the work.
    pub implementation_version: Option<String>,
    /// Version of the experiment's serialized payload schema.
    pub schema_version: u32,
    /// Additional stable provenance values defined by the experiment or host.
    pub attributes: BTreeMap<String, String>,
}

/// A typed unit of work issued by an experiment.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkUnit<T> {
    /// Stable identifier of the experiment definition.
    pub experiment_id: String,
    /// Identifier of the particular experiment run.
    pub instance_id: String,
    /// Identifier unique to this work unit within the run.
    pub unit_id: String,
    /// Revision of the run state that produced this work unit.
    pub generation: u64,
    /// Reproducibility metadata for this computation.
    pub run: RunMetadata,
    /// Experiment-defined work payload.
    pub payload: T,
}

/// A typed result returned for one work unit.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct WorkResult<T> {
    /// Stable identifier of the experiment definition.
    pub experiment_id: String,
    /// Identifier of the experiment run that issued the work.
    pub instance_id: String,
    /// Identifier copied from the completed work unit.
    pub unit_id: String,
    /// Run-state revision copied from the completed work unit.
    pub generation: u64,
    /// Reproducibility metadata used for the computation.
    pub run: RunMetadata,
    /// Wall-clock execution time reported by the executor.
    pub elapsed_ms: u64,
    /// Experiment-defined scalar measurements for observability or reduction.
    pub metrics: BTreeMap<String, f64>,
    /// Experiment-defined result payload.
    pub payload: T,
}

/// Outcome of integrating a result.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SubmitOutcome {
    /// The result was integrated and the experiment can issue more work.
    Accepted,
    /// The result was integrated and completed the experiment.
    Completed,
    /// The result could not be integrated, with a human-readable reason.
    Rejected(String),
}

/// Outcome of validating a result before integration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ValidationOutcome {
    /// The result passed validation.
    Valid,
    /// The result failed validation, with a human-readable reason.
    Invalid(String),
    /// Validation requires more evidence or asynchronous work.
    Pending,
}

/// Typed experiment lifecycle independent of any runtime or wire format.
pub trait Experiment: Send + Sync + 'static {
    /// Configuration supplied when starting an experiment run.
    type Config: Serialize + DeserializeOwned;
    /// Payload carried by each issued [`WorkUnit`].
    type Work: Serialize + DeserializeOwned;
    /// Payload carried by each submitted [`WorkResult`].
    type Result: Serialize + DeserializeOwned;
    /// Serializable snapshot exposed to operators and clients.
    type Status: Serialize;
    /// Error produced by experiment lifecycle methods.
    type Error: Error + Send + Sync + 'static;

    /// Return the stable identifier of this experiment definition.
    fn experiment_id(&self) -> &str;
    /// Describe how work is divided among executors.
    fn topology(&self) -> ExperimentTopology;
    /// Describe how returned results should be verified.
    fn verification_strategy(&self) -> VerificationStrategy;
    /// Initialize a new run from `config`.
    fn start(&mut self, config: Self::Config) -> Result<(), Self::Error>;
    /// Return the next work unit eligible for `worker`, if one is available.
    fn next_work_unit(
        &mut self,
        worker: &WorkerContext,
    ) -> Result<Option<WorkUnit<Self::Work>>, Self::Error>;
    /// Validate a result without integrating it into experiment state.
    fn validate_result(
        &self,
        worker: &WorkerContext,
        result: &WorkResult<Self::Result>,
    ) -> Result<ValidationOutcome, Self::Error>;
    /// Integrate a validated result and report its effect on the run.
    fn submit_result(
        &mut self,
        worker: &WorkerContext,
        result: WorkResult<Self::Result>,
    ) -> Result<SubmitOutcome, Self::Error>;
    /// Return the current serializable experiment status.
    fn status(&self) -> Self::Status;
    /// Request an orderly stop of the current run.
    fn stop(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn topology_tags_are_stable() {
        assert_eq!(
            ExperimentTopology::IndependentReplica.tag(),
            "independent_replica"
        );
        assert_eq!(ExperimentTopology::MapReduce.tag(), "map_reduce");
        assert_eq!(
            ExperimentTopology::DomainDecomposition {
                dimensions: 3,
                uses_ghost_cells: true,
            }
            .tag(),
            "domain_decomposition"
        );
    }

    #[test]
    fn run_metadata_defaults_are_deterministic() {
        let metadata = RunMetadata::default();
        assert_eq!(metadata.seed, 0);
        assert_eq!(metadata.schema_version, 0);
        assert!(metadata.attributes.is_empty());
    }

    #[test]
    fn typed_work_unit_json_roundtrip() {
        let unit = WorkUnit {
            experiment_id: "demo".into(),
            instance_id: "demo-1".into(),
            unit_id: "unit-7".into(),
            generation: 3,
            run: RunMetadata {
                seed: 42,
                schema_version: 1,
                ..RunMetadata::default()
            },
            payload: vec![1_u32, 2, 3],
        };

        let encoded = serde_json::to_string(&unit).unwrap();
        let decoded: WorkUnit<Vec<u32>> = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, unit);
    }
}
