use std::collections::BTreeMap;
use std::marker::PhantomData;

use serde::Serialize;

use crate::{
    canonical_digest, partition_range, AdapterError, BackendDescriptor, BackendIdentity,
    BackendKind, BackendOutput, ComputeBackend, Precision,
};

const PARTITION_ALGORITHM: &str = "contiguous-ceiling-at-most-one-per-child-v1";
const SEED_ALGORITHM: &str = "splitmix64-parent-seed-and-shard-ordinal-v1";

#[derive(Serialize)]
struct ShardedExecutionIdentity<'a> {
    schema: &'static str,
    partition_algorithm: &'static str,
    seed_algorithm: &'static str,
    child_count: usize,
    children: Vec<ShardedChildIdentity<'a>>,
}

#[derive(Serialize)]
struct ShardedChildIdentity<'a> {
    ordinal: usize,
    descriptor: &'a BackendDescriptor,
    execution_fingerprint_sha256: [u8; 32],
}

/// Derives a deterministic, distinct child seed from a parent seed and shard ordinal.
#[must_use]
pub fn shard_seed(parent: u64, ordinal: usize) -> u64 {
    let mut value = parent ^ (ordinal as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

/// Failure from deterministic threaded shard execution.
#[derive(Debug, PartialEq)]
pub enum ShardedError<E> {
    /// One child backend failed its assigned shard.
    Child {
        /// Stable zero-based shard ordinal.
        shard: usize,
        /// Child error.
        source: E,
    },
    /// A child thread panicked.
    ChildPanicked {
        /// Stable zero-based shard ordinal.
        shard: usize,
    },
    /// A child returned a different number of results than inputs.
    ResultLength {
        /// Stable zero-based shard ordinal.
        shard: usize,
        /// Required result count.
        expected: usize,
        /// Returned result count.
        actual: usize,
    },
    /// Partition construction exceeded an addressable allocation.
    Partition(AdapterError),
    /// Aggregated metrics violated the finite-value contract.
    InvalidMetrics(AdapterError),
}

impl<E: std::fmt::Display> std::fmt::Display for ShardedError<E> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Child { shard, source } => write!(formatter, "shard {shard} failed: {source}"),
            Self::ChildPanicked { shard } => write!(formatter, "shard {shard} panicked"),
            Self::ResultLength {
                shard,
                expected,
                actual,
            } => write!(
                formatter,
                "shard {shard} returned {actual} results, expected {expected}"
            ),
            Self::Partition(error) => write!(formatter, "failed to partition work: {error}"),
            Self::InvalidMetrics(error) => write!(formatter, "invalid shard metrics: {error}"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for ShardedError<E> {}

/// Deterministic in-process distributed adapter with one threaded child per shard.
///
/// The adapter partitions a vector into at most `children.len()` contiguous
/// shards. Every child executes once with a derived seed. Results are joined in
/// shard order, independent of completion order, and child metrics are namespaced
/// as `shard.<ordinal>.<name>`.
#[derive(Debug)]
pub struct ThreadedShardedBackend<B, W, R> {
    descriptor: BackendDescriptor,
    children: Vec<B>,
    item_types: PhantomData<fn(W) -> R>,
}

impl<B, W, R> ThreadedShardedBackend<B, W, R>
where
    B: ComputeBackend<Vec<W>, Vec<R>>,
{
    /// Constructs a distributed-proxy descriptor derived from nonempty children.
    ///
    /// Determinism is claimed only when every child claims it. Precision is the
    /// common child precision or [`Precision::Mixed`] when children differ.
    /// Numeric requirements are all retained, while conflicting exact label
    /// requirements are rejected because no worker could satisfy them.
    pub fn try_new(
        id: impl Into<String>,
        implementation_version: impl Into<String>,
        children: Vec<B>,
    ) -> Result<Self, AdapterError> {
        if children.is_empty() {
            return Err(AdapterError::EmptyIdentifier("sharded child workers"));
        }
        let deterministic = children
            .iter()
            .all(|child| child.descriptor().deterministic());
        let first_precision = children[0].descriptor().precision();
        let precision = if children
            .iter()
            .all(|child| child.descriptor().precision() == first_precision)
        {
            first_precision
        } else {
            Precision::Mixed
        };
        let requirements = children
            .iter()
            .flat_map(|child| child.descriptor().requirements().iter().cloned())
            .collect();
        let mut required_labels = BTreeMap::new();
        for child in &children {
            for (name, value) in child.descriptor().required_labels() {
                if required_labels
                    .insert(name.clone(), value.clone())
                    .is_some_and(|previous| previous != *value)
                {
                    return Err(AdapterError::InvalidCapability);
                }
            }
        }
        Ok(Self {
            descriptor: BackendDescriptor::try_new(
                id,
                implementation_version,
                BackendKind::DistributedProxy,
                precision,
                deterministic,
                requirements,
                required_labels,
            )?,
            children,
            item_types: PhantomData,
        })
    }

    /// Returns the number of in-process child workers.
    #[must_use]
    pub fn child_count(&self) -> usize {
        self.children.len()
    }

    /// Returns immutable child backends in stable shard-assignment order.
    #[must_use]
    pub fn children(&self) -> &[B] {
        &self.children
    }
}

impl<B, W, R> BackendIdentity for ThreadedShardedBackend<B, W, R>
where
    B: BackendIdentity,
{
    fn descriptor(&self) -> &BackendDescriptor {
        &self.descriptor
    }

    fn execution_fingerprint(&self) -> Result<[u8; 32], AdapterError> {
        let children = self
            .children
            .iter()
            .enumerate()
            .map(|(ordinal, child)| {
                Ok(ShardedChildIdentity {
                    ordinal,
                    descriptor: child.descriptor(),
                    execution_fingerprint_sha256: child.execution_fingerprint()?,
                })
            })
            .collect::<Result<Vec<_>, AdapterError>>()?;
        canonical_digest(
            b"commutator.threaded-sharded-execution.v1",
            &ShardedExecutionIdentity {
                schema: "commutator.threaded-sharded-execution.v1",
                partition_algorithm: PARTITION_ALGORITHM,
                seed_algorithm: SEED_ALGORITHM,
                child_count: children.len(),
                children,
            },
        )
    }
}

impl<W, R, B> ComputeBackend<Vec<W>, Vec<R>> for ThreadedShardedBackend<B, W, R>
where
    W: Clone + Send + Sync,
    R: Send,
    B: ComputeBackend<Vec<W>, Vec<R>> + Send,
    B::Error: Send,
{
    type Error = ShardedError<B::Error>;

    fn execute(
        &mut self,
        payload: &Vec<W>,
        seed: u64,
    ) -> Result<BackendOutput<Vec<R>>, Self::Error> {
        if payload.is_empty() {
            let metrics = BTreeMap::from([
                ("shard.count".to_owned(), 0.0),
                ("item.count".to_owned(), 0.0),
            ]);
            return BackendOutput::try_new(Vec::new(), metrics)
                .map_err(ShardedError::InvalidMetrics);
        }

        let chunk_size = payload.len().div_ceil(self.children.len());
        let chunks = partition_range(payload.len(), chunk_size).map_err(ShardedError::Partition)?;
        let outcomes = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(chunks.len());
            for (shard, (chunk, child)) in chunks
                .iter()
                .copied()
                .zip(self.children.iter_mut())
                .enumerate()
            {
                let input = payload[chunk.start()..chunk.end()].to_vec();
                handles.push((
                    shard,
                    chunk,
                    scope.spawn(move || child.execute(&input, shard_seed(seed, shard))),
                ));
            }

            let mut outcomes = Vec::with_capacity(handles.len());
            let mut first_error = None;
            for (shard, chunk, handle) in handles {
                let outcome = match handle.join() {
                    Ok(Ok(output)) if output.payload().len() == chunk.len() => {
                        Ok((shard, chunk, output))
                    }
                    Ok(Ok(output)) => Err(ShardedError::ResultLength {
                        shard,
                        expected: chunk.len(),
                        actual: output.payload().len(),
                    }),
                    Ok(Err(source)) => Err(ShardedError::Child { shard, source }),
                    Err(_) => Err(ShardedError::ChildPanicked { shard }),
                };
                match outcome {
                    Ok(output) => outcomes.push(output),
                    Err(error) if first_error.is_none() => first_error = Some(error),
                    Err(_) => {}
                }
            }
            first_error.map_or(Ok(outcomes), Err)
        })?;

        let mut ordered = Vec::new();
        ordered
            .try_reserve_exact(payload.len())
            .map_err(|_| ShardedError::Partition(AdapterError::SizeOverflow))?;
        let mut metrics = BTreeMap::new();
        metrics.insert("shard.count".to_owned(), outcomes.len() as f64);
        metrics.insert("item.count".to_owned(), payload.len() as f64);
        for (shard, chunk, output) in outcomes {
            let (values, child_metrics) = output.into_parts();
            debug_assert_eq!(values.len(), chunk.len());
            ordered.extend(values);
            metrics.extend(
                child_metrics
                    .into_iter()
                    .map(|(name, value)| (format!("shard.{shard}.{name}"), value)),
            );
        }
        BackendOutput::try_new(ordered, metrics).map_err(ShardedError::InvalidMetrics)
    }
}
