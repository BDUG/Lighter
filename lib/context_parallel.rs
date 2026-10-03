//! Exact context-parallel attention primitives.
//!
//! The implementation is deliberately backend-neutral: it models the local
//! work performed by each rank and the numerically-stable `(max, sum, value)`
//! reduction needed to combine ranks. A GPU/NCCL backend can therefore use the
//! same partitioning and reduction contracts without changing the math.

use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextParallelError(pub String);

impl fmt::Display for ContextParallelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for ContextParallelError {}

/// How tokens are assigned to context-parallel ranks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartitionStrategy {
    /// Consecutive, equally-sized ranges. Best for decode and paged KV caches.
    Contiguous,
    /// Round-robin tokens. Balances causal prefill work across ranks.
    Striped,
    /// Mirrored stripes (the first and last blocks share a rank), balancing the
    /// triangular causal-attention workload while preserving block locality.
    ZigZag { block_size: usize },
}

/// A deterministic context partition plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContextPlan {
    pub assignments: Vec<Vec<usize>>,
}

impl ContextPlan {
    pub fn new(
        tokens: usize,
        ranks: usize,
        strategy: PartitionStrategy,
    ) -> Result<Self, ContextParallelError> {
        if ranks == 0 {
            return Err(ContextParallelError("rank count must be non-zero".into()));
        }
        if matches!(strategy, PartitionStrategy::ZigZag { block_size: 0 }) {
            return Err(ContextParallelError(
                "zig-zag block size must be non-zero".into(),
            ));
        }
        let mut assignments = vec![Vec::new(); ranks];
        match strategy {
            PartitionStrategy::Contiguous => {
                for token in 0..tokens {
                    assignments[token * ranks / tokens.max(1)].push(token);
                }
            }
            PartitionStrategy::Striped => {
                for token in 0..tokens {
                    assignments[token % ranks].push(token);
                }
            }
            PartitionStrategy::ZigZag { block_size } => {
                for token in 0..tokens {
                    let block = token / block_size;
                    let cycle = block / ranks;
                    let offset = block % ranks;
                    let rank = if cycle.is_multiple_of(2) {
                        offset
                    } else {
                        ranks - 1 - offset
                    };
                    assignments[rank].push(token);
                }
            }
        }
        Ok(Self { assignments })
    }

    pub fn owner(&self, token: usize) -> Option<usize> {
        self.assignments
            .iter()
            .position(|tokens| tokens.contains(&token))
    }
}

/// Key/value tensors owned by one rank.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvShard {
    pub rank: usize,
    /// Absolute position for each local token.
    pub positions: Vec<usize>,
    /// `[token][kv_head][head_dim]`.
    pub keys: Vec<Vec<Vec<f32>>>,
    /// `[token][kv_head][value_dim]`.
    pub values: Vec<Vec<Vec<f32>>>,
}

/// Decode-time attention settings.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DecodeConfig {
    /// Optional local-attention window. The current token is included.
    pub sliding_window: Option<usize>,
    /// Scale applied to QK scores; normally `1 / sqrt(head_dim)`.
    pub scale: f32,
    /// Cap logits using `cap * tanh(logit / cap)`, when configured.
    pub soft_cap: Option<f32>,
}

impl DecodeConfig {
    pub fn for_head_dim(head_dim: usize) -> Self {
        Self {
            sliding_window: None,
            scale: 1.0 / (head_dim as f32).sqrt(),
            soft_cap: None,
        }
    }
}

/// Associative online-softmax state. It is the payload reduced across ranks.
#[derive(Debug, Clone, PartialEq)]
pub struct SoftmaxState {
    pub max_logit: f32,
    pub exp_sum: f32,
    /// Exponentially weighted, not-yet-normalized value accumulator.
    pub weighted_value: Vec<f32>,
}

impl SoftmaxState {
    pub fn empty(value_dim: usize) -> Self {
        Self {
            max_logit: f32::NEG_INFINITY,
            exp_sum: 0.0,
            weighted_value: vec![0.0; value_dim],
        }
    }

    pub fn push(&mut self, logit: f32, value: &[f32]) -> Result<(), ContextParallelError> {
        if value.len() != self.weighted_value.len() {
            return Err(ContextParallelError(
                "value width changed during reduction".into(),
            ));
        }
        let new_max = self.max_logit.max(logit);
        let old_scale = (self.max_logit - new_max).exp();
        let new_scale = (logit - new_max).exp();
        self.exp_sum = self.exp_sum * old_scale + new_scale;
        for (acc, v) in self.weighted_value.iter_mut().zip(value) {
            *acc = *acc * old_scale + v * new_scale;
        }
        self.max_logit = new_max;
        Ok(())
    }

    /// Combine independently computed states. This operation is associative.
    pub fn merge(&mut self, other: &Self) -> Result<(), ContextParallelError> {
        if self.weighted_value.len() != other.weighted_value.len() {
            return Err(ContextParallelError(
                "cannot merge different value widths".into(),
            ));
        }
        if other.exp_sum == 0.0 {
            return Ok(());
        }
        if self.exp_sum == 0.0 {
            *self = other.clone();
            return Ok(());
        }
        let new_max = self.max_logit.max(other.max_logit);
        let a = (self.max_logit - new_max).exp();
        let b = (other.max_logit - new_max).exp();
        self.exp_sum = self.exp_sum * a + other.exp_sum * b;
        for (x, y) in self.weighted_value.iter_mut().zip(&other.weighted_value) {
            *x = *x * a + y * b;
        }
        self.max_logit = new_max;
        Ok(())
    }

    pub fn finish(&self) -> Result<Vec<f32>, ContextParallelError> {
        if self.exp_sum == 0.0 {
            return Err(ContextParallelError("attention has no visible keys".into()));
        }
        Ok(self
            .weighted_value
            .iter()
            .map(|v| v / self.exp_sum)
            .collect())
    }
}

/// Local result for one rank, suitable for a max/sum/weighted-value all-reduce.
#[derive(Debug, Clone, PartialEq)]
pub struct RankAttentionState {
    pub rank: usize,
    pub heads: Vec<SoftmaxState>,
    pub visited_tokens: usize,
}

/// Compute one rank's contribution for a single decode query.
pub fn local_decode(
    queries: &[Vec<f32>],
    query_position: usize,
    shard: &KvShard,
    config: DecodeConfig,
) -> Result<RankAttentionState, ContextParallelError> {
    validate_shard(queries, shard, config)?;
    let kv_heads = shard.keys[0].len();
    if !queries.len().is_multiple_of(kv_heads) {
        return Err(ContextParallelError(
            "query heads must be divisible by KV heads (GQA/MQA)".into(),
        ));
    }
    let value_dim = shard.values[0][0].len();
    let mut heads = vec![SoftmaxState::empty(value_dim); queries.len()];
    let mut visited_tokens = 0;
    for token in 0..shard.positions.len() {
        let position = shard.positions[token];
        if position > query_position {
            continue;
        }
        if config
            .sliding_window
            .is_some_and(|w| position + w <= query_position)
        {
            continue;
        }
        visited_tokens += 1;
        for (qh, query) in queries.iter().enumerate() {
            let kvh = qh * kv_heads / queries.len();
            let mut score = dot(query, &shard.keys[token][kvh])? * config.scale;
            if let Some(cap) = config.soft_cap {
                if cap <= 0.0 {
                    return Err(ContextParallelError("soft cap must be positive".into()));
                }
                score = cap * (score / cap).tanh();
            }
            heads[qh].push(score, &shard.values[token][kvh])?;
        }
    }
    Ok(RankAttentionState {
        rank: shard.rank,
        heads,
        visited_tokens,
    })
}

/// Merge rank-local decode results into the exact dense-attention result.
pub fn reduce_decode(states: &[RankAttentionState]) -> Result<Vec<Vec<f32>>, ContextParallelError> {
    let first = states
        .first()
        .ok_or_else(|| ContextParallelError("no rank states supplied".into()))?;
    let mut combined = first.heads.clone();
    for state in &states[1..] {
        if state.heads.len() != combined.len() {
            return Err(ContextParallelError("rank head counts differ".into()));
        }
        for (dst, src) in combined.iter_mut().zip(&state.heads) {
            dst.merge(src)?;
        }
    }
    combined.iter().map(SoftmaxState::finish).collect()
}

/// Simulate exact decode-context parallelism over all shards.
pub fn decode_context_parallel(
    queries: &[Vec<f32>],
    query_position: usize,
    shards: &[KvShard],
    config: DecodeConfig,
) -> Result<Vec<Vec<f32>>, ContextParallelError> {
    let states: Result<Vec<_>, _> = shards
        .iter()
        .map(|s| local_decode(queries, query_position, s, config))
        .collect();
    reduce_decode(&states?)
}

/// Run context-parallel causal prefill. Each query may have multiple Q heads.
pub fn causal_prefill(
    queries: &[Vec<Vec<f32>>],
    query_positions: &[usize],
    shards: &[KvShard],
    config: DecodeConfig,
) -> Result<Vec<Vec<Vec<f32>>>, ContextParallelError> {
    if queries.len() != query_positions.len() {
        return Err(ContextParallelError(
            "query and position counts differ".into(),
        ));
    }
    queries
        .iter()
        .zip(query_positions)
        .map(|(q, &p)| decode_context_parallel(q, p, shards, config))
        .collect()
}

/// Split dense KV tensors according to a plan (useful for tests and CPU serving).
pub fn shard_kv(
    keys: &[Vec<Vec<f32>>],
    values: &[Vec<Vec<f32>>],
    plan: &ContextPlan,
) -> Result<Vec<KvShard>, ContextParallelError> {
    if keys.len() != values.len() {
        return Err(ContextParallelError("key/value token counts differ".into()));
    }
    let mut seen = vec![false; keys.len()];
    plan.assignments
        .iter()
        .enumerate()
        .map(|(rank, positions)| {
            let mut k = Vec::with_capacity(positions.len());
            let mut v = Vec::with_capacity(positions.len());
            for &p in positions {
                if p >= keys.len() {
                    return Err(ContextParallelError(format!(
                        "token {p} is outside KV cache"
                    )));
                }
                if seen[p] {
                    return Err(ContextParallelError(format!(
                        "token {p} assigned more than once"
                    )));
                }
                seen[p] = true;
                k.push(keys[p].clone());
                v.push(values[p].clone());
            }
            Ok(KvShard {
                rank,
                positions: positions.clone(),
                keys: k,
                values: v,
            })
        })
        .collect::<Result<Vec<_>, _>>()
        .and_then(|shards| {
            if seen.iter().any(|x| !x) {
                Err(ContextParallelError("partition omits KV tokens".into()))
            } else {
                Ok(shards)
            }
        })
}

fn validate_shard(
    q: &[Vec<f32>],
    s: &KvShard,
    c: DecodeConfig,
) -> Result<(), ContextParallelError> {
    if q.is_empty() {
        return Err(ContextParallelError(
            "at least one query head is required".into(),
        ));
    }
    if s.keys.is_empty() || s.keys.len() != s.values.len() || s.keys.len() != s.positions.len() {
        return Err(ContextParallelError(
            "shard key/value/position lengths are empty or differ".into(),
        ));
    }
    let kvh = s.keys[0].len();
    if kvh == 0 || s.values[0].len() != kvh {
        return Err(ContextParallelError("KV head count is invalid".into()));
    }
    let d = q[0].len();
    let vd = s.values[0][0].len();
    if d == 0 || vd == 0 || !c.scale.is_finite() {
        return Err(ContextParallelError(
            "dimensions and scale must be finite/non-zero".into(),
        ));
    }
    if q.iter().any(|x| x.len() != d)
        || s.keys
            .iter()
            .any(|t| t.len() != kvh || t.iter().any(|x| x.len() != d))
        || s.values
            .iter()
            .any(|t| t.len() != kvh || t.iter().any(|x| x.len() != vd))
    {
        return Err(ContextParallelError("ragged Q/K/V tensor".into()));
    }
    Ok(())
}

fn dot(a: &[f32], b: &[f32]) -> Result<f32, ContextParallelError> {
    if a.len() != b.len() {
        return Err(ContextParallelError("Q/K head dimensions differ".into()));
    }
    Ok(a.iter().zip(b).map(|(x, y)| x * y).sum())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn data() -> (Vec<Vec<Vec<f32>>>, Vec<Vec<Vec<f32>>>) {
        let k = (0..7)
            .map(|i| vec![vec![i as f32 * 0.1, 1.], vec![1., i as f32 * 0.2]])
            .collect();
        let v = (0..7)
            .map(|i| vec![vec![i as f32, 1.], vec![-(i as f32), 2.]])
            .collect();
        (k, v)
    }
    #[test]
    fn partition_owns_every_token_once() {
        for strategy in [
            PartitionStrategy::Contiguous,
            PartitionStrategy::Striped,
            PartitionStrategy::ZigZag { block_size: 2 },
        ] {
            let p = ContextPlan::new(17, 3, strategy).unwrap();
            let mut all = p.assignments.concat();
            all.sort_unstable();
            assert_eq!(all, (0..17).collect::<Vec<_>>());
        }
    }
    #[test]
    fn distributed_matches_single_rank_for_gqa() {
        let (k, v) = data();
        let q = vec![
            vec![0.2, 0.7],
            vec![0.4, 0.3],
            vec![0.5, 0.1],
            vec![0.1, 0.9],
        ];
        let c = DecodeConfig::for_head_dim(2);
        let one = shard_kv(
            &k,
            &v,
            &ContextPlan::new(7, 1, PartitionStrategy::Contiguous).unwrap(),
        )
        .unwrap();
        let expected = decode_context_parallel(&q, 6, &one, c).unwrap();
        for strategy in [
            PartitionStrategy::Contiguous,
            PartitionStrategy::Striped,
            PartitionStrategy::ZigZag { block_size: 2 },
        ] {
            let shards = shard_kv(&k, &v, &ContextPlan::new(7, 3, strategy).unwrap()).unwrap();
            let got = decode_context_parallel(&q, 6, &shards, c).unwrap();
            for (a, b) in got.iter().flatten().zip(expected.iter().flatten()) {
                assert!((a - b).abs() < 1e-5);
            }
        }
    }
    #[test]
    fn causal_window_and_prefill_work() {
        let (k, v) = data();
        let shards = shard_kv(
            &k,
            &v,
            &ContextPlan::new(7, 2, PartitionStrategy::Striped).unwrap(),
        )
        .unwrap();
        let q = vec![vec![0.2, 0.7], vec![0.4, 0.3]];
        let mut c = DecodeConfig::for_head_dim(2);
        c.sliding_window = Some(2);
        let states: Vec<_> = shards
            .iter()
            .map(|s| local_decode(&q, 4, s, c).unwrap())
            .collect();
        assert_eq!(states.iter().map(|s| s.visited_tokens).sum::<usize>(), 2);
        assert_eq!(
            causal_prefill(&[q.clone(), q], &[3, 4], &shards, c)
                .unwrap()
                .len(),
            2
        );
    }
    #[test]
    fn online_merge_is_associative_and_stable() {
        let mut a = SoftmaxState::empty(1);
        a.push(10000., &[2.]).unwrap();
        let mut b = SoftmaxState::empty(1);
        b.push(9999., &[4.]).unwrap();
        a.merge(&b).unwrap();
        let expected = (2. + 4. * (-1.0f32).exp()) / (1. + (-1.0f32).exp());
        assert!((a.finish().unwrap()[0] - expected).abs() < 1e-5);
    }
}
