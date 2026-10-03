//! Vector-valued attention for sets and point clouds.
//!
//! Unlike dot-product attention, which assigns one scalar to every query/key
//! pair, vector attention learns one weight per output channel.  This is the
//! formulation commonly used by point-transformer layers:
//!
//! `y_i = sum_j softmax_j(gamma(q_i - k_j + delta_ij)) * (v_j + delta_ij)`.
//!
//! This module is backend independent and intentionally operates on ordinary
//! Rust slices.  It supports self- and cross-attention, dense or k-nearest
//! neighbourhoods, masks, grouped attention weights, and batched inputs.

use rand::{rngs::StdRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt;

/// Errors returned by vector-attention operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VecAttentionError(pub String);

impl fmt::Display for VecAttentionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for VecAttentionError {}

/// A small, serializable affine projection (`y = x W^T + b`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Linear {
    pub input_dim: usize,
    pub output_dim: usize,
    /// Row-major `[output_dim, input_dim]` weights.
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

impl Linear {
    pub fn new(input_dim: usize, output_dim: usize, rng: &mut impl Rng) -> Self {
        let limit = (6.0f32 / (input_dim + output_dim).max(1) as f32).sqrt();
        let weight = (0..input_dim * output_dim)
            .map(|_| rng.gen_range(-limit..=limit))
            .collect();
        Self {
            input_dim,
            output_dim,
            weight,
            bias: vec![0.0; output_dim],
        }
    }

    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>, VecAttentionError> {
        if x.len() != self.input_dim {
            return Err(VecAttentionError(format!(
                "linear expected {} features, got {}",
                self.input_dim,
                x.len()
            )));
        }
        Ok((0..self.output_dim)
            .map(|o| {
                self.bias[o]
                    + self.weight[o * self.input_dim..(o + 1) * self.input_dim]
                        .iter()
                        .zip(x)
                        .map(|(w, v)| w * v)
                        .sum::<f32>()
            })
            .collect())
    }
}

/// Neighbourhood used for each query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Neighborhood {
    /// Attend to every key.
    Global,
    /// Attend to the `k` closest keys according to Euclidean position.
    KNearest(usize),
}

/// Construction options for [`VecAttention`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VecAttentionConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    pub position_dim: usize,
    /// Number of adjacent channels that share an attention logit.
    pub share_planes: usize,
    pub neighborhood: Neighborhood,
    pub seed: u64,
}

impl VecAttentionConfig {
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dim,
            output_dim: input_dim,
            position_dim: 3,
            share_planes: 1,
            neighborhood: Neighborhood::Global,
            seed: 42,
        }
    }

    fn validate(&self) -> Result<(), VecAttentionError> {
        if self.input_dim == 0
            || self.hidden_dim == 0
            || self.output_dim == 0
            || self.position_dim == 0
        {
            return Err(VecAttentionError("all dimensions must be non-zero".into()));
        }
        if self.share_planes == 0 || !self.hidden_dim.is_multiple_of(self.share_planes) {
            return Err(VecAttentionError(
                "hidden_dim must be divisible by non-zero share_planes".into(),
            ));
        }
        if matches!(self.neighborhood, Neighborhood::KNearest(0)) {
            return Err(VecAttentionError("KNearest requires k > 0".into()));
        }
        Ok(())
    }
}

/// Optional pair mask. Rows are queries and columns are keys; `false` blocks a pair.
pub type AttentionMask = Vec<Vec<bool>>;

/// Output values and the normalized vector-valued weights used to produce them.
#[derive(Debug, Clone, PartialEq)]
pub struct VecAttentionOutput {
    pub values: Vec<Vec<f32>>,
    /// `[query][key][hidden_dim / share_planes]`; masked/non-neighbour entries are zero.
    pub weights: Vec<Vec<Vec<f32>>>,
}

/// Learnable vector-attention layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VecAttention {
    pub config: VecAttentionConfig,
    pub query: Linear,
    pub key: Linear,
    pub value: Linear,
    /// First positional MLP projection, followed by ReLU.
    pub position_in: Linear,
    pub position_out: Linear,
    /// Two-layer relation MLP `hidden -> hidden -> groups`.
    pub relation_in: Linear,
    pub relation_out: Linear,
    pub output: Linear,
}

impl VecAttention {
    pub fn new(config: VecAttentionConfig) -> Result<Self, VecAttentionError> {
        config.validate()?;
        let mut rng = StdRng::seed_from_u64(config.seed);
        let groups = config.hidden_dim / config.share_planes;
        Ok(Self {
            query: Linear::new(config.input_dim, config.hidden_dim, &mut rng),
            key: Linear::new(config.input_dim, config.hidden_dim, &mut rng),
            value: Linear::new(config.input_dim, config.hidden_dim, &mut rng),
            position_in: Linear::new(config.position_dim, config.hidden_dim, &mut rng),
            position_out: Linear::new(config.hidden_dim, config.hidden_dim, &mut rng),
            relation_in: Linear::new(config.hidden_dim, config.hidden_dim, &mut rng),
            relation_out: Linear::new(config.hidden_dim, groups, &mut rng),
            output: Linear::new(config.hidden_dim, config.output_dim, &mut rng),
            config,
        })
    }

    /// Self-attention over one set of features and positions.
    pub fn forward(
        &self,
        features: &[Vec<f32>],
        positions: &[Vec<f32>],
        mask: Option<&AttentionMask>,
    ) -> Result<VecAttentionOutput, VecAttentionError> {
        self.forward_cross(features, positions, features, positions, mask)
    }

    /// Cross-attention from `queries` to a separate key/value set.
    pub fn forward_cross(
        &self,
        queries: &[Vec<f32>],
        query_positions: &[Vec<f32>],
        keys_values: &[Vec<f32>],
        key_positions: &[Vec<f32>],
        mask: Option<&AttentionMask>,
    ) -> Result<VecAttentionOutput, VecAttentionError> {
        self.validate_inputs(queries, query_positions, keys_values, key_positions, mask)?;
        let q = project_rows(&self.query, queries)?;
        let k = project_rows(&self.key, keys_values)?;
        let v = project_rows(&self.value, keys_values)?;
        let nq = queries.len();
        let nk = keys_values.len();
        let groups = self.config.hidden_dim / self.config.share_planes;
        let mut all_weights = vec![vec![vec![0.0; groups]; nk]; nq];
        let mut values = Vec::with_capacity(nq);

        for i in 0..nq {
            let neighbours =
                self.neighbours(&query_positions[i], key_positions, mask.map(|m| &m[i]))?;
            if neighbours.is_empty() {
                return Err(VecAttentionError(format!("query {i} has no unmasked keys")));
            }
            let mut enriched = Vec::with_capacity(neighbours.len());
            let mut logits = Vec::with_capacity(neighbours.len());
            for &j in &neighbours {
                let offset: Vec<f32> = query_positions[i]
                    .iter()
                    .zip(&key_positions[j])
                    .map(|(a, b)| a - b)
                    .collect();
                let mut delta = relu(self.position_in.forward(&offset)?);
                delta = self.position_out.forward(&delta)?;
                let relation: Vec<f32> = q[i]
                    .iter()
                    .zip(&k[j])
                    .zip(&delta)
                    .map(|((qv, kv), d)| qv - kv + d)
                    .collect();
                let score = self
                    .relation_out
                    .forward(&relu(self.relation_in.forward(&relation)?))?;
                enriched.push(
                    v[j].iter()
                        .zip(&delta)
                        .map(|(vv, d)| vv + d)
                        .collect::<Vec<_>>(),
                );
                logits.push(score);
            }
            // Stable softmax independently for every vector/group component.
            for g in 0..groups {
                let max = logits
                    .iter()
                    .map(|x| x[g])
                    .fold(f32::NEG_INFINITY, f32::max);
                let denom: f32 = logits.iter().map(|x| (x[g] - max).exp()).sum();
                for (n, &j) in neighbours.iter().enumerate() {
                    all_weights[i][j][g] = (logits[n][g] - max).exp() / denom;
                }
            }
            let mut aggregate = vec![0.0; self.config.hidden_dim];
            for (n, &j) in neighbours.iter().enumerate() {
                for (c, out) in aggregate.iter_mut().enumerate() {
                    *out += all_weights[i][j][c / self.config.share_planes] * enriched[n][c];
                }
            }
            values.push(self.output.forward(&aggregate)?);
        }
        Ok(VecAttentionOutput {
            values,
            weights: all_weights,
        })
    }

    /// Apply self-attention independently to a batch of variable-sized sets.
    pub fn forward_batch(
        &self,
        features: &[Vec<Vec<f32>>],
        positions: &[Vec<Vec<f32>>],
    ) -> Result<Vec<Vec<Vec<f32>>>, VecAttentionError> {
        if features.len() != positions.len() {
            return Err(VecAttentionError(
                "feature and position batch sizes differ".into(),
            ));
        }
        features
            .iter()
            .zip(positions)
            .map(|(x, p)| self.forward(x, p, None).map(|o| o.values))
            .collect()
    }

    fn neighbours(
        &self,
        query: &[f32],
        keys: &[Vec<f32>],
        row_mask: Option<&Vec<bool>>,
    ) -> Result<Vec<usize>, VecAttentionError> {
        let mut candidates: Vec<(usize, f32)> = keys
            .iter()
            .enumerate()
            .filter(|(j, _)| row_mask.is_none_or(|m| m[*j]))
            .map(|(j, p)| (j, query.iter().zip(p).map(|(a, b)| (a - b) * (a - b)).sum()))
            .collect();
        if let Neighborhood::KNearest(k) = self.config.neighborhood {
            candidates.sort_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(Ordering::Equal)
                    .then(a.0.cmp(&b.0))
            });
            candidates.truncate(k.min(candidates.len()));
        }
        Ok(candidates.into_iter().map(|x| x.0).collect())
    }

    fn validate_inputs(
        &self,
        q: &[Vec<f32>],
        qp: &[Vec<f32>],
        kv: &[Vec<f32>],
        kp: &[Vec<f32>],
        mask: Option<&AttentionMask>,
    ) -> Result<(), VecAttentionError> {
        if q.is_empty() || kv.is_empty() {
            return Err(VecAttentionError(
                "query and key sets must be non-empty".into(),
            ));
        }
        if q.len() != qp.len() || kv.len() != kp.len() {
            return Err(VecAttentionError("every feature needs a position".into()));
        }
        if q.iter().chain(kv).any(|x| x.len() != self.config.input_dim) {
            return Err(VecAttentionError(format!(
                "feature width must be {}",
                self.config.input_dim
            )));
        }
        if qp
            .iter()
            .chain(kp)
            .any(|x| x.len() != self.config.position_dim)
        {
            return Err(VecAttentionError(format!(
                "position width must be {}",
                self.config.position_dim
            )));
        }
        if let Some(m) = mask {
            if m.len() != q.len() || m.iter().any(|r| r.len() != kv.len()) {
                return Err(VecAttentionError(
                    "mask shape must be [queries, keys]".into(),
                ));
            }
        }
        Ok(())
    }
}

fn project_rows(layer: &Linear, rows: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, VecAttentionError> {
    rows.iter().map(|x| layer.forward(x)).collect()
}
fn relu(mut x: Vec<f32>) -> Vec<f32> {
    x.iter_mut().for_each(|v| *v = v.max(0.0));
    x
}

/// Create the usual lower-triangular causal mask for sequence self-attention.
pub fn causal_mask(length: usize) -> AttentionMask {
    (0..length)
        .map(|i| (0..length).map(|j| j <= i).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        (
            vec![vec![1., 0., 0.5], vec![0., 1., 0.5], vec![1., 1., 0.]],
            vec![vec![0., 0.], vec![1., 0.], vec![0., 2.]],
        )
    }
    fn layer(neighborhood: Neighborhood) -> VecAttention {
        let mut c = VecAttentionConfig::new(3, 4);
        c.output_dim = 2;
        c.position_dim = 2;
        c.share_planes = 2;
        c.neighborhood = neighborhood;
        VecAttention::new(c).unwrap()
    }
    #[test]
    fn vector_weights_are_normalized() {
        let (x, p) = fixture();
        let y = layer(Neighborhood::Global).forward(&x, &p, None).unwrap();
        assert_eq!(y.values.len(), 3);
        assert_eq!(y.values[0].len(), 2);
        for i in 0..3 {
            for g in 0..2 {
                let s: f32 = y.weights[i].iter().map(|w| w[g]).sum();
                assert!((s - 1.).abs() < 1e-5);
            }
        }
    }
    #[test]
    fn mask_and_knn_zero_non_neighbours() {
        let (x, p) = fixture();
        let y = layer(Neighborhood::KNearest(1))
            .forward(&x, &p, None)
            .unwrap();
        assert!(y
            .weights
            .iter()
            .all(|row| row.iter().filter(|w| w[0] > 0.).count() == 1));
        let m = causal_mask(3);
        let y = layer(Neighborhood::Global)
            .forward(&x, &p, Some(&m))
            .unwrap();
        assert!(y.weights[0][1][0] == 0. && y.weights[1][2][1] == 0.);
    }
    #[test]
    fn deterministic_and_serializable() {
        let a = layer(Neighborhood::Global);
        let b = layer(Neighborhood::Global);
        assert_eq!(a.query.weight, b.query.weight);
        let restored: VecAttention =
            serde_json::from_str(&serde_json::to_string(&a).unwrap()).unwrap();
        let (x, p) = fixture();
        assert_eq!(
            a.forward(&x, &p, None).unwrap().values,
            restored.forward(&x, &p, None).unwrap().values
        );
    }
    #[test]
    fn cross_attention_and_validation() {
        let (x, p) = fixture();
        let l = layer(Neighborhood::Global);
        let y = l
            .forward_cross(&x[..1], &p[..1], &x[1..], &p[1..], None)
            .unwrap();
        assert_eq!(y.weights[0].len(), 2);
        assert!(l.forward(&[], &[], None).is_err());
    }
}
