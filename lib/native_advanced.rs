//! Backend-neutral optimization and memory primitives for native inference.
//!
//! The traits in this module deliberately exchange ordinary vectors. Tensor
//! runtimes retain ownership of forward/backward execution while Lighter owns
//! the numerically sensitive objectives, cache lifecycle, and quantization.

use crate::native::{NativeError, NativeResult};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationType {
    Int8,
    Int4,
}

#[derive(Debug, Clone, Copy)]
pub struct QuantizationConfig {
    pub dtype: QuantizationType,
    pub group_size: usize,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            dtype: QuantizationType::Int8,
            group_size: 64,
        }
    }
}

/// Symmetric group-wise quantized row-major matrix. Int4 values are packed two
/// per byte; scales remain f32 to avoid accumulating scale quantization error.
#[derive(Debug, Clone)]
pub struct QuantizedMatrix {
    rows: usize,
    cols: usize,
    config: QuantizationConfig,
    values: Vec<u8>,
    scales: Vec<f32>,
}

impl QuantizedMatrix {
    pub fn quantize(
        rows: usize,
        cols: usize,
        values: &[f32],
        config: QuantizationConfig,
    ) -> NativeResult<Self> {
        if rows.checked_mul(cols) != Some(values.len()) {
            return Err(NativeError("quantized matrix shape mismatch".into()));
        }
        if config.group_size == 0 {
            return Err(NativeError(
                "quantization group_size must be positive".into(),
            ));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(NativeError("cannot quantize non-finite values".into()));
        }
        let levels = match config.dtype {
            QuantizationType::Int8 => 127.0,
            QuantizationType::Int4 => 7.0,
        };
        let groups = values.len().div_ceil(config.group_size);
        let mut scales = Vec::with_capacity(groups);
        let mut codes = Vec::with_capacity(values.len());
        for group in values.chunks(config.group_size) {
            let maximum = group.iter().copied().map(f32::abs).fold(0.0, f32::max);
            let scale = if maximum == 0.0 {
                1.0
            } else {
                maximum / levels
            };
            scales.push(scale);
            for value in group {
                codes.push((value / scale).round().clamp(-levels, levels) as i8);
            }
        }
        let values = match config.dtype {
            QuantizationType::Int8 => codes.into_iter().map(|value| value as u8).collect(),
            QuantizationType::Int4 => codes
                .chunks(2)
                .map(|pair| {
                    let low = (pair[0] as u8) & 0x0f;
                    let high = pair.get(1).copied().unwrap_or(0) as u8 & 0x0f;
                    low | (high << 4)
                })
                .collect(),
        };
        Ok(Self {
            rows,
            cols,
            config,
            values,
            scales,
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn storage_bytes(&self) -> usize {
        self.values.len() + self.scales.len() * 4
    }

    pub fn dequantize(&self) -> Vec<f32> {
        (0..self.rows * self.cols)
            .map(|index| self.value(index))
            .collect()
    }

    pub fn get(&self, row: usize, column: usize) -> NativeResult<f32> {
        if row >= self.rows || column >= self.cols {
            return Err(NativeError("quantized matrix index out of bounds".into()));
        }
        Ok(self.value(row * self.cols + column))
    }

    pub fn matvec(&self, input: &[f32]) -> NativeResult<Vec<f32>> {
        if input.len() != self.cols {
            return Err(NativeError("quantized matvec dimension mismatch".into()));
        }
        Ok((0..self.rows)
            .map(|row| {
                (0..self.cols)
                    .map(|column| self.value(row * self.cols + column) * input[column])
                    .sum()
            })
            .collect())
    }

    fn value(&self, index: usize) -> f32 {
        let code = match self.config.dtype {
            QuantizationType::Int8 => self.values[index] as i8,
            QuantizationType::Int4 => {
                let packed = self.values[index / 2];
                let nibble = if index % 2 == 0 {
                    packed & 0x0f
                } else {
                    packed >> 4
                };
                if nibble & 0x08 != 0 {
                    (nibble | 0xf0) as i8
                } else {
                    nibble as i8
                }
            }
        };
        code as f32 * self.scales[index / self.config.group_size]
    }
}

#[derive(Debug, Clone)]
struct KvPage {
    keys: Vec<f32>,
    values: Vec<f32>,
    tokens: usize,
}

/// Copy-on-write paged KV cache. Forking a sequence shares completed pages,
/// which makes beam search and prefix reuse cheap, while append mutates only the
/// active tail page.
#[derive(Debug)]
pub struct PagedKvCache {
    page_tokens: usize,
    width: usize,
    sequences: HashMap<String, Vec<Arc<KvPage>>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCacheStats {
    pub sequences: usize,
    pub logical_tokens: usize,
    pub unique_pages: usize,
}

impl PagedKvCache {
    pub fn new(page_tokens: usize, width: usize) -> NativeResult<Self> {
        if page_tokens == 0 || width == 0 {
            return Err(NativeError(
                "KV page size and width must be positive".into(),
            ));
        }
        Ok(Self {
            page_tokens,
            width,
            sequences: HashMap::new(),
        })
    }

    pub fn append(&mut self, sequence: &str, key: &[f32], value: &[f32]) -> NativeResult<()> {
        if key.len() != self.width || value.len() != self.width {
            return Err(NativeError("KV vector width mismatch".into()));
        }
        let pages = self.sequences.entry(sequence.into()).or_default();
        if pages
            .last()
            .is_none_or(|page| page.tokens == self.page_tokens)
        {
            pages.push(Arc::new(KvPage {
                keys: Vec::with_capacity(self.page_tokens * self.width),
                values: Vec::with_capacity(self.page_tokens * self.width),
                tokens: 0,
            }));
        }
        let page = Arc::make_mut(pages.last_mut().unwrap());
        page.keys.extend_from_slice(key);
        page.values.extend_from_slice(value);
        page.tokens += 1;
        Ok(())
    }

    pub fn fork(&mut self, source: &str, destination: &str) -> NativeResult<()> {
        if self.sequences.contains_key(destination) {
            return Err(NativeError("destination sequence already exists".into()));
        }
        let pages = self
            .sequences
            .get(source)
            .ok_or_else(|| NativeError("source sequence does not exist".into()))?
            .clone();
        self.sequences.insert(destination.into(), pages);
        Ok(())
    }

    pub fn remove(&mut self, sequence: &str) -> bool {
        self.sequences.remove(sequence).is_some()
    }

    /// Retains only the newest `keep_tokens`, releasing whole prefix pages and
    /// copy-on-write trimming a partially retained first page.
    pub fn truncate_left(&mut self, sequence: &str, keep_tokens: usize) -> NativeResult<()> {
        let pages = self
            .sequences
            .get_mut(sequence)
            .ok_or_else(|| NativeError("sequence does not exist".into()))?;
        let mut total: usize = pages.iter().map(|page| page.tokens).sum();
        while pages
            .first()
            .is_some_and(|page| total.saturating_sub(page.tokens) >= keep_tokens)
        {
            total -= pages.remove(0).tokens;
        }
        if total > keep_tokens && !pages.is_empty() {
            let remove_tokens = total - keep_tokens;
            let remove_values = remove_tokens * self.width;
            let first = Arc::make_mut(&mut pages[0]);
            first.keys.drain(..remove_values);
            first.values.drain(..remove_values);
            first.tokens -= remove_tokens;
        }
        Ok(())
    }

    pub fn tokens(&self, sequence: &str) -> usize {
        self.sequences
            .get(sequence)
            .map(|pages| pages.iter().map(|page| page.tokens).sum())
            .unwrap_or(0)
    }

    pub fn read(&self, sequence: &str) -> NativeResult<(Vec<f32>, Vec<f32>)> {
        let pages = self
            .sequences
            .get(sequence)
            .ok_or_else(|| NativeError("sequence does not exist".into()))?;
        let mut keys = Vec::with_capacity(self.tokens(sequence) * self.width);
        let mut values = Vec::with_capacity(keys.capacity());
        for page in pages {
            keys.extend_from_slice(&page.keys);
            values.extend_from_slice(&page.values);
        }
        Ok((keys, values))
    }

    pub fn stats(&self) -> KvCacheStats {
        let logical_tokens = self
            .sequences
            .values()
            .flatten()
            .map(|page| page.tokens)
            .sum();
        let mut addresses: Vec<_> = self.sequences.values().flatten().map(Arc::as_ptr).collect();
        addresses.sort_unstable();
        addresses.dedup();
        KvCacheStats {
            sequences: self.sequences.len(),
            logical_tokens,
            unique_pages: addresses.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PpoConfig {
    pub clip_range: f32,
    pub value_clip_range: f32,
    pub entropy_coefficient: f32,
    pub value_coefficient: f32,
}
impl Default for PpoConfig {
    fn default() -> Self {
        Self {
            clip_range: 0.2,
            value_clip_range: 0.2,
            entropy_coefficient: 0.01,
            value_coefficient: 0.5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PpoBatch {
    pub old_log_probs: Vec<f32>,
    pub new_log_probs: Vec<f32>,
    pub advantages: Vec<f32>,
    pub old_values: Vec<f32>,
    pub new_values: Vec<f32>,
    pub returns: Vec<f32>,
    pub entropies: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct PpoOutput {
    pub loss: f32,
    pub policy_loss: f32,
    pub value_loss: f32,
    pub entropy: f32,
    pub policy_gradients: Vec<f32>,
}

pub fn generalized_advantage_estimate(
    rewards: &[f32],
    values: &[f32],
    dones: &[bool],
    gamma: f32,
    lambda: f32,
) -> NativeResult<(Vec<f32>, Vec<f32>)> {
    if rewards.len() != dones.len() || values.len() != rewards.len() + 1 {
        return Err(NativeError("GAE dimensions are inconsistent".into()));
    }
    if !(0.0..=1.0).contains(&gamma) || !(0.0..=1.0).contains(&lambda) {
        return Err(NativeError("gamma and lambda must be in [0, 1]".into()));
    }
    let mut advantages = vec![0.0; rewards.len()];
    let mut next = 0.0;
    for index in (0..rewards.len()).rev() {
        let continuation = if dones[index] { 0.0 } else { 1.0 };
        let delta = rewards[index] + gamma * values[index + 1] * continuation - values[index];
        next = delta + gamma * lambda * continuation * next;
        advantages[index] = next;
    }
    let returns = advantages
        .iter()
        .zip(values)
        .map(|(advantage, value)| advantage + value)
        .collect();
    Ok((advantages, returns))
}

pub fn ppo_objective(batch: &PpoBatch, config: &PpoConfig) -> NativeResult<PpoOutput> {
    if config.clip_range < 0.0
        || config.value_clip_range < 0.0
        || config.entropy_coefficient < 0.0
        || config.value_coefficient < 0.0
    {
        return Err(NativeError("PPO coefficients cannot be negative".into()));
    }
    let n = batch.advantages.len();
    if n == 0
        || [
            batch.old_log_probs.len(),
            batch.new_log_probs.len(),
            batch.old_values.len(),
            batch.new_values.len(),
            batch.returns.len(),
            batch.entropies.len(),
        ]
        .iter()
        .any(|len| *len != n)
    {
        return Err(NativeError("PPO batch dimensions are inconsistent".into()));
    }
    let mut policy_loss = 0.0;
    let mut value_loss = 0.0;
    let mut gradients = Vec::with_capacity(n);
    for i in 0..n {
        let ratio = (batch.new_log_probs[i] - batch.old_log_probs[i]).exp();
        let clipped = ratio.clamp(1.0 - config.clip_range, 1.0 + config.clip_range);
        let raw = ratio * batch.advantages[i];
        let bounded = clipped * batch.advantages[i];
        policy_loss -= raw.min(bounded);
        gradients.push(if raw <= bounded {
            -ratio * batch.advantages[i]
        } else {
            0.0
        });
        let delta = batch.new_values[i] - batch.old_values[i];
        let clipped_value =
            batch.old_values[i] + delta.clamp(-config.value_clip_range, config.value_clip_range);
        value_loss += (batch.new_values[i] - batch.returns[i])
            .powi(2)
            .max((clipped_value - batch.returns[i]).powi(2));
    }
    let entropy = batch.entropies.iter().sum::<f32>() / n as f32;
    policy_loss /= n as f32;
    value_loss /= 2.0 * n as f32;
    Ok(PpoOutput {
        loss: policy_loss + config.value_coefficient * value_loss
            - config.entropy_coefficient * entropy,
        policy_loss,
        value_loss,
        entropy,
        policy_gradients: gradients
            .into_iter()
            .map(|value| value / n as f32)
            .collect(),
    })
}

pub fn dpo_loss(
    chosen_log_ratio: &[f32],
    rejected_log_ratio: &[f32],
    beta: f32,
    label_smoothing: f32,
) -> NativeResult<(f32, Vec<f32>)> {
    if chosen_log_ratio.is_empty() || chosen_log_ratio.len() != rejected_log_ratio.len() {
        return Err(NativeError("DPO batch dimensions are inconsistent".into()));
    }
    if beta <= 0.0 || !(0.0..0.5).contains(&label_smoothing) {
        return Err(NativeError("invalid DPO beta or label smoothing".into()));
    }
    let mut loss = 0.0;
    let mut gradients = Vec::with_capacity(chosen_log_ratio.len());
    for (&chosen, &rejected) in chosen_log_ratio.iter().zip(rejected_log_ratio) {
        let z = beta * (chosen - rejected);
        let positive = softplus(-z);
        let negative = softplus(z);
        loss += (1.0 - label_smoothing) * positive + label_smoothing * negative;
        gradients.push(beta * (sigmoid(z) - (1.0 - label_smoothing)));
    }
    let n = chosen_log_ratio.len() as f32;
    Ok((
        loss / n,
        gradients.into_iter().map(|value| value / n).collect(),
    ))
}

#[derive(Debug, Clone)]
pub struct DistillationConfig {
    pub temperature: f32,
    pub soft_target_weight: f32,
}
impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            temperature: 2.0,
            soft_target_weight: 0.9,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DistillationOutput {
    pub loss: f32,
    pub soft_loss: f32,
    pub hard_loss: f32,
    pub student_gradients: Vec<f32>,
}

pub fn distillation_loss(
    teacher_logits: &[f32],
    student_logits: &[f32],
    target: Option<usize>,
    config: &DistillationConfig,
) -> NativeResult<DistillationOutput> {
    if teacher_logits.is_empty() || teacher_logits.len() != student_logits.len() {
        return Err(NativeError(
            "distillation logits dimensions are inconsistent".into(),
        ));
    }
    if config.temperature <= 0.0 || !(0.0..=1.0).contains(&config.soft_target_weight) {
        return Err(NativeError("invalid distillation configuration".into()));
    }
    if target.is_some_and(|target| target >= student_logits.len()) {
        return Err(NativeError(
            "distillation target is outside vocabulary".into(),
        ));
    }
    let teacher = probabilities(teacher_logits, config.temperature);
    let student = probabilities(student_logits, config.temperature);
    let soft_loss = teacher
        .iter()
        .zip(&student)
        .map(|(t, s)| {
            if *t == 0.0 {
                0.0
            } else {
                t * (t.ln() - s.max(f32::MIN_POSITIVE).ln())
            }
        })
        .sum::<f32>()
        * config.temperature.powi(2);
    let hard_probs = probabilities(student_logits, 1.0);
    let hard_loss = target
        .map(|target| -hard_probs[target].max(f32::MIN_POSITIVE).ln())
        .unwrap_or(0.0);
    let hard_weight = if target.is_some() {
        1.0 - config.soft_target_weight
    } else {
        0.0
    };
    let mut gradients: Vec<f32> = student
        .iter()
        .zip(&teacher)
        .map(|(s, t)| config.soft_target_weight * config.temperature * (s - t))
        .collect();
    if let Some(target) = target {
        for (index, gradient) in gradients.iter_mut().enumerate() {
            *gradient += hard_weight * (hard_probs[index] - usize::from(index == target) as f32);
        }
    }
    Ok(DistillationOutput {
        loss: config.soft_target_weight * soft_loss + hard_weight * hard_loss,
        soft_loss,
        hard_loss,
        student_gradients: gradients,
    })
}

/// Backend hook used by PPO/DPO or distillation loops after Lighter computes
/// objective gradients. Implementations can use autograd, finite differences,
/// adapters, or a remote optimizer.
pub trait OptimizationTarget {
    fn apply_output_gradients(&mut self, gradients: &[f32], learning_rate: f32)
        -> NativeResult<()>;
}

pub fn apply_ppo_update<T: OptimizationTarget>(
    target: &mut T,
    batch: &PpoBatch,
    config: &PpoConfig,
    learning_rate: f32,
) -> NativeResult<PpoOutput> {
    if learning_rate <= 0.0 {
        return Err(NativeError("learning rate must be positive".into()));
    }
    let output = ppo_objective(batch, config)?;
    target.apply_output_gradients(&output.policy_gradients, learning_rate)?;
    Ok(output)
}

pub fn apply_dpo_update<T: OptimizationTarget>(
    target: &mut T,
    chosen_log_ratio: &[f32],
    rejected_log_ratio: &[f32],
    beta: f32,
    label_smoothing: f32,
    learning_rate: f32,
) -> NativeResult<f32> {
    if learning_rate <= 0.0 {
        return Err(NativeError("learning rate must be positive".into()));
    }
    let (loss, gradients) = dpo_loss(chosen_log_ratio, rejected_log_ratio, beta, label_smoothing)?;
    target.apply_output_gradients(&gradients, learning_rate)?;
    Ok(loss)
}

pub fn apply_distillation_update<T: OptimizationTarget>(
    target: &mut T,
    teacher_logits: &[f32],
    student_logits: &[f32],
    label: Option<usize>,
    config: &DistillationConfig,
    learning_rate: f32,
) -> NativeResult<DistillationOutput> {
    if learning_rate <= 0.0 {
        return Err(NativeError("learning rate must be positive".into()));
    }
    let output = distillation_loss(teacher_logits, student_logits, label, config)?;
    target.apply_output_gradients(&output.student_gradients, learning_rate)?;
    Ok(output)
}

fn probabilities(logits: &[f32], temperature: f32) -> Vec<f32> {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut values: Vec<f32> = logits
        .iter()
        .map(|value| ((value - maximum) / temperature).exp())
        .collect();
    let total: f32 = values.iter().sum();
    for value in &mut values {
        *value /= total;
    }
    values
}
fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}
fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else {
        (1.0 + value.exp()).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn int4_quantization_round_trips_and_multiplies() {
        let matrix = QuantizedMatrix::quantize(
            2,
            2,
            &[1.0, -1.0, 0.5, 0.25],
            QuantizationConfig {
                dtype: QuantizationType::Int4,
                group_size: 2,
            },
        )
        .unwrap();
        let output = matrix.matvec(&[2.0, 1.0]).unwrap();
        assert!((output[0] - 1.0).abs() < 0.2);
        assert!((output[1] - 1.25).abs() < 0.2);
        assert!(matrix.storage_bytes() < 4 * 4);
    }
    #[test]
    fn paged_cache_shares_prefix_then_copies_tail() {
        let mut cache = PagedKvCache::new(2, 1).unwrap();
        cache.append("a", &[1.0], &[2.0]).unwrap();
        cache.append("a", &[3.0], &[4.0]).unwrap();
        cache.fork("a", "b").unwrap();
        assert_eq!(cache.stats().unique_pages, 1);
        cache.append("b", &[5.0], &[6.0]).unwrap();
        assert_eq!(cache.tokens("a"), 2);
        assert_eq!(cache.tokens("b"), 3);
        assert_eq!(cache.stats().unique_pages, 2);
        cache.truncate_left("b", 1).unwrap();
        assert_eq!(cache.read("b").unwrap(), (vec![5.0], vec![6.0]));
    }
    #[test]
    fn gae_respects_terminal_boundaries() {
        let (advantages, returns) = generalized_advantage_estimate(
            &[1.0, 2.0],
            &[0.5, 0.25, 10.0],
            &[false, true],
            1.0,
            1.0,
        )
        .unwrap();
        assert_eq!(advantages, vec![2.5, 1.75]);
        assert_eq!(returns, vec![3.0, 2.0]);
    }
    #[test]
    fn distillation_gradient_moves_student_toward_teacher() {
        let output = distillation_loss(
            &[4.0, 0.0],
            &[0.0, 4.0],
            Some(0),
            &DistillationConfig::default(),
        )
        .unwrap();
        assert!(output.loss > 0.0);
        assert!(output.student_gradients[0] < 0.0);
        assert!(output.student_gradients[1] > 0.0);
    }
}
