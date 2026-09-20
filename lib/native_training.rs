//! Transformer fine-tuning and reinforcement-model building blocks.

use crate::native::{NativeError, NativeResult};

#[derive(Debug, Clone)]
pub struct LoraConfig {
    pub rank: usize,
    pub alpha: f32,
    pub dropout: f32,
}

impl LoraConfig {
    pub fn validate(&self) -> NativeResult<()> {
        if self.rank == 0 || self.alpha <= 0.0 || !(0.0..1.0).contains(&self.dropout) {
            return Err(NativeError("invalid LoRA rank, alpha, or dropout".into()));
        }
        Ok(())
    }
}

/// Trainable low-rank adapter for a frozen `[output, input]` projection.
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    input: usize,
    output: usize,
    config: LoraConfig,
    a: Vec<f32>,
    b: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct LoraGradients {
    pub a: Vec<f32>,
    pub b: Vec<f32>,
}

impl LoraAdapter {
    pub fn new(input: usize, output: usize, config: LoraConfig, seed: u64) -> NativeResult<Self> {
        config.validate()?;
        if input == 0 || output == 0 {
            return Err(NativeError("LoRA dimensions must be positive".into()));
        }
        let mut state = seed.max(1);
        let scale = (1.0 / input as f32).sqrt();
        let a = (0..config.rank * input)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state as u32) as f32 / u32::MAX as f32 * 2.0 - 1.0) * scale
            })
            .collect();
        Ok(Self {
            input,
            output,
            b: vec![0.0; output * config.rank],
            a,
            config,
        })
    }

    pub fn from_weights(
        input: usize,
        output: usize,
        config: LoraConfig,
        a: Vec<f32>,
        b: Vec<f32>,
    ) -> NativeResult<Self> {
        config.validate()?;
        if a.len() != config.rank * input || b.len() != output * config.rank {
            return Err(NativeError(
                "LoRA weight dimensions are inconsistent".into(),
            ));
        }
        Ok(Self {
            input,
            output,
            config,
            a,
            b,
        })
    }

    pub fn forward(&self, input: &[f32], training: bool, seed: u64) -> NativeResult<Vec<f32>> {
        if input.len() != self.input {
            return Err(NativeError("LoRA input dimension mismatch".into()));
        }
        let masked = dropout(input, self.config.dropout, training, seed);
        let hidden = matvec(&self.a, self.config.rank, self.input, &masked);
        let mut output = matvec(&self.b, self.output, self.config.rank, &hidden);
        let scale = self.config.alpha / self.config.rank as f32;
        for value in &mut output {
            *value *= scale;
        }
        Ok(output)
    }

    /// Backpropagates through the adapter and returns `(parameter gradients,
    /// input gradient)`. The frozen base projection is intentionally excluded.
    pub fn backward(
        &self,
        input: &[f32],
        output_gradient: &[f32],
    ) -> NativeResult<(LoraGradients, Vec<f32>)> {
        if input.len() != self.input || output_gradient.len() != self.output {
            return Err(NativeError("LoRA backward dimension mismatch".into()));
        }
        let hidden = matvec(&self.a, self.config.rank, self.input, input);
        let scale = self.config.alpha / self.config.rank as f32;
        let mut grad_b = vec![0.0; self.b.len()];
        for row in 0..self.output {
            for column in 0..self.config.rank {
                grad_b[row * self.config.rank + column] =
                    output_gradient[row] * hidden[column] * scale;
            }
        }
        let mut grad_hidden = vec![0.0; self.config.rank];
        for row in 0..self.output {
            for column in 0..self.config.rank {
                grad_hidden[column] +=
                    self.b[row * self.config.rank + column] * output_gradient[row] * scale;
            }
        }
        let mut grad_a = vec![0.0; self.a.len()];
        for row in 0..self.config.rank {
            for column in 0..self.input {
                grad_a[row * self.input + column] = grad_hidden[row] * input[column];
            }
        }
        let mut grad_input = vec![0.0; self.input];
        for row in 0..self.config.rank {
            for column in 0..self.input {
                grad_input[column] += self.a[row * self.input + column] * grad_hidden[row];
            }
        }
        Ok((
            LoraGradients {
                a: grad_a,
                b: grad_b,
            },
            grad_input,
        ))
    }

    pub fn apply_gradients(
        &mut self,
        gradients: &LoraGradients,
        optimizer: &mut AdamW,
    ) -> NativeResult<()> {
        if gradients.a.len() != self.a.len() || gradients.b.len() != self.b.len() {
            return Err(NativeError(
                "LoRA gradient dimensions are inconsistent".into(),
            ));
        }
        optimizer.update("lora_a", &mut self.a, &gradients.a)?;
        optimizer.update("lora_b", &mut self.b, &gradients.b)
    }

    pub fn weights(&self) -> (&[f32], &[f32]) {
        (&self.a, &self.b)
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.input, self.output)
    }

    pub fn zero_gradients(&self) -> LoraGradients {
        LoraGradients {
            a: vec![0.0; self.a.len()],
            b: vec![0.0; self.b.len()],
        }
    }

    pub fn accumulate(target: &mut LoraGradients, source: &LoraGradients) -> NativeResult<()> {
        if target.a.len() != source.a.len() || target.b.len() != source.b.len() {
            return Err(NativeError("LoRA gradient accumulation mismatch".into()));
        }
        for (target, source) in target.a.iter_mut().zip(&source.a) {
            *target += source;
        }
        for (target, source) in target.b.iter_mut().zip(&source.b) {
            *target += source;
        }
        Ok(())
    }

    pub fn apply_sgd(&mut self, gradients: &LoraGradients, learning_rate: f32) -> NativeResult<()> {
        if learning_rate <= 0.0
            || gradients.a.len() != self.a.len()
            || gradients.b.len() != self.b.len()
        {
            return Err(NativeError("invalid LoRA SGD update".into()));
        }
        for (parameter, gradient) in self.a.iter_mut().zip(&gradients.a) {
            *parameter -= learning_rate * gradient;
        }
        for (parameter, gradient) in self.b.iter_mut().zip(&gradients.b) {
            *parameter -= learning_rate * gradient;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct AdamWConfig {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
    pub max_gradient_norm: Option<f32>,
}
impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
            max_gradient_norm: Some(1.0),
        }
    }
}

#[derive(Default)]
struct AdamState {
    first: Vec<f32>,
    second: Vec<f32>,
    step: u64,
}
pub struct AdamW {
    config: AdamWConfig,
    states: std::collections::HashMap<String, AdamState>,
}

impl AdamW {
    pub fn new(config: AdamWConfig) -> NativeResult<Self> {
        if config.learning_rate <= 0.0
            || !(0.0..1.0).contains(&config.beta1)
            || !(0.0..1.0).contains(&config.beta2)
            || config.epsilon <= 0.0
            || config.weight_decay < 0.0
        {
            return Err(NativeError("invalid AdamW configuration".into()));
        }
        Ok(Self {
            config,
            states: Default::default(),
        })
    }
    pub fn update(
        &mut self,
        name: &str,
        parameters: &mut [f32],
        gradients: &[f32],
    ) -> NativeResult<()> {
        if parameters.len() != gradients.len() {
            return Err(NativeError("AdamW gradient dimension mismatch".into()));
        }
        let norm = gradients.iter().map(|g| g * g).sum::<f32>().sqrt();
        let clip = self
            .config
            .max_gradient_norm
            .filter(|max| norm > *max)
            .map(|max| max / norm)
            .unwrap_or(1.0);
        let state = self.states.entry(name.into()).or_insert_with(|| AdamState {
            first: vec![0.0; parameters.len()],
            second: vec![0.0; parameters.len()],
            step: 0,
        });
        if state.first.len() != parameters.len() {
            return Err(NativeError(
                "AdamW parameter name reused with another shape".into(),
            ));
        }
        state.step += 1;
        let correction1 = 1.0 - self.config.beta1.powi(state.step as i32);
        let correction2 = 1.0 - self.config.beta2.powi(state.step as i32);
        for i in 0..parameters.len() {
            let gradient = gradients[i] * clip;
            state.first[i] =
                self.config.beta1 * state.first[i] + (1.0 - self.config.beta1) * gradient;
            state.second[i] = self.config.beta2 * state.second[i]
                + (1.0 - self.config.beta2) * gradient * gradient;
            let update = state.first[i]
                / correction1
                / ((state.second[i] / correction2).sqrt() + self.config.epsilon);
            parameters[i] -=
                self.config.learning_rate * (update + self.config.weight_decay * parameters[i]);
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct TokenLoss {
    pub loss: f32,
    pub gradients: Vec<f32>,
}

/// Causal-language-model cross entropy. `ignore_index` supports padded labels.
pub fn causal_lm_loss(
    logits: &[Vec<f32>],
    labels: &[i64],
    ignore_index: i64,
    label_smoothing: f32,
) -> NativeResult<TokenLoss> {
    if logits.len() != labels.len() || logits.is_empty() || !(0.0..1.0).contains(&label_smoothing) {
        return Err(NativeError("invalid causal LM loss inputs".into()));
    }
    let vocab = logits[0].len();
    if vocab == 0 || logits.iter().any(|row| row.len() != vocab) {
        return Err(NativeError("inconsistent vocabulary dimensions".into()));
    }
    let active = labels
        .iter()
        .filter(|label| **label != ignore_index)
        .count();
    if active == 0 {
        return Err(NativeError(
            "causal LM batch contains no active labels".into(),
        ));
    }
    let mut loss = 0.0;
    let mut gradients = Vec::with_capacity(logits.len() * vocab);
    for (row, &label) in logits.iter().zip(labels) {
        if label == ignore_index {
            gradients.extend(std::iter::repeat(0.0).take(vocab));
            continue;
        }
        let target = usize::try_from(label)
            .ok()
            .filter(|target| *target < vocab)
            .ok_or_else(|| NativeError("causal LM label outside vocabulary".into()))?;
        let probabilities = softmax(row);
        let uniform = label_smoothing / vocab as f32;
        for (index, probability) in probabilities.iter().enumerate() {
            let expected = uniform
                + if index == target {
                    1.0 - label_smoothing
                } else {
                    0.0
                };
            loss -= expected * probability.max(f32::MIN_POSITIVE).ln();
            gradients.push((probability - expected) / active as f32);
        }
    }
    Ok(TokenLoss {
        loss: loss / active as f32,
        gradients,
    })
}

/// Linear scalar value/reward head suitable for PPO critics and pairwise reward models.
#[derive(Debug, Clone)]
pub struct RewardHead {
    weights: Vec<f32>,
    bias: f32,
}
impl RewardHead {
    pub fn new(hidden_size: usize) -> NativeResult<Self> {
        if hidden_size == 0 {
            return Err(NativeError(
                "reward head hidden size must be positive".into(),
            ));
        }
        Ok(Self {
            weights: vec![0.0; hidden_size],
            bias: 0.0,
        })
    }
    pub fn from_weights(weights: Vec<f32>, bias: f32) -> NativeResult<Self> {
        if weights.is_empty() || !bias.is_finite() || weights.iter().any(|value| !value.is_finite())
        {
            return Err(NativeError("invalid reward head weights".into()));
        }
        Ok(Self { weights, bias })
    }
    pub fn weights(&self) -> (&[f32], f32) {
        (&self.weights, self.bias)
    }
    pub fn score(&self, hidden: &[f32]) -> NativeResult<f32> {
        if hidden.len() != self.weights.len() {
            return Err(NativeError("reward head dimension mismatch".into()));
        }
        Ok(self
            .weights
            .iter()
            .zip(hidden)
            .map(|(w, x)| w * x)
            .sum::<f32>()
            + self.bias)
    }
    pub fn pairwise_loss(
        &self,
        chosen: &[f32],
        rejected: &[f32],
    ) -> NativeResult<(f32, Vec<f32>, Vec<f32>)> {
        let difference = self.score(chosen)? - self.score(rejected)?;
        let loss = softplus(-difference);
        let factor = sigmoid(difference) - 1.0;
        let chosen_gradient: Vec<f32> = self.weights.iter().map(|weight| factor * weight).collect();
        let rejected_gradient: Vec<f32> =
            chosen_gradient.iter().map(|gradient| -gradient).collect();
        Ok((loss, chosen_gradient, rejected_gradient))
    }
    pub fn apply_pairwise_update(
        &mut self,
        chosen: &[f32],
        rejected: &[f32],
        optimizer: &mut AdamW,
    ) -> NativeResult<f32> {
        let difference = self.score(chosen)? - self.score(rejected)?;
        let loss = softplus(-difference);
        let factor = sigmoid(difference) - 1.0;
        let gradient: Vec<f32> = chosen
            .iter()
            .zip(rejected)
            .map(|(c, r)| factor * (c - r))
            .collect();
        optimizer.update("reward_head", &mut self.weights, &gradient)?;
        Ok(loss)
    }
}

/// Runtime-neutral transformer fine-tuning contract. Implementations return
/// token logits and consume their gradients; this supports full fine-tuning,
/// LoRA/QLoRA, remote trainers, and architecture-specific autograd engines.
pub trait FineTunableTransformer {
    fn token_logits(&mut self, input_ids: &[u32]) -> NativeResult<Vec<Vec<f32>>>;
    fn apply_token_gradients(&mut self, gradients: &[f32], learning_rate: f32) -> NativeResult<()>;
}

pub fn supervised_fine_tune_step<T: FineTunableTransformer>(
    model: &mut T,
    input_ids: &[u32],
    labels: &[i64],
    ignore_index: i64,
    label_smoothing: f32,
    learning_rate: f32,
) -> NativeResult<f32> {
    if learning_rate <= 0.0 {
        return Err(NativeError("learning rate must be positive".into()));
    }
    let logits = model.token_logits(input_ids)?;
    let output = causal_lm_loss(&logits, labels, ignore_index, label_smoothing)?;
    model.apply_token_gradients(&output.gradients, learning_rate)?;
    Ok(output.loss)
}

fn matvec(matrix: &[f32], rows: usize, cols: usize, input: &[f32]) -> Vec<f32> {
    (0..rows)
        .map(|row| {
            matrix[row * cols..(row + 1) * cols]
                .iter()
                .zip(input)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect()
}
fn softmax(logits: &[f32]) -> Vec<f32> {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut values: Vec<f32> = logits.iter().map(|value| (value - maximum).exp()).collect();
    let total: f32 = values.iter().sum();
    for value in &mut values {
        *value /= total;
    }
    values
}
fn dropout(input: &[f32], probability: f32, training: bool, mut state: u64) -> Vec<f32> {
    if !training || probability == 0.0 {
        return input.to_vec();
    }
    let scale = 1.0 / (1.0 - probability);
    input
        .iter()
        .map(|value| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            if (state as u32) as f32 / (u32::MAX as f32) < probability {
                0.0
            } else {
                value * scale
            }
        })
        .collect()
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
    fn lora_starts_as_identity_delta_and_updates() {
        let mut adapter = LoraAdapter::new(
            2,
            2,
            LoraConfig {
                rank: 1,
                alpha: 1.0,
                dropout: 0.0,
            },
            7,
        )
        .unwrap();
        assert_eq!(
            adapter.forward(&[1.0, 2.0], false, 0).unwrap(),
            vec![0.0, 0.0]
        );
        let (gradients, _) = adapter.backward(&[1.0, 2.0], &[1.0, -1.0]).unwrap();
        let mut optimizer = AdamW::new(AdamWConfig::default()).unwrap();
        adapter.apply_gradients(&gradients, &mut optimizer).unwrap();
        assert_ne!(adapter.weights().1, &[0.0, 0.0]);
    }
    #[test]
    fn smoothed_causal_loss_has_zero_sum_gradient() {
        let result = causal_lm_loss(&[vec![2.0, 0.0]], &[0], -100, 0.1).unwrap();
        assert!(result.loss > 0.0);
        assert!(result.gradients.iter().sum::<f32>().abs() < 1e-6);
    }
    #[test]
    fn reward_pairwise_update_reduces_loss() {
        let mut head = RewardHead {
            weights: vec![0.1, -0.1],
            bias: 0.0,
        };
        let chosen = [1.0, 0.0];
        let rejected = [0.0, 1.0];
        let before = head.pairwise_loss(&chosen, &rejected).unwrap().0;
        let mut optimizer = AdamW::new(AdamWConfig {
            learning_rate: 0.1,
            weight_decay: 0.0,
            ..Default::default()
        })
        .unwrap();
        head.apply_pairwise_update(&chosen, &rejected, &mut optimizer)
            .unwrap();
        assert!(head.pairwise_loss(&chosen, &rejected).unwrap().0 < before);
    }
}
