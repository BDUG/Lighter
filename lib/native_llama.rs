//! A dependency-light CPU reference implementation of the Llama decoder family.
//!
//! It intentionally favors portability and auditability over SIMD throughput.
//! Production GPU runtimes can implement `NativeModel` directly while sharing
//! the scheduler, tokenizer, artifact loader, and sampling stack.

use crate::native::{BackendInput, HuggingFaceArtifacts, NativeError, NativeModel, NativeResult};
use crate::native_advanced::{PagedKvCache, QuantizationConfig, QuantizedMatrix};
use crate::native_training::{FineTunableTransformer, LoraAdapter, LoraConfig};
use half::{bf16, f16};
use rayon::prelude::*;
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::fs;
use std::sync::Arc;

#[derive(Clone)]
enum MatrixData {
    F32(Arc<Vec<f32>>),
    F16(Arc<Vec<u16>>),
    BF16(Arc<Vec<u16>>),
    Quantized(QuantizedMatrix),
}

enum TensorData {
    F32(Vec<f32>),
    F16(Vec<u16>),
    BF16(Vec<u16>),
    Quantized(QuantizedMatrix),
}

struct StoredTensor {
    shape: Vec<usize>,
    data: TensorData,
}

impl TensorData {
    fn len(&self) -> usize {
        match self {
            Self::F32(data) => data.len(),
            Self::F16(data) | Self::BF16(data) => data.len(),
            Self::Quantized(data) => data.rows() * data.cols(),
        }
    }

    fn into_f32(self) -> Vec<f32> {
        match self {
            Self::F32(data) => data,
            Self::F16(data) => data
                .into_iter()
                .map(|v| f16::from_bits(v).to_f32())
                .collect(),
            Self::BF16(data) => data
                .into_iter()
                .map(|v| bf16::from_bits(v).to_f32())
                .collect(),
            Self::Quantized(data) => data.dequantize(),
        }
    }
}

#[derive(Clone)]
struct Matrix {
    rows: usize,
    cols: usize,
    data: MatrixData,
    lora: Option<LoraAdapter>,
}

impl Matrix {
    fn row(&self, row: usize) -> NativeResult<Vec<f32>> {
        if row >= self.rows {
            return Err(NativeError(format!("token {row} exceeds vocabulary")));
        }
        match &self.data {
            MatrixData::F32(data) => Ok(data[row * self.cols..(row + 1) * self.cols].to_vec()),
            MatrixData::F16(data) => Ok(data[row * self.cols..(row + 1) * self.cols]
                .iter()
                .map(|&v| f16::from_bits(v).to_f32())
                .collect()),
            MatrixData::BF16(data) => Ok(data[row * self.cols..(row + 1) * self.cols]
                .iter()
                .map(|&v| bf16::from_bits(v).to_f32())
                .collect()),
            MatrixData::Quantized(data) => {
                (0..self.cols).map(|column| data.get(row, column)).collect()
            }
        }
    }
    fn mv(&self, x: &[f32]) -> NativeResult<Vec<f32>> {
        if x.len() != self.cols {
            return Err(NativeError(format!(
                "matrix input mismatch: {} != {}",
                x.len(),
                self.cols
            )));
        }
        let mut output = match &self.data {
            MatrixData::F32(data) => Ok(data
                .par_chunks_exact(self.cols)
                .map(|row| crate::arm::dot_auto(row, x))
                .collect()),
            MatrixData::F16(data) => Ok(data
                .par_chunks_exact(self.cols)
                .map(|row| {
                    row.iter()
                        .zip(x)
                        .map(|(&a, b)| f16::from_bits(a).to_f32() * b)
                        .sum()
                })
                .collect()),
            MatrixData::BF16(data) => Ok(data
                .par_chunks_exact(self.cols)
                .map(|row| {
                    row.iter()
                        .zip(x)
                        .map(|(&a, b)| bf16::from_bits(a).to_f32() * b)
                        .sum()
                })
                .collect()),
            MatrixData::Quantized(data) => data.matvec(x),
        }?;
        if let Some(adapter) = &self.lora {
            let delta = adapter.forward(x, false, 0)?;
            for (value, delta) in output.iter_mut().zip(delta) {
                *value += delta;
            }
        }
        Ok(output)
    }
    fn quantize(&mut self, config: QuantizationConfig) -> NativeResult<()> {
        if !matches!(self.data, MatrixData::Quantized(_)) {
            let data: Vec<f32> = match &self.data {
                MatrixData::F32(data) => data.as_ref().clone(),
                MatrixData::F16(data) => data.iter().map(|&v| f16::from_bits(v).to_f32()).collect(),
                MatrixData::BF16(data) => {
                    data.iter().map(|&v| bf16::from_bits(v).to_f32()).collect()
                }
                MatrixData::Quantized(_) => unreachable!(),
            };
            self.data = MatrixData::Quantized(QuantizedMatrix::quantize(
                self.rows, self.cols, &data, config,
            )?);
        }
        Ok(())
    }
}

struct Layer {
    input_norm: Vec<f32>,
    post_norm: Vec<f32>,
    q: Matrix,
    k: Matrix,
    v: Matrix,
    o: Matrix,
    q_bias: Option<Vec<f32>>,
    k_bias: Option<Vec<f32>>,
    v_bias: Option<Vec<f32>>,
    q_norm: Option<Vec<f32>>,
    k_norm: Option<Vec<f32>>,
    gate: Matrix,
    up: Matrix,
    down: Matrix,
}

#[derive(Default)]
struct LayerCache {
    keys: Vec<f32>,
    values: Vec<f32>,
}
#[derive(Clone, Copy)]
enum RopeScaling {
    None,
    Linear {
        factor: f32,
    },
    Llama3 {
        factor: f32,
        low_frequency_factor: f32,
        high_frequency_factor: f32,
        original_context: f32,
    },
}

/// Native CPU implementation for Llama, Mistral, and Qwen2-style dense
/// decoder-only safetensors checkpoints. It supports GQA/MQA, RoPE, optional
/// QKV biases, tied embeddings, and a per-request KV cache.
pub struct NativeLlama {
    embedding: Matrix,
    layers: Vec<Layer>,
    norm: Vec<f32>,
    output: Matrix,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    eps: f32,
    rope_theta: f32,
    rope_scaling: RopeScaling,
    sliding_window: Option<usize>,
    max_positions: Option<usize>,
    processed: HashMap<String, usize>,
    kv_cache: PagedKvCache,
    training_lm_inputs: Vec<Vec<f32>>,
    capture_training: bool,
}

impl NativeLlama {
    pub fn load(artifacts: &HuggingFaceArtifacts) -> NativeResult<Self> {
        Self::load_with_quantization(artifacts, None)
    }

    /// Loads and quantizes each matrix as its checkpoint shard is decoded.
    /// This avoids first retaining the complete half-precision 8B checkpoint
    /// in memory, which is essential on hosts sized for the quantized model.
    pub fn load_quantized(
        artifacts: &HuggingFaceArtifacts,
        config: QuantizationConfig,
    ) -> NativeResult<Self> {
        Self::load_with_quantization(artifacts, Some(config))
    }

    fn load_with_quantization(
        artifacts: &HuggingFaceArtifacts,
        quantization: Option<QuantizationConfig>,
    ) -> NativeResult<Self> {
        let c = &artifacts.config;
        if !is_supported_decoder_config(c) {
            return Err(NativeError(format!(
                "native CPU decoder does not support model_type {:?} with architectures {:?}",
                c.model_type, c.architectures
            )));
        }
        if c.hidden_size == 0
            || c.intermediate_size == 0
            || c.num_hidden_layers == 0
            || c.num_attention_heads == 0
        {
            return Err(NativeError(
                "config is missing transformer dimensions".into(),
            ));
        }
        let mut tensors = load_tensors(&artifacts.weights, quantization)?;
        let embedding = matrix(&mut tensors, "model.embed_tokens.weight")?;
        let heads = c.num_attention_heads;
        let kv_heads = c.num_key_value_heads.unwrap_or(heads);
        if heads % kv_heads != 0 {
            return Err(NativeError(
                "attention heads must be divisible by KV heads".into(),
            ));
        }
        let head_dim = c.head_dim.unwrap_or(c.hidden_size / heads);
        if head_dim * heads != c.hidden_size || head_dim % 2 != 0 {
            return Err(NativeError(
                "hidden_size must equal num_attention_heads * an even head_dim".into(),
            ));
        }
        let rope_scaling = parse_rope_scaling(c.extensions.get("rope_scaling"))?;
        let sliding_window = c
            .extensions
            .get("sliding_window")
            .and_then(|value| value.as_u64())
            .map(|value| value as usize);
        let mut layers = Vec::with_capacity(c.num_hidden_layers);
        for i in 0..c.num_hidden_layers {
            let p = format!("model.layers.{i}");
            layers.push(Layer {
                input_norm: vector(&mut tensors, &format!("{p}.input_layernorm.weight"))?,
                post_norm: vector(
                    &mut tensors,
                    &format!("{p}.post_attention_layernorm.weight"),
                )?,
                q: matrix(&mut tensors, &format!("{p}.self_attn.q_proj.weight"))?,
                k: matrix(&mut tensors, &format!("{p}.self_attn.k_proj.weight"))?,
                v: matrix(&mut tensors, &format!("{p}.self_attn.v_proj.weight"))?,
                o: matrix(&mut tensors, &format!("{p}.self_attn.o_proj.weight"))?,
                q_bias: optional_vector(&mut tensors, &format!("{p}.self_attn.q_proj.bias")),
                k_bias: optional_vector(&mut tensors, &format!("{p}.self_attn.k_proj.bias")),
                v_bias: optional_vector(&mut tensors, &format!("{p}.self_attn.v_proj.bias")),
                q_norm: optional_vector(&mut tensors, &format!("{p}.self_attn.q_norm.weight")),
                k_norm: optional_vector(&mut tensors, &format!("{p}.self_attn.k_norm.weight")),
                gate: matrix(&mut tensors, &format!("{p}.mlp.gate_proj.weight"))?,
                up: matrix(&mut tensors, &format!("{p}.mlp.up_proj.weight"))?,
                down: matrix(&mut tensors, &format!("{p}.mlp.down_proj.weight"))?,
            });
        }
        let norm = vector(&mut tensors, "model.norm.weight")?;
        let output = match matrix(&mut tensors, "lm_head.weight") {
            Ok(value) => value,
            Err(_) if c.tie_word_embeddings => embedding.clone(),
            Err(error) => return Err(error),
        };
        Ok(Self {
            embedding,
            layers,
            norm,
            output,
            heads,
            kv_heads,
            head_dim,
            eps: c.rms_norm_eps as f32,
            rope_theta: c.rope_theta.unwrap_or(10_000.0) as f32,
            rope_scaling,
            sliding_window,
            max_positions: c.max_position_embeddings,
            processed: HashMap::new(),
            kv_cache: PagedKvCache::new(16, kv_heads * head_dim)?,
            training_lm_inputs: Vec::new(),
            capture_training: false,
        })
    }

    /// Quantizes all embedding, attention, MLP, and output projections in
    /// place. RMSNorm vectors intentionally remain f32.
    pub fn quantize(&mut self, config: QuantizationConfig) -> NativeResult<()> {
        self.embedding.quantize(config)?;
        self.output.quantize(config)?;
        for layer in &mut self.layers {
            layer.q.quantize(config)?;
            layer.k.quantize(config)?;
            layer.v.quantize(config)?;
            layer.o.quantize(config)?;
            layer.gate.quantize(config)?;
            layer.up.quantize(config)?;
            layer.down.quantize(config)?;
        }
        Ok(())
    }

    /// Enables parameter-efficient supervised fine-tuning on the language-model
    /// head. The base (including quantized) projection remains frozen.
    pub fn enable_lm_head_lora(&mut self, config: LoraConfig, seed: u64) -> NativeResult<()> {
        self.output.lora = Some(LoraAdapter::new(
            self.output.cols,
            self.output.rows,
            config,
            seed,
        )?);
        Ok(())
    }

    pub fn lm_head_lora(&self) -> Option<&LoraAdapter> {
        self.output.lora.as_ref()
    }

    /// Installs a trained output-head adapter. Reuse its original base model and tokenizer.
    pub fn set_lm_head_lora(&mut self, adapter: LoraAdapter) -> NativeResult<()> {
        if adapter.dimensions() != (self.output.cols, self.output.rows) {
            return Err(NativeError("adapter does not match language-model head dimensions".into()));
        }
        let (a, b) = adapter.weights();
        if a.iter().chain(b).any(|value| !value.is_finite()) {
            return Err(NativeError("adapter weights must be finite".into()));
        }
        self.output.lora = Some(adapter);
        Ok(())
    }

    /// Forks an existing sequence's paged KV state for beam search or
    /// speculative decoding without copying shared prefix pages.
    pub fn fork_sequence(&mut self, source: &str, destination: &str) -> NativeResult<()> {
        if self.processed.contains_key(destination) {
            return Err(NativeError("destination sequence already exists".into()));
        }
        let processed = *self
            .processed
            .get(source)
            .ok_or_else(|| NativeError("source sequence does not exist".into()))?;
        let mut forked: Vec<String> = Vec::new();
        for layer in 0..self.layers.len() {
            let source_key = layer_cache_id(source, layer);
            let destination_key = layer_cache_id(destination, layer);
            if let Err(error) = self.kv_cache.fork(&source_key, &destination_key) {
                for key in forked {
                    self.kv_cache.remove(&key);
                }
                return Err(error);
            }
            forked.push(destination_key);
        }
        self.processed.insert(destination.into(), processed);
        Ok(())
    }

    pub fn kv_cache_stats(&self) -> crate::native_advanced::KvCacheStats {
        self.kv_cache.stats()
    }

    fn clear_sequence(&mut self, id: &str) {
        self.processed.remove(id);
        for layer in 0..self.layers.len() {
            self.kv_cache.remove(&layer_cache_id(id, layer));
        }
    }

    fn run(&mut self, id: &str, tokens: &[u32]) -> NativeResult<Vec<f32>> {
        if tokens.is_empty() {
            return Err(NativeError("a decoder prompt cannot be empty".into()));
        }
        let mut start = self.processed.get(id).copied().unwrap_or(0);
        if start > tokens.len() {
            self.clear_sequence(id);
            start = 0;
        }
        let mut last = None;
        for (position, &token) in tokens.iter().enumerate().skip(start) {
            if self.max_positions.is_some_and(|maximum| {
                position >= maximum && matches!(self.rope_scaling, RopeScaling::None)
            }) {
                return Err(NativeError(format!(
                    "position {position} exceeds max_position_embeddings without RoPE scaling"
                )));
            }
            let mut x = self.embedding.row(token as usize)?;
            for (layer_index, layer) in self.layers.iter().enumerate() {
                let residual = x.clone();
                let n = rms_norm(&x, &layer.input_norm, self.eps)?;
                let mut q = add_bias(layer.q.mv(&n)?, layer.q_bias.as_deref())?;
                let mut k = add_bias(layer.k.mv(&n)?, layer.k_bias.as_deref())?;
                let v = add_bias(layer.v.mv(&n)?, layer.v_bias.as_deref())?;
                if let Some(weight) = layer.q_norm.as_deref() {
                    q = rms_norm_heads(&q, weight, self.heads, self.head_dim, self.eps)?;
                }
                if let Some(weight) = layer.k_norm.as_deref() {
                    k = rms_norm_heads(&k, weight, self.kv_heads, self.head_dim, self.eps)?;
                }
                apply_rope(
                    &mut q,
                    self.heads,
                    self.head_dim,
                    position,
                    self.rope_theta,
                    self.rope_scaling,
                )?;
                apply_rope(
                    &mut k,
                    self.kv_heads,
                    self.head_dim,
                    position,
                    self.rope_theta,
                    self.rope_scaling,
                )?;
                let cache_id = layer_cache_id(id, layer_index);
                self.kv_cache.append(&cache_id, &k, &v)?;
                if let Some(window) = self.sliding_window {
                    self.kv_cache.truncate_left(&cache_id, window)?;
                }
                let (keys, values) = self.kv_cache.read(&cache_id)?;
                let cache = LayerCache { keys, values };
                let attention = attend(&q, &cache, self.heads, self.kv_heads, self.head_dim)?;
                x = add(residual, layer.o.mv(&attention)?)?;
                let residual = x.clone();
                let n = rms_norm(&x, &layer.post_norm, self.eps)?;
                let gate = layer.gate.mv(&n)?;
                let up = layer.up.mv(&n)?;
                let activated: Vec<f32> =
                    gate.into_iter().zip(up).map(|(g, u)| silu(g) * u).collect();
                x = add(residual, layer.down.mv(&activated)?)?;
            }
            last = Some(x);
            self.processed.insert(id.to_owned(), position + 1);
        }
        // A repeated call without a new token is unusual but remains defined.
        let hidden = match last {
            Some(x) => x,
            None => {
                self.clear_sequence(id);
                return self.run(id, tokens);
            }
        };
        let normalized = rms_norm(&hidden, &self.norm, self.eps)?;
        if self.capture_training {
            self.training_lm_inputs.push(normalized.clone());
        }
        self.output.mv(&normalized)
    }
}

/// Some fine-tuned repositories register a checkpoint-specific `model_type`
/// even though their tensors retain the standard Llama layout. Prefer the
/// architecture declaration in that case so those checkpoints do not require
/// rewriting `config.json` before loading.
pub(crate) fn is_supported_decoder_config(config: &crate::native::HuggingFaceConfig) -> bool {
    matches!(
        config.model_type.as_str(),
        "llama" | "mistral" | "qwen2" | "qwen3" | "clm" | "contrastive_lm"
    ) || config.architectures.iter().any(|architecture| {
        matches!(
            architecture.as_str(),
            "LlamaForCausalLM"
                | "MistralForCausalLM"
                | "Qwen2ForCausalLM"
                | "Qwen3ForCausalLM"
                | "CLMForCausalLM"
        )
    })
}

impl NativeModel for NativeLlama {
    fn forward(&mut self, id: &str, tokens: &[u32], _: usize) -> NativeResult<Vec<f32>> {
        self.run(id, tokens)
    }
    fn forward_batch(&mut self, batch: &[BackendInput<'_>]) -> NativeResult<Vec<Vec<f32>>> {
        batch
            .iter()
            .map(|x| self.run(x.sequence_id, x.tokens))
            .collect()
    }
    fn remove_sequence(&mut self, id: &str) {
        self.clear_sequence(id);
    }
}

impl FineTunableTransformer for NativeLlama {
    fn token_logits(&mut self, input_ids: &[u32]) -> NativeResult<Vec<Vec<f32>>> {
        if self.output.lora.is_none() {
            return Err(NativeError("enable LM-head LoRA before fine-tuning".into()));
        }
        if input_ids.is_empty() {
            return Err(NativeError("fine-tuning input cannot be empty".into()));
        }
        const TRAINING_SEQUENCE: &str = "\0lighter-sft";
        self.clear_sequence(TRAINING_SEQUENCE);
        self.training_lm_inputs.clear();
        self.capture_training = true;
        let result = (1..=input_ids.len())
            .map(|end| self.run(TRAINING_SEQUENCE, &input_ids[..end]))
            .collect();
        self.capture_training = false;
        self.clear_sequence(TRAINING_SEQUENCE);
        result
    }

    fn apply_token_gradients(&mut self, gradients: &[f32], learning_rate: f32) -> NativeResult<()> {
        let adapter = self
            .output
            .lora
            .as_mut()
            .ok_or_else(|| NativeError("enable LM-head LoRA before fine-tuning".into()))?;
        let vocabulary = self.output.rows;
        if gradients.len() != self.training_lm_inputs.len() * vocabulary {
            return Err(NativeError(
                "LM-head gradient dimensions are inconsistent".into(),
            ));
        }
        let mut accumulated = adapter.zero_gradients();
        for (hidden, gradient) in self
            .training_lm_inputs
            .iter()
            .zip(gradients.chunks_exact(vocabulary))
        {
            let (current, _) = adapter.backward(hidden, gradient)?;
            LoraAdapter::accumulate(&mut accumulated, &current)?;
        }
        adapter.apply_sgd(&accumulated, learning_rate)?;
        self.training_lm_inputs.clear();
        Ok(())
    }
}

fn layer_cache_id(sequence: &str, layer: usize) -> String {
    format!("{sequence}\0{layer}")
}

fn attend(
    q: &[f32],
    cache: &LayerCache,
    heads: usize,
    kv_heads: usize,
    dim: usize,
) -> NativeResult<Vec<f32>> {
    if q.len() != heads * dim
        || cache.keys.len() != cache.values.len()
        || cache.keys.len() % (kv_heads * dim) != 0
    {
        return Err(NativeError("invalid attention dimensions".into()));
    }
    let positions = cache.keys.len() / (kv_heads * dim);
    let group = heads / kv_heads;
    let mut out = vec![0.0; q.len()];
    for h in 0..heads {
        let kh = h / group;
        let qs = &q[h * dim..(h + 1) * dim];
        let mut scores = Vec::with_capacity(positions);
        for p in 0..positions {
            let offset = (p * kv_heads + kh) * dim;
            scores.push(dot(qs, &cache.keys[offset..offset + dim]) / (dim as f32).sqrt());
        }
        softmax(&mut scores);
        for (p, probability) in scores.into_iter().enumerate() {
            let offset = (p * kv_heads + kh) * dim;
            for d in 0..dim {
                out[h * dim + d] += probability * cache.values[offset + d];
            }
        }
    }
    Ok(out)
}

fn parse_rope_scaling(value: Option<&serde_json::Value>) -> NativeResult<RopeScaling> {
    let Some(map) = value.and_then(|value| value.as_object()) else {
        return Ok(RopeScaling::None);
    };
    let kind = map
        .get("rope_type")
        .or_else(|| map.get("type"))
        .and_then(|value| value.as_str())
        .unwrap_or("linear");
    let number = |name: &str| {
        map.get(name)
            .and_then(|value| value.as_f64())
            .map(|value| value as f32)
            .ok_or_else(|| NativeError(format!("rope_scaling is missing numeric {name}")))
    };
    match kind {
        "linear" => {
            let factor = number("factor")?;
            if factor <= 0.0 {
                return Err(NativeError("RoPE scaling factor must be positive".into()));
            }
            Ok(RopeScaling::Linear { factor })
        }
        "llama3" => {
            let factor = number("factor")?;
            let low_frequency_factor = number("low_freq_factor")?;
            let high_frequency_factor = number("high_freq_factor")?;
            let original_context = number("original_max_position_embeddings")?;
            if factor <= 0.0
                || low_frequency_factor <= 0.0
                || high_frequency_factor <= low_frequency_factor
                || original_context <= 0.0
            {
                return Err(NativeError("invalid llama3 RoPE scaling values".into()));
            }
            Ok(RopeScaling::Llama3 {
                factor,
                low_frequency_factor,
                high_frequency_factor,
                original_context,
            })
        }
        other => Err(NativeError(format!(
            "unsupported rope_scaling type {other:?}"
        ))),
    }
}

fn scaled_frequency(frequency: f32, scaling: RopeScaling) -> f32 {
    match scaling {
        RopeScaling::None => frequency,
        RopeScaling::Linear { factor } => frequency / factor,
        RopeScaling::Llama3 {
            factor,
            low_frequency_factor,
            high_frequency_factor,
            original_context,
        } => {
            let wavelength = std::f32::consts::TAU / frequency;
            let low_wavelength = original_context / low_frequency_factor;
            let high_wavelength = original_context / high_frequency_factor;
            if wavelength > low_wavelength {
                frequency / factor
            } else if wavelength < high_wavelength {
                frequency
            } else {
                let smooth = (original_context / wavelength - low_frequency_factor)
                    / (high_frequency_factor - low_frequency_factor);
                (1.0 - smooth) * frequency / factor + smooth * frequency
            }
        }
    }
}

fn apply_rope(
    x: &mut [f32],
    heads: usize,
    dim: usize,
    position: usize,
    theta: f32,
    scaling: RopeScaling,
) -> NativeResult<()> {
    if x.len() != heads * dim || dim % 2 != 0 {
        return Err(NativeError("invalid RoPE dimensions".into()));
    }
    // Hugging Face Llama uses split-half rotary layout, not adjacent pairs.
    for head in 0..heads {
        let base = head * dim;
        for i in 0..dim / 2 {
            let base_frequency = theta.powf(-((2 * i) as f32 / dim as f32));
            let frequency = scaled_frequency(base_frequency, scaling);
            let angle = position as f32 * frequency;
            let (sin, cos) = angle.sin_cos();
            let a = x[base + i];
            let b = x[base + i + dim / 2];
            x[base + i] = a * cos - b * sin;
            x[base + i + dim / 2] = b * cos + a * sin;
        }
    }
    Ok(())
}

fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> NativeResult<Vec<f32>> {
    if x.len() != weight.len() {
        return Err(NativeError("RMSNorm dimension mismatch".into()));
    }
    let scale = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32 + eps)
        .sqrt()
        .recip();
    Ok(x.iter().zip(weight).map(|(v, w)| v * scale * w).collect())
}

fn rms_norm_heads(
    x: &[f32],
    weight: &[f32],
    heads: usize,
    head_dim: usize,
    eps: f32,
) -> NativeResult<Vec<f32>> {
    if x.len() != heads * head_dim || weight.len() != head_dim {
        return Err(NativeError("per-head RMSNorm dimension mismatch".into()));
    }
    let mut output = Vec::with_capacity(x.len());
    for head in x.chunks_exact(head_dim) {
        output.extend(rms_norm(head, weight, eps)?);
    }
    Ok(output)
}
fn add(a: Vec<f32>, b: Vec<f32>) -> NativeResult<Vec<f32>> {
    if a.len() != b.len() {
        return Err(NativeError("residual dimension mismatch".into()));
    }
    Ok(a.into_iter().zip(b).map(|(x, y)| x + y).collect())
}
fn add_bias(mut value: Vec<f32>, bias: Option<&[f32]>) -> NativeResult<Vec<f32>> {
    if let Some(bias) = bias {
        if value.len() != bias.len() {
            return Err(NativeError("projection bias dimension mismatch".into()));
        }
        for (v, b) in value.iter_mut().zip(bias) {
            *v += b;
        }
    }
    Ok(value)
}
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
fn softmax(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    for value in values {
        *value /= sum;
    }
}

fn matrix(tensors: &mut HashMap<String, StoredTensor>, name: &str) -> NativeResult<Matrix> {
    let tensor = tensors
        .remove(name)
        .ok_or_else(|| NativeError(format!("missing tensor {name}")))?;
    if tensor.shape.len() != 2 {
        return Err(NativeError(format!("tensor {name} is not a matrix")));
    }
    let data = match tensor.data {
        TensorData::F32(data) => MatrixData::F32(Arc::new(data)),
        TensorData::F16(data) => MatrixData::F16(Arc::new(data)),
        TensorData::BF16(data) => MatrixData::BF16(Arc::new(data)),
        TensorData::Quantized(data) => MatrixData::Quantized(data),
    };
    Ok(Matrix {
        rows: tensor.shape[0],
        cols: tensor.shape[1],
        data,
        lora: None,
    })
}
fn vector(tensors: &mut HashMap<String, StoredTensor>, name: &str) -> NativeResult<Vec<f32>> {
    let tensor = tensors
        .remove(name)
        .ok_or_else(|| NativeError(format!("missing tensor {name}")))?;
    if tensor.shape.len() != 1 {
        return Err(NativeError(format!("tensor {name} is not a vector")));
    }
    Ok(tensor.data.into_f32())
}
fn optional_vector(tensors: &mut HashMap<String, StoredTensor>, name: &str) -> Option<Vec<f32>> {
    tensors
        .remove(name)
        .and_then(|tensor| (tensor.shape.len() == 1).then(|| tensor.data.into_f32()))
}

fn load_tensors(
    paths: &[std::path::PathBuf],
    quantization: Option<QuantizationConfig>,
) -> NativeResult<HashMap<String, StoredTensor>> {
    let mut output = HashMap::new();
    for path in paths {
        let bytes = fs::read(path)
            .map_err(|e| NativeError(format!("cannot read {}: {e}", path.display())))?;
        let file = SafeTensors::deserialize(&bytes)
            .map_err(|e| NativeError(format!("invalid {}: {e}", path.display())))?;
        for name in file.names() {
            let view = file
                .tensor(name)
                .map_err(|e| NativeError(format!("cannot read tensor {name}: {e}")))?;
            let mut data = decode(view.dtype(), view.data())?;
            let expected: usize = view.shape().iter().product();
            if data.len() != expected {
                return Err(NativeError(format!(
                    "tensor {name} has invalid byte length"
                )));
            }
            if let (Some(config), [rows, cols]) = (quantization, view.shape()) {
                let values = data.into_f32();
                data = TensorData::Quantized(QuantizedMatrix::quantize(
                    *rows, *cols, &values, config,
                )?);
            }
            if output
                .insert(
                    name.to_owned(),
                    StoredTensor {
                        shape: view.shape().to_vec(),
                        data,
                    },
                )
                .is_some()
            {
                return Err(NativeError(format!("duplicate tensor {name}")));
            }
        }
    }
    Ok(output)
}

fn decode(dtype: Dtype, bytes: &[u8]) -> NativeResult<TensorData> {
    match dtype {
        Dtype::F32 => Ok(TensorData::F32(
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
        )),
        Dtype::F16 => Ok(TensorData::F16(
            bytes
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
                .collect(),
        )),
        Dtype::BF16 => Ok(TensorData::BF16(
            bytes
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
                .collect(),
        )),
        other => Err(NativeError(format!(
            "unsupported native tensor dtype {other:?}; expected F32, F16, or BF16"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn half_precision_matrices_stay_compact_and_multiply() {
        let f16_matrix = Matrix {
            rows: 1,
            cols: 2,
            data: MatrixData::F16(Arc::new(vec![
                f16::from_f32(2.0).to_bits(),
                f16::from_f32(3.0).to_bits(),
            ])),
            lora: None,
        };
        let bf16_matrix = Matrix {
            rows: 1,
            cols: 2,
            data: MatrixData::BF16(Arc::new(vec![
                bf16::from_f32(2.0).to_bits(),
                bf16::from_f32(3.0).to_bits(),
            ])),
            lora: None,
        };
        assert_eq!(f16_matrix.mv(&[4.0, 5.0]).unwrap(), vec![23.0]);
        assert_eq!(bf16_matrix.mv(&[4.0, 5.0]).unwrap(), vec![23.0]);
    }
    #[test]
    fn accepts_checkpoint_specific_clm_model_type() {
        let config = crate::native::HuggingFaceConfig::from_json(
            r#"{"model_type":"clm","architectures":["CLMForCausalLM"]}"#,
        )
        .unwrap();
        assert!(is_supported_decoder_config(&config));

        let qwen3 = crate::native::HuggingFaceConfig::from_json(
            r#"{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"]}"#,
        )
        .unwrap();
        assert!(is_supported_decoder_config(&qwen3));
    }
    #[test]
    fn applies_qwen3_per_head_qk_normalization() {
        let normalized = rms_norm_heads(&[3.0, 4.0, 6.0, 8.0], &[1.0, 1.0], 2, 2, 0.0).unwrap();
        let expected = 2.0_f32.sqrt();
        assert!((normalized[0] - 0.6 * expected).abs() < 1e-6);
        assert!((normalized[1] - 0.8 * expected).abs() < 1e-6);
        assert!((normalized[2] - 0.6 * expected).abs() < 1e-6);
        assert!((normalized[3] - 0.8 * expected).abs() < 1e-6);
    }
    #[test]
    fn rope_position_zero_is_identity() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let original = x.clone();
        apply_rope(&mut x, 1, 4, 0, 10_000.0, RopeScaling::None).unwrap();
        assert_eq!(x, original);
    }
    #[test]
    fn grouped_query_attention_reuses_kv_heads() {
        let cache = LayerCache {
            keys: vec![1.0, 0.0],
            values: vec![3.0, 4.0],
        };
        assert_eq!(
            attend(&[1.0, 0.0, 0.0, 1.0], &cache, 2, 1, 2).unwrap(),
            vec![3.0, 4.0, 3.0, 4.0]
        );
    }
    #[test]
    fn parses_llama3_rope_scaling() {
        let value = serde_json::json!({
            "rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
            "high_freq_factor": 4.0, "original_max_position_embeddings": 8192
        });
        let scaling = parse_rope_scaling(Some(&value)).unwrap();
        assert!(scaled_frequency(0.00001, scaling) < 0.00001);
    }
}
