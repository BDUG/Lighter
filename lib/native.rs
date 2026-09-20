//! Backend-neutral, Candle-free primitives for modern autoregressive inference.
//!
//! This module owns scheduling and decoding policy while [`NativeBackend`] owns
//! model-specific tensor execution.  Consequently an application can integrate
//! CPU, CUDA, Metal, ONNX, GGML, or a remote accelerator without pulling Candle.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, VecDeque};
use std::error::Error;
use std::fmt::{Display, Formatter};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::native_prompt::{ConstraintSpec, OutputConstraint};

pub type NativeResult<T> = Result<T, NativeError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeError(pub String);

impl Display for NativeError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl Error for NativeError {}

/// The common subset of Hugging Face `config.json`, with unknown fields kept so
/// newer architectures remain loadable by a backend without a library release.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HuggingFaceConfig {
    #[serde(default)]
    pub architectures: Vec<String>,
    #[serde(default)]
    pub model_type: String,
    #[serde(default)]
    pub vocab_size: usize,
    #[serde(default)]
    pub hidden_size: usize,
    #[serde(default)]
    pub intermediate_size: usize,
    #[serde(default)]
    pub num_hidden_layers: usize,
    #[serde(default)]
    pub num_attention_heads: usize,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub max_position_embeddings: Option<usize>,
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub torch_dtype: Option<String>,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<TokenIds>,
    #[serde(flatten)]
    pub extensions: HashMap<String, serde_json::Value>,
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}

impl HuggingFaceConfig {
    pub fn from_json(json: &str) -> NativeResult<Self> {
        serde_json::from_str(json)
            .map_err(|e| NativeError(format!("invalid Hugging Face config: {e}")))
    }

    pub fn eos_token_ids(&self) -> Vec<u32> {
        match &self.eos_token_id {
            Some(TokenIds::One(id)) => vec![*id],
            Some(TokenIds::Many(ids)) => ids.clone(),
            None => Vec::new(),
        }
    }
}

/// Files needed to initialize a model from a local Hugging Face snapshot.
#[derive(Debug, Clone)]
pub struct HuggingFaceArtifacts {
    pub root: PathBuf,
    pub config: HuggingFaceConfig,
    pub tokenizer: PathBuf,
    /// Every safetensors shard, in deterministic filename order.
    pub weights: Vec<PathBuf>,
}

impl HuggingFaceArtifacts {
    /// Discovers both single-file and sharded safetensors checkpoints.
    pub fn from_dir(path: impl AsRef<Path>) -> NativeResult<Self> {
        let root = path.as_ref().to_path_buf();
        let config_path = root.join("config.json");
        let tokenizer = root.join("tokenizer.json");
        let config = HuggingFaceConfig::from_json(&read_text(&config_path)?)?;
        if !tokenizer.is_file() {
            return Err(NativeError(format!("missing {}", tokenizer.display())));
        }
        let index = root.join("model.safetensors.index.json");
        let mut weights = if index.is_file() {
            let value: serde_json::Value = serde_json::from_str(&read_text(&index)?)
                .map_err(|e| NativeError(format!("invalid {}: {e}", index.display())))?;
            let map = value
                .get("weight_map")
                .and_then(|v| v.as_object())
                .ok_or_else(|| NativeError("safetensors index has no weight_map".into()))?;
            let mut names: Vec<_> = map.values().filter_map(|v| v.as_str()).collect();
            names.sort_unstable();
            names.dedup();
            names.into_iter().map(|name| root.join(name)).collect()
        } else {
            vec![root.join("model.safetensors")]
        };
        weights.sort();
        if weights.is_empty() || weights.iter().any(|p| !p.is_file()) {
            return Err(NativeError(
                "checkpoint contains missing safetensors shards".into(),
            ));
        }
        Ok(Self {
            root,
            config,
            tokenizer,
            weights,
        })
    }

    /// Downloads the standard snapshot files and all shards through `hf-hub`.
    pub fn from_hub(
        model_id: &str,
        revision: Option<&str>,
        token: Option<String>,
    ) -> NativeResult<Self> {
        match Self::from_hf_hub(model_id, revision, token.clone()) {
            Err(error) if error.0.contains("RelativeUrlWithoutBase") => {
                // hf-hub 0.3.x reads the Hub's Location header and issues a new
                // request with it. The Hub may return a relative resolve-cache
                // URL, which that client cannot parse. ureq follows relative
                // redirects correctly, so use it as a compatibility fallback.
                Self::from_hub_direct(model_id, revision, token)
            }
            result => result,
        }
    }

    fn from_hf_hub(
        model_id: &str,
        revision: Option<&str>,
        token: Option<String>,
    ) -> NativeResult<Self> {
        use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
        let api = ApiBuilder::new()
            .with_progress(false)
            .with_token(token)
            .build()
            .map_err(|e| NativeError(format!("cannot initialize Hugging Face client: {e}")))?;
        let repo = api.repo(Repo::with_revision(
            model_id.into(),
            RepoType::Model,
            revision.unwrap_or("main").into(),
        ));
        let config = repo.get("config.json").map_err(hf_error)?;
        let tokenizer = repo.get("tokenizer.json").map_err(hf_error)?;
        let root = config
            .parent()
            .ok_or_else(|| NativeError("invalid hub cache path".into()))?
            .to_path_buf();
        match repo.get("model.safetensors.index.json") {
            Ok(index) => {
                let value: serde_json::Value = serde_json::from_str(&read_text(&index)?)
                    .map_err(|e| NativeError(format!("invalid safetensors index: {e}")))?;
                let map = value
                    .get("weight_map")
                    .and_then(|v| v.as_object())
                    .ok_or_else(|| NativeError("safetensors index has no weight_map".into()))?;
                let mut names: Vec<_> = map.values().filter_map(|v| v.as_str()).collect();
                names.sort_unstable();
                names.dedup();
                for name in names {
                    repo.get(name).map_err(hf_error)?;
                }
            }
            Err(_) => {
                repo.get("model.safetensors").map_err(hf_error)?;
            }
        }
        let result = Self::from_dir(root)?;
        debug_assert_eq!(result.tokenizer, tokenizer);
        Ok(result)
    }

    fn from_hub_direct(
        model_id: &str,
        revision: Option<&str>,
        token: Option<String>,
    ) -> NativeResult<Self> {
        validate_hub_path(model_id, "model ID")?;
        let revision = revision.unwrap_or("main");
        validate_hub_path(revision, "revision")?;

        let cache = hf_hub::Cache::default();
        let token = token.or_else(|| cache.token());
        let root = cache
            .path()
            .join("lighter")
            .join(model_id.replace('/', "--"))
            .join(revision.replace('/', "--"));
        fs::create_dir_all(&root)
            .map_err(|e| NativeError(format!("cannot create Hub cache {}: {e}", root.display())))?;

        let download = |filename: &str| {
            download_hub_file(model_id, revision, filename, token.as_deref(), &root)
        };
        download("config.json")?;
        download("tokenizer.json")?;
        match download("model.safetensors.index.json") {
            Ok(index) => {
                let value: serde_json::Value = serde_json::from_str(&read_text(&index)?)
                    .map_err(|e| NativeError(format!("invalid safetensors index: {e}")))?;
                let map = value
                    .get("weight_map")
                    .and_then(|v| v.as_object())
                    .ok_or_else(|| NativeError("safetensors index has no weight_map".into()))?;
                let mut names: Vec<_> = map.values().filter_map(|v| v.as_str()).collect();
                names.sort_unstable();
                names.dedup();
                for name in names {
                    validate_hub_filename(name)?;
                    download(name)?;
                }
            }
            Err(error) if error.0.contains("HTTP 404") => {
                download("model.safetensors")?;
            }
            Err(error) => return Err(error),
        }
        Self::from_dir(root)
    }
}

fn download_hub_file(
    model_id: &str,
    revision: &str,
    filename: &str,
    token: Option<&str>,
    root: &Path,
) -> NativeResult<PathBuf> {
    validate_hub_filename(filename)?;
    let destination = root.join(filename);
    if destination.is_file() {
        return Ok(destination);
    }
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            NativeError(format!("cannot create Hub cache {}: {e}", parent.display()))
        })?;
    }
    let url = format!("https://huggingface.co/{model_id}/resolve/{revision}/{filename}");
    let mut request = ureq::get(&url);
    if let Some(token) = token {
        request = request.set("Authorization", &format!("Bearer {token}"));
    }
    let response = request.call().map_err(|error| match error {
        ureq::Error::Status(status, _) => NativeError(format!(
            "Hugging Face download failed for {filename}: HTTP {status}"
        )),
        error => NativeError(format!(
            "Hugging Face download failed for {filename}: {error}"
        )),
    })?;
    let temporary = destination.with_extension(format!("download-{}", std::process::id()));
    let mut file = fs::File::create(&temporary)
        .map_err(|e| NativeError(format!("cannot create {}: {e}", temporary.display())))?;
    if let Err(error) = io::copy(&mut response.into_reader(), &mut file) {
        let _ = fs::remove_file(&temporary);
        return Err(NativeError(format!(
            "cannot download {filename} to {}: {error}",
            destination.display()
        )));
    }
    fs::rename(&temporary, &destination).map_err(|e| {
        let _ = fs::remove_file(&temporary);
        NativeError(format!("cannot save {}: {e}", destination.display()))
    })?;
    Ok(destination)
}

fn validate_hub_path(value: &str, description: &str) -> NativeResult<()> {
    if value.is_empty()
        || value.starts_with('/')
        || value
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
    {
        return Err(NativeError(format!(
            "invalid Hugging Face {description}: {value}"
        )));
    }
    Ok(())
}

fn validate_hub_filename(filename: &str) -> NativeResult<()> {
    validate_hub_path(filename, "filename")
}

fn read_text(path: &Path) -> NativeResult<String> {
    fs::read_to_string(path)
        .map_err(|e| NativeError(format!("cannot read {}: {e}", path.display())))
}

fn hf_error(error: hf_hub::api::sync::ApiError) -> NativeError {
    NativeError(format!("Hugging Face download failed: {error}"))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TokenIds {
    One(u32),
    Many(Vec<u32>),
}

/// Generation controls modelled after widely used vLLM/OpenAI parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SamplingParams {
    pub min_tokens: usize,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub min_p: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
    pub stop_token_ids: Vec<u32>,
    pub stop: Vec<String>,
    pub include_stop_str_in_output: bool,
    pub bad_token_ids: Vec<u32>,
    pub logit_bias: HashMap<u32, f32>,
    pub ignore_eos: bool,
    pub logprobs: Option<usize>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            min_tokens: 0,
            max_tokens: 256,
            temperature: 1.0,
            top_p: 1.0,
            top_k: None,
            min_p: 0.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            repetition_penalty: 1.0,
            seed: 0,
            stop_token_ids: Vec::new(),
            stop: Vec::new(),
            include_stop_str_in_output: false,
            bad_token_ids: Vec::new(),
            logit_bias: HashMap::new(),
            ignore_eos: false,
            logprobs: None,
        }
    }
}

impl SamplingParams {
    pub fn validate(&self) -> NativeResult<()> {
        if self.max_tokens == 0 {
            return Err(NativeError("max_tokens must be positive".into()));
        }
        if self.min_tokens > self.max_tokens {
            return Err(NativeError("min_tokens cannot exceed max_tokens".into()));
        }
        if self.temperature < 0.0 {
            return Err(NativeError("temperature cannot be negative".into()));
        }
        if !(0.0..=1.0).contains(&self.top_p) || self.top_p == 0.0 {
            return Err(NativeError("top_p must be in (0, 1]".into()));
        }
        if !(0.0..=1.0).contains(&self.min_p) {
            return Err(NativeError("min_p must be in [0, 1]".into()));
        }
        if self.repetition_penalty <= 0.0 {
            return Err(NativeError("repetition_penalty must be positive".into()));
        }
        if self.logit_bias.values().any(|value| !value.is_finite()) {
            return Err(NativeError("logit_bias values must be finite".into()));
        }
        if self.stop.iter().any(String::is_empty) {
            return Err(NativeError("stop strings cannot be empty".into()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub id: String,
    pub prompt: String,
    #[serde(default)]
    pub sampling: SamplingParams,
    /// Optional token-level output constraint. The engine masks every token
    /// whose decoded prefix is rejected and stops as soon as it is complete.
    #[serde(default)]
    pub constraint: Option<ConstraintSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenLogprob {
    pub token_id: u32,
    pub logprob: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub id: String,
    pub text: String,
    pub token_ids: Vec<u32>,
    pub prompt_tokens: usize,
    pub finish_reason: FinishReason,
    pub logprobs: Vec<Vec<TokenLogprob>>,
}

/// Model implementations provide tokenization and one next-token logits pass.
/// `position` enables backends to use a paged KV cache or prefix cache.
pub trait NativeBackend {
    fn encode(&self, text: &str) -> NativeResult<Vec<u32>>;
    fn decode(&self, tokens: &[u32]) -> NativeResult<String>;
    fn logits(
        &mut self,
        sequence_id: &str,
        tokens: &[u32],
        position: usize,
    ) -> NativeResult<Vec<f32>>;
    fn remove_sequence(&mut self, _sequence_id: &str) {}

    /// Override to execute a true tensor batch. The default preserves source
    /// compatibility for simple backends while the engine still schedules a
    /// continuous batch.
    fn logits_batch(&mut self, batch: &[BackendInput<'_>]) -> NativeResult<Vec<Vec<f32>>> {
        batch
            .iter()
            .map(|input| self.logits(input.sequence_id, input.tokens, input.position))
            .collect()
    }
}

pub struct BackendInput<'a> {
    pub sequence_id: &'a str,
    pub tokens: &'a [u32],
    pub position: usize,
}

/// Tensor runtimes implement only model execution; this adapter supplies the
/// production Hugging Face tokenizer and implements [`NativeBackend`].
pub trait NativeModel {
    fn forward(
        &mut self,
        sequence_id: &str,
        tokens: &[u32],
        position: usize,
    ) -> NativeResult<Vec<f32>>;
    fn forward_batch(&mut self, batch: &[BackendInput<'_>]) -> NativeResult<Vec<Vec<f32>>> {
        batch
            .iter()
            .map(|x| self.forward(x.sequence_id, x.tokens, x.position))
            .collect()
    }
    fn remove_sequence(&mut self, _sequence_id: &str) {}
}

pub struct HuggingFaceBackend<M> {
    tokenizer: tokenizers::Tokenizer,
    model: M,
}

impl<M> HuggingFaceBackend<M> {
    pub fn new(model: M, artifacts: &HuggingFaceArtifacts) -> NativeResult<Self> {
        let tokenizer = load_tokenizer(&artifacts.tokenizer)?;
        Ok(Self { tokenizer, model })
    }
    pub fn into_model(self) -> M {
        self.model
    }
}

fn load_tokenizer(path: &Path) -> NativeResult<tokenizers::Tokenizer> {
    let bytes = fs::read(path)
        .map_err(|e| NativeError(format!("cannot read tokenizer {}: {e}", path.display())))?;
    match tokenizers::Tokenizer::from_bytes(&bytes) {
        Ok(tokenizer) => Ok(tokenizer),
        Err(original_error) => {
            // Newer tokenizers versions serialize BPE merges as pairs, while
            // tokenizers 0.19 expects the legacy "left right" representation.
            // Convert only well-formed pairs so malformed or unrelated
            // tokenizer files still report the original loading error.
            let mut json: serde_json::Value = serde_json::from_slice(&bytes)
                .map_err(|_| NativeError(format!("cannot load tokenizer: {original_error}")))?;
            let converted_merges = json
                .get_mut("model")
                .and_then(serde_json::Value::as_object_mut)
                .filter(|model| model.get("type").and_then(serde_json::Value::as_str) == Some("BPE"))
                .and_then(|model| model.get_mut("merges"))
                .and_then(serde_json::Value::as_array_mut)
                .map(|merges| {
                    let converted = merges
                        .iter()
                        .map(|merge| {
                            let pair = merge.as_array()?;
                            match pair.as_slice() {
                                [left, right] => Some(serde_json::Value::String(format!(
                                    "{} {}",
                                    left.as_str()?,
                                    right.as_str()?
                                ))),
                                _ => None,
                            }
                        })
                        .collect::<Option<Vec<_>>>();
                    if let Some(converted) = converted {
                        let changed = converted != *merges;
                        *merges = converted;
                        changed
                    } else {
                        false
                    }
                })
                .unwrap_or(false);
            if !converted_merges {
                return Err(NativeError(format!(
                    "cannot load tokenizer: {original_error}"
                )));
            }
            let compatible = serde_json::to_vec(&json)
                .map_err(|e| NativeError(format!("cannot rewrite tokenizer: {e}")))?;
            tokenizers::Tokenizer::from_bytes(&compatible).map_err(|e| {
                NativeError(format!(
                    "cannot load tokenizer after converting BPE merges: {e}"
                ))
            })
        }
    }
}

impl<M: NativeModel> NativeBackend for HuggingFaceBackend<M> {
    fn encode(&self, text: &str) -> NativeResult<Vec<u32>> {
        self.tokenizer
            .encode(text, true)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| NativeError(format!("tokenization failed: {e}")))
    }
    fn decode(&self, tokens: &[u32]) -> NativeResult<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|e| NativeError(format!("decode failed: {e}")))
    }
    fn logits(&mut self, id: &str, tokens: &[u32], position: usize) -> NativeResult<Vec<f32>> {
        self.model.forward(id, tokens, position)
    }
    fn logits_batch(&mut self, batch: &[BackendInput<'_>]) -> NativeResult<Vec<Vec<f32>>> {
        self.model.forward_batch(batch)
    }
    fn remove_sequence(&mut self, id: &str) {
        self.model.remove_sequence(id);
    }
}

struct Sequence {
    request: GenerateRequest,
    prompt_len: usize,
    tokens: Vec<u32>,
    generated: Vec<u32>,
    logprobs: Vec<Vec<TokenLogprob>>,
    rng: StdRng,
    final_text: Option<String>,
    constraint: Option<Box<dyn OutputConstraint + Send + Sync>>,
}

/// A fair continuous-batching queue. Each step advances every active sequence
/// once and then admits queued work into newly available slots.
pub struct NativeEngine<B> {
    backend: B,
    capacity: usize,
    eos_token_ids: Vec<u32>,
    waiting: VecDeque<GenerateRequest>,
    active: Vec<Sequence>,
    cancelled: Vec<String>,
}

impl<B: NativeBackend> NativeEngine<B> {
    pub fn new(backend: B, capacity: usize, eos_token_ids: Vec<u32>) -> NativeResult<Self> {
        if capacity == 0 {
            return Err(NativeError("batch capacity must be positive".into()));
        }
        Ok(Self {
            backend,
            capacity,
            eos_token_ids,
            waiting: VecDeque::new(),
            active: Vec::new(),
            cancelled: Vec::new(),
        })
    }

    pub fn submit(&mut self, request: GenerateRequest) -> NativeResult<()> {
        request.sampling.validate()?;
        if request.id.is_empty() {
            return Err(NativeError("request id cannot be empty".into()));
        }
        if self.waiting.iter().any(|r| r.id == request.id)
            || self.active.iter().any(|s| s.request.id == request.id)
        {
            return Err(NativeError(format!("duplicate request id: {}", request.id)));
        }
        self.waiting.push_back(request);
        Ok(())
    }

    pub fn cancel(&mut self, id: &str) -> bool {
        if let Some(i) = self.waiting.iter().position(|r| r.id == id) {
            self.waiting.remove(i);
            return true;
        }
        if self.active.iter().any(|s| s.request.id == id) {
            self.cancelled.push(id.to_owned());
            return true;
        }
        false
    }

    pub fn is_idle(&self) -> bool {
        self.waiting.is_empty() && self.active.is_empty()
    }
    pub fn queued(&self) -> usize {
        self.waiting.len()
    }
    pub fn active(&self) -> usize {
        self.active.len()
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }

    /// Runs all admitted and queued requests to completion. Servers normally
    /// call [`Self::step`] instead so cancellation and new requests can be
    /// interleaved between decode iterations.
    pub fn run_to_completion(&mut self) -> NativeResult<Vec<GenerateResponse>> {
        let mut responses = Vec::new();
        while !self.is_idle() {
            responses.extend(self.step()?);
        }
        Ok(responses)
    }

    pub fn step(&mut self) -> NativeResult<Vec<GenerateResponse>> {
        self.admit()?;
        let mut completed = Vec::new();
        let inputs: Vec<_> = self
            .active
            .iter()
            .map(|sequence| BackendInput {
                sequence_id: &sequence.request.id,
                tokens: &sequence.tokens,
                position: sequence.tokens.len(),
            })
            .collect();
        let mut batch_logits = self.backend.logits_batch(&inputs)?.into_iter();
        let mut i = 0;
        while i < self.active.len() {
            if self
                .cancelled
                .iter()
                .any(|id| id == &self.active[i].request.id)
            {
                // The batch contains one row per active sequence, including a
                // sequence cancelled after admission.
                batch_logits.next();
                completed.push(self.finish(i, FinishReason::Cancelled)?);
                continue;
            }
            let logits = batch_logits.next().ok_or_else(|| {
                NativeError("backend returned fewer logits rows than requested".into())
            })?;
            let sequence = &mut self.active[i];
            let logits = if let Some(constraint) = sequence.constraint.as_deref() {
                mask_constrained_logits(&self.backend, &logits, &sequence.generated, constraint)?
            } else {
                logits
            };
            let mut effective_sampling = sequence.request.sampling.clone();
            if sequence.generated.len() < effective_sampling.min_tokens {
                effective_sampling
                    .bad_token_ids
                    .extend(effective_sampling.stop_token_ids.iter().copied());
                if !effective_sampling.ignore_eos {
                    effective_sampling
                        .bad_token_ids
                        .extend(self.eos_token_ids.iter().copied());
                }
                effective_sampling.bad_token_ids.sort_unstable();
                effective_sampling.bad_token_ids.dedup();
            }
            let (token, alternatives) = sample(
                &logits,
                &sequence.tokens,
                &effective_sampling,
                &mut sequence.rng,
            )?;
            self.active[i].tokens.push(token);
            self.active[i].generated.push(token);
            self.active[i].logprobs.push(alternatives);
            let p = self.active[i].request.sampling.clone();
            let can_stop = self.active[i].generated.len() >= p.min_tokens;
            let stopped = can_stop
                && (p.stop_token_ids.contains(&token)
                    || (!p.ignore_eos && self.eos_token_ids.contains(&token)));
            let text_stop = if can_stop && !p.stop.is_empty() {
                let text = self.backend.decode(&self.active[i].generated)?;
                p.stop.iter().find_map(|stop| {
                    text.find(stop).map(|at| {
                        if p.include_stop_str_in_output {
                            text[..at + stop.len()].to_owned()
                        } else {
                            text[..at].to_owned()
                        }
                    })
                })
            } else {
                None
            };
            if let Some(text) = text_stop {
                self.active[i].final_text = Some(text);
            }
            let constraint_complete = if let Some(constraint) = self.active[i].constraint.as_deref()
            {
                let generated = self.backend.decode(&self.active[i].generated)?;
                constraint.is_complete(&generated)
            } else {
                false
            };
            if stopped || constraint_complete || self.active[i].final_text.is_some() {
                completed.push(self.finish(i, FinishReason::Stop)?);
            } else if self.active[i].generated.len() >= p.max_tokens {
                completed.push(self.finish(i, FinishReason::Length)?);
            } else {
                i += 1;
            }
        }
        self.cancelled
            .retain(|id| self.active.iter().any(|s| &s.request.id == id));
        self.admit()?;
        Ok(completed)
    }

    fn admit(&mut self) -> NativeResult<()> {
        while self.active.len() < self.capacity {
            let Some(request) = self.waiting.pop_front() else {
                break;
            };
            let tokens = self.backend.encode(&request.prompt)?;
            let prompt_len = tokens.len();
            let seed = request.sampling.seed;
            let constraint = request
                .constraint
                .as_ref()
                .map(ConstraintSpec::compile)
                .transpose()?;
            self.active.push(Sequence {
                request,
                prompt_len,
                tokens,
                generated: Vec::new(),
                logprobs: Vec::new(),
                rng: StdRng::seed_from_u64(seed),
                final_text: None,
                constraint,
            });
        }
        Ok(())
    }

    fn finish(&mut self, index: usize, reason: FinishReason) -> NativeResult<GenerateResponse> {
        let sequence = self.active.remove(index);
        self.backend.remove_sequence(&sequence.request.id);
        Ok(GenerateResponse {
            id: sequence.request.id,
            text: match sequence.final_text {
                Some(text) => text,
                None => self.backend.decode(&sequence.generated)?,
            },
            token_ids: sequence.generated,
            prompt_tokens: sequence.prompt_len,
            finish_reason: reason,
            logprobs: sequence.logprobs,
        })
    }
}

fn mask_constrained_logits<B: NativeBackend>(
    backend: &B,
    logits: &[f32],
    generated: &[u32],
    constraint: &dyn OutputConstraint,
) -> NativeResult<Vec<f32>> {
    let mut candidate = generated.to_vec();
    candidate.push(0);
    let mut masked = logits.to_vec();
    let mut allowed = 0usize;
    for (token_id, score) in masked.iter_mut().enumerate() {
        if !score.is_finite() {
            continue;
        }
        candidate[generated.len()] = token_id as u32;
        let text = backend.decode(&candidate)?;
        if constraint.allows_prefix(&text) {
            allowed += 1;
        } else {
            *score = f32::NEG_INFINITY;
        }
    }
    if allowed == 0 {
        return Err(NativeError(format!(
            "no token can satisfy output constraint {} after {:?}",
            constraint.description(),
            backend.decode(generated)?
        )));
    }
    Ok(masked)
}

fn sample(
    logits: &[f32],
    history: &[u32],
    p: &SamplingParams,
    rng: &mut StdRng,
) -> NativeResult<(u32, Vec<TokenLogprob>)> {
    if logits.is_empty() {
        return Err(NativeError("backend returned empty logits".into()));
    }
    if logits.iter().any(|x| x.is_nan()) {
        return Err(NativeError("backend returned NaN logits".into()));
    }
    let mut counts = HashMap::<u32, usize>::new();
    for &token in history {
        *counts.entry(token).or_default() += 1;
    }
    let mut scores: Vec<(u32, f32)> = logits
        .iter()
        .copied()
        .enumerate()
        .map(|(id, mut score)| {
            score += p.logit_bias.get(&(id as u32)).copied().unwrap_or(0.0);
            if p.bad_token_ids.contains(&(id as u32)) {
                return (id as u32, f32::NEG_INFINITY);
            }
            let count = counts.get(&(id as u32)).copied().unwrap_or(0);
            if count > 0 {
                score = if score >= 0.0 {
                    score / p.repetition_penalty
                } else {
                    score * p.repetition_penalty
                };
                score -= p.presence_penalty + p.frequency_penalty * count as f32;
            }
            (id as u32, score)
        })
        .collect();
    scores.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    if !scores[0].1.is_finite() {
        return Err(NativeError(
            "all tokens were masked or had invalid logits".into(),
        ));
    }
    if p.temperature == 0.0 {
        return Ok((
            scores[0].0,
            top_logprobs(&scores, p.logprobs.unwrap_or(0), 1.0),
        ));
    }
    if let Some(k) = p.top_k {
        scores.truncate(k.max(1).min(scores.len()));
    }
    let max = scores[0].1;
    let mut probs: Vec<f32> = scores
        .iter()
        .map(|(_, x)| ((x - max) / p.temperature).exp())
        .collect();
    let total: f32 = probs.iter().sum();
    for prob in &mut probs {
        *prob /= total;
    }
    let max_prob = probs[0];
    let mut kept = 0;
    let mut cumulative = 0.0;
    for prob in &probs {
        if kept > 0 && (cumulative >= p.top_p || *prob < max_prob * p.min_p) {
            break;
        }
        cumulative += *prob;
        kept += 1;
    }
    scores.truncate(kept.max(1));
    probs.truncate(kept.max(1));
    let norm: f32 = probs.iter().sum();
    let mut draw = rng.gen::<f32>() * norm;
    let mut selected = scores.last().unwrap().0;
    for ((id, _), prob) in scores.iter().zip(&probs) {
        draw -= prob;
        if draw <= 0.0 {
            selected = *id;
            break;
        }
    }
    Ok((
        selected,
        top_logprobs(&scores, p.logprobs.unwrap_or(0), p.temperature),
    ))
}

fn top_logprobs(scores: &[(u32, f32)], n: usize, temperature: f32) -> Vec<TokenLogprob> {
    if n == 0 {
        return Vec::new();
    }
    let t = temperature.max(f32::EPSILON);
    let max = scores[0].1;
    let z: f32 = scores.iter().map(|(_, x)| ((x - max) / t).exp()).sum();
    scores
        .iter()
        .take(n)
        .map(|(token_id, x)| TokenLogprob {
            token_id: *token_id,
            logprob: (x - max) / t - z.ln(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::sync::Arc;

    struct Backend;
    impl NativeBackend for Backend {
        fn encode(&self, text: &str) -> NativeResult<Vec<u32>> {
            Ok(text.bytes().map(u32::from).collect())
        }
        fn decode(&self, tokens: &[u32]) -> NativeResult<String> {
            Ok(tokens.iter().map(|x| char::from_u32(*x).unwrap()).collect())
        }
        fn logits(&mut self, _: &str, _: &[u32], _: usize) -> NativeResult<Vec<f32>> {
            let mut v = vec![-10.0; 128];
            v[b'x' as usize] = 10.0;
            Ok(v)
        }
    }

    #[test]
    fn parses_forward_compatible_hf_config() {
        let c = HuggingFaceConfig::from_json(
            r#"{"model_type":"qwen2","eos_token_id":[1,2],"new_rope_feature":true}"#,
        )
        .unwrap();
        assert_eq!(c.eos_token_ids(), vec![1, 2]);
        assert!(c.extensions.contains_key("new_rope_feature"));
    }

    #[test]
    fn loads_bpe_tokenizer_with_pair_merges() {
        let root = std::env::temp_dir().join(format!(
            "lighter-tokenizer-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("tokenizer.json");
        fs::write(
            &path,
            r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"a":0,"b":1,"ab":2},"merges":[["a","b"]]}}"#,
        )
        .unwrap();

        assert!(tokenizers::Tokenizer::from_file(&path).is_err());
        let tokenizer = load_tokenizer(&path).unwrap();
        assert_eq!(tokenizer.get_vocab_size(false), 3);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn continuously_batches_and_finishes() {
        let mut e = NativeEngine::new(Backend, 1, vec![]).unwrap();
        for id in ["a", "b"] {
            e.submit(GenerateRequest {
                id: id.into(),
                prompt: "p".into(),
                constraint: None,
                sampling: SamplingParams {
                    max_tokens: 2,
                    temperature: 0.0,
                    ..Default::default()
                },
            })
            .unwrap();
        }
        assert!(e.step().unwrap().is_empty());
        let first = e.step().unwrap();
        assert_eq!(first[0].text, "xx");
        assert_eq!(e.active(), 1);
    }

    #[test]
    fn penalties_change_greedy_choice() {
        let p = SamplingParams {
            temperature: 0.0,
            presence_penalty: 2.0,
            ..Default::default()
        };
        let (token, _) = sample(&[2.0, 1.0], &[0], &p, &mut StdRng::seed_from_u64(1)).unwrap();
        assert_eq!(token, 1);
    }

    #[test]
    fn uses_backend_batch_and_honors_text_stop() {
        struct BatchBackend(Arc<AtomicUsize>);
        impl NativeBackend for BatchBackend {
            fn encode(&self, _: &str) -> NativeResult<Vec<u32>> {
                Ok(vec![b'p' as u32])
            }
            fn decode(&self, ids: &[u32]) -> NativeResult<String> {
                Ok(ids.iter().map(|id| char::from_u32(*id).unwrap()).collect())
            }
            fn logits(&mut self, _: &str, _: &[u32], _: usize) -> NativeResult<Vec<f32>> {
                panic!("the batch implementation must be used")
            }
            fn logits_batch(&mut self, batch: &[BackendInput<'_>]) -> NativeResult<Vec<Vec<f32>>> {
                self.0.fetch_add(1, AtomicOrdering::Relaxed);
                Ok(batch
                    .iter()
                    .map(|_| {
                        let mut logits = vec![-10.0; 128];
                        logits[b'x' as usize] = 10.0;
                        logits
                    })
                    .collect())
            }
        }
        let calls = Arc::new(AtomicUsize::new(0));
        let mut engine = NativeEngine::new(BatchBackend(calls.clone()), 2, vec![]).unwrap();
        for id in ["a", "b"] {
            engine
                .submit(GenerateRequest {
                    id: id.into(),
                    prompt: "p".into(),
                    constraint: None,
                    sampling: SamplingParams {
                        temperature: 0.0,
                        stop: vec!["xx".into()],
                        max_tokens: 5,
                        ..Default::default()
                    },
                })
                .unwrap();
        }
        let responses = engine.run_to_completion().unwrap();
        assert_eq!(calls.load(AtomicOrdering::Relaxed), 2);
        assert!(responses.iter().all(|response| response.text.is_empty()));
    }

    #[test]
    fn discovers_sharded_hugging_face_checkpoint() {
        let root = std::env::temp_dir().join(format!("lighter-native-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("config.json"), r#"{"model_type":"llama"}"#).unwrap();
        fs::write(root.join("tokenizer.json"), "{}").unwrap();
        fs::write(root.join("a.safetensors"), []).unwrap();
        fs::write(root.join("b.safetensors"), []).unwrap();
        fs::write(
            root.join("model.safetensors.index.json"),
            r#"{"weight_map":{"a":"a.safetensors","b":"b.safetensors","c":"a.safetensors"}}"#,
        )
        .unwrap();
        let artifacts = HuggingFaceArtifacts::from_dir(&root).unwrap();
        assert_eq!(
            artifacts.weights,
            vec![root.join("a.safetensors"), root.join("b.safetensors")]
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn bad_tokens_and_bias_are_applied() {
        let params = SamplingParams {
            temperature: 0.0,
            bad_token_ids: vec![0],
            logit_bias: HashMap::from([(2, 5.0)]),
            ..Default::default()
        };
        let (token, _) = sample(
            &[10.0, 2.0, 0.0],
            &[],
            &params,
            &mut StdRng::seed_from_u64(1),
        )
        .unwrap();
        assert_eq!(token, 2);
    }

    #[test]
    fn minimum_tokens_masks_eos_instead_of_emitting_it() {
        struct EosBackend;
        impl NativeBackend for EosBackend {
            fn encode(&self, _: &str) -> NativeResult<Vec<u32>> {
                Ok(vec![3])
            }
            fn decode(&self, ids: &[u32]) -> NativeResult<String> {
                Ok(format!("{ids:?}"))
            }
            fn logits(&mut self, _: &str, _: &[u32], _: usize) -> NativeResult<Vec<f32>> {
                Ok(vec![0.0, 10.0, 9.0, 0.0])
            }
        }
        let mut engine = NativeEngine::new(EosBackend, 1, vec![1]).unwrap();
        engine
            .submit(GenerateRequest {
                id: "minimum".into(),
                prompt: "p".into(),
                constraint: None,
                sampling: SamplingParams {
                    min_tokens: 2,
                    max_tokens: 4,
                    temperature: 0.0,
                    ..Default::default()
                },
            })
            .unwrap();
        let response = engine.run_to_completion().unwrap().remove(0);
        assert_eq!(response.token_ids, vec![2, 2, 1]);
        assert_eq!(response.finish_reason, FinishReason::Stop);
    }

    #[test]
    fn constraint_masks_logits_and_stops_on_complete_value() {
        struct ChoiceBackend;
        impl NativeBackend for ChoiceBackend {
            fn encode(&self, _: &str) -> NativeResult<Vec<u32>> {
                Ok(vec![])
            }
            fn decode(&self, ids: &[u32]) -> NativeResult<String> {
                Ok(ids.iter().filter_map(|id| char::from_u32(*id)).collect())
            }
            fn logits(&mut self, _: &str, _: &[u32], _: usize) -> NativeResult<Vec<f32>> {
                let mut logits = vec![f32::NEG_INFINITY; 128];
                logits[b'x' as usize] = 20.0;
                logits[b'o' as usize] = 10.0;
                logits[b'k' as usize] = 9.0;
                Ok(logits)
            }
        }
        let mut engine = NativeEngine::new(ChoiceBackend, 1, vec![]).unwrap();
        engine
            .submit(GenerateRequest {
                id: "guided".into(),
                prompt: String::new(),
                sampling: SamplingParams {
                    temperature: 0.0,
                    max_tokens: 8,
                    ..Default::default()
                },
                constraint: Some(ConstraintSpec::Choice {
                    choices: vec!["ok".into()],
                }),
            })
            .unwrap();
        let response = engine.run_to_completion().unwrap().remove(0);
        assert_eq!(response.text, "ok");
        assert_eq!(response.finish_reason, FinishReason::Stop);
    }
}
