//! Native Transformers-style loading, tokenization and causal generation (stage 1).
//! This module implements a documented subset, not every Hugging Face architecture.
use crate::native::*;
use crate::native_advanced::QuantizationConfig;
use crate::NativeLlama;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default)]
pub struct LoadOptions {
    pub revision: Option<String>,
    pub token: Option<String>,
    pub quantization: Option<QuantizationConfig>,
}
fn local_or_hub_file(source: &str, filename: &str, options: &LoadOptions) -> NativeResult<PathBuf> {
    if Path::new(source).is_dir() {
        return Ok(Path::new(source).join(filename));
    }
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_progress(false)
        .with_token(options.token.clone())
        .build()
        .map_err(|e| NativeError(e.to_string()))?;
    let revision = options.revision.as_deref().unwrap_or("main");
    let repo = api.repo(hf_hub::Repo::with_revision(
        source.into(),
        hf_hub::RepoType::Model,
        revision.into(),
    ));
    match repo.get(filename) {
        Ok(path) => Ok(path),
        Err(error) if error.to_string().contains("RelativeUrlWithoutBase") => {
            let cache = hf_hub::Cache::default();
            let token = options.token.clone().or_else(|| cache.token());
            let root = cache
                .path()
                .join("lighter")
                .join(source.replace('/', "--"))
                .join(revision.replace('/', "--"));
            crate::native::download_hub_file(source, revision, filename, token.as_deref(), &root)
        }
        Err(error) => Err(NativeError(format!("cannot load {filename}: {error}"))),
    }
}

fn optional_file(
    source: &str,
    filename: &str,
    options: &LoadOptions,
) -> NativeResult<Option<PathBuf>> {
    if Path::new(source).is_dir() {
        let path = Path::new(source).join(filename);
        return Ok(path.is_file().then_some(path));
    }
    match local_or_hub_file(source, filename, options) {
        Ok(path) => Ok(Some(path)),
        Err(error) if error.0.contains("404") => Ok(None),
        Err(error) => Err(error),
    }
}

pub struct AutoConfig;
impl AutoConfig {
    pub fn from_pretrained(source: &str, options: &LoadOptions) -> NativeResult<HuggingFaceConfig> {
        let path = local_or_hub_file(source, "config.json", options)?;
        HuggingFaceConfig::from_json(
            &fs::read_to_string(path).map_err(|e| NativeError(e.to_string()))?,
        )
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub enum Padding {
    #[default]
    None,
    Longest,
    MaxLength(usize),
}
#[derive(Debug, Clone, Copy, Default)]
pub enum Side {
    Left,
    #[default]
    Right,
}
#[derive(Debug, Clone)]
pub struct TokenizationOptions {
    pub add_special_tokens: bool,
    pub padding: Padding,
    pub padding_side: Side,
    pub truncation: bool,
    pub truncation_side: Side,
    pub max_length: Option<usize>,
}
impl Default for TokenizationOptions {
    fn default() -> Self {
        Self {
            add_special_tokens: true,
            padding: Padding::None,
            padding_side: Side::Right,
            truncation: false,
            truncation_side: Side::Right,
            max_length: None,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchEncoding {
    pub input_ids: Vec<Vec<u32>>,
    pub attention_mask: Vec<Vec<u32>>,
    pub token_type_ids: Vec<Vec<u32>>,
}
pub struct AutoTokenizer {
    tokenizer: tokenizers::Tokenizer,
    pub pad_token_id: Option<u32>,
}
impl AutoTokenizer {
    /// Loads tokenizer.json without downloading model weights. Padding ID can be
    /// supplied explicitly when the tokenizer has no configured padding.
    pub fn from_pretrained(source: &str, options: &LoadOptions) -> NativeResult<Self> {
        let path = local_or_hub_file(source, "tokenizer.json", options)?;
        let mut result = Self::from_file(path)?;
        if !Path::new(source).is_dir() {
            for filename in [
                "config.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
            ] {
                if let Some(path) = optional_file(source, filename, options)? {
                    let value: Value = serde_json::from_str(
                        &fs::read_to_string(path).map_err(|e| NativeError(e.to_string()))?,
                    )
                    .map_err(|e| NativeError(e.to_string()))?;
                    if let Some(id) = value
                        .get("pad_token_id")
                        .and_then(Value::as_u64)
                        .and_then(|n| u32::try_from(n).ok())
                    {
                        result.pad_token_id = Some(id);
                    }
                    if let Some(value) = value.get("pad_token") {
                        if let Some(token) = value
                            .as_str()
                            .or_else(|| value.get("content").and_then(Value::as_str))
                        {
                            result.pad_token_id =
                                result.tokenizer.token_to_id(token).or(result.pad_token_id);
                        }
                    }
                }
            }
        }
        Ok(result)
    }
    pub fn from_file(path: impl AsRef<Path>) -> NativeResult<Self> {
        let path = path.as_ref();
        let mut tokenizer = crate::native::load_tokenizer(path)?;
        let configured = tokenizer.get_padding().map(|p| p.pad_id);
        let mut pad = configured;
        if let Some(parent) = path.parent() {
            for filename in [
                "config.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
            ] {
                let file = parent.join(filename);
                if !file.exists() {
                    continue;
                }
                let value: Value = serde_json::from_str(
                    &fs::read_to_string(file).map_err(|e| NativeError(e.to_string()))?,
                )
                .map_err(|e| NativeError(e.to_string()))?;
                if let Some(id) = value
                    .get("pad_token_id")
                    .and_then(Value::as_u64)
                    .and_then(|n| u32::try_from(n).ok())
                {
                    pad = Some(id);
                }
                if let Some(value) = value.get("pad_token") {
                    if let Some(text) = value
                        .as_str()
                        .or_else(|| value.get("content").and_then(Value::as_str))
                    {
                        pad = tokenizer.token_to_id(text).or(pad);
                    }
                }
            }
        }
        tokenizer.with_padding(None);
        tokenizer
            .with_truncation(None)
            .map_err(|e| NativeError(e.to_string()))?;
        Ok(Self {
            tokenizer,
            pad_token_id: pad,
        })
    }
    pub fn encode_batch(
        &self,
        texts: &[String],
        options: &TokenizationOptions,
    ) -> NativeResult<BatchEncoding> {
        if options.max_length == Some(0) || matches!(options.padding, Padding::MaxLength(0)) {
            return Err(NativeError("token limits must be positive".into()));
        }
        if options.truncation && options.max_length.is_none() {
            return Err(NativeError("truncation requires max_length".into()));
        }
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), options.add_special_tokens)
            .map_err(|e| NativeError(e.to_string()))?;
        let mut ids = Vec::new();
        let mut types = Vec::new();
        for e in encodings {
            let mut row = e.get_ids().to_vec();
            let mut type_row = e.get_type_ids().to_vec();
            if let Some(limit) = options.max_length {
                if row.len() > limit {
                    if !options.truncation {
                        return Err(NativeError(
                            "tokenized input exceeds max_length; enable truncation explicitly"
                                .into(),
                        ));
                    }
                    match options.truncation_side {
                        Side::Right => {
                            row.truncate(limit);
                            type_row.truncate(limit);
                        }
                        Side::Left => {
                            let start = row.len() - limit;
                            row = row[start..].to_vec();
                            type_row = type_row[start..].to_vec();
                        }
                    }
                }
            }
            ids.push(row);
            types.push(type_row);
        }
        let target = match options.padding {
            Padding::None => None,
            Padding::Longest => Some(ids.iter().map(Vec::len).max().unwrap_or(0)),
            Padding::MaxLength(n) => Some(n),
        };
        let mut masks = Vec::new();
        for (row, type_row) in ids.iter_mut().zip(&mut types) {
            let mut mask = vec![1; row.len()];
            if let Some(target) = target {
                if row.len() > target {
                    return Err(NativeError("input exceeds fixed padding length".into()));
                }
                let count = target - row.len();
                if count > 0 {
                    let pad = self
                        .pad_token_id
                        .ok_or_else(|| NativeError("padding requires pad_token_id".into()))?;
                    if self.tokenizer.id_to_token(pad).is_none() {
                        return Err(NativeError("pad_token_id is absent from vocabulary".into()));
                    }
                    match options.padding_side {
                        Side::Right => {
                            row.extend(vec![pad; count]);
                            type_row.extend(vec![0; count]);
                            mask.extend(vec![0; count]);
                        }
                        Side::Left => {
                            row.splice(0..0, vec![pad; count]);
                            type_row.splice(0..0, vec![0; count]);
                            mask.splice(0..0, vec![0; count]);
                        }
                    }
                }
            }
            masks.push(mask);
        }
        Ok(BatchEncoding {
            input_ids: ids,
            attention_mask: masks,
            token_type_ids: types,
        })
    }
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> NativeResult<String> {
        self.tokenizer
            .decode(ids, skip_special_tokens)
            .map_err(|e| NativeError(e.to_string()))
    }
    pub fn batch_decode(
        &self,
        ids: &[Vec<u32>],
        skip_special_tokens: bool,
    ) -> NativeResult<Vec<String>> {
        ids.iter()
            .map(|row| self.decode(row, skip_special_tokens))
            .collect()
    }
    pub fn save_pretrained(&self, directory: impl AsRef<Path>) -> NativeResult<()> {
        fs::create_dir_all(&directory).map_err(|e| NativeError(e.to_string()))?;
        let mut tokenizer = self.tokenizer.clone();
        if let Some(id) = self.pad_token_id {
            let token = tokenizer
                .id_to_token(id)
                .ok_or_else(|| NativeError("pad ID absent from vocabulary".into()))?;
            tokenizer.with_padding(Some(tokenizers::PaddingParams {
                pad_id: id,
                pad_token: token,
                ..Default::default()
            }));
        }
        tokenizer
            .save(directory.as_ref().join("tokenizer.json"), true)
            .map_err(|e| NativeError(e.to_string()))
    }
}

fn null_list<'de, D, T>(deserializer: D) -> Result<Vec<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Ok(Option::<Vec<T>>::deserialize(deserializer)?.unwrap_or_default())
}

/// Supported HF generation fields. Unknown non-neutral settings fail validation
/// instead of silently pretending to implement another decoding algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GenerationConfig {
    pub max_new_tokens: Option<usize>,
    pub max_length: Option<usize>,
    pub min_new_tokens: Option<usize>,
    pub min_length: usize,
    pub do_sample: bool,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub no_repeat_ngram_size: usize,
    #[serde(deserialize_with = "null_list")]
    pub bad_words_ids: Vec<Vec<u32>>,
    pub num_beams: usize,
    pub num_return_sequences: usize,
    pub eos_token_id: Option<TokenIds>,
    #[serde(deserialize_with = "null_list")]
    pub stop_strings: Vec<String>,
    #[serde(flatten)]
    pub extensions: BTreeMap<String, Value>,
}
impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: None,
            max_length: Some(20),
            min_new_tokens: None,
            min_length: 0,
            do_sample: false,
            temperature: 1.0,
            top_p: 1.0,
            top_k: 50,
            repetition_penalty: 1.0,
            no_repeat_ngram_size: 0,
            bad_words_ids: Vec::new(),
            num_beams: 1,
            num_return_sequences: 1,
            eos_token_id: None,
            stop_strings: Vec::new(),
            extensions: BTreeMap::new(),
        }
    }
}
impl GenerationConfig {
    pub fn from_pretrained(source: &str, options: &LoadOptions) -> NativeResult<Self> {
        let path = local_or_hub_file(source, "generation_config.json", options)?;
        serde_json::from_str(&fs::read_to_string(path).map_err(|e| NativeError(e.to_string()))?)
            .map_err(|e| NativeError(e.to_string()))
    }
    pub fn save_pretrained(&self, directory: impl AsRef<Path>) -> NativeResult<()> {
        fs::create_dir_all(&directory).map_err(|e| NativeError(e.to_string()))?;
        fs::write(
            directory.as_ref().join("generation_config.json"),
            serde_json::to_vec_pretty(self).map_err(|e| NativeError(e.to_string()))?,
        )
        .map_err(|e| NativeError(e.to_string()))
    }
    pub fn sampling(
        &self,
        prompt_tokens: usize,
        context: Option<usize>,
        seed: u64,
    ) -> NativeResult<SamplingParams> {
        if self.num_beams != 1 {
            return Err(NativeError(
                "beam search is not implemented in native parity stage 1".into(),
            ));
        }
        if self.num_return_sequences != 1 {
            return Err(NativeError(
                "stage 1 supports num_return_sequences=1".into(),
            ));
        }
        for (key, value) in &self.extensions {
            let neutral = match key.as_str() {
                "_from_model_config" | "transformers_version" | "bos_token_id" | "pad_token_id" => {
                    true
                }
                "use_cache" => value == &Value::Bool(true),
                "early_stopping"
                | "output_scores"
                | "output_logits"
                | "output_attentions"
                | "output_hidden_states"
                | "return_dict_in_generate" => value == &Value::Bool(false),
                "length_penalty" => value.as_f64() == Some(1.0),
                "num_beam_groups" => value.as_u64() == Some(1),
                "diversity_penalty" | "penalty_alpha" => {
                    value.is_null() || value.as_f64() == Some(0.0)
                }
                "typical_p" => value.as_f64() == Some(1.0),
                _ => value.is_null(),
            };
            if !neutral {
                return Err(NativeError(format!(
                    "unsupported generation setting {key}={value}"
                )));
            }
        }
        let max_tokens = match self.max_new_tokens {
            Some(n) => n,
            None => self
                .max_length
                .and_then(|n| n.checked_sub(prompt_tokens))
                .filter(|&n| n > 0)
                .ok_or_else(|| NativeError("max_length leaves no room for generation".into()))?,
        };
        if context.is_some_and(|limit| {
            prompt_tokens
                .checked_add(max_tokens)
                .is_none_or(|total| total > limit)
        }) {
            return Err(NativeError(
                "prompt plus output exceeds model context".into(),
            ));
        }
        let min_tokens = self
            .min_new_tokens
            .unwrap_or_else(|| self.min_length.saturating_sub(prompt_tokens));
        let mut sampling = SamplingParams {
            max_tokens,
            min_tokens,
            temperature: if self.do_sample {
                self.temperature
            } else {
                0.0
            },
            top_p: self.top_p,
            top_k: if self.top_k == 0 {
                None
            } else {
                Some(self.top_k)
            },
            repetition_penalty: self.repetition_penalty,
            no_repeat_ngram_size: self.no_repeat_ngram_size,
            bad_words_ids: self.bad_words_ids.clone(),
            stop: self.stop_strings.clone(),
            seed,
            ..Default::default()
        };
        if !self.temperature.is_finite() || self.temperature <= 0.0 {
            return Err(NativeError(
                "temperature must be finite and positive; select greedy with do_sample=false"
                    .into(),
            ));
        }
        if let Some(ids) = &self.eos_token_id {
            sampling.ignore_eos = true;
            sampling.stop_token_ids = match ids {
                TokenIds::One(id) => vec![*id],
                TokenIds::Many(ids) => ids.clone(),
            };
        }
        sampling.validate()?;
        Ok(sampling)
    }
}

pub struct AutoModelForCausalLM;
impl AutoModelForCausalLM {
    pub fn from_pretrained(
        source: &str,
        options: &LoadOptions,
    ) -> NativeResult<(NativeLlama, HuggingFaceArtifacts)> {
        let config = AutoConfig::from_pretrained(source, options)?;
        if !crate::native_llama::is_supported_decoder_config(&config) {
            return Err(NativeError(format!("native causal loader does not support model_type {:?}; no model weights were loaded", config.model_type)));
        }
        let artifacts = if Path::new(source).is_dir() {
            HuggingFaceArtifacts::from_dir(source)?
        } else {
            HuggingFaceArtifacts::from_hub(
                source,
                options.revision.as_deref(),
                options.token.clone(),
            )?
        };
        let model = match options.quantization {
            Some(config) => NativeLlama::load_quantized(&artifacts, config)?,
            None => NativeLlama::load(&artifacts)?,
        };
        Ok((model, artifacts))
    }
}

/// Reusable, continuously batched native text generation. Requests retain input order.
pub struct TextGenerationPipeline<B: NativeBackend> {
    engine: NativeEngine<B>,
    pub generation_config: GenerationConfig,
    context: Option<usize>,
    next_id: u64,
}
impl TextGenerationPipeline<HuggingFaceBackend<NativeLlama>> {
    pub fn from_pretrained(
        source: &str,
        options: &LoadOptions,
        batch_size: usize,
    ) -> NativeResult<Self> {
        let (model, artifacts) = AutoModelForCausalLM::from_pretrained(source, options)?;
        let generation_path = optional_file(source, "generation_config.json", options)?;
        let generation_config = if let Some(generation_path) = generation_path {
            serde_json::from_str(
                &fs::read_to_string(generation_path).map_err(|e| NativeError(e.to_string()))?,
            )
            .map_err(|e| NativeError(e.to_string()))?
        } else {
            GenerationConfig::default()
        };
        Self::new(
            HuggingFaceBackend::new(model, &artifacts)?,
            artifacts.config.eos_token_ids(),
            artifacts.config.max_position_embeddings,
            batch_size,
            generation_config,
        )
    }
}
impl<B: NativeBackend> TextGenerationPipeline<B> {
    pub fn new(
        backend: B,
        eos_ids: Vec<u32>,
        context: Option<usize>,
        batch_size: usize,
        generation_config: GenerationConfig,
    ) -> NativeResult<Self> {
        Ok(Self {
            engine: NativeEngine::new(backend, batch_size, eos_ids)?,
            generation_config,
            context,
            next_id: 0,
        })
    }
    pub fn generate(
        &mut self,
        prompts: &[String],
        seed: u64,
    ) -> NativeResult<Vec<GenerateResponse>> {
        // Validate the entire batch before mutating the engine.
        let mut requests = Vec::new();
        for (index, prompt) in prompts.iter().enumerate() {
            let ids = self.engine.backend().encode(prompt)?;
            if ids.is_empty() {
                return Err(NativeError(
                    "prompt must tokenize to at least one token".into(),
                ));
            }
            let sampling = self.generation_config.sampling(
                ids.len(),
                self.context,
                seed.wrapping_add(index as u64),
            )?;
            let id = format!("pipeline-{}-{index}", self.next_id);
            requests.push(GenerateRequest {
                id,
                prompt: prompt.clone(),
                sampling,
                constraint: None,
            });
        }
        self.next_id = self.next_id.wrapping_add(1);
        let order: Vec<_> = requests.iter().map(|r| r.id.clone()).collect();
        for request in requests {
            if let Err(error) = self.engine.submit(request) {
                self.engine.abort_all();
                return Err(error);
            }
        }
        let responses = match self.engine.run_to_completion() {
            Ok(responses) => responses,
            Err(error) => {
                self.engine.abort_all();
                return Err(error);
            }
        };
        let mut map: std::collections::HashMap<_, _> =
            responses.into_iter().map(|r| (r.id.clone(), r)).collect();
        order
            .iter()
            .map(|id| {
                map.remove(id)
                    .ok_or_else(|| NativeError("missing generation result".into()))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Directory(PathBuf);
    impl Directory {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "lighter-tokenizer-{}-{}",
                std::process::id(),
                rand::random::<u64>()
            ));
            fs::create_dir(&path).unwrap();
            fs::write(
                path.join("config.json"),
                r#"{"model_type":"llama","pad_token_id":0}"#,
            )
            .unwrap();
            fs::write(path.join("tokenizer.json"),r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":0,"content":"<pad>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<pad>":0,"hello":1,"world":2,"<unk>":3},"unk_token":"<unk>"}}"#).unwrap();
            Self(path)
        }
    }
    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    #[test]
    fn tokenization_padding_masks_truncation_and_roundtrip() {
        let directory = Directory::new();
        let tokenizer =
            AutoTokenizer::from_pretrained(directory.0.to_str().unwrap(), &LoadOptions::default())
                .unwrap();
        let texts = vec!["hello world".into(), "world".into()];
        let options = TokenizationOptions {
            padding: Padding::Longest,
            padding_side: Side::Left,
            ..Default::default()
        };
        let encoded = tokenizer.encode_batch(&texts, &options).unwrap();
        assert_eq!(encoded.input_ids, vec![vec![1, 2], vec![0, 2]]);
        assert_eq!(encoded.attention_mask, vec![vec![1, 1], vec![0, 1]]);
        assert_eq!(encoded.token_type_ids, vec![vec![0, 0], vec![0, 0]]);
        let options = TokenizationOptions {
            max_length: Some(1),
            truncation: true,
            truncation_side: Side::Left,
            ..Default::default()
        };
        assert_eq!(
            tokenizer.encode_batch(&texts, &options).unwrap().input_ids,
            vec![vec![2], vec![2]]
        );
        let options = TokenizationOptions {
            padding: Padding::MaxLength(3),
            ..Default::default()
        };
        assert_eq!(
            tokenizer.encode_batch(&texts, &options).unwrap().input_ids[0],
            vec![1, 2, 0]
        );
        let target = directory.0.join("saved");
        tokenizer.save_pretrained(&target).unwrap();
        let reloaded = AutoTokenizer::from_file(target.join("tokenizer.json")).unwrap();
        assert_eq!(reloaded.pad_token_id, Some(0));
        assert_eq!(
            reloaded.batch_decode(&encoded.input_ids, true).unwrap(),
            vec!["hello world", "world"]
        );
        assert!(tokenizer
            .encode_batch(
                &texts,
                &TokenizationOptions {
                    max_length: Some(1),
                    ..Default::default()
                }
            )
            .is_err());
        assert!(tokenizer
            .encode_batch(
                &texts,
                &TokenizationOptions {
                    truncation: true,
                    ..Default::default()
                }
            )
            .is_err());
        assert!(tokenizer
            .encode_batch(
                &texts,
                &TokenizationOptions {
                    padding: Padding::MaxLength(1),
                    ..Default::default()
                }
            )
            .is_err());
        assert!(tokenizer
            .encode_batch(&[], &TokenizationOptions::default())
            .unwrap()
            .input_ids
            .is_empty());
    }
    #[test]
    fn metadata_loading_needs_no_weight_files() {
        let directory = Directory::new();
        let config =
            AutoConfig::from_pretrained(directory.0.to_str().unwrap(), &LoadOptions::default())
                .unwrap();
        assert_eq!(config.model_type, "llama");
        let saved = directory.0.join("saved-config");
        config.save_pretrained(&saved).unwrap();
        let restored =
            AutoConfig::from_pretrained(saved.to_str().unwrap(), &LoadOptions::default()).unwrap();
        assert_eq!(
            restored.extensions.get("pad_token_id"),
            Some(&serde_json::json!(0))
        );
        fs::write(
            directory.0.join("config.json"),
            r#"{"model_type":"t5","architectures":["T5ForConditionalGeneration"]}"#,
        )
        .unwrap();
        let error = AutoModelForCausalLM::from_pretrained(
            directory.0.to_str().unwrap(),
            &LoadOptions::default(),
        )
        .err()
        .unwrap();
        assert!(error.0.contains("no model weights were loaded"));
    }
    #[test]
    fn generation_lengths_greedy_sampling_and_unsupported_flags() {
        let mut c = GenerationConfig {
            max_new_tokens: Some(4),
            min_new_tokens: Some(2),
            ..Default::default()
        };
        let p = c.sampling(10, Some(14), 42).unwrap();
        assert_eq!(
            (p.max_tokens, p.min_tokens, p.temperature, p.seed),
            (4, 2, 0., 42)
        );
        assert!(c.sampling(10, Some(13), 42).is_err());
        c.max_new_tokens = None;
        c.max_length = Some(12);
        c.min_new_tokens = None;
        c.min_length = 11;
        assert_eq!(
            (
                c.sampling(10, None, 0).unwrap().max_tokens,
                c.sampling(10, None, 0).unwrap().min_tokens
            ),
            (2, 1)
        );
        assert!(c.sampling(12, None, 0).is_err());
        c = GenerationConfig {
            max_new_tokens: Some(4),
            do_sample: true,
            temperature: 0.5,
            top_k: 0,
            ..Default::default()
        };
        assert_eq!(c.sampling(1, None, 0).unwrap().temperature, 0.5);
        assert!(c.sampling(1, None, 0).unwrap().top_k.is_none());
        c.num_beams = 2;
        assert!(c.sampling(1, None, 0).is_err());
        c.num_beams = 1;
        c.extensions
            .insert("typical_p".into(), serde_json::json!(0.8));
        assert!(c.sampling(1, None, 0).is_err());
    }
    #[test]
    fn generation_config_roundtrip_and_eos_override() {
        let nullable: GenerationConfig =
            serde_json::from_str(r#"{"bad_words_ids":null,"stop_strings":null}"#).unwrap();
        assert!(nullable.bad_words_ids.is_empty() && nullable.stop_strings.is_empty());
        let directory = Directory::new();
        let config = GenerationConfig {
            max_new_tokens: Some(3),
            eos_token_id: Some(TokenIds::Many(vec![2, 3])),
            ..Default::default()
        };
        config.save_pretrained(&directory.0).unwrap();
        let restored = GenerationConfig::from_pretrained(
            directory.0.to_str().unwrap(),
            &LoadOptions::default(),
        )
        .unwrap();
        let p = restored.sampling(1, None, 0).unwrap();
        assert!(p.ignore_eos);
        assert_eq!(p.stop_token_ids, vec![2, 3]);
    }
    struct Backend {
        fail: bool,
    }
    impl NativeBackend for Backend {
        fn encode(&self, prompt: &str) -> NativeResult<Vec<u32>> {
            Ok(prompt.bytes().map(|n| n as u32).collect())
        }
        fn decode(&self, ids: &[u32]) -> NativeResult<String> {
            Ok(ids.iter().map(|_| "x").collect())
        }
        fn logits(&mut self, _: &str, _: &[u32], _: usize) -> NativeResult<Vec<f32>> {
            if self.fail {
                self.fail = false;
                return Err(NativeError("injected failure".into()));
            }
            Ok(vec![0., 10.])
        }
    }
    #[test]
    fn pipeline_orders_unequal_completion_lengths_and_recovers_after_failure() {
        let c = GenerationConfig {
            max_new_tokens: None,
            max_length: Some(5),
            ..Default::default()
        };
        let mut pipeline =
            TextGenerationPipeline::new(Backend { fail: false }, vec![], Some(8), 2, c).unwrap();
        let responses = pipeline.generate(&["a".into(), "abcd".into()], 42).unwrap();
        assert_eq!(
            responses
                .iter()
                .map(|r| r.text.as_str())
                .collect::<Vec<_>>(),
            vec!["xxxx", "x"]
        );
        assert!(pipeline.generate(&["abcdefgh".into()], 42).is_err());
        pipeline.engine.backend_mut().fail = true;
        assert!(pipeline.generate(&["a".into()], 42).is_err());
        assert!(pipeline.engine.is_idle());
        assert_eq!(
            pipeline.generate(&["a".into()], 42).unwrap()[0].text,
            "xxxx"
        );
        assert!(pipeline.generate(&[], 42).unwrap().is_empty());
    }
    #[test]
    #[ignore = "requires public Hugging Face network access; downloads metadata/tokenizer only"]
    fn public_hub_metadata_without_model_weights() {
        let source = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";
        let options = LoadOptions::default();
        assert_eq!(
            AutoConfig::from_pretrained(source, &options)
                .unwrap()
                .model_type,
            "llama"
        );
        let tokenizer = AutoTokenizer::from_pretrained(source, &options).unwrap();
        let encoded = tokenizer
            .encode_batch(&["hello world".into()], &TokenizationOptions::default())
            .unwrap();
        assert!(!encoded.input_ids[0].is_empty());
    }
}
