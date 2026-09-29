//! Turn-key support for the `Contrastive-LM/CLM-v0.1-8B` checkpoint.

use crate::native::{
    GenerateRequest, GenerateResponse, HuggingFaceArtifacts, HuggingFaceBackend, HuggingFaceConfig,
    NativeEngine, NativeError, NativeResult, SamplingParams,
};
use crate::native_advanced::QuantizationConfig;
use crate::NativeLlama;
use std::path::Path;

/// The canonical Hugging Face repository used by [`ContrastiveLm::from_hub`].
pub const CLM_V01_8B_MODEL_ID: &str = "Contrastive-LM/CLM-v0.1-8B";

/// A loaded CLM v0.1 8B model, tokenizer, KV cache, and generation scheduler.
///
/// Unlike the generic native building blocks, this type verifies that the
/// downloaded checkpoint has the Llama causal-decoder layout expected by CLM
/// before attempting to allocate its weights.
pub struct ContrastiveLm {
    engine: NativeEngine<HuggingFaceBackend<NativeLlama>>,
    next_request: u64,
}

impl ContrastiveLm {
    /// Downloads the official checkpoint (or reuses the Hugging Face cache)
    /// and loads it. `token` is only needed when Hub access requires auth.
    pub fn from_hub(token: Option<String>) -> NativeResult<Self> {
        let artifacts = HuggingFaceArtifacts::from_hub(CLM_V01_8B_MODEL_ID, None, token)?;
        Self::from_artifacts(artifacts)
    }

    /// Loads a previously downloaded CLM snapshot without network access.
    pub fn from_dir(path: impl AsRef<Path>) -> NativeResult<Self> {
        Self::from_artifacts(HuggingFaceArtifacts::from_dir(path)?)
    }

    /// Constructs the runner from already resolved Hugging Face artifacts.
    pub fn from_artifacts(artifacts: HuggingFaceArtifacts) -> NativeResult<Self> {
        Self::from_artifacts_with_quantization(artifacts, None)
    }

    /// Constructs the runner and optionally quantizes matrix weights before
    /// the first generation. Int8 approximately halves the resident weight
    /// memory of an F16/BF16 checkpoint; packed Int4 approximately quarters it.
    pub fn from_artifacts_with_quantization(
        artifacts: HuggingFaceArtifacts,
        quantization: Option<QuantizationConfig>,
    ) -> NativeResult<Self> {
        validate_clm_config(&artifacts.config)?;
        let eos = artifacts.config.eos_token_ids();
        let model = match quantization {
            Some(config) => NativeLlama::load_quantized(&artifacts, config)?,
            None => NativeLlama::load(&artifacts)?,
        };
        let backend = HuggingFaceBackend::new(model, &artifacts)?;
        Ok(Self {
            engine: NativeEngine::new(backend, 1, eos)?,
            next_request: 0,
        })
    }

    /// Downloads and loads the official checkpoint, then quantizes it before
    /// generation. This is the convenient lower-resident-memory Hub path.
    pub fn from_hub_quantized(
        token: Option<String>,
        quantization: QuantizationConfig,
    ) -> NativeResult<Self> {
        let artifacts = HuggingFaceArtifacts::from_hub(CLM_V01_8B_MODEL_ID, None, token)?;
        Self::from_artifacts_with_quantization(artifacts, Some(quantization))
    }

    /// Loads and quantizes an offline snapshot.
    pub fn from_dir_quantized(
        path: impl AsRef<Path>,
        quantization: QuantizationConfig,
    ) -> NativeResult<Self> {
        Self::from_artifacts_with_quantization(
            HuggingFaceArtifacts::from_dir(path)?,
            Some(quantization),
        )
    }

    /// Generates one completion while retaining the loaded weights for later
    /// calls. Sampling validation and EOS handling are delegated to the native
    /// engine.
    pub fn generate(
        &mut self,
        prompt: impl Into<String>,
        sampling: SamplingParams,
    ) -> NativeResult<GenerateResponse> {
        let prompt = prompt.into();
        if prompt.trim().is_empty() {
            return Err(NativeError("CLM prompt cannot be empty".into()));
        }
        let id = format!("clm-{}", self.next_request);
        self.next_request = self.next_request.wrapping_add(1);
        self.engine.submit(GenerateRequest {
            id,
            prompt,
            sampling,
            constraint: None,
        })?;
        self.engine
            .run_to_completion()?
            .into_iter()
            .next()
            .ok_or_else(|| NativeError("CLM generation returned no response".into()))
    }
}

fn validate_clm_config(config: &HuggingFaceConfig) -> NativeResult<()> {
    let llama_architecture = config.architectures.iter().any(|architecture| {
        matches!(
            architecture.as_str(),
            "LlamaForCausalLM" | "Qwen3ForCausalLM" | "CLMForCausalLM"
        )
    });
    let clm_model_type = matches!(
        config.model_type.as_str(),
        "llama" | "qwen3" | "clm" | "contrastive_lm"
    );
    if !clm_model_type && !llama_architecture {
        return Err(NativeError(format!(
            "CLM v0.1 8B requires a LlamaForCausalLM checkpoint, found model_type {:?} and architectures {:?}",
            config.model_type, config.architectures
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize_to_file, TensorView};
    use safetensors::Dtype;
    use std::collections::BTreeMap;
    use std::fs;

    #[test]
    fn accepts_clm_llama_config() {
        let config = HuggingFaceConfig::from_json(
            r#"{"model_type":"llama","architectures":["LlamaForCausalLM"]}"#,
        )
        .unwrap();
        validate_clm_config(&config).unwrap();

        let custom_config = HuggingFaceConfig::from_json(
            r#"{"model_type":"clm","architectures":["CLMForCausalLM"]}"#,
        )
        .unwrap();
        validate_clm_config(&custom_config).unwrap();

        let qwen3_config = HuggingFaceConfig::from_json(
            r#"{"model_type":"qwen3","architectures":["Qwen3ForCausalLM"]}"#,
        )
        .unwrap();
        validate_clm_config(&qwen3_config).unwrap();
    }

    #[test]
    fn rejects_an_incompatible_checkpoint_before_loading_weights() {
        let config = HuggingFaceConfig::from_json(
            r#"{"model_type":"gpt2","architectures":["GPT2LMHeadModel"]}"#,
        )
        .unwrap();
        let error = validate_clm_config(&config).unwrap_err();
        assert!(error.0.contains("LlamaForCausalLM"));
    }

    #[test]
    fn loads_a_snapshot_and_generates_end_to_end() {
        let root = std::env::temp_dir().join(format!(
            "lighter-clm-smoke-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("config.json"),
            r#"{
                "model_type":"clm","architectures":["CLMForCausalLM"],
                "vocab_size":2,"hidden_size":2,"intermediate_size":2,
                "num_hidden_layers":1,"num_attention_heads":1,
                "num_key_value_heads":1,"head_dim":2,"rms_norm_eps":0.000001
            }"#,
        )
        .unwrap();
        fs::write(
            root.join("tokenizer.json"),
            r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{"a":0,"b":1},"merges":[]}}"#,
        )
        .unwrap();

        let mut tensors = BTreeMap::new();
        let mut add = |name: &str, shape: Vec<usize>| {
            tensors.insert(
                name.to_owned(),
                (shape.clone(), vec![0; shape.iter().product::<usize>() * 2]),
            );
        };
        add("model.embed_tokens.weight", vec![2, 2]);
        add("model.layers.0.input_layernorm.weight", vec![2]);
        add("model.layers.0.post_attention_layernorm.weight", vec![2]);
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            add(
                &format!("model.layers.0.self_attn.{name}.weight"),
                vec![2, 2],
            );
        }
        for name in ["gate_proj", "up_proj", "down_proj"] {
            add(&format!("model.layers.0.mlp.{name}.weight"), vec![2, 2]);
        }
        add("model.norm.weight", vec![2]);
        add("lm_head.weight", vec![2, 2]);
        let views = tensors
            .iter()
            .map(|(name, (shape, data))| {
                (
                    name,
                    TensorView::new(Dtype::F16, shape.clone(), data).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(views, &None, &root.join("model.safetensors")).unwrap();

        let mut model = ContrastiveLm::from_dir(&root).unwrap();
        let response = model
            .generate(
                "a",
                SamplingParams {
                    max_tokens: 1,
                    temperature: 0.0,
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(response.prompt_tokens, 1);
        assert_eq!(response.token_ids.len(), 1);
        fs::remove_dir_all(root).unwrap();
    }
}
