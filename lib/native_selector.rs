//! Host capability detection and Hugging Face text-generation model selection.

use crate::native::{HuggingFaceConfig, NativeError, NativeResult};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Accelerator {
    NvidiaCuda,
    AmdRocm,
    AppleMetal,
}

/// Capabilities relevant to choosing an inference backend and model size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemCapabilities {
    pub operating_system: String,
    pub architecture: String,
    pub logical_cpus: usize,
    pub memory_bytes: Option<u64>,
    /// Memory currently available without swapping, when the OS exposes it.
    pub available_memory_bytes: Option<u64>,
    /// Largest detected accelerator memory. This is advisory: the selected
    /// execution backend still determines whether that accelerator is usable.
    pub accelerator_memory_bytes: Option<u64>,
    pub accelerators: Vec<Accelerator>,
    pub avx2: bool,
    pub neon: bool,
}

impl SystemCapabilities {
    pub fn detect() -> Self {
        Self {
            operating_system: std::env::consts::OS.into(),
            architecture: std::env::consts::ARCH.into(),
            logical_cpus: std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1),
            memory_bytes: detect_memory("MemTotal:"),
            available_memory_bytes: detect_memory("MemAvailable:"),
            accelerators: detect_accelerators(),
            accelerator_memory_bytes: detect_accelerator_memory(),
            avx2: detect_avx2(),
            neon: detect_neon(),
        }
    }

    pub fn has_gpu(&self) -> bool {
        !self.accelerators.is_empty()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceModel {
    pub id: String,
    pub downloads: u64,
    pub likes: u64,
    pub model_type: Option<String>,
    pub parameter_count: Option<u64>,
    pub gated: bool,
    pub private: bool,
    pub tags: Vec<String>,
}

impl HuggingFaceModel {
    pub fn estimated_runtime_bytes(&self, bytes_per_parameter: u64) -> Option<u64> {
        // Account for weights plus a conservative 25% for activations and KV cache.
        self.parameter_count?
            .checked_mul(bytes_per_parameter)?
            .checked_mul(5)?
            .checked_div(4)
    }

    pub fn is_native_llama_compatible(&self) -> bool {
        matches!(
            self.model_type.as_deref(),
            Some("llama" | "mistral" | "qwen2")
        )
    }
}

#[derive(Debug, Clone)]
pub struct SelectionPolicy {
    pub memory_fraction: f32,
    pub bytes_per_parameter: u64,
    pub allow_gated: bool,
    pub allow_private: bool,
    pub require_native_llama: bool,
}

impl Default for SelectionPolicy {
    fn default() -> Self {
        Self {
            memory_fraction: 0.75,
            // NativeLlama materializes all supported checkpoint dtypes as f32.
            bytes_per_parameter: 4,
            allow_gated: false,
            allow_private: false,
            require_native_llama: true,
        }
    }
}

impl SelectionPolicy {
    pub fn validate(&self) -> NativeResult<()> {
        if !(0.0..=1.0).contains(&self.memory_fraction) || self.memory_fraction == 0.0 {
            return Err(NativeError("memory_fraction must be in (0, 1]".into()));
        }
        if self.bytes_per_parameter == 0 {
            return Err(NativeError("bytes_per_parameter must be positive".into()));
        }
        Ok(())
    }
}

/// Queries the public Hub API and ranks models that fit the current host.
pub struct ModelSelector {
    capabilities: SystemCapabilities,
    token: Option<String>,
    api_base: String,
}

impl ModelSelector {
    pub fn new(token: Option<String>) -> Self {
        Self {
            capabilities: SystemCapabilities::detect(),
            token,
            api_base: "https://huggingface.co".into(),
        }
    }

    pub fn with_capabilities(capabilities: SystemCapabilities, token: Option<String>) -> Self {
        Self {
            capabilities,
            token,
            api_base: "https://huggingface.co".into(),
        }
    }

    pub fn capabilities(&self) -> &SystemCapabilities {
        &self.capabilities
    }

    /// Returns popular Hub models tagged for text generation. `search` is
    /// matched by the Hub against model IDs, tags, and metadata.
    pub fn available_models(
        &self,
        search: Option<&str>,
        limit: usize,
    ) -> NativeResult<Vec<HuggingFaceModel>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let mut request = ureq::get(&format!("{}/api/models", self.api_base))
            .query("filter", "text-generation")
            .query("sort", "downloads")
            .query("direction", "-1")
            .query("limit", &limit.min(1000).to_string())
            .query("full", "true");
        if let Some(search) = search.filter(|value| !value.is_empty()) {
            request = request.query("search", search);
        }
        let response = self.authorize(request).call().map_err(http_error)?;
        let raw: Vec<HubModel> = response
            .into_json()
            .map_err(|e| NativeError(format!("invalid Hugging Face models response: {e}")))?;
        raw.into_iter().map(HubModel::into_model).collect()
    }

    /// Discovers models and fills missing architecture metadata by fetching
    /// only their small `config.json` files. Individual inaccessible or stale
    /// search results are skipped rather than failing the whole discovery run.
    pub fn available_compatible_models(
        &self,
        search: Option<&str>,
        limit: usize,
        policy: &SelectionPolicy,
    ) -> NativeResult<Vec<HuggingFaceModel>> {
        policy.validate()?;
        let mut models = self.available_models(search, limit)?;
        for model in &mut models {
            if model.model_type.is_none() {
                if let Ok(config) = self.model_config(&model.id, None) {
                    model.model_type = Some(config.model_type);
                }
            }
        }
        models.retain(|model| {
            (policy.allow_gated || !model.gated)
                && (policy.allow_private || !model.private)
                && (!policy.require_native_llama || model.is_native_llama_compatible())
        });
        Ok(models)
    }

    /// Complete discovery-and-selection operation used by applications that do
    /// not need to inspect the intermediate Hub result list.
    pub fn select_from_hub(
        &self,
        search: Option<&str>,
        limit: usize,
        policy: &SelectionPolicy,
    ) -> NativeResult<Option<HuggingFaceModel>> {
        let models = self.available_compatible_models(search, limit, policy)?;
        Ok(self.select(&models, policy)?.cloned())
    }

    /// Downloads only `config.json`, useful for filling metadata omitted by a
    /// search result before committing to a multi-gigabyte checkpoint download.
    pub fn model_config(
        &self,
        model_id: &str,
        revision: Option<&str>,
    ) -> NativeResult<HuggingFaceConfig> {
        validate_model_id(model_id)?;
        let revision = revision.unwrap_or("main");
        let url = format!(
            "{}/{}/resolve/{}/config.json",
            self.api_base, model_id, revision
        );
        let response = self.authorize(ureq::get(&url)).call().map_err(http_error)?;
        let value: Value = response
            .into_json()
            .map_err(|e| NativeError(format!("invalid config for {model_id}: {e}")))?;
        serde_json::from_value(value)
            .map_err(|e| NativeError(format!("invalid config for {model_id}: {e}")))
    }

    pub fn select<'a>(
        &self,
        models: &'a [HuggingFaceModel],
        policy: &SelectionPolicy,
    ) -> NativeResult<Option<&'a HuggingFaceModel>> {
        policy.validate()?;
        let budget = self
            .capabilities
            .available_memory_bytes
            .or(self.capabilities.memory_bytes)
            .map(|bytes| (bytes as f64 * policy.memory_fraction as f64) as u64);
        Ok(models
            .iter()
            .filter(|model| {
                (policy.allow_gated || !model.gated)
                    && (policy.allow_private || !model.private)
                    && (!policy.require_native_llama || model.is_native_llama_compatible())
                    && match (
                        budget,
                        model.estimated_runtime_bytes(policy.bytes_per_parameter),
                    ) {
                        (Some(budget), Some(required)) => required <= budget,
                        _ => true,
                    }
            })
            .max_by(|left, right| {
                left.parameter_count
                    .unwrap_or(0)
                    .cmp(&right.parameter_count.unwrap_or(0))
                    .then_with(|| left.downloads.cmp(&right.downloads))
            }))
    }

    fn authorize(&self, request: ureq::Request) -> ureq::Request {
        match &self.token {
            Some(token) => request.set("Authorization", &format!("Bearer {token}")),
            None => request,
        }
    }
}

#[derive(Deserialize)]
struct HubModel {
    // The Hub currently returns both `id` and `modelId` for some models. An
    // alias cannot be used here because serde treats the second key as a
    // duplicate field. Keep them separate and prefer the canonical `id`.
    #[serde(default)]
    id: Option<String>,
    #[serde(default, rename = "modelId")]
    model_id: Option<String>,
    #[serde(default)]
    downloads: u64,
    #[serde(default)]
    likes: u64,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    private: bool,
    #[serde(default)]
    gated: Value,
    #[serde(default)]
    config: HashMap<String, Value>,
    #[serde(default)]
    safetensors: Option<HubSafetensors>,
}

#[derive(Deserialize)]
struct HubSafetensors {
    #[serde(default)]
    total: Option<u64>,
    #[serde(default)]
    parameters: HashMap<String, u64>,
}

impl HubModel {
    fn into_model(self) -> NativeResult<HuggingFaceModel> {
        let id = self.id.or(self.model_id).ok_or_else(|| {
            NativeError("invalid Hugging Face model entry: missing `id` and `modelId`".into())
        })?;
        let parameter_count = self.safetensors.and_then(|metadata| {
            metadata.total.or_else(|| {
                let total: u64 = metadata.parameters.values().sum();
                (total > 0).then_some(total)
            })
        });
        let model_type = self
            .config
            .get("model_type")
            .and_then(Value::as_str)
            .map(str::to_owned);
        let gated = match self.gated {
            Value::Bool(value) => value,
            Value::String(value) => value != "false",
            _ => false,
        };
        Ok(HuggingFaceModel {
            id,
            downloads: self.downloads,
            likes: self.likes,
            model_type,
            parameter_count,
            gated,
            private: self.private,
            tags: self.tags,
        })
    }
}

fn validate_model_id(value: &str) -> NativeResult<()> {
    if value.is_empty()
        || value.contains("..")
        || value.chars().any(|c| c.is_control() || c.is_whitespace())
    {
        Err(NativeError("invalid Hugging Face model ID".into()))
    } else {
        Ok(())
    }
}

fn http_error(error: ureq::Error) -> NativeError {
    NativeError(format!("Hugging Face request failed: {error}"))
}

fn detect_memory(field: &str) -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let text = fs::read_to_string("/proc/meminfo").ok()?;
        let kb = text
            .lines()
            .find(|line| line.starts_with(field))?
            .split_whitespace()
            .nth(1)?
            .parse::<u64>()
            .ok()?;
        return kb.checked_mul(1024);
    }
    #[cfg(target_os = "macos")]
    {
        if field != "MemTotal:" {
            return None;
        }
        let output = Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        return String::from_utf8(output.stdout).ok()?.trim().parse().ok();
    }
    #[allow(unreachable_code)]
    None
}

fn detect_accelerator_memory() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let output = Command::new("nvidia-smi")
            .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        return String::from_utf8(output.stdout)
            .ok()?
            .lines()
            .filter_map(|line| line.trim().parse::<u64>().ok())
            .max()?
            .checked_mul(1024 * 1024);
    }
    #[allow(unreachable_code)]
    None
}

fn detect_accelerators() -> Vec<Accelerator> {
    let mut result = Vec::new();
    #[cfg(target_os = "macos")]
    if matches!(std::env::consts::ARCH, "aarch64" | "x86_64") {
        result.push(Accelerator::AppleMetal);
    }
    #[cfg(target_os = "linux")]
    {
        if std::path::Path::new("/dev/nvidiactl").exists()
            || command_succeeds("nvidia-smi", &["-L"])
        {
            result.push(Accelerator::NvidiaCuda);
        }
        if std::path::Path::new("/dev/kfd").exists() {
            result.push(Accelerator::AmdRocm);
        }
    }
    result
}

fn command_succeeds(command: &str, args: &[&str]) -> bool {
    Command::new(command)
        .args(args)
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn detect_avx2() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        return std::arch::is_x86_feature_detected!("avx2");
    }
    #[allow(unreachable_code)]
    false
}

fn detect_neon() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        return std::arch::is_aarch64_feature_detected!("neon");
    }
    #[cfg(target_arch = "arm")]
    {
        return std::arch::is_arm_feature_detected!("neon");
    }
    #[allow(unreachable_code)]
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selects_largest_compatible_model_that_fits() {
        let selector = ModelSelector::with_capabilities(
            SystemCapabilities {
                operating_system: "test".into(),
                architecture: "x86_64".into(),
                logical_cpus: 8,
                memory_bytes: Some(10_000),
                available_memory_bytes: Some(10_000),
                accelerator_memory_bytes: None,
                accelerators: vec![],
                avx2: true,
                neon: false,
            },
            None,
        );
        let model = |id: &str, kind: &str, parameters: u64| HuggingFaceModel {
            id: id.into(),
            downloads: 1,
            likes: 0,
            model_type: Some(kind.into()),
            parameter_count: Some(parameters),
            gated: false,
            private: false,
            tags: vec![],
        };
        let models = vec![
            model("small", "llama", 1_000),
            model("large", "llama", 5_000),
            model("unsupported", "gpt2", 2_000),
        ];
        let selected = selector
            .select(&models, &SelectionPolicy::default())
            .unwrap()
            .unwrap();
        assert_eq!(selected.id, "small");
    }

    #[test]
    fn parses_realistic_hub_metadata() {
        let raw: HubModel = serde_json::from_value(serde_json::json!({
            "id":"org/model", "downloads":42, "gated":"auto", "config":{"model_type":"qwen2"},
            "safetensors":{"parameters":{"BF16":123}}
        }))
        .unwrap();
        let model = raw.into_model().unwrap();
        assert_eq!(model.parameter_count, Some(123));
        assert!(model.gated);
        assert!(model.is_native_llama_compatible());
    }

    #[test]
    fn parses_hub_metadata_with_both_model_identifiers() {
        let raw: HubModel = serde_json::from_value(serde_json::json!({
            "id": "org/canonical", "modelId": "org/legacy"
        }))
        .unwrap();

        let model = raw.into_model().unwrap();

        assert_eq!(model.id, "org/canonical");
    }

    #[test]
    fn falls_back_to_legacy_hub_model_identifier() {
        let raw: HubModel = serde_json::from_value(serde_json::json!({
            "modelId": "org/legacy"
        }))
        .unwrap();

        let model = raw.into_model().unwrap();

        assert_eq!(model.id, "org/legacy");
    }
}
