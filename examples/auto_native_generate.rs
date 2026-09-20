//! Select, download, and run a compatible Hugging Face model without Candle.
//!
//! This example downloads model weights. Use a narrow search term for a small
//! checkpoint, for example `TinyLlama`.

#[cfg(feature = "native")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use candlelighter::native::{
        GenerateRequest, HuggingFaceArtifacts, HuggingFaceBackend, NativeEngine, SamplingParams,
    };
    use candlelighter::native_advanced::{QuantizationConfig, QuantizationType};
    use candlelighter::{ModelSelector, NativeLlama, SelectionPolicy};

    let mut args = std::env::args().skip(1);
    let search = args.next().unwrap_or_else(|| "TinyLlama".into());
    let prompt = args
        .next()
        .unwrap_or_else(|| "Explain why Rust is safe:".into());
    let token = std::env::var("HF_TOKEN").ok();
    let selector = ModelSelector::new(token.clone());
    let policy = SelectionPolicy::default();
    let selected = selector
        .select_from_hub(Some(&search), 25, &policy)?
        .ok_or("no compatible Hugging Face model fits this system")?;
    eprintln!(
        "selected {} for {:#?}",
        selected.id,
        selector.capabilities()
    );

    let artifacts = HuggingFaceArtifacts::from_hub(&selected.id, None, token)?;
    let eos = artifacts.config.eos_token_ids();
    let mut model = NativeLlama::load(&artifacts)?;
    model.quantize(QuantizationConfig {
        dtype: QuantizationType::Int8,
        group_size: 64,
    })?;
    let backend = HuggingFaceBackend::new(model, &artifacts)?;
    let mut engine = NativeEngine::new(backend, 1, eos)?;
    engine.submit(GenerateRequest {
        id: "automatic-example".into(),
        prompt,
        constraint: None,
        sampling: SamplingParams {
            max_tokens: 64,
            temperature: 0.7,
            top_p: 0.9,
            seed: 42,
            ..Default::default()
        },
    })?;
    println!("{}", engine.run_to_completion()?.remove(0).text);
    Ok(())
}

#[cfg(not(feature = "native"))]
fn main() {
    eprintln!("enable this example with --no-default-features --features native");
}
