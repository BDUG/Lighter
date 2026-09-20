//! Generate text from a local Hugging Face snapshot without Candle.
//!
//! Run with:
//! `cargo run --example native_generate --no-default-features --features native -- MODEL_DIR "prompt"`

#[cfg(feature = "native")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use candlelighter::native::{
        GenerateRequest, HuggingFaceArtifacts, HuggingFaceBackend, NativeEngine, SamplingParams,
    };
    use candlelighter::NativeLlama;

    let mut args = std::env::args().skip(1);
    let directory = args
        .next()
        .ok_or("usage: native_generate MODEL_DIR [PROMPT]")?;
    let prompt = args.next().unwrap_or_else(|| "Once upon a time".into());
    let artifacts = HuggingFaceArtifacts::from_dir(directory)?;
    let eos = artifacts.config.eos_token_ids();
    let model = NativeLlama::load(&artifacts)?;
    let backend = HuggingFaceBackend::new(model, &artifacts)?;
    let mut engine = NativeEngine::new(backend, 1, eos)?;
    engine.submit(GenerateRequest {
        id: "example".into(),
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
    let response = engine.run_to_completion()?.remove(0);
    println!("{}", response.text);
    eprintln!(
        "generated {} tokens ({:?})",
        response.token_ids.len(),
        response.finish_reason
    );
    Ok(())
}

#[cfg(not(feature = "native"))]
fn main() {
    eprintln!("enable this example with --no-default-features --features native");
}
