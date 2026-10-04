//! JSON config points to a compiled .tflite model, runtime and external delegate.
use candlelighter::{
    arm_npu::{TfliteNpuBackend, TfliteNpuConfig},
    native::{GenerateRequest, NativeBackend, NativeEngine, SamplingParams},
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        return Err(
            "usage: arm_npu_generate CONFIG.json TOKENIZER.json EOS_TOKEN_ID PROMPT".into(),
        );
    }
    let config: TfliteNpuConfig = serde_json::from_slice(&std::fs::read(&args[0])?)?;
    let backend = TfliteNpuBackend::from_files(config, &args[1])?;
    eprintln!("{}", serde_json::to_string_pretty(backend.report())?);
    let prompt_len = backend.encode(&args[3])?.len();
    let remaining = backend
        .report()
        .context_length
        .checked_sub(prompt_len)
        .ok_or("prompt exceeds compiled context")?;
    if prompt_len == 0 || remaining == 0 {
        return Err("prompt must be nonempty and leave room for generation".into());
    }
    let mut engine = NativeEngine::new(backend, 1, vec![args[2].parse()?])?;
    engine.submit(GenerateRequest {
        id: "arm-npu".into(),
        prompt: args[3].clone(),
        sampling: SamplingParams {
            max_tokens: remaining.min(32),
            temperature: 0.,
            ..Default::default()
        },
        constraint: None,
    })?;
    for response in engine.run_to_completion()? {
        println!("{}", response.text);
    }
    Ok(())
}
