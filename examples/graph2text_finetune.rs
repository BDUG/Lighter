//! Native Graph2Text SFT: consistent triple prompts, answer-only labels, shuffled
//! graph order, frozen base + output-head LoRA, holdout loss and adapter persistence.
use candlelighter::{
    graph_training::*,
    native::*,
    native_advanced::{QuantizationConfig, QuantizationType},
    native_training::*,
    NativeLlama,
};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use std::path::Path;

fn dataset(path: &str) -> Result<Vec<InstructionExample>, Box<dyn std::error::Error>> {
    let data: Vec<InstructionExample> = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    if data.is_empty() {
        return Err("dataset must contain at least one record".into());
    }
    for example in &data {
        example.triples()?;
    }
    Ok(data)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(4..=6).contains(&args.len()) {
        return Err("usage: graph2text_finetune MODEL_DIR TRAIN_JSON VALIDATION_JSON ADAPTER_JSON [EPOCHS=3; 0=reload/evaluate] [none|int4|int8]".into());
    }
    let epochs: usize = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(3);
    let quantization = match args.get(5).map(String::as_str).unwrap_or("none") {
        "none" => None,
        "int4" => Some(QuantizationType::Int4),
        "int8" => Some(QuantizationType::Int8),
        _ => return Err("quantization must be none, int4 or int8".into()),
    };
    if epochs > 0 && Path::new(&args[3]).exists() {
        return Err(
            "adapter output already exists; choose a new path or use EPOCHS=0 to reload".into(),
        );
    }
    let train = dataset(&args[1])?;
    let validation = dataset(&args[2])?;
    let artifacts = HuggingFaceArtifacts::from_dir(&args[0])?;
    let tokenizer = tokenizers::Tokenizer::from_file(&artifacts.tokenizer)
        .map_err(|e| NativeError(e.to_string()))?;
    let eos_ids = artifacts.config.eos_token_ids();
    let eos = eos_ids.first().copied();
    let context = artifacts.config.max_position_embeddings.unwrap_or(2048);
    let mut model = match quantization {
        Some(dtype) => NativeLlama::load_quantized(
            &artifacts,
            QuantizationConfig {
                dtype,
                group_size: 64,
            },
        )?,
        None => NativeLlama::load(&artifacts)?,
    };
    if epochs == 0 {
        let saved: HeadAdapterCheckpoint =
            serde_json::from_str(&std::fs::read_to_string(&args[3])?)?;
        if saved.base_model != args[0] {
            return Err("adapter base-model path differs; use its original base model".into());
        }
        model.set_lm_head_lora(saved.into_adapter()?)?;
    } else {
        model.enable_lm_head_lora(
            LoraConfig {
                rank: 8,
                alpha: 16.0,
                dropout: 0.0,
            },
            42,
        )?;
    }
    let mut rng = StdRng::seed_from_u64(42);
    let mut order: Vec<_> = (0..train.len()).collect();
    for epoch in 0..epochs {
        order.shuffle(&mut rng);
        let mut loss = 0.0;
        for &index in &order {
            let row = &train[index];
            let prompt = row.prompt(true, &mut rng)?;
            let tokens = answer_only_tokens(&tokenizer, &prompt, &row.output, eos, context)?;
            loss += supervised_fine_tune_step(
                &mut model,
                &tokens.input_ids,
                &tokens.labels,
                -100,
                0.0,
                1e-4,
            )?;
        }
        println!(
            "epoch={} mean_train_loss={:.6}",
            epoch + 1,
            loss / train.len() as f32
        );
    }
    if epochs > 0 {
        let checkpoint = HeadAdapterCheckpoint::from_adapter(
            args[0].clone(),
            model.lm_head_lora().ok_or("missing trained adapter")?,
        );
        let mut output = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args[3])?;
        serde_json::to_writer_pretty(&mut output, &checkpoint)?;
        println!("saved_adapter={}", args[3]);
    }
    let mut eval_loss = 0.0;
    for row in &validation {
        let prompt = row.prompt(false, &mut rng)?;
        let tokens = answer_only_tokens(&tokenizer, &prompt, &row.output, eos, context)?;
        let logits = model.token_logits(&tokens.input_ids)?;
        eval_loss += causal_lm_loss(&logits, &tokens.labels, -100, 0.0)?.loss;
    }
    println!(
        "mean_validation_loss={:.6}",
        eval_loss / validation.len() as f32
    );
    let backend = HuggingFaceBackend::new(model, &artifacts)?;
    let mut engine = NativeEngine::new(backend, 1, eos_ids)?;
    for (i, row) in validation.iter().enumerate() {
        let prompt = row.prompt(false, &mut rng)?;
        let prompt_len = tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| NativeError(e.to_string()))?
            .len();
        let available = context
            .checked_sub(prompt_len)
            .filter(|&n| n > 0)
            .ok_or("validation prompt fills model context")?;
        engine.submit(GenerateRequest {
            id: format!("validation-{i}"),
            prompt,
            sampling: SamplingParams {
                max_tokens: available.min(128),
                temperature: 0.0,
                ..Default::default()
            },
            constraint: None,
        })?;
        let response = engine.run_to_completion()?.remove(0);
        println!(
            "{}",
            serde_json::json!({"input":row.input,"reference":row.output,"prediction":response.text,"entity_mention_recall":entity_mention_recall(&row.triples()?,&response.text),"finish_reason":format!("{:?}",response.finish_reason)})
        );
    }
    Ok(())
}
