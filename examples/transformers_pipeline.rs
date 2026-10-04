//! cargo run --release --no-default-features --features native --example transformers_pipeline -- MODEL_DIR_OR_HUB_ID [PROMPT ...]
use candlelighter::transformers::*;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let source = args
        .next()
        .ok_or("usage: transformers_pipeline MODEL_DIR_OR_HUB_ID [PROMPT ...]")?;
    let mut prompts: Vec<_> = args.collect();
    if prompts.is_empty() {
        prompts = vec![
            "Explain Rust ownership:".into(),
            "Explain graph retrieval:".into(),
        ];
    }
    let options = LoadOptions {
        token: std::env::var("HF_TOKEN").ok(),
        ..Default::default()
    };
    let config = AutoConfig::from_pretrained(&source, &options)?;
    eprintln!(
        "model_type={} context={:?}",
        config.model_type, config.max_position_embeddings
    );
    let tokenizer = AutoTokenizer::from_pretrained(&source, &options)?;
    let tokens = tokenizer.encode_batch(&prompts, &TokenizationOptions::default())?;
    eprintln!(
        "prompt_lengths={:?}",
        tokens.input_ids.iter().map(Vec::len).collect::<Vec<_>>()
    );
    let mut pipeline = TextGenerationPipeline::from_pretrained(&source, &options, 4)?;
    pipeline.generation_config.max_new_tokens = Some(64);
    pipeline.generation_config.do_sample = false;
    pipeline.generation_config.num_beams = 1;
    pipeline.generation_config.num_return_sequences = 1;
    pipeline.generation_config.no_repeat_ngram_size = 3;
    for (prompt, response) in prompts.iter().zip(pipeline.generate(&prompts, 42)?) {
        println!(
            "{}",
            serde_json::json!({"prompt": prompt, "generated_text": response.text, "prompt_tokens": response.prompt_tokens, "generated_tokens": response.token_ids.len(), "finish_reason": format!("{:?}", response.finish_reason)})
        );
    }
    Ok(())
}
