//! cargo run --no-default-features --features native --example graph2text_native -- MODEL_DIR GRAPH_JSON
use candlelighter::{
    graph2text::{generate, Graph, Options},
    native::{HuggingFaceArtifacts, HuggingFaceBackend, NativeEngine, SamplingParams},
    NativeLlama,
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: graph2text_native MODEL_DIR GRAPH_JSON".into());
    }
    let graph = Graph::from_json(&std::fs::read_to_string(&args[1])?)?;
    let artifacts = HuggingFaceArtifacts::from_dir(&args[0])?;
    let eos = artifacts.config.eos_token_ids();
    let model = NativeLlama::load(&artifacts)?;
    let backend = HuggingFaceBackend::new(model, &artifacts)?;
    let mut engine = NativeEngine::new(backend, 1, eos)?;
    let (plan, response) = generate(
        &mut engine,
        &graph,
        &Options::default(),
        SamplingParams {
            max_tokens: 256,
            temperature: 0.0,
            ..Default::default()
        },
    )?;
    println!("{}", response.text);
    eprintln!(
        "Selected {} facts; omitted {}",
        plan.facts.len(),
        plan.omitted_fact_ids.len()
    );
    Ok(())
}
