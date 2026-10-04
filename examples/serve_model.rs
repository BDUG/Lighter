//! Embed the HTTP host in a Rust application using a local snapshot.
#[cfg(feature = "server")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    use candlelighter::model_host::{serve, HostConfig};
    use candlelighter::native::{HuggingFaceArtifacts, HuggingFaceBackend};
    use candlelighter::NativeLlama;
    let directory = std::env::args()
        .nth(1)
        .ok_or("usage: serve_model MODEL_DIR")?;
    let artifacts = HuggingFaceArtifacts::from_dir(directory)?;
    let model = NativeLlama::load(&artifacts)?;
    let backend = HuggingFaceBackend::new(model, &artifacts)?;
    let config = HostConfig {
        model_name: "local-model".into(),
        max_context_tokens: artifacts.config.max_position_embeddings,
        api_key: std::env::var("LIGHTER_API_KEY").ok(),
        // Set chat_template only after matching the snapshot's training format.
        ..Default::default()
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:8000").await?;
    eprintln!("hosting local-model at {}", listener.local_addr()?);
    serve(listener, backend, artifacts.config.eos_token_ids(), config).await?;
    Ok(())
}
#[cfg(not(feature = "server"))]
fn main() {
    eprintln!("enable this example with --no-default-features --features server");
}
