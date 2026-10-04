//! OpenAI-compatible single-model native CPU host.
use candlelighter::model_host::{serve, ChatTemplate, HostConfig};
use candlelighter::native::{HuggingFaceArtifacts, HuggingFaceBackend};
use candlelighter::native_advanced::{QuantizationConfig, QuantizationType};
use candlelighter::NativeLlama;
use clap::Parser;
use std::net::SocketAddr;

#[derive(Parser)]
#[command(
    name = "lighter-serve",
    about = "Host a native model through OpenAI-compatible HTTP endpoints"
)]
struct Args {
    /// Local snapshot directory or Hugging Face repository ID.
    #[arg(long)]
    model: String,
    /// Public API model name. Defaults to the model argument.
    #[arg(long)]
    served_model_name: Option<String>,
    #[arg(long, default_value = "127.0.0.1:8000")]
    bind: SocketAddr,
    /// Explicit chat formatting; chat endpoint is disabled unless specified.
    #[arg(long, value_parser = ["chatml", "llama3"])]
    chat_template: Option<String>,
    #[arg(long, value_parser = ["int8", "int4"])]
    quantization: Option<String>,
    #[arg(long, default_value_t = 4)]
    max_batch_size: usize,
    #[arg(long, default_value_t = 64)]
    max_pending_requests: usize,
    #[arg(long, default_value_t = 4096)]
    max_input_tokens: usize,
    #[arg(long, default_value_t = 512)]
    max_tokens: usize,
    #[arg(long, default_value_t = 300)]
    request_timeout_seconds: u64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let mut config = HostConfig {
        model_name: args.served_model_name.unwrap_or_else(|| args.model.clone()),
        batch_size: args.max_batch_size,
        max_pending_requests: args.max_pending_requests,
        max_input_tokens: args.max_input_tokens,
        max_output_tokens: args.max_tokens,
        request_timeout: std::time::Duration::from_secs(args.request_timeout_seconds),
        api_key: std::env::var("LIGHTER_API_KEY").ok(),
        chat_template: args.chat_template.map(|v| {
            if v == "chatml" {
                ChatTemplate::ChatMl
            } else {
                ChatTemplate::Llama3
            }
        }),
        ..Default::default()
    };
    config.validate()?;
    eprintln!("loading model...");
    let artifacts = if std::path::Path::new(&args.model).is_dir() {
        HuggingFaceArtifacts::from_dir(&args.model)?
    } else {
        HuggingFaceArtifacts::from_hub(&args.model, None, std::env::var("HF_TOKEN").ok())?
    };
    config.max_context_tokens = artifacts.config.max_position_embeddings;
    let eos = artifacts.config.eos_token_ids();
    let model = if let Some(mode) = args.quantization {
        NativeLlama::load_quantized(
            &artifacts,
            QuantizationConfig {
                dtype: if mode == "int8" {
                    QuantizationType::Int8
                } else {
                    QuantizationType::Int4
                },
                group_size: 64,
            },
        )?
    } else {
        NativeLlama::load(&artifacts)?
    };
    let backend = HuggingFaceBackend::new(model, &artifacts)?;
    let listener = tokio::net::TcpListener::bind(args.bind).await?;
    eprintln!("model loaded; listening on {}", listener.local_addr()?);
    serve(listener, backend, eos, config).await?;
    Ok(())
}
