//! The factory creates and drops the delegate on the HTTP host's decoding thread.
use candlelighter::{
    arm_npu::{TfliteNpuBackend, TfliteNpuConfig},
    model_host::{router_with_backend_factory, HostConfig},
    native::NativeError,
};
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 5 {
        return Err(
            "usage: arm_npu_host CONFIG.json TOKENIZER.json EOS_TOKEN_ID CONTEXT_LENGTH ADDRESS"
                .into(),
        );
    }
    let config: TfliteNpuConfig = serde_json::from_slice(&std::fs::read(&args[0])?)?;
    let tokenizer = args[1].clone();
    let eos = args[2].parse()?;
    let context: usize = args[3].parse()?;
    if context < 2 {
        return Err("context must be at least 2".into());
    }
    let host = HostConfig {
        model_name: "arm-npu".into(),
        max_context_tokens: Some(context),
        max_input_tokens: context - 1,
        max_output_tokens: context - 1,
        api_key: std::env::var("LIGHTER_API_KEY").ok(),
        ..Default::default()
    };
    let app = router_with_backend_factory(
        move || {
            let backend = TfliteNpuBackend::from_files(config, tokenizer)?;
            if backend.report().context_length != context {
                return Err(NativeError(
                    "configured context differs from compiled model".into(),
                ));
            }
            eprintln!(
                "{}",
                serde_json::to_string_pretty(backend.report())
                    .map_err(|e| NativeError(e.to_string()))?
            );
            Ok(backend)
        },
        vec![eos],
        host,
    )?;
    let listener = tokio::net::TcpListener::bind(&args[4]).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}
