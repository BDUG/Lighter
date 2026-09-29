//! Download and run Contrastive-LM/CLM-v0.1-8B with the native runtime.
//!
//! Run with:
//! `cargo run --release --example clm_generate --no-default-features --features native -- "Your prompt"`

#[cfg(feature = "native")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use candlelighter::native::SamplingParams;
    use candlelighter::native_advanced::{QuantizationConfig, QuantizationType};
    use candlelighter::{ContrastiveLm, CLM_V01_8B_MODEL_ID};

    let prompt = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let prompt = if prompt.is_empty() {
        "Explain contrastive language modeling in a few sentences.".to_owned()
    } else {
        prompt
    };
    let token = std::env::var("HF_TOKEN").ok();

    let quantization = match std::env::var("CLM_QUANTIZATION").as_deref() {
        Ok("int8") => Some(QuantizationType::Int8),
        Ok("int4") => Some(QuantizationType::Int4),
        Ok(value) => {
            return Err(format!("CLM_QUANTIZATION must be int8 or int4, got {value:?}").into())
        }
        Err(_) => None,
    }
    .map(|dtype| QuantizationConfig {
        dtype,
        group_size: 64,
    });

    let mut model = if let Some(directory) = std::env::var_os("CLM_MODEL_DIR") {
        eprintln!("loading {}...", std::path::Path::new(&directory).display());
        match quantization {
            Some(config) => ContrastiveLm::from_dir_quantized(directory, config)?,
            None => ContrastiveLm::from_dir(directory)?,
        }
    } else {
        eprintln!("downloading/loading {CLM_V01_8B_MODEL_ID}...");
        match quantization {
            Some(config) => ContrastiveLm::from_hub_quantized(token, config)?,
            None => ContrastiveLm::from_hub(token)?,
        }
    };
    let response = model.generate(
        prompt,
        SamplingParams {
            max_tokens: 128,
            temperature: 0.7,
            top_p: 0.9,
            seed: 42,
            ..Default::default()
        },
    )?;
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
