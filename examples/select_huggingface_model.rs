//! Discover and optionally download a Hugging Face model appropriate for the
//! current system.
//!
//! `HF_TOKEN` is optional and enables gated/private models when the policy does.

#[cfg(feature = "native")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use candlelighter::{native::HuggingFaceArtifacts, ModelSelector, SelectionPolicy};
    use std::io::{self, Write};

    let search = std::env::args().nth(1);
    let token = std::env::var("HF_TOKEN").ok();
    let selector = ModelSelector::new(token.clone());
    println!("system: {:#?}", selector.capabilities());
    match selector.select_from_hub(search.as_deref(), 50, &SelectionPolicy::default())? {
        Some(model) => {
            println!(
                "selected {} ({:?} parameters, {} downloads)",
                model.id, model.parameter_count, model.downloads
            );
            print!("Download this model now? [y/N] ");
            io::stdout().flush()?;

            let mut answer = String::new();
            io::stdin().read_line(&mut answer)?;
            if matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
                println!("downloading {}...", model.id);
                let artifacts = HuggingFaceArtifacts::from_hub(&model.id, None, token)?;
                println!("downloaded to {}", artifacts.root.display());
                println!("run it later with:");
                println!(
                    "cargo run --example native_generate --no-default-features --features native -- {:?} \"Your prompt\"",
                    artifacts.root
                );
            } else {
                println!("download skipped");
            }
        }
        None => println!("no compatible model fits the detected system"),
    }
    Ok(())
}

#[cfg(not(feature = "native"))]
fn main() {
    eprintln!("enable this example with --no-default-features --features native");
}
