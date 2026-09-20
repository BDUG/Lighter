//! Discover a Hugging Face model appropriate for the current system.
//!
//! `HF_TOKEN` is optional and enables gated/private models when the policy does.

#[cfg(feature = "native")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use candlelighter::{ModelSelector, SelectionPolicy};

    let search = std::env::args().nth(1);
    let selector = ModelSelector::new(std::env::var("HF_TOKEN").ok());
    println!("system: {:#?}", selector.capabilities());
    match selector.select_from_hub(search.as_deref(), 50, &SelectionPolicy::default())? {
        Some(model) => println!(
            "selected {} ({:?} parameters, {} downloads)",
            model.id, model.parameter_count, model.downloads
        ),
        None => println!("no compatible model fits the detected system"),
    }
    Ok(())
}

#[cfg(not(feature = "native"))]
fn main() {
    eprintln!("enable this example with --no-default-features --features native");
}
