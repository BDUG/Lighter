#[cfg(feature = "native")]
pub mod native;
#[cfg(feature = "native")]
mod native_llama;
#[cfg(feature = "native")]
pub use native_llama::NativeLlama;
#[cfg(feature = "native")]
mod native_selector;
#[cfg(feature = "native")]
pub use native_selector::{
    Accelerator, HuggingFaceModel, ModelSelector, SelectionPolicy, SystemCapabilities,
};
#[cfg(feature = "native")]
pub mod native_advanced;
#[cfg(feature = "native")]
pub mod native_prompt;
#[cfg(feature = "native")]
pub mod native_training;

#[cfg(feature = "candle")]
pub mod prelude;

#[cfg(feature = "candle")]
pub mod examples;
#[cfg(feature = "candle")]
pub mod layer;
#[cfg(feature = "candle")]
pub mod preprocessing;

#[cfg(feature = "candle")]
pub mod activations;
#[cfg(feature = "candle")]
pub mod convolutiontypes;
#[cfg(feature = "candle")]
pub mod densetypes;
#[cfg(feature = "candle")]
pub mod embeddingtypes;
#[cfg(feature = "candle")]
pub mod layers;
#[cfg(feature = "candle")]
pub mod losses;
#[cfg(feature = "candle")]
pub mod models;
#[cfg(feature = "candle")]
pub mod optimizers;
#[cfg(feature = "candle")]
pub mod parallelmodel;
#[cfg(feature = "candle")]
pub mod parallelmodeltypes;
#[cfg(feature = "candle")]
pub mod poolingtypes;
#[cfg(feature = "candle")]
pub mod recurrenttypes;
#[cfg(feature = "candle")]
pub mod saveweightstype;
#[cfg(feature = "candle")]
pub mod sequentialmodel;
#[cfg(feature = "candle")]
pub mod serializationtensor;
#[cfg(feature = "candle")]
pub mod topk;
#[cfg(feature = "candle")]
pub mod utils;
