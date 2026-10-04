#[cfg(feature = "native")]
pub mod contrastive_lm;
pub mod graph2text;
#[cfg(feature = "server")]
pub mod model_host;
#[cfg(feature = "native")]
pub mod native;
#[cfg(feature = "native")]
pub use contrastive_lm::{ContrastiveLm, CLM_V01_8B_MODEL_ID};
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

/// Joint-embedding predictive architectures for images and videos.
pub mod jepa;
/// Continuous-time neural ODE and liquid neural-network building blocks.
pub mod liquid;
/// Exact distributed prefill and decode context parallelism.
pub mod context_parallel;

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

#[cfg(feature = "native")]
pub mod graph_training;

#[cfg(feature = "native")]
pub mod transformers;

#[cfg(feature = "native")]
pub mod arm;
#[cfg(feature = "arm-npu")]
pub mod arm_npu;
