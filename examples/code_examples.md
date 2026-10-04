<a id="examples"></a>

# Complete code examples

This is the code cookbook for the usage handbook. Each standalone listing below
is a complete checked-in program with imports, feature gates, inputs, execution,
and error handling. Copy a listing to `examples/<name>.rs` in a checkout and run
its command from the root; dependencies are already declared in `Cargo.toml`.
There are no omitted function bodies in these listings. For use in another crate,
add `candlelighter` and the upstream crates imported by the selected program;
Candle dependency versions must match the library's manifest.

Model-free examples validate small operations and APIs. The XOR example checks
its complete truth table after training and checkpoint reload. Toy logits,
TinyStudent and flat objective-output parameters are labeled in source. Model-backed
workflows require matching artifacts; local checkpoint integration tests do not
establish compatibility with every Hub model. Downloaded-model and physical NPU
execution are unverified unless an individual entry states otherwise. Large CLM runs need roughly 16 GB of
half-precision weights plus working memory, or quantized loading and temporary
shard conversion memory. Use `--release` for real decoder inference.

<a id="examples-example-index"></a>

### Example index

| Program | Feature selection | Validation / prerequisites | Capabilities |
| --- | --- | --- | --- |
| [serve_model](#examples-serve_model) | server | Compile-checked; compatible MODEL_DIR required | Embed the OpenAI-compatible host around a single native backend. See [host guide](../docs/handbook.md#host) for CLI, curl, Python SDK and limits. |
| [handbook_keras](#examples-handbook_keras) | default | Local: validated | Sequential regression, persistent classification optimizer, shape checks and prediction-equivalent safetensors reload. |
| [handbook_layers](#examples-handbook_layers) | default | Local: validated | Feature shaping/scaling and inverse, every dense activation, Conv1D, both pooling paths, normalization/flatten, LSTM/GRU, embedding, autoencoder and fit/predict. |
| [handbook_experimental](#examples-handbook_experimental) | default | Local: validated narrow cases | Conv2D, feature masks, dropout, classification CE, minimal attention/MoE, PEFT and split constructors, manual merge, ODE solvers and liquid step. Constructor-only checks do not establish training support. |
| [handbook_native](#examples-handbook_native) | native | Local: validated | A complete toy NativeBackend, capacity-two scheduling, queued/active cancellation, greedy generation/EOS/logprobs, sampling controls, choices/JSON/regex/GBNF, templates, every transform, fork/join, constraint semantics, schemas/signatures and MCP request/response round trips. |
| [handbook_training](#examples-handbook_training) | native | Local: validated | LoRA gradient accumulation/averaging, AdamW clipping and SGD, adapter restoration, ignored labels, reward training, PPO/DPO/distillation hooks, both quantization types and cache lifecycle. |
| [handbook_jepa_liquid](#examples-handbook_jepa_liquid) | none | Local: validated | Custom PatchEncoder, I-JEPA and V-JEPA, online/EMA image and video updates, three solvers, imported liquid parameters, explicit CfC/LFM states and readout fitting. |
| [native_advanced](#examples-native_advanced) | native | Local: validated | Int4 projection, copy-on-write KV cache, terminal-aware GAE, clipped PPO, DPO and temperature-scaled distillation. |
| [native_finetune](#examples-native_finetune) | native | Local path validated; snapshot path compile-checked | LoRA forward/backward, AdamW, mock SFT contract, pairwise reward model; optional MODEL_DIR performs a real NativeLlama LM-head update. |
| [native_prompt](#examples-native_prompt) | native | Local: validated | Choice/generation directives, static bindings, finite grammar, ReAct parsing, typed signatures and complete MCP JSON-RPC request serialization. |
| [jepa](#examples-jepa) | none | Local: validated | I-JEPA image and tube-masked V-JEPA forward passes, plus five trainable image updates. |
| [liquid_networks](#examples-liquid_networks) | none | Local: validated | RK4 analytical decay check, stacked LFM sequence/readout training and irregular CfC elapsed times. |
| [native_generate](#examples-native_generate) | native | Compile-checked; compatible MODEL_DIR required | Local Hugging Face artifacts/config/tokenizer/shards, NativeLlama, HuggingFaceBackend, seeded generation and EOS reporting. No downloads when snapshot is complete. |
| [select_huggingface_model](#examples-select_huggingface_model) | native | Compile-checked; network required | Machine capability detection, Hub metadata search, policy selection and an explicit y/N download prompt. |
| [auto_native_generate](#examples-auto_native_generate) | native | Compile-checked; network/weights required | Policy-based selection, automatic snapshot download, Int8 conversion and autoregressive generation. |
| [clm_generate](#examples-clm_generate) | native | Compile-checked; network or local 8B snapshot required | CLM 8B offline/Hub paths, optional Int8/Int4 loading, joined prompts, generation and stop accounting. |
| [internet_jepa](#examples-internet_jepa) | native | Compile-checked; network required | Download a real image and exercise I-JEPA, V-JEPA and trainable masked prediction. |
| [internet_liquid](#examples-internet_liquid) | native | Compile-checked; network required | Download temperature observations, normalize, fit LFM readout, handle irregular CfC time and produce an ODE baseline. |
| [graph2text](#examples-graph2textrs) | none | Local: validated | Exact graph realization and provenance. |
| [graph2text_native](#examples-graph2text_nativers) | native | Compatible local model required | Generate text from selected graph facts. |
| [G-Retriever Python programs](#examples-g-retriever-python-implementation-and-examples) | Python environment | Core tests validated; model/graph data required for generation | PCST retrieval, graph soft prompts and training. |
| [graph2text_finetune](#examples-graph2text-finetune) | native | Compatible local model required | Train and reload LM-head LoRA using graph instructions. |
| [transformers_pipeline](#examples-transformers-pipeline) | native | Local snapshot or Hub model required | Reusable batched causal generation. |
| [arm_acceleration](#examples-arm-acceleration) | native; optional arm-sve/arm-sme | x86 fallback validated; ARM codegen checked | Runtime vector and matrix dispatch. |
| [arm_npu_generate](#examples-arm-npu-generate) | arm-npu | Compiled model and vendor SDK required | TFLite delegate generation. |
| [arm_npu_host](#examples-arm-npu-host) | arm-npu,server | Compiled model and vendor SDK required | Thread-affine delegate HTTP hosting. |
| [ai_winter](#examples-ai-winter) | candle | CPU training and separate inference validated | XOR learning and safetensors persistence. |
| [Candle ARM bridge](../docs/handbook.md#candle-arm-integration) | candle,native | CPU comparison validated | Explicit inference-only tensor/buffer bridge. |
| [context_parallel_attention](#examples-context-parallel-attention) | none | CPU example and four unit tests validated | Exact sharded attention reference. |

`server` means `--no-default-features --features server` (and includes native).
`default` selects Candle and native; `native` means
`--no-default-features --features native`; `none` means `--no-default-features`.
The table covers runnable public workflows. The main handbook covers the status
of unimplemented KAN/DoRA/other roadmap APIs; there is no invented runnable code
for capabilities absent from the crate.

<a id="examples-choose-a-learning-path"></a>

### Choose a learning path

1. Porting Keras layers: run `handbook_layers`, then `handbook_keras`, and compare
   the [Python program](../docs/handbook.md#keras-complete-executable-python-listing).
2. Building a native generator: run `handbook_native`, then use `native_generate`
   with a compatible snapshot; tune request sampling and lifecycle explicitly.
3. Training adapters/objectives: run `native_finetune` without a snapshot, then
   `handbook_training`; add a snapshot only for an actual LM-head update.
4. Image/video self-supervision: run `jepa` and `handbook_jepa_liquid`, then optionally
   `internet_jepa`. Keep target patches out of visible context.
5. Continuous-time sequences: run `liquid_networks` and `handbook_jepa_liquid`, then
   optionally `internet_liquid`; time intervals are elapsed durations.
6. Legacy Candle experiments: consult the final section, including its execution
   limits, rather than treating every selector entry as a passing smoke test.

<a id="examples-complete-standalone-rust-programs"></a>

### Complete standalone Rust programs

<a id="examples-serve_model"></a>

#### serve_model

Embed the single-model HTTP host in a Rust application. Compatible local snapshot
required. This example is compile-checked; the host's real HTTP inference path was
validated with a tiny synthetic safetensors checkpoint.

```bash
cargo run --locked --release --example serve_model --no-default-features --features server -- MODEL_DIR
```

[Source](serve_model.rs) · [Complete API guide](../docs/handbook.md#host)

<!-- source: serve_model.rs -->
```rust
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
```

<a id="examples-lighter-serve-cli"></a>

#### lighter-serve CLI

The complete CLI loads an actual local/Hub checkpoint, chooses quantized loading,
and starts the host. See the API guide for a full request/response walkthrough.

<!-- source: ../src/serve.rs -->
```rust
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
```

<a id="examples-handbook_keras"></a>

#### handbook_keras

Sequential regression, persistent classification optimizer, shape checks and prediction-equivalent safetensors reload.

**Validation:** Local: validated.

[Source](handbook_keras.rs)

```bash
cargo run --locked --example handbook_keras
```

<!-- source: handbook_keras.rs -->
```rust
//! Complete Rust training/persistence counterparts to the Python Keras examples.
#[cfg(feature = "candle")]
fn main() -> anyhow::Result<()> {
    use candlelighter::prelude::*;
    let dev = Device::Cpu;
    let vars = VarMap::new();
    let mut model = SequentialModel::new(
        vars.clone(),
        vec![
            Box::new(Dense::new(
                4,
                2,
                Activations::Relu,
                &dev,
                &vars,
                "hidden".into(),
            )),
            Box::new(Dense::new(
                1,
                4,
                Activations::Linear,
                &dev,
                &vars,
                "output".into(),
            )),
        ],
    );
    let x = Tensor::new(&[[[1f32, 2.]], [[2., 3.]], [[3., 4.]], [[4., 5.]]], &dev)?;
    let y = Tensor::new(&[[[3f32]], [[5.]], [[7.]], [[9.]]], &dev)?;
    model.compile(Optimizers::SGD(0.01), Loss::MSE);
    model.fit(x.clone(), y, 3, false);
    let predictions = model.predict(&x).unwrap();
    assert_eq!(predictions.len(), 4);
    for prediction in predictions {
        assert_eq!(prediction.dims(), &[1, 1]);
    }

    // A custom classification loop supplies a loss the Lighter fit wrapper rejects.
    let mut class_vars = VarMap::new();
    let classifier = Dense::new(
        2,
        2,
        Activations::Linear,
        &dev,
        &class_vars,
        "classifier".into(),
    );
    let inputs = Tensor::new(&[[1f32, 2.], [2., 3.], [3., 4.], [4., 5.]], &dev)?;
    let labels = Tensor::new(&[0u32, 0, 1, 1], &dev)?;
    let mut optimizer = candle_nn::AdamW::new(
        class_vars.all_vars(),
        candle_nn::ParamsAdamW {
            lr: 1e-3,
            weight_decay: 0.01,
            ..Default::default()
        },
    )?;
    for _ in 0..3 {
        let logits = classifier.forward(inputs.clone());
        let loss = candle_nn::loss::cross_entropy(&logits, &labels)?;
        assert!(loss.to_scalar::<f32>()?.is_finite());
        optimizer.backward_step(&loss)?;
    }
    // Restore into the SAME registered variables, preserving tensor shapes.
    let before = classifier.forward(inputs.clone()).to_vec2::<f32>()?;
    let directory =
        std::env::temp_dir().join(format!("lighter-handbook-keras-{}", std::process::id()));
    std::fs::create_dir(&directory)?; // Refuse to overwrite an existing directory.
    let path = directory.join("classification.safetensors");
    let result = (|| -> anyhow::Result<()> {
        class_vars.save(&path)?;
        // Destroy the in-memory weights so equality demonstrates actual restoration.
        for variable in class_vars.all_vars() {
            variable.set(&variable.as_tensor().zeros_like()?)?;
        }
        assert_ne!(before, classifier.forward(inputs.clone()).to_vec2::<f32>()?);
        class_vars.load(&path)?;
        assert_eq!(before, classifier.forward(inputs).to_vec2::<f32>()?);
        Ok(())
    })();
    if path.exists() {
        std::fs::remove_file(&path)?;
    }
    std::fs::remove_dir(&directory)?;
    result?;
    println!(
        "Rust regression, custom classification and prediction-equivalent weight reload completed"
    );
    Ok(())
}
#[cfg(not(feature = "candle"))]
fn main() {
    eprintln!("enable the candle feature");
}
```

<a id="examples-handbook_layers"></a>

#### handbook_layers

Feature shaping/scaling and inverse, every dense activation, Conv1D, both pooling paths, normalization/flatten, LSTM/GRU, embedding, autoencoder and fit/predict.

**Validation:** Local: validated.

[Source](handbook_layers.rs)

```bash
cargo run --locked --example handbook_layers
```

<!-- source: handbook_layers.rs -->
```rust
//! Small CPU examples for the usage handbook; no downloads or output files.
#[cfg(feature = "candle")]
fn main() -> anyhow::Result<()> {
    use candlelighter::embeddingtypes::EmbeddingType;
    use candlelighter::layer::embeddinglayer::{Embed, EmbeddingLayerTrait};
    use candlelighter::prelude::*;
    use candlelighter::preprocessing::features::{Features, FeaturesTrait};
    use candlelighter::recurrenttypes::RecurrentType;

    let dev = Device::Cpu;
    let mut features = Features::new(dev.clone());
    features.add_feature_1_d(vec![1., 2.]);
    features.add_feature_1_d(vec![3., 4.]);
    assert_eq!(features.get_data_tensor().dims(), &[2, 1, 2]);

    let values = Tensor::new(&[[1f32, 2., 3.]], &dev)?;
    let scaling = FeatureScaling::new(values);
    assert_eq!(
        scaling.min_max_normalization().to_vec2::<f32>()?,
        vec![vec![0., 0.5, 1.]]
    );
    println!("z-score: {:?}", scaling.z_score().to_vec2::<f32>()?);
    let restored = scaling.min_max_normalization_reverse(scaling.min_max_normalization());
    assert_eq!(restored.to_vec1::<f32>()?, vec![1., 2., 3.]);

    let vars = VarMap::new();
    let x = Tensor::new(&[[1f32, 2.]], &dev)?;
    for (i, activation) in [
        Activations::Linear,
        Activations::Relu,
        Activations::Silu,
        Activations::Sigmoid,
        Activations::Softmax,
    ]
    .into_iter()
    .enumerate()
    {
        let dense = Dense::new(3, 2, activation, &dev, &vars, format!("dense_{i}"));
        assert_eq!(dense.forward(x.clone()).dims(), &[1, 3]);
    }

    let kernel = Tensor::ones((1, 1, 2), DType::F32, &dev)?;
    let conv = Conv::new(kernel, 1, 0, 1, 1, 1, &dev, &vars, "conv".into());
    let signal = Tensor::new(&[[[1f32, 2., 3., 4.]]], &dev)?;
    assert_eq!(
        conv.forward(signal).flatten_all()?.to_vec1::<f32>()?,
        vec![3., 5., 7.]
    );

    let image = Tensor::new(&[[[1f32, 2.], [3., 4.]]], &dev)?;
    // Current wrapper behavior: MAX averages; AVERAGE takes the maximum.
    for (kind, expected) in [(PoolingType::MAX, 2.5f32), (PoolingType::AVERAGE, 4.)] {
        let pool = Pooling::new(kind, 2, 2, &dev, &vars, "pool".into());
        assert_eq!(
            pool.forward(image.clone())
                .flatten_all()?
                .to_vec1::<f32>()?,
            vec![expected]
        );
    }
    let flat = Flatten::new(&dev, &vars, "flatten".into());
    assert_eq!(flat.forward(image).dims(), &[1, 4]);
    // Normalization's current forward method discards the normalized tensor.
    let normalization = Normalization::new(1, &dev, &vars, "norm".into());
    assert_eq!(
        normalization.forward(x.clone()).to_vec2::<f32>()?,
        x.to_vec2::<f32>()?
    );
    let normalized = x.broadcast_div(&x.sqr()?.sum_keepdim(1)?.sqrt()?)?;
    println!(
        "explicit L2 normalization: {:?}",
        normalized.to_vec2::<f32>()?
    );

    for kind in [RecurrentType::LSTM, RecurrentType::GRU] {
        // Separate maps: the wrapper does not namespace its recurrent weights.
        let rnn = Recurrent::new(kind, 2, 3, &dev, &VarMap::new(), "rnn".into());
        assert_eq!(rnn.forward(x.clone()).dims(), &[1, 3]);
    }
    let embedding = Embed::new(
        EmbeddingType::Standard,
        8,
        3,
        &dev,
        &VarMap::new(),
        "embed".into(),
    );
    assert_eq!(
        embedding.forward(Tensor::new(&[0u32, 2, 7], &dev)?).dims(),
        &[3, 3]
    );

    // An autoencoder architecture assembled from existing dense layers.
    let ae_vars = VarMap::new();
    let autoencoder = SequentialModel::new(
        ae_vars.clone(),
        vec![
            Box::new(Dense::new(
                1,
                2,
                Activations::Relu,
                &dev,
                &ae_vars,
                "encoder".into(),
            )),
            Box::new(Dense::new(
                2,
                1,
                Activations::Linear,
                &dev,
                &ae_vars,
                "decoder".into(),
            )),
        ],
    );
    assert_eq!(autoencoder.forward(x).dims(), &[1, 2]);

    // Small supervised regression using the existing fit/predict API.
    let train_vars = VarMap::new();
    let mut model = SequentialModel::new(
        train_vars.clone(),
        vec![Box::new(Dense::new(
            1,
            2,
            Activations::Linear,
            &dev,
            &train_vars,
            "regression".into(),
        ))],
    );
    model.compile(Optimizers::SGD(0.01), Loss::MSE);
    let inputs = Tensor::new(&[[[1f32, 2.]], [[2., 3.]]], &dev)?;
    let targets = Tensor::new(&[[[3f32]], [[5.]]], &dev)?;
    model.fit(inputs.clone(), targets, 2, false);
    assert_eq!(model.predict(&inputs).unwrap().len(), 2);
    println!("handbook layer examples completed");
    Ok(())
}

#[cfg(not(feature = "candle"))]
fn main() {
    eprintln!("run this example with the candle feature enabled");
}
```

<a id="examples-handbook_experimental"></a>

#### handbook_experimental

Conv2D, feature masks, dropout, classification CE, minimal attention/MoE, PEFT and split constructors, manual merge, ODE solvers and liquid step. Constructor-only checks do not establish training support.

**Validation:** Local: validated narrow cases.

[Source](handbook_experimental.rs)

```bash
cargo run --locked --example handbook_experimental
```

<!-- source: handbook_experimental.rs -->
```rust
//! Narrow demonstrations of experimental Candle APIs and upstream alternatives.
//! See docs/handbook.md for limits; no downloads or persistent output files.
#[cfg(feature = "candle")]
fn main() -> anyhow::Result<()> {
    use candlelighter::densetypes::DenseType;
    use candlelighter::layer::sparsemoe::{SparseMoE, SparseMoETrait};
    use candlelighter::liquid::*;
    use candlelighter::prelude::*;
    let dev = Device::Cpu;
    let vars = VarMap::new();
    let kernel = Tensor::ones((1, 1, 2, 2), DType::F32, &dev)?;
    let conv = Conv::new(kernel, 2, 0, 1, 1, 1, &dev, &vars, "conv2d".into());
    assert_eq!(
        conv.forward(Tensor::ones((1, 1, 4, 4), DType::F32, &dev)?)
            .dims(),
        &[1, 1, 3, 3]
    );
    let x = Tensor::new(&[[1f32, 2., 3.]], &dev)?;
    let mask = Tensor::new(&[[1f32, 0., 1.]], &dev)?;
    assert_eq!(x.mul(&mask)?.to_vec2::<f32>()?, vec![vec![1., 0., 3.]]);
    let dropout = candle_nn::Dropout::new(0.1);
    assert_eq!(
        dropout.forward_t(&x, false)?.to_vec2::<f32>()?,
        x.to_vec2::<f32>()?
    );
    let _training = dropout.forward_t(&x, true)?;
    let config = candle_nn::ParamsAdamW {
        lr: 1e-3,
        weight_decay: 0.01,
        ..Default::default()
    };
    let _optimizer = candle_nn::AdamW::new(vars.all_vars(), config)?;
    let logits = Tensor::new(&[[2f32, -1.], [-1., 2.]], &dev)?;
    let labels = Tensor::new(&[0u32, 1], &dev)?;
    assert!(candle_nn::loss::cross_entropy(&logits, &labels)?.to_scalar::<f32>()? > 0.);

    let attention = SelfAttention::new(1, 1, 1, 1, &dev, &VarMap::new(), "attention".into());
    assert_eq!(
        attention
            .forward(Tensor::ones((1, 1), DType::F32, &dev)?)
            .dims(),
        &[1, 1]
    );
    let moe = SparseMoE::new(2, 4, 4, &dev, &VarMap::new(), "moe".into());
    assert_eq!(
        moe.forward(Tensor::ones((1, 4), DType::F32, &dev)?).dims(),
        &[1, 4]
    );
    // Only construction: these variants are not a working adapter trainer.
    for kind in [DenseType::LORA, DenseType::DORA] {
        let rank = Tensor::zeros((2, 2), DType::F32, &dev)?;
        let _adapter = Dense::new2(
            2,
            2,
            Activations::Linear,
            kind,
            rank,
            1.,
            &dev,
            &VarMap::new(),
            "adapter".into(),
        );
    }
    let branches: Vec<Box<dyn Trainable>> = vec![
        Box::new(Dense::new(
            2,
            2,
            Activations::Linear,
            &dev,
            &vars,
            "branch_a".into(),
        )),
        Box::new(Dense::new(
            2,
            2,
            Activations::Linear,
            &dev,
            &vars,
            "branch_b".into(),
        )),
    ];
    let _split = ParallelModel::new(ParallelModelType::Split, &dev, vars, branches);
    // Explicit averaging is a working ensemble alternative.
    let a = Tensor::new(&[[1f32, 3.]], &dev)?;
    let b = Tensor::new(&[[3f32, 5.]], &dev)?;
    assert_eq!(((a + b)? * 0.5)?.to_vec2::<f32>()?, vec![vec![2., 4.]]);

    for solver in [OdeSolver::Euler, OdeSolver::Heun, OdeSolver::RungeKutta4] {
        let ode = NeuralOde::new(|_, state: &[f32]| vec![-state[0]], solver);
        let state = ode.solve(&[1.], 0., 1., 20)?;
        assert!((state[0] - (-1f32).exp()).abs() < 0.02);
    }
    let cell = LiquidCell::new(2, 4)?.with_solver(OdeSolver::Heun);
    assert_eq!(cell.step(&[1., 0.], &[0.; 4], 0.1)?.len(), 4);
    println!("experimental handbook recipes completed (see documented limits)");
    Ok(())
}

#[cfg(not(feature = "candle"))]
fn main() {
    eprintln!("run this example with the candle feature enabled");
}
```

<a id="examples-handbook_native"></a>

#### handbook_native

A complete toy NativeBackend, capacity-two scheduling, queued/active cancellation, greedy generation/EOS/logprobs, sampling controls, choices/JSON/regex/GBNF, templates, every transform, fork/join, constraint semantics, schemas/signatures and MCP request/response round trips.

**Validation:** Local: validated.

[Source](handbook_native.rs)

```bash
cargo run --locked --example handbook_native --no-default-features --features native
```

<!-- source: handbook_native.rs -->
```rust
//! Model-free scheduler, sampling, constraints, and workflow examples.
#[cfg(feature = "native")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use candlelighter::native::*;
    use candlelighter::native_prompt::*;
    use serde_json::json;

    // Deliberately tiny backend: token 0 = a, token 1 = b, token 2 = EOS.
    struct Demo;
    impl NativeBackend for Demo {
        fn encode(&self, _: &str) -> NativeResult<Vec<u32>> {
            Ok(vec![0])
        }
        fn decode(&self, tokens: &[u32]) -> NativeResult<String> {
            Ok(tokens
                .iter()
                .filter_map(|id| match id {
                    0 => Some('a'),
                    1 => Some('b'),
                    _ => None,
                })
                .collect())
        }
        fn logits(&mut self, _: &str, _: &[u32], position: usize) -> NativeResult<Vec<f32>> {
            Ok(if position < 3 {
                vec![4., 1., -10.]
            } else {
                vec![-10., -10., 4.]
            })
        }
    }
    let mut engine = NativeEngine::new(Demo, 2, vec![2])?;
    let sampling = SamplingParams {
        max_tokens: 8,
        temperature: 0.0,
        seed: 42,
        logprobs: Some(2),
        ..Default::default()
    };
    for id in ["first", "second", "cancel"] {
        engine.submit(GenerateRequest {
            id: id.into(),
            prompt: "demo".into(),
            sampling: sampling.clone(),
            constraint: None,
        })?;
    }
    assert!(engine.cancel("cancel"));
    let responses = engine.run_to_completion()?;
    assert_eq!(responses.len(), 2); // Queued cancellation removes the request without a response.
    assert_eq!(responses.iter().filter(|r| r.text == "aa").count(), 2);
    for response in responses {
        println!(
            "{}: {:?} {:?}",
            response.id, response.text, response.finish_reason
        );
    }

    engine.submit(GenerateRequest {
        id: "active-cancel".into(),
        prompt: "demo".into(),
        sampling: sampling.clone(),
        constraint: None,
    })?;
    assert!(engine.step()?.is_empty());
    assert!(engine.cancel("active-cancel"));
    assert_eq!(
        engine.run_to_completion()?[0].finish_reason,
        FinishReason::Cancelled
    );

    // Sampling controls are independent of the model implementation.
    SamplingParams {
        temperature: 0.7,
        top_k: Some(2),
        top_p: 0.9,
        min_p: 0.05,
        presence_penalty: 0.1,
        frequency_penalty: 0.1,
        repetition_penalty: 1.1,
        min_tokens: 1,
        max_tokens: 8,
        stop: vec!["ab".into()],
        stop_token_ids: vec![2],
        bad_token_ids: vec![1],
        logit_bias: [(0, 0.2)].into(),
        seed: 42,
        ..Default::default()
    }
    .validate()?;

    for spec in [
        ConstraintSpec::Choice {
            choices: vec!["a".into(), "b".into()],
        },
        ConstraintSpec::Regex {
            pattern: "a+".into(),
        },
        ConstraintSpec::Json,
        ConstraintSpec::Gbnf {
            grammar: "root ::= \"a\" | \"b\"".into(),
            root: "root".into(),
        },
    ] {
        let constraint = spec.compile()?;
        assert!(constraint.allows_prefix(""));
    }
    engine.submit(GenerateRequest {
        id: "constrained".into(),
        prompt: "demo".into(),
        sampling,
        constraint: Some(ConstraintSpec::Choice {
            choices: vec!["a".into()],
        }),
    })?;
    assert_eq!(engine.run_to_completion()?[0].text, "a");

    struct Executor;
    impl PromptExecutor for Executor {
        fn generate(
            &mut self,
            _: &str,
            _: usize,
            _: Option<&str>,
            _: Option<&dyn OutputConstraint>,
        ) -> NativeResult<String> {
            Ok("hello".into())
        }
        fn select(&mut self, _: &str, choices: &[String]) -> NativeResult<String> {
            Ok(choices[0].clone())
        }
    }
    let branch = Workflow {
        steps: vec![WorkflowStep::Transform {
            input: "answer".into(),
            output: "uppercase".into(),
            operation: Transform::Uppercase,
        }],
    };
    let workflow = Workflow {
        steps: vec![
            WorkflowStep::Prompt {
                template: PromptTemplate::parse("{{gen answer max_tokens=8}}")?,
            },
            WorkflowStep::Fork {
                branches: vec![branch],
            },
        ],
    };
    let mut state = Variables::new();
    workflow.execute(&mut Executor, &mut state)?;
    assert_eq!(state["branch_0"]["uppercase"], json!("HELLO"));
    let call = ToolCall::from_openai_json(r#"{"name":"search","arguments":{"query":"Rust"}}"#)?;
    // Each constraint exposes distinct prefix/completion semantics.
    let regex = RegexConstraint::new("a+")?;
    assert!(regex.is_complete("aaa"));
    assert!(!regex.is_complete("b"));
    assert!(regex.allows_prefix("b")); // Current regex API does not prune prefixes.
    assert!(JsonConstraint.is_complete(r#"{"ok":true}"#));
    let choice = ChoiceConstraint::new(vec!["yes".into(), "no".into()])?;
    assert!(choice.allows_prefix("ye"));
    assert!(!choice.allows_prefix("maybe"));
    let grammar = GbnfGrammar::parse("root ::= \"start\" | \"stop\"", "root")?;
    assert!(grammar.is_complete("start"));
    assert!(!grammar.allows_prefix("quit"));
    let mut json_state = Variables::from([("record".into(), json!({"name":"Rust"}))]);
    let transforms = Workflow {
        steps: vec![
            WorkflowStep::Transform {
                input: "record".into(),
                output: "name".into(),
                operation: Transform::JsonPointer("/name".into()),
            },
            WorkflowStep::Transform {
                input: "name".into(),
                output: "lower".into(),
                operation: Transform::Lowercase,
            },
        ],
    };
    transforms.execute(&mut Executor, &mut json_state)?;
    assert_eq!(json_state["lower"], json!("rust"));
    let definition = ToolDefinition {
        name: "search".into(),
        description: "Demo search schema".into(),
        input_schema: json!({"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}),
    };
    let request = McpRequest {
        jsonrpc: "2.0".into(),
        id: json!(1),
        method: "tools/call".into(),
        params: json!({"name":definition.name,"arguments":call.arguments}),
    };
    let request_text = serde_json::to_string(&request)?;
    let parsed: McpRequest = serde_json::from_str(&request_text)?;
    assert_eq!(parsed.method, "tools/call");
    let response = McpResponse {
        jsonrpc: "2.0".into(),
        id: parsed.id,
        result: Some(json!({"content":[{"type":"text","text":"demo result"}]})),
        error: None,
    };
    let response_text = serde_json::to_string(&response)?;
    let decoded: McpResponse = serde_json::from_str(&response_text)?;
    assert!(decoded.result.is_some());
    let signature = Signature::parse("question, context -> answer")?;
    assert_eq!(signature.inputs, vec!["question", "context"]);
    assert_eq!(signature.outputs, vec!["answer"]);
    println!("parsed tool call: {call:?}; workflow: {state:?}");
    Ok(())
}

#[cfg(not(feature = "native"))]
fn main() {
    eprintln!("run this example with the native feature enabled");
}
```

<a id="examples-handbook_training"></a>

#### handbook_training

LoRA gradient accumulation/averaging, AdamW clipping and SGD, adapter restoration, ignored labels, reward training, PPO/DPO/distillation hooks, both quantization types and cache lifecycle.

**Validation:** Local: validated.

[Source](handbook_training.rs)

```bash
cargo run --locked --example handbook_training --no-default-features --features native
```

<!-- source: handbook_training.rs -->
```rust
//! Explicit gradient accumulation, optimizer hooks, token loss and cache operations.
#[cfg(feature = "native")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use candlelighter::native::NativeResult;
    use candlelighter::native_advanced::*;
    use candlelighter::native_training::*;

    let mut adapter = LoraAdapter::new(
        3,
        2,
        LoraConfig {
            rank: 2,
            alpha: 4.,
            dropout: 0.,
        },
        42,
    )?;
    let inputs = [[1f32, 0.5, -0.5], [0.5, 1., 0.]];
    let before = adapter.forward(&inputs[0], false, 42)?;
    let mut accumulated = adapter.zero_gradients();
    for input in &inputs {
        let (gradient, input_gradient) = adapter.backward(input, &[0.2, -0.1])?;
        assert_eq!(input_gradient.len(), 3);
        LoraAdapter::accumulate(&mut accumulated, &gradient)?;
    }
    for gradient in accumulated.a.iter_mut().chain(&mut accumulated.b) {
        *gradient /= inputs.len() as f32;
    }
    let mut optimizer = AdamW::new(AdamWConfig {
        learning_rate: 1e-3,
        max_gradient_norm: Some(0.5),
        ..Default::default()
    })?;
    adapter.apply_gradients(&accumulated, &mut optimizer)?;
    assert_ne!(before, adapter.forward(&inputs[0], false, 42)?);
    adapter.apply_sgd(&accumulated, 1e-3)?;
    let (a, b) = adapter.weights();
    let restored = LoraAdapter::from_weights(
        3,
        2,
        LoraConfig {
            rank: 2,
            alpha: 4.,
            dropout: 0.,
        },
        a.to_vec(),
        b.to_vec(),
    )?;
    assert_eq!(
        adapter.forward(&inputs[0], false, 1)?,
        restored.forward(&inputs[0], false, 1)?
    );

    let loss = causal_lm_loss(
        &[vec![2., 0., -1.], vec![0., 2., -1.]],
        &[0, -100],
        -100,
        0.1,
    )?;
    assert!(loss.loss.is_finite());
    assert_eq!(&loss.gradients[3..], &[0., 0., 0.]);
    let mut reward = RewardHead::new(3)?;
    assert!(reward
        .apply_pairwise_update(&[1., 0., 0.], &[0., 0., 1.], &mut optimizer)?
        .is_finite());
    assert!(reward.score(&[1., 0., 0.])?.is_finite());

    // These parameters represent objective outputs, not transformer weights.
    // A real backend must propagate output gradients through its model.
    struct OutputParameters(Vec<f32>);
    impl OptimizationTarget for OutputParameters {
        fn apply_output_gradients(&mut self, gradient: &[f32], rate: f32) -> NativeResult<()> {
            assert_eq!(self.0.len(), gradient.len());
            for (parameter, derivative) in self.0.iter_mut().zip(gradient) {
                *parameter -= rate * derivative;
            }
            Ok(())
        }
    }
    let (advantages, returns) =
        generalized_advantage_estimate(&[1., 0.5], &[0.2, 0.3, 0.], &[false, true], 0.99, 0.95)?;
    let batch = PpoBatch {
        old_log_probs: vec![-0.8, -0.7],
        new_log_probs: vec![-0.75, -0.8],
        advantages,
        old_values: vec![0.2, 0.3],
        new_values: vec![0.25, 0.35],
        returns,
        entropies: vec![0.5, 0.4],
    };
    let mut policy = OutputParameters(batch.new_log_probs.clone());
    let ppo = apply_ppo_update(&mut policy, &batch, &PpoConfig::default(), 0.01)?;
    assert!(ppo.loss.is_finite());
    let mut preference = OutputParameters(vec![0.4, 0.3]);
    assert!(
        apply_dpo_update(&mut preference, &[0.4, 0.3], &[-0.2, -0.1], 0.1, 0., 0.01)?.is_finite()
    );
    let mut student = OutputParameters(vec![1., 2., 0.]);
    let distilled = apply_distillation_update(
        &mut student,
        &[4., 1., 0.],
        &[1., 2., 0.],
        Some(0),
        &DistillationConfig::default(),
        0.01,
    )?;
    assert!(distilled.loss.is_finite());

    for dtype in [QuantizationType::Int8, QuantizationType::Int4] {
        let matrix = QuantizedMatrix::quantize(
            2,
            2,
            &[1., -1., 0.5, 0.25],
            QuantizationConfig {
                dtype,
                group_size: 2,
            },
        )?;
        assert_eq!(matrix.dequantize().len(), 4);
        assert_eq!(matrix.matvec(&[2., 1.])?.len(), 2);
        println!(
            "quantized matrix: {}x{}, {} bytes",
            matrix.rows(),
            matrix.cols(),
            matrix.storage_bytes()
        );
    }
    let mut cache = PagedKvCache::new(16, 2)?;
    cache.append("prompt", &[1., 2.], &[3., 4.])?;
    cache.fork("prompt", "branch")?;
    cache.append("branch", &[5., 6.], &[7., 8.])?;
    cache.truncate_left("branch", 1)?;
    assert_eq!(cache.tokens("prompt"), 1);
    assert_eq!(cache.read("branch")?, (vec![5., 6.], vec![7., 8.]));
    assert!(cache.remove("branch"));
    println!("gradient accumulation, loss/update hooks, quantization and cache checks completed");
    Ok(())
}
#[cfg(not(feature = "native"))]
fn main() {
    eprintln!("enable the native feature");
}
```

<a id="examples-handbook_jepa_liquid"></a>

#### handbook_jepa_liquid

Custom PatchEncoder, I-JEPA and V-JEPA, online/EMA image and video updates, three solvers, imported liquid parameters, explicit CfC/LFM states and readout fitting.

**Validation:** Local: validated.

[Source](handbook_jepa_liquid.rs)

```bash
cargo run --locked --example handbook_jepa_liquid --no-default-features
```

<!-- source: handbook_jepa_liquid.rs -->
```rust
//! Complete custom encoder, image/video training and stateful continuous-time recipes.
use candlelighter::jepa::*;
use candlelighter::liquid::*;

struct SummaryEncoder;
impl PatchEncoder for SummaryEncoder {
    fn latent_dim(&self) -> usize {
        2
    }
    fn encode(&self, patch: &[f32]) -> Result<Vec<f32>, JepaError> {
        if patch.is_empty() {
            return Err(JepaError("empty patch".into()));
        }
        let mean = patch.iter().sum::<f32>() / patch.len() as f32;
        let energy = patch.iter().map(|x| x * x).sum::<f32>() / patch.len() as f32;
        Ok(vec![mean, energy])
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let pixels: Vec<f32> = (0..16).map(|value| value as f32 / 15.).collect();
    let mask = [BlockMask {
        row: 0,
        col: 1,
        height: 1,
        width: 1,
    }];
    let image = IJepa::new(SummaryEncoder, 2, 2)?.forward(&pixels, 4, 4, 1, &mask)?;
    assert!(image.loss.is_finite());
    let mut video = pixels.clone();
    video.extend(pixels.iter().rev().copied());
    let prediction = VJepa::new(SummaryEncoder, 2, 2)?.forward(&video, 2, 4, 4, 1, &mask)?;
    assert!(prediction.loss.is_finite());
    let geometry = JepaGeometry {
        height: 4,
        width: 4,
        channels: 1,
        patch_height: 2,
        patch_width: 2,
    };
    let mut trainer = JepaTrainer::new(
        JepaConfig {
            patch_elements: 4,
            latent_dim: 8,
            learning_rate: 0.01,
            target_momentum: 0.99,
        },
        42,
    )?;
    for _ in 0..3 {
        assert!(trainer.train_image(&pixels, geometry, &mask)?.is_finite());
        assert!(trainer.train_video(&video, 2, geometry, &mask)?.is_finite());
    }
    assert_eq!(trainer.steps(), 6);

    for solver in [OdeSolver::Euler, OdeSolver::Heun, OdeSolver::RungeKutta4] {
        let ode = NeuralOde::new(|_, state: &[f32]| vec![-state[0]], solver);
        assert!((ode.solve(&[1.], 0., 1., 20)?[0] - (-1f32).exp()).abs() < 0.02);
    }
    let cell = LiquidCell::from_parameters(
        1,
        2,
        vec![0.2, -0.1],
        vec![0.; 4],
        vec![0.; 2],
        vec![1., 2.],
        OdeSolver::Heun,
    )?;
    let mut hidden = vec![0.; cell.hidden_size()];
    for (input, elapsed) in [(1., 0.1), (0.5, 0.4), (0., 1.2)] {
        hidden = cell.step(&[input], &hidden, elapsed)?;
        assert!(hidden.iter().all(|x| x.is_finite()));
    }
    let samples = vec![vec![1., 0.], vec![0.5, 0.5], vec![0., 1.]];
    let elapsed = [0.1, 0.4, 1.2];
    let cfc = CfcCell::new(2, 4)?;
    let mut state = vec![0.; 4];
    let mut streamed = vec![];
    for (input, dt) in samples.iter().zip(elapsed) {
        state = cfc.step(input, &state, dt)?;
        streamed.push(state.clone());
    }
    assert_eq!(streamed, cfc.forward_irregular(&samples, &elapsed)?);
    let mut lfm = Lfm::new(2, &[8, 4], 1)?;
    let mut states = lfm.zero_state();
    for input in &samples {
        assert_eq!(lfm.step(input, &mut states, 0.1)?.len(), 1);
    }
    let targets = vec![vec![1.], vec![0.5], vec![0.]];
    for _ in 0..5 {
        assert!(lfm.fit_readout(&samples, &targets, 0.1, 0.05)?.is_finite());
    }
    assert_eq!(lfm.forward(&samples, 0.1)?.len(), samples.len());
    println!("custom JEPA encoder, six image/video updates and stateful liquid examples completed");
    Ok(())
}
```

<a id="examples-native_advanced"></a>

#### native_advanced

Int4 projection, copy-on-write KV cache, terminal-aware GAE, clipped PPO, DPO and temperature-scaled distillation.

**Validation:** Local: validated.

[Source](native_advanced.rs)

```bash
cargo run --locked --example native_advanced --no-default-features --features native
```

<!-- source: native_advanced.rs -->
```rust
//! Exhaustive tour of native quantization, paged KV caching, reinforcement
//! learning objectives, and teacher/student distillation.

#[cfg(feature = "native")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use candlelighter::native_advanced::*;

    // Quantize a row-major projection and execute it without dequantizing the
    // complete matrix. Int8 is also available via QuantizationType::Int8.
    let weights = [1.0, -1.0, 0.5, 0.25];
    let projection = QuantizedMatrix::quantize(
        2,
        2,
        &weights,
        QuantizationConfig {
            dtype: QuantizationType::Int4,
            group_size: 2,
        },
    )?;
    println!(
        "quantized projection: {:?}",
        projection.matvec(&[2.0, 1.0])?
    );

    // Pages are shared across forks until one branch writes to its tail. This
    // is suitable for prompt-prefix reuse and beam-search branches.
    let mut cache = PagedKvCache::new(16, 2)?;
    cache.append("prompt", &[1.0, 2.0], &[3.0, 4.0])?;
    cache.fork("prompt", "beam-2")?;
    cache.append("beam-2", &[5.0, 6.0], &[7.0, 8.0])?;
    println!("KV cache: {:?}", cache.stats());

    // Compute terminal-aware GAE, followed by the clipped PPO objective.
    let (advantages, returns) =
        generalized_advantage_estimate(&[1.0, 0.5], &[0.2, 0.3, 0.0], &[false, true], 0.99, 0.95)?;
    let ppo = ppo_objective(
        &PpoBatch {
            old_log_probs: vec![-0.8, -0.7],
            new_log_probs: vec![-0.75, -0.8],
            advantages,
            old_values: vec![0.2, 0.3],
            new_values: vec![0.25, 0.35],
            returns,
            entropies: vec![0.5, 0.4],
        },
        &PpoConfig::default(),
    )?;
    println!("PPO loss: {}", ppo.loss);

    // DPO consumes policy-vs-reference log-ratios for chosen/rejected answers.
    let (dpo, dpo_gradients) = dpo_loss(&[0.4, 0.3], &[-0.2, -0.1], 0.1, 0.0)?;
    println!("DPO loss: {dpo}, gradients: {dpo_gradients:?}");

    // Distillation combines temperature-scaled teacher KL with an optional
    // hard-label cross entropy term and returns student-logit gradients.
    let distilled = distillation_loss(
        &[4.0, 1.0, 0.0],
        &[1.0, 2.0, 0.0],
        Some(0),
        &DistillationConfig::default(),
    )?;
    println!("distillation loss: {}", distilled.loss);
    Ok(())
}

#[cfg(not(feature = "native"))]
fn main() {
    eprintln!("enable this example with --no-default-features --features native");
}
```

<a id="examples-native_finetune"></a>

#### native_finetune

LoRA forward/backward, AdamW, mock SFT contract, pairwise reward model; optional MODEL_DIR performs a real NativeLlama LM-head update.

**Validation:** Local path validated; snapshot path compile-checked.

[Source](native_finetune.rs)

```bash
cargo run --locked --example native_finetune --no-default-features --features native
```

For a real local snapshot, append `-- MODEL_DIR`; use release for larger models.

<!-- source: native_finetune.rs -->
```rust
//! Supervised fine-tuning, LoRA, AdamW, and reward-model example.

#[cfg(feature = "native")]
fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use candlelighter::native::{NativeError, NativeResult};
    use candlelighter::native_training::*;
    use candlelighter::{native::HuggingFaceArtifacts, NativeLlama};

    // With a snapshot argument, run a real LM-head LoRA update through the
    // native transformer. Without one, continue with the small inspectable
    // examples below.
    if let Some(directory) = std::env::args().nth(1) {
        let artifacts = HuggingFaceArtifacts::from_dir(directory)?;
        let tokenizer = tokenizers::Tokenizer::from_file(&artifacts.tokenizer)?;
        let encoded = tokenizer.encode("Fine-tuning a native transformer", true)?;
        let ids = encoded.get_ids();
        if ids.len() < 2 {
            return Err("tokenized fine-tuning text must contain two tokens".into());
        }
        let mut model = NativeLlama::load(&artifacts)?;
        model.enable_lm_head_lora(
            LoraConfig {
                rank: 8,
                alpha: 16.0,
                dropout: 0.0,
            },
            42,
        )?;
        let labels: Vec<i64> = ids[1..].iter().map(|id| i64::from(*id)).collect();
        let loss =
            supervised_fine_tune_step(&mut model, &ids[..ids.len() - 1], &labels, -100, 0.0, 1e-4)?;
        println!("NativeLlama LM-head LoRA step loss={loss}");
        return Ok(());
    }

    let mut adapter = LoraAdapter::new(
        3,
        2,
        LoraConfig {
            rank: 2,
            alpha: 4.0,
            dropout: 0.05,
        },
        42,
    )?;
    let input = [1.0, 0.5, -0.5];
    let delta = adapter.forward(&input, true, 7)?;
    let (gradients, input_gradient) = adapter.backward(&input, &[0.2, -0.1])?;
    let mut optimizer = AdamW::new(AdamWConfig::default())?;
    adapter.apply_gradients(&gradients, &mut optimizer)?;
    println!("LoRA delta={delta:?}, input gradient={input_gradient:?}");

    struct TinyStudent {
        logits: Vec<Vec<f32>>,
        last_gradient: Vec<f32>,
    }
    impl FineTunableTransformer for TinyStudent {
        fn token_logits(&mut self, _: &[u32]) -> NativeResult<Vec<Vec<f32>>> {
            Ok(self.logits.clone())
        }
        fn apply_token_gradients(
            &mut self,
            gradients: &[f32],
            learning_rate: f32,
        ) -> NativeResult<()> {
            if gradients.iter().any(|value| !value.is_finite()) {
                return Err(NativeError("non-finite training gradient".into()));
            }
            self.last_gradient = gradients
                .iter()
                .map(|value| value * learning_rate)
                .collect();
            Ok(())
        }
    }
    let mut student = TinyStudent {
        logits: vec![vec![2.0, 0.0, -1.0], vec![0.0, 2.0, -1.0]],
        last_gradient: vec![],
    };
    let loss = supervised_fine_tune_step(&mut student, &[10, 11], &[0, 1], -100, 0.1, 1e-4)?;
    println!(
        "SFT loss={loss}, gradient elements={}",
        student.last_gradient.len()
    );

    let mut reward = RewardHead::new(3)?;
    let pair_loss =
        reward.apply_pairwise_update(&[1.0, 0.5, 0.0], &[0.0, 0.5, 1.0], &mut optimizer)?;
    println!("reward-model pairwise loss={pair_loss}");
    Ok(())
}

#[cfg(not(feature = "native"))]
fn main() {
    eprintln!("enable this example with --no-default-features --features native");
}
```

<a id="examples-native_prompt"></a>

#### native_prompt

Choice/generation directives, static bindings, finite grammar, ReAct parsing, typed signatures and complete MCP JSON-RPC request serialization.

**Validation:** Local: validated.

[Source](native_prompt.rs)

```bash
cargo run --locked --example native_prompt --no-default-features --features native
```

<!-- source: native_prompt.rs -->
```rust
//! Declarative prompting, grammar constraints, workflows, and tool protocols.

#[cfg(feature = "native")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use candlelighter::native::{NativeError, NativeResult};
    use candlelighter::native_prompt::*;
    use serde_json::json;

    struct DemoExecutor;
    impl PromptExecutor for DemoExecutor {
        fn generate(
            &mut self,
            _: &str,
            _: usize,
            _: Option<&str>,
            constraint: Option<&dyn OutputConstraint>,
        ) -> NativeResult<String> {
            let output = "hello".to_string();
            if constraint.is_some_and(|constraint| !constraint.is_complete(&output)) {
                return Err(NativeError("demo output violates constraint".into()));
            }
            Ok(output)
        }
        fn select(&mut self, _: &str, choices: &[String]) -> NativeResult<String> {
            choices
                .first()
                .cloned()
                .ok_or_else(|| NativeError("no choices".into()))
        }
    }

    let template = PromptTemplate::parse(
        "Question: {{question}}\nFormat: {{select format choices=json|text}}\nAnswer: {{gen answer max_tokens=32 regex=hello stop=\"\\n\"}}",
    )?;
    let mut variables = Variables::from([("question".into(), json!("Say hello"))]);
    let output = template.execute(&mut DemoExecutor, &mut variables)?;
    println!("{output}");

    let grammar = GbnfGrammar::parse("root ::= command\ncommand ::= \"start\" | \"stop\"", "root")?;
    assert!(grammar.allows_prefix("sta"));
    assert!(grammar.is_complete("start"));

    let call = ToolCall::from_react(
        "Thought: look it up\nAction: search\nAction Input: {\"query\":\"Rust\"}",
    )?;
    println!("tool call: {call:?}");

    let request = McpRequest {
        jsonrpc: "2.0".into(),
        id: json!(1),
        method: "tools/call".into(),
        params: json!({"name": call.name, "arguments": call.arguments}),
    };
    println!("MCP: {}", serde_json::to_string(&request)?);
    println!(
        "DSPy-style signature: {:?}",
        Signature::parse("question, context -> answer")?
    );
    Ok(())
}

#[cfg(not(feature = "native"))]
fn main() {
    eprintln!("enable this example with --no-default-features --features native");
}
```

<a id="examples-jepa"></a>

#### jepa

I-JEPA image and tube-masked V-JEPA forward passes, plus five trainable image updates.

**Validation:** Local: validated.

[Source](jepa.rs)

```bash
cargo run --locked --example jepa --no-default-features
```

<!-- source: jepa.rs -->
```rust
//! Minimal I-JEPA and V-JEPA forward passes.
use candlelighter::jepa::{
    BlockMask, IJepa, JepaConfig, JepaGeometry, JepaTrainer, MeanEncoder, VJepa,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let target = [BlockMask {
        row: 0,
        col: 1,
        height: 1,
        width: 1,
    }];
    let pixels: Vec<f32> = (0..16).map(|x| x as f32 / 15.0).collect();

    let image_model = IJepa::new(MeanEncoder::new(4)?, 2, 2)?;
    let image = image_model.forward(&pixels, 4, 4, 1, &target)?;
    println!(
        "I-JEPA targets: {}, MSE: {:.6}",
        image.targets.len(),
        image.loss
    );

    let mut video_pixels = pixels.clone();
    video_pixels.extend(pixels.iter().rev());
    let video_model = VJepa::new(MeanEncoder::new(4)?, 2, 2)?;
    let video = video_model.forward(&video_pixels, 2, 4, 4, 1, &target)?;
    println!(
        "V-JEPA tube targets: {}, MSE: {:.6}",
        video.targets.len(),
        video.loss
    );

    // A real training step uses an online encoder and predictor, stop-gradient
    // target representations, and an EMA update of the target encoder.
    let mut trainer = JepaTrainer::new(
        JepaConfig {
            patch_elements: 4,
            latent_dim: 8,
            learning_rate: 1e-2,
            target_momentum: 0.99,
        },
        42,
    )?;
    let geometry = JepaGeometry {
        height: 4,
        width: 4,
        channels: 1,
        patch_height: 2,
        patch_width: 2,
    };
    for epoch in 0..5 {
        let loss = trainer.train_image(&pixels, geometry, &target)?;
        println!("training step {epoch}: {loss:.6}");
    }
    Ok(())
}
```

<a id="examples-liquid_networks"></a>

#### liquid_networks

RK4 analytical decay check, stacked LFM sequence/readout training and irregular CfC elapsed times.

**Validation:** Local: validated.

[Source](liquid_networks.rs)

```bash
cargo run --locked --example liquid_networks --no-default-features
```

<!-- source: liquid_networks.rs -->
```rust
//! Neural ODE and liquid foundation model examples.
use candlelighter::liquid::{CfcCell, Lfm, NeuralOde, OdeSolver};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let decay = NeuralOde::new(|_, state: &[f32]| vec![-state[0]], OdeSolver::RungeKutta4);
    let state = decay.solve(&[1.0], 0.0, 1.0, 20)?;
    println!(
        "Neural ODE y(1): {:.6} (expected {:.6})",
        state[0],
        (-1.0f32).exp()
    );

    let mut model = Lfm::new(2, &[8, 4], 1)?;
    let samples = vec![vec![1.0, 0.0], vec![0.5, 0.5], vec![0.0, 1.0]];
    println!("LFM outputs: {:?}", model.forward(&samples, 0.1)?);
    let targets = vec![vec![1.0], vec![0.5], vec![0.0]];
    println!(
        "LFM readout training loss: {:.6}",
        model.fit_readout(&samples, &targets, 0.1, 0.05)?
    );

    let cfc = CfcCell::new(2, 4)?;
    let elapsed = [0.1, 0.4, 1.2];
    println!(
        "CfC irregular-time states: {:?}",
        cfc.forward_irregular(&samples, &elapsed)?
    );
    Ok(())
}
```

<a id="examples-native_generate"></a>

#### native_generate

Local Hugging Face artifacts/config/tokenizer/shards, NativeLlama, HuggingFaceBackend, seeded generation and EOS reporting. No downloads when snapshot is complete.

**Validation:** Compile-checked; compatible MODEL_DIR required.

[Source](native_generate.rs)

```bash
cargo run --locked --release --example native_generate --no-default-features --features native -- MODEL_DIR "Once upon a time"
```

<!-- source: native_generate.rs -->
```rust
//! Generate text from a local Hugging Face snapshot without Candle.
//!
//! Run with:
//! `cargo run --example native_generate --no-default-features --features native -- MODEL_DIR "prompt"`

#[cfg(feature = "native")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use candlelighter::native::{
        GenerateRequest, HuggingFaceArtifacts, HuggingFaceBackend, NativeEngine, SamplingParams,
    };
    use candlelighter::NativeLlama;

    let mut args = std::env::args().skip(1);
    let directory = args
        .next()
        .ok_or("usage: native_generate MODEL_DIR [PROMPT]")?;
    let prompt = args.next().unwrap_or_else(|| "Once upon a time".into());
    let artifacts = HuggingFaceArtifacts::from_dir(directory)?;
    let eos = artifacts.config.eos_token_ids();
    let model = NativeLlama::load(&artifacts)?;
    let backend = HuggingFaceBackend::new(model, &artifacts)?;
    let mut engine = NativeEngine::new(backend, 1, eos)?;
    engine.submit(GenerateRequest {
        id: "example".into(),
        prompt,
        constraint: None,
        sampling: SamplingParams {
            max_tokens: 64,
            temperature: 0.7,
            top_p: 0.9,
            seed: 42,
            ..Default::default()
        },
    })?;
    let response = engine.run_to_completion()?.remove(0);
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
```

<a id="examples-select_huggingface_model"></a>

#### select_huggingface_model

Machine capability detection, Hub metadata search, policy selection and an explicit y/N download prompt.

**Validation:** Compile-checked; network required.

[Source](select_huggingface_model.rs)

```bash
cargo run --locked --example select_huggingface_model --no-default-features --features native -- "Llama"
```

<!-- source: select_huggingface_model.rs -->
```rust
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
```

<a id="examples-auto_native_generate"></a>

#### auto_native_generate

Policy-based selection, automatic snapshot download, Int8 conversion and autoregressive generation.

**Validation:** Compile-checked; network/weights required.

[Source](auto_native_generate.rs)

```bash
cargo run --locked --release --example auto_native_generate --no-default-features --features native -- "TinyLlama" "Hello"
```

<!-- source: auto_native_generate.rs -->
```rust
//! Select, download, and run a compatible Hugging Face model without Candle.
//!
//! This example downloads model weights. Use a narrow search term for a small
//! checkpoint, for example `TinyLlama`.

#[cfg(feature = "native")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use candlelighter::native::{
        GenerateRequest, HuggingFaceArtifacts, HuggingFaceBackend, NativeEngine, SamplingParams,
    };
    use candlelighter::native_advanced::{QuantizationConfig, QuantizationType};
    use candlelighter::{ModelSelector, NativeLlama, SelectionPolicy};

    let mut args = std::env::args().skip(1);
    let search = args.next().unwrap_or_else(|| "TinyLlama".into());
    let prompt = args
        .next()
        .unwrap_or_else(|| "Explain why Rust is safe:".into());
    let token = std::env::var("HF_TOKEN").ok();
    let selector = ModelSelector::new(token.clone());
    let policy = SelectionPolicy::default();
    let selected = selector
        .select_from_hub(Some(&search), 25, &policy)?
        .ok_or("no compatible Hugging Face model fits this system")?;
    eprintln!(
        "selected {} for {:#?}",
        selected.id,
        selector.capabilities()
    );

    let artifacts = HuggingFaceArtifacts::from_hub(&selected.id, None, token)?;
    let eos = artifacts.config.eos_token_ids();
    let mut model = NativeLlama::load(&artifacts)?;
    model.quantize(QuantizationConfig {
        dtype: QuantizationType::Int8,
        group_size: 64,
    })?;
    let backend = HuggingFaceBackend::new(model, &artifacts)?;
    let mut engine = NativeEngine::new(backend, 1, eos)?;
    engine.submit(GenerateRequest {
        id: "automatic-example".into(),
        prompt,
        constraint: None,
        sampling: SamplingParams {
            max_tokens: 64,
            temperature: 0.7,
            top_p: 0.9,
            seed: 42,
            ..Default::default()
        },
    })?;
    println!("{}", engine.run_to_completion()?.remove(0).text);
    Ok(())
}

#[cfg(not(feature = "native"))]
fn main() {
    eprintln!("enable this example with --no-default-features --features native");
}
```

<a id="examples-clm_generate"></a>

#### clm_generate

CLM 8B offline/Hub paths, optional Int8/Int4 loading, joined prompts, generation and stop accounting.

**Validation:** Compile-checked; network or local 8B snapshot required.

[Source](clm_generate.rs)

```bash
cargo run --locked --release --example clm_generate --no-default-features --features native -- "Explain ownership in Rust."
```

<!-- source: clm_generate.rs -->
```rust
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
```

<a id="examples-internet_jepa"></a>

#### internet_jepa

Download a real image and exercise I-JEPA, V-JEPA and trainable masked prediction.

**Validation:** Compile-checked; network required.

[Source](internet_jepa.rs)

```bash
cargo run --locked --example internet_jepa --no-default-features --features native
```

<!-- source: internet_jepa.rs -->
```rust
//! I-JEPA and V-JEPA training on an image downloaded from the internet.
//!
//! The architecture and masking follow the official Meta I-JEPA and V-JEPA
//! examples: https://github.com/facebookresearch/ijepa and
//! https://github.com/facebookresearch/jepa. The Rust logo is only a compact,
//! redistributable input that keeps this example quick to run.

#[cfg(feature = "native")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use candlelighter::jepa::{BlockMask, JepaConfig, JepaGeometry, JepaTrainer};
    use image::ImageReader;
    use std::io::{Cursor, Read};

    const IMAGE_URL: &str = "https://www.rust-lang.org/logos/rust-logo-512x512.png";
    let response = ureq::get(IMAGE_URL).call()?;
    let mut bytes = Vec::new();
    response.into_reader().read_to_end(&mut bytes)?;
    let source = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()?
        .decode()?
        .resize_exact(32, 32, image::imageops::FilterType::Triangle)
        .to_luma8();
    let pixels = source
        .as_raw()
        .iter()
        .map(|pixel| *pixel as f32 / 255.0)
        .collect::<Vec<_>>();

    let geometry = JepaGeometry {
        height: 32,
        width: 32,
        channels: 1,
        patch_height: 8,
        patch_width: 8,
    };
    // Mask two spatially separated blocks, as I-JEPA predicts several target
    // regions from the same visible context.
    let masks = [
        BlockMask {
            row: 0,
            col: 0,
            height: 1,
            width: 2,
        },
        BlockMask {
            row: 2,
            col: 2,
            height: 2,
            width: 1,
        },
    ];
    let config = JepaConfig {
        patch_elements: 64,
        latent_dim: 32,
        learning_rate: 0.02,
        target_momentum: 0.996,
    };
    let mut ijepa = JepaTrainer::new(config, 23)?;
    for epoch in 0..20 {
        let loss = ijepa.train_image(&pixels, geometry, &masks)?;
        if epoch % 5 == 0 || epoch == 19 {
            println!("I-JEPA epoch {epoch:2}: loss={loss:.6}");
        }
    }

    // Turn the downloaded image into a two-frame clip. Repeating the same
    // spatial mask over both frames exercises V-JEPA tube masking.
    let mirrored = source
        .rows()
        .flat_map(|row| row.rev().map(|pixel| pixel[0] as f32 / 255.0))
        .collect::<Vec<_>>();
    let video = [pixels, mirrored].concat();
    let mut vjepa = JepaTrainer::new(config, 29)?;
    for epoch in 0..20 {
        let loss = vjepa.train_video(&video, 2, geometry, &masks)?;
        if epoch % 5 == 0 || epoch == 19 {
            println!("V-JEPA epoch {epoch:2}: loss={loss:.6}");
        }
    }
    Ok(())
}

#[cfg(not(feature = "native"))]
fn main() {
    eprintln!("enable this internet-backed example with --no-default-features --features native");
}
```

<a id="examples-internet_liquid"></a>

#### internet_liquid

Download temperature observations, normalize, fit LFM readout, handle irregular CfC time and produce an ODE baseline.

**Validation:** Compile-checked; network required.

[Source](internet_liquid.rs)

```bash
cargo run --locked --example internet_liquid --no-default-features --features native
```

<!-- source: internet_liquid.rs -->
```rust
//! Liquid-network forecasting on a public internet time-series dataset.
//!
//! The daily minimum-temperature dataset is the same real-world forecasting
//! dataset published by Jason Brownlee at
//! https://github.com/jbrownlee/Datasets. The continuous-time model follows the
//! official LTC examples at https://github.com/raminmh/liquid_time_constant_networks.

#[cfg(feature = "native")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use candlelighter::liquid::{CfcCell, Lfm, NeuralOde, OdeSolver};

    const DATA_URL: &str =
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv";
    let csv = ureq::get(DATA_URL).call()?.into_string()?;
    let temperatures = csv
        .lines()
        .skip(1)
        .filter_map(|line| {
            line.rsplit_once(',')?
                .1
                .trim_matches('"')
                .parse::<f32>()
                .ok()
        })
        .take(365)
        .collect::<Vec<_>>();
    if temperatures.len() < 2 {
        return Err("the downloaded dataset did not contain enough samples".into());
    }
    let mean = temperatures.iter().sum::<f32>() / temperatures.len() as f32;
    let scale = temperatures
        .iter()
        .map(|value| (value - mean).abs())
        .fold(0.0f32, f32::max)
        .max(f32::EPSILON);
    let normalized = temperatures
        .iter()
        .map(|x| (x - mean) / scale)
        .collect::<Vec<_>>();
    let inputs = normalized[..normalized.len() - 1]
        .iter()
        .map(|x| vec![*x])
        .collect::<Vec<_>>();
    let targets = normalized[1..].iter().map(|x| vec![*x]).collect::<Vec<_>>();

    let mut lfm = Lfm::new(1, &[12, 6], 1)?;
    for epoch in 0..25 {
        let loss = lfm.fit_readout(&inputs, &targets, 1.0, 0.01)?;
        if epoch % 5 == 0 || epoch == 24 {
            println!("LFM epoch {epoch:2}: normalized MSE={loss:.6}");
        }
    }

    // CfC directly accepts irregular elapsed times (e.g. missing observations).
    let cfc = CfcCell::new(1, 6)?;
    let elapsed = (0..inputs.len())
        .map(|i| if i % 17 == 0 { 2.0 } else { 1.0 })
        .collect::<Vec<_>>();
    let states = cfc.forward_irregular(&inputs, &elapsed)?;
    println!("CfC processed {} irregular observations", states.len());

    // A Neural ODE baseline models exponential relaxation toward the dataset mean.
    let relaxation = NeuralOde::new(
        |_, state: &[f32]| vec![-0.15 * state[0]],
        OdeSolver::RungeKutta4,
    );
    let forecast = relaxation.solve(&[normalized[0]], 0.0, 7.0, 28)?;
    println!(
        "Neural ODE seven-day forecast: {:.2} C",
        forecast[0] * scale + mean
    );
    Ok(())
}

#[cfg(not(feature = "native"))]
fn main() {
    eprintln!("enable this internet-backed example with --no-default-features --features native");
}
```

<a id="examples-complete-historical-candle-training-examples"></a>

### Complete historical Candle training examples

The following are full library example modules, not Cargo `--example` targets.
They are compiled with the Candle library. Run their named choices through:

```bash
cargo run --locked --bin candlelighter
```

Arrow keys select an example; Enter runs it. Run from the root for clock data.
These original demonstrations retain their original behavior: long training
loops, hardcoded dimensions, experimental APIs and BERT/Llama credential/path
placeholders. They are **compiled but not validated as end-to-end workflows**.
The small handbook programs above provide passing alternatives and document
known API behavior. Set up transformer tokens at runtime in a custom application;
do not commit secret values to these modules. A complete source listing does not
mean every architecture is implemented or every historical example succeeds.

<a id="examples-simple_dnn"></a>

#### simple_dnn

Selector entries: **Simple DNN, Simple DNN2, Simple DNN3**. [Source](../lib/examples/simple_dnn.rs).

<!-- source: ../lib/examples/simple_dnn.rs -->
```rust
use rand::distributions::Distribution;

#[allow(unused)]
use crate::prelude::*;
use crate::preprocessing::{self, features::{Features, FeaturesTrait}};

pub struct Dataitem {
    x: Vec<usize>,
    y: Vec<usize>
}

pub fn generatedata(numofelements: usize, limit: usize) -> Vec<Dataitem> {
    let mut result: Vec<Dataitem> = vec![];

    let vals: Vec<u64> = (0..numofelements as u64).collect();
    for (_i,_valuee) in vals.iter().enumerate() {
        let index = Uniform::new(0, limit);
        let mut rng = rand::thread_rng();
        let a = index.sample(&mut rng);
        let b = index.sample(&mut rng);

        let mut resultelement =  Dataitem {
            x: Vec::new(), // e.g., 1. , 2.
            y: Vec::new() // e.g., 3.
        };
        resultelement.x.push(a);
        resultelement.x.push(b);

        resultelement.y.push( a+b );

        result.push(resultelement);
    }
    return result;
}


pub fn simple_dnn() {
    let varmap = VarMap::new();
    let dev = candle_core::Device::cuda_if_available(0).unwrap();

    let dataset = generatedata(2000, 10);
    let mut featurehelper_x = Features::new(dev.clone());
    let mut featurehelper_y = Features::new(dev.clone());

    for (_j, value) in dataset.iter().enumerate() {
        let tmp_x_value = value.x.iter().filter_map( |s| Some(*s as f32) ) .collect();
        featurehelper_x.add_feature_1_d(tmp_x_value);

        let tmp_y_value = value.y.iter().filter_map( |s| Some(*s as f32) ) .collect();
        featurehelper_y.add_feature_1_d(tmp_y_value);
    }
    let tmp_x = featurehelper_x.get_data_tensor();
    let tmp_y = featurehelper_y.get_data_tensor();

    let mut layers: Vec<Box<dyn Trainable>> = vec![];
    let mut name1 = String::new();
    name1.push_str("fc1");
    layers.push(Box::new(Dense::new(20, 2, Activations::Relu, &dev, &varmap, name1 )));
    let mut name2 = String::new();
    name2.push_str("fc2");
    layers.push(Box::new(Dense::new(2, 20, Activations::Relu, &dev, &varmap, name2 )));
    let mut name3 = String::new();
    name3.push_str("fc3");
    layers.push(Box::new(Dense::new(1, 2, Activations::Relu, &dev, &varmap, name3 )));

    let mut model = SequentialModel::new(varmap, layers);
    
    let numbers: Vec<f32> = (0..=100).map(|x| x as f32).collect();
    let scaling = preprocessing::featurescaling::FeatureScaling::new(Tensor::new( numbers, &dev).unwrap());

    model.compile(Optimizers::SGD(0.0001), Loss::MSE);
    model.fit(
        scaling.min_max_normalization_other(tmp_x), 
        scaling.min_max_normalization_other(tmp_y), 
        1000, 
        false);
    
    let mut featurehelper_x_test = Features::new(dev.clone());
    //let x_test: [[f32; 2]; 1] = [ [4., 5.] ];
    let x_test: [f32; 2] = [4., 5.];
    let _tmp_tensor = Tensor::new(&x_test, &dev).unwrap();
    featurehelper_x_test.add_feature(_tmp_tensor);

    let tmp_tensor = scaling.min_max_normalization_other(featurehelper_x_test.get_data_tensor());
    let prediction = model.predict(&tmp_tensor).unwrap();
    println!("Prediction: {}", scaling.min_max_normalization_reverse( prediction.get(0).unwrap().clone() ));
}



pub fn simple_dnn2() {
    let varmap = VarMap::new();
    let dev = candle_core::Device::cuda_if_available(0).unwrap();

    let dataset = generatedata(2000, 10);
    let mut featurehelper_x = Features::new(dev.clone());
    let mut featurehelper_y = Features::new(dev.clone());

    for (_j, value) in dataset.iter().enumerate() {
        let tmp_x_value = value.x.iter().filter_map( |s| Some(*s as f32) ) .collect();
        featurehelper_x.add_feature_1_d(tmp_x_value);

        let tmp_y_value = value.y.iter().filter_map( |s| Some(*s as f32) ) .collect();
        featurehelper_y.add_feature_1_d(tmp_y_value);
    }
    let tmp_x = featurehelper_x.get_data_tensor();
    let tmp_y = featurehelper_y.get_data_tensor();

    let mut _rank = Tensor::rand(0.0, 1.0, tmp_x.get(0).unwrap().shape(), &dev).unwrap().to_dtype(DType::F32).unwrap();
    let mut _alpha = 0.25;

    let mut layers: Vec<Box<dyn Trainable>> = vec![];
    let mut name1 = String::new();
    name1.push_str("fc1");
    layers.push(Box::new(Dense::new2(20, 2, Activations::Relu, crate::densetypes::DenseType::LORA,_rank.clone(), _alpha, &dev, &varmap, name1 )));
    let mut name2 = String::new();
    name2.push_str("fc2");
    layers.push(Box::new(Dense::new(2, 20, Activations::Relu,  &dev, &varmap, name2 )));
    let mut name3 = String::new();
    name3.push_str("fc3");
    layers.push(Box::new(Dense::new(1, 2, Activations::Relu,&dev, &varmap, name3 )));

    let mut model = SequentialModel::new(varmap, layers);
    
    let numbers: Vec<f32> = (0..=100).map(|x| x as f32).collect();
    let scaling = preprocessing::featurescaling::FeatureScaling::new(Tensor::new( numbers, &dev).unwrap());

    model.compile(Optimizers::SGD(0.0001), Loss::MSE);
    model.fit(
        scaling.min_max_normalization_other(tmp_x), 
        scaling.min_max_normalization_other(tmp_y), 
        10, 
        false);
    
    let mut featurehelper_x_test = Features::new(dev.clone());
    //let x_test: [[f32; 2]; 1] = [ [4., 5.] ];
    let x_test: [f32; 2] = [4., 5.];
    let _tmp_tensor = Tensor::new(&x_test, &dev).unwrap();
    featurehelper_x_test.add_feature(_tmp_tensor);

    let tmp_tensor = scaling.min_max_normalization_other(featurehelper_x_test.get_data_tensor());
    let prediction = model.predict(&tmp_tensor).unwrap();
    println!("Prediction: {}", scaling.min_max_normalization_reverse( prediction.get(0).unwrap().clone() ));
}



pub fn simple_dnn3() {
    let varmap = VarMap::new();
    let dev = candle_core::Device::cuda_if_available(0).unwrap();

    let dataset = generatedata(2000, 10);
    let mut featurehelper_x = Features::new(dev.clone());
    let mut featurehelper_y = Features::new(dev.clone());

    for (_j, value) in dataset.iter().enumerate() {
        let tmp_x_value = value.x.iter().filter_map( |s| Some(*s as f32) ) .collect();
        featurehelper_x.add_feature_1_d(tmp_x_value);

        let tmp_y_value = value.y.iter().filter_map( |s| Some(*s as f32) ) .collect();
        featurehelper_y.add_feature_1_d(tmp_y_value);
    }
    let tmp_x = featurehelper_x.get_data_tensor();
    let tmp_y = featurehelper_y.get_data_tensor();

    let mut _rank = Tensor::rand(0.0, 1.0, tmp_x.get(0).unwrap().shape(), &dev).unwrap().to_dtype(DType::F32).unwrap();
    let mut _alpha = 0.25;

    let mut layers: Vec<Box<dyn Trainable>> = vec![];
    let mut name1 = String::new();
    name1.push_str("fc1");
    layers.push(Box::new(Dense::new2(20, 2, Activations::Relu, crate::densetypes::DenseType::DORA,_rank.clone(), _alpha, &dev, &varmap, name1 )));
    let mut name2 = String::new();
    name2.push_str("fc2");
    layers.push(Box::new(Dense::new(2, 20, Activations::Relu,  &dev, &varmap, name2 )));
    let mut name3 = String::new();
    name3.push_str("fc3");
    layers.push(Box::new(Dense::new(1, 2, Activations::Relu,&dev, &varmap, name3 )));

    let mut model = SequentialModel::new(varmap, layers);
    
    let numbers: Vec<f32> = (0..=100).map(|x| x as f32).collect();
    let scaling = preprocessing::featurescaling::FeatureScaling::new(Tensor::new( numbers, &dev).unwrap());

    model.compile(Optimizers::SGD(0.0001), Loss::MSE);
    model.fit(
        scaling.min_max_normalization_other(tmp_x), 
        scaling.min_max_normalization_other(tmp_y), 
        100, 
        false);

    let mut featurehelper_x_test = Features::new(dev.clone());
    //let x_test: [[f32; 2]; 1] = [ [4., 5.] ];
    let x_test: [f32; 2] = [4., 5.];
    let _tmp_tensor = Tensor::new(&x_test, &dev).unwrap();
    featurehelper_x_test.add_feature(_tmp_tensor);

    let tmp_tensor = scaling.min_max_normalization_other(featurehelper_x_test.get_data_tensor());
    let prediction = model.predict(&tmp_tensor).unwrap();
    println!("Prediction: {}", scaling.min_max_normalization_reverse( prediction.get(0).unwrap().clone() ));
}
```

<a id="examples-simple_cnn"></a>

#### simple_cnn

Selector entries: **Simple CNN**. [Source](../lib/examples/simple_cnn.rs).

<!-- source: ../lib/examples/simple_cnn.rs -->
```rust

#[allow(unused)]
use crate::prelude::*;
use crate::preprocessing::{self, features::{Features, FeaturesTrait}};

pub fn simple_cnn(){
    let varmap = VarMap::new();
    let dev = candle_core::Device::Cpu;

    let images = Tensor::read_npy("data/clock/clock_image.npy").unwrap();
    let results = Tensor::read_npy("data/clock/clock_time.npy").unwrap();

    let mut featurehelper_x = Features::new(dev.clone());
    let mut featurehelper_y = Features::new(dev.clone());

    let _images_vec: Vec<Vec<Vec<f32>>> = images.to_dtype(DType::F32).unwrap().to_vec3().unwrap();
    let _clock_vec: Vec<Vec<f32>> = results.to_dtype(DType::F32).unwrap().to_vec2().unwrap();
    for (_position, data) in _images_vec.iter().enumerate(){
        featurehelper_x.add_feature(Tensor::new(data.clone(), &dev).unwrap());
    }
    for (_position, data) in _clock_vec.iter().enumerate(){
        featurehelper_y.add_feature_1_d(data.clone());
    }

    let mut layers: Vec<Box<dyn Trainable>> = vec![];
    let mut name1 = String::new();
    name1.push_str("convolution 1");
    layers.push(Box::new(Conv::new2( ConvolutionTypes::Default, 2, 2, 1, 1, 1, &dev, &varmap, name1)));
    
    let mut name2 = String::new();
    name2.push_str("maxpooling 1");
    layers.push(Box::new(Pooling::new( PoolingType::MAX, 2, 2, &dev, &varmap, name2)));

    let mut name5 = String::new();
    name5.push_str("flatten");
    layers.push(Box::new(Flatten::new( &dev, &varmap, name5)));

    let mut name6 = String::new();
    name6.push_str("fully connected 1");
    layers.push(Box::new(Dense::new(8, 1024, Activations::Relu, &dev, &varmap, name6 )));

    let mut name7 = String::new();
    name7.push_str("fully connected 2");
    layers.push(Box::new(Dense::new(2, 8, Activations::Relu, &dev, &varmap, name7 )));
    
    let mut model = SequentialModel::new(varmap, layers); 

    let numbers: Vec<f32> = (0..=60).map(|x| x as f32).collect();
    let scaling = preprocessing::featurescaling::FeatureScaling::new(Tensor::new( numbers, &dev).unwrap());

    let tmp_x = featurehelper_x.get_data_tensor();
    let tmp_y = featurehelper_y.get_data_tensor();

    let x_testimage = images.get(0).unwrap().to_dtype(DType::F32).unwrap();
    let mut featurehelper_x_test = Features::new(dev.clone());
    featurehelper_x_test.add_feature(x_testimage);
    let x_testtensor = featurehelper_x_test.get_data_tensor();

    model.compile(Optimizers::SGD(0.0005), Loss::MSE);
    model.fit(
        scaling.min_max_normalization_other(tmp_x), 
        scaling.min_max_normalization_other(tmp_y), 
        4000, 
        false);


    let prediction = model.predict(&scaling.min_max_normalization_other(x_testtensor)).unwrap();
    println!("Prediction: {}", scaling.min_max_normalization_reverse( prediction.get(0).unwrap().clone() ));
    println!("Expected: {}", results.get(0).unwrap() );
}
```

<a id="examples-simple_rnn"></a>

#### simple_rnn

Selector entries: **Simple RNN, Simple RNN2**. [Source](../lib/examples/simple_rnn.rs).

<!-- source: ../lib/examples/simple_rnn.rs -->
```rust
use std::ops::Add;

#[allow(unused)]
use crate::prelude::*;
use crate::recurrenttypes::RecurrentType;
use rand::Rng;

pub fn simple_rnn() {
    let varmap = VarMap::new();
    let dev = candle_core::Device::cuda_if_available(0).unwrap();
    let x: [[[f32; 2]; 1]; 6] = [ [[1., 2.]] , [[2., 1.]] ,[[3., 4.]], [[5., 6.]], [[5., 5.]] , [[4., 5.]]];
    let y: [[[f32; 1]; 1]; 6] = [ [[3.]], [[3.]], [[7.]], [[11.]] , [[10.]], [[9.]]];

    let mut layers: Vec<Box<dyn Trainable>> = vec![];
    let mut name1 = String::new();
    name1.push_str("rnn1");
    layers.push(Box::new(Recurrent::new(RecurrentType::LSTM,2, 4, &dev, &varmap, name1 )));
    let mut name3 = String::new();
    name3.push_str("fc1");
    layers.push(Box::new(Dense::new(1, 4, Activations::Relu, &dev, &varmap, name3 )));

    let mut model = SequentialModel::new(varmap, layers);
    
    model.compile(Optimizers::SGD(0.01), Loss::MSE);
    model.fit(
        Tensor::new(&x, &dev).unwrap(), 
        Tensor::new(&y, &dev).unwrap(), 
        2000, 
        true);
    
    let x_test: [[f32; 2]; 1] = [ [1., 1.] ];
    let prediction = model.predict(&Tensor::new(&x_test, &dev).unwrap()).unwrap();
    println!("prediction: {}", prediction.get(0).unwrap().clone() );
}

pub fn to_tensor(input: &Vec<Vec<f32>>, device: &Device) -> Tensor{
    let dimension1: usize = input.len();
    let dimension2: usize = input.get(0).unwrap().len();
    let mut result = Vec::new();
    for i in 0..dimension1 {
        for j in 0..dimension2 {
            let val = input.get(i).unwrap().get(j).unwrap();
            result.push(val.clone().to_owned());
        }
    }
    return Tensor::from_vec(result, (dimension1,1,dimension2), device ).unwrap().clone();
}

pub fn generate_sum_pair(batchsize: usize) -> Vec<Vec<Vec<f32>>> {
    let mut rng = rand::thread_rng();

    let mut input_vector : Vec<Vec<f32>> = Vec::new();
    let mut output_vector : Vec<Vec<f32>> = Vec::new();

    for _i in 0..batchsize {
        let n1: f32 = rng.gen_range(0.0..100.0);
        let n2: f32 = rng.gen_range(0.0..100.0);
        let mut input_pair : Vec<f32> = Vec::new();
        input_pair.push(n1);
        input_pair.push(n2);
    
        let sum: f32 = n1.add(n2);
        let mut output_pair : Vec<f32> = Vec::new();
        output_pair.push(sum);

        input_vector.push(input_pair);
        output_vector.push(output_pair);
        
    }

    let mut result: Vec<Vec<Vec<f32>>> = Vec::new();
    result.push(input_vector);
    result.push(output_vector);
    return result;
}


// TBD: DO addition with given textual description e.g. 1+1
pub fn simple_rnn2() {
    let varmap = VarMap::new();
    let dev = candle_core::Device::cuda_if_available(0).unwrap();
    let generateddata: Vec<Vec<Vec<f32>>> = generate_sum_pair(100);
    let x_tmp: &Vec<Vec<f32>> = generateddata.get(0).unwrap();
    let y_tmp: &Vec<Vec<f32>> = generateddata.get(1).unwrap();

    let _xsize : usize = x_tmp.len();
    let x = to_tensor(x_tmp, &dev);
    let y = to_tensor(y_tmp, &dev);

    let mut layers: Vec<Box<dyn Trainable>> = vec![];
    let mut name1 = String::new();
    name1.push_str("rnn1");
    layers.push(Box::new(Recurrent::new(RecurrentType::LSTM,2, 4, &dev, &varmap, name1 )));
    let mut name3 = String::new();
    name3.push_str("fc1");
    layers.push(Box::new(Dense::new(1, 4, Activations::Relu, &dev, &varmap, name3 )));

    let mut model = SequentialModel::new(varmap, layers);
    model.compile(Optimizers::SGD(0.01), Loss::MSE);
    model.fit(
        x, 
        y, 
        2000, 
        true);
    
    let x_test: [[f32; 2]; 1] = [ [2., 1.] ];
    let prediction = model.predict(&Tensor::new(&x_test, &dev).unwrap()).unwrap();
    println!("prediction: {}", prediction.get(0).unwrap().clone() );
}```

<a id="examples-simple_s2s"></a>

#### simple_s2s

Selector entries: **Simple S2S**. [Source](../lib/examples/simple_s2s.rs).

<!-- source: ../lib/examples/simple_s2s.rs -->
```rust

use crate::embeddingtypes::EmbeddingType;
#[allow(unused)]
use crate::prelude::*;
use flatten::embeddinglayer::Embed;
use flatten::embeddinglayer::EmbeddingLayerTrait;
use ndarray_rand::rand_distr::num_traits::ToPrimitive;
use std::collections::HashMap;

/** This example implements initial parts of the steps given on the following site:
 *  https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html 
 */

pub fn simple_s2s() {
    let sentence = "Life is short, eat dessert first";
    let sentence_binding = sentence.replace(",", "");
    let sentence_splitted: Vec<&str> = sentence_binding.split(" ").collect();
    assert_eq!(sentence_splitted, ["Life", "is", "short", "eat", "dessert", "first"]);
    let mut sentence_splitted_sorted = sentence_splitted.clone();

    // ***************************************************
    // #### 1. Embedding an Input Sentence

    // To avoid a simple increasing order do a alphanumerical sorting 
    sentence_splitted_sorted.sort();
    assert_eq!(sentence_splitted_sorted, ["Life", "dessert", "eat", "first", "is", "short"]);
    let mut dc: HashMap<&str, usize> = HashMap::new();
    for (pos, e) in sentence_splitted_sorted.iter().enumerate() {
        dc.insert(e, pos);
    }
    let mut resulttensor : Vec<u32> = Vec::new();
    for (_pos, e) in sentence_splitted.iter().enumerate() {
        let rst = dc.get(e).clone().unwrap();
        resulttensor.push(rst.clone().to_u32().unwrap());
    }
    let varmap = VarMap::new();
    let dev = candle_core::Device::cuda_if_available(0).unwrap();

    let mut name1 = String::new();
    name1.push_str("embed0");
    let inputtensor : Tensor = Tensor::from_vec(resulttensor, (1,6), &dev ).unwrap().clone();
    let embedding_oepration = Embed::new(EmbeddingType::Standard, 6, 16, &dev , &varmap, name1);
    let embedded_sentence : Tensor = embedding_oepration.forward(inputtensor);
    println!("{}",embedded_sentence.to_string());
    let d = embedded_sentence.dims().to_vec()[2];

    // ***************************************************
    // #### 2. Computing the Unnormalized Attention Weights

    let d_q: usize = 24;
    let d_k: usize = 24;
    let d_v: usize = 28;

    let w_query = Tensor::rand(0.0, 1.0, (d_q,d), &dev).unwrap().to_dtype(DType::F32).unwrap();
    let w_key = Tensor::rand(0.0, 1.0, (d_k,d), &dev).unwrap().to_dtype(DType::F32).unwrap();
    let w_value = Tensor::rand(0.0, 1.0, (d_v,d), &dev).unwrap().to_dtype(DType::F32).unwrap();

    let embedded_sentence_vector = embedded_sentence.to_vec3::<f32>().unwrap();

    let mut query_attention_weight = Vec::new();
    for  (_pos, e) in embedded_sentence_vector[0].iter().enumerate(){
        let word: Tensor = Tensor::from_vec(e.clone(), (d,1), &dev ).unwrap().clone();
        query_attention_weight.push(w_query.matmul(&word).unwrap().clone());
    }
    println!("Vector: {:?}", query_attention_weight);

    let keys = w_key.matmul(&embedded_sentence.reshape( (6,16) ).unwrap().t().unwrap()).unwrap();
    let _values = w_value.matmul(&embedded_sentence.reshape( (6,16) ).unwrap().t().unwrap()).unwrap();

    // omega = w
    let mut omegas = Vec::new();
    for  (_pos, e) in query_attention_weight.iter().enumerate(){
        let omega_tmp = e.reshape( (1,24) ).unwrap().matmul(&keys).unwrap();
        omegas.push(omega_tmp.clone());
    }

}

```

<a id="examples-simple_tnn"></a>

#### simple_tnn

Selector entries: **Simple TNN**. [Source](../lib/examples/simple_tnn.rs).

<!-- source: ../lib/examples/simple_tnn.rs -->
```rust

#[allow(unused)]
use crate::prelude::*;
use crate::{preprocessing::features::{Features, FeaturesTrait}, recurrenttypes::RecurrentType};
use ndarray_rand::rand_distr::num_traits::ToPrimitive;
use rand::distributions::Distribution;
use crate::preprocessing;

/** This example base on the idea of the following site: https://github.com/javierlorenzod/pytorch-attention-mechanism
 * 
 * A sequence of numbers has given delimiter e.g., 0. The numbers after the delimiter will be added e.g.,
 * 
 * 1 2 3 4 0 5 0 7. 
 * 
 * 5+7 = 12 
 */

pub struct TNNDataitem {
    x: Vec<usize>,
    y: usize
}

pub fn generatedata(sizeofsequence: usize, numofelements: usize, delimiter: f32) -> Vec<TNNDataitem> {
    let mut result: Vec<TNNDataitem> = vec![];

    let vals: Vec<u64> = (0..numofelements as u64).collect();
    for (_i, _value) in vals.iter().enumerate() {
        let index_1 = Uniform::new(0, sizeofsequence/ 2);
        let index_2 = Uniform::new(sizeofsequence/2, sizeofsequence);
   
        let mut rng = rand::thread_rng();
        let a = index_1.sample(&mut rng);
        let b = index_2.sample(&mut rng);


        let mut resultelement =  TNNDataitem {
            x: Vec::new(),
            y : a+b
        };

        let vals2: Vec<u64> = (0..sizeofsequence as u64).collect();
        for (_j, value2) in vals2.iter().enumerate() {

            if value2.to_usize() == Some(a) || value2.to_usize() == Some(b) {
                resultelement.x.push(delimiter as usize);
            }
            else{
                resultelement.x.push(value2.to_usize().unwrap());
            }
        }
        
        result.push(resultelement);
    }
    return result;
}


pub fn to_tensor(input: &Vec<Vec<f32>>, device: &Device) -> Tensor{
    let dimension1: usize = input.len();
    let dimension2: usize = input.get(0).unwrap().len();
    let mut result = Vec::new();
    for i in 0..dimension1 {
        for j in 0..dimension2 {
            let val = input.get(i).unwrap().get(j).unwrap();
            result.push(val.clone().to_owned());
        }
    }
    return Tensor::from_vec(result, (dimension1,1,dimension2), device ).unwrap().clone();
}

pub fn simple_tnn() {
    let sizeofsequence = 20;
    let numofelements = 20000;

    let varmap = VarMap::new();
    let dev = candle_core::Device::cuda_if_available(0).unwrap();
    let dataset = generatedata(sizeofsequence, numofelements, 0.0);

    let mut featurehelper_x = Features::new(dev.clone());
    let mut featurehelper_y = Features::new(dev.clone());

    for (_j, value) in dataset.iter().enumerate() {
        let tmp_x_value = value.x.iter().filter_map( |s| s.to_f32() ) .collect();
        featurehelper_x.add_feature_1_d(tmp_x_value);

        let mut tmp_y_value: Vec<f32> = Vec::new();
        tmp_y_value.push(value.y.to_f32().unwrap());
        featurehelper_y.add_feature_1_d(tmp_y_value);
    }


    let mut layers: Vec<Box<dyn Trainable>> = vec![];

    let mut name1 = String::new();
    name1.push_str("rnn1");
    layers.push(Box::new(Recurrent::new(RecurrentType::LSTM,sizeofsequence, sizeofsequence, &dev, &varmap, name1 )));

    let mut name2 = String::new();
    name2.push_str("attention1");
    // Query dim tells us what the attention sees from the given sequence 
    layers.push(Box::new(SelfAttention::new( 1, 1, sizeofsequence, sizeofsequence, &dev, &varmap,  name2)));
    
    let mut name3 = String::new();
    name3.push_str("fc1");
    layers.push(Box::new(Dense::new(4, sizeofsequence, Activations::Relu, &dev, &varmap, name3 )));
    
    let mut name4 = String::new();
    name4.push_str("fc2");
    layers.push(Box::new(Dense::new(1, 4, Activations::Relu, &dev, &varmap, name4 )));

    let mut model = SequentialModel::new(varmap, layers);
    model.compile(Optimizers::Adam(0.0005), Loss::MSE);       

    let numbers: Vec<f32> = (0..=100).map(|x| x as f32).collect();
    let scaling = preprocessing::featurescaling::FeatureScaling::new(Tensor::new( numbers, &dev).unwrap());

    let tmp_x = featurehelper_x.get_data_tensor();
    let tmp_y = featurehelper_y.get_data_tensor();

    model.fit( 
        scaling.min_max_normalization_other(tmp_x), 
        scaling.min_max_normalization_other(tmp_y), 
        10, 
        false);
    

    
    let mut featurehelper_x_test = Features::new(dev.clone());
    let x_test: [f32; 20] = [1., 2., 3., 4., 0., 6., 7., 8., 9., 0. ,11., 12.,12., 13., 14., 25., 16., 17., 18., 19., ];
    let _tmp_tensor = Tensor::new(&x_test, &dev).unwrap();
    featurehelper_x_test.add_feature(_tmp_tensor);

    let tmp_tensor = scaling.min_max_normalization_other(featurehelper_x_test.get_data_tensor());
    let prediction = model.predict(&tmp_tensor).unwrap();
    
    // 6 + 11 = 17 
    println!("Done {}", scaling.min_max_normalization_reverse( prediction.get(0).unwrap().clone() ) );
}

```

<a id="examples-simple_enn"></a>

#### simple_enn

Selector entries: **Simple ENN**. [Source](../lib/examples/simple_enn.rs).

<!-- source: ../lib/examples/simple_enn.rs -->
```rust

#[allow(unused)]
use crate::prelude::*;
use crate::{layer, preprocessing::features::{Features, FeaturesTrait}};
use layer::sparsemoe::{SparseMoE, SparseMoETrait};
use ndarray_rand::rand_distr::num_traits::ToPrimitive;
use rand::distributions::Distribution;
use crate::preprocessing;

/** This example base on the idea of the following site: https://github.com/javierlorenzod/pytorch-attention-mechanism
 * 
 * A sequence of numbers has given delimiter e.g., 0. The numbers after the delimiter will be added e.g.,
 * 
 * 1 2 3 4 0 5 0 7. 
 * 
 * 5+7 = 12 
 */

pub struct ENNDataitem {
    x: Vec<usize>,
    y: usize
}

pub fn generatedata(sizeofsequence: usize, numofelements: usize, delimiter: f32) -> Vec<ENNDataitem> {
    let mut result: Vec<ENNDataitem> = vec![];

    let vals: Vec<u64> = (0..numofelements as u64).collect();
    for (_i, _value) in vals.iter().enumerate() {
        let index_1 = Uniform::new(0, sizeofsequence/ 2);
        let index_2 = Uniform::new(sizeofsequence/2, sizeofsequence);
   
        let mut rng = rand::thread_rng();
        let a = index_1.sample(&mut rng);
        let b = index_2.sample(&mut rng);


        let mut resultelement =  ENNDataitem {
            x: Vec::new(),
            y : a+b
        };

        let vals2: Vec<u64> = (0..sizeofsequence as u64).collect();
        for (_j, value2) in vals2.iter().enumerate() {

            if value2.to_usize() == Some(a) || value2.to_usize() == Some(b) {
                resultelement.x.push(delimiter as usize);
            }
            else{
                resultelement.x.push(value2.to_usize().unwrap());
            }
        }
        
        result.push(resultelement);
    }
    return result;
}


pub fn to_tensor(input: &Vec<Vec<f32>>, device: &Device) -> Tensor{
    let dimension1: usize = input.len();
    let dimension2: usize = input.get(0).unwrap().len();
    let mut result = Vec::new();
    for i in 0..dimension1 {
        for j in 0..dimension2 {
            let val = input.get(i).unwrap().get(j).unwrap();
            result.push(val.clone().to_owned());
        }
    }
    return Tensor::from_vec(result, (dimension1,1,dimension2), device ).unwrap().clone();
}

pub fn simple_enn() {
    let sizeofsequence = 20;
    let numofelements = 20000;

    let varmap = VarMap::new();
    let dev = candle_core::Device::cuda_if_available(0).unwrap();
    let dataset = generatedata(sizeofsequence, numofelements, 0.0);

    let mut featurehelper_x = Features::new(dev.clone());
    let mut featurehelper_y = Features::new(dev.clone());

    for (_j, value) in dataset.iter().enumerate() {
        let tmp_x_value = value.x.iter().filter_map( |s| s.to_f32() ) .collect();
        featurehelper_x.add_feature_1_d(tmp_x_value);

        let mut tmp_y_value: Vec<f32> = Vec::new();
        tmp_y_value.push(value.y.to_f32().unwrap());
        featurehelper_y.add_feature_1_d(tmp_y_value);
    }


    let mut layers: Vec<Box<dyn Trainable>> = vec![];

    let mut name2 = String::new();
    name2.push_str("attention1");
    // Query dim tells us what the attention sees from the given sequence 
    layers.push(Box::new(SparseMoE::new(10, sizeofsequence, sizeofsequence, &dev, &varmap,  name2)));
    
    let mut name3 = String::new();
    name3.push_str("fc1");
    layers.push(Box::new(Dense::new(4, sizeofsequence, Activations::Relu, &dev, &varmap, name3 )));
    
    let mut name4 = String::new();
    name4.push_str("fc2");
    layers.push(Box::new(Dense::new(1, 4, Activations::Relu, &dev, &varmap, name4 )));

    let mut model = SequentialModel::new(varmap, layers);
    model.compile(Optimizers::Adam(0.0005), Loss::MSE);       

    let numbers: Vec<f32> = (0..=100).map(|x| x as f32).collect();
    let scaling = preprocessing::featurescaling::FeatureScaling::new(Tensor::new( numbers, &dev).unwrap());

    let tmp_x = featurehelper_x.get_data_tensor();
    let tmp_y = featurehelper_y.get_data_tensor();

    model.fit( 
        scaling.min_max_normalization_other(tmp_x), 
        scaling.min_max_normalization_other(tmp_y), 
        10, 
        false);
    

    
    let mut featurehelper_x_test = Features::new(dev.clone());
    let x_test: [f32; 20] = [1., 2., 3., 4., 0., 6., 7., 8., 9., 0. ,11., 12.,12., 13., 14., 25., 16., 17., 18., 19., ];
    let _tmp_tensor = Tensor::new(&x_test, &dev).unwrap();
    featurehelper_x_test.add_feature(_tmp_tensor);

    let tmp_tensor = scaling.min_max_normalization_other(featurehelper_x_test.get_data_tensor());
    let prediction = model.predict(&tmp_tensor).unwrap();
    
    // 6 + 11 = 17 
    println!("Done {}", scaling.min_max_normalization_reverse( prediction.get(0).unwrap().clone() ) );
}

```

<a id="examples-simple_pnn"></a>

#### simple_pnn

Selector entries: **Simple PNN**. [Source](../lib/examples/simple_pnn.rs).

<!-- source: ../lib/examples/simple_pnn.rs -->
```rust

#[allow(unused)]
use crate::prelude::*;
use crate::preprocessing::features::{Features, FeaturesTrait};
use ndarray_rand::rand_distr::num_traits::ToPrimitive;
use rand::distributions::Distribution;
use crate::preprocessing;

/** This example base on the idea of the following site: https://github.com/javierlorenzod/pytorch-attention-mechanism
 * 
 * A sequence of numbers has given delimiter e.g., 0. The numbers after the delimiter will be added e.g.,
 * 
 * 1 2 3 4 0 5 0 7. 
 * 
 * 5+7 = 12 
 */

pub struct ENNDataitem {
    x: Vec<usize>,
    y: usize
}

pub fn generatedata(sizeofsequence: usize, numofelements: usize, delimiter: f32) -> Vec<ENNDataitem> {
    let mut result: Vec<ENNDataitem> = vec![];

    let vals: Vec<u64> = (0..numofelements as u64).collect();
    for (_i, _value) in vals.iter().enumerate() {
        let index_1 = Uniform::new(0, sizeofsequence/ 2);
        let index_2 = Uniform::new(sizeofsequence/2, sizeofsequence);
   
        let mut rng = rand::thread_rng();
        let a = index_1.sample(&mut rng);
        let b = index_2.sample(&mut rng);


        let mut resultelement =  ENNDataitem {
            x: Vec::new(),
            y : a+b
        };

        let vals2: Vec<u64> = (0..sizeofsequence as u64).collect();
        for (_j, value2) in vals2.iter().enumerate() {

            if value2.to_usize() == Some(a) || value2.to_usize() == Some(b) {
                resultelement.x.push(delimiter as usize);
            }
            else{
                resultelement.x.push(value2.to_usize().unwrap());
            }
        }
        
        result.push(resultelement);
    }
    return result;
}


pub fn to_tensor(input: &Vec<Vec<f32>>, device: &Device) -> Tensor{
    let dimension1: usize = input.len();
    let dimension2: usize = input.get(0).unwrap().len();
    let mut result = Vec::new();
    for i in 0..dimension1 {
        for j in 0..dimension2 {
            let val = input.get(i).unwrap().get(j).unwrap();
            result.push(val.clone().to_owned());
        }
    }
    return Tensor::from_vec(result, (dimension1,1,dimension2), device ).unwrap().clone();
}

pub fn simple_pnn() {
    let sizeofsequence = 20;
    let numofelements = 20000;

    let varmap = VarMap::new();
    let dev = candle_core::Device::cuda_if_available(0).unwrap();
    let dataset = generatedata(sizeofsequence, numofelements, 0.0);

    let mut featurehelper_x = Features::new(dev.clone());
    let mut featurehelper_y = Features::new(dev.clone());

    for (_j, value) in dataset.iter().enumerate() {
        let tmp_x_value = value.x.iter().filter_map( |s| s.to_f32() ) .collect();
        featurehelper_x.add_feature_1_d(tmp_x_value);

        let mut tmp_y_value: Vec<f32> = Vec::new();
        tmp_y_value.push(value.y.to_f32().unwrap());
        featurehelper_y.add_feature_1_d(tmp_y_value);
    }


    let mut layers: Vec<Box<dyn Trainable>> = Vec::new();


    let mut name1 = String::new();
    name1.push_str("fc0");
    layers.push(Box::new(Dense::new(sizeofsequence, sizeofsequence, Activations::Relu, &dev, &varmap, name1 )));


    let mut name2 = String::new();
    name2.push_str("split");


    let mut name2_1 = String::new();
    name2_1.push_str("split_fc1");
    let mut name4 = String::new();
    name4.push_str("fc2");

    let mut layers_parallel: Vec<Box<dyn Trainable>> = Vec::new();

    let mut name2_1 = String::new();
    name2_1.push_str("split_fc1");
    let mut name2_2 = String::new();
    name2_2.push_str("split_fc2");

    let varmap2 = VarMap::new();
    layers_parallel.push(Box::new(Dense::new(sizeofsequence, sizeofsequence, Activations::Relu, &dev, &varmap2, name2_1 )));
    layers_parallel.push(Box::new(Dense::new(sizeofsequence, sizeofsequence, Activations::Relu, &dev, &varmap2, name2_2 )));

    layers.push(Box::new(ParallelModel::new(ParallelModelType::Split, &dev, varmap2, layers_parallel)));

    // FIX ME
    let varmap3 = VarMap::new();
    let layers_parallel2: Vec<Box<dyn Trainable>> = Vec::new();
    layers.push(Box::new(ParallelModel::new(ParallelModelType::Merge, &dev, varmap3, layers_parallel2)));

    let mut name3 = String::new();
    name3.push_str("fc1");
    layers.push(Box::new(Dense::new(4, sizeofsequence, Activations::Relu, &dev, &varmap, name3 )));

    let mut name4 = String::new();
    name4.push_str("fc2");
    layers.push(Box::new(Dense::new(1, 4, Activations::Relu, &dev, &varmap, name4 )));

    let mut model = SequentialModel::new(varmap, layers);
    model.compile(Optimizers::Adam(0.0005), Loss::MSE);       

    let numbers: Vec<f32> = (0..=100).map(|x| x as f32).collect();
    let scaling = preprocessing::featurescaling::FeatureScaling::new(Tensor::new( numbers, &dev).unwrap());

    let tmp_x = featurehelper_x.get_data_tensor();
    let tmp_y = featurehelper_y.get_data_tensor();

    model.fit( 
        scaling.min_max_normalization_other(tmp_x), 
        scaling.min_max_normalization_other(tmp_y), 
        10, 
        false);

    
    let mut featurehelper_x_test = Features::new(dev.clone());
    let x_test: [f32; 20] = [1., 2., 3., 4., 0., 6., 7., 8., 9., 0. ,11., 12.,12., 13., 14., 25., 16., 17., 18., 19., ];
    let _tmp_tensor = Tensor::new(&x_test, &dev).unwrap();
    featurehelper_x_test.add_feature(_tmp_tensor);

    let tmp_tensor = scaling.min_max_normalization_other(featurehelper_x_test.get_data_tensor());
    let prediction = model.predict(&tmp_tensor).unwrap();
    
    // 6 + 11 = 17 
    println!("Done {}", scaling.min_max_normalization_reverse( prediction.get(0).unwrap().clone() ) );
}

```

<a id="examples-simple_llm"></a>

#### simple_llm

Selector entries: **Simple LLM, Simple LLM2**. [Source](../lib/examples/simple_llm.rs).

<!-- source: ../lib/examples/simple_llm.rs -->
```rust

use std::{collections::HashMap};

#[allow(unused)]
use crate::prelude::*;
use crate::{preprocessing::features::{Features, FeaturesTrait}, saveweightstype::SaveWeightsType};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use selfattention::flatten::transformers::{tokeoutputstream::TokenOutputStream, transformermodels::TransformerTrait};
use tokenizers::PaddingParams;

use self::selfattention::flatten::transformers::{berthiddenacttype::BertHiddenActType, bertmodel::{LighterBertModel, LighterBertModelTrait}, bertpositionembdtypes::BertPositionEmbeddingType, llamamodel::{LighterLLamaModel, LighterLLamaModelTrait}};


pub fn to_tensor(input: &Vec<Vec<f32>>, device: &Device) -> Tensor{
    let dimension1: usize = input.len();
    let dimension2: usize = input.get(0).unwrap().len();
    let mut result = Vec::new();
    for i in 0..dimension1 {
        for j in 0..dimension2 {
            let val = input.get(i).unwrap().get(j).unwrap();
            result.push(val.clone().to_owned());
        }
    }
    return Tensor::from_vec(result, (dimension1,1,dimension2), device ).unwrap().clone();
}
pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}

pub fn simple_llm() {

    let varmap = VarMap::new();
    let dev = candle_core::Device::cuda_if_available(0).unwrap();
    
    // Adapted from https://github.com/huggingface/candle/blob/a0facd0e67b546215ea62b53dc28a1cb2e6dcd47/candle-examples/examples/bert/main.rs
    let sentences = [
            "The cat sits outside",
            "A man is playing guitar",
            "I love pasta",
            "The new movie is awesome",
            "The cat plays in the garden",
            "A woman watches TV",
            "The new movie is so great",
            "Do you like pizza?",
     ];
    let pp = PaddingParams {
        strategy: tokenizers::PaddingStrategy::BatchLongest,
        ..Default::default()
    };

    let _bert_model = LighterBertModel::new(
        30522, 
        768, 
        12, 
        12, 
        3072, 
        BertHiddenActType::Gelu, 
        0.1, 
        512, 
        2, 
        0.02, 
        1e-12, 
        0, 
        BertPositionEmbeddingType::Absolute, 
        &dev, 
        &varmap, 
        "bert1".to_string());

    let mut parameter = HashMap::new();
    parameter.insert("hf_token".into(), Value::String("<YOUR TOKEN>".into()));
    parameter.insert("hf_model".into(), Value::String("sentence-transformers/all-MiniLM-L6-v2".into()));
    // Or your pathes after first download
    // parameter.insert("hf_model.safetensors".into(), Value::String("<YOUR PATH>".into()));
    // parameter.insert("hf_tokenizer.json".into(), Value::String("<YOUR PATH>".into()));

    

    let mut _tokenizer =  _bert_model.get_tokenizer(&parameter);    

    _tokenizer.with_padding(Some(pp));
    let tokens = _tokenizer
        .encode_batch(sentences.to_vec(), true).unwrap();
    let token_ids = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec();
            Ok(Tensor::new(tokens.as_slice(), &dev)?)
        })
        .collect::<Result<Vec<_>>>().unwrap();

    _bert_model.load_weights(SaveWeightsType::HuggingfaceHub, &parameter, &varmap, &dev); 

    let mut featurehelper_x_test = Features::new(dev.clone());
    for element in token_ids{
        featurehelper_x_test.add_feature(element);
    }
   
    let input = &featurehelper_x_test.get_data_tensor();
    let result = _bert_model.predict(input).unwrap();
    let embeddings = result.get(0).unwrap();
    let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3().unwrap();
    let n_sentences = sentences.len();
    let embeddings = (embeddings.sum(1).unwrap() / (n_tokens as f64)).unwrap();
   
    let mut similarities = vec![];
    for i in 0..n_sentences {
        let e_i = embeddings.get(i).unwrap();
        for j in (i + 1)..n_sentences {
            let e_j = embeddings.get(j).unwrap();
            let sum_ij = (&e_i * &e_j).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
            let sum_i2 = (&e_i * &e_i).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
            let sum_j2 = (&e_j * &e_j).unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap();
            let cosine_similarity = sum_ij / (sum_i2 * sum_j2).sqrt();
            similarities.push((cosine_similarity, i, j))
        }
    }
    similarities.sort_by(|u, v| v.0.total_cmp(&u.0));
    for &(score, i, j) in similarities[..5].iter() {
        println!("score: {score:.2} '{}' '{}'", sentences[i], sentences[j])
    }

}


// https://huggingface.co/tasks
// one task is https://huggingface.co/tasks/text-generation
pub fn simple_llm2() {

    let varmap = VarMap::new();
    let dev = candle_core::Device::cuda_if_available(0).unwrap();
    
    // TinyLlama-1.1B-Chat-v1.0 
    let llamamodel = LighterLLamaModel::new(
        0.85,
        20,
        3.2,
        0.7,
        50,
        32000, 
        768, 
        3072, 
        12, 
        12, 
        12, 
        1e-06, 
        1.0, 
        &dev, 
        "llama1".to_string());

    let mut parameter = HashMap::new();
    parameter.insert("hf_token".into(), Value::String("<YOUR TOKEN>".into()));
    parameter.insert("hf_model".into(), Value::String("JackFram/llama-160m".into()));
    // Or your pathes after first download
    // parameter.insert("hf_model.safetensors".into(), Value::String("<YOUR PATH>".into()));
    // parameter.insert("hf_tokenizer.json".into(), Value::String("<YOUR PATH>".into()));

    let mut _tokenizer =  llamamodel.get_tokenizer(&parameter);    
    llamamodel.load_weights(SaveWeightsType::HuggingfaceHub, &parameter, &varmap, &dev); 

    let pp = PaddingParams {
        strategy: tokenizers::PaddingStrategy::BatchLongest,
        ..Default::default()
    };
    _tokenizer.with_padding(Some(pp));

    let tokens = _tokenizer.encode("The world is ", true).unwrap();
    let mut token_ids = tokens.get_ids().to_vec();
    println!("Given query: {}",_tokenizer.decode(&token_ids, true).unwrap());

    let mut _tokenizer_stream = TokenOutputStream::new(_tokenizer.clone());

    let mut featurehelper_x_test = Features::new(dev.clone());
    let _tmp: Vec<f32> = token_ids.iter().map(|&e| e as f32).collect();
    let _tmp2= Tensor::new(_tmp, &dev).unwrap();
    featurehelper_x_test.add_feature(_tmp2);


    let input = &featurehelper_x_test.get_data_tensor();
    let result = llamamodel.predict(input).unwrap();

    let vv= result.get(0).unwrap().flatten_all().unwrap().to_vec1().unwrap();


    let mut string = String::new();
    for _token in vv {
        if let t = _tokenizer_stream.next_token(_token).unwrap() {
            if t.clone() != None {
                string.push_str(&t.unwrap());
            }
        }
    }
    println!("Given answer: {}", string);
   
}

```


<a id="examples-graph2textrs"></a>

### graph2text.rs

<!-- source: graph2text.rs -->
```rust
//! cargo run --no-default-features --example graph2text -- [graph.json] [output.json]
use candlelighter::graph2text::{verbalize, Graph, Options, Triple};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() > 2 {
        return Err("usage: graph2text [graph.json] [output.json]".into());
    }
    let graph = if let Some(path) = args.first() {
        Graph::from_json(&std::fs::read_to_string(path)?)?
    } else {
        Graph::from_triples(&[
            Triple {
                subject: "Ada Lovelace".into(),
                predicate: "wrote notes about".into(),
                object: "Analytical Engine".into(),
            },
            Triple {
                subject: "Charles Babbage".into(),
                predicate: "designed".into(),
                object: "Analytical Engine".into(),
            },
        ])?
    };
    let output = verbalize(&graph, &Options::default())?;
    if let Some(path) = args.get(1) {
        std::fs::write(path, serde_json::to_string_pretty(&output)?)?;
    } else {
        println!("{}", output.text);
    }
    Ok(())
}
```

<a id="examples-graph2text_nativers"></a>

### graph2text_native.rs

<!-- source: graph2text_native.rs -->
```rust
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
```

<a id="examples-g-retriever-python-implementation-and-examples"></a>

### G-Retriever Python implementation and examples

These listings adapt the [MIT-licensed upstream G-Retriever](../third_party/g_retriever/LICENSE). See the [integration guide](../docs/handbook.md#retriever) for installation, pretrained models and behavior differences.

<a id="examples-corepy"></a>

#### core.py

<!-- source: python/g_retriever/core.py -->
```python
"""Adapted from XiaoxinHe/G-Retriever (MIT, copyright 2024 Xiaoxin He).

PCST retrieval follows src/dataset/utils/retrieval.py; graph soft prompting follows
src/model/graph_llm.py. Original source and license: third_party/g_retriever/.
"""
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

# pcst-fast 1.0.10 wheels return corrupted index arrays with NumPy 2.x.
if int(np.__version__.split(".")[0]) >= 2:
    raise ImportError("G-Retriever requires numpy>=1.26,<2 for pcst-fast 1.0.10")

from pcst_fast import pcst_fast
import torch
from torch import nn
from torch_geometric.data import Batch, Data

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("lighter_upstream_gnn", ROOT / "third_party/g_retriever/src/model/gnn.py")
_gnn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gnn)


class HashEncoder:
    """Deterministic lexical demo embeddings, not pretrained semantic embeddings."""
    def __init__(self, dimension=64):
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.dimension = dimension

    def encode(self, texts):
        output = torch.zeros(len(texts), self.dimension)
        for row, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                index = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "little") % self.dimension
                output[row, index] += 1
        return nn.functional.normalize(output, dim=-1)


class SentenceEncoder:
    """Hugging Face encoder with masked mean pooling, as in upstream preprocessing."""
    def __init__(self, model="sentence-transformers/all-roberta-large-v1", device="cpu"):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).to(device).eval()
        self.dimension = self.model.config.hidden_size
        self.device = device

    @torch.no_grad()
    def encode(self, texts, batch_size=32):
        if not texts:
            return torch.empty(0, self.dimension)
        batches = []
        for start in range(0, len(texts), batch_size):
            inputs = self.tokenizer(texts[start:start + batch_size], padding=True, truncation=True, return_tensors="pt").to(self.device)
            hidden = self.model(**inputs).last_hidden_state
            mask = inputs.attention_mask.unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            batches.append(nn.functional.normalize(pooled, dim=-1).cpu())
        return torch.cat(batches)


@dataclass
class Retrieval:
    graph: Data
    property_graph: dict
    description: str
    node_ids: list
    edge_ids: list
    node_prizes: list
    edge_prizes: list
    omitted_node_ids: list
    omitted_edge_ids: list

    def metadata(self):
        return {key: getattr(self, key) for key in ("node_ids", "edge_ids", "node_prizes", "edge_prizes", "omitted_node_ids", "omitted_edge_ids")}


def _validate(graph):
    if not isinstance(graph, dict) or set(graph) - {"nodes", "edges"}:
        raise ValueError("graph must contain only nodes and edges")
    nodes, edges = graph.get("nodes"), graph.get("edges")
    if not isinstance(nodes, list) or not nodes or not isinstance(edges, list):
        raise ValueError("graph requires a nonempty nodes array and an edges array")
    def text(value):
        return isinstance(value, str) and bool(value.strip())
    node_ids = set()
    for node in nodes:
        if not isinstance(node, dict) or set(node) - {"id", "label", "properties"} or not text(node.get("id")) or not text(node.get("label")) or node["id"] in node_ids:
            raise ValueError("invalid or duplicate node")
        node_ids.add(node["id"])
    edge_ids = set()
    for edge in edges:
        if not isinstance(edge, dict) or set(edge) - {"id", "source", "relation", "target", "properties"} or not text(edge.get("id")) or not text(edge.get("relation")) or edge["id"] in edge_ids:
            raise ValueError("invalid or duplicate edge")
        if not text(edge.get("source")) or not text(edge.get("target")) or edge["source"] not in node_ids or edge["target"] not in node_ids:
            raise ValueError("edge endpoint is absent")
        edge_ids.add(edge["id"])
    for item in nodes + edges:
        properties = item.get("properties", {})
        if not isinstance(properties, dict) or any(not text(k) for k in properties):
            raise ValueError("properties require nonblank keys")
        json.dumps(properties, allow_nan=False)
    return nodes, edges


def retrieve(graph, question, encoder, topk=3, topk_edges=3, edge_cost=0.5):
    """Cosine rank prizes, virtual edge-prize nodes and unrooted GW PCST pruning.

    Solve connectivity as undirected, then restore original directed relations.
    Returned embeddings are CPU tensors, with original IDs recorded separately.
    """
    nodes, edges = _validate(graph)
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be nonblank")
    if any(isinstance(k, bool) or not isinstance(k, int) or k < 0 for k in (topk, topk_edges)) or not np.isfinite(edge_cost) or edge_cost < 0:
        raise ValueError("rank counts must be nonnegative integers and edge cost finite/nonnegative")
    if topk == 0 and (topk_edges == 0 or not edges):
        raise ValueError("at least one prize source must be enabled")
    node_text = [n["label"] + " " + json.dumps(n.get("properties", {}), sort_keys=True, ensure_ascii=False) for n in nodes]
    edge_text = [e["relation"] + " " + json.dumps(e.get("properties", {}), sort_keys=True, ensure_ascii=False) for e in edges]
    x = torch.as_tensor(encoder.encode(node_text)).detach().cpu().float()
    query = torch.as_tensor(encoder.encode([question])).detach().cpu().float()
    attrs = torch.as_tensor(encoder.encode(edge_text)).detach().cpu().float()
    dimension = x.shape[1] if x.ndim == 2 else 0
    if x.shape != (len(nodes), dimension) or dimension == 0 or query.shape != (1, dimension) or attrs.shape != (len(edges), dimension) or not all(torch.isfinite(t).all() for t in (x, query, attrs)):
        raise ValueError("encoder must return finite, equally sized embedding rows")
    indices = {n["id"]: i for i, n in enumerate(nodes)}
    original_edges = np.asarray([(indices[e["source"]], indices[e["target"]]) for e in edges], dtype=np.int64).reshape(-1, 2)
    nprizes = np.zeros(len(nodes), dtype=np.float64)
    nscores = nn.functional.cosine_similarity(query, x).numpy()
    count = min(topk, len(nodes))
    # Stable index tie breaking; descending integer ranks match upstream.
    for rank, index in enumerate(np.argsort(-nscores, kind="stable")[:count]):
        nprizes[index] = count - rank
    eprizes = np.zeros(len(edges), dtype=np.float64)
    if edges and topk_edges:
        scores = nn.functional.cosine_similarity(query, attrs).numpy()
        values = np.unique(scores)[::-1][:topk_edges]
        previous = float(len(values))
        for rank, score in enumerate(values):
            members = scores == score
            prize = min((len(values) - rank) / int(members.sum()), previous)
            eprizes[members] = prize
            previous = prize * 0.99
        edge_cost = min(edge_cost, float(eprizes.max()) * 0.995)
    transformed, costs, virtual_prizes = [], [], []
    real_mapping, virtual_mapping = {}, {}
    # Nonpositive edge prizes become discounted costs; positive surplus is a
    # virtual node prize attached to both original endpoints by zero-cost edges.
    for i, (source, target) in enumerate(original_edges):
        if eprizes[i] <= edge_cost:
            real_mapping[len(transformed)] = i
            transformed.append((source, target))
            costs.append(edge_cost - eprizes[i])
    real_count = len(transformed)
    for i, (source, target) in enumerate(original_edges):
        if eprizes[i] > edge_cost:
            virtual = len(nodes) + len(virtual_prizes)
            virtual_mapping[virtual] = i
            transformed.extend([(source, virtual), (virtual, target)])
            costs.extend([0.0, 0.0])
            virtual_prizes.append(eprizes[i] - edge_cost)
    if not edges:
        # Upstream returns all nodes for edge-free graphs; preserve that behavior.
        selected_nodes, selected_edges = list(range(len(nodes))), []
    else:
        vertices, selected = pcst_fast(
            np.asarray(transformed, dtype=np.int64).reshape(-1, 2),
            np.concatenate([nprizes, virtual_prizes]), np.asarray(costs, dtype=np.float64),
            -1, 1, "gw", 0)
        selected_edges = sorted({real_mapping[int(e)] for e in selected if e < real_count} | {virtual_mapping[int(v)] for v in vertices if v >= len(nodes)})
        selected_nodes = sorted({int(v) for v in vertices if v < len(nodes)} | {int(v) for i in selected_edges for v in original_edges[i]})
    if not selected_nodes:
        raise ValueError("PCST selected no original nodes; adjust retrieval parameters")
    remap = {node: i for i, node in enumerate(selected_nodes)}
    edge_index = torch.tensor([(remap[int(original_edges[i, 0])], remap[int(original_edges[i, 1])]) for i in selected_edges], dtype=torch.long).reshape(-1, 2).T.contiguous()
    data = Data(x=x[selected_nodes], edge_index=edge_index, edge_attr=attrs[selected_edges], num_nodes=len(selected_nodes))
    node_frame = pd.DataFrame([{"node_id": nodes[i]["id"], "node_attr": nodes[i]["label"], "properties": json.dumps(nodes[i].get("properties", {}), sort_keys=True)} for i in selected_nodes])
    edge_frame = pd.DataFrame([{"edge_id": edges[i]["id"], "src": edges[i]["source"], "edge_attr": edges[i]["relation"], "dst": edges[i]["target"], "properties": json.dumps(edges[i].get("properties", {}), sort_keys=True)} for i in selected_edges], columns=["edge_id", "src", "edge_attr", "dst", "properties"])
    return Retrieval(data, {"nodes": [nodes[i] for i in selected_nodes], "edges": [edges[i] for i in selected_edges]}, node_frame.to_csv(index=False) + "\n" + edge_frame.to_csv(index=False),
        [nodes[i]["id"] for i in selected_nodes], [edges[i]["id"] for i in selected_edges],
        nprizes.tolist(), eprizes.tolist(), [n["id"] for i,n in enumerate(nodes) if i not in selected_nodes],
        [e["id"] for i,e in enumerate(edges) if i not in selected_edges])


class GraphRetriever(nn.Module):
    """Upstream GNN -> mean pool -> projector -> single learned LM soft token.

    The language model is frozen; gradients flow into the GNN and projector.
    Accepts Hugging Face causal LMs with inputs_embeds support.
    """
    def __init__(self, model, tokenizer, input_dim, hidden_dim=64, layers=2, heads=4,
                 gnn="gt", max_text_tokens=512, max_new_tokens=64):
        super().__init__()
        if gnn not in _gnn.load_gnn_model or min(input_dim, hidden_dim, heads, max_text_tokens, max_new_tokens) < 1 or layers < 2 or hidden_dim % heads:
            raise ValueError("invalid GNN dimensions, architecture or token limits")
        self.model, self.tokenizer = model, tokenizer
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()
        embedding = model.get_input_embeddings().weight
        self.graph_encoder = _gnn.load_gnn_model[gnn](input_dim, hidden_dim, hidden_dim, layers, 0.0, heads).to(device=embedding.device, dtype=torch.float32)
        self.projector = nn.Sequential(nn.Linear(hidden_dim, 2048), nn.Sigmoid(), nn.Linear(2048, embedding.shape[1])).to(embedding.device)
        self.max_text_tokens, self.max_new_tokens = max_text_tokens, max_new_tokens
        self.input_dim = input_dim
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("tokenizer needs a pad or EOS token")
            tokenizer.pad_token = tokenizer.eos_token

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()  # Keep frozen LM dropout disabled during adapter training.
        return self

    def encode_graphs(self, retrievals):
        if not retrievals or any(r.graph.num_nodes < 1 or r.graph.x.shape[1] != self.input_dim for r in retrievals):
            raise ValueError("nonempty graphs with matching input dimensions are required")
        device = self.model.get_input_embeddings().weight.device
        graphs = Batch.from_data_list([r.graph for r in retrievals]).to(device)
        # BatchNorm in the unchanged upstream GNN needs at least two node rows.
        if self.training and graphs.x.shape[0] < 2:
            raise ValueError("training requires at least two total nodes per batch")
        hidden, _ = self.graph_encoder(graphs.x, graphs.edge_index, graphs.edge_attr)
        pooled = hidden.new_zeros((len(retrievals), hidden.shape[1]))
        pooled.index_add_(0, graphs.batch, hidden)
        counts = torch.bincount(graphs.batch, minlength=len(retrievals)).clamp_min(1)
        return self.projector(pooled / counts.unsqueeze(-1))

    def _inputs(self, retrievals, questions, answers=None):
        if len(retrievals) != len(questions) or (answers is not None and len(answers) != len(questions)) or not questions or any(not isinstance(q,str) or not q.strip() for q in questions):
            raise ValueError("aligned nonempty graph/question/answer batches are required")
        embedding = self.model.get_input_embeddings()
        device, dtype = embedding.weight.device, embedding.weight.dtype
        graph_tokens = self.encode_graphs(retrievals).to(dtype=dtype)
        rows, labels = [], []
        for i, (retrieval, question) in enumerate(zip(retrievals, questions)):
            description = self.tokenizer.encode(retrieval.description, add_special_tokens=False)[:self.max_text_tokens]
            question_ids = self.tokenizer.encode("\nQuestion: " + question + "\nAnswer:", add_special_tokens=False)
            bos = [] if self.tokenizer.bos_token_id is None else [self.tokenizer.bos_token_id]
            bos_emb = embedding(torch.tensor(bos, device=device, dtype=torch.long))
            ids = description + question_ids
            target = []
            if answers is not None:
                target = self.tokenizer.encode(answers[i], add_special_tokens=False)[:self.max_new_tokens]
                if self.tokenizer.eos_token_id is not None:
                    target.append(self.tokenizer.eos_token_id)
                if not target:
                    raise ValueError("answer must tokenize to at least one target token")
            row = torch.cat([bos_emb, graph_tokens[i:i+1], embedding(torch.tensor(ids + target, device=device, dtype=torch.long))])
            rows.append(row)
            labels.append([-100] * (len(row) - len(target)) + target)
        length = max(len(row) for row in rows)
        context = getattr(self.model.config, "max_position_embeddings", None)
        if context and length + (self.max_new_tokens if answers is None else 0) > context:
            raise ValueError("graph prompt exceeds language-model context; reduce text/output limits")
        padded, masks, padded_labels = [], [], []
        pad = embedding(torch.tensor(self.tokenizer.pad_token_id, device=device))
        for row, label in zip(rows, labels):
            count = length - len(row)
            padded.append(torch.cat([pad.expand(count, -1), row]))
            masks.append([0] * count + [1] * len(row))
            padded_labels.append([-100] * count + label)
        return torch.stack(padded), torch.tensor(masks, device=device), torch.tensor(padded_labels, device=device)

    def forward(self, retrievals, questions, answers):
        embeddings, mask, labels = self._inputs(retrievals, questions, answers)
        return self.model(inputs_embeds=embeddings, attention_mask=mask, labels=labels).loss

    @torch.no_grad()
    def generate(self, retrievals, questions):
        was_training = self.training
        self.eval()
        try:
            embeddings, mask, _ = self._inputs(retrievals, questions)
            output = self.model.generate(inputs_embeds=embeddings, attention_mask=mask,
                max_new_tokens=self.max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id)
            return self.tokenizer.batch_decode(output, skip_special_tokens=True)
        finally:
            self.train(was_training)

    def save_adapter(self, path):
        torch.save({"graph_encoder": self.graph_encoder.state_dict(), "projector": self.projector.state_dict()}, path)

    def load_adapter(self, path):
        state = torch.load(path, map_location=self.model.get_input_embeddings().weight.device, weights_only=True)
        self.graph_encoder.load_state_dict(state["graph_encoder"])
        self.projector.load_state_dict(state["projector"])
```

<a id="examples-demopy"></a>

#### demo.py

<!-- source: python/g_retriever/demo.py -->
```python
"""Tiny randomly initialized Llama and byte tokenizer for offline plumbing checks."""
import torch
from transformers import LlamaConfig, LlamaForCausalLM


class ByteTokenizer:
    bos_token_id, eos_token_id, pad_token_id = 1, 2, 0
    def encode(self, text, add_special_tokens=False):
        return [byte + 3 for byte in text.encode("utf-8")]
    def batch_decode(self, rows, skip_special_tokens=True):
        return [bytes(token - 3 for token in row.tolist() if 3 <= token < 259).decode("utf-8", errors="replace") for row in rows]


def tiny_model():
    torch.manual_seed(42)
    torch.set_num_threads(2)
    config = LlamaConfig(vocab_size=259, hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=2048, bos_token_id=1, eos_token_id=2, pad_token_id=0)
    return LlamaForCausalLM(config), ByteTokenizer()
```

<a id="examples-g_retriever_retrievepy"></a>

#### g_retriever_retrieve.py

<!-- source: python/g_retriever_retrieve.py -->
```python
"""Run actual PCST retrieval without a language model."""
import argparse
import json
from g_retriever import HashEncoder, SentenceEncoder, retrieve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="examples/data/graph2text.json")
    parser.add_argument("--question", default="Who wrote notes about the Analytical Engine?")
    parser.add_argument("--encoder", help="HF/local sentence encoder; omit for offline hash demo")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--topk-edges", type=int, default=3)
    parser.add_argument("--edge-cost", type=float, default=0.5)
    parser.add_argument("--output", help="write retrieved property graph JSON (replaces existing file)")
    args = parser.parse_args()
    with open(args.graph) as source:
        graph = json.load(source)
    encoder = SentenceEncoder(args.encoder) if args.encoder else HashEncoder()
    result = retrieve(graph, args.question, encoder, args.topk, args.topk_edges, args.edge_cost)
    if args.output:
        with open(args.output, "w") as output:
            json.dump(result.property_graph, output, indent=2)
    print(result.description)
    print(json.dumps(result.metadata(), indent=2))


if __name__ == "__main__":
    main()
```

<a id="examples-g_retriever_trainpy"></a>

#### g_retriever_train.py

<!-- source: python/g_retriever_train.py -->
```python
"""Train graph encoder/projector on JSONL graph/question/answer records; frozen LM."""
import argparse
import json
import torch
from g_retriever import GraphRetriever, HashEncoder, SentenceEncoder, retrieve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="examples/data/g_retriever_train.jsonl")
    parser.add_argument("--model", help="HF/local causal LM; omit for offline random-model demo")
    parser.add_argument("--encoder", help="HF/local sentence encoder; omit for hash demo")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--resume", help="adapter checkpoint to continue training")
    parser.add_argument("--output", required=True, help="output adapter checkpoint (replaces existing file)")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.learning_rate < float("inf"):
        parser.error("epochs, batch size and finite learning rate must be positive")
    torch.manual_seed(42)
    with open(args.data) as source:
        records = [json.loads(line) for line in source if line.strip()]
    if not records or any(set(r) != {"graph", "question", "answer"} or not isinstance(r["answer"], str) or not r["answer"].strip() for r in records):
        parser.error("JSONL records must contain graph, question and nonempty answer")
    encoder = SentenceEncoder(args.encoder) if args.encoder else HashEncoder()
    retrievals = [retrieve(r["graph"], r["question"], encoder) for r in records]
    if args.model:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    else:
        from g_retriever.demo import tiny_model
        model, tokenizer = tiny_model()
        print("Offline random-model training demo; not a pretrained QA system.")
    retriever = GraphRetriever(model, tokenizer, encoder.dimension, hidden_dim=args.hidden_dim)
    if args.resume:
        retriever.load_adapter(args.resume)
    optimizer = torch.optim.AdamW([p for p in retriever.parameters() if p.requires_grad], lr=args.learning_rate)
    retriever.train()
    for epoch in range(args.epochs):
        total = 0.0
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            optimizer.zero_grad()
            loss = retriever(retrievals[start:start + args.batch_size],
                [r["question"] for r in batch], [r["answer"] for r in batch])
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in retriever.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            total += loss.item() * len(batch)
        print(f"epoch={epoch + 1} loss={total / len(records):.6f}")
    retriever.save_adapter(args.output)
    print("Saved graph encoder/projector:", args.output)


if __name__ == "__main__":
    main()
```

<a id="examples-g_retriever_generatepy"></a>

#### g_retriever_generate.py

<!-- source: python/g_retriever_generate.py -->
```python
"""PCST retrieval -> upstream GNN -> learned soft prompt -> causal language model."""
import argparse
import json
from g_retriever import GraphRetriever, HashEncoder, SentenceEncoder, retrieve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="examples/data/graph2text.json")
    parser.add_argument("--question", default="Who wrote notes about the Analytical Engine?")
    parser.add_argument("--model", help="HF/local causal LM; omit for offline random-model demo")
    parser.add_argument("--encoder", help="HF/local sentence encoder; omit for hash demo")
    parser.add_argument("--adapter", help="trained GNN/projector checkpoint")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    with open(args.graph) as source:
        graph = json.load(source)
    encoder = SentenceEncoder(args.encoder) if args.encoder else HashEncoder()
    result = retrieve(graph, args.question, encoder)
    if args.model:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    else:
        from g_retriever.demo import tiny_model
        model, tokenizer = tiny_model()
        print("Offline random-model demo: output is not a trained QA answer.")
    retriever = GraphRetriever(model, tokenizer, encoder.dimension, hidden_dim=args.hidden_dim,
        max_new_tokens=args.max_new_tokens)
    if args.adapter:
        retriever.load_adapter(args.adapter)
    else:
        print("GNN/projector are untrained; train or load an adapter for meaningful graph conditioning.")
    print(json.dumps(result.metadata(), indent=2))
    print("Answer:", retriever.generate([result], [args.question])[0])


if __name__ == "__main__":
    main()
```


<a id="examples-graph2text-finetune"></a>

### graph2text_finetune

<!-- source: graph2text_finetune.rs -->
```rust
//! Native Graph2Text SFT: consistent triple prompts, answer-only labels, shuffled
//! graph order, frozen base + output-head LoRA, holdout loss and adapter persistence.
use candlelighter::{
    graph_training::*,
    native::*,
    native_advanced::{QuantizationConfig, QuantizationType},
    native_training::*,
    NativeLlama,
};
use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use std::path::Path;

fn dataset(path: &str) -> Result<Vec<InstructionExample>, Box<dyn std::error::Error>> {
    let data: Vec<InstructionExample> = serde_json::from_str(&std::fs::read_to_string(path)?)?;
    if data.is_empty() {
        return Err("dataset must contain at least one record".into());
    }
    for example in &data {
        example.triples()?;
    }
    Ok(data)
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(4..=6).contains(&args.len()) {
        return Err("usage: graph2text_finetune MODEL_DIR TRAIN_JSON VALIDATION_JSON ADAPTER_JSON [EPOCHS=3; 0=reload/evaluate] [none|int4|int8]".into());
    }
    let epochs: usize = args.get(4).map(|s| s.parse()).transpose()?.unwrap_or(3);
    let quantization = match args.get(5).map(String::as_str).unwrap_or("none") {
        "none" => None,
        "int4" => Some(QuantizationType::Int4),
        "int8" => Some(QuantizationType::Int8),
        _ => return Err("quantization must be none, int4 or int8".into()),
    };
    if epochs > 0 && Path::new(&args[3]).exists() {
        return Err(
            "adapter output already exists; choose a new path or use EPOCHS=0 to reload".into(),
        );
    }
    let train = dataset(&args[1])?;
    let validation = dataset(&args[2])?;
    let artifacts = HuggingFaceArtifacts::from_dir(&args[0])?;
    let tokenizer = tokenizers::Tokenizer::from_file(&artifacts.tokenizer)
        .map_err(|e| NativeError(e.to_string()))?;
    let eos_ids = artifacts.config.eos_token_ids();
    let eos = eos_ids.first().copied();
    let context = artifacts.config.max_position_embeddings.unwrap_or(2048);
    let mut model = match quantization {
        Some(dtype) => NativeLlama::load_quantized(
            &artifacts,
            QuantizationConfig {
                dtype,
                group_size: 64,
            },
        )?,
        None => NativeLlama::load(&artifacts)?,
    };
    if epochs == 0 {
        let saved: HeadAdapterCheckpoint =
            serde_json::from_str(&std::fs::read_to_string(&args[3])?)?;
        if saved.base_model != args[0] {
            return Err("adapter base-model path differs; use its original base model".into());
        }
        model.set_lm_head_lora(saved.into_adapter()?)?;
    } else {
        model.enable_lm_head_lora(
            LoraConfig {
                rank: 8,
                alpha: 16.0,
                dropout: 0.0,
            },
            42,
        )?;
    }
    let mut rng = StdRng::seed_from_u64(42);
    let mut order: Vec<_> = (0..train.len()).collect();
    for epoch in 0..epochs {
        order.shuffle(&mut rng);
        let mut loss = 0.0;
        for &index in &order {
            let row = &train[index];
            let prompt = row.prompt(true, &mut rng)?;
            let tokens = answer_only_tokens(&tokenizer, &prompt, &row.output, eos, context)?;
            loss += supervised_fine_tune_step(
                &mut model,
                &tokens.input_ids,
                &tokens.labels,
                -100,
                0.0,
                1e-4,
            )?;
        }
        println!(
            "epoch={} mean_train_loss={:.6}",
            epoch + 1,
            loss / train.len() as f32
        );
    }
    if epochs > 0 {
        let checkpoint = HeadAdapterCheckpoint::from_adapter(
            args[0].clone(),
            model.lm_head_lora().ok_or("missing trained adapter")?,
        );
        let mut output = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args[3])?;
        serde_json::to_writer_pretty(&mut output, &checkpoint)?;
        println!("saved_adapter={}", args[3]);
    }
    let mut eval_loss = 0.0;
    for row in &validation {
        let prompt = row.prompt(false, &mut rng)?;
        let tokens = answer_only_tokens(&tokenizer, &prompt, &row.output, eos, context)?;
        let logits = model.token_logits(&tokens.input_ids)?;
        eval_loss += causal_lm_loss(&logits, &tokens.labels, -100, 0.0)?.loss;
    }
    println!(
        "mean_validation_loss={:.6}",
        eval_loss / validation.len() as f32
    );
    let backend = HuggingFaceBackend::new(model, &artifacts)?;
    let mut engine = NativeEngine::new(backend, 1, eos_ids)?;
    for (i, row) in validation.iter().enumerate() {
        let prompt = row.prompt(false, &mut rng)?;
        let prompt_len = tokenizer
            .encode(prompt.as_str(), true)
            .map_err(|e| NativeError(e.to_string()))?
            .len();
        let available = context
            .checked_sub(prompt_len)
            .filter(|&n| n > 0)
            .ok_or("validation prompt fills model context")?;
        engine.submit(GenerateRequest {
            id: format!("validation-{i}"),
            prompt,
            sampling: SamplingParams {
                max_tokens: available.min(128),
                temperature: 0.0,
                ..Default::default()
            },
            constraint: None,
        })?;
        let response = engine.run_to_completion()?.remove(0);
        println!(
            "{}",
            serde_json::json!({"input":row.input,"reference":row.output,"prediction":response.text,"entity_mention_recall":entity_mention_recall(&row.triples()?,&response.text),"finish_reason":format!("{:?}",response.finish_reason)})
        );
    }
    Ok(())
}
```

<a id="examples-transformers-pipeline"></a>

### transformers_pipeline

<!-- source: transformers_pipeline.rs -->
```rust
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
```

<a id="examples-arm-acceleration"></a>

### Arm Acceleration

<!-- source: arm_acceleration.rs -->
```rust
//! No weights or vendor SDK required. Runtime dispatch checks OS-enabled features.
use candlelighter::arm::{dot, matmul, ArmCapabilities, Kernel};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "{}",
        serde_json::to_string_pretty(&ArmCapabilities::detect())?
    );
    let a = vec![1., 2., 3., 4., 5.];
    let b = vec![5., 4., 3., 2., 1.];
    assert!((dot(&a, &b, Kernel::Auto)? - 35.).abs() < 0.001);
    let product = matmul(
        &[1., 2., 3., 4., 5., 6.],
        &[7., 8., 9., 10., 11., 12.],
        2,
        3,
        2,
        Kernel::Auto,
    )?;
    assert_eq!(product, vec![58., 64., 139., 154.]);
    println!("dot=35; matrix product={product:?}");
    // Explicit requests report an error rather than silently running another kernel.
    for kernel in [Kernel::Neon, Kernel::Sve, Kernel::Sme] {
        match matmul(&a, &b, 1, 5, 1, kernel) {
            Ok(value) => println!("{kernel:?}: {value:?}"),
            Err(error) => println!("{kernel:?}: {error}"),
        }
    }
    Ok(())
}
```

<a id="examples-arm-npu-generate"></a>

### Arm Npu Generate

<!-- source: arm_npu_generate.rs -->
```rust
//! JSON config points to a compiled .tflite model, runtime and external delegate.
use candlelighter::{
    arm_npu::{TfliteNpuBackend, TfliteNpuConfig},
    native::{GenerateRequest, NativeBackend, NativeEngine, SamplingParams},
};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 4 {
        return Err(
            "usage: arm_npu_generate CONFIG.json TOKENIZER.json EOS_TOKEN_ID PROMPT".into(),
        );
    }
    let config: TfliteNpuConfig = serde_json::from_slice(&std::fs::read(&args[0])?)?;
    let backend = TfliteNpuBackend::from_files(config, &args[1])?;
    eprintln!("{}", serde_json::to_string_pretty(backend.report())?);
    let prompt_len = backend.encode(&args[3])?.len();
    let remaining = backend
        .report()
        .context_length
        .checked_sub(prompt_len)
        .ok_or("prompt exceeds compiled context")?;
    if prompt_len == 0 || remaining == 0 {
        return Err("prompt must be nonempty and leave room for generation".into());
    }
    let mut engine = NativeEngine::new(backend, 1, vec![args[2].parse()?])?;
    engine.submit(GenerateRequest {
        id: "arm-npu".into(),
        prompt: args[3].clone(),
        sampling: SamplingParams {
            max_tokens: remaining.min(32),
            temperature: 0.,
            ..Default::default()
        },
        constraint: None,
    })?;
    for response in engine.run_to_completion()? {
        println!("{}", response.text);
    }
    Ok(())
}
```

<a id="examples-arm-npu-host"></a>

### Arm Npu Host

<!-- source: arm_npu_host.rs -->
```rust
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
```

<a id="examples-ai-winter"></a>

### AI winter example: XOR training and inference

Follow the [step-by-step XOR tutorial](../README.md#ai-winter-example-learn-xor-then-run-inference) to train, save and reload this CPU network.

<!-- source: ai_winter.rs -->
```rust
//! Train XOR, save weights, then run inference in a separate process.
use candle_core::{Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarMap};
use candlelighter::prelude::{Activations, Dense, DenseLayerTrait, Trainable};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 || !matches!(args[0].as_str(), "train" | "infer") {
        return Err("usage: ai_winter {train|infer} CHECKPOINT.safetensors".into());
    }
    let device = Device::Cpu;
    let mut vars = VarMap::new();
    let hidden = Dense::new(8, 2, Activations::Sigmoid, &device, &vars, "hidden".into());
    let output = Dense::new(1, 8, Activations::Sigmoid, &device, &vars, "output".into());
    let inputs = Tensor::new(&[[0f32, 0.], [0., 1.], [1., 0.], [1., 1.]], &device)?;
    let targets = Tensor::new(&[[0f32], [1.], [1.], [0.]], &device)?;
    let forward = |x: Tensor| output.forward(hidden.forward(x));

    if args[0] == "train" {
        // Fixed seeded values make this CPU demonstration reproducible.
        let mut rng = StdRng::seed_from_u64(42);
        for (name, shape) in [
            ("hidden.weight", vec![8, 2]),
            ("hidden.bias", vec![8]),
            ("output.weight", vec![1, 8]),
            ("output.bias", vec![1]),
        ] {
            let count: usize = shape.iter().product();
            let values: Vec<f32> = (0..count).map(|_| rng.gen_range(-1.0..1.0)).collect();
            vars.set_one(name, Tensor::from_vec(values, shape, &device)?)?;
        }
        // Keep optimizer state across steps; train all four rows in one batch.
        let mut optimizer = AdamW::new(
            vars.all_vars(),
            ParamsAdamW {
                lr: 0.05,
                weight_decay: 0.0,
                ..Default::default()
            },
        )?;
        let initial =
            candle_nn::loss::mse(&forward(inputs.clone()), &targets)?.to_scalar::<f32>()?;
        println!("initial MSE: {initial:.6}");
        let mut converged = false;
        for step in 1..=5000 {
            let loss = candle_nn::loss::mse(&forward(inputs.clone()), &targets)?;
            optimizer.backward_step(&loss)?;
            if step % 100 == 0 {
                let value =
                    candle_nn::loss::mse(&forward(inputs.clone()), &targets)?.to_scalar::<f32>()?;
                println!("step {step:4}: MSE={value:.6}");
                if value < 0.001 {
                    converged = true;
                    break;
                }
            }
        }
        if !converged {
            return Err("XOR training did not reach the target loss".into());
        }
        vars.save(&args[1])?;
        println!("saved {}", args[1]);
    } else {
        // Reconstruct the same variable names/shapes before loading saved weights.
        vars.load(&args[1])?;
        println!("loaded {}", args[1]);
    }

    let probabilities = forward(inputs).to_vec2::<f32>()?;
    println!("a b | probability | XOR");
    for (index, expected) in [0u8, 1, 1, 0].into_iter().enumerate() {
        let probability = probabilities[index][0];
        let prediction = u8::from(probability >= 0.5);
        println!(
            "{} {} | {:.4}      | {}",
            index / 2,
            index % 2,
            probability,
            prediction
        );
        if !probability.is_finite() || prediction != expected {
            return Err(format!("incorrect XOR prediction for row {index}").into());
        }
    }
    Ok(())
}
```

<a id="examples-context-parallel-attention"></a>

### Context-parallel attention

Run with `cargo run --locked --no-default-features --example context_parallel_attention`.

<!-- source: context_parallel_attention.rs -->
```rust
//! Exact decode-context parallel attention on three simulated workers.
use candlelighter::context_parallel::{
    decode_context_parallel, shard_kv, ContextPlan, DecodeConfig, PartitionStrategy,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let keys: Vec<_> = (0..12).map(|i| vec![vec![i as f32 / 12.0, 1.0]]).collect();
    let values: Vec<_> = (0..12)
        .map(|i| vec![vec![i as f32, (i * i) as f32]])
        .collect();
    let plan = ContextPlan::new(12, 3, PartitionStrategy::Contiguous)?;
    let shards = shard_kv(&keys, &values, &plan)?;
    println!("DCP assignments: {:?}", plan.assignments);

    // One decode token; KV cache remains distributed and only three small
    // online-softmax states need to be reduced.
    let query = vec![vec![0.5, 0.75]];
    let output = decode_context_parallel(&query, 11, &shards, DecodeConfig::for_head_dim(2))?;
    println!("decode output: {:.4?}", output[0]);
    Ok(())
}
```
