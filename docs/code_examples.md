# Complete code examples

[Back to the handbook](handbook.md) · [Python Keras comparison](keras_comparison.md)

This is the code cookbook for the usage handbook. Each standalone listing below
is a complete checked-in program with imports, feature gates, inputs, execution,
and error handling. Copy a listing to `examples/<name>.rs` in a checkout and run
its command from the root; dependencies are already declared in `Cargo.toml`.
There are no omitted function bodies in these listings. For use in another crate,
add `candlelighter` and the upstream crates imported by the selected program;
Candle dependency versions must match the library's manifest.

Model-free examples validate small operations and APIs, not accuracy on a real
ML task. Toy logits, TinyStudent and flat objective-output parameters are labeled
in source. GPU, model-backed and internet examples are compile-checked; they were
not executed with downloaded weights. Large CLM runs need roughly 16 GB of
half-precision weights plus working memory, or quantized loading and temporary
shard conversion memory. Use `--release` for real decoder inference.

## Example index

| Program | Feature selection | Validation / prerequisites | Capabilities |
| --- | --- | --- | --- |
| [serve_model](#serve_model) | server | Compile-checked; compatible MODEL_DIR required | Embed the OpenAI-compatible host around a single native backend. See [host guide](model_host.md) for CLI, curl, Python SDK and limits. |
| [handbook_keras](#handbook_keras) | default | Local: validated | Sequential regression, persistent classification optimizer, shape checks and prediction-equivalent safetensors reload. |
| [handbook_layers](#handbook_layers) | default | Local: validated | Feature shaping/scaling and inverse, every dense activation, Conv1D, both pooling paths, normalization/flatten, LSTM/GRU, embedding, autoencoder and fit/predict. |
| [handbook_experimental](#handbook_experimental) | default | Local: validated narrow cases | Conv2D, feature masks, dropout, classification CE, minimal attention/MoE, PEFT and split constructors, manual merge, ODE solvers and liquid step. Constructor-only checks do not establish training support. |
| [handbook_native](#handbook_native) | native | Local: validated | A complete toy NativeBackend, capacity-two scheduling, queued/active cancellation, greedy generation/EOS/logprobs, sampling controls, choices/JSON/regex/GBNF, templates, every transform, fork/join, constraint semantics, schemas/signatures and MCP request/response round trips. |
| [handbook_training](#handbook_training) | native | Local: validated | LoRA gradient accumulation/averaging, AdamW clipping and SGD, adapter restoration, ignored labels, reward training, PPO/DPO/distillation hooks, both quantization types and cache lifecycle. |
| [handbook_jepa_liquid](#handbook_jepa_liquid) | none | Local: validated | Custom PatchEncoder, I-JEPA and V-JEPA, online/EMA image and video updates, three solvers, imported liquid parameters, explicit CfC/LFM states and readout fitting. |
| [native_advanced](#native_advanced) | native | Local: validated | Int4 projection, copy-on-write KV cache, terminal-aware GAE, clipped PPO, DPO and temperature-scaled distillation. |
| [native_finetune](#native_finetune) | native | Local path validated; snapshot path compile-checked | LoRA forward/backward, AdamW, mock SFT contract, pairwise reward model; optional MODEL_DIR performs a real NativeLlama LM-head update. |
| [native_prompt](#native_prompt) | native | Local: validated | Choice/generation directives, static bindings, finite grammar, ReAct parsing, typed signatures and complete MCP JSON-RPC request serialization. |
| [jepa](#jepa) | none | Local: validated | I-JEPA image and tube-masked V-JEPA forward passes, plus five trainable image updates. |
| [liquid_networks](#liquid_networks) | none | Local: validated | RK4 analytical decay check, stacked LFM sequence/readout training and irregular CfC elapsed times. |
| [native_generate](#native_generate) | native | Compile-checked; compatible MODEL_DIR required | Local Hugging Face artifacts/config/tokenizer/shards, NativeLlama, HuggingFaceBackend, seeded generation and EOS reporting. No downloads when snapshot is complete. |
| [select_huggingface_model](#select_huggingface_model) | native | Compile-checked; network required | Machine capability detection, Hub metadata search, policy selection and an explicit y/N download prompt. |
| [auto_native_generate](#auto_native_generate) | native | Compile-checked; network/weights required | Policy-based selection, automatic snapshot download, Int8 conversion and autoregressive generation. |
| [clm_generate](#clm_generate) | native | Compile-checked; network or local 8B snapshot required | CLM 8B offline/Hub paths, optional Int8/Int4 loading, joined prompts, generation and stop accounting. |
| [internet_jepa](#internet_jepa) | native | Compile-checked; network required | Download a real image and exercise I-JEPA, V-JEPA and trainable masked prediction. |
| [internet_liquid](#internet_liquid) | native | Compile-checked; network required | Download temperature observations, normalize, fit LFM readout, handle irregular CfC time and produce an ODE baseline. |

`server` means `--no-default-features --features server` (and includes native).
`default` selects Candle and native; `native` means
`--no-default-features --features native`; `none` means `--no-default-features`.
The table covers runnable public workflows. The main handbook covers the status
of unimplemented KAN/DoRA/other roadmap APIs; there is no invented runnable code
for capabilities absent from the crate.

## Choose a learning path

1. Porting Keras layers: run `handbook_layers`, then `handbook_keras`, and compare
   the [Python program](keras_comparison.md#complete-executable-python-listing).
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

## Complete standalone Rust programs

### serve_model

Embed the single-model HTTP host in a Rust application. Compatible local snapshot
required. This example is compile-checked; the host's real HTTP inference path was
validated with a tiny synthetic safetensors checkpoint.

```bash
cargo run --locked --release --example serve_model --no-default-features --features server -- MODEL_DIR
```

[Source](../examples/serve_model.rs) · [Complete API guide](model_host.md)

<!-- source: ../examples/serve_model.rs -->
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

### lighter-serve CLI

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

### handbook_keras

Sequential regression, persistent classification optimizer, shape checks and prediction-equivalent safetensors reload.

**Validation:** Local: validated.

[Source](../examples/handbook_keras.rs)

```bash
cargo run --locked --example handbook_keras
```

<!-- source: ../examples/handbook_keras.rs -->
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

### handbook_layers

Feature shaping/scaling and inverse, every dense activation, Conv1D, both pooling paths, normalization/flatten, LSTM/GRU, embedding, autoencoder and fit/predict.

**Validation:** Local: validated.

[Source](../examples/handbook_layers.rs)

```bash
cargo run --locked --example handbook_layers
```

<!-- source: ../examples/handbook_layers.rs -->
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

### handbook_experimental

Conv2D, feature masks, dropout, classification CE, minimal attention/MoE, PEFT and split constructors, manual merge, ODE solvers and liquid step. Constructor-only checks do not establish training support.

**Validation:** Local: validated narrow cases.

[Source](../examples/handbook_experimental.rs)

```bash
cargo run --locked --example handbook_experimental
```

<!-- source: ../examples/handbook_experimental.rs -->
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

### handbook_native

A complete toy NativeBackend, capacity-two scheduling, queued/active cancellation, greedy generation/EOS/logprobs, sampling controls, choices/JSON/regex/GBNF, templates, every transform, fork/join, constraint semantics, schemas/signatures and MCP request/response round trips.

**Validation:** Local: validated.

[Source](../examples/handbook_native.rs)

```bash
cargo run --locked --example handbook_native --no-default-features --features native
```

<!-- source: ../examples/handbook_native.rs -->
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

### handbook_training

LoRA gradient accumulation/averaging, AdamW clipping and SGD, adapter restoration, ignored labels, reward training, PPO/DPO/distillation hooks, both quantization types and cache lifecycle.

**Validation:** Local: validated.

[Source](../examples/handbook_training.rs)

```bash
cargo run --locked --example handbook_training --no-default-features --features native
```

<!-- source: ../examples/handbook_training.rs -->
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

### handbook_jepa_liquid

Custom PatchEncoder, I-JEPA and V-JEPA, online/EMA image and video updates, three solvers, imported liquid parameters, explicit CfC/LFM states and readout fitting.

**Validation:** Local: validated.

[Source](../examples/handbook_jepa_liquid.rs)

```bash
cargo run --locked --example handbook_jepa_liquid --no-default-features
```

<!-- source: ../examples/handbook_jepa_liquid.rs -->
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

### native_advanced

Int4 projection, copy-on-write KV cache, terminal-aware GAE, clipped PPO, DPO and temperature-scaled distillation.

**Validation:** Local: validated.

[Source](../examples/native_advanced.rs)

```bash
cargo run --locked --example native_advanced --no-default-features --features native
```

<!-- source: ../examples/native_advanced.rs -->
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

### native_finetune

LoRA forward/backward, AdamW, mock SFT contract, pairwise reward model; optional MODEL_DIR performs a real NativeLlama LM-head update.

**Validation:** Local path validated; snapshot path compile-checked.

[Source](../examples/native_finetune.rs)

```bash
cargo run --locked --example native_finetune --no-default-features --features native
```

For a real local snapshot, append `-- MODEL_DIR`; use release for larger models.

<!-- source: ../examples/native_finetune.rs -->
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

### native_prompt

Choice/generation directives, static bindings, finite grammar, ReAct parsing, typed signatures and complete MCP JSON-RPC request serialization.

**Validation:** Local: validated.

[Source](../examples/native_prompt.rs)

```bash
cargo run --locked --example native_prompt --no-default-features --features native
```

<!-- source: ../examples/native_prompt.rs -->
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

### jepa

I-JEPA image and tube-masked V-JEPA forward passes, plus five trainable image updates.

**Validation:** Local: validated.

[Source](../examples/jepa.rs)

```bash
cargo run --locked --example jepa --no-default-features
```

<!-- source: ../examples/jepa.rs -->
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

### liquid_networks

RK4 analytical decay check, stacked LFM sequence/readout training and irregular CfC elapsed times.

**Validation:** Local: validated.

[Source](../examples/liquid_networks.rs)

```bash
cargo run --locked --example liquid_networks --no-default-features
```

<!-- source: ../examples/liquid_networks.rs -->
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

### native_generate

Local Hugging Face artifacts/config/tokenizer/shards, NativeLlama, HuggingFaceBackend, seeded generation and EOS reporting. No downloads when snapshot is complete.

**Validation:** Compile-checked; compatible MODEL_DIR required.

[Source](../examples/native_generate.rs)

```bash
cargo run --locked --release --example native_generate --no-default-features --features native -- MODEL_DIR "Once upon a time"
```

<!-- source: ../examples/native_generate.rs -->
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

### select_huggingface_model

Machine capability detection, Hub metadata search, policy selection and an explicit y/N download prompt.

**Validation:** Compile-checked; network required.

[Source](../examples/select_huggingface_model.rs)

```bash
cargo run --locked --example select_huggingface_model --no-default-features --features native -- "Llama"
```

<!-- source: ../examples/select_huggingface_model.rs -->
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

### auto_native_generate

Policy-based selection, automatic snapshot download, Int8 conversion and autoregressive generation.

**Validation:** Compile-checked; network/weights required.

[Source](../examples/auto_native_generate.rs)

```bash
cargo run --locked --release --example auto_native_generate --no-default-features --features native -- "TinyLlama" "Hello"
```

<!-- source: ../examples/auto_native_generate.rs -->
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

### clm_generate

CLM 8B offline/Hub paths, optional Int8/Int4 loading, joined prompts, generation and stop accounting.

**Validation:** Compile-checked; network or local 8B snapshot required.

[Source](../examples/clm_generate.rs)

```bash
cargo run --locked --release --example clm_generate --no-default-features --features native -- "Explain ownership in Rust."
```

<!-- source: ../examples/clm_generate.rs -->
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

### internet_jepa

Download a real image and exercise I-JEPA, V-JEPA and trainable masked prediction.

**Validation:** Compile-checked; network required.

[Source](../examples/internet_jepa.rs)

```bash
cargo run --locked --example internet_jepa --no-default-features --features native
```

<!-- source: ../examples/internet_jepa.rs -->
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

### internet_liquid

Download temperature observations, normalize, fit LFM readout, handle irregular CfC time and produce an ODE baseline.

**Validation:** Compile-checked; network required.

[Source](../examples/internet_liquid.rs)

```bash
cargo run --locked --example internet_liquid --no-default-features --features native
```

<!-- source: ../examples/internet_liquid.rs -->
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

## Complete historical Candle training examples

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

### simple_dnn

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

### simple_cnn

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

### simple_rnn

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

### simple_s2s

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

### simple_tnn

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

### simple_enn

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

### simple_pnn

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

### simple_llm

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


## graph2text.rs

<!-- source: ../examples/graph2text.rs -->
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

## graph2text_native.rs

<!-- source: ../examples/graph2text_native.rs -->
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
