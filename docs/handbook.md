# Lighter usage handbook

Lighter (`candlelighter`) combines a Keras-inspired layer/model API on top of
[Candle](https://github.com/huggingface/candle) with a portable native CPU
language-model runtime. JEPA and liquid-network modules can run without either
feature. This is an experimental learning project; implemented APIs do not imply
production readiness.

## Contents

- [Model hosting and OpenAI API](model_host.md)
- [Complete code cookbook](code_examples.md)
- [Python Keras comparison](keras_comparison.md)
- [Setup and feature selection](#setup-and-feature-selection)
- [Capability table](#capability-table)
- [Candle layers and training](#candle-layers-and-training)
- [Experimental architectures](#experimental-architectures)
- [Native generation and model loading](#native-generation-and-model-loading)
- [Quantization, caches, and reinforcement objectives](#quantization-caches-and-reinforcement-objectives)
- [Fine-tuning and reward modeling](#fine-tuning-and-reward-modeling)
- [Prompts, grammars, workflows, and tools](#prompts-grammars-workflows-and-tools)
- [JEPA image and video learning](#jepa-image-and-video-learning)
- [Liquid networks and neural ODEs](#liquid-networks-and-neural-odes)
- [Validation and troubleshooting](#validation-and-troubleshooting)

## Complete examples and Python Keras

For complete programs instead of individual recipes, use the
[code cookbook](code_examples.md). It includes full source listings for every
standalone example and all historical Candle training examples, with commands,
feature requirements, prerequisites, and validation status. Every listing includes
its imports and executable entry point or original library functions.

The [Python Keras comparison](keras_comparison.md) maps the APIs and actual behavior,
then supplies a complete runnable Keras 3 program. It covers regression,
classification, preprocessing, layers, recurrent networks, attention, embedding,
regularization, autoencoders, multi-output/ensemble models, custom MoE, LoRA,
persistence, custom objectives, and JEPA/continuous-time reference implementations.
Capabilities without a direct core-Keras counterpart are identified explicitly.

Additional complete Rust workflows:

| Program | What it demonstrates | Command |
| --- | --- | --- |
| [handbook_keras](../examples/handbook_keras.rs) | Full regression, persistent custom classification training, prediction-equivalent safetensors restoration | `cargo run --locked --example handbook_keras` |
| [handbook_training](../examples/handbook_training.rs) | Averaged adapter gradients, clipping/SGD/AdamW, ignored labels, reward updates, PPO/DPO/distillation optimizer hooks, quantization and KV lifecycle | `cargo run --locked --example handbook_training --no-default-features --features native` |
| [handbook_jepa_liquid](../examples/handbook_jepa_liquid.rs) | Custom PatchEncoder, image/video training, imported recurrent parameters, solver accuracy and explicit CfC/LFM state | `cargo run --locked --example handbook_jepa_liquid --no-default-features` |

## Setup and feature selection

Install stable Rust and a native C/C++ compiler, then run commands from the
repository root. The first build downloads and compiles dependencies. `--locked`
keeps the checkout's dependency resolution intact.

```bash
cargo build --locked
cargo test --locked
```

For another Rust project:

```bash
cargo add candlelighter
# Or choose only the native runtime:
cargo add candlelighter --no-default-features --features native
```

| Feature selection | Available modules | Example command |
| --- | --- | --- |
| Default (`candle,native`) | Both runtimes, JEPA, liquid | `cargo run --locked --example handbook_layers` |
| `--no-default-features --features candle` | Candle layers and models, JEPA, liquid | `cargo run --locked --bin candlelighter --no-default-features --features candle` |
| `--no-default-features --features server` | Native runtime and OpenAI-compatible HTTP host | `cargo run --locked --bin lighter-serve --no-default-features --features server -- --help` |
| `--no-default-features --features native` | Native inference/training/prompts, JEPA, liquid | `cargo run --locked --example handbook_native --no-default-features --features native` |
| `--no-default-features` | JEPA and liquid only | `cargo run --locked --example jepa --no-default-features` |

CPU execution needs no CUDA installation. Some Candle examples call
`Device::cuda_if_available(0)` and fall back to CPU; this crate does not expose a
CUDA feature switch. The native built-in decoder is a CPU reference implementation.
Hardware detection does not enable GPU execution by itself.

## Capability table

**Implemented** means an execution path exists, with limits explained below.
**Experimental** means the API has known restrictions or incomplete behavior.
**Composition** means an example builds the architecture from existing primitives.
**Planned** means there is no Lighter implementation or runnable API example yet.
The table replaces the old checkmarks and running-person icons; duplicate
recurrent entries have been combined. Every implemented capability has an example
or recipe linked here. Planned entries include a concrete alternative or design
recipe, without claiming an unavailable API exists.

| Capability | Runtime / status | Example or usage recipe |
| --- | --- | --- |
| Feature shaping: samples, time steps, spatial features | Candle / implemented | [Layer tour](../examples/handbook_layers.rs), [shapes](#feature-shapes-and-scaling) |
| Min-max and z-score feature scaling, reuse and inverse transforms | Candle / implemented with edge cases | [Scaling recipe](#feature-shapes-and-scaling), [DNN](../lib/examples/simple_dnn.rs), [TNN](../lib/examples/simple_tnn.rs) |
| Dense layers | Candle / implemented | [Layer tour](../examples/handbook_layers.rs), [DNN](../lib/examples/simple_dnn.rs) |
| Linear, ReLU, SiLU, sigmoid, log-softmax activations | Candle / implemented | [Layer tour](../examples/handbook_layers.rs), [activation semantics](#dense-layers-and-training) |
| 1D and 2D convolution | Candle / implemented with explicit kernel | [Layer tour](../examples/handbook_layers.rs), [CNN](../lib/examples/simple_cnn.rs), [2D recipe](#convolution-pooling-normalization-and-flatten) |
| Average and max pooling | Candle / experimental: enum names reversed | [Layer tour](../examples/handbook_layers.rs), [pooling behavior](#convolution-pooling-normalization-and-flatten) |
| Normalization | Candle / experimental: wrapper returns input | [Layer tour](../examples/handbook_layers.rs), [explicit L2 recipe](#convolution-pooling-normalization-and-flatten) |
| Flatten | Candle / implemented, flattens all dimensions | [Layer tour](../examples/handbook_layers.rs) |
| LSTM and GRU recurrence | Candle / implemented one-step wrapper | [Layer tour](../examples/handbook_layers.rs), [RNN](../lib/examples/simple_rnn.rs) |
| Regulation / regularization | Candle / no dedicated Lighter layer | [Dropout and weight-decay recipe](#regularization) |
| Sequential model, forward, fit, predict, summary | Candle / implemented with MSE training | [Layer tour](../examples/handbook_layers.rs), [training recipe](#dense-layers-and-training) |
| SGD and Adam training | Candle / implemented; Adam uses Candle AdamW | [Optimizer recipe](#dense-layers-and-training) |
| MSE, NLL, BCE-with-logits, cross-entropy losses | Candle / only MSE accepted by `fit` | [Loss recipe](#dense-layers-and-training) |
| Model JSON and JSON/safetensors weights | Candle / experimental round-trip support | [Persistence recipe](#saving-and-loading), [existing tests](../tests/test_importexport.rs) |
| Autoencoder | Candle / composition; dedicated VAE planned | [Layer tour](../examples/handbook_layers.rs), [autoencoder recipe](#autoencoders) |
| Standard feature embedding | Candle / implemented | [Layer tour](../examples/handbook_layers.rs), [S2S](../lib/examples/simple_s2s.rs) |
| Timestep, absolute, rotary, sinusoidal embedding layers | Candle wrappers / planned | [Embedding alternatives](#recurrent-layers-and-embeddings); native decoder already has RoPE |
| Self-attention | Candle / experimental shape-sensitive wrapper | [Attention recipe](#attention-and-mixture-of-experts), [TNN](../lib/examples/simple_tnn.rs) |
| Cross, causal, multi-head/query, grouped-query attention wrappers | Candle / planned beyond current wrapper | [Attention alternatives](#attention-and-mixture-of-experts); native decoder implements causal GQA/MQA |
| Sparse Mixture of Experts | Candle / experimental | [MoE recipe](#attention-and-mixture-of-experts), [ENN](../lib/examples/simple_enn.rs) |
| Multi-task, gated-training, hierarchical, conditional MoE variants | Candle / planned | [MoE alternatives](#attention-and-mixture-of-experts) |
| Feature/sequence masking | Candle / planned dedicated layer | [Masking recipe](#masking-and-kan-dense); JEPA block/tube masks are implemented |
| Feature/weight quantization | Native / implemented Int8 and packed Int4; Candle feature layer planned | [Quantization example](../examples/native_advanced.rs), [recipe](#quantization-caches-and-reinforcement-objectives) |
| KAN-Dense | Planned | [KAN design recipe](#masking-and-kan-dense) |
| Candle LoRA and DoRA dense variants (PEFT) | Candle / experimental | [PEFT constructor recipe](#candle-peft), [DNN2/DNN3](../lib/examples/simple_dnn.rs) |
| Parallel split and merge models | Candle / experimental | [Parallel recipe](#parallel-split-and-merge), [PNN](../lib/examples/simple_pnn.rs) |
| BERT text similarity | Candle / implemented wrapper; checkpoint setup required | [BERT recipe](#bert-and-candle-llama), [LLM](../lib/examples/simple_llm.rs) |
| Candle Llama completion | Candle / implemented wrapper; checkpoint setup required | [Llama recipe](#bert-and-candle-llama), [LLM2](../lib/examples/simple_llm.rs) |
| Other Candle transformer architectures | No dedicated Lighter wrapper | [Transformer scope](#bert-and-candle-llama), [upstream model catalog](transformers.MD) |
| Model hosting: OpenAI completion/chat API, incremental SSE, discovery | Server feature / implemented CPU host | [Hosting guide](model_host.md), [embedding source](../examples/serve_model.rs) |
| Host authentication, health/metrics, bounded admission, context/timeout/cancellation | Server feature / implemented | [Hosting configuration](model_host.md#limits-streaming-and-failure-behavior) |
| JSON-object output mode | Native/server / object-root constraint | [API recipe](model_host.md#curl-examples); token limits/stop strings can truncate output |
| Continuous batching, fair scheduling, step/run-to-completion | Native / implemented; tensor batching depends on backend | [Native tour](../examples/handbook_native.rs) |
| Greedy/seeded temperature, top-k/top-p/min-p sampling | Native / implemented | [Native tour](../examples/handbook_native.rs), [sampling recipe](#sampling-and-scheduling) |
| Presence/frequency/repetition penalties, bad tokens, logit bias | Native / implemented | [Native tour](../examples/handbook_native.rs) |
| EOS/stop tokens/strings, min/max tokens, cancellation, log probabilities | Native / implemented | [Native tour](../examples/handbook_native.rs), [scheduler semantics](#sampling-and-scheduling) |
| Custom tensor backend/model adapter | Native / implemented extension traits | [Demo backend](../examples/handbook_native.rs), [adapter recipe](#snapshots-and-custom-backends) |
| Hugging Face config, tokenizer, single/sharded safetensors downloads | Native / implemented | [Snapshot recipe](#snapshots-and-custom-backends), [native_generate](../examples/native_generate.rs) |
| Dense Llama/Mistral/Qwen-family decoder | Native / implemented portable reference | [Generation recipe](#native-generation-and-model-loading), [native_generate](../examples/native_generate.rs) |
| F32/F16/BF16 weights, RMSNorm, SiLU MLP, RoPE, sliding window, KV cache | Native decoder / implemented | [Decoder scope](#snapshots-and-custom-backends), [CLM guide](contrastive_lm.MD) |
| CPU/memory/accelerator detection and compatible-model selection | Native / implemented | [Selection recipe](#model-selection), [select_huggingface_model](../examples/select_huggingface_model.rs) |
| Automatic model download and generation | Native / implemented, network/model required | [auto_native_generate](../examples/auto_native_generate.rs), [selection recipe](#model-selection) |
| CLM v0.1 8B, cached/offline/repeated generation and quantized loading | Native / implemented | [CLM recipe](#contrastive-lm), [complete guide](contrastive_lm.MD) |
| Paged KV cache, shared prefixes, copy-on-write, truncation | Native / implemented primitive | [native_advanced](../examples/native_advanced.rs), [cache recipe](#quantization-caches-and-reinforcement-objectives) |
| GAE, clipped PPO, DPO preference loss | Native / implemented objectives/gradients | [native_advanced](../examples/native_advanced.rs), [reinforcement recipe](#quantization-caches-and-reinforcement-objectives) |
| Distillation and optimizer update hooks | Native / implemented primitives | [native_advanced](../examples/native_advanced.rs), [update recipe](#quantization-caches-and-reinforcement-objectives) |
| LoRA initialization, forward/backward, gradient accumulation, SGD | Native / implemented | [native_finetune](../examples/native_finetune.rs), [fine-tuning recipe](#fine-tuning-and-reward-modeling) |
| Gradient-clipped AdamW, label-smoothed causal LM loss, SFT | Native / implemented | [native_finetune](../examples/native_finetune.rs) |
| NativeLlama LM-head LoRA update | Native / implemented, local snapshot required | [Snapshot SFT command](#fine-tuning-and-reward-modeling) |
| Reward/value head, pairwise preference updates | Native / implemented | [native_finetune](../examples/native_finetune.rs) |
| Prompt variables, generation and choice directives | Native / implemented | [native_prompt](../examples/native_prompt.rs), [prompt recipe](#prompts-grammars-workflows-and-tools) |
| Chained transforms, state fork/join | Native / implemented; branches execute sequentially | [Native tour](../examples/handbook_native.rs) |
| Choice, JSON, regex and finite GBNF output constraints | Native / implemented subset; regex completion only | [Native tour](../examples/handbook_native.rs), [constraint recipe](#prompts-grammars-workflows-and-tools) |
| Typed signatures, OpenAI/ReAct tool parsing, MCP JSON-RPC types | Native / implemented parsing/serialization | [native_prompt](../examples/native_prompt.rs), [Native tour](../examples/handbook_native.rs) |
| I-JEPA image and V-JEPA tube masking / prediction | Feature-independent / implemented | [jepa](../examples/jepa.rs), [JEPA recipe](#jepa-image-and-video-learning) |
| JEPA online encoder/predictor, EMA target training | Feature-independent / implemented compact trainer | [jepa](../examples/jepa.rs), [internet_jepa](../examples/internet_jepa.rs) |
| Euler, Heun, RK4 Neural ODE solvers | Feature-independent / implemented | [Solver recipe](#liquid-networks-and-neural-odes), [liquid_networks](../examples/liquid_networks.rs) |
| Liquid time-constant and CfC recurrent cells, irregular time steps | Feature-independent / implemented | [Cell recipe](#liquid-networks-and-neural-odes), [liquid_networks](../examples/liquid_networks.rs) |
| Stacked LFM and trainable readout | Feature-independent / implemented | [liquid_networks](../examples/liquid_networks.rs), [internet_liquid](../examples/internet_liquid.rs) |

## Candle layers and training

Run the complete small CPU tour, with assertions and no model downloads:

```bash
cargo run --locked --example handbook_layers
```

It covers feature shaping/scaling, every dense activation, convolution, both
pooling choices, flattening, normalization behavior, LSTM/GRU, embedding,
a composed autoencoder, and supervised regression.

The older training examples use the interactive selector:

```bash
cargo run --locked --bin candlelighter
```

Choose a named example with arrow keys and Enter, or select `exit`.
Run from the root so `Simple CNN` can read `data/clock/`. DNN, CNN, RNN, S2S,
TNN, ENN, and PNN refer to this selector, not Cargo `--example` targets.
The historical examples illustrate experiments and are not all smoke tests.

### Feature shapes and scaling

`Features` builds tensors with samples first, time steps second, and spatial
features after that. A simple regression input has shape `(samples, 1, features)`.
`fit` and `predict` iterate over the sample dimension; `forward` applies layers
to the tensor directly. Keep each sample's shape consistent.

```rust
use candlelighter::prelude::*;
use candlelighter::preprocessing::features::{Features, FeaturesTrait};
let device = Device::Cpu;
let mut features = Features::new(device.clone());
features.add_feature_1_d(vec![1., 2.]);
features.add_feature_1_d(vec![3., 4.]);
let x = features.get_data_tensor(); // [2, 1, 2]
let scaling = FeatureScaling::new(x);
let normalized = scaling.min_max_normalization();
let standardized = scaling.z_score();
let held_out = Tensor::new(&[[[5f32, 6.]]], &device)?;
let scaled_held_out = scaling.min_max_normalization_other(held_out.clone());
let standardized_held_out = scaling.z_score_other(held_out);
let original = scaling.min_max_normalization_reverse(normalized);
let original_z = scaling.z_score_reverse(standardized);
```

These transforms use global statistics across the tensor, rather than per-column
statistics. Fit them on training data and use the `_other` methods for validation
or inference. Constant data gives a zero denominator; handle it before scaling.
Reverse methods return flattened tensors; reshape explicitly if needed.
[Feature-shape notes](featuredatastructure.MD) describe the conventions.

### Dense layers and training

`Dense::new(output_width, input_width, activation, ...)` uses a shared `VarMap`;
use distinct names for distinct dense layers. Available activations are `Linear`,
`Relu`, `Silu`, `Sigmoid`, and `Softmax`. The last currently computes **log-softmax**,
so its output is log probabilities, not probabilities.

```rust
use candlelighter::prelude::*;
let dev = Device::Cpu;
let vars = VarMap::new();
let mut model = SequentialModel::new(vars.clone(), vec![
    Box::new(Dense::new(4, 2, Activations::Relu, &dev, &vars, "hidden".into())),
    Box::new(Dense::new(1, 4, Activations::Linear, &dev, &vars, "output".into())),
]);
model.compile(Optimizers::SGD(0.01), Loss::MSE);
let x = Tensor::new(&[[[1f32, 2.]], [[2., 3.]]], &dev)?;
let y = Tensor::new(&[[[3f32]], [[5.]]], &dev)?;
model.fit(x.clone(), y, 5, false);
let predictions = model.predict(&x).unwrap();
model.summary();
```

Use `Optimizers::Adam(0.001)` in `compile` to select Candle AdamW. The fitting
loop recreates the optimizer per sample, so it does not preserve Adam moments
across updates. `summary` gives a rough layer-width count, not a precise parameter
count. `fit` needs positive epochs and a usable optimizer; its snapshot logic can
panic for no updates or an exactly zero loss.

Although `Loss` also lists `NLL`, `BinaryCrossEntropyWithLogit`, `CrossEntropy`,
and `None`, the current fitting loop rejects every non-MSE loss. For classification,
use a custom Candle training loop, for example:

```rust
let logits = Tensor::new(&[[2f32, -1.], [-1., 2.]], &dev)?;
let labels = Tensor::new(&[0u32, 1], &dev)?;
let loss = candle_nn::loss::cross_entropy(&logits, &labels)?;
```

This demonstrates the upstream loss, not support for `model.fit(..., CrossEntropy)`.
See the [DNN source](../lib/examples/simple_dnn.rs) for longer training examples.

### Convolution, pooling, normalization, and flatten

For explicit kernels, use `Conv::new(kernel, dimensions, padding, stride,
dilation, groups, ...)`. A 1D tensor is `[batch, channels, length]`; a 2D tensor
is `[batch, channels, height, width]`.

```rust
let vars = VarMap::new();
let kernel = Tensor::ones((1, 1, 2, 2), DType::F32, &dev)?;
let conv = Conv::new(kernel, 2, 0, 1, 1, 1, &dev, &vars, "conv2d".into());
let image = Tensor::ones((1, 1, 4, 4), DType::F32, &dev)?;
let convolved = conv.forward(image); // [1, 1, 3, 3]
```

`Conv::new2` initialization choices currently return initialized tensors rather
than perform ordinary convolution. Use an explicit kernel for a convolution
example; only dimensions 1 and 2 are implemented.

`Pooling` accepts a rank-3 `[channels, height, width]` tensor, inserts a batch
axis internally, and removes it afterward. Currently `PoolingType::MAX` calls
average pooling and `PoolingType::AVERAGE` calls max pooling. The layer tour
checks the actual outputs `2.5` and `4.0` for a 2×2 patch `[1,2;3,4]`.
For conventional names, call Candle directly:

```rust
let image = Tensor::new(&[[[[1f32, 2.], [3., 4.]]]], &dev)?;
let average = image.avg_pool2d_with_stride(2, 2)?;
let maximum = image.max_pool2d_with_stride(2, 2)?;
```

`Normalization::forward` currently discards the result of `normalize_axis` and
returns unchanged input. To normalize an `[N, features]` tensor explicitly:

```rust
let x = Tensor::new(&[[3f32, 4.]], &dev)?;
let normalized = x.broadcast_div(&x.sqr()?.sum_keepdim(1)?.sqrt()?)?;
// [0.6, 0.8]; handle zero-norm vectors before division.
```

`Flatten::new(&dev, &vars, name).forward(input)` produces `[1, total_elements]`.
It flattens the entire tensor, including any batch dimension; use it after the
sample axis has been removed when batching through `predict` or `fit`.

### Recurrent layers and embeddings

```rust
use candlelighter::recurrenttypes::RecurrentType;
let rnn = Recurrent::new(RecurrentType::LSTM, 2, 3, &dev, &VarMap::new(), "rnn".into());
let state = rnn.forward(Tensor::new(&[[1f32, 2.]], &dev)?); // [1, 3]
```

Substitute `RecurrentType::GRU` for GRU. Each call initializes a zero state with
batch size 1 and performs one step. LSTM returns its cell state `c`, while GRU
returns `h`. The wrapper does not retain state or unroll a sequence. Use Candle's
`RNN` methods or the liquid cells below for explicit stateful stepping. Separate
`VarMap`s avoid recurrent weight-name collisions.

```rust
use candlelighter::embeddingtypes::EmbeddingType;
use candlelighter::layer::embeddinglayer::{Embed, EmbeddingLayerTrait};
let embed = Embed::new(EmbeddingType::Standard, 8, 3, &dev, &VarMap::new(), "embed".into());
let vectors = embed.forward(Tensor::new(&[0u32, 2, 7], &dev)?); // [3, 3]
```

IDs must be in the vocabulary range. The wrapper converts input to `U32`.
Timestep, absolute, rotary, and sinusoidal standalone embedding wrappers from the
old roadmap do not exist. For an absolute-position recipe, create a second
standard embedding with `vocab_size = max_positions`, look up the position IDs,
and add its vectors to token embeddings. NativeLlama applies RoPE internally.
See [embedding notes](embedding.MD) for the historical roadmap.

### Regularization

The old “Regulation” row did not name a dedicated implementation. Use Candle
primitives for dropout and weight decay in a custom training loop:

```rust
let dropout = candle_nn::Dropout::new(0.1);
let training_output = candle_core::ModuleT::forward_t(&dropout, &x, true)?;
let inference_output = candle_core::ModuleT::forward_t(&dropout, &x, false)?;
let config = candle_nn::ParamsAdamW { lr: 1e-3, weight_decay: 0.01, ..Default::default() };
let optimizer = candle_nn::AdamW::new(vars.all_vars(), config)?;
```

There is no Lighter `Dropout` Trainable wrapper to add directly to a sequential
model. Native `LoraConfig::dropout` and `AdamWConfig::weight_decay` provide the
corresponding controls for native adapter training.

### Saving and loading

Architecture JSON and weights are separate files:

```rust
use candlelighter::saveweightstype::SaveWeightsType;
model.save_model("network.model");
model.save_weights(SaveWeightsType::SafeTensor, &model.varmap, "network.safetensors");
model.save_weights(SaveWeightsType::PlainJSON, &model.varmap, "network.weights.json");
let architecture = model.load_model("network.model", &dev);
```

The existing [import/export tests](../tests/test_importexport.rs) exercise file
creation and API calls, not prediction-equivalent round trips. Current JSON
weight loading flattens values and replaces map entries; model reconstruction
also has VarMap ownership issues. Flatten serialization is unsupported, embedding
JSON omits its type, normalization uses inconsistent field names, and SparseMoE
serialization calls unfinished `as_any`. Do not assume arbitrary models can be
restored correctly. For a reliable custom weight restoration path, construct
layers with a mutable Candle `VarMap`, then use its `save`/`load` methods and check
predictions before and after. For native snapshots, use the loader below.

## Experimental architectures

### Autoencoders

The [layer tour](../examples/handbook_layers.rs) composes a 2→1→2 autoencoder:

```rust
let vars = VarMap::new();
let mut autoencoder = SequentialModel::new(vars.clone(), vec![
    Box::new(Dense::new(1, 2, Activations::Relu, &dev, &vars, "encoder".into())),
    Box::new(Dense::new(2, 1, Activations::Linear, &dev, &vars, "decoder".into())),
]);
autoencoder.compile(Optimizers::SGD(0.01), Loss::MSE);
// With a nonconstant [samples, 1, 2] tensor:
autoencoder.fit(x.clone(), x, 5, false);
```

This is a reconstruction architecture built from dense layers. There is no
dedicated variational autoencoder API; [autoencoder notes](autoencoder.MD) point
to an upstream VAE implementation.

### Attention and Mixture of Experts

Run the narrow experimental recipes with `cargo run --locked --example handbook_experimental`.
[The source](../examples/handbook_experimental.rs) checks 2D convolution, masking,
upstream dropout/losses, a tiny attention/MoE forward pass, PEFT/split construction,
explicit averaging, every ODE solver, and a liquid-cell step. These checks do not
establish general training support for experimental wrappers.

The current self-attention constructor can be explored with a minimal shape:

```rust
let attention_vars = VarMap::new();
let attention = SelfAttention::new(1, 1, 1, 1, &dev, &attention_vars, "attention".into());
let output = attention.forward(Tensor::ones((1, 1), DType::F32, &dev)?);
```

The wrapper uses an image-oriented Candle attention module, hardcodes batch and
channel sizes to 1, and uses an all-ones KV tensor. The key/value mapper is not
applied. This is a limited experiment rather than general transformer attention.
The [TNN](../lib/examples/simple_tnn.rs) contains the original larger experiment.
Cross-attention and the other standalone wrappers in [attention notes](attention.MD)
remain roadmap items. NativeLlama already implements causal attention with
multi-query/grouped-query layouts; use a compatible snapshot for those capabilities.

SparseMoE can be constructed for an equal-width toy forward pass:

```rust
use candlelighter::layer::sparsemoe::{SparseMoE, SparseMoETrait};
let moe = SparseMoE::new(2, 4, 4, &dev, &VarMap::new(), "moe".into());
let output = moe.forward(Tensor::ones((1, 4), DType::F32, &dev)?);
```

The gate selects a hardcoded top two indices after evaluating every expert; it
is not sparse execution. The output is reshaped to the input shape, so widths
must match. Expert/gate maps are separate from the constructor's map, limiting
training integration; `as_any` is unfinished, so serialization panics.
See [ENN](../lib/examples/simple_enn.rs) and [MoE notes](moe.MD). Multi-task,
hierarchical, conditional, and gated-training variants are planned. A simple
multi-task alternative is to call independent `SequentialModel::forward`s and
retain each output rather than claim a multi-task MoE API.

### Masking and KAN-Dense

There is no dedicated feature-masking layer. In a custom Candle pipeline, an
explicit element mask can be applied before the model:

```rust
let x = Tensor::new(&[[1f32, 2., 3.]], &dev)?;
let mask = Tensor::new(&[[1f32, 0., 1.]], &dev)?;
let masked = x.mul(&mask)?; // [1, 0, 3]
```

This zeros features; it does not implement variable-length attention masking.
JEPA's `BlockMask` implements patch exclusion/tube masking, and native generation
constraints mask candidate tokens. Int8/Int4 matrix quantization is available in
the native runtime, not as a Candle feature layer.

KAN-Dense has no implementation. An illustrative design recipe is
`y_j = sum_i spline_ji(x_i)`: choose knots, evaluate spline bases for each input,
then train their coefficients. Ordinary `Dense` is a runnable baseline, not a
KAN substitute. The original [KAN discussion](https://www.holeoftherabbit.com/2024/06/16/may-kan-will-be-the-next-ai-disruption-step/)
is background reading; no `KAN` constructor is provided.

### Candle PEFT

The historical DNN2/DNN3 examples use the experimental constructor:

```rust
use candlelighter::densetypes::DenseType;
let rank_tensor = Tensor::zeros((2, 2), DType::F32, &dev)?;
let adapter = Dense::new2(2, 2, Activations::Linear, DenseType::LORA,
    rank_tensor, 1.0, &dev, &VarMap::new(), "adapter".into());
```

Replace `LORA` with `DORA` to construct its experimental variant. The constructor
uses the tensor's **number of dimensions** as rank, not its values. Adapter
matrices are plain tensors outside the VarMap, with restrictive multiplication
shapes; the DoRA path contains unfinished weight-update logic. Treat these as
constructor experiments rather than a validated PEFT training workflow.
Use [native_finetune](../examples/native_finetune.rs) for explicit trainable LoRA
and an actual NativeLlama LM-head update. Native DoRA is not implemented.

### Parallel split and merge

```rust
let vars = VarMap::new();
let branches: Vec<Box<dyn Trainable>> = vec![
    Box::new(Dense::new(2, 2, Activations::Linear, &dev, &vars, "branch_a".into())),
    Box::new(Dense::new(2, 2, Activations::Linear, &dev, &vars, "branch_b".into())),
];
let split = ParallelModel::new(ParallelModelType::Split, &dev, vars, branches);
```

This shows construction. `Split::predict` currently delegates to the sequential
prediction helper; `ParallelModel::forward` also chains layers sequentially.
`Merge::predict` attempts gated weighting but its constructor creates a gate with
input width 1 and a zero-width intermediate layer. General merge inference and
training are not complete. See [PNN](../lib/examples/simple_pnn.rs) and
[merging notes](modelmerging.MD). For a working two-model average, apply equal-width
models independently and combine their tensors explicitly:

```rust
let a = model_a.forward(x.clone());
let b = model_b.forward(x);
let mean = ((a + b)? * 0.5)?;
```

This recipe is an ensemble average, not validation of the `Merge` implementation.

### BERT and Candle Llama

`Simple LLM` illustrates BERT sentence embedding and cosine-style similarity;
`Simple LLM2` illustrates tokenization and Llama completion. Both are in
[the transformer example source](../lib/examples/simple_llm.rs).

```bash
cargo run --locked --bin candlelighter
# Choose Simple LLM or Simple LLM2 after preparing the checkpoint configuration.
```

These historical examples contain token/path placeholders and fixed model
configuration. Match vocabulary, hidden width, layer count, and other settings to
the checkpoint before running them; do not paste credentials into source.
For a custom program, populate the parameter map at runtime with `hf_model`,
optional `hf_token` from `HF_TOKEN`, or local `hf_model.safetensors` and
`hf_tokenizer.json` paths. Use `get_tokenizer`, `load_weights`, and `predict` as
shown in the example. Lighter wraps BERT and Llama; the other text, vision, audio,
and multimodal architectures listed in [transformers.MD](transformers.MD) are
upstream Candle possibilities, not automatically exposed Lighter capabilities.

## Model hosting

Run `lighter-serve` to expose a single loaded native model through OpenAI-compatible
HTTP endpoints. The [hosting guide](model_host.md) includes complete CLI, curl,
Python SDK and embedding examples, supported fields, and vLLM differences.
The optional `server` feature includes `native`. Actual inference uses the existing
CPU decoder; endpoint compatibility does not add vLLM GPU/distributed execution.

```bash
cargo run --locked --release --bin lighter-serve \
  --no-default-features --features server -- \
  --model MODEL_DIR --served-model-name local-model
```

## Native generation and model loading

### Sampling and scheduling

Start with a deterministic toy backend requiring no weights:

```bash
cargo run --locked --example handbook_native --no-default-features --features native
```

[The source](../examples/handbook_native.rs) implements `NativeBackend`, submits
multiple requests to a capacity-two engine, checks EOS termination, exercises
queued and active cancellation, returns log probabilities, and enforces a choice
constraint. A queued cancellation removes the request without a response; an
active cancellation produces `FinishReason::Cancelled` on a later step.
`run_to_completion` drives the loop, while `step` supports an application event loop.
This is scheduling over a toy backend, not language-model quality evaluation.
Override `logits_batch` or `NativeModel::forward_batch` for actual tensor batching;
the default implementation calls each sequence in turn.

Generation controls are fields of `SamplingParams`:

```rust
use candlelighter::native::SamplingParams;
let sampling = SamplingParams {
    max_tokens: 128, min_tokens: 1, temperature: 0.7,
    top_k: Some(40), top_p: 0.9, min_p: 0.05, seed: 42,
    presence_penalty: 0.1, frequency_penalty: 0.1, repetition_penalty: 1.1,
    stop: vec!["\nUser:".into()], stop_token_ids: vec![2],
    bad_token_ids: vec![0], logit_bias: [(3, 0.2)].into(),
    logprobs: Some(5), ..Default::default()
};
sampling.validate()?;
```

Token IDs must match the tokenizer. Use `temperature: 0.0` for greedy decoding.
`ignore_eos` and `include_stop_str_in_output` control EOS and stop-string behavior.
Limits are per request; sampling seeds make the same backend/request reproducible.

### Snapshots and custom backends

A local snapshot needs `config.json`, `tokenizer.json`, and either a single
`model.safetensors` or an index plus every referenced shard. The built-in loader
accepts F32/F16/BF16 tensors and retains half-precision matrix storage.

```bash
cargo run --locked --release --example native_generate \
  --no-default-features --features native -- MODEL_DIR "Once upon a time"
```

A library loading recipe:

```rust
use candlelighter::native::{HuggingFaceArtifacts, HuggingFaceBackend, NativeEngine};
use candlelighter::NativeLlama;
let artifacts = HuggingFaceArtifacts::from_dir("MODEL_DIR")?;
let eos = artifacts.config.eos_token_ids();
let model = NativeLlama::load(&artifacts)?;
let backend = HuggingFaceBackend::new(model, &artifacts)?;
let mut engine = NativeEngine::new(backend, 4, eos)?;
// Submit GenerateRequest values, then call step or run_to_completion.
```

`HuggingFaceArtifacts::from_hub(model_id, revision, token)` downloads configuration,
tokenizer, index, and shards. `HuggingFaceConfig` preserves unknown config fields.
For another tensor runtime, implement `NativeModel` and wrap it in
`HuggingFaceBackend`, or implement `NativeBackend` directly as in the native tour.

The built-in decoder handles dense Llama/Mistral/Qwen-family layouts, GQA/MQA,
rotary position embeddings including Llama 3 extended context, Mistral sliding
windows, RMS normalization, gated SiLU MLPs, tied embeddings, projection biases,
Qwen3 per-head Q/K normalization, and per-sequence KV caching. A model-family
name alone does not guarantee checkpoint compatibility; MoE/other tensor layouts
can be unsupported. These decoder internals are exercised by loading a compatible
snapshot and generating; they are not separately configurable standalone layers.

### Model selection

```bash
cargo run --locked --example select_huggingface_model \
  --no-default-features --features native -- "Llama"
cargo run --locked --release --example auto_native_generate \
  --no-default-features --features native -- "TinyLlama" "Hello"
```

The selection example inspects OS, architecture, CPU count, memory, SIMD and
accelerators, queries public model metadata/configuration, estimates memory, and
asks whether to download the selected snapshot. The automatic example selects,
downloads, and generates without the manual download prompt. Set `HF_TOKEN`
securely for authenticated access; public models may not need it. Neither example
runs offline. Selection is a memory/compatibility estimate, not a throughput
promise or a guarantee of enough memory for every prompt.

### Contrastive-LM

```bash
cargo run --locked --release --example clm_generate \
  --no-default-features --features native -- "Explain ownership in Rust."
# Offline, with reduced matrix storage:
CLM_MODEL_DIR=/models/CLM-v0.1-8B CLM_QUANTIZATION=int4 RAYON_NUM_THREADS=4 \
cargo run --locked --release --example clm_generate \
  --no-default-features --features native -- "Hello"
```

CLM downloads `Contrastive-LM/CLM-v0.1-8B` on its first Hub run. Unquantized
half-precision weights need roughly 16 GB plus working memory and enough disk
for the entire snapshot. Int8/Int4 conversion works shard by shard, reducing
resident matrix memory; it still needs temporary memory. CPU inference on 8B is
expensive: use release builds. `ContrastiveLm::from_dir`/`from_hub` and their
quantized variants retain weights across repeated `generate` calls.
See the [complete CLM guide](contrastive_lm.MD) for library examples and failures.

## Quantization, caches, and reinforcement objectives

```bash
cargo run --locked --example native_advanced --no-default-features --features native
```

The [example](../examples/native_advanced.rs) demonstrates Int4 matrix-vector
products, copy-on-write cache prefixes, GAE, clipped PPO, DPO, and distillation.
For Int8, change the matrix config to `QuantizationType::Int8`. `group_size`
controls per-group scales; quantization is lossy.

```rust
use candlelighter::native_advanced::*;
let matrix = QuantizedMatrix::quantize(2, 2, &[1., -1., 0.5, 0.25],
    QuantizationConfig { dtype: QuantizationType::Int8, group_size: 2 })?;
let output = matrix.matvec(&[2., 1.])?;
let mut cache = PagedKvCache::new(16, 2)?;
cache.append("prompt", &[1., 2.], &[3., 4.])?;
cache.fork("prompt", "branch")?;
cache.append("branch", &[5., 6.], &[7., 8.])?;
cache.truncate_left("branch", 1)?;
let (keys, values) = cache.read("branch")?;
cache.remove("branch");
```

The standalone `PagedKvCache` is a backend-neutral primitive. NativeLlama also
exposes `fork_sequence` and `kv_cache_stats`; application integration still needs
correct sequence lifecycle handling.

The reinforcement APIs calculate objectives and explicit gradients, not an entire
RL agent/environment. `generalized_advantage_estimate` handles terminal states;
`ppo_objective` consumes old/new log probabilities, values, returns, advantages,
and entropies. `dpo_loss` compares policy-vs-reference log ratios.
`distillation_loss` combines teacher KL and optional hard-label cross entropy.

To apply gradients, implement `OptimizationTarget::apply_output_gradients` in your
optimizer/backend and call `apply_ppo_update`, `apply_dpo_update`, or
`apply_distillation_update`. For example:

```rust
use candlelighter::native::NativeResult;
use candlelighter::native_advanced::*;
struct Parameters(Vec<f32>);
impl OptimizationTarget for Parameters {
    fn apply_output_gradients(&mut self, gradients: &[f32], learning_rate: f32) -> NativeResult<()> {
        assert_eq!(self.0.len(), gradients.len());
        for (weight, gradient) in self.0.iter_mut().zip(gradients) {
            *weight -= learning_rate * gradient;
        }
        Ok(())
    }
}
```

Map objective gradients to the correct model parameters in a real training loop;
these flat updates are an integration example, not a complete PPO trainer.

## Fine-tuning and reward modeling

```bash
# Local primitives: no snapshot required.
cargo run --locked --example native_finetune --no-default-features --features native
# Actual NativeLlama LM-head LoRA/SFT update:
cargo run --locked --release --example native_finetune \
  --no-default-features --features native -- MODEL_DIR
```

The [example](../examples/native_finetune.rs) initializes a seeded `LoraAdapter`,
computes its delta and backward gradients, updates it with gradient-clipped
`AdamW`, runs label-smoothed SFT through `FineTunableTransformer`, and trains a
pairwise `RewardHead`. Without a snapshot its `TinyStudent` illustrates the
training contract, not a transformer model. With a snapshot it loads NativeLlama,
enables LM-head LoRA, shifts labels, and performs one supervised update.

`LoraAdapter::zero_gradients` and `accumulate` support minibatch gradient
accumulation; `apply_sgd` provides a simple alternative optimizer. `causal_lm_loss`
uses an ignore-label value and optional label smoothing. Adapter `dropout` applies
only in training calls. `RewardHead::score` can serve as a scalar reward/value
prediction; `apply_pairwise_update` learns preferred vs rejected hidden vectors.
The built-in update targets the LM head, not full-network fine-tuning or a complete
QLoRA training pipeline. DoRA is available only as the experimental Candle variant.

## Prompts, grammars, workflows, and tools

```bash
cargo run --locked --example native_prompt --no-default-features --features native
cargo run --locked --example handbook_native --no-default-features --features native
```

[native_prompt](../examples/native_prompt.rs) shows variable binding, choice and
generation directives, regex completion, finite GBNF, ReAct tool parsing, typed
signatures, and MCP request serialization. [handbook_native](../examples/handbook_native.rs)
adds constrained engine generation, the four general constraint kinds, OpenAI tool-call
JSON, chained transforms and state fork/join.

```rust
use candlelighter::native_prompt::*;
let template = PromptTemplate::parse(
    "Question: {{question}}\nFormat: {{select format choices=json|text}}\nAnswer: {{gen answer max_tokens=32 regex=hello}}"
)?;
let constraint = ConstraintSpec::Choice { choices: vec!["yes".into(), "no".into()] };
// Assign to GenerateRequest::constraint to enforce it while decoding.
let json_constraint = ConstraintSpec::Json;
let regex = ConstraintSpec::Regex { pattern: "hello".into() };
let grammar = GbnfGrammar::parse("root ::= \"start\" | \"stop\"", "root")?;
assert!(grammar.allows_prefix("sta"));
assert!(grammar.is_complete("start"));
```

`RegexConstraint::allows_prefix` currently accepts every prefix; regex checks strict
completion only. It cannot prevent off-pattern tokens during decoding. Choice, JSON
and finite GBNF have prefix checks; test a selected constraint against the intended
vocabulary rather than assuming every constraint has identical guarantees.

`PromptExecutor` connects templates to an actual generator. The examples use a
mock executor and do not call an LLM. JSON constraints check JSON completion (`JsonObject` additionally requires an object root),
not an arbitrary JSON Schema. GBNF supports finite languages rather than a general
recursive grammar engine; `OutputConstraint` is the extension point.
`Workflow` supports JSON-pointer, upper/lowercase transforms and fork/join state;
branches currently execute sequentially. These are inspired by LMQL, Guidance,
SGLang, LCEL, and DSPy, not integrations with those frameworks.

`ToolCall::from_openai_json` and `from_react` parse calls; `McpRequest`/
`McpResponse` serialize JSON-RPC; `Signature::parse("question, context -> answer")`
parses typed-signature-style field lists. None automatically executes tools,
starts an MCP server, or provides an autonomous agent loop.

## JEPA image and video learning

```bash
cargo run --locked --example jepa --no-default-features
# Downloads the Rust logo for image/video training:
cargo run --locked --example internet_jepa --no-default-features --features native
```

The [local example](../examples/jepa.rs) constructs a `BlockMask`, uses `IJepa`
for image patch prediction and `VJepa` for tube-masked video prediction, then trains
an online encoder/predictor with an EMA target encoder via `JepaTrainer`.
Image inputs are flattened pixels; geometry describes height, width, channels,
and patch size. Video masks apply over the same spatial region across frames.

```rust
use candlelighter::jepa::*;
let mask = [BlockMask { row: 0, col: 1, height: 1, width: 1 }];
let pixels: Vec<f32> = (0..16).map(|x| x as f32 / 15.).collect();
let model = IJepa::new(MeanEncoder::new(4)?, 2, 2)?;
let prediction = model.forward(&pixels, 4, 4, 1, &mask)?;
```

`MeanEncoder` is a compact reference encoder. Implement `PatchEncoder` for a
production CNN/ViT encoder. These are compact implementations informed by the
official Meta I-JEPA/V-JEPA projects, not downloaded pretrained Meta weights.
The internet example requires network access on each run and cites its sources
in its [module documentation](../examples/internet_jepa.rs).

## Liquid networks and neural ODEs

```bash
cargo run --locked --example liquid_networks --no-default-features
# Downloads daily-minimum-temperature data for forecasting:
cargo run --locked --example internet_liquid --no-default-features --features native
```

The [local example](../examples/liquid_networks.rs) solves exponential decay,
runs a stacked LFM, trains its readout, and evaluates CfC with irregular elapsed
times. Exercise all solver choices and a liquid cell directly:

```rust
use candlelighter::liquid::*;
for solver in [OdeSolver::Euler, OdeSolver::Heun, OdeSolver::RungeKutta4] {
    let ode = NeuralOde::new(|_, state: &[f32]| vec![-state[0]], solver);
    let state = ode.solve(&[1.], 0., 1., 20)?;
}
let cell = LiquidCell::new(2, 4)?.with_solver(OdeSolver::Heun);
let next = cell.step(&[1., 0.], &[0.; 4], 0.1)?;
let cfc = CfcCell::new(2, 4)?;
let states = cfc.forward_irregular(&[vec![1., 0.], vec![0., 1.]], &[0.1, 0.5])?;
```

`Lfm::zero_state` and `step` support explicit recurrent state; `forward` starts
from zero state for a sequence. `fit_readout` trains the readout rather than all
recurrent parameters. Euler/Heun/RK4 accuracy depends on step size; elapsed values
are time intervals, not absolute timestamps. Implementations cite the official
Liquid Time-Constant Networks project in their source documentation.

## Validation and troubleshooting

Required local checks:

```bash
cargo build --locked
cargo test --locked
cargo run --locked --example handbook_layers
cargo run --locked --example handbook_keras
cargo run --locked --example handbook_training --no-default-features --features native
cargo run --locked --example handbook_jepa_liquid --no-default-features
cargo run --locked --example handbook_experimental
cargo run --locked --example handbook_native --no-default-features --features native
cargo run --locked --example native_advanced --no-default-features --features native
cargo run --locked --example native_finetune --no-default-features --features native
cargo run --locked --example native_prompt --no-default-features --features native
cargo run --locked --example jepa --no-default-features
cargo run --locked --example liquid_networks --no-default-features
```

The test suite writes `model.model`, `model4.model`, `model5.model`, `test.model`,
and `test.weights` in its working directory. Preserve existing user files and
remove only outputs created by your test run. Example assertions check small
functional paths; they do not establish model accuracy or production performance.

| Symptom | Next check |
| --- | --- |
| Missing Cargo/compiler | Install stable Rust and native C/C++ build tools; activate the toolchain in the shell. |
| Tensor shape panic | Check batch/sample axes, kernel channel counts, and layer widths. Many legacy wrappers unwrap errors. |
| Non-MSE fitting panic | Use MSE in the wrapper or implement a custom loss/training loop. |
| Constant-data scaling / zero-norm output | Handle zero variance, range, or norm before division. |
| Hugging Face 401/403 | Check access/license acceptance and securely configured `HF_TOKEN`. |
| Missing tensors or shards | Verify checkpoint architecture and every index-referenced shard. |
| Out of memory / slow generation | Use release, smaller models, Int8/Int4, and appropriate `RAYON_NUM_THREADS`. |
| Download blocked | Allow the actual Hugging Face/download destination required by the request; offline snapshots avoid Hub access. |

GPU execution, internet-backed examples, and large downloaded checkpoints are
optional checks and require their respective hardware, network access, disk, and
memory. For architecture-specific details, use the [CLM guide](contrastive_lm.MD)
and the linked source examples. Historical topic notes are useful design context;
the status table above describes the current implementation.

## Graph-to-text

See the [complete graph-to-text guide](graph2text.md) for property graphs, triples, rooted planning, exact realization, native model generation, HTTP streaming and a Python comparison.

| Capability | Status | Example |
| --- | --- | --- |
| Property graph and triple ingestion | Implemented | [Graph CLI](../examples/graph2text.rs) |
| Graph validation, breadth-first planning and fact provenance | Implemented | [Guide](graph2text.md) |
| Exact graph-to-text realization | Implemented | [JSON fixture](../examples/data/graph2text.json) |
| Native graph-grounded language generation | Implemented | [Native example](../examples/graph2text_native.rs) |
| Hosted graph generation and SSE | Implemented (`server`) | [Endpoint guide](graph2text.md#hosted-graph-generation) |
