# Rust Lighter

This project was started as my RUST exercise to abstract the Rust minimalist ML framework Candle (https://github.com/huggingface/candle) and introduce a more convenient way of programming neural network machine learning models.

The behaviour is inspired by Python KERAS (https://keras.io) and the initial step based on the Rust-Keras-like code (https://github.com/AhmedBoin/Rust-Keras-Like).

So let's call the project **Candle Lighter** &#128367;, because it helps to turn on the candle light and is even easier to implement.

Examples can be found in the **lib/examples/** and **examples/** directories.

To use it as library just call 'cargo add candlelighter'

## Running the examples

Run the commands below from the repository root after installing a stable Rust
toolchain. The first build downloads and compiles the dependencies, so it can
take a few minutes.

### Candle training examples

The examples in `lib/examples/` are exposed through the project's interactive
example selector. Start it with:

```bash
cargo run --bin candlelighter
```

Use the arrow keys to choose an example, press Enter to run it, or select
`exit` to close the selector. These examples use the default features and may
automatically use a CUDA device when one is available; otherwise they run on
the CPU. Run the command from the repository root so data-backed examples such
as `Simple CNN` can find the files under `data/`.

### Standalone native examples

The files in `examples/` are regular Cargo examples. They use the Candle-free
native runtime and can be run individually with this command pattern:

```bash
cargo run --example <example-name> --no-default-features --features native -- [arguments]
```

Examples that do not require a downloaded model can be tried immediately:

```bash
cargo run --example native_advanced --no-default-features --features native
cargo run --example native_finetune --no-default-features --features native
cargo run --example native_prompt --no-default-features --features native
```

### Backend-independent architecture examples

The backend-independent `jepa` module supplies image and video joint-embedding
predictive architectures (I-JEPA and tube-masked V-JEPA), including a trainable
online encoder/predictor and exponential-moving-average target encoder. The
`liquid` module supplies Euler, Heun, and RK4 Neural ODE solvers, liquid
time-constant and closed-form continuous-time (CfC) recurrent cells, irregular
time-series inference, and a compact stacked LFM with trainable readout. Both
examples run without Candle:

```bash
cargo run --example jepa --no-default-features
cargo run --example liquid_networks --no-default-features
```

### Attention examples

```bash
cargo run --example vec_attention --no-default-features
cargo run --example context_parallel_attention --no-default-features
```

See the **[attention usage book](docs/attention_usage_book.MD)** for complete
VecAttention and context-parallel examples, tensor shapes, validation,
research references, current capabilities, and the roadmap for each feature.

Internet-backed examples download the Rust logo for I-JEPA/V-JEPA training and
the public daily-minimum-temperature dataset for liquid forecasting:

```bash
cargo run --example internet_jepa --no-default-features --features native
cargo run --example internet_liquid --no-default-features --features native
```

These examples cite and follow the official Meta I-JEPA/V-JEPA repositories and
the official Liquid Time-Constant Networks implementation in their module-level
documentation. They require network access on each run.

`MeanEncoder` is an intentionally small reference encoder. Implement
`PatchEncoder` to connect JEPA masking and prediction to a CNN, ViT, or another
production encoder.

Model-backed examples accept arguments after `--`:

```bash
# Inspect the machine, select a compatible model, and optionally download it.
cargo run --example select_huggingface_model \
  --no-default-features --features native -- "Llama"

# Generate with a Hugging Face snapshot already downloaded to MODEL_DIR.
cargo run --example native_generate \
  --no-default-features --features native -- MODEL_DIR "Once upon a time"

# Select, download, and run a model. Both arguments are optional.
cargo run --example auto_native_generate \
  --no-default-features --features native -- "TinyLlama" "Hello"

# Fine-tune the LM head of a local snapshot with LoRA.
cargo run --example native_finetune \
  --no-default-features --features native -- MODEL_DIR

# Download and run Contrastive-LM/CLM-v0.1-8B directly from Hugging Face.
cargo run --release --example clm_generate \
  --no-default-features --features native -- "Your prompt"
```

After selecting a model, the selection example asks whether to download it. If
you answer `y`, it downloads the snapshot into the Hugging Face cache and prints
a `native_generate` command containing the cache path so you can run it later.

The selection and automatic-generation examples require network access.
Set `HF_TOKEN` before running them when access to a gated or private Hugging
Face model is required. Local model directories must contain the model config,
tokenizer, and safetensors files expected in a Hugging Face snapshot.

`clm_generate` is a ready-to-run entry point for
[`Contrastive-LM/CLM-v0.1-8B`](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B).
It downloads and caches the official config, tokenizer, and all safetensors
shards on its first run, then performs generation with the built-in native
Llama/Qwen-family decoder, including Qwen3 per-head Q/K normalization. F16 and
BF16 matrix weights remain in their checkpoint
precision during inference instead of being expanded to `f32`, so the 8B
checkpoint uses roughly 16 GB for weights plus working memory. Use a release
build as shown above; set `HF_TOKEN` if the Hub requires authentication.
Set `CLM_MODEL_DIR` to a downloaded snapshot directory to run fully offline
and skip all Hub requests.
Set `CLM_QUANTIZATION=int8` or `CLM_QUANTIZATION=int4` to reduce resident matrix
weight memory. Quantized loading converts matrices one checkpoint shard at a
time, so it does not require enough RAM to hold the full half-precision model
first. Native dense matrix-vector products use all
available CPU cores; control their number with Rayon's `RAYON_NUM_THREADS`.
Applications can use the same implementation through `ContrastiveLm::from_hub`
or load an existing snapshot without network access with
`ContrastiveLm::from_dir`, then call `generate` repeatedly without reloading
the model. See the [complete CLM guide](docs/contrastive_lm.MD) for authenticated
downloads, offline snapshots, quantization, library examples, resource
requirements, and troubleshooting.

## Native inference runtime

Lighter also exposes a backend-neutral autoregressive runtime in
`candlelighter::native`. It does not depend on Candle and can be built on its
own:

```bash
cargo build --no-default-features --features native
```

Implement `NativeBackend` for the tensor runtime of your choice, then use
`NativeEngine` for continuous batching and fair request scheduling. The engine
supports seeded sampling, greedy decoding, temperature, top-k, top-p and min-p,
presence/frequency/repetition penalties, multiple EOS and stop token IDs,
cancellation, token log probabilities, and per-request generation limits.
`HuggingFaceConfig` reads the common `config.json` fields and preserves unknown
fields, allowing backend implementations to support newer Hugging Face model
architectures without waiting for this crate to update its schema.

`HuggingFaceArtifacts::from_dir` discovers single-file and sharded safetensors
snapshots. `HuggingFaceArtifacts::from_hub` downloads config, tokenizer, index,
and every referenced weight shard. Implement `NativeModel` for a tensor runtime
and wrap it in `HuggingFaceBackend` to get production Hugging Face tokenization,
true batched forward calls, stop strings, bad-token masking, logit bias, and
per-sequence cache cleanup. Use `NativeEngine::step` in a server event loop or
`run_to_completion` for synchronous jobs.

For a fully built-in Candle-free path, `NativeLlama::load(&artifacts)` loads
F32, F16, or BF16 safetensors for dense Llama, Mistral, and Qwen2-family
decoders. It implements grouped-query/multi-query attention, rotary position
embeddings, gated SiLU MLPs, RMS normalization, tied embeddings, projection
biases, Llama 3 extended-context RoPE, Mistral sliding-window attention, and
per-request KV caching. Wrap it with `HuggingFaceBackend` and pass
the backend to `NativeEngine`. This built-in implementation is a portable CPU
reference; high-throughput GPU implementations should implement `NativeModel`
and can reuse the rest of the native serving stack unchanged.

### Automatic model selection

`ModelSelector` detects the host operating system, architecture, logical CPU
count, total/available memory, accelerator memory, AVX2/NEON support, and NVIDIA,
AMD, or Apple accelerators. It can
query the Hugging Face model API, inspect lightweight model configs, filter out
private or gated repositories, estimate runtime memory from safetensors
parameter metadata, and select the largest compatible model within a configured
memory budget.

```bash
# Select a current text-generation model and choose whether to download it.
cargo run --example select_huggingface_model \
  --no-default-features --features native -- "Llama"

# Generate using an already downloaded Hugging Face snapshot.
cargo run --example native_generate \
  --no-default-features --features native -- ./model "Once upon a time"

# Or select, download, and run a compatible model end to end.
cargo run --example auto_native_generate \
  --no-default-features --features native -- TinyLlama "Hello"

# Exercise quantization, paged KV cache, PPO, DPO, GAE, and distillation.
cargo run --example native_advanced \
  --no-default-features --features native

# Exercise LoRA, AdamW, supervised fine-tuning, and reward-model training.
cargo run --example native_finetune \
  --no-default-features --features native

# Run an actual NativeLlama LM-head LoRA/SFT update on a local snapshot.
cargo run --example native_finetune \
  --no-default-features --features native -- ./model
```

### Training and memory capabilities

`native_advanced` contains backend-neutral production primitives for symmetric
group-wise Int8/packed-Int4 matrix quantization, copy-on-write paged KV caches
with prefix sharing, terminal-aware generalized advantage estimation, clipped
PPO policy/value objectives, DPO preference loss, and temperature-scaled
teacher/student distillation. Objective functions return explicit gradients so
an implementation of `OptimizationTarget` can apply them through its own
autograd engine, adapter trainer, or remote optimizer.

`native_training` adds deterministic LoRA initialization and inference,
adapter backpropagation, gradient-clipped AdamW, label-smoothed causal-language
model loss, a `FineTunableTransformer` training contract, supervised fine-tune
steps, and a trainable pairwise reward/value head. This keeps transformer
architecture execution pluggable while providing the full loss, gradient, and
optimizer path required for PEFT, QLoRA-style adapters, SFT, reward modeling,
PPO critics, and preference optimization.

### Prompt languages, grammars, agents, and tools

`native_prompt` provides LMQL/Guidance-style variable binding, generation and
selection directives; SGLang/LCEL-style chained transforms and fork/join state;
token-prefix constraints for choices and JSON; strict regex completion; a GBNF
subset for finite command and schema languages; DSPy-style typed signatures;
OpenAI JSON and ReAct tool-call parsing; and serializable MCP JSON-RPC request
and response types. `GenerateRequest::constraint` accepts a serializable
choice, JSON, regex, or finite-GBNF constraint. `NativeEngine` decodes every
candidate token, masks prefixes rejected by the compiled constraint, and stops
as soon as the result is complete. `OutputConstraint` remains the extension
point for custom CFG/FSM engines. Prompt `gen` directives accept `json`,
`regex=...`, or `choices=a|b` using the same constraint implementations.

```bash
cargo run --example native_prompt \
  --no-default-features --features native
```

The default feature set keeps the existing Candle training API enabled. To use
both runtimes explicitly, select `--features candle,native`.

**MAINTAINERS AND CONTRIBUTORS ARE HIGHLY WELCOME**


**Note:** It is by far not production ready and is only used for own training purposes. No warranty and liability is given. I am a private person and not targeting any commercial benefits.


# Supported Layer types

| Meta Layer | Type         |      State    |  Example      |
|-----| --------------|---------------|---------------|
| Sequential model | - |   &#9989;     |     |
| - | Feature scaling      |  &#x1F3C3;     | [DNN](./lib/examples/simple_dnn.rs) and [TNN](./lib/examples/simple_tnn.rs)             |
| - | Dense        |  &#9989;      | [DNN](./lib/examples/simple_dnn.rs)           |
| - | Convolution  |  &#9989;      | [CNN](./lib/examples/simple_cnn.rs)           |
| - | Pooling      |  &#9989;      | -             |
| - | Normalization|  &#9989;      | -             |
| - | Flatten      |  &#9989;      | -             |
| - | Recurrent    |  &#9989;      | [RNN](./lib/examples/simple_rnn.rs) 1st throw |
| - | Regulation   |  &#9989;      | -             |
| - | Recurrent    |  &#9989;      | [RNN](./lib/examples/simple_rnn.rs) 1st throw |
| - | [Autoencoder](./docs/autoencoder.MD)     | &#x1F3C3;    | -             |
| - | [Feature embedding](./docs/embedding.MD)     |  &#9989;      | [S2S](./lib/examples/simple_s2s.rs) 1st throw |
| - | [Attention](./docs/attention.MD)    |  &#x1F3C3;    | [TNN](./lib/examples/simple_tnn.rs) 1st throw  |
| - | [Mixture of Experts](./docs/moe.MD)   |  &#x1F3C3;    | [ENN](./lib/examples/simple_enn.rs) 1st throw             |
| - |  [Feature masking and -quantization](./docs/masking.MD)   |  &#x1F3C3;    | -             |
| - |  [KAN-Dense](https://www.holeoftherabbit.com/2024/06/16/may-kan-will-be-the-next-ai-disruption-step/)   |  &#x1F3C3;    | -             |
| [Model fine tuning  (PEFT)](https://www.holeoftherabbit.com/2024/06/14/fine-tuning-as-playfield/)  |  -  |  &#x1F3C3;    | *In development:* [DNN2](./lib/examples/simple_dnn.rs) & [DNN3](./lib/examples/simple_dnn.rs)            |
| Parallel model (in sense of split) |  -   |  &#x1F3C3;    | [PNN](./lib/examples/simple_pnn.rs) 1st throw           |
| Parallel model |  [Merging](./docs/modelmerging.MD)      |  &#x1F3C3;    | [PNN](./lib/examples/simple_pnn.rs) 1st throw             |
| Transformer models |  [see](./docs/transformers.MD)     |  &#x1F3C3;    |          |
| * BERT |  Text similarity    |  &#9989;  |   [LLM](./lib/examples/simple_llm.rs)       |
| * LLAMA |  Completion (Chat)    |  &#9989;  |   [LLM2](./lib/examples/simple_llm.rs)       |
| Reinforcement models |  [see](./docs/reinforcement.MD)     |  &#x1F3C3;    |          |


# License
Tripple-licensed to be compatible with the Rust project and the source roots.

Licensed under the [MPL 2.0](./LICENSE), [MIT license](http://opensource.org/licenses/MIT) or the [Apache license, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0) at your option.
