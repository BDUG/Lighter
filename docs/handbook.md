# Lighter handbook

A consolidated reference for setup, capabilities, concepts, model hosting, graph-to-text, G-Retriever, Python Keras comparisons and complete source examples.

## Contents

- [How to use this handbook](#how-to-use-this-handbook)
- [Data and training explained](#data-and-training-explained)
- [Generation and retrieval explained](#generation-and-retrieval-explained)
- [Usage and capability reference](#usage)
- [Model hosting and the HTTP API](#host)
- [Graph-to-text](#graph)
- [G-Retriever integration](#retriever)
- [Contrastive-LM runner](#clm)
- [Python Keras comparison](#keras)
- [Feature data structures](#feature-shapes)
- [Integrating custom layers](#custom-layers)
- [Autoencoders](#autoencoders)
- [Attention mechanisms](#attention)
- [Embeddings and positional information](#embeddings)
- [Mixture of Experts](#mixture-of-experts)
- [Masking and quantization](#masking)
- [Model merging and ensembles](#model-merging)
- [Reinforcement learning objectives](#reinforcement)
- [Upstream transformer catalog](#transformer-catalog)
- [Native Transformers parity and remaining stages](#transformers-parity)
- [Graph-to-text instruction fine-tuning](#graph-finetuning)
- [ARM vector, matrix and NPU backends](#arm-backends)
- [How ARM acceleration integrates with Candle](#candle-arm-integration)
- [Complete code examples](#examples)

## How to use this handbook

This is the consolidated documentation for Lighter. Commands assume the repository
root. Full source listings are included near the end, so a reader can inspect both
an API recipe and the complete program that surrounds it. The examples directory
also contains a short run guide. The G-Retriever source retains its own upstream
README and license under `third_party/`; those files describe the external research
implementation rather than another Lighter manual.

### Choose a workflow

| Goal | Start with | Why this path fits |
| --- | --- | --- |
| Learn layers and train a small numerical model | [Candle recipes](#usage-candle-layers-and-training) | Shapes, forward passes and gradients are easy to inspect on generated data. |
| Generate language from existing compatible weights | [Native loading](#usage-native-generation-and-model-loading) | Load one tokenizer/model pair and reuse the engine across requests. |
| Expose a model to applications | [HTTP host](#host) | The host adds admission, sampling, streaming, cancellation and usage reporting around the runtime. |
| Describe every selected graph fact exactly | [Graph-to-text](#graph) | Deterministic realization preserves identities, literal values and relation direction. |
| Answer a question using a relevant part of a graph | [G-Retriever](#retriever) | Retrieval selects evidence; a GNN supplies a learned graph embedding to the Python language model. |
| Compare Rust concepts with familiar Python APIs | [Keras comparison](#keras) | The mappings distinguish shared concepts from different numerical behavior. |
| Work with image/video masking or irregular time steps | [JEPA](#usage-jepa-image-and-video-learning) and [liquid networks](#usage-liquid-networks-and-neural-odes) | These modules do not require the Candle or native features for their core examples. |

Start with a self-contained example before supplying a downloaded model. A passing
small example verifies your toolchain and the demonstrated API; it does not establish
that a particular pretrained checkpoint or dataset is compatible.

### Understand the three execution paths

The Candle path represents inputs as tensors and composes layer forward passes. The
native path represents prompts as token IDs and schedules autoregressive decoding.
The G-Retriever soft-prompt path represents graphs as PyG data and feeds a learned
floating-point embedding directly into a Transformers causal LM. These paths can
share concepts and data, but their model objects and checkpoints are different.

For example, a selected property graph can move from Python retrieval into Rust exact
verbalization because both accept the same graph schema. A learned GNN projector
checkpoint cannot be loaded as a native Rust Llama checkpoint: it contains graph
encoder/projector weights, and the native decoder currently lacks that input-embedding
interface. Choose the execution path before choosing a checkpoint format.

The `candle` and `native` Cargo features select dependencies at build time. They are
not CPU/GPU switches. `server` enables the HTTP host and includes `native`. Python
requirements are installed independently of Cargo. The Keras and G-Retriever examples
use separate environments because their dependency constraints differ.

### Read capability status and example output

“Implemented” means an executable API exists in this checkout. “Experimental” means
the implementation has limitations described with its recipe. “Upstream reference”
means the linked architecture exists in an external project and is not a promise of
a Lighter wrapper. The historical architecture notes later in this handbook are
retained as context; the capability table and current recipes describe working scope.

Examples using tiny backends or random models verify plumbing. A generated string
from a random model has no factual meaning. Shapes, finite losses, gradients and
checkpoint round trips are the useful outputs of those demonstrations. A downloaded
pretrained model may produce useful prose, but graph fidelity and task accuracy still
need evaluation against your data.

## Data and training explained

### Trace a numerical sample through a model

A dense layer takes a feature vector and computes a weighted projection plus bias.
With an input tensor shaped `[batch, features]`, a layer with four outputs produces
`[batch, 4]`. An activation transforms those values; it does not add another batch or
time dimension. A second dense layer can map the four hidden values to a prediction.

Suppose an example learns `y = x1 + x2`. The model receives two features per sample,
returns one prediction and compares it with one target. Mean squared error measures
squared differences. Backpropagation computes derivatives of the loss with respect
to trainable weights; the optimizer applies an update. Repeating this process can
reduce error on the training samples. Evaluate on separate samples to check whether
that improvement extends beyond the examples used for updates.

The legacy `fit` wrapper uses its documented MSE behavior. Choosing a different loss
name does not establish that its training loop implements that objective. For a
classification task, inspect the classification examples and actual loss path rather
than treating every Keras compile option as interchangeable.

### Keep shape, axis and identity separate

A tensor shape describes how values are arranged; an axis describes what a dimension
means. Two arrays containing the same numbers can compute different results if one
interprets a dimension as channels and another as time steps.

| Input | Typical interpretation | Check before execution |
| --- | --- | --- |
| Dense `[B,F]` | `B` samples, `F` features | Layer input width matches `F`. |
| Legacy feature helper `[B,T,F]` | Samples, time steps, features | A singleton `T` is still a dimension. |
| Image `[B,C,H,W]` | Samples, channels, height, width | The selected convolution expects channels first. |
| Token IDs | Integer vocabulary indices | Use embedding indices, not floating-point feature vectors. |
| PyG node features `[N,D]` | `N` graph nodes, `D` embedding values | `D` matches the GNN input width. |
| PyG edge index `[2,E]` | Source/target indices for `E` edges | Both rows reference the current node array. |

Flattening changes layout into a vector; it does not create new information. Pooling
reduces resolution by aggregating values. Reshaping must preserve the intended ordering,
otherwise later layers see a different representation even if element counts agree.

Graph IDs are a different kind of identity: `"ada"` identifies an entity regardless
of its row index. Retrieval remaps selected node rows for the GNN, while the exported
property graph retains original IDs. This is why the retrieval metadata matters when
matching evidence back to the original graph.

### Fit preprocessing on training data

Feature scaling makes numerical magnitudes comparable. Min-max scaling uses the
observed range; standardization uses a mean and standard deviation. Fit those
statistics on training data, then reuse them for validation and inference. Recomputing
statistics on each prediction batch changes the model's input distribution.

Lighter's documented global scaling and Keras per-feature normalization can differ:
for a two-column matrix, a global minimum is one number for the entire matrix, while
per-column normalization learns separate statistics for each feature. Compare the
axes and the actual transformed values before comparing model predictions. Preserve
the preprocessing configuration alongside weights if the application needs to reproduce
training-time inputs.

### Distinguish weights, adapters and model artifacts

A checkpoint may contain weights without the architecture or tokenizer needed to use
them. A Hugging Face native snapshot combines configuration, tokenizer and supported
weights. A Candle weight round trip assumes the matching model structure. A
G-Retriever adapter contains the graph encoder/projector and relies on the same base
LM and embedding encoder at inference time.

Freezing a language model prevents optimizer updates to its parameters. It does not
prevent gradients from passing through its computations to a trainable soft prompt.
The G-Retriever integration uses exactly this distinction. LoRA instead adds trainable
low-rank parameter updates; its location and coverage depend on the implementation.
The native fine-tuning example's LM-head LoRA is narrower than tuning every attention
projection in a language model.

## Generation and retrieval explained

### Follow one hosted request

A client submits a prompt and sampling options. The host validates the model name,
request fields and limits, then admits the request into its bounded queue. The worker
owns the backend, tokenizes the prompt and advances active requests through decoding.
Sampling chooses the next token from logits; completion stops at an allowed end token,
a configured stop condition or an output limit.

The prompt and generated tokens both occupy context. `max_tokens` limits generated
output; it does not give the prompt unlimited space. Increasing it can cause a request
to exceed the model context or consume more memory and time. Temperature affects
sampling randomness, not model knowledge. Greedy decoding can be repeatable while
still producing an incorrect answer.

Streaming returns pieces of the same generation as they become available. Network
chunks are not necessarily complete words or tokens. Clients should assemble the
content fields in order and handle the final event; rendering each raw transport chunk
as a separate sentence corrupts the output. Usage counts describe tokens processed,
not factual completeness.

The CPU host provides API behavior familiar from model servers, but its throughput
comes from the backend it wraps. Continuous batching and queue management do not turn
a CPU reference decoder into a GPU execution engine. Measure time to first output,
total latency and tokens per second on the actual model and hardware.

### Separate graph evidence selection from realization

Exact graph-to-text planning selects facts according to traversal, radius and an
optional fact budget. Each selected fact has an identity and a deterministic textual
realization. If every original fact is needed, use an unrestricted plan and check that
there are no omissions. A small `max_facts` budget can exclude evidence even though
the underlying graph is valid.

Language-model realization asks a model to express selected facts in prose. The
input plan explains which facts were supplied, but it cannot prove which facts the
model expressed correctly. Compare generated entity names, relation direction,
attributes and values against the plan when assessing fidelity.

G-Retriever adds question-guided evidence selection before learned realization. A
question embedding is compared with node and relation embeddings. Cosine scores rank
relevance; ranking prizes encourage the solver to retain useful evidence while edge
costs discourage unnecessary connections. Similarity is a retrieval signal, not a
proof that a relation answers the question.

### Work through an edge-prize transformation

The upstream algorithm needs to reward useful relations as well as useful nodes.
The PCST solver accepts node prizes and edge costs, so relation prizes are transformed:

| Relation prize | Edge cost | Transformed representation |
| --- | --- | --- |
| `0.2` | `0.5` | Keep the edge with cost `0.3`. |
| `0.8` | `0.5` | Add a virtual node with prize `0.3`, connected to both endpoints by zero-cost edges. |

The solver trades collected prizes against connection costs. Its selected tree can
include a lower-ranked node because that node connects highly rewarded evidence.
After solving, virtual nodes are removed and their original relations restored. The
result is a selected subgraph with original IDs, not a graph containing new fictional
entities. The solver uses undirected connectivity; the exported relation still retains
its original source and target.

G-Retriever runs an unrooted, one-component PCST procedure. It is an approximate
retrieval method, so a relevant fact can be omitted. Check selected and omitted IDs,
then test retrieval recall against labeled evidence. Increasing the rank counts or
reducing costs may retain more context, but also introduce irrelevant evidence and
longer prompts.

### Understand the learned graph soft prompt

The GNN updates node representations using their neighborhood structure. Mean pooling
turns all node representations in one retrieved graph into one vector. The projector
maps that vector into the language model's embedding width. The integration inserts
this vector as one soft token alongside the textual graph description and question.
It therefore supplies both structural conditioning and explicit text evidence.

This soft token is a continuous vector, not a tokenizer vocabulary ID. Its usefulness
depends on training the GNN/projector for the chosen base LM and encoder. Random
projector weights do not acquire graph knowledge merely because the language model
is pretrained. Conversely, a trained adapter is tied to its representation choices;
loading it with a different sentence encoder can change what every graph feature means.

### Evaluate each stage independently

| Stage | Useful check | What a failure suggests |
| --- | --- | --- |
| Input | Unique IDs, valid endpoints, finite properties and correct shapes | Fix the data before changing model parameters. |
| Retrieval | Recall of annotated answer-bearing entities/relations | Adjust embeddings, prizes or graph construction. |
| Prompt construction | Evidence survives token truncation and fits context | Change limits or select less irrelevant context. |
| Training | Finite loss and gradients in intended parameters | Inspect label masks, frozen parameters and batch shapes. |
| Generation | Answer accuracy and graph-fact fidelity | Evaluate the trained adapter/base LM against held-out questions. |
| Hosting | Queue saturation, latency, cancellation and token throughput | Tune admission and runtime settings for real workloads. |

A falling training loss does not establish retrieval recall or factual generation.
The included offline tests validate computation and interfaces; a production task
needs held-out data and application-specific accuracy checks.

<a id="usage"></a>

## Usage and capability reference

Lighter (`candlelighter`) combines a Keras-inspired layer/model API on top of
[Candle](https://github.com/huggingface/candle) with a portable native CPU
language-model runtime. JEPA and liquid-network modules can run without either
feature. This is an experimental learning project; implemented APIs do not imply
production readiness.

<a id="usage-complete-examples-and-python-keras"></a>

### Complete examples and Python Keras

For complete programs instead of individual recipes, use the
[code cookbook](#examples). It includes full source listings for every
standalone example and all historical Candle training examples, with commands,
feature requirements, prerequisites, and validation status. Every listing includes
its imports and executable entry point or original library functions.

The [Python Keras comparison](#keras) maps the APIs and actual behavior,
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

<a id="usage-setup-and-feature-selection"></a>

### Setup and feature selection

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

<a id="usage-capability-table"></a>

### Capability table

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
| Feature shaping: samples, time steps, spatial features | Candle / implemented | [Layer tour](../examples/handbook_layers.rs), [shapes](#usage-feature-shapes-and-scaling) |
| Min-max and z-score feature scaling, reuse and inverse transforms | Candle / implemented with edge cases | [Scaling recipe](#usage-feature-shapes-and-scaling), [DNN](../lib/examples/simple_dnn.rs), [TNN](../lib/examples/simple_tnn.rs) |
| Dense layers | Candle / implemented | [Layer tour](../examples/handbook_layers.rs), [DNN](../lib/examples/simple_dnn.rs) |
| Linear, ReLU, SiLU, sigmoid, log-softmax activations | Candle / implemented | [Layer tour](../examples/handbook_layers.rs), [activation semantics](#usage-dense-layers-and-training) |
| 1D and 2D convolution | Candle / implemented with explicit kernel | [Layer tour](../examples/handbook_layers.rs), [CNN](../lib/examples/simple_cnn.rs), [2D recipe](#usage-convolution-pooling-normalization-and-flatten) |
| Average and max pooling | Candle / experimental: enum names reversed | [Layer tour](../examples/handbook_layers.rs), [pooling behavior](#usage-convolution-pooling-normalization-and-flatten) |
| Normalization | Candle / experimental: wrapper returns input | [Layer tour](../examples/handbook_layers.rs), [explicit L2 recipe](#usage-convolution-pooling-normalization-and-flatten) |
| Flatten | Candle / implemented, flattens all dimensions | [Layer tour](../examples/handbook_layers.rs) |
| LSTM and GRU recurrence | Candle / implemented one-step wrapper | [Layer tour](../examples/handbook_layers.rs), [RNN](../lib/examples/simple_rnn.rs) |
| Regulation / regularization | Candle / no dedicated Lighter layer | [Dropout and weight-decay recipe](#usage-regularization) |
| Sequential model, forward, fit, predict, summary | Candle / implemented with MSE training | [Layer tour](../examples/handbook_layers.rs), [training recipe](#usage-dense-layers-and-training) |
| SGD and Adam training | Candle / implemented; Adam uses Candle AdamW | [Optimizer recipe](#usage-dense-layers-and-training) |
| MSE, NLL, BCE-with-logits, cross-entropy losses | Candle / only MSE accepted by `fit` | [Loss recipe](#usage-dense-layers-and-training) |
| Model JSON and JSON/safetensors weights | Candle / experimental round-trip support | [Persistence recipe](#usage-saving-and-loading), [existing tests](../tests/test_importexport.rs) |
| Autoencoder | Candle / composition; dedicated VAE planned | [Layer tour](../examples/handbook_layers.rs), [autoencoder recipe](#usage-autoencoders) |
| Standard feature embedding | Candle / implemented | [Layer tour](../examples/handbook_layers.rs), [S2S](../lib/examples/simple_s2s.rs) |
| Timestep, absolute, rotary, sinusoidal embedding layers | Candle wrappers / planned | [Embedding alternatives](#usage-recurrent-layers-and-embeddings); native decoder already has RoPE |
| Self-attention | Candle / experimental shape-sensitive wrapper | [Attention recipe](#usage-attention-and-mixture-of-experts), [TNN](../lib/examples/simple_tnn.rs) |
| Cross, causal, multi-head/query, grouped-query attention wrappers | Candle / planned beyond current wrapper | [Attention alternatives](#usage-attention-and-mixture-of-experts); native decoder implements causal GQA/MQA |
| Sparse Mixture of Experts | Candle / experimental | [MoE recipe](#usage-attention-and-mixture-of-experts), [ENN](../lib/examples/simple_enn.rs) |
| Multi-task, gated-training, hierarchical, conditional MoE variants | Candle / planned | [MoE alternatives](#usage-attention-and-mixture-of-experts) |
| Feature/sequence masking | Candle / planned dedicated layer | [Masking recipe](#usage-masking-and-kan-dense); JEPA block/tube masks are implemented |
| Feature/weight quantization | Native / implemented Int8 and packed Int4; Candle feature layer planned | [Quantization example](../examples/native_advanced.rs), [recipe](#usage-quantization-caches-and-reinforcement-objectives) |
| KAN-Dense | Planned | [KAN design recipe](#usage-masking-and-kan-dense) |
| Candle LoRA and DoRA dense variants (PEFT) | Candle / experimental | [PEFT constructor recipe](#usage-candle-peft), [DNN2/DNN3](../lib/examples/simple_dnn.rs) |
| Parallel split and merge models | Candle / experimental | [Parallel recipe](#usage-parallel-split-and-merge), [PNN](../lib/examples/simple_pnn.rs) |
| BERT text similarity | Candle / implemented wrapper; checkpoint setup required | [BERT recipe](#usage-bert-and-candle-llama), [LLM](../lib/examples/simple_llm.rs) |
| Candle Llama completion | Candle / implemented wrapper; checkpoint setup required | [Llama recipe](#usage-bert-and-candle-llama), [LLM2](../lib/examples/simple_llm.rs) |
| Other Candle transformer architectures | No dedicated Lighter wrapper | [Transformer scope](#usage-bert-and-candle-llama), [upstream model catalog](#transformer-catalog) |
| Model hosting: OpenAI completion/chat API, incremental SSE, discovery | Server feature / implemented CPU host | [Hosting guide](#host), [embedding source](../examples/serve_model.rs) |
| Host authentication, health/metrics, bounded admission, context/timeout/cancellation | Server feature / implemented | [Hosting configuration](#host-limits-streaming-and-failure-behavior) |
| JSON-object output mode | Native/server / object-root constraint | [API recipe](#host-curl-examples); token limits/stop strings can truncate output |
| Continuous batching, fair scheduling, step/run-to-completion | Native / implemented; tensor batching depends on backend | [Native tour](../examples/handbook_native.rs) |
| Greedy/seeded temperature, top-k/top-p/min-p sampling | Native / implemented | [Native tour](../examples/handbook_native.rs), [sampling recipe](#usage-sampling-and-scheduling) |
| Presence/frequency/repetition penalties, bad tokens, logit bias | Native / implemented | [Native tour](../examples/handbook_native.rs) |
| EOS/stop tokens/strings, min/max tokens, cancellation, log probabilities | Native / implemented | [Native tour](../examples/handbook_native.rs), [scheduler semantics](#usage-sampling-and-scheduling) |
| Custom tensor backend/model adapter | Native / implemented extension traits | [Demo backend](../examples/handbook_native.rs), [adapter recipe](#usage-snapshots-and-custom-backends) |
| Hugging Face config, tokenizer, single/sharded safetensors downloads | Native / implemented | [Snapshot recipe](#usage-snapshots-and-custom-backends), [native_generate](../examples/native_generate.rs) |
| Dense Llama/Mistral/Qwen-family decoder | Native / implemented portable reference | [Generation recipe](#usage-native-generation-and-model-loading), [native_generate](../examples/native_generate.rs) |
| F32/F16/BF16 weights, RMSNorm, SiLU MLP, RoPE, sliding window, KV cache | Native decoder / implemented | [Decoder scope](#usage-snapshots-and-custom-backends), [CLM guide](#clm) |
| ARM F32 NEON/SVE vector kernels and SME matrix kernels | Native / runtime dispatch; optional Linux SVE/SME | [ARM kernel recipe](#arm-backends), [arm_acceleration](../examples/arm_acceleration.rs) |
| Arm NN / TFLite NPU external delegates | Native / compiled full-prefix models; vendor SDK required | [NPU recipe](#arm-backends), [generation](../examples/arm_npu_generate.rs), [hosting](../examples/arm_npu_host.rs) |
| CPU/memory/accelerator detection and compatible-model selection | Native / implemented | [Selection recipe](#usage-model-selection), [select_huggingface_model](../examples/select_huggingface_model.rs) |
| Automatic model download and generation | Native / implemented, network/model required | [auto_native_generate](../examples/auto_native_generate.rs), [selection recipe](#usage-model-selection) |
| CLM v0.1 8B, cached/offline/repeated generation and quantized loading | Native / implemented | [CLM recipe](#usage-contrastive-lm), [complete guide](#clm) |
| Paged KV cache, shared prefixes, copy-on-write, truncation | Native / implemented primitive | [native_advanced](../examples/native_advanced.rs), [cache recipe](#usage-quantization-caches-and-reinforcement-objectives) |
| GAE, clipped PPO, DPO preference loss | Native / implemented objectives/gradients | [native_advanced](../examples/native_advanced.rs), [reinforcement recipe](#usage-quantization-caches-and-reinforcement-objectives) |
| Distillation and optimizer update hooks | Native / implemented primitives | [native_advanced](../examples/native_advanced.rs), [update recipe](#usage-quantization-caches-and-reinforcement-objectives) |
| LoRA initialization, forward/backward, gradient accumulation, SGD | Native / implemented | [native_finetune](../examples/native_finetune.rs), [fine-tuning recipe](#usage-fine-tuning-and-reward-modeling) |
| Gradient-clipped AdamW, label-smoothed causal LM loss, SFT | Native / implemented | [native_finetune](../examples/native_finetune.rs) |
| NativeLlama LM-head LoRA update | Native / implemented, local snapshot required | [Snapshot SFT command](#usage-fine-tuning-and-reward-modeling) |
| Native Transformers-style factories, tokenizer batches, generation config and pipeline | Native / stage 1 implemented | [Parity guide](#transformers-parity), [pipeline source](../examples/transformers_pipeline.rs) |
| Repeated n-gram and multi-token bad-word restrictions | Native/server / implemented | [Generation behavior](#transformers-parity) |
| Graph2Text instruction SFT, answer-only labels, triple shuffling, head-adapter checkpoints | Native / implemented, local snapshot required | [Complete workflow](#graph-finetuning), [program](../examples/graph2text_finetune.rs) |
| Reward/value head, pairwise preference updates | Native / implemented | [native_finetune](../examples/native_finetune.rs) |
| Prompt variables, generation and choice directives | Native / implemented | [native_prompt](../examples/native_prompt.rs), [prompt recipe](#usage-prompts-grammars-workflows-and-tools) |
| Chained transforms, state fork/join | Native / implemented; branches execute sequentially | [Native tour](../examples/handbook_native.rs) |
| Choice, JSON, regex and finite GBNF output constraints | Native / implemented subset; regex completion only | [Native tour](../examples/handbook_native.rs), [constraint recipe](#usage-prompts-grammars-workflows-and-tools) |
| Typed signatures, OpenAI/ReAct tool parsing, MCP JSON-RPC types | Native / implemented parsing/serialization | [native_prompt](../examples/native_prompt.rs), [Native tour](../examples/handbook_native.rs) |
| I-JEPA image and V-JEPA tube masking / prediction | Feature-independent / implemented | [jepa](../examples/jepa.rs), [JEPA recipe](#usage-jepa-image-and-video-learning) |
| JEPA online encoder/predictor, EMA target training | Feature-independent / implemented compact trainer | [jepa](../examples/jepa.rs), [internet_jepa](../examples/internet_jepa.rs) |
| Euler, Heun, RK4 Neural ODE solvers | Feature-independent / implemented | [Solver recipe](#usage-liquid-networks-and-neural-odes), [liquid_networks](../examples/liquid_networks.rs) |
| Liquid time-constant and CfC recurrent cells, irregular time steps | Feature-independent / implemented | [Cell recipe](#usage-liquid-networks-and-neural-odes), [liquid_networks](../examples/liquid_networks.rs) |
| Stacked LFM and trainable readout | Feature-independent / implemented | [liquid_networks](../examples/liquid_networks.rs), [internet_liquid](../examples/internet_liquid.rs) |

<a id="usage-candle-layers-and-training"></a>

### Candle layers and training

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

<a id="usage-feature-shapes-and-scaling"></a>

#### Feature shapes and scaling

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
[Feature-shape notes](#feature-shapes) describe the conventions.

<a id="usage-dense-layers-and-training"></a>

#### Dense layers and training

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

<a id="usage-convolution-pooling-normalization-and-flatten"></a>

#### Convolution, pooling, normalization, and flatten

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

<a id="usage-recurrent-layers-and-embeddings"></a>

#### Recurrent layers and embeddings

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
See [embedding notes](#embeddings) for the historical roadmap.

<a id="usage-regularization"></a>

#### Regularization

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

<a id="usage-saving-and-loading"></a>

#### Saving and loading

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

<a id="usage-experimental-architectures"></a>

### Experimental architectures

<a id="usage-autoencoders"></a>

#### Autoencoders

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
dedicated variational autoencoder API; [autoencoder notes](#autoencoders) point
to an upstream VAE implementation.

<a id="usage-attention-and-mixture-of-experts"></a>

#### Attention and Mixture of Experts

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
Cross-attention and the other standalone wrappers in [attention notes](#attention)
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
See [ENN](../lib/examples/simple_enn.rs) and [MoE notes](#mixture-of-experts). Multi-task,
hierarchical, conditional, and gated-training variants are planned. A simple
multi-task alternative is to call independent `SequentialModel::forward`s and
retain each output rather than claim a multi-task MoE API.

<a id="usage-masking-and-kan-dense"></a>

#### Masking and KAN-Dense

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

<a id="usage-candle-peft"></a>

#### Candle PEFT

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

<a id="usage-parallel-split-and-merge"></a>

#### Parallel split and merge

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
[merging notes](#model-merging). For a working two-model average, apply equal-width
models independently and combine their tensors explicitly:

```rust
let a = model_a.forward(x.clone());
let b = model_b.forward(x);
let mean = ((a + b)? * 0.5)?;
```

This recipe is an ensemble average, not validation of the `Merge` implementation.

<a id="usage-bert-and-candle-llama"></a>

#### BERT and Candle Llama

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
and multimodal architectures listed in [transformers.MD](#transformer-catalog) are
upstream Candle possibilities, not automatically exposed Lighter capabilities.

<a id="usage-model-hosting"></a>

### Model hosting

Run `lighter-serve` to expose a single loaded native model through OpenAI-compatible
HTTP endpoints. The [hosting guide](#host) includes complete CLI, curl,
Python SDK and embedding examples, supported fields, and vLLM differences.
The optional `server` feature includes `native`. Actual inference uses the existing
CPU decoder; endpoint compatibility does not add vLLM GPU/distributed execution.

```bash
cargo run --locked --release --bin lighter-serve \
  --no-default-features --features server -- \
  --model MODEL_DIR --served-model-name local-model
```

<a id="usage-native-generation-and-model-loading"></a>

### Native generation and model loading

<a id="usage-sampling-and-scheduling"></a>

#### Sampling and scheduling

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

<a id="usage-snapshots-and-custom-backends"></a>

#### Snapshots and custom backends

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

<a id="usage-model-selection"></a>

#### Model selection

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

<a id="usage-contrastive-lm"></a>

#### Contrastive-LM

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
See the [complete CLM guide](#clm) for library examples and failures.

<a id="usage-quantization-caches-and-reinforcement-objectives"></a>

### Quantization, caches, and reinforcement objectives

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

<a id="usage-fine-tuning-and-reward-modeling"></a>

### Fine-tuning and reward modeling

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

<a id="usage-prompts-grammars-workflows-and-tools"></a>

### Prompts, grammars, workflows, and tools

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

<a id="usage-jepa-image-and-video-learning"></a>

### JEPA image and video learning

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

<a id="usage-liquid-networks-and-neural-odes"></a>

### Liquid networks and neural ODEs

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

<a id="usage-validation-and-troubleshooting"></a>

### Validation and troubleshooting

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
memory. For architecture-specific details, use the [CLM guide](#clm)
and the linked source examples. Historical topic notes are useful design context;
the status table above describes the current implementation.

<a id="usage-graph-to-text"></a>

### Graph-to-text

See the [complete graph-to-text guide](#graph) for property graphs, triples, rooted planning, exact realization, native model generation, HTTP streaming and a Python comparison.

| Capability | Status | Example |
| --- | --- | --- |
| Property graph and triple ingestion | Implemented | [Graph CLI](../examples/graph2text.rs) |
| Graph validation, breadth-first planning and fact provenance | Implemented | [Guide](#graph) |
| Exact graph-to-text realization | Implemented | [JSON fixture](../examples/data/graph2text.json) |
| Native graph-grounded language generation | Implemented | [Native example](../examples/graph2text_native.rs) |
| Hosted graph generation and SSE | Implemented (`server`) | [Endpoint guide](#graph-hosted-graph-generation) |

<a id="usage-g-retriever"></a>

### G-Retriever

The [G-Retriever integration guide](#retriever) covers the pinned upstream
implementation, actual PCST retrieval, learned graph soft prompts, training and examples.

| Capability | Status | Example |
| --- | --- | --- |
| Cosine ranking and prize-collecting Steiner-tree retrieval | Implemented (Python) | [Retrieval](../examples/python/g_retriever_retrieve.py) |
| Upstream GCN, GAT and graph Transformer encoders | Vendored and integrated (Python/PyG) | [Core](../examples/python/g_retriever/core.py) |
| Graph soft-prompt generation | Implemented (Python/Transformers) | [Generation](../examples/python/g_retriever_generate.py) |
| Frozen-LM GNN/projector training and checkpoints | Implemented (Python) | [Training](../examples/python/g_retriever_train.py) |
| Retrieved property-graph export to Rust | Implemented | [Guide](#retriever-retrieve-and-export-to-rust) |

The Python integration uses an embedding-input causal LM; native Rust generation can
consume exported graph text but does not inject learned graph embeddings.

<a id="host"></a>

## Model hosting and the HTTP API

`lighter-serve` loads one native model and exposes OpenAI-compatible HTTP completion
and chat endpoints. Like a vLLM server, it keeps weights loaded, accepts concurrent
requests, interleaves decoding in a continuous batch, and can stream SSE responses.
Execution uses Lighter's portable **CPU** decoder. This does not provide vLLM's
GPU kernels, tensor/pipeline parallelism, distributed serving, or throughput.

<a id="host-start-a-local-snapshot"></a>

### Start a local snapshot

The snapshot must contain `config.json`, `tokenizer.json`, and complete single-file
or sharded safetensors weights supported by NativeLlama. Dense Llama, Mistral and
Qwen-family layouts are supported; an arbitrary Hugging Face model cannot be hosted.

```bash
cargo run --locked --release --bin lighter-serve \
  --no-default-features --features server -- \
  --model /models/my-snapshot \
  --served-model-name local-model \
  --bind 127.0.0.1:8000
```

This serves text completions. Enable chat only with an explicit template that
matches the checkpoint:

```bash
cargo run --locked --release --bin lighter-serve \
  --no-default-features --features server -- \
  --model /models/qwen-snapshot --served-model-name local-model \
  --chat-template chatml --quantization int4 \
  --max-batch-size 4 --max-pending-requests 64 \
  --max-input-tokens 4096 --max-tokens 512 \
  --request-timeout-seconds 300 --bind 127.0.0.1:8000
```

`chatml` formats system/user/assistant turns with `<|im_start|>` and `<|im_end|>`;
`llama3` uses begin-of-text/header/end-of-turn tokens. There is no automatic
`tokenizer_config.json` Jinja-template application. Models requiring a different
format need preformatted `/v1/completions` prompts or a separate template
implementation. Explicit beginning-of-sequence tokens are not inserted a second time when the
backend tokenizer has a matching configured BOS token. A ChatML example is not proof that every Qwen checkpoint uses that
exact format; check its model instructions. Without `--chat-template`, chat calls
return a clear 400 error rather than guess the model's format.

The host binds to loopback by default. For access from another machine, select
`--bind 0.0.0.0:8000` and use the network/proxy configuration appropriate to your
deployment. For authenticated serving, set `LIGHTER_API_KEY` securely in the
process environment; clients send `Authorization: Bearer <key>`. The host does not
log the key. TLS and external ingress are handled by a reverse proxy, not this CLI.

<a id="host-start-from-the-hugging-face-hub"></a>

### Start from the Hugging Face Hub

`--model` also accepts a repository ID:

```bash
cargo run --locked --release --bin lighter-serve \
  --no-default-features --features server -- \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --served-model-name tinyllama
```

This downloads/cache-loads a real checkpoint; it is not a model-free smoke test.
TinyLlama's training chat template is not provided by the host's two template modes;
use its correctly formatted text prompt with `/v1/completions`. Hub access and disk/
memory must be available. Use an existing secure `HF_TOKEN` binding for gated/private
models, and set `LIGHTER_API_KEY` separately for client authentication.

`--quantization int8|int4` quantizes matrix weights shard by shard during loading;
normalization vectors remain float32. Quantization is lossy and needs temporary
conversion memory. CPU workers use Rayon; set `RAYON_NUM_THREADS` to limit them.
`--max-batch-size` bounds active sequences but does not imply a fused GPU tensor batch.

<a id="host-endpoint-table"></a>

### Endpoint table

| Endpoint | Behavior |
| --- | --- |
| `GET /health` | Worker readiness and served model. Remains unauthenticated for health checks. |
| `GET /v1/models` | OpenAI-style list containing the single served model ID. |
| `POST /v1/completions` | One string prompt; buffered JSON or incremental SSE. |
| `POST /v1/chat/completions` | Text-only system/user/assistant messages; configured template required. |
| `GET /metrics` | Prometheus-compatible numeric readiness, inflight/active/queued and completed/failed counters. |

When a key is configured, all listed endpoints except `/health` require bearer
authentication. Metrics count inference work, not every rejected HTTP parse/auth
attempt. Health means the model worker is available; it does not assess model quality.

<a id="host-curl-examples"></a>

### curl examples

Model discovery and readiness:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models
```

Text completion:

```bash
curl --fail http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-model","prompt":"Once upon a time","max_tokens":64,"temperature":0.7,"top_p":0.9,"seed":42}'
```

Chat (start the host with the matching `--chat-template` first):

```bash
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-model","messages":[{"role":"system","content":"Answer briefly."},{"role":"user","content":"What is Rust ownership?"}],"max_tokens":64,"temperature":0.0}'
```

Streaming with optional final token usage:

```bash
curl --no-buffer --fail http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-model","prompt":"Explain memory safety:","max_tokens":64,"stream":true,"stream_options":{"include_usage":true}}'
```

Each event is `data: <JSON>`, followed by a finish-reason chunk and `data: [DONE]`.
Chat chunks contain `choices[0].delta`; text chunks contain `choices[0].text`.
Requested final usage appears in a separate chunk with an empty choices array.
For authenticated requests, add `-H "Authorization: Bearer $LIGHTER_API_KEY"`.
Keep values in your local environment; do not commit keys into example commands.

Request a JSON **object** root:

```bash
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-model","messages":[{"role":"user","content":"Return a JSON object with a greeting field."}],"max_tokens":128,"response_format":{"type":"json_object"}}'
```

This uses the new `ConstraintSpec::JsonObject` object-root prefix/completion
constraint. It does not enforce a JSON Schema or guarantee a complete object if
`max_tokens` is exhausted or a user stop string truncates output. Always inspect `finish_reason` and parse the final text.

<a id="host-complete-python-openai-client"></a>

### Complete Python OpenAI client

Install `openai` in a Python environment. The program below targets a host started
with `--served-model-name local-model` and a correct chat template:

```python
import os
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key=os.environ.get("LIGHTER_API_KEY", "unused-local-key"),
)
print([model.id for model in client.models.list().data])

completion = client.completions.create(
    model="local-model",
    prompt="Once upon a time",
    max_tokens=64,
    temperature=0.7,
    seed=42,
)
print(completion.choices[0].text)
print(completion.usage)

chat = client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "Explain Rust ownership."}],
    max_tokens=64,
)
print(chat.choices[0].message.content)

with client.chat.completions.create(
    model="local-model",
    messages=[{"role": "user", "content": "Write a short greeting."}],
    max_tokens=64,
    stream=True,
    stream_options={"include_usage": True},
) as stream:
    for chunk in stream:
        if chunk.choices:
            print(chunk.choices[0].delta.content or "", end="", flush=True)
        if chunk.usage:
            print("\nusage:", chunk.usage)
print()
```

Native sampling extensions can be supplied with the SDK's `extra_body`, for
example `extra_body={"top_k": 40, "min_p": 0.05, "repetition_penalty": 1.1}`.
Closing a streaming connection cancels its request between decode iterations and
releases its backend sequence cache. A request ID is returned in the
`x-request-id` header and completion/chunk body; there is no separate cancellation
HTTP endpoint.

<a id="host-supported-request-fields"></a>

### Supported request fields

Both endpoints support the common fields below, plus either `prompt` or `messages`.
Unknown fields are rejected rather than silently ignored.

| Field | Default / restriction |
| --- | --- |
| `model` | Required; must equal the served model ID. |
| `prompt` | Completion endpoint only; one nonempty string. Arrays and token-ID prompts are unsupported. |
| `messages` | Chat only; nonempty list of `{role, content}` strings. No images, audio, tools or structured content. |
| `max_tokens` | Default `min(256, configured limit)`; positive, at most `--max-tokens`. |
| `temperature`, `top_p`, `seed` | Defaults 0.7, 0.9, 0. Seed is a nonnegative integer; 0 temperature is greedy. |
| `top_k`, `min_p` | Native extensions; optional positive top-k, default min-p 0. |
| `presence_penalty`, `frequency_penalty`, `repetition_penalty` | Defaults 0, 0, 1; repetition penalty must be positive. |
| `stop` | String or string list; at most 16 nonempty strings, each at most 1024 bytes. |
| `n` | Only 1 supported. |
| `stream` | Default false. |
| `stream_options` | `{"include_usage": true}`; valid only with `stream=true`. |
| `response_format` | `{"type":"text"}` or `{"type":"json_object"}`; schema mode unsupported. |

Usage counts the backend's prompt tokens and sampled completion tokens, including
sampled stop/EOS tokens even when omitted from visible text. Standard finish
reasons are `stop` and `length`. This is an OpenAI-compatible **subset**, not a
complete clone: tools/function calling, multimodal inputs, logprobs, embeddings,
multiple choices, beam search, speculative decoding, arbitrary chat templates,
model hot swaps and distributed/GPU execution are not exposed by this host.
Some of these primitives exist elsewhere in Lighter but are not API fields here.

<a id="host-limits-streaming-and-failure-behavior"></a>

### Limits, streaming, and failure behavior

- HTTP bodies are limited to 1 MiB. The total active plus waiting request count is
  bounded by `--max-pending-requests`; full admission returns 429.
- Input token count is checked after tokenization. Prompt plus requested output
  must fit the checkpoint's `max_position_embeddings` when present.
- Timeouts include queue time and are checked between CPU decode operations;
  an in-progress matrix computation cannot be interrupted instantly.
- SSE buffers incomplete Unicode suffixes and possible stop prefixes, emitting
  text stable across decode steps. This introduces a small delay but avoids leaking
  partial stop strings. A backend that rewrites already emitted text causes a
  stream error; use buffered responses for such a custom decoder.
- Stream buffers are bounded. Disconnected/slow consumers are cancelled rather
  than allowing unlimited queued output; an interrupted/error stream may close
  without `[DONE]`. Discard it as an incomplete response.
- Batch-level inference errors fail the affected pending batch, clear caches, and
  allow new requests. Invalid bodies/unsupported fields/limits use OpenAI-style
  error objects. Errors after an SSE response begins appear as a JSON error event.
- Ctrl-C stops admission and drains HTTP connections. Processes are not retained
  in an environment snapshot; start the host again when a task needs it.

<a id="host-embed-the-host-in-rust"></a>

### Embed the host in Rust

A full embedding example is [serve_model.rs](../examples/serve_model.rs):

```bash
cargo run --locked --release --example serve_model \
  --no-default-features --features server -- /models/my-snapshot
```

`model_host::serve(listener, backend, eos_ids, config)` accepts any `NativeBackend +
Send + 'static`. `model_host::router` returns an Axum router for applications that
manage TLS, shutdown, or routing themselves. One worker owns the backend; all
HTTP handlers communicate through bounded channels. True tensor batching depends
on the backend's `logits_batch`, rather than automatically becoming GPU batching.

<a id="host-validation"></a>

### Validation

```bash
cargo build --locked --no-default-features --features server --bin lighter-serve
cargo test --locked --features server
# Candle-free host checks (legacy Candle integration tests need default features):
cargo test --locked --no-default-features --features server --lib
cargo test --locked --no-default-features --features server --test test_model_host
```

Tests cover overlapping decode batches, buffered/SSE responses, stop-prefix and
Unicode handling, model IDs, explicit chat formatting, authentication, invalid
fields, input/output limits, timeout/admission, client cancellation, and batch
failure recovery. Real TCP tests construct a tiny local safetensors decoder,
load its actual tokenizer/weights, and verify completions/chat/streaming. Int8 and
Int4 model loaders are also tested through the HTTP router. The CLI was also exercised through the Python OpenAI SDK for model discovery,
completions, chat, incremental streaming and final usage. A tokenizer regression
check verifies that an explicit template BOS is not duplicated. Synthetic fixture
weights validate integration, not language quality or production throughput.
Large downloaded model and GPU-serving benchmarks were not run.

<a id="host-graph-to-text-extension"></a>

### Graph-to-text extension

`POST /v1/graph/completions` generates from a validated property graph using the same runtime and admission controls. See [graph inputs, examples and response semantics](#graph-hosted-graph-generation).

<a id="graph"></a>

## Graph-to-text

Lighter implements property-graph ingestion, validation, reproducible content planning,
exact realization and optional native language-model realization. The core runs without
Candle, a tokenizer, network access or model weights. It supports directed relations,
parallel edges, cycles, self-loops, isolated entities and arbitrary JSON-valued properties.
The deterministic graph module does not include a pretrained graph neural encoder.
For learned conditioning and training, see [G-Retriever](#retriever) and the
[native instruction fine-tuning example](#graph-finetuning).

<a id="graph-run-a-complete-example"></a>

### Run a complete example

```bash
cargo run --no-default-features --example graph2text
cargo run --no-default-features --example graph2text -- examples/data/graph2text.json /tmp/description.json
```

The second command writes `text` plus a `plan` containing every selected fact and its
provenance ID. Output includes literal values, entity IDs, edge IDs and direction:

```text
Entity "ada" has label "Ada Lovelace".
"Ada Lovelace" ("ada") has "birth_year" = 1815.
Entity "engine" has label "Analytical Engine".
"Analytical Engine" ("engine") has "type" = "mechanical computer".
Entity "babbage" has label "Charles Babbage".
"Charles Babbage" ("babbage") — "designed" → "Analytical Engine" ("engine") [edge "design"].
"Ada Lovelace" ("ada") — "wrote notes about" → "Analytical Engine" ("engine") [edge "notes"; attributes: "year" = 1843].
```

<a id="graph-input-contract"></a>

### Input contract

[Complete input fixture](../examples/data/graph2text.json). Each node requires a unique,
nonblank `id` and nonblank `label`. Each edge requires its own unique nonblank `id`,
existing `source` and `target` node IDs, and a nonblank `relation`. `properties` defaults
to an empty object; its keys must be nonblank. JSON scalars, arrays, nested objects and
null values remain literal data. Labels need not be unique. Node and edge IDs occupy
separate namespaces. Unknown fields fail JSON decoding; empty graphs fail validation.

Graph arrays are public for ergonomic construction; every processing entry point validates
them. `Graph::from_json` and `to_json` validate too. `GraphError` implements `Error`.
Property and label strings are JSON-escaped in output so newlines or quotes cannot
break fact boundaries.

<a id="graph-rust-construction-selection-and-provenance"></a>

### Rust construction, selection and provenance

```rust
use candlelighter::graph2text::{Edge, Graph, Node, Options, plan, verbalize};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut ada = Node::new("ada", "Ada Lovelace");
    ada.properties.insert("birth_year".into(), json!(1815));
    let graph = Graph {
        nodes: vec![ada, Node::new("engine", "Analytical Engine")],
        edges: vec![Edge::new("notes", "ada", "wrote notes about", "engine")],
    };
    let options = Options {
        root: Some("ada".into()), radius: Some(1), max_facts: Some(100),
    };
    let content = plan(&graph, &options)?;
    assert!(content.omitted_fact_ids.is_empty());
    println!("{}", verbalize(&graph, &options)?.text);
    let json = graph.to_json()?;
    assert_eq!(Graph::from_json(&json)?, graph);
    Ok(())
}
```

Planning is breadth-first, traversing both incoming and outgoing edges while preserving
original direction in every relation fact. Neighbors and component seeds use ID order;
edges and properties have stable ordering. Reordering input arrays leaves output unchanged.
With no radius, all disconnected components are included, even with a root. A radius
requires a root and includes only its undirected neighborhood; radius zero retains that
node, its properties and self-loops. An edge is selected only if both endpoints are selected.

Each node contributes one identity fact plus one fact per property. Each edge contributes
one relation fact with all edge properties. `max_facts` is a positive fact budget,
not a token budget. Every excluded fact is listed in `omitted_fact_ids`, whether excluded
by neighborhood selection or budget. Budgeting can omit node identity facts while retaining
others; callers needing full coverage should leave it unset and check omissions.

<a id="graph-knowledge-triples"></a>

### Knowledge triples

```rust
use candlelighter::graph2text::{Graph, Options, Triple, verbalize};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let graph = Graph::from_triples(&[
        Triple { subject: "Paris".into(), predicate: "is capital of".into(), object: "France".into() },
        Triple { subject: "France".into(), predicate: "is in".into(), object: "Europe".into() },
    ])?;
    println!("{}", verbalize(&graph, &Options::default())?.text);
    Ok(())
}
```

Triple ingestion merges exact entity labels, allocates deterministic `n0`, `n1`, … IDs
in lexical label order, and assigns `e0`, `e1`, … IDs in triple order. Duplicate triples
retain separate edges. Use explicit property graphs to distinguish equal-label entities.

<a id="graph-native-language-model-realization"></a>

### Native language-model realization

```bash
cargo run --no-default-features --features native --example graph2text_native -- /path/to/hf-model examples/data/graph2text.json
```

[Complete native example](../examples/graph2text_native.rs) loads local Hugging Face
artifacts once, constructs a native engine and calls `graph2text::generate`. It returns
both the content plan and the runtime's full `GenerateResponse` (text, tokens, usage and
finish reason). Reuse the idle engine for subsequent graphs. The helper rejects busy
engines; runtime failures abort its requests and clear their caches. The sample uses
256 output tokens and greedy decoding. Choose output limits to cover the graph and a
model whose context can accommodate the serialized facts. The native example uses a
plain completion prompt; use the HTTP endpoint with a chat template for instruction models.

`graph2text::prompt` serializes selected facts as JSON and instructs the model to preserve
identity, direction and literal values. Graph strings are untrusted data. Prompt instructions
reduce, but cannot guarantee protection from prompt injection or factual errors. Generated
prose may omit, distort or invent facts. The returned plan records supplied facts, not proof
that generated prose expresses them. Use `verbalize` when exact coverage is required.
No model download is necessary for exact realization; language-model realization needs
compatible weights and tokenizers as described in the [model host guide](#host).

<a id="graph-hosted-graph-generation"></a>

### Hosted graph generation

```bash
cargo run --no-default-features --features server --bin lighter-serve -- --model /path/to/hf-model --chat-template llama3
python3 - <<'PY'
import json, urllib.request
with open("examples/data/graph2text.json") as source:
    graph = json.load(source)
payload = {"model": "lighter", "graph": graph,
           "options": {"root": "ada", "radius": 1},
           "max_tokens": 256, "temperature": 0}
request = urllib.request.Request(
    "http://127.0.0.1:8000/v1/graph/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"})
with urllib.request.urlopen(request) as response:
    result = json.load(response)
print(result["choices"][0].get("message", {}).get("content",
      result["choices"][0].get("text", "")))
PY
```

`POST /v1/graph/completions` accepts `graph`, optional `options` and the
[host's completion generation fields](#host). `prompt` and `messages` are
rejected. Invalid graphs/options return 400 before admission. A configured chat template
wraps the graph prompt as a user message and returns chat completion fields; otherwise
it returns text completion fields. `stream: true` uses the same SSE format, queue,
authentication, token/context limits, cancellation and timeout handling as other endpoints.
This is a Lighter extension; OpenAI clients can call it through raw HTTP. Hosted replies
contain standard completion usage, not the plan; compute `plan` locally for provenance.

<a id="graph-keras-comparison"></a>

### Keras comparison

Keras has no built-in property-graph-to-text layer. A learned encoder/decoder requires a
chosen graph architecture, vocabulary, paired training data and decoding policy. Calling
a pretrained language model with serialized graph facts implements the same realization
approach as Lighter's native/hosted path. Exact realization corresponds to a deterministic
Python data transformation, independent of Keras:

```python
import json

def exact_graph_text(graph):
    nodes = {n["id"]: n for n in graph["nodes"]}
    if not nodes or len(nodes) != len(graph["nodes"]):
        raise ValueError("empty graph or duplicate node IDs")
    q = lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True,
                                separators=(",", ":"))
    lines = []
    for node in sorted(nodes.values(), key=lambda n: n["id"]):
        lines.append(f'Entity {q(node["id"])} has label {q(node["label"])}.')
        for key, value in sorted(node.get("properties", {}).items()):
            lines.append(f'{q(node["label"])} ({q(node["id"])}) has {q(key)} = {q(value)}.')
    for edge in sorted(graph["edges"], key=lambda e: e["id"]):
        src, dst = nodes[edge["source"]], nodes[edge["target"]]
        attributes = ", ".join(f"{q(k)} = {q(v)}" for k, v in sorted(edge.get("properties", {}).items()))
        suffix = f"; attributes: {attributes}" if attributes else ""
        lines.append(f'{q(src["label"])} ({q(src["id"])}) — {q(edge["relation"])} → {q(dst["label"])} ({q(dst["id"])}) [edge {q(edge["id"])}{suffix}].')
    return "\n".join(lines)

with open("examples/data/graph2text.json") as source:
    print(exact_graph_text(json.load(source)))
```

This Python comparison uses ID ordering rather than rooted traversal and illustrates
realization only; Lighter additionally validates the full schema, plans neighborhoods and
reports omissions. See the [general Keras comparison](#keras) for trainable
layers and inference differences.

<a id="graph-verification"></a>

### Verification

```bash
cargo test --no-default-features --test test_graph2text
cargo test --no-default-features --features server --test test_graph2text
cargo test --no-default-features --features server --lib graph_api_tests
```

Tests cover graph serialization, missing endpoints, duplicate identities, bad options,
cycles, self-loops, disconnected components, incoming edges, budgets, parallel edges,
Unicode/escaping, arbitrary properties, native decoding/cache cleanup and HTTP validation.

<a id="graph-g-retriever-retrieval-and-learned-graph-conditioning"></a>

### G-Retriever retrieval and learned graph conditioning

For question-guided PCST subgraph retrieval, upstream GNN encoders and trainable graph
soft prompts, see the [G-Retriever integration](#retriever). Its retrieval example
exports the same property-graph schema used by this pipeline.

<a id="retriever"></a>

## G-Retriever integration

Lighter includes the implementation from [XiaoxinHe/G-Retriever](https://github.com/XiaoxinHe/G-Retriever),
pinned to revision `315b0ff8a206536067602fb97e77c10f4d646d5d`, plus CPU-compatible
examples integrating its retrieval and graph soft-prompt architecture. The upstream
Python sources are preserved unchanged in [third_party/g_retriever](../third_party/g_retriever/README.md),
with the [MIT license](../third_party/g_retriever/LICENSE) and [provenance](../third_party/g_retriever/PROVENANCE.md).

The integration implements actual PCST retrieval and learned graph conditioning:

1. Embed the question, node labels/properties and edge relations/properties.
2. Convert cosine rankings into node and edge prizes.
3. Convert edge prizes into discounted costs or virtual prize-bearing nodes.
4. Run the unrooted prize-collecting Steiner-tree solver with GW pruning.
5. Restore original graph IDs, relation directions and selected textual facts.
6. Encode the retrieved graph with the upstream GCN, GAT or graph Transformer.
7. Mean-pool graph node representations and project them into one language-model
   embedding, inserted as a learned soft prompt.
8. Train the graph encoder/projector using answer-only causal language-model loss,
   or generate answers with a frozen causal language model.

This implementation uses PyTorch, PyG, Transformers and `pcst-fast`. Lighter's native
Rust decoder currently accepts token inputs, not arbitrary input embeddings; the learned
soft-prompt path runs through the Python integration. Exported retrieved property graphs
work with Rust exact realization, native generation and the HTTP graph endpoint.

<a id="retriever-install"></a>

### Install

Use Python 3.10–3.12 and a separate environment from the Keras examples:

```bash
python3 -m venv .venv-g-retriever
. .venv-g-retriever/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r examples/python/g_retriever/requirements.txt
```

All commands assume the repository root. For CUDA, install an appropriate PyTorch
build from [the official selector](https://pytorch.org/get-started/locally/) instead.
The provided CLI examples use CPU; the core also accepts a language model already
placed on a single CUDA device. Use a model that fits that device's memory.

NumPy is constrained to `>=1.26,<2`: the tested `pcst-fast==1.0.10` wheel returned
corrupted vertex/edge arrays under NumPy 2. The integration rejects that combination.
No custom `torch_scatter` extension is needed by the adapted pooling code. The unchanged
upstream research scripts have additional dependencies described in their README.

<a id="retriever-run-offline-examples"></a>

### Run offline examples

The examples need no downloaded models or datasets when run with these defaults:

```bash
# Cosine rank prizes and actual PCST retrieval.
python examples/python/g_retriever_retrieve.py

# Frozen tiny Llama + trainable upstream GNN/projector; save the learned adapter.
python examples/python/g_retriever_train.py --epochs 3 --output /tmp/g-retriever-demo.pt

# Reload the adapter and generate through the graph soft-prompt path.
python examples/python/g_retriever_generate.py --adapter /tmp/g-retriever-demo.pt
```

Offline embeddings are deterministic hashed word counts. The offline causal LM is a
small **randomly initialized** Llama with a byte tokenizer. These defaults demonstrate
retrieval, gradients, serialization and decoding; they do not supply trained QA answers.
Generated byte sequences may contain unreadable text. Use semantic embeddings, a
pretrained LM and a trained compatible adapter for meaningful QA. Retrieval and
training can be inspected independently of generation quality.

| Example | Purpose | Source |
| --- | --- | --- |
| Retrieval | Graph validation, ranking, virtual nodes, PCST, text and provenance | [g_retriever_retrieve.py](../examples/python/g_retriever_retrieve.py) |
| Training | JSONL batches, frozen LM, GNN/projector optimization and checkpoint output | [g_retriever_train.py](../examples/python/g_retriever_train.py) |
| Generation | Graph retrieval, soft-prompt injection and checkpoint inference | [g_retriever_generate.py](../examples/python/g_retriever_generate.py) |
| Integration core | Retrieval, sentence encoder and graph-conditioned LM | [core.py](../examples/python/g_retriever/core.py) |
| Offline model | Tiny Llama and byte tokenizer | [demo.py](../examples/python/g_retriever/demo.py) |

<a id="retriever-retrieve-and-export-to-rust"></a>

### Retrieve and export to Rust

```bash
python examples/python/g_retriever_retrieve.py \
  --graph examples/data/graph2text.json \
  --question "Who wrote notes about the Analytical Engine?" \
  --topk 3 --topk-edges 3 --edge-cost 0.5 \
  --output /tmp/retrieved-graph.json

cargo run --no-default-features --example graph2text -- /tmp/retrieved-graph.json
```

`--output` writes a selected property graph and replaces an existing output file.
The program also prints a CSV description and retrieval metadata: selected original
node/edge IDs, per-input node/edge prizes, and omitted original IDs. Output graph
nodes retain their properties and directed relations. PCST solves connectivity as
undirected; direction is restored for the GNN and text description.

Node IDs and edge IDs must be unique within their respective namespaces, endpoints
must exist, labels/relations must be nonblank, and properties must be finite JSON data.
The schema matches [Lighter graph-to-text](#graph-input-contract). Edgeless graphs
retain all nodes, matching upstream behavior. With edges, unrooted PCST selects one
connected component; it may drop unrelated entities or relations. Edge-prize ranking
can be disabled with `--topk-edges 0`; node ranking with `--topk 0`, provided another
prize source is enabled. Ties among nodes use input index order; tied edge scores share
their prize. Retrieval is approximate, not a guarantee that the answer is present.

<a id="retriever-use-semantic-embeddings-and-pretrained-models"></a>

### Use semantic embeddings and pretrained models

The upstream sentence encoder is `sentence-transformers/all-roberta-large-v1`.
The integration uses its Hugging Face encoder with masked mean pooling and normalization:

```bash
python examples/python/g_retriever_retrieve.py \
  --encoder sentence-transformers/all-roberta-large-v1 \
  --question "Who designed the Analytical Engine?"
```

Encoder IDs and `--model` accept Hugging Face IDs or local directories. Downloading
requires network access; gated models require publisher access and Hub credentials
such as `HF_TOKEN`. Models with `inputs_embeds` support and compatible causal-LM
forward/generation behavior are required. The tested architecture is Llama.

For example, train using a local pretrained model and semantic encoder:

```bash
python examples/python/g_retriever_train.py \
  --model /path/to/causal-lm \
  --encoder sentence-transformers/all-roberta-large-v1 \
  --data examples/data/g_retriever_train.jsonl \
  --epochs 3 --batch-size 2 --learning-rate 0.001 \
  --hidden-dim 64 --output /tmp/g-retriever-adapter.pt

python examples/python/g_retriever_generate.py \
  --model /path/to/causal-lm \
  --encoder sentence-transformers/all-roberta-large-v1 \
  --adapter /tmp/g-retriever-adapter.pt --hidden-dim 64 \
  --graph examples/data/graph2text.json \
  --question "Who wrote notes about the Analytical Engine?"
```

The supplied two-record fixture is a smoke-training dataset, not sufficient for a
useful general QA system. Train on representative labeled graphs. Without `--adapter`,
the GNN/projector are randomly initialized even when the language model is pretrained.

<a id="retriever-training-data-and-checkpoints"></a>

#### Training data and checkpoints

[Complete JSONL fixture](../examples/data/g_retriever_train.jsonl). Each line contains:

```json
{"graph":{"nodes":[{"id":"a","label":"Ada"},{"id":"b","label":"Engine"}],"edges":[{"id":"e","source":"a","relation":"wrote notes about","target":"b"}]},"question":"Who wrote notes?","answer":"Ada."}
```

Retrieval is precomputed before epochs. Each batch pools its graphs separately,
left-pads model input embeddings, masks padding and prompt positions with `-100`,
and trains only on answer tokens plus EOS. AdamW optimizes the GNN/projector with
gradient clipping. The frozen LM stays in evaluation mode while retaining the backward
path to its input soft prompt. Because upstream GNNs use BatchNorm, a training batch
needs at least two total retrieved nodes; inference supports a single-node graph.

`--resume PATH` reloads an adapter before optimization; it does not restore optimizer
state. Checkpoints contain GNN and projector weights, not the frozen LM or sentence
encoder. Reuse the same encoder, GNN architecture/dimensions and base language model
for inference. `save_adapter` replaces its target file. Core `load_adapter` uses
`torch.load(weights_only=True)`. Default graph Transformer uses two layers, four heads
and hidden size 64. The core also supports the unchanged upstream `gcn` and `gat`.

Description and training-answer token lengths are truncated to configured limits
(default 512 and 64 in the core). Questions remain intact. Context overflow raises an
error. The generation CLI defaults to 32 new tokens. Truncation can remove evidence,
and generated answers remain probabilistic; the selected graph is not proof of fidelity.

<a id="retriever-api-example"></a>

### API example

```python
# From the repository root; install dependencies above first.
import json, sys
sys.path.insert(0, "examples/python")
from g_retriever import HashEncoder, GraphRetriever, retrieve
from g_retriever.demo import tiny_model

with open("examples/data/graph2text.json") as source:
    graph = json.load(source)
encoder = HashEncoder(64)
retrieval = retrieve(graph, "Who wrote notes?", encoder)
model, tokenizer = tiny_model()
retriever = GraphRetriever(model, tokenizer, input_dim=64)
loss = retriever([retrieval], ["Who wrote notes?"], ["Ada Lovelace."])
loss.backward()
print("loss:", loss.item())
print(retriever.generate([retrieval], ["Who wrote notes?"]))
```

<a id="retriever-differences-from-upstream-and-reproduction"></a>

### Differences from upstream and reproduction

The upstream research source is preserved verbatim. The adapted integration changes:

- GPU-specific two-device memory limits to an injectable, CPU-compatible causal LM.
- The fixed 4096-dimensional soft prompt to the model's actual embedding width.
- `torch_scatter` pooling to native PyTorch `index_add_` and per-graph counts.
- Llama-2-specific instruction markers to a generic `Question:` / `Answer:` prompt.
- Retrieval input preparation to Lighter property graphs, preserving IDs and properties.
- Top-ranked edge tie handling to use immutable cosine scores; empty edge shapes and
  invalid data are handled explicitly.

The adapted training path freezes the LM and trains the GNN/projector. The vendored
research implementation additionally supports LM LoRA and prompt-only baselines.
Its original defaults, pretrained model choices and benchmark scripts remain in
[upstream README](../third_party/g_retriever/README.md). Datasets and figures are not
vendored; obtain and preprocess ExplaGraphs, SceneGraphs or WebQSP following upstream
instructions before invoking those scripts from `third_party/g_retriever/`. Their CUDA,
`torch_scatter`, PEFT, datasets and tracking dependencies are separate from the CPU
integration requirements. The offline tests do not reproduce published benchmark scores
or validate a downloaded pretrained checkpoint.

<a id="retriever-verify"></a>

### Verify

```bash
python -m unittest discover -s examples/python/g_retriever -v
```

Tests exercise actual PCST connectivity, direction/ID restoration, prize ties,
parallel edges/self-loops, disconnected and edgeless graphs, invalid inputs, deterministic
embeddings, all three upstream GNNs, gradient propagation/frozen LM behavior,
checkpoint round trips, batch padding, generation and context limits.

<a id="retriever-citation"></a>

### Citation

```bibtex
@inproceedings{he2024gretriever,
  title={G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering},
  author={Xiaoxin He and Yijun Tian and Yifei Sun and Nitesh V Chawla and Thomas Laurent and Yann LeCun and Xavier Bresson and Bryan Hooi},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://openreview.net/forum?id=MPJ3oXtTZl}
}
```

<a id="clm"></a>

## Contrastive-LM runner

Lighter provides a native, Candle-free runner for
[`Contrastive-LM/CLM-v0.1-8B`](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B).
It supports Hugging Face downloads, offline snapshots, F16/BF16 weights,
optional Int8 or packed Int4 quantization, and repeated generation without
reloading the model.

<a id="clm-requirements"></a>

### Requirements

- A stable Rust toolchain.
- Enough disk space for the complete Hugging Face snapshot.
- Approximately 16 GB plus working memory for F16/BF16 weights. Int8 and Int4
  reduce resident matrix storage, but quantization still needs temporary memory
  while each checkpoint shard is converted.
- Network access for the first Hub-backed run, unless `CLM_MODEL_DIR` points to
  an existing snapshot.

The built-in implementation is a portable CPU reference runtime. It uses all
available CPU cores by default, but an 8B model remains computationally
expensive. Use `--release` for generation.

<a id="clm-quick-start"></a>

### Quick start

Run the model with a prompt:

```bash
cargo run --release --example clm_generate \
  --no-default-features --features native -- \
  "Explain contrastive language modeling."
```

The first invocation downloads `config.json`, `tokenizer.json`, and every
safetensors shard to the Hugging Face cache. Set `HF_TOKEN` when the repository
requires authentication:

```bash
HF_TOKEN=hf_example cargo run --release --example clm_generate \
  --no-default-features --features native -- "Hello"
```

Arguments after `--` are joined with spaces, so quoted and unquoted multiword
prompts are both accepted. With no prompt, the example uses a built-in default.

<a id="clm-offline-snapshots"></a>

### Offline snapshots

Set `CLM_MODEL_DIR` to a Hugging Face snapshot containing:

- `config.json`
- `tokenizer.json`
- either `model.safetensors`, or `model.safetensors.index.json` and every shard
  referenced by its `weight_map`

```bash
CLM_MODEL_DIR=/models/CLM-v0.1-8B \
cargo run --release --example clm_generate \
  --no-default-features --features native -- "Write a short Rust example."
```

When `CLM_MODEL_DIR` is set, the runner does not contact Hugging Face.

<a id="clm-quantization-and-cpu-threads"></a>

### Quantization and CPU threads

Set `CLM_QUANTIZATION` to `int8` or `int4`. Matrices are quantized as each
checkpoint shard is decoded, avoiding a second full half-precision copy:

```bash
CLM_QUANTIZATION=int4 RAYON_NUM_THREADS=8 \
cargo run --release --example clm_generate \
  --no-default-features --features native -- "Summarize ownership in Rust."
```

`RAYON_NUM_THREADS` limits the CPU worker pool. If omitted, Rayon uses the
machine's available parallelism. Quantization is lossy; compare output quality
before choosing it for an application.

| Variable | Values | Purpose |
| --- | --- | --- |
| `HF_TOKEN` | Hugging Face token | Authenticates Hub downloads. |
| `CLM_MODEL_DIR` | Snapshot directory | Enables fully offline loading. |
| `CLM_QUANTIZATION` | `int8` or `int4` | Reduces resident matrix storage. |
| `RAYON_NUM_THREADS` | Positive integer | Controls CPU inference parallelism. |

<a id="clm-library-api"></a>

### Library API

Use `ContrastiveLm` to retain loaded weights across requests:

```rust,no_run
use candlelighter::native::SamplingParams;
use candlelighter::ContrastiveLm;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let token = std::env::var("HF_TOKEN").ok();
    let mut model = ContrastiveLm::from_hub(token)?;
    let response = model.generate(
        "Explain why Rust prevents data races.",
        SamplingParams {
            max_tokens: 128,
            temperature: 0.7,
            top_p: 0.9,
            seed: 42,
            ..Default::default()
        },
    )?;
    println!("{}", response.text);
    Ok(())
}
```

For a quantized local snapshot:

```rust,no_run
use candlelighter::native::SamplingParams;
use candlelighter::native_advanced::{QuantizationConfig, QuantizationType};
use candlelighter::ContrastiveLm;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let mut model = ContrastiveLm::from_dir_quantized(
    "/models/CLM-v0.1-8B",
    QuantizationConfig {
        dtype: QuantizationType::Int4,
        group_size: 64,
    },
)?;
let response = model.generate("Hello", SamplingParams::default())?;
println!("{}", response.text);
# Ok(())
# }
```

Available constructors are `from_hub`, `from_dir`, `from_hub_quantized`,
`from_dir_quantized`, and `from_artifacts_with_quantization`.

<a id="clm-troubleshooting"></a>

### Troubleshooting

- **401/403 from Hugging Face:** accept any model license requirements and set
  `HF_TOKEN` to a token with read access.
- **Out of memory:** select Int8 or Int4, close other memory-heavy processes,
  and ensure the largest individual shard plus the quantized model fits RAM.
- **Missing tensor or shard:** verify the snapshot is complete and that all
  files named by `model.safetensors.index.json` are present.
- **Unsupported dtype:** the native loader accepts F32, F16, and BF16 source
  tensors. Convert other checkpoint formats to safetensors in one of these
  dtypes.
- **Generation is slow:** use a release build, enable quantization, and tune
  `RAYON_NUM_THREADS`. This runtime is CPU-oriented and prioritizes portability
  over accelerator throughput.

<a id="keras"></a>

## Python Keras comparison

This chapter translates Lighter concepts to **standalone Keras 3** with a JAX CPU
backend. The examples use `import keras`, not `tensorflow.keras`. A familiar name
does not imply matching numerical behavior: shape conventions, initialization,
optimizer state, recurrence, and serialization differ.

<a id="keras-candle-arm-and-npu-comparison"></a>

### Candle, ARM acceleration and NPU deployment compared with Keras

Keras is a Python model API whose selected backend executes tensor operations.
Lighter's Candle layers serve a similar role for Rust, while its native decoder
and TensorFlow Lite delegate are separate execution paths. Choose the path first:
a training model, an inference tensor operation and a compiled NPU model have
different interfaces and gradient behavior.

| Concern | Lighter / Candle | Python Keras 3 |
| --- | --- | --- |
| Choose tensor runtime | Enable `candle`; select a Candle `Device` | Set `KERAS_BACKEND` before importing Keras: TensorFlow, JAX or PyTorch |
| Ordinary matrix multiplication | `Tensor::matmul` uses Candle's implementation | `keras.ops.matmul` uses the selected backend's implementation |
| Explicit ARM vector/matrix kernels | `arm::dot` and `arm::matmul` select NEON/SVE/SME at runtime | Core Keras has no matching NEON/SVE/SME selector; backend libraries choose their CPU kernels |
| Connect tensors to Lighter ARM kernels | Copy CPU F32 tensor values to row-major vectors, invoke kernel, reconstruct tensor | A Python native extension would need an equivalent explicit bridge |
| Gradients through copied buffers | Candle autograd connection is lost | Converting tensors to NumPy/native buffers and rebuilding tensors also generally loses the differentiable connection |
| Preserve gradients | Keep supported Candle tensor operations or implement a custom operation with backward | Keep supported Keras/backend operations or implement the backend's custom-gradient mechanism |
| Native token generation | `NativeEngine` adds scheduling, sampling and constraints | A Keras LM may implement generation itself or use another model/serving library |
| Arm NN / TFLite inference | `TfliteNpuBackend` dynamically loads the C runtime and external delegate | A Python TFLite interpreter can load an external delegate; this is a separate inference runtime, not `model.fit` |
| Export to NPU | Supply an already compiled model meeting Lighter's fixed-prefix tensor contract | Export/convert a supported model using TensorFlow Lite and vendor tooling; conversion is model/backend dependent |
| NPU training | Delegate path is inference-only | TFLite delegates do not add Keras training or backpropagation |
| HTTP serving | Shared Rust host, including a factory for thread-affine delegates | Serve the model/interpreter through an application or dedicated serving runtime |
| Check actual NPU use | Delegate status plus vendor partition/profile evidence | Loading a delegate likewise does not prove every operation ran on the NPU |

Enabling `arm-npu`, `arm-sve` or `arm-sme` does not change Candle's device dispatch.
Likewise, selecting a Keras backend does not automatically select an Arm NN external
delegate. CPU acceleration, GPU execution and vendor NPU execution need their own
runtime configuration. Refer to [Candle integration](#candle-arm-integration) and
the [NPU deployment contract](#arm-backends) for the Rust boundaries.

#### Compare the same matrix operation

Use fixed F32 inputs to compare numerical results independently of random model
initialization. This Keras example produces the same matrix as the complete
[Rust Candle/ARM bridge](../examples/candle_arm_bridge.rs):

```python
import os
os.environ.setdefault("KERAS_BACKEND", "jax")  # Set before importing Keras.
import keras
import numpy as np

left = keras.ops.convert_to_tensor([[1., 2., 3.], [4., 5., 6.]], dtype="float32")
right = keras.ops.convert_to_tensor([[7., 8.], [9., 10.], [11., 12.]], dtype="float32")
product = keras.ops.matmul(left, right)
actual = keras.ops.convert_to_numpy(product)
np.testing.assert_allclose(actual, [[58., 64.], [139., 154.]], rtol=1e-5, atol=1e-5)
print(actual)
```

```bash
cargo run --no-default-features --features candle,native --example candle_arm_bridge
```

Both examples validate the arithmetic. The Python operation remains a backend tensor
operation until its result is converted for inspection; the Rust bridge deliberately
crosses an owned-buffer boundary before multiplication. It is an inference recipe,
not a replacement for a differentiable Candle layer. Benchmark complete calls,
including conversion and allocation, before choosing either execution path.

#### Move a Keras model toward the delegate contract

A Keras training checkpoint is not a `.tflite` deployment artifact. For a compatible
TensorFlow-backed model, TensorFlow Lite tooling can convert a supported exported
model; vendor NPU tooling may require additional quantization or compilation. JAX
and PyTorch Keras backends require an appropriate supported export route rather than
passing their backend tensors directly to the TensorFlow Lite converter.

For Lighter's current token-generation backend, the exported graph must accept
fixed `[1, context]` token IDs and any configured mask/position inputs, and return
causal F32 logits in a supported layout. A usual Keras regression or image classifier
does not satisfy that contract. The same trained architecture, tokenizer, token IDs,
masking and position semantics must survive export. Some causal architectures or
operators may be unsupported by the selected NPU, even if they train successfully
in Keras.

Compare the original Keras/model-runtime outputs with compiled-model outputs before
serving them. Include padded prefixes, multiple prefix lengths and context boundaries;
check logits with a dtype/quantization-appropriate tolerance, then check generated
text. Attach the delegate and inspect vendor diagnostics to establish NPU coverage.
Model conversion, NPU execution and gradient parity are not established by the small
matrix comparison above.

'''
<a id="keras-run-the-complete-comparison"></a>

### Run the complete comparison

From the repository root, create an isolated Python environment:

```bash
python3 -m venv /tmp/lighter-keras
/tmp/lighter-keras/bin/python -m pip install keras==3.15.1 jax==0.11.2 jaxlib==0.11.2 numpy==2.5.3
KERAS_BACKEND=jax /tmp/lighter-keras/bin/python examples/python/keras_comparison.py
```

These are the versions used to validate this chapter, rather than dependencies
of the Rust library. JAX/JAXlib need a supported Python version/platform; the
validated interpreter is Python 3.12.14. Set `KERAS_BACKEND` before
importing Keras. Backend-specific accelerator setup is outside these CPU examples.
No model download or token is needed; temporary persistence files are cleaned up.

```bash
cargo run --locked --example handbook_keras
cargo run --locked --example handbook_layers
cargo run --locked --example handbook_experimental
cargo run --locked --example handbook_training --no-default-features --features native
```

The Rust programs are complete listings in the [code cookbook](#examples).
The full Python program appears at the end of this chapter and is also available
as [keras_comparison.py](../examples/python/keras_comparison.py).

<a id="keras-api-and-behavior-mapping"></a>

### API and behavior mapping

| Task | Lighter / Candle Rust | Python Keras 3 | Important difference |
| --- | --- | --- | --- |
| Assemble a sequential network | `SequentialModel::new(vars, layers)` | `keras.Sequential([...])` | Rust shares an explicit `VarMap`; Keras layers own tracked weights. |
| Dense projection | `Dense::new(out, in, activation, ...)` | `layers.Dense(out, activation=...)` | Keras infers input width after build. Rust orders output width before input width. |
| Select optimizer/loss | `model.compile(Optimizers::SGD(rate), Loss::MSE)` | `model.compile(optimizer=SGD(rate), loss="mse")` | Compile selects configuration; it does not train. |
| Train regression | `model.fit(x, y, epochs, verbose)` | `model.fit(x, y, epochs=..., batch_size=...)` | Rust sample loop lacks Keras batching, shuffling, callbacks, and History. |
| Infer | `model.predict(&x)` / `forward(x)` | `model.predict(x)` / `model(x, training=False)` | Rust predict returns a vector of per-sample tensors. |
| Summarize | `model.summary()` | `model.summary()` | Rust's width-based count is not a precise parameter count. |
| Adam optimizer | `Optimizers::Adam(rate)` | `keras.optimizers.Adam` / `AdamW` | Rust wrapper uses AdamW and recreates optimizer state per sample. Use a persistent custom loop for fair comparison. |
| Classification | Custom Candle `cross_entropy` loop | Built-in categorical/sparse/binary losses | Rust wrapper's `fit` currently accepts only MSE. |
| Global min-max/z-score | `FeatureScaling` | NumPy/global `keras.ops` expressions | `layers.Normalization(axis=-1)` uses per-feature statistics; it is not a direct replacement. |
| Reuse scaling statistics | `_other` methods | Retain training minimum/maximum/mean/std | Compute statistics on training data only. |
| Inverse transforms | `_reverse` methods | `scaled * range + minimum` | Rust inverse methods return flattened tensors. |
| Linear/ReLU/SiLU/sigmoid | `Activations` | Dense `activation=` or `layers.Activation` | Comparable functions; parameters/initialization still differ. |
| Softmax | `Activations::Softmax` | `ops.log_softmax(logits)` | Rust's named Softmax currently produces log-softmax. Keras `"softmax"` produces probabilities. |
| 1D/2D convolution | `Conv::new(explicit_kernel, ...)` | `layers.Conv1D` / `Conv2D` | Rust is channels-first; Keras defaults to channels-last. Keras kernels train normally. |
| Convolution initialization | Experimental `Conv::new2` | `kernel_initializer=` | Rust `new2` currently returns initialized tensors instead of convolving input. |
| Average/max pooling | `Pooling` | `AveragePooling2D` / `MaxPooling2D` | Rust MAX and AVERAGE names are reversed. Python example uses their conventional meaning. |
| Unit/L2 normalization | Explicit Candle square/sum/divide | `layers.UnitNormalization` | Rust Normalization wrapper returns unchanged input. |
| Layer/batch normalization | No equivalent functioning Lighter wrapper | `LayerNormalization` / `BatchNormalization` | Learned scale/offset and running statistics are additional semantics. |
| Flatten | `Flatten` | `layers.Flatten` | Rust collapses every dimension to `[1, total]`; Keras preserves batch. |
| LSTM/GRU | `Recurrent` | `layers.LSTM` / `GRU` | Rust resets state each call, batch=1, one step; LSTM returns cell `c`. Keras normally returns final hidden output after a sequence. |
| Explicit recurrent state | Upstream Candle RNN methods / liquid `step` | `initial_state`, `return_state`, RNN cell | Match state/output choices before porting. |
| Embedding | `Embed(Standard, vocabulary, width, ...)` | `layers.Embedding(vocabulary, width)` | Use integer IDs and consistent vocabulary. |
| Position embeddings | Compose embedding layers; native decoder RoPE | Compose second `Embedding` lookup | Learned absolute positions differ from RoPE/sinusoidal encodings. |
| Dropout | Candle `Dropout` custom pipeline | `layers.Dropout` | No Lighter `Trainable` dropout wrapper. Training flag matters. |
| Weight decay / L2 penalty | Candle/native AdamW | Keras AdamW / `regularizers.L2` | Decoupled weight decay and L2 loss penalties are distinct. |
| Autoencoder | Sequential encoder/decoder composition | Sequential or Functional model | Both can train reconstruction; neither example is a VAE. |
| Self/cross/causal attention | Restricted legacy SelfAttention; native decoder attention | `layers.MultiHeadAttention` | Keras attention example is general sequence attention; Rust legacy wrapper is image-oriented and restricted. |
| MoE | Experimental `SparseMoE` | Custom `TopTwoExperts` layer | Both tiny examples evaluate all experts; Python gate uses softmax top-2 weighting, not identical Rust routing. |
| Mask features | Tensor multiply; JEPA masks | Multiply / `Masking` / attention mask | Zeroing, propagated sequence masks, and attention exclusions serve different purposes. |
| KAN-Dense | No implemented Lighter layer | No built-in Keras KAN layer | Requires a custom spline/basis layer or separate library; no fictitious API example is provided. |
| LoRA | Native `LoraAdapter`; legacy Dense PEFT experimental | `layers.Dense(..., lora_rank=...)` | Keras Dense LoRA is not Lighter legacy `Dense::new2`; native adapter has explicit gradient APIs. |
| DoRA | Legacy experimental Dense DORA | No built-in Keras Dense DoRA option | A custom weight parameterization is needed. |
| Split/multiple heads | Restricted `ParallelModel::Split` | Functional multiple outputs | Keras outputs branch independently; current Rust Split predicts sequentially. |
| Merge/ensemble | Explicit average of independent Rust models | `layers.Average` | Rust legacy Merge is incomplete. |
| Model persistence | Architecture JSON and separate weights; Candle VarMap safetensors | `.keras` model / `.weights.h5` | Keras model save includes structure and optimizer state for supported models. Rust legacy JSON round trips are incomplete. Formats are not interchangeable. |
| Hugging Face tokenizer/config/shards | `HuggingFaceArtifacts`, `HuggingFaceBackend` | No automatic core-Keras equivalent | Needs compatible model/tokenizer integration, e.g. another model library. |
| BERT/Llama inference | Candle wrappers / built-in NativeLlama | No pretrained BERT/Llama from plain `Dense` | A pretrained architecture/checkpoint integration is needed; toy attention is not a pretrained model. |
| Model selection and hardware budget | `ModelSelector` | No core-Keras selector | Keras backend device detection does not implement Hub selection/memory budgeting. |
| CLM 8B/offline generation | `ContrastiveLm` | No core-Keras CLM loader | Use Rust examples or a compatible external Python model implementation. |
| HTTP model hosting | `lighter-serve`: OpenAI-compatible completion/chat/SSE API | No core-Keras model server equivalent | [Hosting guide](#host); CPU reference execution, not vLLM GPU throughput. |
| Continuous batching/cancel/EOS/logprobs | `NativeEngine` | No core-Keras serving scheduler | A forward pass/`predict` is not an autoregressive serving engine. |
| Sampling and penalties | `SamplingParams` | Custom decoding or model-library sampler | Keras classification softmax is not token sampling. |
| Int8/Int4 storage | `QuantizedMatrix` / NativeLlama quantized loader | Backend/layer-specific quantization APIs | Custom native grouped scales/packed storage are not equivalent to casting or generic Keras quantization. |
| Paged/shared-prefix KV cache | `PagedKvCache` | No core-Keras cache abstraction | Needs a custom serving backend. |
| PPO/DPO/GAE | Objective and explicit-gradient functions | Custom `keras.ops` objectives | Both need a real optimizer, policy/environment, and gradient propagation for full RL. |
| Distillation | `distillation_loss`/update hook | KL and supervised loss in a custom loop | Match temperature scaling, coefficients and reduction. |
| SFT and label smoothing | `causal_lm_loss`, shifted labels, ignore index | Categorical CE plus masks/sample weights | Integer ignore IDs need explicit masking in Keras. |
| Reward/value model | `RewardHead` | `Dense(1)` with shared pairwise head | Compare scalar scores, not token classifier outputs. |
| Prompt/template/grammar generation | Native prompt types and constraints | No core-Keras prompt/grammar engine | Python string templating or regex post-validation does not constrain decoding. |
| Workflow/signature/tool/MCP parsing | Native workflow/protocol structs | No core-Keras equivalents | Ordinary Python orchestration/protocol code is separate from the ML model. |
| I-JEPA/V-JEPA | Built-in mask/trainer modules | Custom online/predictor/target architecture | Python example is a compact JEPA-like image reference; not full Meta training or V-JEPA. |
| Neural ODE | Euler/Heun/RK4 `NeuralOde` | Custom solver or external library | Python RK4 example validates the equation, not differentiable Keras ODE training. |
| Liquid/CfC/irregular time | Built-in cells and LFM | Custom recurrent layer or external LTC library | Python elapsed-time gate is illustrative; it is not identical to Lighter CfC/LFM. |

<a id="keras-regression-translate-an-entire-workflow"></a>

### Regression: translate an entire workflow

Python uses `[batch, features]`; the Rust wrapper uses `[samples, time=1, features]`.
The dataset below expresses `target = first_feature + second_feature`. Both
programs train a 2→4→1 network using MSE and SGD, then check output shapes. They do
not promise identical weights or loss trajectories.

<a id="keras-rust"></a>

#### Rust

The full `main` is in [handbook_keras](#examples-handbook_keras). Its core:

```rust
use candlelighter::prelude::*;
let dev = Device::Cpu;
let vars = VarMap::new();
let mut model = SequentialModel::new(vars.clone(), vec![
    Box::new(Dense::new(4, 2, Activations::Relu, &dev, &vars, "hidden".into())),
    Box::new(Dense::new(1, 4, Activations::Linear, &dev, &vars, "output".into())),
]);
let x = Tensor::new(&[[[1f32, 2.]], [[2., 3.]], [[3., 4.]], [[4., 5.]]], &dev)?;
let y = Tensor::new(&[[[3f32]], [[5.]], [[7.]], [[9.]]], &dev)?;
model.compile(Optimizers::SGD(0.01), Loss::MSE);
model.fit(x.clone(), y, 3, false);
let predictions = model.predict(&x).unwrap();
assert_eq!(predictions.len(), 4);
```

<a id="keras-python-keras"></a>

#### Python Keras

```python
import keras
import numpy as np

x = np.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype="float32")
y = x.sum(axis=1, keepdims=True)
model = keras.Sequential([
    keras.Input(shape=(2,)),
    keras.layers.Dense(4, activation="relu"),
    keras.layers.Dense(1),
])
model.compile(optimizer=keras.optimizers.SGD(0.01), loss="mse")
history = model.fit(x, y, epochs=3, batch_size=1, shuffle=False, verbose=0)
predictions = model.predict(x, verbose=0)
assert predictions.shape == (4, 1)
```

Keras retains optimizer state and exposes loss history. Lighter's wrapper prints
best loss and returns per-sample tensors. Use persistent optimizer custom loops
when comparing training mechanics. Keras callbacks, metric tracking, validation
splits, and distributed training are not provided by Lighter's fitting wrapper.

<a id="keras-convolution-and-pooling-compare-actual-values"></a>

### Convolution and pooling: compare actual values

The complete layer programs verify an all-ones length-two convolution kernel on
`[1,2,3,4]` yields `[3,5,7]`. Rust input is `[1,1,4]`; Python is `[1,4,1]`.
For two-dimensional convolution, Rust kernel shape is
`[output_channels, input_channels, height, width]`; Keras kernel shape is
`[height, width, input_channels, output_channels]`. For a transfer, transpose the
kernel and input axes, then compare results explicitly.

A 2×2 patch `[1,2;3,4]` averages to `2.5` and max-pools to `4`. The Rust tour
asserts its wrapper's current reversed enum mapping; the Python tour asserts
Keras' conventional mapping. These assertions deliberately reveal the difference.

<a id="keras-normalization-activation-and-flatten-semantics"></a>

### Normalization, activation, and flatten semantics

Three different operations often get called normalization:

1. Dataset standardization uses training mean/std (Keras `Normalization.adapt`).
2. Unit normalization divides each vector by its norm (`UnitNormalization`).
3. Layer normalization uses per-example statistics and optional learned scale/offset.

Lighter `FeatureScaling` provides the first using global tensor statistics.
Its `Normalization` wrapper provides none of these correctly at present; use the
explicit L2 recipe for the second. The examples test `[3,4] → [0.6,0.8]`.
Keras batch normalization additionally has training/inference moving-statistic state.

Keras `Flatten` preserves batch; Lighter `Flatten` collapses all dimensions.
Keras `"softmax"` outputs probabilities; Lighter `Activations::Softmax` currently
outputs log probabilities. Compare Rust to Python `ops.log_softmax`, or exponentiate
Rust output when probability semantics are required.

<a id="keras-recurrence-embeddings-and-attention"></a>

### Recurrence, embeddings, and attention

Keras LSTM can return output, hidden state, and cell state separately. Rust's
wrapper resets zero state for a single batch-1 step and returns cell state. A
sequence-wide Keras LSTM output therefore cannot be compared directly to that
Rust return value. The Python program checks both LSTM and GRU sequence shapes.

Embedding lookups are the closest direct counterpart: integer token IDs select
rows from a learned matrix. The Python program also adds learned position vectors,
then runs masked causal attention and cross-attention. These are capabilities of
Keras `MultiHeadAttention`; the Rust legacy SelfAttention wrapper is much more
restricted. NativeLlama's internally managed RoPE/attention is a separate inference
path, exercised with model snapshots rather than arbitrary Keras tensors.

<a id="keras-autoencoders-branches-and-mixtures"></a>

### Autoencoders, branches, and mixtures

Both languages can compose a 2→1→2 reconstruction network. Python also exposes
the encoder as a second Functional model. Neither example is a variational model:
there is no posterior parameterization, sampling/reparameterization, or KL term.

The Python split example has two independent named outputs from the same input;
the merge example averages their tensors. In Rust use independent model forwards
and explicit averaging for comparable semantics. Legacy ParallelModel does not
currently implement the same general behavior.

`TopTwoExperts` is a full custom Keras layer in the Python listing. It runs every
expert, selects two logits, normalizes their weights, and combines their outputs.
It illustrates routing, not sparse compute or a training-equivalent port of Rust
SparseMoE. Output width, gating normalization, and optimizer ownership must be
matched before attempting a faithful implementation.

<a id="keras-training-lora-reward-modeling-and-objectives"></a>

### Training, LoRA, reward modeling, and objectives

The Python script demonstrates:

- Built-in Dense LoRA with `lora_rank=1`; dropout/L2/AdamW controls.
- Sparse-label classification with logits and label-smoothed categorical CE.
- Terminal-aware GAE, a clipped PPO **policy term**, DPO loss, and temperature-scaled teacher KL.
- A shared scalar reward head trained on a preferred/rejected pair.

The Rust [handbook_training](#examples-handbook_training) additionally
shows averaged gradient accumulation, SGD and clipped AdamW adapter updates,
adapter weight reconstruction, ignored padding labels, reward updates, all three
optimizer hooks, Int8/Int4 storage, and shared-prefix cache truncation/removal.
Native LoRA backward currently does not accept a dropout mask; the accumulation
example sets dropout to zero so its forward/backward contract is consistent.

Keras gradients are normally computed by its backend. Rust objective hooks return
output-space derivatives; a real backend must propagate them through model
execution. The toy hook's flat parameters represent objective outputs, not
transformer weights. Python's PPO snippet includes only the clipped policy term;
Rust `ppo_objective` also includes configured value and entropy terms. Python's
KL snippet omits Rust's configurable supervised mixture. These demonstrate related
components, not numerical equivalence of the entire losses.

DoRA, paged caches, native checkpoint formats, continuous batching, and grammar
masking have no direct core-Keras counterparts. A Keras Dense layer with a quantized
option is not automatically equivalent to Lighter's group-wise packed Int4 matrix.
Use the complete native programs for those operations. The native regex constraint
checks completed text but does not currently prune invalid prefixes; choice, JSON
and finite GBNF expose separate prefix checks.

<a id="keras-persistence-and-portability"></a>

### Persistence and portability

The Python program writes/reads a `.keras` model and `.weights.h5`, checking equal
predictions in a temporary directory. The Rust comparison constructs registered
variables, saves safetensors, and reloads into the **same VarMap**, also checking
equal predictions. This avoids the incomplete legacy architecture/JSON-weight
restoration paths described in the handbook.

To create a separate restored Rust model, reconstruct the exact same names and
shapes in a fresh VarMap, then load its weights. Keras `.keras` serialization and
Candle safetensors are not drop-in replacements. Cross-language weight transfer
requires architecture recreation, name mapping, kernel transposes, dtype checks,
and a prediction comparison on identical inputs. Custom Keras layers additionally
need serialization registration/configuration; this script only saves its standard
regression model, not the custom expert layer.

<a id="keras-jepa-and-continuous-time-references"></a>

### JEPA and continuous-time references

The Python JEPA-like example excludes a target image patch, predicts its target
encoder representation from visible patches, and updates target weights by EMA.
It is a small training demonstration, not an exact port of Lighter's geometry-aware
predictor or a complete I-JEPA/V-JEPA research implementation. Full Rust image and
video execution/training examples are in the cookbook.

The Python continuous-time section runs a scalar RK4 decay solver and an elapsed-time
gated recurrence. RK4 validates `y(1) ≈ exp(-1)`. The recurrence is a custom reference,
not a Keras-provided CfC and not equivalent to Lighter's recurrent parameterization.
For Euler, Heun, RK4, LiquidCell, CfC, and stacked LFM, use the complete Rust listings.

<a id="keras-complete-executable-python-listing"></a>

### Complete executable Python listing

The program below is the full contents of the linked source file. Every function
is called by `main`; assertions check shapes, finiteness, deterministic layer
values, and persistence. These checks establish runnable examples, not model quality.

```python
"""Executable Keras 3 counterparts for docs/handbook.md#keras.

Run with KERAS_BACKEND=jax python examples/python/keras_comparison.py.
No downloaded models, credentials, or persistent output files are required.
These compare concepts and shapes, not identical initialization/training behavior.
"""
import os
os.environ.setdefault("KERAS_BACKEND", "jax")

import tempfile
from pathlib import Path
import keras
from keras import layers, ops
import numpy as np

keras.utils.set_random_seed(42)


def array(value):
    return np.asarray(ops.convert_to_numpy(value))


def finite(value):
    assert np.isfinite(array(value)).all()


def regression_and_classification():
    x = np.array([[1, 2], [2, 3], [3, 4], [4, 5]], dtype="float32")
    y = x.sum(axis=1, keepdims=True)
    model = keras.Sequential([
        keras.Input(shape=(2,)), layers.Dense(4, activation="relu"), layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.SGD(0.01), loss="mse")
    history = model.fit(x, y, epochs=3, batch_size=1, shuffle=False, verbose=0)
    predictions = model.predict(x, verbose=0)
    assert predictions.shape == (4, 1)
    finite(history.history["loss"])
    classifier = keras.Sequential([keras.Input(shape=(2,)), layers.Dense(2)])
    classifier.compile(
        optimizer=keras.optimizers.AdamW(1e-3, weight_decay=0.01, global_clipnorm=1.0),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    )
    finite(classifier.train_on_batch(x, np.array([0, 0, 1, 1], dtype="int32")))
    print("regression and classification:", predictions.shape)
    return model, x


def scaling():
    x = np.array([[1, 2], [3, 4]], dtype="float32")
    minimum, maximum = x.min(), x.max()
    scaled = (x - minimum) / max(maximum - minimum, np.finfo("float32").eps)
    np.testing.assert_allclose(scaled * (maximum - minimum) + minimum, x)
    standardized = (x - x.mean()) / max(x.std(), np.finfo("float32").eps)
    np.testing.assert_allclose(standardized.mean(), 0, atol=1e-6)
    # Per-feature preprocessing differs from Lighter's global statistics.
    normalizer = layers.Normalization(axis=-1)
    normalizer.adapt(x)
    assert array(normalizer(x)).shape == x.shape
    print("global min-max/z-score; per-column Keras Normalization")


def layer_shapes():
    x = np.array([[1, 2]], dtype="float32")
    for activation in ["linear", "relu", "silu", "sigmoid", "softmax"]:
        assert array(layers.Dense(3, activation=activation)(x)).shape == (1, 3)
    logits = layers.Dense(3)(x)
    np.testing.assert_allclose(array(ops.exp(ops.log_softmax(logits))).sum(), 1, atol=1e-6)
    # Channels-last layouts; the Rust wrapper examples use channels first.
    signal = np.array([1, 2, 3, 4], dtype="float32").reshape(1, 4, 1)
    conv = layers.Conv1D(1, 2, use_bias=False, kernel_initializer="ones")
    np.testing.assert_allclose(array(conv(signal)).reshape(-1), [3, 5, 7])
    image = np.ones((1, 4, 4, 1), dtype="float32")
    assert array(layers.Conv2D(1, 2)(image)).shape == (1, 3, 3, 1)
    patch = np.array([1, 2, 3, 4], dtype="float32").reshape(1, 2, 2, 1)
    np.testing.assert_allclose(array(layers.AveragePooling2D(2)(patch)).reshape(-1), [2.5])
    np.testing.assert_allclose(array(layers.MaxPooling2D(2)(patch)).reshape(-1), [4])
    np.testing.assert_allclose(array(layers.UnitNormalization(axis=-1)(np.array([[3, 4]], dtype="float32"))), [[0.6, 0.8]])
    assert array(layers.Flatten()(np.ones((2, 3, 4), dtype="float32"))).shape == (2, 12)
    norm = layers.LayerNormalization(axis=-1)(x)
    assert array(norm).shape == x.shape
    regularized = layers.Dense(2, kernel_regularizer=keras.regularizers.L2(0.01))
    regularized(x)
    assert len(regularized.losses) == 1
    dropout = layers.Dropout(0.1)
    np.testing.assert_allclose(array(dropout(x, training=False)), x)
    finite(dropout(x, training=True))
    # Dense LoRA is Keras' implementation, not Lighter's legacy Dense::new2.
    lora = layers.Dense(2, lora_rank=1)
    assert array(lora(x)).shape == (1, 2)
    print("dense, activations, convolution, pooling, normalization, flatten, dropout, LoRA")


def recurrence_embedding_attention():
    sequence = np.ones((1, 3, 2), dtype="float32")
    lstm = layers.LSTM(4, return_state=True)
    output, hidden, cell = lstm(sequence)
    assert array(output).shape == array(hidden).shape == array(cell).shape == (1, 4)
    gru = layers.GRU(4, return_sequences=True)
    assert array(gru(sequence)).shape == (1, 3, 4)
    ids = np.array([[0, 2, 7]], dtype="int32")
    tokens = layers.Embedding(8, 4)(ids)
    positions = layers.Embedding(3, 4)(ops.arange(3))
    embedded = tokens + positions
    mask = np.array([[[True, True, False]] * 3])
    attended = layers.MultiHeadAttention(num_heads=2, key_dim=2)(
        embedded, embedded, attention_mask=mask, use_causal_mask=True,
    )
    assert array(attended).shape == (1, 3, 4)
    cross = layers.MultiHeadAttention(num_heads=2, key_dim=2)(
        embedded, np.ones((1, 2, 4), dtype="float32"),
    )
    assert array(cross).shape == (1, 3, 4)
    # Masking is propagated only to layers that support it.
    masked = layers.Masking(mask_value=0.0)(sequence)
    assert array(layers.LSTM(4)(masked)).shape == (1, 4)
    print("LSTM/GRU, token and position embeddings, causal/cross attention, masking")


def autoencoder_and_parallel_models():
    x = np.array([[1, 2], [2, 3]], dtype="float32")
    inputs = keras.Input(shape=(2,))
    latent = layers.Dense(1, activation="relu", name="encoder")(inputs)
    reconstruction = layers.Dense(2, name="decoder")(latent)
    autoencoder = keras.Model(inputs, reconstruction)
    autoencoder.compile(optimizer="sgd", loss="mse")
    finite(autoencoder.train_on_batch(x, x))
    encoder = keras.Model(inputs, latent)
    assert array(encoder(x)).shape == (2, 1)
    branch_a = layers.Dense(2, name="branch_a")(inputs)
    branch_b = layers.Dense(2, name="branch_b")(inputs)
    split = keras.Model(inputs, {"a": branch_a, "b": branch_b})
    assert array(split(x)["a"]).shape == (2, 2)
    merged = keras.Model(inputs, layers.Average()([branch_a, branch_b]))
    assert array(merged(x)).shape == (2, 2)
    print("autoencoder, multi-output branches, averaged ensemble")


class TopTwoExperts(layers.Layer):
    """Toy top-2 routing after all expert forwards; not sparse computation."""
    def __init__(self, width=4, experts=3):
        super().__init__()
        self.experts = [layers.Dense(width, activation="relu") for _ in range(experts)]
        self.gate = layers.Dense(experts)

    def call(self, x):
        scores = self.gate(x)
        values, indices = ops.top_k(scores, k=2)
        selected = ops.sum(ops.one_hot(indices, len(self.experts)), axis=-2)
        weights = ops.softmax(ops.where(selected > 0, scores, -1e9), axis=-1)
        outputs = ops.stack([expert(x) for expert in self.experts], axis=1)
        return ops.sum(outputs * ops.expand_dims(weights, -1), axis=1)


def mixture_of_experts():
    x = np.ones((2, 4), dtype="float32")
    moe = TopTwoExperts()
    assert array(moe(x)).shape == (2, 4)
    print("custom top-2 expert mixture (all experts evaluated)")


def save_and_restore(model, x):
    before = array(model(x, training=False))
    with tempfile.TemporaryDirectory(prefix="lighter-keras-") as directory:
        path = Path(directory) / "regression.keras"
        model.save(path)
        restored = keras.models.load_model(path)
        np.testing.assert_allclose(array(restored(x, training=False)), before, atol=1e-6)
        weight_path = Path(directory) / "regression.weights.h5"
        model.save_weights(weight_path)
        clone = keras.models.clone_model(model)
        clone.load_weights(weight_path)
        np.testing.assert_allclose(array(clone(x, training=False)), before, atol=1e-6)
    print("model and weight persistence: equivalent predictions")


def reinforcement_and_distillation():
    # Numerical objective examples. Full agents require an environment and optimizer loop.
    rewards, values, done = [1.0, 0.5], [0.2, 0.3, 0.0], [False, True]
    advantages = np.zeros(2, dtype="float32")
    running = 0.0
    for t in reversed(range(2)):
        alive = 0.0 if done[t] else 1.0
        delta = rewards[t] + 0.99 * values[t + 1] * alive - values[t]
        running = delta + 0.99 * 0.95 * alive * running
        advantages[t] = running
    ratios = ops.exp(ops.array([-0.75, -0.8]) - ops.array([-0.8, -0.7]))
    policy_loss = -ops.mean(ops.minimum(
        ratios * advantages, ops.clip(ratios, 0.8, 1.2) * advantages,
    ))
    chosen, rejected = ops.array([0.4, 0.3]), ops.array([-0.2, -0.1])
    dpo_loss = ops.mean(ops.softplus(-0.1 * (chosen - rejected)))
    teacher, student = ops.array([4., 1., 0.]), ops.array([1., 2., 0.])
    temperature = 2.0
    teacher_p = ops.softmax(teacher / temperature)
    kl = ops.sum(teacher_p * (ops.log(teacher_p) - ops.log_softmax(student / temperature)))
    distillation = temperature ** 2 * kl
    finite(policy_loss); finite(dpo_loss); finite(distillation)
    logits = np.array([[2., 0., -1.], [0., 2., -1.]], dtype="float32")
    targets = np.eye(3, dtype="float32")[[0, 1]]
    ce = keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
    finite(ce(targets, logits))
    # Shared scalar reward head; the difference models pairwise preference.
    chosen_input = keras.Input((3,)); rejected_input = keras.Input((3,))
    head = layers.Dense(1)
    preference = keras.Model([chosen_input, rejected_input], head(chosen_input) - head(rejected_input))
    preference.compile(optimizer="adam", loss=keras.losses.BinaryCrossentropy(from_logits=True))
    finite(preference.train_on_batch(
        [np.array([[1., 0., 0.]], dtype="float32"), np.array([[0., 0., 1.]], dtype="float32")],
        np.ones((1, 1), dtype="float32"),
    ))
    print("GAE, clipped policy objective, DPO, distillation, smoothed CE, reward preference")


def jepa_reference():
    # Tiny patch predictor inspired by JEPA; not identical to Lighter JepaTrainer.
    online = keras.Sequential([keras.Input((4,)), layers.Dense(8)])
    target = keras.models.clone_model(online)
    target.set_weights(online.get_weights())
    target.trainable = False
    predictor = keras.Sequential([keras.Input((8,)), layers.Dense(8)])
    context = keras.Input((4,))
    learner = keras.Model(context, predictor(online(context)))
    learner.compile(optimizer=keras.optimizers.SGD(0.01), loss="mse")
    pixels = np.arange(16, dtype="float32").reshape(4, 4) / 15
    # Exclude the upper-right target patch from the visible context.
    patches = pixels.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4)
    visible = patches[[0, 2, 3]].mean(axis=0, keepdims=True)
    withheld = patches[[1]]
    for _ in range(3):
        expected = array(target(withheld, training=False))
        finite(learner.train_on_batch(visible, expected))
        target.set_weights([
            0.99 * old + 0.01 * new
            for old, new in zip(target.get_weights(), online.get_weights())
        ])
    print("tiny JEPA-like masked predictor with an EMA target")


def continuous_time_reference():
    # Same qualitative decay example as NeuralOde; NumPy scalar solver.
    state, dt = 1.0, 0.05
    for _ in range(20):
        k1 = -state
        k2 = -(state + dt * k1 / 2)
        k3 = -(state + dt * k2 / 2)
        k4 = -(state + dt * k3)
        state += dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    np.testing.assert_allclose(state, np.exp(-1), atol=1e-6)
    # A compact elapsed-time gated recurrence; no built-in Keras CfC layer.
    candidate = layers.Dense(4, activation="tanh")
    hidden = ops.zeros((1, 4))
    samples = [np.array([[1., 0.]], dtype="float32"), np.array([[0., 1.]], dtype="float32")]
    for sample, elapsed in zip(samples, [0.1, 0.5]):
        gate = np.exp(-elapsed)
        hidden = gate * hidden + (1 - gate) * candidate(sample)
    finite(hidden)
    print("RK4 decay and custom elapsed-time recurrence")


def main():
    print(f"Keras {keras.__version__}; backend={keras.backend.backend()}")
    model, x = regression_and_classification()
    scaling()
    layer_shapes()
    recurrence_embedding_attention()
    autoencoder_and_parallel_models()
    mixture_of_experts()
    save_and_restore(model, x)
    reinforcement_and_distillation()
    jepa_reference()
    continuous_time_reference()
    print("all Keras comparison examples completed")


if __name__ == "__main__":
    main()
```


<a id="keras-graph-to-text"></a>

### Graph-to-text

See the [graph-to-text comparison and executable Python realization](#graph-keras-comparison).

<a id="keras-g-retriever-graph-soft-prompts"></a>

### G-Retriever graph soft prompts

The [G-Retriever integration](#retriever) uses PyTorch/PyG for the upstream graph
encoders and Transformers for embedding-input language modeling. Keras has no built-in
PCST retriever or equivalent upstream GNN implementation; reproducing this path requires
implementing compatible graph message passing, pooling, projection and an LM accepting
input embeddings. The included Python examples execute the actual PyTorch architecture.

<a id="feature-shapes"></a>

## Feature data structures

A feature helper stores values using samples, time steps and spatial feature dimensions. A single sample still needs a batch dimension when the consuming layer expects one. Use the shape table in the data explanation chapter to trace each axis. The original notes below describe the design conventions; a particular layer can require a narrower shape contract.

The hardest part in machine learning often revolves around ensuring data is in the correct shape for various machine learning models and operations. This is because machine learning algorithms are highly sensitive to the shape of the input data. Common challenges are:

- **Understanding Data Representation**: Knowing how data should be structured for different types of machine learning tasks can be complex. Each type of data and model architecture might require a unique arrangement of dimensions.
- **Batch Size Dimension**: Many machine learning frameworks expect the first dimension of your data array to represent the batch size. Remembering to include this dimension, even if you're processing a single sample (batch size of 1), can be a source of error.
- **Channels or Features Dimension**: For image data, it's crucial to know the expected order of dimensions, particularly the position of the channels dimension (e.g., RGB channels in an image). Some frameworks expect channels first (channels, height, width), while others expect channels last (height, width, channels).
- **Time Series and Sequences**: For sequential data, such as time series or text, managing the sequence length dimension can be challenging. Different models and parts of the same model might require this dimension to be in different places or to be of a specific length, sometimes necessitating padding or truncation of sequences.
- **Broadcasting Rules**: Understanding how arrays with different shapes are treated in operations (like addition, multiplication) according to broadcasting rules can be tricky. Inadvertently broadcasting arrays can lead to unexpected results or performance issues.
- **Reshaping and Squeezing**: Knowing when and how to reshape (change the dimensionality) or squeeze/unsqueeze (adding or removing singleton dimensions) without losing the meaningful structure of your data requires a good understanding of your data and the model's expectations.

And many more aspects...

With Lighter this should become a bit easier. A helper *class* will support to describe the intended input features dimensionality:
- *1st dimension (**rational**):* Samples, at least one
- *2nd dimension (**temporal**):* Time steps, at least one
- *3rd dimension (**spatial**):* Features in different dimensions, at least a 1 dimensional sequence

Per definition *batches* are not taken under consideration. This kind of shuffling operation we move to the *model* fitting consideration. The model fittign will consider multiple given samples for the batching.

<a id="custom-layers"></a>

## Integrating custom layers

A custom layer needs a defined input/output shape, a forward computation and a plan for parameters and persistence. Registering a name alone does not teach the model loader how to reconstruct it. Inspect a working existing layer, then connect its parameter handling and deserialization path before relying on save/load. The historical extension checklist is retained below.

The following extensions are to make for integrating an own layer:
- lib/layers.rs   Implement trainable and serialization for the own layer.
- lib/models.rs   Implement the deserialization for the own layer.
- lib/layer/      Add the layer implementation in an own .rs file
    * mod.rs      add the specific .rs file

<a id="autoencoders"></a>

## Autoencoders

An autoencoder trains a model to reconstruct its own input. An encoder maps features into a smaller latent representation; a decoder maps that representation back to the input shape. For `[batch, features]`, a dense encoder can produce `[batch, latent]` and the decoder must restore `[batch, features]`. Reconstruction loss compares the output with the original input rather than an external class label.

Lighter demonstrates this composition with existing layers; it does not expose a dedicated variational autoencoder training API. A variational autoencoder additionally learns a latent probability distribution and combines reconstruction loss with a distribution regularizer. That objective is not implied by composing two dense layers. Use the autoencoder recipe in the usage chapter for current executable scope.

A small latent width encourages compression, but low reconstruction error does not guarantee useful semantic features. Evaluate reconstruction on held-out inputs and inspect what changes when the latent representation is perturbed.
https://github.com/huggingface/candle/blob/fa06f5f5f9a05c8d0c246e761e94a73680c510a6/candle-transformers/src/models/stable_diffusion/vae.rs#L2

<a id="attention"></a>

## Attention mechanisms

Attention combines value vectors according to weights computed from query/key compatibility. Self-attention uses representations from the same sequence; cross-attention obtains keys/values from another input. Causal attention masks future positions so next-token prediction cannot inspect the answer tokens ahead of it. Multiple heads learn several projections, while grouped-query attention shares key/value heads across groups of query heads.

The native decoder contains its supported causal/grouped-query computations as model internals. The following table is a historical roadmap for standalone Candle wrappers; its “not integrated” labels do not describe the internal native decoder. Consult the usage capability table for current coverage.

We will support different attention approaches. [Candle](https://github.com/huggingface/candle) provides us a brought varierty on existing implementations.

| Type                  | From where                                                                                        |
|-----------------------|---------------------------------------------------------------------------------------------------|
| SelfAttention         | Integrated- [here](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/wuerstchen/attention_processor.rs) for one input sequence. Depending on the implementation it is also called a *dot product attention* or *global attention*.|
| CrossAttention (aka Co-Attention)       | Not Integrated so far - [here](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/stable_diffusion/attention.rs) for multiple input sequences |
| CausalSelfAttention   | Not Integrated so far - [here](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/llama2_c.rs) for parts of one or multiple input sequences e.g., only all token before the present. Depending on the implementation also called *local attention*. |
| MultiHeadAttention   | Not Integrated so far - [here](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/segment_anything/transformer.rs) for multiple concerns/ questions |
| MultiQueryAttention   | Not Integrated so far - [here](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/chatglm.rs) for multiple concerns/ questions but knowing the other concerns/ questions |
| GroupQueryAttention   | Not Integrated so far - [here](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/quantized_mpt.rs) for building logical groups between the questions |


*Terms*:
- Heads: amount on parallel questions on a given stream.
- Contexts: amount of parallel streams.
- Temporal: Time.
- Spatial: Dimensionality.

*Note: All attention should be available for multiple dimensions. This includes spatial transformer which acts in >= 2D space (=spatial) as required for CNN applications.*

More complex models mappes as own layer:
- https://arxiv.org/html/2312.06635v3
    * [here](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/rwkv_v6.rs)

<a id="embeddings"></a>

## Embeddings and positional information

An embedding lookup maps discrete vocabulary IDs to learned vectors. ID `7` is a category index, not the numerical feature value seven. Position mechanisms add or encode order information so a model can distinguish otherwise identical tokens at different locations. RoPE rotates query/key representations according to position; a generic embedding lookup is a different operation.

The native decoder implements its supported positional mechanism internally. The historical references below concern additional standalone wrappers and future experiments. Their performance figures are external reference claims, not benchmarks of this checkout.

Both embedding ( e.g., Word2Vec) and encoding (e.g., Bag of words) is about representing data in a different space. Embedding isually talks about continous vector spaces (aka sequences), usually capturing semantic relationships, where encoding also includes compressing and dimensional reduction.

The following embeddings shall be integrated from [Candle](https://github.com/huggingface/candle):


| Type                  | From where                                                                                        |
|-----------------------|---------------------------------------------------------------------------------------------------|
| Embedding         | Integrated - Standard layer |
| Timestep Embedding         | Not Integrated so far - [here](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/stable_diffusion/embeddings.rs) for relative (aka local) positional encoding |
| Positional Embedding         | Not Integrated so far - [here](https://github.com/huggingface/candle/blob/0c5eecbc0faa7e642210800c735ad8137d5a9e08/candle-transformers/src/models/segment_anything/prompt_encoder.rs#L25) for absolute positional encoding |
| Falcon Rotary Positional Embedding         | Not Integrated so far - [here](https://github.com/huggingface/candle/blob/0c5eecbc0faa7e642210800c735ad8137d5a9e08/candle-transformers/src/models/falcon.rs#L163) for absolute and relative positional encoding |
| Sinusoidal Positional Embedding         | Not Integrated so far - [here](https://github.com/huggingface/candle/blob/0c5eecbc0faa7e642210800c735ad8137d5a9e08/candle-transformers/src/models/marian.rs#L108) for absolute and relative positional encoding |


Notes:
- Dimensional reduction such as PCA tend to perform poorly. Check e.g., [here](https://arxiv.org/abs/2205.11498).
- Next steps will be
    * [here](https://arxiv.org/abs/2205.13147), by just using randomly n dimensions.
    * Quantization via
        - Converting to binary quatization (aka **hemming distance** ). Speedup is roughly a factor 3 (compare [here](https://huggingface.co/blog/embedding-quantization)).
        - Converting to scalar quantization (aka **integer** mapping). Speedup is roughly a factor 24 (compare [here](https://huggingface.co/blog/embedding-quantization)).

<a id="mixture-of-experts"></a>

## Mixture of Experts

A Mixture of Experts has multiple parameterized expert functions and a router that chooses or weights their contributions. Sparse routing evaluates only some experts for an input; attention instead computes weighted relationships among representations. Neither mechanism inherently guarantees a particular arithmetic interpretation, task partition or speedup. Routing, capacity limits and execution determine the result and synchronization requirements.

Lighter's standalone SparseMoE is experimental; the working example and limitations are in the usage chapter. The text below is retained as historical motivation and taxonomy. Its arithmetic illustration is conceptual, not an executable result or a general property of attention versus MoE.

For the same sequence, a Mixture of Experts (MoE) model would use a gating mechanism to **decide** which 'expert' is best suited to handle the prediction at each point in the sequence. This implies it takes one out of many experts. In comparison e.g. Multi-Head Attention is about splitting focus to **simultaneously** capture different types of relationships in the data. It's like having multiple lenses to look at the data from different angles at once.

From the parallelism perspective MoE does not have a bottleneck, because the sequence can be splittet into simultanous processed snippets of the input. A Multi-Head Attention whoudl require a syncronization point (merge).

Example:

Input sequence: 1 2 3 4 - 5 6 + 8 9
Expected outcome: 4-5 = - 1 and 6+8= 14

MoE: 2 results -1 and 14, because of different tasks in a sequence
Attention: Might be 13, because an sense within the given sequence.

This is a classical MoE case. We will have an expert for the **minus** and one for **plus** operation.


Fields of application:

**MoE**
- Distributing/ Sharding of large sequences
- Heterogenous data which can be distinguid into subsets
- Different tasks to perform

**Multi Head Attention**
- Capture and attent different, complex dependencies and relationships within a stream
- Cross and multi modal data sequences

We will support different attention approaches. [Candle](https://github.com/huggingface/candle) provides us a brought varierty on existing implementations.

| Type                  | From where                                                                                        |
|-----------------------|---------------------------------------------------------------------------------------------------|
| SparseMoe         | Not integrated, because of own implementation - Can be found in Candle [here](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/qwen2_moe.rs) and [here](https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/mixtral.rs). Only run a few tasks (aka top k experts) and teh other will be *nulled*. |
| Multi Task Moe         | **N/A** in Candal. Run multiple tasks in parallel and collects each results individually. |
| Gated Moe         | **N/A** in Candle. Not only performance an expert selection, but also adapts the trainign process. |
| Hierachical Moe         | **N/A** in Candle. Stacks experts to evolutionary determine a result. |
| Conditional Moe         | **N/A** in Candle. Also know as "Switch MoE". Purpose is the selection of at least one expert for a specifi task. |

*Note:*: Sparse (aka Sparsamkeit).

<a id="masking"></a>

## Masking and quantization

Masking suppresses selected inputs or interactions. A feature mask can be multiplied elementwise with data, while a causal attention mask prevents selected query/key pairs from contributing. These masks solve different problems: replacing a feature value with zero is not the same as excluding a token from attention. Check the expected mask shape and whether the layer interprets zero as a value or as an exclusion.

Quantization represents values at lower precision. Native int8/int4 projection examples store quantized weights and associated scaling information; reconstruction is approximate. This affects memory and numerical error, while masking changes which information participates in a computation. Validate quality on your task when selecting a quantization configuration.

The executable masking and quantization recipes use existing operations. There is no implied universal feature-masking or feature-quantization layer that covers every model. Weight quantization, input quantization and output embeddings require separate representation choices.
**Masking** is about handling sequences with varying lengths.

<a id="model-merging"></a>

## Model merging and ensembles

Prediction averaging and parameter merging are different operations. An ensemble can average two models' compatible output scores without modifying either model. Parameter averaging requires compatible architectures, tensor names and shapes, and is not guaranteed to preserve quality even then. Concatenating outputs creates a larger representation rather than averaging predictions.

The parallel-model recipes explain the current experimental split/merge behavior. The historical list below describes possible strategies; it does not mean each strategy is implemented by the same merge switch. Validate dimensions and a held-out metric before adopting a combination.

**Merging** refers to the concept of *ensembling learning* that combines multiple models to create a stronger and more robust one (aka model merging).
Some commonly methods we tend to implement the following:

- **Ensemble learning**: This method involves training multiple models separately and then combining their predictions. The combination can be done in various ways, such as averaging, weighted averaging or voting.
- **Model stacking** (aka meta learning): The predictions of multiple models are used as input for a new model - the merging one.
- **Model blending**: Similar to model stacking, blending combines the predictions of multiple models. However, instead of using a meta-learner, blending typically involves a simpler approach such as taking the average of predictions.
- **Bayesian Model Combination**: This advanced technique involves using Bayesian methods to combine models. It takes into account the uncertainty in the predictions of each model and can be more effective than simple averaging or voting in certain cases.

Not in scope:
- **Feature union**: This technique is used primarily in data preprocessing, where features generated by different models or transformations are combined into a single feature set.
- **Cascade generalization**: This method involves using the predictions of one model as an input feature for another model. Unlike stacking, where the meta-learner is trained after all base models are trained, in cascade generalization, each model can be trained sequentially, with each new model incorporating the predictions of the previous models as features.

<a id="reinforcement"></a>

## Reinforcement learning objectives

Reinforcement learning optimizes decisions using rewards and state transitions. A reward model estimates preference or utility; a policy produces actions. Reward-model training alone does not implement an environment, collect rollouts or update a policy.

Lighter provides objective and gradient-related primitives such as advantage estimation, PPO/DPO-related calculations, reward-head updates and distillation. The native examples demonstrate those computations on inspectable values. They are not a complete agent-training service or a ready-made rollout system.

When using advantage estimation, terminal boundaries matter: the return after an episode ends must not inherit value from the next episode. When comparing policy objectives, record how rewards, reference-policy scores and masks were computed. Use the native objective and fine-tuning recipes for actual signatures and current integration limits.
compare https://github.com/huggingface/candle/tree/main/candle-examples/examples/reinforcement-learning

<a id="transformer-catalog"></a>

## Upstream transformer catalog

A model catalog lists architectures available in an upstream project. Loading one in Lighter still requires a compatible configuration, tokenizer, supported weights and an implemented execution path. The native decoder and legacy BERT/Llama wrappers have the scope described in the usage chapter; they do not automatically support every entry below.

This historical catalog retains upstream source links for exploration. File locations and architecture coverage on upstream main branches can change independently of the versions pinned by this repository.

Candle Rust source code reflected to the list given here https://huggingface.co/docs/transformers/model_doc/albert

Elements are:
* Model, format e.g., safetensors: https://github.com/huggingface/safetensors
    - Config including architecture, format json, see https://huggingface.co/docs/transformers/en/main_classes/configuration
    - Weights, format e.g., .pkl
* Dataset



* Text models e.g.,
    - BERT, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/bert.rs
    - GPTBigCode, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/bigcode.rs
    - MISTRAL, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/mistral.rs
    - MIXTRAL, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/mixtral.rs
    - BLIPText, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/blip_text.rs
    - ChatGLM, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/chatglm.rs
    - ConvMixer, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/convmixer.rs
    - DistilBert, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/distilbert.rs
    - Falcon, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/falcon.rs
    - Gemma, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/gemma.rs
    - LLama, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/llama.rs
    - LLamaV2, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/llama2_c.rs
    - Mamba, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/mamba.rs
    - Marian, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/marian.rs
    - MPT, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/mpt.rs
    - OLMO, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/olmo.rs
    - Persimmon, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/persimmon.rs
    - Phi, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/phi.rs
    - Phi3, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/phi3.rs
    - QWen2, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/qwen2.rs
    - RWKV, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/rwkv_v5.rs
    - StableLM, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/stable_lm.rs
    - Starcoder2, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/starcoder2.rs
    - T5, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/t5.rs
    -
* Vision models e.g.,
    - ConvNext,https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/convnext.rs
    - DinoV2, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/dinov2.rs
    - EfficientNet, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/efficientnet.rs
    - EfficientVit, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/efficientvit.rs
    - MobileOne, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/mobileone.rs
    - VGG, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/repvgg.rs
    - ResNet, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/resnet.rs
    - Segformer, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/segformer.rs
    - VID, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/vit.rs
    -
* Audio models e.g.,
    - Encodec, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/encodec.rs
    -
* Multimodal models e.g.,
    - BLIP, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/blip.rs
    - TCoCR, https://github.com/huggingface/candle/blob/main/candle-transformers/src/models/trocr.rs
    -
* Reinforcement models
* Time series models
* Graph models

<a id="transformers-parity"></a>

## Native Transformers parity: staged implementation

The compatibility target is native Rust implementation in stages. The comparison uses
[Hugging Face Transformers](https://github.com/huggingface/transformers) revision
`469230357aab0f2b303b0d638c1f8d06edb14184`, whose source tree contains **521 model
directories**. A directory can represent variants, helpers or deprecated code; this
count is not a count of architectures supported by Lighter. The complete inspected
model-directory and pipeline-module inventory is recorded in
[transformers_inventory.json](../tools/transformers_inventory.json).

Stage 1 provides shared loading, batch tokenization and causal-generation APIs in
[`candlelighter::transformers`](../lib/transformers.rs). The module uses Rust, the
existing native decoder and the Rust `tokenizers` crate. It does not call Python or
claim support for every upstream architecture.

### Completed first-stage capabilities

| Transformers concept | Native API | Current behavior |
| --- | --- | --- |
| `AutoConfig.from_pretrained` | `AutoConfig::from_pretrained` | Read local/Hub config without loading model weights; preserve unknown fields. |
| Config export | `HuggingFaceConfig::save_pretrained` | Write `config.json`; weights are not exported. |
| `AutoTokenizer.from_pretrained` | `AutoTokenizer::from_pretrained` | Load tokenizer JSON and available pad metadata independently of weights. |
| Batched tokenization | `encode_batch` and `BatchEncoding` | IDs, token-type IDs and attention masks with left/right padding or truncation. |
| Tokenizer export/decode | `save_pretrained`, `decode`, `batch_decode` | Save tokenizer JSON including explicit pad configuration; decode with special-token control. |
| `GenerationConfig` | `GenerationConfig` | Load/save a supported configuration subset and map it to native sampling. |
| `AutoModelForCausalLM` | `AutoModelForCausalLM::from_pretrained` | Dispatch supported native decoder layouts; reject unsupported model types before loading weights. |
| Text-generation pipeline | `TextGenerationPipeline::from_pretrained` and `generate` | Reuse a loaded model, continuously batch independent prompts and return results in input order. |
| Repeated n-gram blocking | `no_repeat_ngram_size` | Block completion of n-grams already found in prompt/generated history. |
| Multi-token bad words | `bad_words_ids` | Block a word's final token only when its preceding token sequence matches the history suffix. |

Supported decoder model types currently include `llama`, `mistral`, `qwen2`, `qwen3`,
`clm` and `contrastive_lm`, plus recognized causal architecture declarations using
compatible tensor layouts. This is narrower than support for every variant in those
families. Existing configuration, positional encoding, weight-layout and dtype limits
still apply. Auto loading does not manufacture a missing architecture implementation.

### Run the pipeline example

```bash
cargo run --release --no-default-features --features native --example transformers_pipeline -- \
  /path/to/hf-model "Explain Rust ownership:" "Explain graph retrieval:"
```

A Hub model ID can replace the local directory. That invocation downloads compatible
model weights and requires enough memory for the model. `HF_TOKEN` is optional for
public models and supplies credentials when needed. Use a local tiny checkpoint for
an offline smoke run. [Complete source](../examples/transformers_pipeline.rs).

The example loads metadata/tokenizers, creates one pipeline, and requests two
continuations with greedy decoding and a repeated-trigram restriction. Its JSON output
contains the original prompt, generated continuation, token counts and finish reason.
The native output is continuation-only; it is not an exact copy of the Python pipeline's
full-prompt return format.

### Load and tokenize independently

```rust
use candlelighter::transformers::*;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = LoadOptions::default();
    let config = AutoConfig::from_pretrained("/path/to/hf-model", &options)?;
    let mut tokenizer = AutoTokenizer::from_pretrained("/path/to/hf-model", &options)?;
    // If pad metadata is absent, choose an existing appropriate vocabulary token.
    // Do not invent an ID or resize embeddings implicitly.
    tokenizer.pad_token_id = config.extensions.get("pad_token_id")
        .and_then(|v| v.as_u64()).and_then(|n| u32::try_from(n).ok())
        .or(tokenizer.pad_token_id);
    let batch = tokenizer.encode_batch(
        &["hello world".into(), "hello".into()],
        &TokenizationOptions {
            padding: Padding::Longest, padding_side: Side::Left,
            ..Default::default()
        },
    )?;
    println!("{:?}", batch.attention_mask);
    Ok(())
}
```

Padding requires a valid configured pad ID when actual padding is needed. Real tokens
receive mask value `1`; added pad positions receive `0`. Token-type IDs are padded
with zero. Explicit truncation requires `max_length`; overlong inputs otherwise fail.
Left truncation retains the tail and right truncation retains the prefix of the encoded
sequence. This is plain token-level truncation, not pair-aware truncation or a promise
that all special-token boundaries remain intact. Padding helpers do not change model
weights or make the native pipeline use a padded tensor backend: its engine still
tracks each request's unpadded token sequence.

The initial tokenizer API handles single strings and batches. It does not yet implement
text-pair strategies, overflowing windows, offset mappings or arbitrary Jinja chat
templates. Those remain explicit gaps rather than silently emulated outputs.

### Configure native generation

```rust
use candlelighter::transformers::*;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut pipeline = TextGenerationPipeline::from_pretrained(
        "/path/to/hf-model", &LoadOptions::default(), 4,
    )?;
    pipeline.generation_config = GenerationConfig {
        max_new_tokens: Some(64), do_sample: true, temperature: 0.7,
        top_p: 0.9, top_k: 50, no_repeat_ngram_size: 3,
        ..Default::default()
    };
    let results = pipeline.generate(&["Describe the graph:".into()], 42)?;
    println!("{}", results[0].text);
    Ok(())
}
```

`max_new_tokens` takes precedence over total `max_length`. The former counts generated
tokens, the latter includes prompt tokens. Likewise `min_new_tokens` takes precedence
over total `min_length`. The default total length is 20, following the familiar HF
configuration convention; specify an output length for longer prompts. Context overflow
fails before any request in the batch is submitted.

`do_sample=false` uses greedy decoding. Sampling enables temperature, top-k and top-p.
`top_k=0` disables top-k filtering. Repetition penalty, multi-token bad words, n-gram
blocking and stop strings use the native runtime. An explicit EOS token/list overrides
the model's default EOS stopping IDs. Minimum length suppresses stopping tokens until
its requirement is satisfied.

The pipeline reads available `generation_config.json`; missing optional config falls
back to defaults, while malformed files and non-404 Hub errors are reported. Nullable
inactive bad-word/stop-string lists decode as empty lists. Common metadata and neutral
unsupported settings are accepted, but active unsupported algorithms fail. For example,
`num_beams=4`, multiple return sequences or non-default typical sampling produce an
error in stage 1. The implementation does not silently run greedy decoding in place
of the requested beam search.

All prompts are validated before admission. Backend failures abort outstanding pipeline
requests and remove their caches, allowing the pipeline to be reused. A seeded sampling
batch derives a different seed per input index; moving a prompt to another batch position
can therefore change its sampled continuation. Greedy output still depends on backend
numerics and is not a guarantee of factual correctness.

N-gram and bad-word constraints also exist directly in `SamplingParams` and in the
HTTP completion/chat/graph request fields `no_repeat_ngram_size` and `bad_words_ids`.
Bad-word entries are tokenizer ID sequences, not literal strings. The native rule
blocks a word as soon as its complete prefix matches, even in a short initial context.
Transformers 4.57.6 delays multi-token bad-word checks until history length reaches
the full word length; this boundary behavior differs intentionally. Tokenize them with
the same tokenizer used by the model. If all possible tokens become blocked, generation
fails explicitly rather than emitting a forbidden token.

### Remaining gaps and implementation stages

This table covers upstream capability groups; the pinned inventory records the broad
architecture and pipeline surface. “Pending” is an implementation gap, not a claim that
an API with a similar name already provides it.

| Area | Existing Lighter coverage | Remaining native work | Stage |
| --- | --- | --- | --- |
| Shared config/tokenizer/model loading | Stage 1 above | More auto task dispatch, checkpoint export, tokenizer pair/offset APIs | 1 follow-ups |
| Decoding policies | Greedy, sampling, penalties, length/stop controls, constraints | Beam/group/diverse search, multiple candidates, assisted/speculative decoding, contrastive/typical strategies | 2 |
| Generation introspection | Token logprobs and native response metadata | Full score/logit sequences, hidden states, attentions, richer HF-style output objects | 2 |
| Cache policies | Native KV caches, paged sharing and cleanup primitives | Static/offloaded/quantized cache policies and full generation integration | 2 |
| Chat/templates/tools | Explicit ChatML/Llama3 hosting and native prompt/tool helpers | General tokenizer chat-template/Jinja semantics, multimodal chat processors | 2/5 |
| General Trainer API | Native objective/optimizer primitives and LM-head SFT | Full-transformer autograd, dataset collators, optimizer/scheduler state, accumulation, evaluation/callback/checkpoint loops | 3 |
| PEFT/QLoRA | Native head LoRA and grouped quantized base weights | Attention/MLP adapters, NF4/QLoRA-compatible training, broader adapter export/merge | 3 |
| Encoder and encoder-decoder tasks | Legacy Candle BERT wrapper; native causal decoder | Rust task heads and execution paths for masked LM, classification, tagging, QA, T5/BART/seq2seq | 4 |
| Other text architectures | Compatible native decoder families only | GPT variants, additional recurrent/state-space/MoE families and variant-specific layouts | 4 |
| Pipelines beyond causal generation | Graph workflows and dedicated demonstrations | Complete feature extraction, fill-mask, text classification, zero-shot, QA and task dispatch | 4 |
| Vision | JEPA examples and image utilities | Image processors, classification/detection/segmentation/depth task models | 5 |
| Audio, video, multimodal | Video JEPA and limited input utilities | Audio/video processors, ASR, TTS, multimodal model architectures and task pipelines | 5 |
| Time-series/document/table tasks | Liquid time-series examples | Compatible transformer-specific forecasters, table/document processors and task heads | 5 |
| Hardware execution | Reference CPU kernels; backend extension interface | GPU kernels, fused/flash attention, backend dispatch and mixed-precision training | 6 |
| Distributed and deployment/export | Native HTTP host | Tensor/pipeline parallel execution, distributed training, graph export and deployment optimizations | 6 |
| Quantization ecosystem | Grouped native int4/int8 | Additional formats/backends, activation quantization and per-architecture numerical validation | 6 |

A task head and an architecture are separate requirements. A sequence-classification
pipeline, for example, needs tokenizer processing, encoder execution, classification
weights, label metadata and postprocessing. Adding only a dispatcher name does not
complete that task. Each later stage should include execution tests against small
reference checkpoints, not just configuration recognition.

### Verification

```bash
cargo test --no-default-features --features native --lib transformers
cargo test --no-default-features --features server --test test_model_host native_transformers
```

Offline tests cover padding masks, left/right truncation, tokenizer/config round trips,
unsupported architecture preflight, generation length precedence, EOS overrides,
unsupported settings, nullable list decoding, n-gram and bad-word history restrictions,
ordered batch results and recovery after backend failure. A tiny real safetensors model
verifies model reuse and batched generation. The complete CLI also runs against a local
synthetic checkpoint. Shared decoder restriction vectors were independently checked
against Transformers 4.57.6; the short-context bad-word difference is described above. An optional live Hub metadata test is ignored by default;
its attempted run here was blocked by DNS resolution for `huggingface.co`. These tests establish the documented first-stage behavior;
they do not establish parity with all 521 upstream directories or every pretrained model.

<a id="graph-finetuning"></a>

## Graph-to-text instruction fine-tuning with Lighter

This example adapts the instruction-pair Graph2Text workflow to Lighter's native
Rust APIs: validate serialized triples, shuffle their order, tokenize an instruction
and target description, mask prompt labels, update a low-rank adapter, evaluate held-out
loss, save/reload the adapter and generate text. It uses the Alan Bean and Christopher
Nolan records from the supplied example.

### Scope compared with the supplied QLoRA recipe

| Supplied Python workflow | This Lighter example |
| --- | --- |
| Transformers, PEFT, bitsandbytes and TRL on a GPU | Native CPU decoder and `native_training` APIs in Rust |
| LoRA across attention and MLP projections | LoRA on the output language-model head only |
| NF4 double-quantized 4-bit weights | Optional native grouped int4/int8 weights with frozen base projections |
| AdamW and gradient accumulation | One example per SGD update through `supervised_fine_tune_step` |
| Model-specific chat template | Consistent plain-text instruction/graph/answer prompt |
| PEFT adapter export and merged HF model | Native JSON output-head adapter, loaded with `set_lm_head_lora` |
| Llama/Mistral/Qwen or a separate T5 seq2seq trainer | Compatible checkpoints supported by the native decoder; no T5 training path |

Native int4 + output-head LoRA is **not** the full NF4/QLoRA implementation in the
pasted Python recipe. The example does not resize tokenizer embeddings, implement
bitsandbytes optimizers, train every projection or export a merged Hugging Face model.
The narrower adapter may be less expressive for learning graph aggregation and syntax.
Use a small compatible model first; this reference CPU training path is not a claim
that a 7B–14B model fits a specific GPU memory budget.

### Step 1: Prepare graph/description pairs

[Training data](../examples/data/graph2text_train.json) and
[held-out validation data](../examples/data/graph2text_validation.json) are JSON arrays.
Each record has exactly `instruction`, `input` and `output`:

```json
{
  "instruction": "Convert the following graph relations into a fluent, cohesive paragraph.",
  "input": "<H> Alan Bean <R> birthPlace <T> Wheeler, Texas | <H> Alan Bean <R> almaMater <T> UT Austin | <H> Alan Bean <R> mission <T> Apollo 12",
  "output": "Born in Wheeler, Texas, Alan Bean graduated from UT Austin and served on the Apollo 12 mission."
}
```

`InstructionExample::triples` parses the subject/relation/object markers and validates
the resulting graph with Lighter's graph module. Empty fields, missing delimiters,
ambiguous reserved markers and empty instructions/targets fail. The `|`, `<H>`, `<R>`
and `<T>` markers are reserved by this simple serialization: use the property-graph
format for labels containing those markers, rather than guessing how to split them.

The delimiters remain ordinary tokenizer text, so the base tokenizer and vocabulary
stay compatible with the checkpoint. They may span several tokens. Consistency across
training and generation matters more than assuming each marker is one token. Adding
special tokens would require compatible resized embeddings and output projections,
which this example does not implement.

The two supplied training records are smoke data. Real training needs many varied
paired descriptions, relation types and entity combinations. Keep validation/test
records distinct from training records, and consider entity/template overlap when
constructing splits so memorization does not masquerade as generalization.

### Step 2: Load a frozen base and initialize an adapter

Run from the repository root with a compatible local Hugging Face snapshot:

```bash
cargo run --release --no-default-features --features native --example graph2text_finetune -- \
  /path/to/hf-model examples/data/graph2text_train.json \
  examples/data/graph2text_validation.json /tmp/graph2text-head-adapter.json 3 none
```

The positional arguments are `MODEL_DIR TRAIN_JSON VALIDATION_JSON ADAPTER_JSON`,
followed by optional epochs (default `3`) and quantization (`none`, `int4` or `int8`).
For a quantized base:

```bash
cargo run --release --no-default-features --features native --example graph2text_finetune -- \
  /path/to/hf-model examples/data/graph2text_train.json \
  examples/data/graph2text_validation.json /tmp/graph2text-head-int4.json 3 int4
```

The program initializes rank `8`, alpha `16`, zero adapter dropout and seed `42`.
The learning rate is `1e-4`; these inspectable defaults are in the source, not a tuned
configuration for every dataset. Quantized loading processes matrices as weights are
loaded. The base remains frozen while output-head adapter matrices are updated.

Choose a new adapter path for a training run: the example refuses to replace an
existing checkpoint. The JSON format records dimensions, LoRA configuration, low-rank
weights and the original base-model path. Dimensions and finite weights are validated
when loading. The path is informational identity, not a cryptographic fingerprint;
retain the exact original base weights, tokenizer and quantization configuration.

### Step 3: Train with answer-only next-token loss

Training constructs this consistent completion prompt:

```text
You are an expert Graph-to-Text verbalizer.
Convert the following graph relations into a fluent, cohesive paragraph.

Graph Input:
<H> Alan Bean <R> birthPlace <T> Wheeler, Texas | ...

Answer:
```

Both record order and triple order are shuffled reproducibly during training. The
facts stay unchanged, exposing the model to different serialization orders. Validation
uses the original order. Shuffling is augmentation, not a proof that the learned
model is invariant to all graph permutations.

`answer_only_tokens` tokenizes the prompt with its configured special-token behavior
and the answer without extra special tokens, then appends a configured EOS token if
available. It shifts labels once and replaces prompt targets with `-100`. For a
simplified sequence:

| Original sequence | Model inputs | Shifted labels |
| --- | --- | --- |
| `P0 P1 A0 EOS` | `P0 P1 A0` | `-100 A0 EOS` |

The position predicting the **first answer token** contributes to loss; positions
predicting the remaining prompt do not. This boundary is easy to shift incorrectly,
so the test explicitly checks it. The helper rejects context overflow instead of
silently discarding graph evidence or target tokens.

The implementation calls `supervised_fine_tune_step` with zero label smoothing and
SGD updates. It prints the mean per-record training loss after each epoch. Because
the mean weights each record equally, it is not a token-weighted dataset perplexity.
A tokenizer producing many tokens per marker can significantly increase context and
training cost.

In the supplied Python snippet, formatting everything into a `text` column does not
by itself establish answer-only masking. A TRL implementation must configure the
appropriate completion/assistant-only loss behavior or supply explicit masks. The
Rust example supplies shifted labels directly rather than relying on that assumption.

### Step 4: Reload, evaluate and generate

Use epochs `0` to load the saved adapter, compute validation loss and generate held-out
descriptions without performing updates:

```bash
cargo run --release --no-default-features --features native --example graph2text_finetune -- \
  /path/to/hf-model examples/data/graph2text_train.json \
  examples/data/graph2text_validation.json /tmp/graph2text-head-adapter.json 0 none
```

Use the same quantization argument as training (`0 int4` for the quantized example).
Both data paths remain required; the program validates the training input even in
reload mode. Reloading is evaluation, not optimizer-state resumption.

Validation uses teacher-forced answer-only loss. Generation uses greedy decoding
with a maximum of 128 new tokens, reduced if needed to fit the remaining context.
For each validation record, the program prints JSON with the input, reference,
prediction, finish reason and `entity_mention_recall`.

This mention score checks case-insensitive subject/object substrings. It can flag
missing entity names, but aliases and paraphrases can lower it even for correct text.
It cannot validate relation direction or factual precision: “Engine designed Babbage”
still mentions the same entities as “Babbage designed Engine.” Treat it as a diagnostic,
not a grounding metric or an implementation of NLI evaluation.

For a serious evaluation, compare held-out descriptions using text-quality measures
such as BLEU/ROUGE and separately annotate whether each graph relation/value is
expressed correctly. Review unsupported claims and missing facts. Those external
metrics and NLI models are not bundled into this example. Negative or ambiguous
training examples need accurate references; prompts alone cannot enforce grounding.

### Inference and serving after training

A reloaded native output-head adapter changes the model's logits, and the example
places that adapted model in `HuggingFaceBackend` and `NativeEngine` for generation.
An application can use the same adapted model when constructing the HTTP host.
`lighter-serve` does not currently have an adapter-path option; load and attach the
adapter through the Rust API before calling `model_host::serve`.

The adapter JSON is not a PEFT checkpoint and cannot be sent directly to vLLM or
Ollama. Exporting a merged HF checkpoint would require converting the adapter update
into compatible model weights, preserving configuration/tokenizer files and verifying
numerical equivalence. This example does not provide that conversion.

### Validation and complete source

```bash
cargo test --no-default-features --features native --lib graph_training
cargo test --no-default-features --features server --test test_model_host graph_sft
```

Tests cover parsing, seeded shuffling, first-answer/EOS label masking, context overflow,
checkpoint validation, lexical metric limitations, real native adapter updates with
float/int4/int8 bases, dimension mismatches and restored-logit equivalence. A CLI smoke
run on a tiny synthetic checkpoint verified training, saving, reload evaluation and
generation. That checkpoint is not a trained graph verbalizer; useful descriptions
require an appropriate pretrained model and representative training data.

[Complete program](../examples/graph2text_finetune.rs) ·
[Reusable data/checkpoint APIs](../lib/graph_training.rs).

<a id="arm-backends"></a>

## ARM vector, matrix and NPU backends

ARM CPU instruction extensions and an NPU are separate execution targets. NEON,
SVE and SME execute on the CPU. A TensorFlow Lite external delegate sends supported
parts of a compiled graph to an Arm NN backend, which may use a vendor NPU driver.
Installing Arm NN alone does not add an NPU to the machine. The vendor must supply
a compatible Arm NN backend, driver and model compilation flow.

### CPU kernels and runtime selection

`arm::dot` computes a checked F32 dot product. `arm::matmul` multiplies row-major
`A[m,k]` and `B[k,n]`, returning `C[m,n]`. Both validate dimensions, including
multiplication overflow. Empty dimensions return appropriately shaped zeros.
`Kernel::Auto` selects an available implementation; an explicit unavailable kernel
returns an error so a benchmark cannot accidentally measure scalar execution.

| Kernel | Build requirement | Runtime requirement | Operation |
| --- | --- | --- | --- |
| Scalar | `native` | Any supported host | Dot and matrix multiplication |
| NEON | `native`, AArch64 | OS-enabled NEON | Four-lane F32 dot with scalar tails |
| SVE | `arm-sve`, AArch64 Linux | Linux HWCAP SVE | Predicated, vector-length-independent F32 dot |
| SME | `arm-sme`, AArch64 Linux | Linux HWCAP2 SME and SME_F32F32 | Packed streaming outer products using ZA |

For dot products, automatic selection prefers SVE, then NEON, then scalar. Matrix
multiplication prefers SME when available, otherwise uses the selected vector dot
kernel against transposed B columns. SME packs partial tiles with zeros, checks the
streaming vector length before reading packed buffers, and restores the ordinary
calling convention's saved SIMD registers after leaving streaming mode. `dot`
rejects an explicit SME request because SME is exposed as a matrix operation.

The native decoder's **F32 matrix-vector projections** call the automatic vector
kernel, with the selection cached once per process. F16, BF16 and quantized weight
paths retain their existing kernels. The SME matrix API is available to applications;
it is not automatically applied to each single-token decoder projection. SVE/SME
are opt-in and currently Linux-only; AArch64 macOS uses NEON. Floating-point reduction
order and fused operations can change rounding, so compare with a tolerance.

```rust
use candlelighter::arm::{dot, matmul, ArmCapabilities, Kernel};
let capabilities = ArmCapabilities::detect();
println!("{capabilities:?}");
let value = dot(&[1., 2., 3.], &[4., 5., 6.], Kernel::Auto)?;
assert!((value - 32.).abs() < 0.001);
let matrix = matmul(&[1., 2., 3., 4.], &[5., 6., 7., 8.], 2, 2, 2, Kernel::Auto)?;
assert_eq!(matrix, vec![19., 22., 43., 50.]);
```

Run a model-free demonstration on any host, or compile optional kernels on ARM:

```bash
cargo run --no-default-features --features native --example arm_acceleration
cargo run --release --no-default-features --features arm-sve,arm-sme --example arm_acceleration
```

<a id="candle-arm-integration"></a>

### How ARM acceleration integrates with Candle

The `candle` feature enables Lighter's tensor layers and training API; `native`
enables the independent decoding engine and ARM kernels. Both can be enabled in
one application. Their integration is currently explicit at the tensor/buffer
boundary. The ARM additions do not register a new Candle `Device` or replace
Candle's tensor operation dispatch.

| Path | Integration with Candle | Gradient behavior |
| --- | --- | --- |
| Ordinary Candle layers and `Tensor::matmul` | Use Candle's own device, storage and kernels | Candle tracks supported differentiable operations |
| Native decoder F32 projections | Automatically call Lighter's NEON/SVE/scalar dispatch, independently of Candle | Native inference; no Candle autograd graph |
| `arm::dot` / `arm::matmul` | Explicit conversion between CPU F32 tensors and row-major buffers | Buffer conversion detaches the computation from Candle autograd |
| Arm NN / TFLite delegate | Separate compiled model and interpreter; usable alongside Candle preprocessing | Inference only; no gradients through the delegate |

#### Bridge a Candle tensor to an ARM CPU kernel

For a small inference operation, flatten two CPU F32 tensors into owned row-major
vectors, call `arm::matmul`, then reconstruct the result as a Candle tensor. The
example below checks dimensions and compares against Candle's own multiplication.
`flatten_all` preserves logical element order; extracting vectors also copies data,
so this bridge is not zero-copy. On x86 it exercises scalar fallback; on AArch64
it selects the compiled, OS-enabled ARM kernel.

<!-- source: ../examples/candle_arm_bridge.rs -->
```rust
//! Explicit, inference-only bridge between Candle tensors and ARM F32 kernels.
use candle_core::{Device, Tensor};
use candlelighter::arm::{matmul, Kernel};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::Cpu;
    let a = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &device)?;
    let b = Tensor::new(&[[7f32, 8.], [9., 10.], [11., 12.]], &device)?;
    let (m, k) = a.dims2()?;
    let (bk, n) = b.dims2()?;
    if k != bk {
        return Err("matrix inner dimensions differ".into());
    }
    let left = a.flatten_all()?.to_vec1::<f32>()?;
    let right = b.flatten_all()?.to_vec1::<f32>()?;
    let result = matmul(&left, &right, m, k, n, Kernel::Auto)?;
    let bridged = Tensor::from_vec(result, (m, n), &device)?;
    let reference = a.matmul(&b)?;
    let actual = bridged.flatten_all()?.to_vec1::<f32>()?;
    let expected = reference.flatten_all()?.to_vec1::<f32>()?;
    assert!(actual
        .iter()
        .zip(expected)
        .all(|(x, y)| (*x - y).abs() < 0.001));
    println!("{:?}", bridged.to_vec2::<f32>()?);
    Ok(())
}
```

```bash
cargo run --no-default-features --features candle,native --example candle_arm_bridge
# Enable optional SVE/SME on AArch64 Linux:
cargo run --release --no-default-features --features candle,arm-sve,arm-sme --example candle_arm_bridge
```

This recipe uses `Device::Cpu` and F32 inputs. For other devices or dtypes, explicitly
transfer to CPU and convert to F32 before extracting values, then transfer the result
back only if needed. Transfers, allocations and B transposition can dominate small
operations, so benchmark the entire bridge against `Tensor::matmul`. Do not assume
an ARM kernel is faster merely because it was selected.

Reconstructing with `Tensor::from_vec` creates a tensor without a differentiable
connection to the original inputs. For Candle training, retain Candle operations.
A differentiable ARM operator would require a Candle custom operation with a backward
implementation; this bridge does not supply one. For `C = A B`, that implementation
would need gradients `dA = dC Bᵀ` and `dB = Aᵀ dC`, along with shape, dtype and
device handling.

#### Use Candle preprocessing with the NPU backend

A Candle application can prepare inputs and invoke `TfliteNpuModel::forward` with
CPU token IDs, then wrap the returned F32 logits with `Tensor::from_vec` for further
Candle processing. The NPU model uses the [compiled tensor contract](#arm-backends):
fixed batch-one token inputs, optional mask/position inputs and causal logits. It
cannot execute arbitrary Candle tensor graphs or load Candle weights directly.
Model export and vendor compilation happen before inference.

For text generation, prefer `TfliteNpuBackend` with `NativeEngine`; they already
provide tokenizer integration and autoregressive sampling. Candle can remain in the
same application for other tensor work. For HTTP serving, create the delegate inside
`router_with_backend_factory` so construction, invocation and destruction happen
on the decode worker. Pass owned inputs or configuration across threads rather than
moving interpreter handles. Neither this NPU path nor the CPU buffer bridge supports
end-to-end Candle backpropagation.

### Arm NN / TensorFlow Lite external delegates

Enable `arm-npu` to use `TfliteNpuModel` or the tokenizer-backed
`TfliteNpuBackend`. The implementation dynamically loads the TensorFlow Lite C API
and the external plugin's `tflite_plugin_create_delegate` and
`tflite_plugin_destroy_delegate` symbols. It creates a model/interpreter, attaches
the delegate, allocates tensors, copies inputs, invokes the graph and copies logits
through the C API. Libraries and plugin option buffers remain owned for the required
lifetime; teardown destroys the interpreter before its delegate and model.

Use a TensorFlow Lite **C shared library** built for the target machine. A Python
`tensorflow` package or a static `.a` file is not the runtime library this API loads.
Use a matching delegate shared library, its dependencies and vendor backend plugins.
Arm NN builds differ: choose a build exporting the external plugin interface, not
just a library exposing Arm NN's separate C++ delegate constructor. See the
[Arm NN delegate source](https://github.com/ARM-software/armnn/tree/main/delegate)
and [TensorFlow Lite C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/core/c/c_api.h)
for the corresponding runtime versions and supported operators.

A compiled model must obey this contract:

| Tensor | Supported layout and type | Meaning |
| --- | --- | --- |
| Token IDs | Fixed `[1, context]`, int32 or int64 | Entire current prompt plus generated prefix, right-padded |
| Attention mask, optional | Same shape, int32 or int64 | One for prefix tokens, zero for padding |
| Position IDs, optional | Same shape, int32 or int64 | `0..context-1`; padded positions must be masked by the graph |
| Logits | F32 `[1, vocab]`, `[context, vocab]` or `[1, context, vocab]` | Next-token scores; multi-row outputs select the last real prefix row |

Input indices must cover **every model input exactly once**. Tensor shapes and byte
sizes are checked after allocation. All token IDs, including padding and configured
BOS, must fit the model vocabulary. For a single-row output, the exported graph
must itself select logits for the last unmasked token. A graph without a mask input
must implement its own equivalent padding semantics. Export dequantized F32 logits
if the NPU uses int8 internally; quantized logits output is currently rejected.

This backend evaluates the full prefix on every step and holds no per-sequence KV
cache. The native scheduler can interleave requests, but the default backend batch
method invokes them sequentially. Fixed batch-one graphs do not provide true tensor
batching. Stateful models requiring KV-cache inputs/outputs, dynamic sequence lengths,
and graph training are outside this backend's current contract. The `.tflite` graph
may implement any architecture satisfying the contract and supported by the delegate;
Lighter does not convert arbitrary Hugging Face safetensors to NPU binaries.

Prepare a matching model/tokenizer before running:

1. Export the model's causal forward pass with fixed sequence length, explicit
   input bindings and the logits layout above, using the model's supported exporter.
2. Convert/compile the graph with the TensorFlow Lite and vendor tools required by
   the selected NPU. Preserve causal masking and position semantics. Operator and
   memory limits vary by vendor, so conversion success must be checked on the target.
3. Compare next-token logits and generated text against the original model on several
   prompts, including padding, different prefix lengths and context boundaries.
4. Package the matching `tokenizer.json`, compiled `.tflite` graph, runtime/delegate
   libraries, driver dependencies and configuration. Record versions together.

For example, save the following deployment-specific JSON as `arm-npu.json`.
Replace the paths, indices and token IDs with those of your exported model. The
`backends` option is Arm NN-specific; replace `VendorNpuBackend` with the actual
registered vendor backend ID. `CpuAcc` uses an accelerated CPU backend and is useful
for integration checks, but does not demonstrate NPU execution.

```json
{
  "runtime_library": "/opt/tflite/lib/libtensorflowlite_c.so",
  "model": "/opt/models/causal-fixed.tflite",
  "delegate_library": "/opt/armnn/lib/libarmnnDelegate.so",
  "delegate_options": {"backends": "VendorNpuBackend"},
  "allow_cpu_fallback": false,
  "num_threads": 2,
  "token_input": 0,
  "attention_mask_input": 1,
  "position_ids_input": 2,
  "logits_output": 0,
  "pad_token_id": 0,
  "bos_token_id": 1
}
```

`allow_cpu_fallback` defaults to false. Missing plugin libraries/symbols, delegate
creation failures and delegated interpreter/allocation failures return an error.
With explicit fallback enabled, initialization retries without the delegate and
records its reason in `ExecutionReport`. Invoke failures are returned to the caller;
the backend does not switch execution targets during a request. Runtime/model or
tensor-contract failures cannot be repaired by fallback and still fail.

`delegate_active: true` means a delegate was attached and tensor allocation succeeded.
TensorFlow Lite delegates can leave unsupported operators on the CPU, or an Arm NN
backend can itself target the CPU. Use vendor profiling and partition diagnostics to
verify actual NPU coverage and performance. There is no universal NPU detection API
in this module.

### Generate text and expose the HTTP API

The examples use the same native engine and sampling API as the CPU decoder:

```bash
cargo run --release --no-default-features --features arm-npu --example arm_npu_generate -- \
  arm-npu.json /opt/models/tokenizer.json 2 "Explain graph retrieval:"
```

The third argument is the model's EOS token ID. The generation example validates
prompt length and bounds generated tokens by the compiled context. The execution
report prints delegate/fallback status and tensor limits before generation.

Delegate handles are deliberately not `Send` or `Sync`. Construct, invoke and drop
them on one thread. `model_host::router_with_backend_factory` creates the backend
on its decode worker, reports initialization errors before exposing routes, and
keeps the backend on that worker until shutdown. It accepts a `Send` factory even
when the returned backend is not `Send`.

```bash
cargo run --release --no-default-features --features arm-npu,server --example arm_npu_host -- \
  arm-npu.json /opt/models/tokenizer.json 2 512 127.0.0.1:8080
curl http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"arm-npu","prompt":"Explain graph retrieval:","max_tokens":32,"temperature":0}'
```

Set the fourth argument to the compiled context length; the factory checks it against
the model. Set `LIGHTER_API_KEY` to require bearer authentication. This example serves
completions, model discovery, graph completions, health and metrics through the shared
host. Chat templates are disabled by default; applications can set the appropriate
`HostConfig.chat_template` for their checkpoint. All existing admission, streaming,
cancellation and timeout rules apply. A long native delegate invocation cannot be
preempted mid-call by a timeout or cancellation.

### Validation and hardware limits

```bash
cargo test --no-default-features --features arm-npu,arm-sve,arm-sme,server \
  --test test_arm --test test_arm_npu --test test_model_host
rustup target add aarch64-unknown-linux-gnu
python3 tools/check_arm_codegen.py
```

Kernel tests compare auto/available implementations against scalar results for short
vectors, tails, rectangular tiles and malformed shapes. On an ARM machine they also
execute NEON and whichever enabled SVE/SME kernels the OS exposes. On other hosts,
explicit unsupported kernels must fail. The delegate tests compile a small C ABI
shim using `cc` and verify copying, mixed int32/int64 inputs, logits row selection,
engine integration, fallback and handle cleanup. A separate host test verifies
construction, execution and destruction of a non-Send backend on the worker thread.

The cross-codegen tool compiles the kernel bodies and assembly for AArch64 in an
isolated harness; it substitutes sequential iteration and removes serialization
derives, so it is not a full-crate cross-build. These checks passed on the development
x86 host. **No physical ARM CPU or NPU execution has been verified here.** Run the
kernel tests and vendor-model comparisons on the deployment device before relying on
performance or numerical parity claims.

<a id="examples"></a>

## Complete code examples

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

<a id="examples-example-index"></a>

### Example index

| Program | Feature selection | Validation / prerequisites | Capabilities |
| --- | --- | --- | --- |
| [serve_model](#examples-serve_model) | server | Compile-checked; compatible MODEL_DIR required | Embed the OpenAI-compatible host around a single native backend. See [host guide](#host) for CLI, curl, Python SDK and limits. |
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

`server` means `--no-default-features --features server` (and includes native).
`default` selects Candle and native; `native` means
`--no-default-features --features native`; `none` means `--no-default-features`.
The table covers runnable public workflows. The main handbook covers the status
of unimplemented KAN/DoRA/other roadmap APIs; there is no invented runnable code
for capabilities absent from the crate.

<a id="examples-choose-a-learning-path"></a>

### Choose a learning path

1. Porting Keras layers: run `handbook_layers`, then `handbook_keras`, and compare
   the [Python program](#keras-complete-executable-python-listing).
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

[Source](../examples/serve_model.rs) · [Complete API guide](#host)

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

<a id="examples-handbook_layers"></a>

#### handbook_layers

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

<a id="examples-handbook_experimental"></a>

#### handbook_experimental

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

<a id="examples-handbook_native"></a>

#### handbook_native

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

<a id="examples-handbook_training"></a>

#### handbook_training

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

<a id="examples-handbook_jepa_liquid"></a>

#### handbook_jepa_liquid

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

<a id="examples-native_advanced"></a>

#### native_advanced

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

<a id="examples-native_finetune"></a>

#### native_finetune

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

<a id="examples-native_prompt"></a>

#### native_prompt

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

<a id="examples-jepa"></a>

#### jepa

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

<a id="examples-liquid_networks"></a>

#### liquid_networks

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

<a id="examples-native_generate"></a>

#### native_generate

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

<a id="examples-select_huggingface_model"></a>

#### select_huggingface_model

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

<a id="examples-auto_native_generate"></a>

#### auto_native_generate

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

<a id="examples-clm_generate"></a>

#### clm_generate

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

<a id="examples-internet_jepa"></a>

#### internet_jepa

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

<a id="examples-internet_liquid"></a>

#### internet_liquid

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

<a id="examples-graph2text_nativers"></a>

### graph2text_native.rs

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

<a id="examples-g-retriever-python-implementation-and-examples"></a>

### G-Retriever Python implementation and examples

These listings adapt the [MIT-licensed upstream G-Retriever](../third_party/g_retriever/LICENSE). See the [integration guide](#retriever) for installation, pretrained models and behavior differences.

<a id="examples-corepy"></a>

#### core.py

<!-- source: ../examples/python/g_retriever/core.py -->
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

<!-- source: ../examples/python/g_retriever/demo.py -->
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

<!-- source: ../examples/python/g_retriever_retrieve.py -->
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

<!-- source: ../examples/python/g_retriever_train.py -->
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

<!-- source: ../examples/python/g_retriever_generate.py -->
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

<!-- source: ../examples/graph2text_finetune.rs -->
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

<!-- source: ../examples/transformers_pipeline.rs -->
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

### Arm Acceleration

<!-- source: ../examples/arm_acceleration.rs -->
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

### Arm Npu Generate

<!-- source: ../examples/arm_npu_generate.rs -->
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

### Arm Npu Host

<!-- source: ../examples/arm_npu_host.rs -->
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

### AI winter example: XOR training and inference

Follow the [step-by-step XOR tutorial](../README.md#ai-winter-example-learn-xor-then-run-inference) to train, save and reload this CPU network.

<!-- source: ../examples/ai_winter.rs -->
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
