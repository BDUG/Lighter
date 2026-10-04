# Running the examples

[Full source cookbook](code_examples.md) · [CPU and Arm NN serving](model_serving.md) · [Handbook](../docs/handbook.md)

Run these commands from the repository root, where `Cargo.toml` lives. Install
[stable Rust and Cargo](https://rustup.rs/) and a native C/C++ compiler first. Cargo downloads build dependencies
on the first run. Examples use the CPU; model inference benefits from `--release`.

Start with a small example that needs no model weights or external data:

```bash
cargo run --no-default-features --example graph2text
```

Cargo uses the filename without `.rs` as the example name. Arguments after `--`
are passed to the program. The commands below select the features each example
needs; ordinary `cargo run --example NAME` enables both `candle` and `native` by
default. `server` is opt-in. Replace `/path/to/hf-model` with your local model directory.

## Self-contained examples

These examples use generated data or small demonstration backends and need no
model downloads or credentials. They print shapes, losses, generated text or other
results; assertions check the demonstrated behavior.

| Example | What it demonstrates | Run command |
| --- | --- | --- |
| [graph2text.rs](graph2text.rs) | Exact graph-to-text realization and fact provenance | `cargo run --no-default-features --example graph2text` |
| [jepa.rs](jepa.rs) | Image/video JEPA forward passes and training | `cargo run --no-default-features --example jepa` |
| [liquid_networks.rs](liquid_networks.rs) | Neural ODE, liquid readout training and irregular time steps | `cargo run --no-default-features --example liquid_networks` |
| [handbook_jepa_liquid.rs](handbook_jepa_liquid.rs) | Custom patch encoder and image/video/liquid recipes | `cargo run --no-default-features --example handbook_jepa_liquid` |
| [native_advanced.rs](native_advanced.rs) | Quantization, paged caches, reinforcement objectives and distillation | `cargo run --no-default-features --features native --example native_advanced` |
| [native_prompt.rs](native_prompt.rs) | Templates, constraints, workflows and tool protocols with a demo executor | `cargo run --no-default-features --features native --example native_prompt` |
| [native_finetune.rs](native_finetune.rs) | Small LoRA, optimizer, supervised loss and reward-head updates | `cargo run --no-default-features --features native --example native_finetune` |
| [handbook_native.rs](handbook_native.rs) | Scheduling, sampling and constrained generation with a tiny backend | `cargo run --no-default-features --features native --example handbook_native` |
| [handbook_training.rs](handbook_training.rs) | Gradient accumulation, optimizer hooks, token losses and cache operations | `cargo run --no-default-features --features native --example handbook_training` |
| [handbook_layers.rs](handbook_layers.rs) | Feature scaling, activations and Candle layer shapes | `cargo run --no-default-features --features candle --example handbook_layers` |
| [handbook_keras.rs](handbook_keras.rs) | Rust training and persistence counterparts to Keras | `cargo run --no-default-features --features candle --example handbook_keras` |
| [handbook_experimental.rs](handbook_experimental.rs) | Experimental APIs and upstream alternatives | `cargo run --no-default-features --features candle --example handbook_experimental` |

`handbook_keras` creates a temporary directory for weight round trips and removes
it on successful completion. The experimental examples illustrate the limitations
explained in the [handbook](../docs/handbook.md#usage).

### Graph files and saved output

`graph2text` accepts an optional input file and optional output file:

```bash
cargo run --no-default-features --example graph2text -- examples/data/graph2text.json
cargo run --no-default-features --example graph2text -- examples/data/graph2text.json /tmp/graph-description.json
```

Without an output path, it prints text. With an output path, it writes JSON containing
`text` and the content `plan`, replacing any existing file at that path. See the
[input fixture](data/graph2text.json) and [graph-to-text guide](../docs/handbook.md#graph)
for graph validation, neighborhoods and fact budgets.

## Examples requiring a local model

Provide a compatible Hugging Face snapshot containing `config.json`,
`tokenizer.json` and supported single-file or sharded `.safetensors` weights.
These commands load existing files; they do not download a model. Model memory
requirements depend on the checkpoint size. See the [native runtime guide](../docs/handbook.md#usage-native-generation-and-model-loading)
for supported architectures and artifact loading.

```bash
# Generate a completion from a local model.
cargo run --release --no-default-features --features native --example native_generate -- /path/to/hf-model "Explain Rust ownership:"

# Perform a real LM-head LoRA training step instead of the small default demo.
cargo run --release --no-default-features --features native --example native_finetune -- /path/to/hf-model

# Realize graph facts with a native language model.
cargo run --release --no-default-features --features native --example graph2text_native -- /path/to/hf-model examples/data/graph2text.json
```

Source files: [native_generate.rs](native_generate.rs),
[native_finetune.rs](native_finetune.rs), [graph2text_native.rs](graph2text_native.rs).
Language-model graph descriptions are probabilistic; use `graph2text` for exact
selected-fact coverage.

## Examples that access the internet

| Example | External input and behavior | Run command |
| --- | --- | --- |
| [internet_jepa.rs](internet_jepa.rs) | Downloads the Rust logo for an image/video JEPA demonstration | `cargo run --no-default-features --features native --example internet_jepa` |
| [internet_liquid.rs](internet_liquid.rs) | Downloads daily minimum temperatures for a forecasting demonstration | `cargo run --no-default-features --features native --example internet_liquid` |
| [select_huggingface_model.rs](select_huggingface_model.rs) | Searches Hugging Face, selects a compatible model and asks whether to download it | `cargo run --no-default-features --features native --example select_huggingface_model -- TinyLlama` |
| [auto_native_generate.rs](auto_native_generate.rs) | Searches, downloads and runs a compatible checkpoint automatically | `cargo run --release --no-default-features --features native --example auto_native_generate -- TinyLlama "Explain why Rust is safe:"` |
| [clm_generate.rs](clm_generate.rs) | Downloads and runs `Contrastive-LM/CLM-v0.1-8B` unless a local directory is supplied | `cargo run --release --no-default-features --features native --example clm_generate -- "Explain contrastive language modeling."` |

The dataset demonstrations need network access but no Hugging Face account.
Model downloads use the Hugging Face cache and may be large. `HF_TOKEN` supplies
credentials for Hub access where needed; gated models also require access granted
by their publisher. Model selection depends on the available Hub results and your
system's memory. Automatic generation initially loads the model before quantizing it.

For CLM, use a local snapshot and optional load-time quantization:

```bash
CLM_MODEL_DIR=/path/to/clm-model CLM_QUANTIZATION=int4 \
  cargo run --release --no-default-features --features native --example clm_generate -- "Explain contrastive language modeling."
```

`CLM_QUANTIZATION` accepts `int4` or `int8`; leave it unset for the ordinary loader.
See the [CLM guide](../docs/handbook.md#clm) for checkpoint details.

## Host a model over HTTP

[serve_model.rs](serve_model.rs) embeds the host and serves a local snapshot under
the name `local-model` at `127.0.0.1:8000`. It runs until stopped with Ctrl+C.

```bash
cargo run --release --no-default-features --features server --example serve_model -- /path/to/hf-model
```

In another terminal:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-model","prompt":"Explain Rust ownership:","max_tokens":64}'
```

If `LIGHTER_API_KEY` is set when starting the host, add
`-H "Authorization: Bearer $LIGHTER_API_KEY"` to completion requests using the same
key in the client terminal. This example uses plain completion prompts. To configure
a chat template, quantization or limits, use the CLI instead:

```bash
cargo run --release --no-default-features --features server --bin lighter-serve -- \
  --model /path/to/hf-model --chat-template llama3
```

Match the template to the model's training format. The CLI defaults to model name
`lighter`. See the [model host guide](../docs/handbook.md#host) for chat, streaming,
authentication and the graph generation endpoint.

## Python Keras comparison

[python/keras_comparison.py](python/keras_comparison.py) runs the Keras 3 counterparts
on generated data. Install Python 3 and dependencies in a virtual environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install keras jax jaxlib numpy
KERAS_BACKEND=jax python examples/python/keras_comparison.py
```

The program prints shapes and results, checks numerical values and uses temporary
files for persistence. It needs no downloaded models. The Rust counterpart is
`handbook_keras`; the [Keras comparison](../docs/handbook.md#keras) explains API
mappings and behavior differences.

## More documentation

- [Usage handbook and capability table](../docs/handbook.md#usage)
- [Complete example source listings](../docs/handbook.md#examples)
- [Graph-to-text guide](../docs/handbook.md#graph)

The older `simple_*` demonstrations in `lib/examples/` are library functions,
not standalone Cargo example targets. Follow the handbook for their invocation
and prerequisites.

## G-Retriever examples

The [G-Retriever guide](../docs/handbook.md#retriever) includes installation and pretrained-model
instructions. Use its separate PyTorch environment for these runnable Python examples:

```bash
python3 -m venv .venv-g-retriever
. .venv-g-retriever/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r examples/python/g_retriever/requirements.txt

python examples/python/g_retriever_retrieve.py
python examples/python/g_retriever_train.py --output /tmp/g-retriever-demo.pt
python examples/python/g_retriever_generate.py --adapter /tmp/g-retriever-demo.pt
python -m unittest discover -s examples/python/g_retriever -v
```

The defaults run actual PCST retrieval and train a graph soft prompt against a tiny
random language model without downloading weights. They demonstrate the pipeline;
meaningful QA requires a pretrained model, semantic embeddings and a trained adapter.
Source: [retrieval](python/g_retriever_retrieve.py), [training](python/g_retriever_train.py),
[generation](python/g_retriever_generate.py). The original MIT-licensed implementation
is preserved in [third_party/g_retriever](../third_party/g_retriever/README.md).

## Native Graph2Text instruction fine-tuning

[graph2text_finetune.rs](graph2text_finetune.rs) uses Lighter for answer-only supervised training, shuffled triples, native output-head LoRA, validation and adapter saving/reloading. A compatible local model is required. See the [complete workflow and QLoRA differences](../docs/handbook.md#graph-finetuning).

```bash
cargo run --release --no-default-features --features native --example graph2text_finetune -- \
  /path/to/hf-model examples/data/graph2text_train.json \
  examples/data/graph2text_validation.json /tmp/graph2text-adapter.json 3 none

# Reload and evaluate; use the same base and quantization setting.
cargo run --release --no-default-features --features native --example graph2text_finetune -- \
  /path/to/hf-model examples/data/graph2text_train.json \
  examples/data/graph2text_validation.json /tmp/graph2text-adapter.json 0 none
```

The optional final argument accepts `none`, `int4` or `int8`. Training refuses to replace an existing adapter file. This trains output-head LoRA on a frozen native CPU model; it is not the full GPU NF4/QLoRA recipe. The small fixtures demonstrate the workflow rather than training a general graph verbalizer.

## Native Transformers-style pipeline

[transformers_pipeline.rs](transformers_pipeline.rs) demonstrates independent config/tokenizer loading and reusable batched causal generation. A compatible local snapshot or Hub model ID is required:

```bash
cargo run --release --no-default-features --features native --example transformers_pipeline -- \
  /path/to/hf-model "Explain Rust ownership:" "Explain graph retrieval:"
```

A Hub ID downloads model weights. `HF_TOKEN` supplies optional credentials. The [native parity chapter](../docs/handbook.md#transformers-parity) describes supported generation settings, padding/truncation APIs, architecture limits and the remaining implementation stages.

## ARM CPU acceleration and Arm NN / TFLite delegates

```bash
# Works on every host; unavailable explicit ARM kernels are reported.
cargo run --no-default-features --features native --example arm_acceleration
# Opt-in SVE/SME on AArch64 Linux, with runtime capability checks.
cargo run --release --no-default-features --features arm-sve,arm-sme --example arm_acceleration
# Compiled model, matching tokenizer and vendor runtime/delegate are required.
cargo run --release --no-default-features --features arm-npu --example arm_npu_generate -- \
  arm-npu.json /opt/models/tokenizer.json 2 "Explain graph retrieval:"
# EOS ID=2 and compiled context=512 are examples; use your model's actual values.
cargo run --release --no-default-features --features arm-npu,server --example arm_npu_host -- \
  arm-npu.json /opt/models/tokenizer.json 2 512 127.0.0.1:8080
```

See the [ARM backend guide](../docs/handbook.md#arm-backends) for the JSON configuration,
compiled tensor contract, delegate dependencies, fallback policy, HTTP request example
and hardware validation limits. The CPU example needs no external SDK. The NPU examples
load trusted vendor shared libraries and a precompiled `.tflite` causal model; safetensors
are not automatically converted. Set `LIGHTER_API_KEY` for authenticated hosting.

### Candle tensor bridge

```bash
cargo run --no-default-features --features candle,native --example candle_arm_bridge
```

[candle_arm_bridge.rs](candle_arm_bridge.rs) copies CPU F32 Candle tensors into
row-major buffers, calls the ARM matrix kernel and reconstructs a Candle tensor.
It compares the result with Candle's own `matmul`. This inference bridge detaches
Candle gradients. See [Candle integration](../docs/handbook.md#candle-arm-integration)
for device transfers, training limits and the NPU boundary.

## AI winter example: XOR training and inference

```bash
cargo run --locked --no-default-features --features candle --example ai_winter -- train xor.safetensors
cargo run --locked --no-default-features --features candle --example ai_winter -- infer xor.safetensors
```

[ai_winter.rs](ai_winter.rs) trains a sigmoid network on all four XOR rows, saves
safetensors and reloads them for inference in a separate process. Both modes check
the truth table. See the [step-by-step README](../README.md#ai-winter-example-learn-xor-then-run-inference).
Training overwrites the supplied checkpoint path.

## Context-parallel attention

```bash
cargo run --locked --no-default-features --example context_parallel_attention
```

[context_parallel_attention.rs](context_parallel_attention.rs) demonstrates exact
attention over sharded KV data using stable online-softmax reduction. It is an
in-process CPU reference, with no distributed transport required. See the
[capability reference](../docs/handbook.md#context-parallel-attention) for shapes,
masking, GQA/MQA and the remaining distributed-runtime work.

## Model serving

Follow [model_serving.md](model_serving.md) for CPU Qwen3 and Arm NN delegate
startup, configuration, HTTP requests, authentication and runtime prerequisites.
The [complete source cookbook](code_examples.md) contains the host code listings.
