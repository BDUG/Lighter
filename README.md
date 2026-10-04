# Rust Lighter

A Keras-inspired Rust machine-learning library built on [Candle](https://github.com/huggingface/candle), including also a Candle-free native implementation. 


## Disclaimer & Project Status

This repository is an independent, non-commercial hobby project developed purely in spare time. 

- **Non-Commercial:** This project does not generate revenue, offer commercial services, or represent any business entity.
- **Maintenance & Support:** It is provided on an "as-is" basis without dedicated support, formal roadmaps, or service-level agreements (SLAs). Updates and bug fixes occur purely at the author's discretion and availability.
- **No Liability:** The author accepts no liability or responsibility for any direct, indirect, incidental, or consequential damages resulting from the use, inability to use, or reliance on this software. Use it entirely at your own risk.

## Handbook 

See the **[Usage handbook](docs/handbook.md#usage)**.

## AI winter example: learn XOR, then run inference

XOR returns `1` when two bits differ and `0` when they match. It illustrates a
classic limitation of single-layer perceptrons: a nonlinear hidden layer is needed
to learn this decision boundary. The wider AI winters had many causes.

1. Install [stable Rust](https://rustup.rs/) and a C/C++ compiler, then get the code:

   ```bash
   git clone https://github.com/BDUG/Lighter.git
   cd Lighter
   ```

2. Open [the complete AI winter example](examples/ai_winter.rs). It uses all four
   XOR rows as training data and builds a `2 → 8 → 1` network with sigmoid activations:

   ```rust
   let hidden = Dense::new(8, 2, Activations::Sigmoid, &device, &vars, "hidden".into());
   let output = Dense::new(1, 8, Activations::Sigmoid, &device, &vars, "output".into());
   // Inputs:  [0,0], [0,1], [1,0], [1,1]
   // Targets:    0,     1,     1,     0
   ```

3. Train on CPU and save the learned weights:

   ```bash
   cargo run --locked --no-default-features --features candle --example ai_winter -- train xor.safetensors
   ```

   The program uses seeded initialization and a persistent AdamW optimizer with
   learning rate `0.05` and zero weight decay. It trains with mean squared error,
   prints progress every 100 steps, and saves when MSE falls below `0.001`.
   It fails if training does not converge within 5,000 steps. The checkpoint path
   is your choice; training replaces an existing file at that path.

4. Run inference in a separate process using the saved weights:

   ```bash
   cargo run --locked --no-default-features --features candle --example ai_winter -- infer xor.safetensors
   ```

   This reconstructs the same network, loads the checkpoint and predicts all four
   inputs without training. Probabilities at or above `0.5` become `1`; lower ones
   become `0`. Exact probabilities can vary slightly, but the expected classes are:

   | Input | Expected XOR |
   | --- | --- |
   | `[0, 0]` | `0` |
   | `[0, 1]` | `1` |
   | `[1, 0]` | `1` |
   | `[1, 1]` | `0` |

   Both modes check every prediction and return an error if one is incorrect.
   This small example demonstrates training, persistence and inference; it needs
   no dataset download, Python environment or accelerator.

## LLM example

Run inference with [Qwen3-0.6B from Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B)
using Lighter's native Rust decoder. This example downloads pretrained weights;
it does not train the model. Use the checkout and toolchain from the example above,
with internet access, roughly 1.2 GB of disk space for the weights and at least
4 GB of available RAM for weights, caches and runtime overhead. Inference runs on CPU.

1. Prepare a Qwen3 chat prompt. The explicit assistant prefix selects a response
   without a thinking preamble; the pipeline accepts formatted text rather than
   automatically applying the model's chat template:

   ```bash
   prompt='<|im_start|>user
   Explain Rust ownership in two short sentences.<|im_end|>
   <|im_start|>assistant
   <think>

   </think>

   '
   ```

2. Download the model and generate a response:

   ```bash
   cargo run --locked --release --no-default-features --features native \
     --example transformers_pipeline -- Qwen/Qwen3-0.6B "$prompt"
   ```

   The [complete inference example](examples/transformers_pipeline.rs) loads
   configuration, tokenizer and safetensors from Hugging Face, then performs
   greedy decoding with a maximum of 64 new tokens. It prints JSON containing
   `generated_text`, token counts and the finish reason. A response can end at
   that token limit. The public model normally needs no token; an existing
   `HF_TOKEN` environment variable is used if configured.

3. Reuse the command with another prompt. Downloaded files are cached. To use a
   separately downloaded local snapshot, replace `Qwen/Qwen3-0.6B` with its
   directory containing `config.json`, `tokenizer.json` and the model safetensors
   (plus the shard index for sharded weights).

For loading options, supported architectures and generation settings, see the
[Transformers compatibility guide](docs/handbook.md#transformers-parity).

Contributions are welcome. Licensed under [MPL-2.0](LICENSE), [MIT](https://opensource.org/licenses/MIT), or [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0), at your option.
