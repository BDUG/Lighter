# Running Contrastive-LM CLM v0.1 8B

Lighter provides a native, Candle-free runner for
[`Contrastive-LM/CLM-v0.1-8B`](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B).
It supports Hugging Face downloads, offline snapshots, F16/BF16 weights,
optional Int8 or packed Int4 quantization, and repeated generation without
reloading the model.

## Requirements

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

## Quick start

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

## Offline snapshots

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

## Quantization and CPU threads

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

## Library API

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

## Troubleshooting

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

