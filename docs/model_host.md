# Model hosting with lighter-serve

[Back to the handbook](handbook.md) · [Full code cookbook](code_examples.md)

`lighter-serve` loads one native model and exposes OpenAI-compatible HTTP completion
and chat endpoints. Like a vLLM server, it keeps weights loaded, accepts concurrent
requests, interleaves decoding in a continuous batch, and can stream SSE responses.
Execution uses Lighter's portable **CPU** decoder. This does not provide vLLM's
GPU kernels, tensor/pipeline parallelism, distributed serving, or throughput.

## Start a local snapshot

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

## Start from the Hugging Face Hub

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

## Endpoint table

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

## curl examples

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

## Complete Python OpenAI client

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

## Supported request fields

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

## Limits, streaming, and failure behavior

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

## Embed the host in Rust

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

## Validation

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

## Graph-to-text extension

`POST /v1/graph/completions` generates from a validated property graph using the same runtime and admission controls. See [graph inputs, examples and response semantics](graph2text.md#hosted-graph-generation).
