# Model serving: CPU and Arm NN

Run from a repository checkout with stable Rust and a C/C++ compiler.
The CPU Qwen3 example requires internet for the initial download, about 1.2 GB
of weight storage and at least 4 GB of available RAM. The Arm NN path requires
the compiled artifacts and vendor runtime described below.

Both paths expose the same OpenAI-compatible HTTP API. Run a server in one terminal
and the `curl` requests in another. Stop the server with Ctrl-C.

## Serve Qwen3 on CPU

1. Start the native CPU host from the repository root:

   ```bash
   cargo run --locked --release --no-default-features --features server --bin lighter-serve -- \
     --model Qwen/Qwen3-0.6B \
     --served-model-name qwen3 \
     --bind 127.0.0.1:8000 \
     --chat-template chatml \
     --max-input-tokens 1024 \
     --max-tokens 128
   ```

   The host downloads and caches the model as in the [LLM example](../README.md#llm-example); a local
   snapshot directory can replace the Hugging Face ID. Wait for the message
   `model loaded; listening on ...`. `chatml` explicitly enables chat formatting.

2. Check readiness and discover the hosted model:

   ```bash
   curl http://127.0.0.1:8000/health
   curl http://127.0.0.1:8000/v1/models
   ```

3. Request a chat response:

   ```bash
   curl http://127.0.0.1:8000/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"qwen3","messages":[{"role":"user","content":"Explain Rust ownership briefly. /no_think"}],"max_tokens":64,"temperature":0}'
   ```

   Read the reply in `choices[0].message.content`. Add `"stream":true` to the JSON
   and use `curl -N` for incremental server-sent events. The token limit can truncate
   a reply. Qwen3's `/no_think` instruction requests a response without extended reasoning.

## Serve a compiled model via Arm NN

This path requires a TensorFlow Lite causal model, its matching `tokenizer.json`,
a TensorFlow Lite **C shared library**, and an Arm NN delegate library exporting
the external-plugin interface. Install their matching dependencies and vendor
backend/driver on the target machine. The native CPU host's safetensors are not
converted automatically. See the [ARM backend guide](../docs/handbook.md#arm-backends)
for export requirements and supported tensor shapes.

1. Save a configuration as `arm-npu.json`, replacing the paths, input indices,
   padding ID and backend ID with those of your deployment:

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
     "pad_token_id": 0
   }
   ```

   `VendorNpuBackend` is a placeholder for your registered vendor backend ID.
   To exercise Arm NN on the CPU instead, use `CpuAcc`. Arm NN alone does not
   supply a vendor NPU driver. With fallback disabled, delegate initialization
   failures stop startup; successful attachment still permits CPU graph partitions.

2. Start the delegate host:

   ```bash
   cargo run --locked --release --no-default-features --features arm-npu,server \
     --example arm_npu_host -- \
     arm-npu.json /opt/models/tokenizer.json 2 512 127.0.0.1:8001
   ```

   Here `2` is the EOS token ID and `512` is the compiled context length: replace
   both with your model's actual values. The host verifies the context length and
   prints an execution report. It keeps interpreter/delegate handles on the decode
   worker thread. `delegate_active` confirms attachment; vendor profiling establishes
   actual NPU coverage. Physical ARM/NPU execution has not been verified here.

3. Check readiness and request a completion:

   ```bash
   curl http://127.0.0.1:8001/health
   curl http://127.0.0.1:8001/v1/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"arm-npu","prompt":"Explain graph retrieval briefly:","max_tokens":32,"temperature":0}'
   ```

   Read generated text in `choices[0].text`. For a chat-trained checkpoint, pass
   its correctly formatted prompt; this example leaves the chat endpoint disabled.
   Each step evaluates the full prefix in a fixed batch-one graph, so scheduling
   multiple requests does not imply NPU tensor batching.

Set `LIGHTER_API_KEY` before starting either server to require authentication, then
include `-H "Authorization: Bearer $LIGHTER_API_KEY"` on API requests from a shell
with the same variable configured. The examples bind to localhost. See the
[hosting handbook](../docs/handbook.md#host) for limits, streaming and other routes.
