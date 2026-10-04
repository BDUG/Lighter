# Rust Lighter

A Keras-inspired Rust machine-learning library built on [Candle](https://github.com/huggingface/candle), with a Candle-free native implementation. 

Experimental; not production-ready.

From a checkout with stable Rust, try a model-free CPU example:

```bash
cargo run --locked --example native_advanced --no-default-features --features native
```

See the **[Usage handbook](docs/handbook.md)** for the complete capability table, setup, examples, model downloads, and current limitations. For the dedicated 8B runner, see the [CLM guide](docs/contrastive_lm.MD).

Serve models with the [OpenAI-compatible HTTP host](docs/model_host.md).

Contributions are welcome. Licensed under [MPL-2.0](LICENSE), [MIT](https://opensource.org/licenses/MIT), or [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0), at your option.
