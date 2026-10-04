# Python Keras comparison

[Back to the handbook](handbook.md) · [Complete Rust programs](code_examples.md)

This chapter translates Lighter concepts to **standalone Keras 3** with a JAX CPU
backend. The examples use `import keras`, not `tensorflow.keras`. A familiar name
does not imply matching numerical behavior: shape conventions, initialization,
optimizer state, recurrence, and serialization differ.

## Run the complete comparison

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

The Rust programs are complete listings in the [code cookbook](code_examples.md).
The full Python program appears at the end of this chapter and is also available
as [keras_comparison.py](../examples/python/keras_comparison.py).

## API and behavior mapping

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
| HTTP model hosting | `lighter-serve`: OpenAI-compatible completion/chat/SSE API | No core-Keras model server equivalent | [Hosting guide](model_host.md); CPU reference execution, not vLLM GPU throughput. |
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

## Regression: translate an entire workflow

Python uses `[batch, features]`; the Rust wrapper uses `[samples, time=1, features]`.
The dataset below expresses `target = first_feature + second_feature`. Both
programs train a 2→4→1 network using MSE and SGD, then check output shapes. They do
not promise identical weights or loss trajectories.

### Rust

The full `main` is in [handbook_keras](code_examples.md#handbook_keras). Its core:

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

### Python Keras

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

## Convolution and pooling: compare actual values

The complete layer programs verify an all-ones length-two convolution kernel on
`[1,2,3,4]` yields `[3,5,7]`. Rust input is `[1,1,4]`; Python is `[1,4,1]`.
For two-dimensional convolution, Rust kernel shape is
`[output_channels, input_channels, height, width]`; Keras kernel shape is
`[height, width, input_channels, output_channels]`. For a transfer, transpose the
kernel and input axes, then compare results explicitly.

A 2×2 patch `[1,2;3,4]` averages to `2.5` and max-pools to `4`. The Rust tour
asserts its wrapper's current reversed enum mapping; the Python tour asserts
Keras' conventional mapping. These assertions deliberately reveal the difference.

## Normalization, activation, and flatten semantics

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

## Recurrence, embeddings, and attention

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

## Autoencoders, branches, and mixtures

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

## Training, LoRA, reward modeling, and objectives

The Python script demonstrates:

- Built-in Dense LoRA with `lora_rank=1`; dropout/L2/AdamW controls.
- Sparse-label classification with logits and label-smoothed categorical CE.
- Terminal-aware GAE, a clipped PPO **policy term**, DPO loss, and temperature-scaled teacher KL.
- A shared scalar reward head trained on a preferred/rejected pair.

The Rust [handbook_training](code_examples.md#handbook_training) additionally
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

## Persistence and portability

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

## JEPA and continuous-time references

The Python JEPA-like example excludes a target image patch, predicts its target
encoder representation from visible patches, and updates target weights by EMA.
It is a small training demonstration, not an exact port of Lighter's geometry-aware
predictor or a complete I-JEPA/V-JEPA research implementation. Full Rust image and
video execution/training examples are in the cookbook.

The Python continuous-time section runs a scalar RK4 decay solver and an elapsed-time
gated recurrence. RK4 validates `y(1) ≈ exp(-1)`. The recurrence is a custom reference,
not a Keras-provided CfC and not equivalent to Lighter's recurrent parameterization.
For Euler, Heun, RK4, LiquidCell, CfC, and stacked LFM, use the complete Rust listings.

## Complete executable Python listing

The program below is the full contents of the linked source file. Every function
is called by `main`; assertions check shapes, finiteness, deterministic layer
values, and persistence. These checks establish runnable examples, not model quality.

```python
"""Executable Keras 3 counterparts for docs/keras_comparison.md.

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


## Graph-to-text

See the [graph-to-text comparison and executable Python realization](graph2text.md#keras-comparison).
