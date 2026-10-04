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
