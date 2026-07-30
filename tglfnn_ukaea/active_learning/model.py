"""Pure-JAX implementation of the TGLFNN Gaussian MLP ensembles.

Parameters use exactly the same nested-dict layout as the pickled weights
shipped in ``tglfnn_ukaea/weights``::

    params[flux][f"MLP_{i}"][f"FullyConnectedLayer_{j}"]["weight"]  # (out, in)
    params[flux][f"MLP_{i}"][f"FullyConnectedLayer_{j}"]["bias"]    # (out,)

so pretrained checkpoints can be fine-tuned and written back losslessly.

Each network maps normalised inputs to a (mean, variance) pair of a
normalised flux, with ``variance = softplus(raw)``, matching the original
architecture (hidden layers: Dense -> Dropout -> ReLU).
"""

import dataclasses
from typing import Any, Mapping, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

Params = Any  # Nested dict pytree of arrays.

_VAR_FLOOR = 1e-6


@dataclasses.dataclass(frozen=True)
class Normalizer:
    """Z-score normalisation constants taken from the checkpoint ``stats``."""

    input_mean: jax.Array
    input_std: jax.Array
    output_mean: Mapping[str, float]
    output_std: Mapping[str, float]

    @classmethod
    def from_stats(cls, stats, input_labels, output_labels) -> "Normalizer":
        return cls(
            input_mean=jnp.array([stats[l]["mean"] for l in input_labels]),
            input_std=jnp.array([stats[l]["std"] for l in input_labels]),
            output_mean={l: stats[l]["mean"] for l in output_labels},
            output_std={l: stats[l]["std"] for l in output_labels},
        )

    def normalize_inputs(self, x: jax.Array) -> jax.Array:
        return (x - self.input_mean) / self.input_std

    def normalize_output(self, y: jax.Array, label: str) -> jax.Array:
        return (y - self.output_mean[label]) / self.output_std[label]

    def unnormalize_output(self, y: jax.Array, label: str) -> jax.Array:
        return y * self.output_std[label] + self.output_mean[label]


def mlp_apply(
    params: Params,
    x: jax.Array,
    *,
    dropout_rate: float = 0.0,
    key: jax.Array | None = None,
) -> Tuple[jax.Array, jax.Array]:
    """Forward pass of a single Gaussian MLP.

    Args:
        params: ``{"FullyConnectedLayer_{j}": {"weight", "bias"}}``.
        x: Normalised inputs of shape ``(..., n_inputs)``.
        dropout_rate: Dropout applied after each hidden linear layer
            (only if ``key`` is provided, i.e. during training).
        key: PRNG key for dropout; ``None`` disables dropout (inference).

    Returns:
        ``(mean, variance)`` of the normalised output, each ``(...)``.
    """
    n_layers = len(params)
    for j in range(n_layers - 1):
        layer = params[f"FullyConnectedLayer_{j}"]
        x = x @ layer["weight"].T + layer["bias"]
        if key is not None and dropout_rate > 0.0:
            key, subkey = jax.random.split(key)
            keep = jax.random.bernoulli(subkey, 1.0 - dropout_rate, x.shape)
            x = jnp.where(keep, x / (1.0 - dropout_rate), 0.0)
        x = jax.nn.relu(x)
    layer = params[f"FullyConnectedLayer_{n_layers - 1}"]
    out = x @ layer["weight"].T + layer["bias"]
    return out[..., 0], jax.nn.softplus(out[..., 1])


def stack_ensemble(flux_params: Mapping[str, Params]) -> Params:
    """``{"MLP_i": tree}`` -> single tree with a leading ensemble axis."""
    members = [flux_params[f"MLP_{i}"] for i in range(len(flux_params))]
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *members)


def unstack_ensemble(stacked: Params) -> Mapping[str, Params]:
    """Inverse of :func:`stack_ensemble`, converting leaves to numpy."""
    n_members = jax.tree.leaves(stacked)[0].shape[0]
    return {
        f"MLP_{i}": jax.tree.map(lambda leaf, i=i: np.asarray(leaf[i]), stacked)
        for i in range(n_members)
    }


def init_ensemble(
    key: jax.Array,
    n_members: int,
    n_inputs: int,
    hidden_size: int = 512,
    num_hiddens: int = 6,
) -> Params:
    """He-initialised stacked ensemble parameters (for training from scratch)."""

    def init_member(key: jax.Array) -> Params:
        sizes = [n_inputs] + [hidden_size] * (num_hiddens - 1) + [2]
        params = {}
        for j, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            key, subkey = jax.random.split(key)
            params[f"FullyConnectedLayer_{j}"] = {
                "weight": jax.random.normal(subkey, (n_out, n_in))
                * jnp.sqrt(2.0 / n_in),
                "bias": jnp.zeros(n_out),
            }
        return params

    return jax.vmap(init_member)(jax.random.split(key, n_members))


def predict(
    stacked_params: Params, x_norm: jax.Array
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Ensemble prediction in normalised units.

    Returns:
        ``(mean, aleatoric_variance, epistemic_variance)``, each of shape
        ``x_norm.shape[:-1]``. The mean is the average of the member means,
        the aleatoric variance the average of the member variances, and the
        epistemic variance the spread of the member means (the ensemble
        disagreement used as the active learning acquisition signal).
    """
    means, variances = jax.vmap(lambda p: mlp_apply(p, x_norm))(stacked_params)
    return means.mean(axis=0), variances.mean(axis=0), means.var(axis=0)


def gaussian_nll(
    params: Params,
    x: jax.Array,
    y: jax.Array,
    key: jax.Array,
    dropout_rate: float,
) -> jax.Array:
    """Mean Gaussian negative log-likelihood of one member on a batch."""
    mean, var = mlp_apply(params, x, dropout_rate=dropout_rate, key=key)
    var = var + _VAR_FLOOR
    return 0.5 * jnp.mean(jnp.log(var) + (y - mean) ** 2 / var)


def train_ensemble(
    stacked_params: Params,
    x: jax.Array,
    y: jax.Array,
    key: jax.Array,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    dropout_rate: float = 0.0,
    replay: Tuple[jax.Array, jax.Array] | None = None,
    replay_weight: float = 1.0,
) -> Tuple[Params, float]:
    """Trains all ensemble members of one flux with Adam on the Gaussian NLL.

    Each member draws its own bootstrap minibatches (and dropout masks),
    which maintains ensemble diversity during fine-tuning.

    Args:
        stacked_params: Ensemble parameters with a leading member axis.
        x: Normalised inputs, shape ``(n_samples, n_inputs)``.
        y: Normalised targets, shape ``(n_samples,)``.
        key: PRNG key.
        epochs: Number of passes over the dataset (in expectation).
        batch_size: Minibatch size per member.
        learning_rate: Adam learning rate.
        dropout_rate: Hidden-layer dropout rate during training.
        replay: Optional distillation-replay data ``(x_replay, y_replay)``
            with shapes ``(n_replay, n_inputs)`` and *per-member* targets
            ``(n_members, n_replay)`` (normalised units) — typically each
            member's own frozen pretrained prediction. Anchors the model
            outside the acquired data without collapsing member diversity.
        replay_weight: Weight of the replay NLL relative to the data NLL.

    Returns:
        ``(trained_params, final_mean_loss)``.
    """
    n_samples = x.shape[0]
    n_members = jax.tree.leaves(stacked_params)[0].shape[0]
    batch_size = min(batch_size, n_samples)
    optimizer = optax.adam(learning_rate)
    opt_state = jax.vmap(optimizer.init)(stacked_params)

    if replay is not None:
        x_replay, y_replay = replay
        n_replay = x_replay.shape[0]
        replay_batch_size = min(batch_size, n_replay)
    else:
        # Dummy per-member targets so member_step keeps a fixed signature.
        y_replay = jnp.zeros((n_members, 1))

    @jax.jit
    def step(params, opt_state, key):
        def member_step(params, opt_state, key, y_replay_member):
            batch_key, dropout_key, replay_key, replay_dropout_key = (
                jax.random.split(key, 4)
            )

            def loss_fn(params):
                idx = jax.random.randint(
                    batch_key, (batch_size,), 0, n_samples
                )
                loss = gaussian_nll(
                    params, x[idx], y[idx], dropout_key, dropout_rate
                )
                if replay is not None:
                    replay_idx = jax.random.randint(
                        replay_key, (replay_batch_size,), 0, n_replay
                    )
                    loss += replay_weight * gaussian_nll(
                        params,
                        x_replay[replay_idx],
                        y_replay_member[replay_idx],
                        replay_dropout_key,
                        dropout_rate,
                    )
                return loss

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss

        keys = jax.random.split(key, n_members)
        return jax.vmap(member_step)(params, opt_state, keys, y_replay)

    n_steps = epochs * max(1, n_samples // batch_size)
    losses = jnp.zeros(n_members)
    for _ in range(n_steps):
        key, subkey = jax.random.split(key)
        stacked_params, opt_state, losses = step(stacked_params, opt_state, subkey)
    return stacked_params, float(losses.mean())
