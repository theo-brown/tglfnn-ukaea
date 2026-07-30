"""A simple ensemble-disagreement active learning loop for TGLFNN.

Each round:

1. Sample random candidate points from the surrogate's training hypercube
   (``config["param_space"]`` stored in the pickled checkpoint).
2. Score candidates by the ensemble disagreement (epistemic standard
   deviation, summed over the flux outputs) and pick the top batch.
3. Label the batch with TGLF, launched through TORAX's ``tglf2py`` wrapper.
4. Fine-tune the ensembles on all data acquired so far.
5. Save a checkpoint in the same pickle format as the shipped weights, so it
   can be consumed by ``tglfnn_ukaea.load``-style loading.
"""

import dataclasses
import pathlib
import pickle
from typing import Mapping, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from tglfnn_ukaea import loader
from tglfnn_ukaea.active_learning import model as model_lib
from tglfnn_ukaea.active_learning import oracles

# Inputs whose bounds in ``param_space`` are log10 exponents; they are
# sampled log-uniformly (e.g. XNUE in [1e-5, 5]).
LOG10_SAMPLED_LABELS = ("XNUE",)


@dataclasses.dataclass
class ActiveLearningConfig:
    machine: str = "multimachine"
    n_rounds: int = 5
    n_initial: int = 128  # Random points labelled before the first round.
    n_candidates: int = 8192  # Candidate pool size per round.
    acquisition_batch: int = 32  # Points labelled by TGLF per round.
    acquisition: str = "ensemble_variance"  # Or "random" (baseline).
    epochs_per_round: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-4
    dropout: float = 0.05
    validation_fraction: float = 0.2
    seed: int = 0
    warm_start: bool = True  # Start from the shipped pretrained weights.
    output_dir: str = "active_learning_checkpoints"
    # Labels above these gyro-Bohm magnitudes are discarded, matching the
    # cuts applied when the original training data was generated.
    flux_cutoff_gb: Mapping[str, float] = dataclasses.field(
        default_factory=lambda: {"efe_gb": 200.0, "efi_gb": 200.0, "pfi_gb": 100.0}
    )


def sample_inputs(
    key: jax.Array,
    n: int,
    input_labels: Sequence[str],
    param_space: Mapping[str, Sequence[float]],
) -> jax.Array:
    """Samples ``n`` points uniformly from the training hypercube."""
    low = jnp.array([param_space[label][0] for label in input_labels])
    high = jnp.array([param_space[label][1] for label in input_labels])
    x = jax.random.uniform(
        key, (n, len(input_labels)), minval=low, maxval=high
    )
    log10_mask = jnp.array(
        [label in LOG10_SAMPLED_LABELS for label in input_labels]
    )
    return jnp.where(log10_mask, 10.0**x, x)


def acquisition_scores(
    params_by_flux: Mapping[str, model_lib.Params], x_norm: jax.Array
) -> jax.Array:
    """Ensemble disagreement, summed over fluxes (in normalised units)."""
    total = jnp.zeros(x_norm.shape[0])
    for stacked_params in params_by_flux.values():
        _, _, epistemic_var = model_lib.predict(stacked_params, x_norm)
        total += jnp.sqrt(epistemic_var)
    return total


def filter_valid(
    x: np.ndarray,
    y: np.ndarray,
    output_labels: Sequence[str],
    flux_cutoff_gb: Mapping[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Drops failed TGLF runs and fluxes beyond the training-data cuts."""
    cutoffs = np.array([flux_cutoff_gb[label] for label in output_labels])
    valid = np.all(np.isfinite(y), axis=-1) & np.all(
        np.abs(y) <= cutoffs, axis=-1
    )
    return x[valid], y[valid]


def evaluate(
    params_by_flux: Mapping[str, model_lib.Params],
    normalizer: model_lib.Normalizer,
    x: jax.Array,
    y: np.ndarray,
    output_labels: Sequence[str],
) -> Mapping[str, float]:
    """Validation RMSE (gyro-Bohm units) and NLL (normalised) per flux."""
    x_norm = normalizer.normalize_inputs(x)
    metrics = {}
    for i, label in enumerate(output_labels):
        mean_norm, aleatoric, epistemic = model_lib.predict(
            params_by_flux[label], x_norm
        )
        mean = normalizer.unnormalize_output(mean_norm, label)
        y_norm = normalizer.normalize_output(jnp.asarray(y[:, i]), label)
        var = aleatoric + epistemic + model_lib._VAR_FLOOR
        metrics[f"rmse_{label}"] = float(
            jnp.sqrt(jnp.mean((mean - y[:, i]) ** 2))
        )
        metrics[f"nll_{label}"] = float(
            0.5 * jnp.mean(jnp.log(var) + (y_norm - mean_norm) ** 2 / var)
        )
    return metrics


def save_checkpoint(
    path: pathlib.Path,
    model_dict: Mapping,
    params_by_flux: Mapping[str, model_lib.Params],
    round_index: int,
) -> None:
    """Writes a checkpoint in the same pickle format as the shipped weights."""
    checkpoint = {
        "stats": model_dict["stats"],
        "config": {**model_dict["config"], "active_learning_round": round_index},
        "input_labels": model_dict["input_labels"],
        "params": {
            label: model_lib.unstack_ensemble(stacked)
            for label, stacked in params_by_flux.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)


def run_active_learning(
    config: ActiveLearningConfig = ActiveLearningConfig(),
    oracle: oracles.Oracle = oracles.tglf_oracle,
) -> Mapping:
    """Runs the active learning loop and returns the final state.

    Args:
        config: Loop settings.
        oracle: Labelling function ``(x, input_labels) -> fluxes``; defaults
            to running TGLF via the TORAX wrapper. Pass
            ``oracles.mock_oracle`` for a cheap dry run.

    Returns:
        Dict with the trained ``params`` (stacked, by flux), the acquired
        dataset ``x``/``y``, and the per-round metrics ``history``.
    """
    model_dict = loader.load(config.machine)
    input_labels = tuple(model_dict["input_labels"])
    output_labels = tuple(model_dict["params"].keys())
    param_space = model_dict["config"]["param_space"]
    normalizer = model_lib.Normalizer.from_stats(
        model_dict["stats"], input_labels, output_labels
    )
    output_dir = pathlib.Path(config.output_dir)

    key = jax.random.key(config.seed)
    if config.warm_start:
        params_by_flux = {
            label: model_lib.stack_ensemble(model_dict["params"][label])
            for label in output_labels
        }
    else:
        n_members = model_dict["config"].get("num_estimators", 5)
        params_by_flux = {}
        for label in output_labels:
            key, subkey = jax.random.split(key)
            params_by_flux[label] = model_lib.init_ensemble(
                subkey, n_members, len(input_labels)
            )

    # Initial dataset: random points from the hypercube, labelled by TGLF.
    key, subkey = jax.random.split(key)
    x_data = np.asarray(
        sample_inputs(subkey, config.n_initial, input_labels, param_space)
    )
    y_data = oracle(x_data, input_labels)
    x_data, y_data = filter_valid(
        x_data, y_data, output_labels, config.flux_cutoff_gb
    )
    print(f"Initial dataset: {len(x_data)} valid points")

    history = []
    for round_index in range(config.n_rounds):
        # 1-2. Sample candidates and pick the batch to label.
        key, candidate_key, acquisition_key = jax.random.split(key, 3)
        candidates = sample_inputs(
            candidate_key, config.n_candidates, input_labels, param_space
        )
        if config.acquisition == "ensemble_variance":
            scores = acquisition_scores(
                params_by_flux, normalizer.normalize_inputs(candidates)
            )
            batch_indices = jnp.argsort(scores)[-config.acquisition_batch :]
        elif config.acquisition == "random":
            batch_indices = jax.random.choice(
                acquisition_key,
                config.n_candidates,
                (config.acquisition_batch,),
                replace=False,
            )
        else:
            raise ValueError(f"Unknown acquisition: '{config.acquisition}'")
        x_new = np.asarray(candidates[batch_indices])

        # 3. Label with TGLF.
        y_new = oracle(x_new, input_labels)
        x_new, y_new = filter_valid(
            x_new, y_new, output_labels, config.flux_cutoff_gb
        )
        x_data = np.concatenate([x_data, x_new])
        y_data = np.concatenate([y_data, y_new])

        # 4. Fine-tune on all data acquired so far (with a held-out split).
        key, split_key = jax.random.split(key)
        permutation = jax.random.permutation(split_key, len(x_data))
        n_val = int(len(x_data) * config.validation_fraction)
        val_idx, train_idx = permutation[:n_val], permutation[n_val:]
        x_train = jnp.asarray(x_data)[train_idx]
        x_train_norm = normalizer.normalize_inputs(x_train)
        train_losses = {}
        for i, label in enumerate(output_labels):
            y_train_norm = normalizer.normalize_output(
                jnp.asarray(y_data)[train_idx, i], label
            )
            key, train_key = jax.random.split(key)
            params_by_flux[label], train_losses[label] = model_lib.train_ensemble(
                params_by_flux[label],
                x_train_norm,
                y_train_norm,
                train_key,
                epochs=config.epochs_per_round,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                dropout_rate=config.dropout,
            )

        # 5. Report and checkpoint.
        metrics = {
            "round": round_index,
            "n_data": len(x_data),
            "n_new": len(x_new),
            **{f"train_nll_{k}": v for k, v in train_losses.items()},
        }
        if n_val > 0:
            metrics.update(
                evaluate(
                    params_by_flux,
                    normalizer,
                    jnp.asarray(x_data)[val_idx],
                    y_data[np.asarray(val_idx)],
                    output_labels,
                )
            )
        history.append(metrics)
        print(
            f"Round {round_index}: "
            + ", ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in metrics.items()
                if k != "round"
            )
        )
        save_checkpoint(
            output_dir / f"{config.machine}_round_{round_index}.pkl",
            model_dict,
            params_by_flux,
            round_index,
        )

    return {
        "params": params_by_flux,
        "x": x_data,
        "y": y_data,
        "history": history,
    }
