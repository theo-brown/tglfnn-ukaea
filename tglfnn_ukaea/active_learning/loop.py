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
import time
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
    # Candidates only qualify for acquisition if |predicted flux| plus this
    # many predicted standard deviations stays below the flux cuts, so TGLF
    # runs are not spent right at the cut boundary.
    acquisition_margin_sigma: float = 1.0
    # Distillation replay: per round, this many free random points are
    # pseudo-labelled by the *frozen pretrained* ensemble (member-wise) and
    # trained on alongside the TGLF data, anchoring the model against
    # forgetting outside the acquired regions. 0 disables; requires
    # warm_start.
    n_replay: int = 1024
    replay_weight: float = 1.0
    # Scale (gyro-Bohm units) of the asinh loss weighting: samples are
    # weighted by 1 / (1 + (flux / s)^2), the squared Jacobian of
    # asinh(flux / s). This fits the model in asinh space to first order, so
    # few-GB errors near marginal stability (which dominate the behaviour of
    # stiff transport simulations) cost as much as large errors in the
    # strongly driven region. Set to 0 to disable and recover plain NLL.
    asinh_scale_gb: float = 10.0
    # Optional pool of extra candidate inputs (an ``.npz`` file with an
    # ``x`` array of shape ``(n, n_inputs)``), e.g. harvested from the
    # points where an integrated-modelling code actually evaluates the
    # surrogate. Simulation trajectories live on a low-dimensional,
    # near-marginal manifold (and partly outside the training hypercube)
    # that uniform hypercube sampling essentially never hits, so without
    # this neither acquisition nor replay protects the region that
    # determines simulation behaviour. Empty string disables.
    candidate_pool_path: str = ""
    # Fraction of each acquisition batch reserved for the highest-scoring
    # not-yet-acquired pool points (when a pool is given).
    pool_acquisition_fraction: float = 0.5
    # Fraction of the replay points drawn from the pool (when given).
    pool_replay_fraction: float = 0.5
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
    params_by_flux: Mapping[str, model_lib.Params],
    x_norm: jax.Array,
    normalizer: model_lib.Normalizer,
    flux_cutoff_gb: Mapping[str, float],
    margin_sigma: float = 0.0,
) -> jax.Array:
    """Ensemble disagreement, summed over fluxes (in normalised units).

    Candidates whose *predicted* fluxes exceed the training-data cuts, within
    ``margin_sigma`` predicted standard deviations, score ``-inf``: their
    TGLF labels would be discarded by :func:`filter_valid` anyway, so
    labelling them wastes TGLF runs. (Without this mask the loop stalls,
    because disagreement is largest precisely in the super-critical corners
    of the hypercube; without the margin it stalls more slowly, on the
    boundary points whose labels land just above the cuts.)"""
    total = jnp.zeros(x_norm.shape[0])
    within_cuts = jnp.ones(x_norm.shape[0], dtype=bool)
    for label, stacked_params in params_by_flux.items():
        mean_norm, aleatoric_var, epistemic_var = model_lib.predict(
            stacked_params, x_norm
        )
        total += jnp.sqrt(epistemic_var)
        predicted = normalizer.unnormalize_output(mean_norm, label)
        predicted_std = (
            jnp.sqrt(aleatoric_var + epistemic_var)
            * normalizer.output_std[label]
        )
        within_cuts &= (
            jnp.abs(predicted) + margin_sigma * predicted_std
            <= flux_cutoff_gb[label]
        )
    return jnp.where(within_cuts, total, -jnp.inf)


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
    asinh_scale_gb: float = 10.0,
) -> Mapping[str, float]:
    """Validation RMSE (gyro-Bohm and asinh space) and NLL per flux.

    The asinh-space RMSE, ``rmse(asinh(pred/s) - asinh(true/s))``, is
    sensitive to the small-|flux| near-marginal region that the plain RMSE
    (dominated by the largest fluxes) cannot see.
    """
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
        metrics[f"asinh_rmse_{label}"] = float(
            jnp.sqrt(
                jnp.mean(
                    (
                        jnp.arcsinh(mean / asinh_scale_gb)
                        - jnp.arcsinh(y[:, i] / asinh_scale_gb)
                    )
                    ** 2
                )
            )
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

    pool = None
    if config.candidate_pool_path:
        pool = jnp.asarray(np.load(config.candidate_pool_path)["x"])
        pool_available = np.ones(len(pool), dtype=bool)
        print(
            f"Candidate pool: {len(pool)} points from "
            f"{config.candidate_pool_path}"
        )

    key = jax.random.key(config.seed)
    if config.warm_start:
        params_by_flux = {
            label: model_lib.stack_ensemble(model_dict["params"][label])
            for label in output_labels
        }
        # Frozen copy of the pretrained ensembles, used as the distillation
        # replay teacher (stack_ensemble builds fresh arrays, and training
        # never mutates in place, so no explicit copy is needed).
        pretrained_by_flux = {
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
        round_start = time.monotonic()
        # 1-2. Sample candidates and pick the batch to label.
        key, candidate_key, acquisition_key = jax.random.split(key, 3)
        candidates = sample_inputs(
            candidate_key, config.n_candidates, input_labels, param_space
        )
        if config.acquisition == "ensemble_variance":

            def scores_for(x_cand):
                return acquisition_scores(
                    params_by_flux,
                    normalizer.normalize_inputs(x_cand),
                    normalizer,
                    config.flux_cutoff_gb,
                    config.acquisition_margin_sigma,
                )

            n_from_pool = 0
            if pool is not None:
                n_from_pool = min(
                    round(
                        config.acquisition_batch
                        * config.pool_acquisition_fraction
                    ),
                    int(pool_available.sum()),
                )
            batch_indices = jnp.argsort(scores_for(candidates))[
                -(config.acquisition_batch - n_from_pool) :
            ]
            if n_from_pool > 0:
                # Highest-disagreement pool points not yet labelled; once
                # acquired they leave the pool so rounds don't relabel them.
                available = np.flatnonzero(pool_available)
                pool_order = np.asarray(
                    jnp.argsort(scores_for(pool[available]))
                )
                picked = available[pool_order[-n_from_pool:]]
                pool_available[picked] = False
                candidates = jnp.concatenate([candidates, pool[picked]])
                batch_indices = jnp.concatenate(
                    [
                        batch_indices,
                        jnp.arange(len(candidates) - n_from_pool, len(candidates)),
                    ]
                )
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
        oracle_start = time.monotonic()
        y_new = oracle(x_new, input_labels)
        oracle_seconds = time.monotonic() - oracle_start
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
        use_replay = config.n_replay > 0 and config.warm_start
        if use_replay:
            key, replay_key = jax.random.split(key)
            n_pool_replay = 0
            if pool is not None:
                n_pool_replay = min(
                    round(config.n_replay * config.pool_replay_fraction),
                    len(pool),
                )
            x_replay = sample_inputs(
                replay_key,
                config.n_replay - n_pool_replay,
                input_labels,
                param_space,
            )
            if n_pool_replay > 0:
                # Anchor the replay distillation on the pool manifold too —
                # the pretrained teacher is typically accurate there, and
                # hypercube samples alone essentially never cover it.
                key, pool_replay_key = jax.random.split(key)
                pool_idx = jax.random.choice(
                    pool_replay_key,
                    len(pool),
                    (n_pool_replay,),
                    replace=False,
                )
                x_replay = jnp.concatenate([x_replay, pool[pool_idx]])
            x_replay_norm = normalizer.normalize_inputs(x_replay)
        train_losses = {}
        for i, label in enumerate(output_labels):
            y_train_norm = normalizer.normalize_output(
                jnp.asarray(y_data)[train_idx, i], label
            )
            replay = None
            replay_sample_weights = None
            if use_replay:
                # Member-wise pseudo-labels from the frozen pretrained
                # ensemble: member m is distilled towards pretrained member
                # m, preserving the ensemble spread (and hence the
                # acquisition signal) away from the acquired data.
                replay_targets = jax.vmap(
                    lambda p: model_lib.mlp_apply(p, x_replay_norm)[0]
                )(pretrained_by_flux[label])
                replay = (x_replay_norm, replay_targets)
                if config.asinh_scale_gb > 0:
                    replay_gb = normalizer.unnormalize_output(
                        replay_targets, label
                    )
                    replay_sample_weights = 1.0 / (
                        1.0 + (replay_gb / config.asinh_scale_gb) ** 2
                    )
            sample_weights = None
            if config.asinh_scale_gb > 0:
                y_train_gb = jnp.asarray(y_data)[train_idx, i]
                sample_weights = 1.0 / (
                    1.0 + (y_train_gb / config.asinh_scale_gb) ** 2
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
                sample_weights=sample_weights,
                replay=replay,
                replay_weight=config.replay_weight,
                replay_sample_weights=replay_sample_weights,
            )

        # 5. Report and checkpoint.
        metrics = {
            "round": round_index,
            "n_data": len(x_data),
            "n_new": len(x_new),
            "seconds_oracle": oracle_seconds,
            "seconds_round": time.monotonic() - round_start,
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
