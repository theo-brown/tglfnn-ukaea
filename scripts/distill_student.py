"""Distil a TGLFNN-ukaea deep ensemble into a single small student network.

The teacher is one of the released deep-ensemble checkpoints (5 Gaussian MLPs
of 5x512 hidden layers per flux, trained with NLL). The student is a single,
smaller GaussianMLP per flux (default: 4x256 hidden layers, tanh activation)
trained to reproduce the teacher's ensemble mean and total variance
(aleatoric + epistemic) on inputs sampled from the training hypercube
recorded in the teacher checkpoint.

The student is written out in the same pickle schema as the released weights,
with ``num_estimators: 1`` and the reduced ``model_size``/``hidden_size``
recorded in its config. Because ``fusion_surrogates``
(https://github.com/google-deepmind/fusion_surrogates) constructs its
``GaussianMLPEnsemble`` from exactly those config entries, the student loads
through the existing ``TGLFNNukaeaModel`` inference path without any code
changes, at roughly 26x fewer FLOPs per evaluation than the teacher.

The student architecture itself is fusion_surrogates'
``networks.GaussianMLPEnsemble`` (with ``n_ensemble=1``), so training and
inference use literally the same Flax module.

Requires: fusion_surrogates (pip install fusion_surrogates), which brings in
jax, flax and optax.

Example:
    python scripts/distill_student.py --machine multimachine \
        --n-train 1000000 --steps 4000 --batch-size 4096
"""

import argparse
import functools
import json
import pathlib
import pickle
import time
from typing import Any, Mapping

from fusion_surrogates.common import networks
import jax
import jax.numpy as jnp
import numpy as np
import optax

import tglfnn_ukaea


def build_teacher(model_dict: Mapping[str, Any]):
    """Builds the teacher network and stacked params.

    Mirrors the parameter conversion in
    fusion_surrogates.tglfnn_ukaea.tglfnn_ukaea_model.TGLFNNukaeaModel:
    "MLP_{i}" -> "GaussianMLP_{i}", "FullyConnectedLayer_{j}" -> "Dense_{j}",
    "weight" -> transposed "kernel", stacked over fluxes for vmapping.
    """
    config = model_dict["config"]
    network = networks.GaussianMLPEnsemble(
        n_ensemble=config.get("num_estimators", 5),
        num_hiddens=config.get("model_size", 6),
        hidden_size=config.get("hidden_size", 512),
        dropout=config.get("dropout", 0.0),
        activation=config.get("activation", "relu"),
    )
    output_labels = tuple(model_dict["params"].keys())
    params = {}
    for output_label in output_labels:
        ensemble = {}
        for i in range(network.n_ensemble):
            network_params = {}
            for j in range(network.num_hiddens):
                original = model_dict["params"][output_label][f"MLP_{i}"][
                    f"FullyConnectedLayer_{j}"
                ]
                network_params[f"Dense_{j}"] = {
                    "bias": jnp.array(original["bias"].T),
                    "kernel": jnp.array(original["weight"].T),
                }
            ensemble[f"GaussianMLP_{i}"] = network_params
        params[output_label] = ensemble
    stacked = jax.tree.map(
        lambda *args: jnp.stack(args),
        *[params[label] for label in output_labels],
    )
    return network, stacked, output_labels


def input_stats(model_dict: Mapping[str, Any]):
    """Input mean/std vectors, in the same order the inference code uses."""
    input_labels = model_dict["input_labels"]
    means = jnp.array(
        [v["mean"] for k, v in model_dict["stats"].items() if k in input_labels]
    )
    stds = jnp.array(
        [v["std"] for k, v in model_dict["stats"].items() if k in input_labels]
    )
    return means, stds


# Inputs whose param_space bounds are recorded in log10 but which are fed to
# the network in linear units (sampled log-uniformly during training).
_LOG10_SAMPLED_INPUTS = ("XNUE", "BETAE")


def sample_inputs(
    model_dict: Mapping[str, Any],
    n: int,
    rng: np.random.Generator,
    margin: float = 0.0,
) -> np.ndarray:
    """Uniformly samples the training hypercube recorded in the checkpoint.

    With ``margin > 0``, each dimension's bounds are extended by that
    fraction of its range on both sides (in log10 space for log-sampled
    inputs), so the sample covers a neighbourhood of the training box.
    """
    param_space = model_dict["config"]["param_space"]
    columns = []
    for label in model_dict["input_labels"]:
        bounds = param_space[label]
        lo, hi = float(bounds[0]), float(bounds[1])
        pad = margin * (hi - lo)
        values = rng.uniform(lo - pad, hi + pad, size=n)
        if label in _LOG10_SAMPLED_INPUTS:
            values = 10.0**values
        columns.append(values)
    return np.stack(columns, axis=-1).astype(np.float32)


def teacher_labels(network, stacked_params, normalized_inputs, batch_size=65536):
    """Teacher [mean, total variance] per flux, in normalized output space."""

    @jax.jit
    def forward(x):
        return jax.vmap(
            lambda params: network.apply(
                {"params": params}, x, deterministic=True
            )
        )(stacked_params)

    outputs = []
    for start in range(0, normalized_inputs.shape[0], batch_size):
        outputs.append(
            np.asarray(forward(normalized_inputs[start : start + batch_size]))
        )
    # (n_fluxes, n_samples, 2)
    return np.concatenate(outputs, axis=1)


def r2(target: np.ndarray, prediction: np.ndarray) -> float:
    residual = np.sum((target - prediction) ** 2)
    total = np.sum((target - np.mean(target)) ** 2)
    return float(1.0 - residual / total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", default="multimachine",
                        choices=["multimachine", "step"])
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-hiddens", type=int, default=5,
                        help="Dense layers per network incl. the 2-unit "
                        "output layer (matches 'model_size' in the config); "
                        "5 => 4 hidden layers.")
    parser.add_argument("--activation", default="tanh",
                        choices=["relu", "tanh", "sigmoid"],
                        help="tanh gives a smooth surrogate, which helps "
                        "Newton-type transport solvers converge.")
    parser.add_argument("--n-train", type=int, default=1_000_000)
    parser.add_argument("--n-val", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Peak learning rate.")
    parser.add_argument("--schedule", default="cosine",
                        choices=["cosine", "wsd"],
                        help="LR schedule: 'cosine' decays from --lr over "
                        "the whole run; 'wsd' (warmup-stable-decay) warms up "
                        "over --warmup-steps, holds --lr flat, then decays "
                        "over the final 20%% of steps. The stable phase "
                        "makes plateaus diagnostic of convergence rather "
                        "than of the annealing.")
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="AdamW weight decay. Distillation targets are "
                        "noiseless, so 0 is a reasonable choice.")
    parser.add_argument("--var-loss-weight", type=float, default=0.1)
    parser.add_argument("--loss-weight-q0", type=float, default=0.0,
                        help="If > 0 (in GB units), weight the loss "
                        "per-sample-per-flux by q0^2/(q0^2 + flux_GB^2) "
                        "(plus --loss-weight-floor). This is locally "
                        "equivalent to training in asinh(flux/q0) space: "
                        "relative-error matching above q0, absolute below - "
                        "concentrating accuracy at operating-point flux "
                        "magnitudes without changing the checkpoint's "
                        "linear-output contract. 0 disables weighting.")
    parser.add_argument("--loss-weight-floor", type=float, default=0.05,
                        help="Uniform floor mixed into the loss weights so "
                        "the high-flux tail keeps a minimum gradient "
                        "signal.")
    parser.add_argument("--sign-loss-weight", type=float, default=0.0,
                        help="If > 0, adds a sign-consistency hinge "
                        "relu(-pred_GB * teacher_GB)/q0^2 on the physical "
                        "(unnormalised) fluxes. Sign-crossing fluxes (the "
                        "particle flux pinch/outflow transition) feed the "
                        "D_eff/V_eff decomposition in transport solvers, "
                        "where a wrong sign flips the convection direction; "
                        "plain MSE treats such errors as no worse than "
                        "same-sign errors of equal size. The hinge is "
                        "proportional to both magnitudes, so it vanishes "
                        "near zero flux and never fights threshold noise. "
                        "0 disables the term.")
    parser.add_argument("--threshold-q0", type=float, default=10.0,
                        help="Scale (in GB units) of the threshold-weighted "
                        "minibatch sampling: samples are drawn with "
                        "probability proportional to the per-flux average of "
                        "1/(|flux| + q0), concentrating training on the "
                        "near-threshold region that dominates stiff "
                        "flux-driven transport simulations.")
    parser.add_argument("--uniform-fraction", type=float, default=0.3,
                        help="Fraction of the sampling probability assigned "
                        "uniformly, so the high-flux tail stays anchored. "
                        "1.0 recovers unweighted (uniform) sampling.")
    parser.add_argument("--oob-fraction", type=float, default=0.25,
                        help="Fraction of the training pool drawn from a "
                        "margin-extended hypercube instead of the training "
                        "box, so the student matches the teacher in the "
                        "out-of-box neighbourhood that transport solvers "
                        "routinely query (e.g. high elongation). "
                        "0.0 recovers box-only sampling.")
    parser.add_argument("--oob-margin", type=float, default=0.25,
                        help="Margin for --oob-fraction sampling, as a "
                        "fraction of each dimension's range added to both "
                        "sides of the box.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for data sampling (pool and validation "
                        "sets are identical across runs with the same seed).")
    parser.add_argument("--init-seed", type=int, default=None,
                        help="Seed for network initialisation and minibatch "
                        "draws (defaults to --seed). Varying it with a fixed "
                        "--seed isolates optimisation variance from data "
                        "variance.")
    parser.add_argument("--output", default=None,
                        help="Output pickle path (default: "
                        "tglfnn_ukaea/weights/<machine>_student.pkl)")
    args = parser.parse_args()

    teacher_dict = tglfnn_ukaea.load(args.machine)
    teacher_network, teacher_params, output_labels = build_teacher(teacher_dict)
    in_means, in_stds = input_stats(teacher_dict)
    n_inputs = len(teacher_dict["input_labels"])
    n_fluxes = len(output_labels)

    print(f"Teacher: {args.machine}, fluxes={output_labels}, "
          f"{teacher_network.n_ensemble}x{teacher_network.hidden_size}-wide x "
          f"{teacher_network.num_hiddens}-layer per flux")
    print(f"Student: 1x{args.hidden_size}-wide x {args.num_hiddens}-layer "
          f"({args.activation}) per flux")

    # --- Distillation data ------------------------------------------------
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    n_oob = int(args.n_train * args.oob_fraction)
    x_train = np.concatenate([
        sample_inputs(teacher_dict, args.n_train - n_oob, rng),
        sample_inputs(teacher_dict, n_oob, rng, margin=args.oob_margin),
    ])
    x_val = sample_inputs(teacher_dict, args.n_val, rng)
    z_train = (jnp.asarray(x_train) - in_means) / in_stds
    z_val = (jnp.asarray(x_val) - in_means) / in_stds
    y_train = teacher_labels(teacher_network, teacher_params, z_train)
    y_val = teacher_labels(teacher_network, teacher_params, z_val)
    print(f"Labelled {args.n_train}+{args.n_val} samples with the teacher in "
          f"{time.time() - t0:.1f}s")

    # --- Student setup ----------------------------------------------------
    student_network = networks.GaussianMLPEnsemble(
        n_ensemble=1,
        num_hiddens=args.num_hiddens,
        hidden_size=args.hidden_size,
        dropout=0.0,
        activation=args.activation,
    )
    init_seed = args.init_seed if args.init_seed is not None else args.seed
    init_keys = jax.random.split(jax.random.key(init_seed), n_fluxes)
    dummy = jnp.zeros((1, n_inputs))
    student_params = jax.tree.map(
        lambda *args_: jnp.stack(args_),
        *[
            student_network.init(k, dummy, deterministic=True)["params"]
            for k in init_keys
        ],
    )

    if args.schedule == "wsd":
        decay_steps = max(1, int(0.2 * args.steps))
        stable_steps = max(0, args.steps - args.warmup_steps - decay_steps)
        schedule = optax.join_schedules(
            [
                optax.linear_schedule(0.0, args.lr, args.warmup_steps),
                optax.constant_schedule(args.lr),
                optax.cosine_decay_schedule(args.lr, decay_steps),
            ],
            boundaries=[
                args.warmup_steps,
                args.warmup_steps + stable_steps,
            ],
        )
    else:
        schedule = optax.cosine_decay_schedule(args.lr, args.steps)
    optimizer = optax.adamw(schedule, weight_decay=args.weight_decay)

    z_train_dev = jnp.asarray(z_train)
    y_train_dev = jnp.asarray(y_train)
    var_weight = args.var_loss_weight
    eps = 1e-6

    out_stds_dev = jnp.array(
        [teacher_dict["stats"][label]["std"] for label in output_labels]
    )
    out_means_dev = jnp.array(
        [teacher_dict["stats"][label]["mean"] for label in output_labels]
    )
    flux_gb = jnp.abs(
        y_train_dev[..., 0] * out_stds_dev[:, None] + out_means_dev[:, None]
    )  # (n_fluxes, n_samples)

    # Initialise the variance-head bias to the teacher's mean variance so
    # early training is not spent dragging softplus(0) up to scale.
    mean_var = jnp.mean(y_train_dev[..., 1], axis=1)
    last_layer = f"Dense_{args.num_hiddens - 1}"
    last_bias = student_params["GaussianMLP_0"][last_layer]["bias"]
    student_params["GaussianMLP_0"][last_layer]["bias"] = last_bias.at[
        :, 1
    ].set(jnp.log(jnp.expm1(jnp.maximum(mean_var, eps))))

    opt_state = optimizer.init(student_params)

    # Per-sample-per-flux loss weights (asinh-equivalent error allocation).
    if args.loss_weight_q0 > 0:
        q0 = args.loss_weight_q0
        relative = q0**2 / (q0**2 + flux_gb**2)
        loss_weights = (
            args.loss_weight_floor
            + (1.0 - args.loss_weight_floor) * relative
        )
    else:
        loss_weights = jnp.ones_like(flux_gb)
    loss_weights = loss_weights / jnp.mean(loss_weights)

    # Threshold-weighted minibatch sampling: probability proportional to a
    # mixture of uniform and the per-flux average of 1/(|flux_GB| + q0),
    # implemented as inverse-CDF sampling inside the jitted train step.
    near_threshold_weight = jnp.mean(
        1.0 / (flux_gb + args.threshold_q0), axis=0
    )
    n_pool = z_train_dev.shape[0]
    probabilities = (
        args.uniform_fraction / n_pool
        + (1.0 - args.uniform_fraction)
        * near_threshold_weight
        / jnp.sum(near_threshold_weight)
    )
    sampling_cdf = jnp.cumsum(probabilities)
    sampling_cdf = sampling_cdf / sampling_cdf[-1]

    sign_scale = args.loss_weight_q0 if args.loss_weight_q0 > 0 else 10.0

    def loss_fn(params, z, y, w):
        pred = jax.vmap(
            lambda p: student_network.apply(
                {"params": p}, z, deterministic=True
            )
        )(params)
        mean_loss = jnp.mean(w * (pred[..., 0] - y[..., 0]) ** 2)
        logvar_loss = jnp.mean(
            w * (jnp.log(pred[..., 1] + eps) - jnp.log(y[..., 1] + eps)) ** 2
        )
        # Sign-consistency hinge on physical fluxes: active only when the
        # prediction and the teacher disagree in sign, scaled by both
        # magnitudes (a confident wrong-sign costs more).
        q_pred = pred[..., 0] * out_stds_dev[:, None] + out_means_dev[:, None]
        q_true = y[..., 0] * out_stds_dev[:, None] + out_means_dev[:, None]
        sign_loss = jnp.mean(jax.nn.relu(-q_pred * q_true)) / sign_scale**2
        total = (
            mean_loss
            + var_weight * logvar_loss
            + args.sign_loss_weight * sign_loss
        )
        return total, (mean_loss, logvar_loss, sign_loss)

    @jax.jit
    def train_step(params, opt_state, key):
        uniforms = jax.random.uniform(key, (args.batch_size,))
        idx = jnp.searchsorted(sampling_cdf, uniforms)
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, z_train_dev[idx], y_train_dev[:, idx], loss_weights[:, idx]
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, aux

    # --- Training loop ----------------------------------------------------
    t0 = time.time()
    key = jax.random.key(init_seed + 1)
    for step in range(args.steps):
        key, subkey = jax.random.split(key)
        student_params, opt_state, loss, aux = train_step(
            student_params, opt_state, subkey
        )
        if step % 200 == 0 or step == args.steps - 1:
            print(f"step {step:5d} loss={float(loss):.5f} "
                  f"mean={float(aux[0]):.5f} logvar={float(aux[1]):.5f} "
                  f"sign={float(aux[2]):.5f} "
                  f"({time.time() - t0:.0f}s)")

    # --- Validation against the teacher ----------------------------------
    student_val = np.asarray(
        jax.vmap(
            lambda p: student_network.apply(
                {"params": p}, z_val, deterministic=True
            )
        )(student_params)
    )
    out_stds = np.array(
        [teacher_dict["stats"][label]["std"] for label in output_labels]
    )
    out_means = np.array(
        [teacher_dict["stats"][label]["mean"] for label in output_labels]
    )
    metrics = {}
    for i, label in enumerate(output_labels):
        teacher_mean, student_mean = y_val[i, :, 0], student_val[i, :, 0]
        # Near-threshold subset: |flux| < 10 GB. R^2 is affine-invariant, so
        # normalized-space R^2 equals physical-space R^2; RMSE is scaled back
        # to GB units.
        near = np.abs(teacher_mean * out_stds[i] + out_means[i]) < 10.0
        # Sign agreement with the teacher where the physical flux is small
        # enough that sign errors are plausible (|q| < 20 GB) - the region
        # where the D_eff/V_eff decomposition is most sign-sensitive.
        teacher_phys = teacher_mean * out_stds[i] + out_means[i]
        student_phys = student_mean * out_stds[i] + out_means[i]
        band = np.abs(teacher_phys) < 20.0
        metrics[label] = {
            "sign_agreement_lt20gb": float(
                np.mean(
                    np.sign(student_phys[band]) == np.sign(teacher_phys[band])
                )
            ),
            "mean_r2": r2(teacher_mean, student_mean),
            "mean_rmse_gb": float(
                np.sqrt(np.mean((teacher_mean - student_mean) ** 2))
                * out_stds[i]
            ),
            "near_threshold_rmse_gb": float(
                np.sqrt(np.mean((teacher_mean[near] - student_mean[near]) ** 2))
                * out_stds[i]
            ),
            "logvar_r2": r2(
                np.log(y_val[i, :, 1] + eps), np.log(student_val[i, :, 1] + eps)
            ),
        }
        print(f"{label}: {metrics[label]}")

    # --- Package in the released pickle schema ----------------------------
    student_pickle_params = {}
    for i, label in enumerate(output_labels):
        per_flux = jax.tree.map(lambda leaf: leaf[i], student_params)
        layers = {}
        for j in range(args.num_hiddens):
            dense = per_flux["GaussianMLP_0"][f"Dense_{j}"]
            layers[f"FullyConnectedLayer_{j}"] = {
                "weight": np.asarray(dense["kernel"]).T.astype(np.float32),
                "bias": np.asarray(dense["bias"]).T.astype(np.float32),
            }
        student_pickle_params[label] = {"MLP_0": layers}

    student_config = dict(teacher_dict["config"])
    student_config.update(
        num_estimators=1,
        model_size=args.num_hiddens,
        hidden_size=args.hidden_size,
        dropout=0.0,
        activation=args.activation,
        regressor_type="DistilledStudent",
        loss_function="distillation(MSE mean + MSE logvar)",
        distillation={
            "teacher": args.machine,
            "teacher_config": {
                "num_estimators": teacher_network.n_ensemble,
                "model_size": teacher_network.num_hiddens,
                "hidden_size": teacher_network.hidden_size,
            },
            "n_train": args.n_train,
            "n_val": args.n_val,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "schedule": args.schedule,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "var_loss_weight": args.var_loss_weight,
            "loss_weight_q0": args.loss_weight_q0,
            "loss_weight_floor": args.loss_weight_floor,
            "threshold_q0": args.threshold_q0,
            "uniform_fraction": args.uniform_fraction,
            "oob_fraction": args.oob_fraction,
            "oob_margin": args.oob_margin,
            "seed": args.seed,
            "metrics": metrics,
        },
    )
    student_dict = {
        "stats": teacher_dict["stats"],
        "config": student_config,
        "input_labels": teacher_dict["input_labels"],
        "params": student_pickle_params,
    }

    output_path = (
        pathlib.Path(args.output)
        if args.output
        else pathlib.Path(tglfnn_ukaea.__file__).parent
        / "weights"
        / f"{args.machine}_student.pkl"
    )
    with open(output_path, "wb") as f:
        pickle.dump(student_dict, f)
    print(f"Wrote student checkpoint to {output_path} "
          f"({output_path.stat().st_size / 1e6:.1f} MB)")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
