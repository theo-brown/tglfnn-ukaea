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
inference use literally the same Flax module. Alternatively,
``--shared-trunk`` trains a single mean-only network (shared trunk, one
linear head per flux) and exports it per flux with the trunk duplicated and
a constant per-flux variance, so it still loads through the same path.

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

import flax.linen as nn
from fusion_surrogates.common import networks
import jax
import jax.numpy as jnp
import numpy as np
import optax

import tglfnn_ukaea

_ACTIVATIONS = {"relu": jax.nn.relu, "tanh": jnp.tanh, "sigmoid": jax.nn.sigmoid}


class SharedTrunkMeanStudent(nn.Module):
    """Mean-only student: one shared trunk, one linear head per flux.

    The three fluxes share a single representation, so the well-constrained
    heat channels regularise the particle channel (the persistently
    worst-distilled output). Only the ensemble mean is learned; the exported
    checkpoint carries the teacher's average total variance per flux as a
    constant, wired into the (otherwise unused) variance column of the final
    layer so the checkpoint still loads through ``GaussianMLPEnsemble``.

    ``num_hiddens`` counts Dense layers per flux in the *exported* network
    (matching ``model_size`` in the config): ``num_hiddens - 1 -
    head_hiddens`` shared trunk layers, then ``head_hiddens`` per-flux
    hidden layers, then the per-flux output layer. ``head_hiddens`` moves
    capacity from the shared trunk into the branches at fixed total depth.
    """

    num_hiddens: int
    hidden_size: int
    activation: str
    n_heads: int
    head_hiddens: int = 0
    dtype: Any = jnp.float32
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        act = _ACTIVATIONS[self.activation]
        x = x.astype(self.dtype)
        n_trunk = self.num_hiddens - 1 - self.head_hiddens

        def maybe_drop(h):
            if self.dropout <= 0.0:
                return h
            return nn.Dropout(rate=self.dropout, deterministic=deterministic)(h)

        for j in range(n_trunk):
            x = act(
                maybe_drop(
                    nn.Dense(
                        self.hidden_size, name=f"Trunk_{j}", dtype=self.dtype
                    )(x)
                )
            )
        heads = []
        for i in range(self.n_heads):
            h = x
            for k in range(self.head_hiddens):
                h = act(
                    maybe_drop(
                        nn.Dense(
                            self.hidden_size,
                            name=f"Head_{i}_Hidden_{k}",
                            dtype=self.dtype,
                        )(h)
                    )
                )
            heads.append(nn.Dense(1, name=f"Head_{i}", dtype=self.dtype)(h))
        # Params stay float32 (Flax default param_dtype); only the compute
        # runs in self.dtype. Return float32 so the loss is accumulated at
        # full precision.
        return jnp.concatenate(heads, axis=-1).astype(jnp.float32)


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
    parser.add_argument("--mean-loss", default="mse",
                        choices=["mse", "huber", "mae"],
                        help="Regression loss for the mean-matching term "
                        "(shared-trunk mode only). 'huber' is quadratic "
                        "inside --huber-delta-gb and linear outside (tail-"
                        "robust, smooth convergence); 'mae' applies "
                        "constant-magnitude gradients, pressing hardest on "
                        "already-small residuals at the cost of a higher "
                        "noise floor (pair with a deep anneal). Composes "
                        "with the per-sample --loss-weight-q0 weighting.")
    parser.add_argument("--huber-delta-gb", type=float, default=30.0,
                        help="Huber transition scale in GB units, applied "
                        "per flux in normalized space (delta / output std).")
    parser.add_argument("--head-hiddens", type=int, default=0,
                        help="Shared-trunk mode only: number of per-flux "
                        "hidden layers in each head branch. Moves capacity "
                        "from the shared trunk to the branches at fixed "
                        "total depth (num_hiddens counts trunk + head "
                        "hidden + output layers per exported flux network).")
    parser.add_argument("--pool-cache", default=None,
                        help="Path to an .npz cache for the initial "
                        "training pool and validation set. Loaded if it "
                        "exists, created otherwise. Only valid when runs "
                        "share the same --seed/--n-train/--n-val/"
                        "--oob-fraction; useful for architecture scans that "
                        "reuse one labelled pool.")
    parser.add_argument("--shared-trunk", action="store_true",
                        help="Train a single mean-only network with a shared "
                        "trunk and one linear head per flux, instead of an "
                        "independent Gaussian MLP per flux. The shared "
                        "representation regularises the particle channel via "
                        "the heat channels. The variance is NOT learned: the "
                        "exported checkpoint carries the teacher's average "
                        "total variance per flux as a constant (variance "
                        "column of the final layer), so it still loads "
                        "through the released GaussianMLPEnsemble path.")
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
    parser.add_argument("--decay-fraction", type=float, default=0.2,
                        help="Fraction of --steps spent in the final cosine "
                        "decay phase of the 'wsd' schedule.")
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "bfloat16"],
                        help="Compute dtype for the student's forward/"
                        "backward pass (shared-trunk mode only). Parameters "
                        "and the loss stay float32; the exported checkpoint "
                        "is float32 either way.")
    parser.add_argument("--resample-every", type=int, default=0,
                        help="If > 0, regenerate and relabel the entire "
                        "training pool every this many steps (online "
                        "resampling). The distillation targets are "
                        "noiseless and teacher labelling is cheap, so this "
                        "removes the finite-pool memorisation floor: the "
                        "student never sees the same sample twice across "
                        "pool generations. The validation set is fixed. "
                        "0 keeps a single fixed pool.")
    parser.add_argument("--ema-decay", type=float, default=0.0,
                        help="If > 0 (e.g. 0.999), maintain an exponential "
                        "moving average of the student weights and export "
                        "it as the checkpoint; the final raw weights are "
                        "exported alongside with a _raw suffix.")
    parser.add_argument("--sobolev-weight", type=float, default=0.0,
                        help="If > 0, adds a Sobolev term matching teacher "
                        "directional derivatives: per step, random unit "
                        "directions are drawn and the MAE between student "
                        "and teacher directional derivatives (from "
                        "precomputed teacher Jacobians) is penalized. "
                        "Shared-trunk only.")
    parser.add_argument("--sobolev-dirs", type=int, default=2,
                        help="Random directions per step for the Sobolev "
                        "term (ignored with --sobolev-exact).")
    parser.add_argument("--sobolev-exact", action="store_true",
                        help="Match the full student Jacobian against the "
                        "cached teacher Jacobian instead of a random "
                        "directional projection. With 3 outputs the exact "
                        "Jacobian is 3 VJPs, so this removes the estimator "
                        "noise at roughly 2x the Sobolev-term cost.")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Dropout rate applied after each trunk/head "
                        "hidden Dense layer during training. Inference is "
                        "always deterministic, so the exported checkpoint "
                        "records dropout=0 in its top-level config.")
    parser.add_argument("--target-transform", default="linear",
                        choices=["linear", "signed_log"],
                        help="Space in which the mean-matching residual is "
                        "measured. 'signed_log' applies "
                        "sign(q)*log1p(|q|/q0) to both prediction and "
                        "target in physical units, giving relative-error "
                        "behaviour across the ~4 decades of flux while "
                        "staying smooth through zero. The network still "
                        "emits linear normalized flux, so the exported "
                        "checkpoint loads through the released path "
                        "unchanged.")
    parser.add_argument("--target-log-q0", type=float, default=10.0,
                        help="Scale (GB) of the signed-log transform.")
    parser.add_argument("--mix-boxedge", type=float, default=0.0,
                        help="Fraction of the pool drawn near the faces of "
                        "the training hypercube: a random subset of "
                        "dimensions is pushed into the outer "
                        "--boxedge-band of its range.")
    parser.add_argument("--boxedge-band", type=float, default=0.1,
                        help="Width of the near-face band as a fraction of "
                        "each dimension's range.")
    parser.add_argument("--boxedge-dims", type=int, default=2,
                        help="Max number of dimensions pushed to a face per "
                        "box-edge sample (1..this, drawn uniformly).")
    parser.add_argument("--mix-variance", type=float, default=0.0,
                        help="Fraction of the pool drawn by importance "
                        "sampling on the teacher's epistemic ensemble "
                        "variance (variance across member means), the "
                        "query-by-committee signal.")
    parser.add_argument("--variance-candidates", type=int, default=6,
                        help="Candidate oversampling factor for "
                        "--mix-variance: this many candidates are labelled "
                        "per accepted sample.")
    parser.add_argument("--boundary-fraction", type=float, default=0.0,
                        help="Fraction of the training pool rejection-"
                        "sampled from the near-threshold shell: uniform "
                        "in-box draws kept only if any flux satisfies "
                        "|flux_GB| < --boundary-qgb.")
    parser.add_argument("--boundary-qgb", type=float, default=10.0,
                        help="Shell half-width in GB for "
                        "--boundary-fraction sampling.")
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
    parser.add_argument("--init-from", default=None,
                        help="Path to a previously exported shared-trunk "
                        "student pickle to continue training from (warm "
                        "restart). Loads the trunk and head weights; the "
                        "optimizer state is rebuilt from scratch, so pair "
                        "this with --warmup-steps and a reduced --lr (e.g. "
                        "1/3 of the previous cycle's peak) so the fresh "
                        "Adam moments do not kick the model off its "
                        "minimum. Requires --shared-trunk with matching "
                        "architecture.")
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
    if args.shared_trunk:
        n_trunk = args.num_hiddens - 1 - args.head_hiddens
        if n_trunk < 1:
            raise SystemExit("--head-hiddens leaves no trunk layers")
        print(f"Student: shared {args.hidden_size}-wide x "
              f"{n_trunk}-layer trunk ({args.activation}) + "
              f"{n_fluxes} mean heads with {args.head_hiddens} hidden "
              f"layer(s) each (variance not learned)")
    else:
        print(f"Student: 1x{args.hidden_size}-wide x {args.num_hiddens}-layer "
              f"({args.activation}) per flux")

    out_stds_np = np.array(
        [teacher_dict["stats"][label]["std"] for label in output_labels]
    )
    out_means_np = np.array(
        [teacher_dict["stats"][label]["mean"] for label in output_labels]
    )

    def teacher_jacobians(z, batch_size=8192):
        """Teacher Jacobian d(normalized mean)/dz per flux: (n, n_fluxes, 13).

        Costs n_fluxes VJPs per point (~4x the labelling cost); consumed by
        the Sobolev term as directional-derivative labels.
        """

        def flux_means(z_single):
            out = jax.vmap(
                lambda p: teacher_network.apply(
                    {"params": p}, z_single[None, :], deterministic=True
                )
            )(teacher_params)
            return out[:, 0, 0]

        jac_fn = jax.jit(jax.vmap(jax.jacrev(flux_means)))
        chunks = []
        tj = time.time()
        for start in range(0, z.shape[0], batch_size):
            chunks.append(
                np.asarray(jac_fn(z[start : start + batch_size])).astype(
                    np.float32
                )
            )
            if start == 0:
                rate = batch_size / max(time.time() - tj, 1e-9)
                print(f"Jacobian labelling at ~{rate:.0f} samples/s "
                      f"(first batch incl. compile); "
                      f"{z.shape[0] / rate / 60:.0f} min estimated")
        return np.concatenate(chunks)

    def teacher_epistemic(z, batch_size=65536):
        """Variance across ensemble-member means, per flux: (n_fluxes, n).

        The ensemble's second output channel is aleatoric+epistemic; the
        query-by-committee signal is the epistemic part alone, so the
        members are evaluated individually here.
        """
        member = networks.GaussianMLP(
            num_hiddens=teacher_network.num_hiddens,
            hidden_size=teacher_network.hidden_size,
            dropout=0.0,
            activation=teacher_network.activation,
        )

        @jax.jit
        def forward(x):
            members = [
                jax.vmap(
                    lambda p: member.apply(
                        {"params": p}, x, deterministic=True
                    )[..., 0]
                )(teacher_params[f"GaussianMLP_{i}"])
                for i in range(teacher_network.n_ensemble)
            ]
            return jnp.var(jnp.stack(members), axis=0)  # (n_fluxes, batch)

        out = [
            np.asarray(forward(z[s : s + batch_size]))
            for s in range(0, z.shape[0], batch_size)
        ]
        return np.concatenate(out, axis=1)

    # --- Distillation data ------------------------------------------------
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    n_oob = int(args.n_train * args.oob_fraction)
    n_boundary = int(args.n_train * args.boundary_fraction)
    n_boxedge = int(args.n_train * args.mix_boxedge)
    n_variance = int(args.n_train * args.mix_variance)

    def sample_boxedge(pool_rng, n_target):
        """Uniform samples with a few dimensions pushed against a box face.

        Extrapolation risk is concentrated at the faces of the training
        hypercube, and a solver line search reaches them; uniform sampling
        puts almost no mass there in 13 dimensions.
        """
        x = sample_inputs(teacher_dict, n_target, pool_rng)
        param_space = teacher_dict["config"]["param_space"]
        n_dims = x.shape[1]
        k = pool_rng.integers(1, args.boxedge_dims + 1, size=n_target)
        for label_i, label in enumerate(teacher_dict["input_labels"]):
            lo, hi = (float(b) for b in param_space[label][:2])
            span = hi - lo
            # Which rows push THIS dimension to a face.
            hit = pool_rng.random(n_target) < (k / n_dims)
            if not hit.any():
                continue
            u = pool_rng.random(hit.sum()) * args.boxedge_band
            high_side = pool_rng.random(hit.sum()) < 0.5
            edge = np.where(high_side, hi - u * span, lo + u * span)
            if label in _LOG10_SAMPLED_INPUTS:
                edge = 10.0**edge
            x[hit, label_i] = edge.astype(np.float32)
        return x

    def sample_variance(pool_rng, n_target):
        """Importance-samples on the teacher's epistemic ensemble variance."""
        n_cand = n_target * args.variance_candidates
        kept = []
        got = 0
        while got < n_target:
            xc = sample_inputs(teacher_dict, min(n_cand, 2_000_000), pool_rng)
            zc = (jnp.asarray(xc) - in_means) / in_stds
            ep = teacher_epistemic(zc)  # (n_fluxes, n)
            # Normalize per flux so no channel's scale dominates, then take
            # the worst channel as the acquisition score.
            score = np.max(ep / (np.mean(ep, axis=1, keepdims=True) + 1e-12),
                           axis=0)
            p = score / score.sum()
            take = min(n_target - got, xc.shape[0])
            idx = pool_rng.choice(xc.shape[0], size=take, replace=False, p=p)
            kept.append(xc[idx])
            got += take
            print(f"  variance sampling: {got}/{n_target} "
                  f"(score p90/p50 = "
                  f"{np.percentile(score, 90) / np.median(score):.1f}x)")
        return np.concatenate(kept)[:n_target]

    def sample_boundary(pool_rng, n_target):
        """Rejection-samples uniform in-box points with any |flux| in the
        near-threshold shell."""
        kept, got = [], 0
        while got < n_target:
            xc = sample_inputs(teacher_dict, 1_000_000, pool_rng)
            zc = (jnp.asarray(xc) - in_means) / in_stds
            yc = teacher_labels(teacher_network, teacher_params, zc)
            flux_gb = np.abs(
                yc[..., 0] * out_stds_np[:, None] + out_means_np[:, None]
            )
            mask = np.any(flux_gb < args.boundary_qgb, axis=0)
            kept.append(xc[mask])
            got += int(mask.sum())
            print(f"  boundary sampling: {got}/{n_target} "
                  f"(acceptance {100 * mask.mean():.1f}%)")
        return np.concatenate(kept)[:n_target]

    def make_pool(pool_rng):
        """Samples and teacher-labels a fresh training pool."""
        n_uniform = (
            args.n_train - n_oob - n_boundary - n_boxedge - n_variance
        )
        if n_uniform < 0:
            raise SystemExit("pool mix fractions exceed 1.0")
        parts = [
            sample_inputs(teacher_dict, n_uniform, pool_rng),
            sample_inputs(
                teacher_dict, n_oob, pool_rng, margin=args.oob_margin
            ),
        ]
        if n_boundary > 0:
            parts.append(sample_boundary(pool_rng, n_boundary))
        if n_boxedge > 0:
            parts.append(sample_boxedge(pool_rng, n_boxedge))
        if n_variance > 0:
            parts.append(sample_variance(pool_rng, n_variance))
        x = np.concatenate(parts)
        z = (jnp.asarray(x) - in_means) / in_stds
        y = teacher_labels(teacher_network, teacher_params, z)
        return z, jnp.asarray(y)

    need_jac = args.sobolev_weight > 0
    j_train_dev = None
    if args.pool_cache and pathlib.Path(args.pool_cache).exists():
        cached = dict(np.load(args.pool_cache))
        z_train_dev = jnp.asarray(cached["z_train"])
        y_train_dev = jnp.asarray(cached["y_train"])
        z_val = jnp.asarray(cached["z_val"])
        y_val = cached["y_val"]
        print(f"Loaded pool cache from {args.pool_cache} in "
              f"{time.time() - t0:.1f}s")
        if need_jac:
            if "j_train" in cached:
                j_train_dev = jnp.asarray(cached["j_train"])
            else:
                print("Cache lacks Jacobians; computing and re-saving...")
                j_np = teacher_jacobians(z_train_dev)
                cached["j_train"] = j_np
                np.savez(args.pool_cache, **cached)
                j_train_dev = jnp.asarray(j_np)
    else:
        # Validation set FIRST so it is identical across pool compositions
        # (boundary sampling consumes an unpredictable amount of the rng
        # stream).
        x_val = sample_inputs(teacher_dict, args.n_val, rng)
        z_val = (jnp.asarray(x_val) - in_means) / in_stds
        y_val = teacher_labels(teacher_network, teacher_params, z_val)
        z_train_dev, y_train_dev = make_pool(rng)
        print(f"Labelled {args.n_train}+{args.n_val} samples with the "
              f"teacher in {time.time() - t0:.1f}s")
        save = {
            "z_train": np.asarray(z_train_dev),
            "y_train": np.asarray(y_train_dev),
            "z_val": np.asarray(z_val),
            "y_val": np.asarray(y_val),
        }
        if need_jac:
            save["j_train"] = teacher_jacobians(z_train_dev)
            j_train_dev = jnp.asarray(save["j_train"])
        if args.pool_cache:
            np.savez(args.pool_cache, **save)
            print(f"Saved pool cache to {args.pool_cache}")

    # --- Student setup ----------------------------------------------------
    init_seed = args.init_seed if args.init_seed is not None else args.seed
    dummy = jnp.zeros((1, n_inputs))
    if args.dtype != "float32" and not args.shared_trunk:
        raise SystemExit("--dtype bfloat16 requires --shared-trunk")
    if args.shared_trunk:
        student_network = SharedTrunkMeanStudent(
            num_hiddens=args.num_hiddens,
            hidden_size=args.hidden_size,
            activation=args.activation,
            n_heads=n_fluxes,
            head_hiddens=args.head_hiddens,
            dtype=jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32,
        )
        student_params = student_network.init(
            jax.random.key(init_seed), dummy
        )["params"]
        if args.init_from:
            with open(args.init_from, "rb") as f:
                src = pickle.load(f)
            src_labels = tuple(src["params"].keys())
            if src_labels != output_labels:
                raise SystemExit(
                    f"--init-from flux labels {src_labels} do not match "
                    f"teacher {output_labels}"
                )
            src_head_hiddens = (
                src["config"].get("distillation", {}).get("head_hiddens", 0)
            )
            if (
                src["config"]["model_size"] != args.num_hiddens
                or src["config"]["hidden_size"] != args.hidden_size
                or src_head_hiddens != args.head_hiddens
            ):
                raise SystemExit("--init-from architecture mismatch")
            n_trunk_init = args.num_hiddens - 1 - args.head_hiddens
            restored = {}
            for j in range(n_trunk_init):
                lay = src["params"][src_labels[0]]["MLP_0"][
                    f"FullyConnectedLayer_{j}"
                ]
                restored[f"Trunk_{j}"] = {
                    "kernel": jnp.array(lay["weight"].T),
                    "bias": jnp.array(lay["bias"]),
                }
            last = f"FullyConnectedLayer_{args.num_hiddens - 1}"
            for i, label in enumerate(src_labels):
                for k in range(args.head_hiddens):
                    lay = src["params"][label]["MLP_0"][
                        f"FullyConnectedLayer_{n_trunk_init + k}"
                    ]
                    restored[f"Head_{i}_Hidden_{k}"] = {
                        "kernel": jnp.array(lay["weight"].T),
                        "bias": jnp.array(lay["bias"]),
                    }
                lay = src["params"][label]["MLP_0"][last]
                # Row 0 of the exported 2-unit output layer is the mean
                # head; row 1 is the constant-variance column, dropped here.
                restored[f"Head_{i}"] = {
                    "kernel": jnp.array(lay["weight"][0:1].T),
                    "bias": jnp.array(lay["bias"][0:1]),
                }
            student_params = restored
            print(f"Initialised student from {args.init_from}")
    else:
        student_network = networks.GaussianMLPEnsemble(
            n_ensemble=1,
            num_hiddens=args.num_hiddens,
            hidden_size=args.hidden_size,
            dropout=0.0,
            activation=args.activation,
        )
        init_keys = jax.random.split(jax.random.key(init_seed), n_fluxes)
        student_params = jax.tree.map(
            lambda *args_: jnp.stack(args_),
            *[
                student_network.init(k, dummy, deterministic=True)["params"]
                for k in init_keys
            ],
        )

    if args.schedule == "wsd":
        decay_steps = max(1, int(args.decay_fraction * args.steps))
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

    var_weight = args.var_loss_weight
    eps = 1e-6

    out_stds_dev = jnp.array(
        [teacher_dict["stats"][label]["std"] for label in output_labels]
    )
    out_means_dev = jnp.array(
        [teacher_dict["stats"][label]["mean"] for label in output_labels]
    )

    # Teacher's mean total variance per flux: the variance-head bias init for
    # the Gaussian student, or the constant exported variance for the
    # mean-only shared-trunk student.
    mean_var = jnp.mean(y_train_dev[..., 1], axis=1)
    if not args.shared_trunk:
        # Initialise the variance-head bias to the teacher's mean variance so
        # early training is not spent dragging softplus(0) up to scale.
        last_layer = f"Dense_{args.num_hiddens - 1}"
        last_bias = student_params["GaussianMLP_0"][last_layer]["bias"]
        student_params["GaussianMLP_0"][last_layer]["bias"] = last_bias.at[
            :, 1
        ].set(jnp.log(jnp.expm1(jnp.maximum(mean_var, eps))))

    opt_state = optimizer.init(student_params)

    def pool_arrays(y_pool):
        """Per-pool loss weights and minibatch-sampling CDF.

        Loss weights implement the asinh-equivalent error allocation; the CDF
        implements threshold-weighted minibatch sampling (probability
        proportional to a mixture of uniform and the per-flux average of
        1/(|flux_GB| + q0)) via inverse-CDF lookup in the jitted train step.
        """
        flux_gb = jnp.abs(
            y_pool[..., 0] * out_stds_dev[:, None] + out_means_dev[:, None]
        )  # (n_fluxes, n_samples)
        if args.loss_weight_q0 > 0:
            q0 = args.loss_weight_q0
            relative = q0**2 / (q0**2 + flux_gb**2)
            weights = (
                args.loss_weight_floor
                + (1.0 - args.loss_weight_floor) * relative
            )
        else:
            weights = jnp.ones_like(flux_gb)
        weights = weights / jnp.mean(weights)

        near_threshold_weight = jnp.mean(
            1.0 / (flux_gb + args.threshold_q0), axis=0
        )
        n_pool = y_pool.shape[1]
        probabilities = (
            args.uniform_fraction / n_pool
            + (1.0 - args.uniform_fraction)
            * near_threshold_weight
            / jnp.sum(near_threshold_weight)
        )
        cdf = jnp.cumsum(probabilities)
        return weights, cdf / cdf[-1]

    loss_weights, sampling_cdf = pool_arrays(y_train_dev)

    sign_scale = args.loss_weight_q0 if args.loss_weight_q0 > 0 else 10.0

    def _sign_hinge(pred_mean, true_mean):
        # Sign-consistency hinge on physical fluxes: active only when the
        # prediction and the teacher disagree in sign, scaled by both
        # magnitudes (a confident wrong-sign costs more).
        q_pred = pred_mean * out_stds_dev[:, None] + out_means_dev[:, None]
        q_true = true_mean * out_stds_dev[:, None] + out_means_dev[:, None]
        return jnp.mean(jax.nn.relu(-q_pred * q_true)) / sign_scale**2

    if args.shared_trunk:
        # Per-flux Huber scale in normalized space.
        huber_delta = args.huber_delta_gb / out_stds_dev

        def _mean_err(resid):
            if args.mean_loss == "mse":
                return resid**2
            if args.mean_loss == "mae":
                return jnp.abs(resid)
            d = huber_delta[:, None]
            a = jnp.abs(resid)
            return jnp.where(a <= d, 0.5 * resid**2, d * (a - 0.5 * d))

        def _to_physical(norm):
            return norm * out_stds_dev[:, None] + out_means_dev[:, None]

        def _signed_log(q):
            # Smooth (C1) through zero: d/dq = 1/(q0 + |q|).
            return jnp.sign(q) * jnp.log1p(jnp.abs(q) / args.target_log_q0)

        def loss_fn(params, z, y, w, j, dirs, drop_key):
            # (batch, n_fluxes) -> (n_fluxes, batch), matching y/w layout.
            kwargs = {}
            if args.dropout > 0:
                kwargs = {
                    "deterministic": False,
                    "rngs": {"dropout": drop_key},
                }
            pred_mean = student_network.apply(
                {"params": params}, z, **kwargs
            ).T
            if args.target_transform == "signed_log":
                resid = _signed_log(_to_physical(pred_mean)) - _signed_log(
                    _to_physical(y[..., 0])
                )
            else:
                resid = pred_mean - y[..., 0]
            mean_loss = jnp.mean(w * _mean_err(resid))
            sign_loss = _sign_hinge(pred_mean, y[..., 0])
            sob_loss = jnp.zeros(())
            if args.sobolev_weight > 0:
                # The Sobolev term always uses the deterministic network:
                # it matches the teacher's deterministic Jacobian, so
                # injecting dropout noise into it would be counterproductive.
                if args.sobolev_exact:
                    # Full student Jacobian: 3 VJPs per sample, no estimator
                    # noise. j is (batch, n_fluxes, n_inputs).
                    jac_student = jax.vmap(
                        jax.jacrev(
                            lambda s: student_network.apply(
                                {"params": params}, s[None, :]
                            )[0]
                        )
                    )(z)
                    sob_loss = jnp.mean(
                        w.T[:, :, None] * jnp.abs(jac_student - j)
                    )
                else:
                    # Stochastic Sobolev: match teacher directional
                    # derivatives along random unit directions. Teacher side
                    # is a contraction of the precomputed Jacobian; student
                    # side is one JVP per direction.
                    def apply_T(zz):
                        return student_network.apply({"params": params}, zz).T

                    for d in range(args.sobolev_dirs):
                        v = dirs[d]
                        _, dd_student = jax.jvp(
                            apply_T, (z,), (jnp.broadcast_to(v, z.shape),)
                        )
                        dd_teacher = jnp.einsum("bfi,i->fb", j, v)
                        sob_loss = sob_loss + jnp.mean(
                            w * jnp.abs(dd_student - dd_teacher)
                        )
                    sob_loss = sob_loss / args.sobolev_dirs
            total = (
                mean_loss
                + args.sign_loss_weight * sign_loss
                + args.sobolev_weight * sob_loss
            )
            return total, (mean_loss, sob_loss, sign_loss)

    else:
        if args.mean_loss != "mse":
            raise SystemExit("--mean-loss requires --shared-trunk")

        if args.sobolev_weight > 0:
            raise SystemExit("--sobolev-weight requires --shared-trunk")

        def loss_fn(params, z, y, w, j, dirs, drop_key):
            del j, dirs, drop_key
            pred = jax.vmap(
                lambda p: student_network.apply(
                    {"params": p}, z, deterministic=True
                )
            )(params)
            mean_loss = jnp.mean(w * (pred[..., 0] - y[..., 0]) ** 2)
            logvar_loss = jnp.mean(
                w
                * (jnp.log(pred[..., 1] + eps) - jnp.log(y[..., 1] + eps)) ** 2
            )
            sign_loss = _sign_hinge(pred[..., 0], y[..., 0])
            total = (
                mean_loss
                + var_weight * logvar_loss
                + args.sign_loss_weight * sign_loss
            )
            return total, (mean_loss, logvar_loss, sign_loss)

    use_sobolev = args.sobolev_weight > 0

    @jax.jit
    def train_step(
        params, ema, opt_state, key, z_pool, y_pool, w_pool, cdf, j_pool
    ):
        key_idx, key_dirs, key_drop = jax.random.split(key, 3)
        uniforms = jax.random.uniform(key_idx, (args.batch_size,))
        idx = jnp.searchsorted(cdf, uniforms)
        if use_sobolev:
            j_batch = j_pool[idx]
            dirs = jax.random.normal(key_dirs, (args.sobolev_dirs, 13))
            dirs = dirs / jnp.linalg.norm(dirs, axis=1, keepdims=True)
        else:
            j_batch, dirs = None, None
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, z_pool[idx], y_pool[:, idx], w_pool[:, idx],
            j_batch, dirs, key_drop,
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        if args.ema_decay > 0:
            ema = jax.tree.map(
                lambda e, p: args.ema_decay * e + (1 - args.ema_decay) * p,
                ema, params,
            )
        return params, ema, opt_state, loss, aux

    # --- Training loop ----------------------------------------------------
    if args.resample_every > 0 and use_sobolev:
        raise SystemExit(
            "--resample-every with --sobolev-weight is not supported "
            "(Jacobian labels would go stale)"
        )
    aux1_name = "sob" if args.shared_trunk else "logvar"
    ema_params = student_params if args.ema_decay > 0 else None
    t0 = time.time()
    key = jax.random.key(init_seed + 1)
    for step in range(args.steps):
        if (
            args.resample_every > 0
            and step > 0
            and step % args.resample_every == 0
        ):
            # Online resampling: a completely fresh, freshly-labelled pool.
            # Pool shapes are unchanged, so the jitted step does not
            # recompile.
            tr = time.time()
            generation = step // args.resample_every
            z_train_dev, y_train_dev = make_pool(
                np.random.default_rng(args.seed + 100_003 * generation)
            )
            loss_weights, sampling_cdf = pool_arrays(y_train_dev)
            print(f"step {step:5d} resampled pool (generation {generation}, "
                  f"{time.time() - tr:.1f}s)")
        key, subkey = jax.random.split(key)
        student_params, ema_params, opt_state, loss, aux = train_step(
            student_params, ema_params, opt_state, subkey,
            z_train_dev, y_train_dev, loss_weights, sampling_cdf,
            j_train_dev,
        )
        if step % 200 == 0 or step == args.steps - 1:
            print(f"step {step:5d} loss={float(loss):.5f} "
                  f"mean={float(aux[0]):.5f} {aux1_name}={float(aux[1]):.5f} "
                  f"sign={float(aux[2]):.5f} "
                  f"({time.time() - t0:.0f}s)")

    # --- Validation against the teacher ----------------------------------
    # Jacobian validation subset (shared across param sets): teacher
    # Jacobians on 20k val points, compared against student Jacobians.
    n_jval = min(20_000, z_val.shape[0])
    j_val_teacher = None
    if args.shared_trunk:
        j_val_teacher = teacher_jacobians(z_val[:n_jval])

    def compute_metrics(params):
        if args.shared_trunk:
            val_means = np.asarray(
                student_network.apply({"params": params}, z_val)
            ).T  # (n_fluxes, n_val)
            # Constant per-flux variance, as exported.
            val_vars = np.broadcast_to(
                np.asarray(mean_var)[:, None], val_means.shape
            )
            student_val = np.stack([val_means, val_vars], axis=-1)
        else:
            student_val = np.asarray(
                jax.vmap(
                    lambda p: student_network.apply(
                        {"params": p}, z_val, deterministic=True
                    )
                )(params)
            )
        metrics = _pointwise_metrics(student_val)
        if args.shared_trunk:
            s_jac = np.asarray(
                jax.vmap(
                    jax.jacrev(
                        lambda zz: student_network.apply(
                            {"params": params}, zz[None, :]
                        )[0]
                    )
                )(z_val[:n_jval])
            )  # (n_jval, n_fluxes, 13)
            dj = np.abs(s_jac - j_val_teacher)
            drive = [
                list(teacher_dict["input_labels"]).index(k)
                for k in ("RLNS_1", "RLTS_1", "RLTS_2")
            ]
            for i, label in enumerate(output_labels):
                metrics[label]["jac_mae_norm"] = float(np.mean(dj[:, i, :]))
                metrics[label]["jac_mae_drive_norm"] = float(
                    np.mean(dj[:, i, drive])
                )
        for label in output_labels:
            print(f"{label}: {metrics[label]}")
        return metrics

    out_stds = out_stds_np
    out_means = out_means_np

    def _pointwise_metrics(student_val):
        metrics = {}
        for i, label in enumerate(output_labels):
            metrics[label] = _one_flux_metrics(i, label, student_val)
        return metrics

    def _one_flux_metrics(i, label, student_val):
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
        result = {
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
        }
        if not args.shared_trunk:
            result["logvar_r2"] = r2(
                np.log(y_val[i, :, 1] + eps), np.log(student_val[i, :, 1] + eps)
            )
        return result

    # --- Package in the released pickle schema ----------------------------
    def package_params(export_params):
        pickle_params = {}
        if args.shared_trunk:
            # Export per flux as trunk + head, with the head padded to the
            # 2-unit [mean, variance] output layer GaussianMLP expects: the
            # variance column has zero weights and a bias of
            # softplus^-1(teacher mean variance), so the loaded network emits
            # the trained mean and a constant per-flux variance. The trunk
            # weights are duplicated across fluxes, trading file size for
            # loading through the released inference path unchanged.
            mean_var_np = np.maximum(np.asarray(mean_var), eps)
            n_trunk_out = args.num_hiddens - 1 - args.head_hiddens
            for i, label in enumerate(output_labels):
                layers = {}
                for j in range(n_trunk_out):
                    dense = export_params[f"Trunk_{j}"]
                    layers[f"FullyConnectedLayer_{j}"] = {
                        "weight": np.asarray(dense["kernel"]).T.astype(
                            np.float32
                        ),
                        "bias": np.asarray(dense["bias"]).T.astype(np.float32),
                    }
                for k in range(args.head_hiddens):
                    dense = export_params[f"Head_{i}_Hidden_{k}"]
                    layers[f"FullyConnectedLayer_{n_trunk_out + k}"] = {
                        "weight": np.asarray(dense["kernel"]).T.astype(
                            np.float32
                        ),
                        "bias": np.asarray(dense["bias"]).T.astype(np.float32),
                    }
                head = export_params[f"Head_{i}"]
                head_kernel = np.asarray(head["kernel"])  # (hidden, 1)
                out_weight = np.zeros(
                    (2, head_kernel.shape[0]), dtype=np.float32
                )
                out_weight[0] = head_kernel[:, 0]
                out_bias = np.array(
                    [
                        float(np.asarray(head["bias"])[0]),
                        float(np.log(np.expm1(mean_var_np[i]))),
                    ],
                    dtype=np.float32,
                )
                layers[f"FullyConnectedLayer_{args.num_hiddens - 1}"] = {
                    "weight": out_weight,
                    "bias": out_bias,
                }
                pickle_params[label] = {"MLP_0": layers}
        else:
            for i, label in enumerate(output_labels):
                per_flux = jax.tree.map(lambda leaf: leaf[i], export_params)
                layers = {}
                for j in range(args.num_hiddens):
                    dense = per_flux["GaussianMLP_0"][f"Dense_{j}"]
                    layers[f"FullyConnectedLayer_{j}"] = {
                        "weight": np.asarray(dense["kernel"]).T.astype(
                            np.float32
                        ),
                        "bias": np.asarray(dense["bias"]).T.astype(np.float32),
                    }
                pickle_params[label] = {"MLP_0": layers}
        return pickle_params

    student_config = dict(teacher_dict["config"])
    student_config.update(
        num_estimators=1,
        model_size=args.num_hiddens,
        hidden_size=args.hidden_size,
        dropout=0.0,
        activation=args.activation,
        regressor_type="DistilledStudent",
        loss_function=(
            "distillation(MSE mean; shared trunk; constant variance)"
            if args.shared_trunk
            else "distillation(MSE mean + MSE logvar)"
        ),
        distillation={
            "teacher": args.machine,
            "shared_trunk": args.shared_trunk,
            "mean_only": args.shared_trunk,
            "sign_loss_weight": args.sign_loss_weight,
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
            "decay_fraction": args.decay_fraction,
            "dtype": args.dtype,
            "resample_every": args.resample_every,
            "init_from": args.init_from,
            "head_hiddens": args.head_hiddens,
            "mean_loss": args.mean_loss,
            "huber_delta_gb": args.huber_delta_gb,
            "weight_decay": args.weight_decay,
            "var_loss_weight": args.var_loss_weight,
            "loss_weight_q0": args.loss_weight_q0,
            "loss_weight_floor": args.loss_weight_floor,
            "threshold_q0": args.threshold_q0,
            "uniform_fraction": args.uniform_fraction,
            "oob_fraction": args.oob_fraction,
            "oob_margin": args.oob_margin,
            "seed": args.seed,
            "ema_decay": args.ema_decay,
            "sobolev_weight": args.sobolev_weight,
            "sobolev_dirs": args.sobolev_dirs,
            "sobolev_exact": args.sobolev_exact,
            "boundary_fraction": args.boundary_fraction,
            "boundary_qgb": args.boundary_qgb,
            "train_dropout": args.dropout,
            "target_transform": args.target_transform,
            "target_log_q0": args.target_log_q0,
            "mix_boxedge": args.mix_boxedge,
            "boxedge_band": args.boxedge_band,
            "boxedge_dims": args.boxedge_dims,
            "mix_variance": args.mix_variance,
        },
    )

    def export(export_params, path, ema_flag):
        metrics = compute_metrics(export_params)
        cfg = dict(student_config)
        dist = dict(cfg["distillation"])
        dist["metrics"] = metrics
        dist["ema"] = ema_flag
        cfg["distillation"] = dist
        student_dict = {
            "stats": teacher_dict["stats"],
            "config": cfg,
            "input_labels": teacher_dict["input_labels"],
            "params": package_params(export_params),
        }
        with open(path, "wb") as f:
            pickle.dump(student_dict, f)
        print(f"Wrote student checkpoint to {path} "
              f"({path.stat().st_size / 1e6:.1f} MB)")
        return metrics

    output_path = (
        pathlib.Path(args.output)
        if args.output
        else pathlib.Path(tglfnn_ukaea.__file__).parent
        / "weights"
        / f"{args.machine}_student.pkl"
    )
    export_ema = args.ema_decay > 0
    main_params = ema_params if export_ema else student_params
    metrics = export(main_params, output_path, export_ema)
    if export_ema:
        raw_path = output_path.with_name(output_path.stem + "_raw.pkl")
        print("--- raw (non-EMA) weights ---")
        export(student_params, raw_path, False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
