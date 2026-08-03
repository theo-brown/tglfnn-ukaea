"""Validates the distilled student against its teacher via fusion_surrogates.

Loads both checkpoints through the standard inference path
(fusion_surrogates.tglfnn_ukaea.TGLFNNukaeaModel — the same code TORAX
uses), compares predictions on inputs sampled from the training hypercube,
and benchmarks single-call latency at a transport-solver-like batch size.

Requires: fusion_surrogates (pip install fusion_surrogates).

Example:
    python scripts/validate_student.py --machine multimachine
"""

import argparse
import time

from fusion_surrogates.tglfnn_ukaea import tglfnn_ukaea_model
import jax
import numpy as np

import tglfnn_ukaea

_LOG10_SAMPLED_INPUTS = ("XNUE", "BETAE")


def sample_inputs(model_dict, n, rng):
    param_space = model_dict["config"]["param_space"]
    columns = []
    for label in model_dict["input_labels"]:
        lo, hi = (float(b) for b in param_space[label])
        values = rng.uniform(lo, hi, size=n)
        if label in _LOG10_SAMPLED_INPUTS:
            values = 10.0**values
        columns.append(values)
    return np.stack(columns, axis=-1).astype(np.float32)


def benchmark(fn, x, n_repeats=100):
    fn(x)  # compile
    start = time.perf_counter()
    for _ in range(n_repeats):
        jax.block_until_ready(fn(x))
    return (time.perf_counter() - start) / n_repeats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--machine", default="multimachine")
    parser.add_argument("--n-samples", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Benchmark batch size (~number of radial grid "
                        "faces in a transport solver).")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    teacher = tglfnn_ukaea_model.TGLFNNukaeaModel(args.machine)
    student = tglfnn_ukaea_model.TGLFNNukaeaModel(f"{args.machine}_student")
    assert teacher.input_labels == student.input_labels
    assert teacher.output_labels == student.output_labels

    rng = np.random.default_rng(args.seed)
    x = sample_inputs(tglfnn_ukaea.load(args.machine), args.n_samples, rng)

    teacher_pred = teacher.predict(x)
    student_pred = student.predict(x)
    print("Student vs teacher agreement (mean channel):")
    for label in teacher.output_labels:
        t = np.asarray(teacher_pred[label][..., 0])
        s = np.asarray(student_pred[label][..., 0])
        r2 = 1.0 - np.sum((t - s) ** 2) / np.sum((t - np.mean(t)) ** 2)
        print(f"  {label}: R^2={r2:.4f}  RMSE={np.sqrt(np.mean((t-s)**2)):.3f} GB"
              f"  (teacher std {np.std(t):.1f} GB)")

    x_bench = x[: args.batch_size]
    teacher_fn = jax.jit(teacher.predict)
    student_fn = jax.jit(student.predict)
    t_teacher = benchmark(teacher_fn, x_bench)
    t_student = benchmark(student_fn, x_bench)
    print(f"Latency at batch={args.batch_size}: "
          f"teacher {t_teacher*1e6:.0f} us, student {t_student*1e6:.0f} us, "
          f"speedup {t_teacher/t_student:.1f}x")


if __name__ == "__main__":
    main()
