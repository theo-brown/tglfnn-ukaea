"""Runs the TGLFNN active learning loop from the command line.

Requires ``pip install tglfnn_ukaea[active-learning]`` plus, for real TGLF
labelling, TORAX with its compiled TGLF wrapper (``tglf2py_lib``); see
https://torax.readthedocs.io/en/latest/installation.html#optional-install-tglf.
Use ``--mock-oracle`` for a dry run without TGLF installed.

Example:
    python scripts/run_active_learning.py --n-rounds 10 --acquisition-batch 64
"""

import argparse
import dataclasses

from tglfnn_ukaea import active_learning


def main() -> None:
    defaults = active_learning.ActiveLearningConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    for field in dataclasses.fields(defaults):
        if field.name == "flux_cutoff_gb":
            continue
        name = "--" + field.name.replace("_", "-")
        value = getattr(defaults, field.name)
        if field.type == "bool":
            parser.add_argument(
                name,
                type=lambda s: s.lower() in ("true", "1", "yes"),
                default=value,
            )
        else:
            parser.add_argument(name, type=type(value), default=value)
    parser.add_argument(
        "--mock-oracle",
        action="store_true",
        help="Use a cheap analytic oracle instead of TGLF (for testing).",
    )
    parser.add_argument(
        "--oracle-workers",
        type=int,
        default=1,
        help="TGLF worker processes: 1 runs serially, 0 uses all CPU cores.",
    )
    args = vars(parser.parse_args())

    n_workers = args.pop("oracle_workers")
    if args.pop("mock_oracle"):
        oracle = active_learning.mock_oracle
    elif n_workers == 1:
        oracle = active_learning.tglf_oracle
    else:
        oracle = active_learning.ParallelTGLFOracle(n_workers or None)
    config = active_learning.ActiveLearningConfig(**args)
    result = active_learning.run_active_learning(config, oracle=oracle)
    print(f"Done: {len(result['x'])} labelled points, "
          f"checkpoints in '{config.output_dir}'")


if __name__ == "__main__":
    main()
