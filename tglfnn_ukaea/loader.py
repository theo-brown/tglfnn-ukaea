import pathlib
import pickle
from typing import Any, Literal, Mapping

_KNOWN_MODELS = ["step", "multimachine", "multimachine_student"]


def load(
    machine: Literal[
        "step", "multimachine", "multimachine_student"
    ] = "multimachine",
) -> Mapping[str, Any]:
    """Loads a released checkpoint as a nested dict of numpy arrays.

    "multimachine_student" is a single-network distillation of the
    "multimachine" deep ensemble (see scripts/distill_student.py): same
    inputs, outputs and normalisation stats, ~26x fewer FLOPs per
    evaluation. Its epistemic uncertainty is folded into the single
    variance output.
    """
    if machine not in _KNOWN_MODELS:
        raise ValueError(
            f"Unknown machine type: '{machine}' (must be one of "
            f"{_KNOWN_MODELS})"
        )

    pickle_file = pathlib.Path(__file__).parent / "weights" / f"{machine}.pkl"
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)

    return data
