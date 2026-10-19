"""JAX model of the TGLFNN-ukaea networks distributed as ONNX graphs.

Requires JAX (the `jax` extra). The networks are evaluated by interpreting
their ONNX graphs (tglfnn_ukaea.graph), so the predictions are those of the
original exports and are differentiable with JAX in forward and reverse mode.
"""

from __future__ import annotations

import os
from typing import Any, Literal, Mapping

import jax
import jax.numpy as jnp
import numpy as np

from tglfnn_ukaea import graph as graph_lib
from tglfnn_ukaea import loader

GRAPH_FORMAT = "onnx-graph"


class TGLFNNukaeaGraphModel:
  """UKAEA TGLF surrogate, evaluated from its ONNX graphs.

  This object is intended for use as a static argument to jitted JAX
  functions. It therefore is immutable and hashes by value, not id.

  The graphs run in float32 and compare some inputs against thresholds in
  float64, as the exports do; enable jax_enable_x64 to reproduce them exactly.

  Attributes:
    machine: The machine the network was trained for.
    version: The version of the network, e.g. "4.1.2".
    path: The graph npz, if not one distributed with the package.
    input_labels: The order of the inputs expected by `predict`.
    output_labels: The predicted fluxes.
    tglf_MASS_1: The electron mass in units of the reference mass of the
      networks' gyro-Bohm normalisation (TGLF's MASS_1), or None if the export
      does not record it. The inputs and outputs that depend on the sound
      speed (XNUE, VEXB_SHEAR and the gyro-Bohm fluxes) use this reference.
  """

  def __init__(
      self,
      machine: Literal["step", "multimachine"] = "step",
      version: str | None = "4.1.2",
      path: str | os.PathLike[str] | None = None,
  ):
    """Loads the network.

    Args:
      machine: The machine the network was trained for.
      version: The network version, loaded from the package weights.
      path: A graph npz written by scripts/onnx_to_graph_npz.py, loaded instead
        of the package weights. `version` is then taken from the file.
    """
    if path is None:
      arrays = loader.load(machine, version)
    else:
      path = os.fspath(path)
      with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    if str(arrays.get("format", "")) != GRAPH_FORMAT:
      raise ValueError(
          f"Network {machine} {version or path} is not stored as an ONNX graph"
          " (export it with scripts/onnx_to_graph_npz.py)"
      )

    self.machine = machine
    self.version = str(arrays["version"])
    self.path = path
    self.input_labels = tuple(str(label) for label in arrays["input_labels"])
    self.output_labels = tuple(str(label) for label in arrays["output_labels"])
    self.tglf_MASS_1 = (
        float(arrays["tglf_MASS_1"]) if "tglf_MASS_1" in arrays else None
    )
    self._graphs = {
        label: graph_lib.Graph.from_arrays(arrays, label)
        for label in self.output_labels
    }
    self._frozen = True

  def predict(self, inputs: jax.Array) -> Mapping[str, jax.Array]:
    """Predicts the mean and variance of each flux.

    Args:
      inputs: Raw (unnormalised) inputs of shape (..., n_inputs), ordered as
        `input_labels`.

    Returns:
      For each label in `output_labels`, an array of shape (..., 2) holding the
      mean [GB] and variance [GB^2], with the dtype of `inputs`.
    """
    inputs = jnp.asarray(inputs)
    if inputs.shape[-1] != len(self.input_labels):
      raise ValueError(
          f"Expected {len(self.input_labels)} inputs {self.input_labels}, got"
          f" shape {inputs.shape}"
      )
    batch_shape = inputs.shape[:-1]
    # The graphs take a (batch, n_inputs) float32 matrix.
    flat_inputs = inputs.reshape(-1, inputs.shape[-1]).astype(jnp.float32)
    predictions = {}
    for label, graph in self._graphs.items():
      mean, variance = graph(flat_inputs)
      predictions[label] = (
          jnp.concatenate([mean, variance], axis=-1)
          .reshape(batch_shape + (2,))
          .astype(inputs.dtype)
      )
    return predictions

  def __hash__(self) -> int:
    return hash((self.machine, self.version, self.path))

  def __eq__(self, other: Any) -> bool:
    return (
        isinstance(other, TGLFNNukaeaGraphModel)
        and self.machine == other.machine
        and self.version == other.version
        and self.path == other.path
    )

  def __setattr__(self, attr, value):
    if getattr(self, "_frozen", False):
      raise AttributeError("TGLFNNukaeaGraphModel is immutable.")
    super().__setattr__(attr, value)
