"""Generate reference outputs of the ONNX models, distributed as test data.

For each distributed network version, this evaluates the original ONNX models
with onnxruntime on random inputs and writes the inputs and outputs to
tglfnn_ukaea/reference/{machine}/{version}.npz. Implementations of the networks
(e.g. using the parameters from tglfnn_ukaea.load) can be tested against them,
using tglfnn_ukaea.load_reference.

Each npz contains:

    version            ()                      network version, e.g. "3.2.1"
    model_name         ()                      model directory
    input_labels       (n_in,)                 input order
    output_labels      (n_out,)                fluxes, e.g. ["efe_gb", "efi_gb", "pfi_gb"]
    inputs             (num_samples, n_in)  f32
    {flux}             (num_samples, 2)     f64   ONNX (mean, variance)

For networks stored as ONNX graphs (scripts/onnx_to_graph_npz.py), it also
contains the derivatives of the outputs with respect to the inputs:

    {flux}_jacobian        (num_samples, 2, n_in)  f64  d(mean, variance)/d inputs
    {flux}_jacobian_valid  (num_samples, 2, n_in)  bool

The derivatives are central finite differences of the ONNX graph converted to
float64 and run with onnxruntime, so they are independent of any
implementation under test. Entries where two step sizes disagree (a clip or
branch boundary lies within the stencil) are marked invalid.

The inputs are drawn uniformly within +/- 2 standard deviations of every flux's
input normalisation (for graph networks, with positive inputs kept positive).

Usage (from the repository root, with the package and `export` extra installed):
    python scripts/make_onnx_reference.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import TensorProto
import onnxruntime as ort

import tglfnn_ukaea
from tglfnn_ukaea.loader import REFERENCE_DIR
import onnx_to_graph_npz

REPO_DIR = pathlib.Path(__file__).resolve().parent.parent
NUM_SAMPLES = 64
# Finite difference step, relative to each input's normalisation std.
JACOBIAN_STEP = 1e-3
# Relative agreement between the derivatives with steps h and h/2 below which
# an entry is valid.
JACOBIAN_AGREEMENT = 1e-4


def _session(model):
  model = model if isinstance(model, bytes) else str(model)
  return ort.InferenceSession(model, providers=["CPUExecutionProvider"])


def _check_not_lfs_pointer(path):
  if path.read_bytes()[:40].startswith(b"version https://git-lfs"):
    raise SystemExit(f"{path} is a git-lfs pointer; run `git lfs pull` first")


def to_float64(model):
  """`model` with its float32 computation done in float64.

  onnxruntime has no float64 Softplus or Erf: Softplus is rewritten as
  relu(x) + log(1 + exp(-|x|)), and Erf is evaluated in float32.
  """
  graph = model.graph
  for i, tensor in enumerate(graph.initializer):
    if tensor.data_type == TensorProto.FLOAT:
      graph.initializer[i].CopyFrom(numpy_helper.from_array(
          numpy_helper.to_array(tensor).astype(np.float64), tensor.name
      ))
  for value in list(graph.input) + list(graph.output):
    if value.type.tensor_type.elem_type == TensorProto.FLOAT:
      value.type.tensor_type.elem_type = TensorProto.DOUBLE
  del graph.value_info[:]

  nodes = []
  for node in graph.node:
    for attribute in node.attribute:
      if (
          attribute.type == onnx.AttributeProto.TENSOR
          and attribute.t.data_type == TensorProto.FLOAT
      ):
        attribute.t.CopyFrom(numpy_helper.from_array(
            numpy_helper.to_array(attribute.t).astype(np.float64),
            attribute.t.name,
        ))
      if node.op_type == "Cast" and attribute.name == "to":
        if attribute.i == TensorProto.FLOAT:
          attribute.i = TensorProto.DOUBLE
    if node.op_type == "Softplus":
      x, y = node.input[0], node.output[0]
      p = y + "/f64"
      graph.initializer.append(
          numpy_helper.from_array(np.ones((), np.float64), p + "/one")
      )
      nodes += [
          helper.make_node("Relu", [x], [p + "/relu"]),
          helper.make_node("Abs", [x], [p + "/abs"]),
          helper.make_node("Neg", [p + "/abs"], [p + "/neg"]),
          helper.make_node("Exp", [p + "/neg"], [p + "/exp"]),
          helper.make_node("Add", [p + "/exp", p + "/one"], [p + "/sum"]),
          helper.make_node("Log", [p + "/sum"], [p + "/log"]),
          helper.make_node("Add", [p + "/relu", p + "/log"], [y]),
      ]
    elif node.op_type == "Erf":
      x, y = node.input[0], node.output[0]
      p = y + "/f64"
      nodes += [
          helper.make_node("Cast", [x], [p + "/x"], to=TensorProto.FLOAT),
          helper.make_node("Erf", [p + "/x"], [p + "/erf"]),
          helper.make_node("Cast", [p + "/erf"], [y], to=TensorProto.DOUBLE),
      ]
    else:
      nodes.append(node)
  del graph.node[:]
  graph.node.extend(nodes)
  return model


def jacobian(session, inputs, steps):
  """Central finite differences of a float64 session's (mean, variance).

  Returns:
    The derivatives with steps `steps` and `steps / 2`, each of shape
    (num_samples, 2, n_in).
  """
  num_samples, num_inputs = inputs.shape
  results = []
  for scale in (1.0, 0.5):
    h = steps * scale
    # For each sample, +h and -h along each input.
    offsets = np.concatenate([np.diag(h), -np.diag(h)])  # (2 n_in, n_in)
    stencil = (inputs[:, None, :].astype(np.float64) + offsets[None]).reshape(
        -1, num_inputs
    )
    mean, variance = session.run(None, {session.get_inputs()[0].name: stencil})
    outputs = np.stack([mean[:, 0], variance[:, 0]], axis=-1).reshape(
        num_samples, 2, num_inputs, 2
    )  # (sample, +/-, input, output)
    derivative = (outputs[:, 0] - outputs[:, 1]) / (2 * h[:, None])
    results.append(derivative.transpose(0, 2, 1))
  return results


def make_reference(machine, version, seed=0):
  """The ONNX models' outputs for `machine` and `version` on random inputs."""
  params = tglfnn_ukaea.load(machine, version)
  model_dir = REPO_DIR / str(params["model_name"])
  is_graph = str(params.get("format", "")) == onnx_to_graph_npz.FORMAT

  if is_graph:
    inputs = onnx_to_graph_npz.sample_inputs(params, NUM_SAMPLES, seed)
  else:
    mean, std = params["input_mean"], params["input_std"]
    low = (mean - 2 * std).min(axis=0)
    high = (mean + 2 * std).max(axis=0)
    rng = np.random.default_rng(seed)
    inputs = rng.uniform(low, high, (NUM_SAMPLES, len(low))).astype(np.float32)

  reference = {
      "version": params["version"],
      "model_name": params["model_name"],
      "input_labels": params["input_labels"],
      "output_labels": params["output_labels"],
      "inputs": inputs,
  }
  for i, flux in enumerate(params["output_labels"]):
    if is_graph:
      path = model_dir / str(params["onnx_files"][i])
    else:
      path = model_dir / f"regressor_{flux}_onnx.onnx"
    _check_not_lfs_pointer(path)
    session = _session(path)
    flux_mean, flux_variance = session.run(
        None, {session.get_inputs()[0].name: inputs}
    )
    reference[str(flux)] = np.stack(
        [flux_mean[:, 0], flux_variance[:, 0]], axis=-1
    )

    if is_graph:
      session64 = _session(to_float64(onnx.load(str(path))).SerializeToString())
      steps = JACOBIAN_STEP * params["input_std"][i]
      fine, coarse = jacobian(session64, inputs, steps)
      # Entries much smaller than the largest derivative of the same output
      # are compared absolutely.
      row_scale = np.abs(fine).max(axis=-1, keepdims=True)
      reference[f"{flux}_jacobian"] = fine
      reference[f"{flux}_jacobian_valid"] = np.abs(fine - coarse) <= (
          JACOBIAN_AGREEMENT * (np.abs(fine) + row_scale)
      )
  return reference


if __name__ == "__main__":
  for machine in tglfnn_ukaea.DEFAULT_VERSIONS:
    for version in tglfnn_ukaea.available_versions(machine):
      output = REFERENCE_DIR / machine / f"{version}.npz"
      output.parent.mkdir(parents=True, exist_ok=True)
      with open(output, "wb") as f:
        np.savez(f, **make_reference(machine, version))
      print(f"wrote {output}")
