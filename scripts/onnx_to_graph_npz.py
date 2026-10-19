"""Export TGLFNN-ukaea networks from their ONNX files to a graph npz.

Unlike scripts/onnx_to_npz.py, which extracts the parameters of one known
architecture (ParallelDeepEnsemble), this stores the ONNX graph itself: the
node list and every tensor it uses. tglfnn_ukaea.graph.Graph evaluates it node
by node with the ONNX operator semantics, so any network built from the
supported operators is reproduced exactly, including preprocessing, output
transforms and uncertainty corrections embedded in the export.

The npz contains only plain arrays (no pickled objects):

    format             ()               "onnx-graph"
    version            ()               network version, e.g. "4.1.2"
    model_name         ()               model directory
    input_labels       (n_in,)          input order expected by the networks
    output_labels      (n_out,)         fluxes, e.g. ["efe_gb", "efi_gb", "pfi_gb"]
    input_mean         (n_out, n_in) f64  training input mean (config.yaml);
    input_std          (n_out, n_in) f64  informative only, the graph scales itself
    onnx_files         (n_out,)         source file of each output's graph
    onnx_sha256        (n_out,)
    onnx_producer      ()
    tglf_settings_json ()               reference TGLF settings, verbatim (if present)
    tglf_MASS_1        ()         f64   electron mass in units of the reference
                                        mass of the gyro-Bohm normalisation
                                        (tglf_settings.json "fixed"), if present
    {flux}/graph       ()               JSON: input, outputs, opset, nodes
    {flux}/tensor/{name}                each initializer, Constant output and
                                        tensor-valued attribute
    {flux}/config_yaml ()               training config.yaml, verbatim

Each graph maps raw float32 inputs (batch, n_in) to ("mean", "variance"), each
(batch, 1). Constant nodes are folded into tensors; every other node is kept.

The written npz is evaluated with tglfnn_ukaea.graph (NumPy) and compared
against onnxruntime (skip with --no-verify). Run with the package and the
`export` extra installed, from the repository root:

    pip install -e .[export]
    python scripts/onnx_to_graph_npz.py 4.1.2 -o tglfnn_ukaea/weights/step/4.1.2.npz
"""

import argparse
import hashlib
import json
import pathlib

import numpy as np
import onnx
from onnx import numpy_helper
import yaml

from tglfnn_ukaea import graph as graph_lib

FORMAT = "onnx-graph"
DEFAULT_FLUXES = ("efe_gb", "efi_gb", "pfi_gb")
OUTPUT_NAMES = ("mean", "variance")

# Inputs that are physically positive, and are sampled as such when verifying.
POSITIVE_INPUTS = {
    "TAUS_2", "RMIN_LOC", "Q_LOC", "XNUE", "KAPPA_LOC", "ZEFF", "BETAE",
}


def _attribute_value(attribute):
  value = onnx.helper.get_attribute_value(attribute)
  if isinstance(value, bytes):
    return value.decode()
  if isinstance(value, (list, tuple)):
    return [v.decode() if isinstance(v, bytes) else v for v in value]
  return value


def _constant_value(node):
  """The tensor of a Constant node, whichever attribute holds it."""
  if len(node.attribute) != 1:
    raise ValueError(f"Constant {node.output[0]} has {len(node.attribute)} attributes")
  attribute = node.attribute[0]
  value = onnx.helper.get_attribute_value(attribute)
  if attribute.name == "value":
    return numpy_helper.to_array(value)
  dtypes = {
      "value_float": np.float32, "value_floats": np.float32,
      "value_int": np.int64, "value_ints": np.int64,
  }
  if attribute.name not in dtypes:
    raise ValueError(f"Unsupported Constant attribute {attribute.name}")
  return np.asarray(value, dtype=dtypes[attribute.name])


def extract_graph(path):
  """The graph spec and tensors of the ONNX file at `path`."""
  model = onnx.load(str(path))
  graph = model.graph
  opsets = {o.domain: o.version for o in model.opset_import}
  if set(opsets) - {"", "ai.onnx"}:
    raise ValueError(f"{path}: unsupported operator domains {set(opsets)}")
  opset = opsets.get("", opsets.get("ai.onnx"))

  initializer_names = {i.name for i in graph.initializer}
  inputs = [i.name for i in graph.input if i.name not in initializer_names]
  if len(inputs) != 1:
    raise ValueError(f"{path}: expected one graph input, got {inputs}")
  outputs = tuple(o.name for o in graph.output)
  if outputs != OUTPUT_NAMES:
    raise ValueError(f"{path}: expected outputs {OUTPUT_NAMES}, got {outputs}")

  tensors = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
  nodes = []
  for node in graph.node:
    if node.domain not in ("", "ai.onnx"):
      raise ValueError(f"{path}: op {node.op_type} in domain {node.domain}")
    if node.op_type == "Constant":
      tensors[node.output[0]] = _constant_value(node)
      continue
    attrs = {}
    for attribute in node.attribute:
      if attribute.type == onnx.AttributeProto.TENSOR:
        key = f"{node.output[0]}::{attribute.name}"
        tensors[key] = numpy_helper.to_array(attribute.t)
        attrs[attribute.name] = {"tensor": key}
      elif attribute.type in (
          onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS
      ):
        raise ValueError(f"{path}: {node.op_type} has a subgraph attribute")
      else:
        attrs[attribute.name] = _attribute_value(attribute)
    nodes.append({
        "op": node.op_type,
        "inputs": list(node.input),
        "outputs": list(node.output),
        "attrs": attrs,
    })

  spec = {"input": inputs[0], "outputs": list(outputs), "opset": opset, "nodes": nodes}
  return spec, tensors, model.producer_name + " " + model.producer_version


def export(model_dir, fluxes):
  """The npz arrays for the networks in `model_dir`."""
  model_dir = pathlib.Path(model_dir)
  arrays = {}
  input_labels = None
  input_means, input_stds, files, hashes, producers = [], [], [], [], set()
  for flux in fluxes:
    path = model_dir / f"regressor_{flux}.onnx"
    if path.read_bytes()[:40].startswith(b"version https://git-lfs"):
      raise SystemExit(f"{path} is a git-lfs pointer; run `git lfs pull` first")
    config_text = (model_dir / flux / "config.yaml").read_text()
    config = yaml.safe_load(config_text)

    labels = tuple(config["interface"]["inputs"])
    if input_labels is None:
      input_labels = labels
    elif labels != input_labels:
      raise ValueError(f"{flux} expects inputs {labels}, not {input_labels}")
    scaling = config["members"][0]["scaling"]
    for member in config["members"][1:]:
      if member["scaling"]["x_mean"] != scaling["x_mean"]:
        raise ValueError(f"{flux}: ensemble members have different scalings")
    input_means.append(scaling["x_mean"])
    input_stds.append(scaling["x_std"])

    spec, tensors, producer = extract_graph(path)
    arrays[graph_lib.GRAPH_KEY.format(flux=flux)] = np.array(json.dumps(spec))
    prefix = graph_lib.TENSOR_PREFIX.format(flux=flux)
    for name, value in tensors.items():
      arrays[prefix + name] = value
    arrays[f"{flux}/config_yaml"] = np.array(config_text)
    files.append(path.name)
    hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    producers.add(producer)

  metadata = {}
  settings_path = model_dir / "tglf_settings.json"
  if settings_path.exists():
    settings_text = settings_path.read_text()
    metadata["tglf_settings_json"] = np.array(settings_text)
    fixed = json.loads(settings_text).get("fixed", {})
    if "MASS_1" in fixed:
      metadata["tglf_MASS_1"] = np.array(float(fixed["MASS_1"]))

  return {
      "format": np.array(FORMAT),
      "version": np.array(str(config["version"])),
      "model_name": np.array(model_dir.resolve().name),
      "input_labels": np.array(input_labels),
      "output_labels": np.array(fluxes),
      "input_mean": np.array(input_means, dtype=np.float64),
      "input_std": np.array(input_stds, dtype=np.float64),
      "onnx_files": np.array(files),
      "onnx_sha256": np.array(hashes),
      "onnx_producer": np.array(", ".join(sorted(producers))),
      **metadata,
      **arrays,
  }


def sample_inputs(arrays, num_samples, seed=0):
  """Random float32 inputs within +/- 2 std of the training input mean.

  Physically positive inputs are kept above a tenth of their mean.
  """
  mean, std = arrays["input_mean"], arrays["input_std"]
  low = (mean - 2 * std).min(axis=0)
  high = (mean + 2 * std).max(axis=0)
  for i, label in enumerate(arrays["input_labels"]):
    if str(label) in POSITIVE_INPUTS:
      low[i] = max(low[i], mean[:, i].min() / 10)
  rng = np.random.default_rng(seed)
  return rng.uniform(low, high, (num_samples, len(low))).astype(np.float32)


def verify(arrays, model_dir, num_samples=256):
  """Compares the npz graphs, run with NumPy, against onnxruntime."""
  import onnxruntime as ort  # pylint: disable=g-import-not-at-top

  inputs = sample_inputs(arrays, num_samples)
  for flux, file in zip(arrays["output_labels"], arrays["onnx_files"]):
    session = ort.InferenceSession(
        str(pathlib.Path(model_dir) / str(file)),
        providers=["CPUExecutionProvider"],
    )
    expected = session.run(None, {session.get_inputs()[0].name: inputs})
    graph = graph_lib.Graph.from_arrays(arrays, str(flux))
    actual = graph(inputs)
    for name, a, e in zip(OUTPUT_NAMES, actual, expected):
      if a.dtype != e.dtype or a.shape != e.shape:
        raise AssertionError(
            f"{flux} {name}: {a.dtype}{a.shape} != onnxruntime {e.dtype}{e.shape}"
        )
      scale = np.abs(e).max()
      err = np.abs(a.astype(np.float64) - e).max()
      print(f"  {flux} {name}: max |diff| = {err:.3g} (max |value| {scale:.3g})")
      np.testing.assert_allclose(a, e, rtol=1e-4, atol=1e-5 * scale)


def main():
  parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  parser.add_argument("model_dir", type=pathlib.Path)
  parser.add_argument("-o", "--output", type=pathlib.Path, required=True)
  parser.add_argument("--fluxes", nargs="+", default=DEFAULT_FLUXES)
  parser.add_argument("--no-verify", action="store_true")
  args = parser.parse_args()

  arrays = export(args.model_dir, tuple(args.fluxes))
  if not args.no_verify:
    print(f"Verifying {args.model_dir} against onnxruntime")
    verify(arrays, args.model_dir)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  with open(args.output, "wb") as f:
    np.savez(f, **arrays)
  print(f"wrote {args.output}")


if __name__ == "__main__":
  main()
