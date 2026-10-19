"""Tests of the ONNX graph interpreter and the JAX model against onnxruntime.

The references (tglfnn_ukaea/reference) hold onnxruntime outputs of the
original exports and finite-difference derivatives of the float64-converted
graphs; see scripts/make_onnx_reference.py.

Run with `python -m unittest discover tests` (the JAX tests need the `jax`
extra).
"""

import unittest

import numpy as np

import tglfnn_ukaea
from tglfnn_ukaea import graph as graph_lib

try:
  import jax
  import jax.numpy as jnp
  from tglfnn_ukaea import model as model_lib
except ImportError:
  jax = None

GRAPH_VERSIONS = [
    (machine, version)
    for machine in tglfnn_ukaea.DEFAULT_VERSIONS
    for version in tglfnn_ukaea.available_versions(machine)
    if str(tglfnn_ukaea.load(machine, version).get("format", ""))
    == "onnx-graph"
]


def _assert_matches_reference(test, predictions, reference, labels, rtol):
  """(mean, variance) per flux agree to rtol of the output's largest value."""
  for label in labels:
    expected = reference[label]
    actual = np.asarray(predictions[label], dtype=np.float64)
    for i, name in enumerate(("mean", "variance")):
      scale = np.abs(expected[:, i]).max()
      with test.subTest(flux=label, output=name):
        np.testing.assert_allclose(
            actual[:, i], expected[:, i], rtol=rtol, atol=rtol * scale
        )


class GraphTest(unittest.TestCase):

  def test_graph_versions_present(self):
    self.assertIn(("step", "4.1.2"), GRAPH_VERSIONS)

  def test_numpy_matches_onnxruntime(self):
    for machine, version in GRAPH_VERSIONS:
      arrays = tglfnn_ukaea.load(machine, version)
      reference = tglfnn_ukaea.load_reference(machine, version)
      predictions = {}
      for label in arrays["output_labels"]:
        graph = graph_lib.Graph.from_arrays(arrays, str(label))
        mean, variance = graph(reference["inputs"])
        self.assertEqual(mean.dtype, np.float32)
        predictions[str(label)] = np.concatenate([mean, variance], axis=-1)
      _assert_matches_reference(
          self, predictions, reference, predictions, rtol=1e-5
      )

  def test_unsupported_op_fails(self):
    arrays = tglfnn_ukaea.load("step", "4.1.2")
    graph = graph_lib.Graph.from_arrays(arrays, "efe_gb")
    nodes = (dict(graph.nodes[0], op="NotAnOp"),) + graph.nodes[1:]
    broken = graph_lib.Graph(
        nodes, graph.tensors, graph.input_name, graph.output_names, graph.opset
    )
    with self.assertRaisesRegex(NotImplementedError, "NotAnOp"):
      broken.check()


@unittest.skipIf(jax is None, "JAX is not installed")
class GraphModelTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    # The compiled graphs are large; free them between tests.
    self.addCleanup(jax.clear_caches)
    self.addCleanup(jax.config.update, "jax_enable_x64", jax.config.x64_enabled)
    jax.config.update("jax_enable_x64", True)

  def _per_point(self, model):
    """inputs (n_in,) -> predictions (n_out, 2)."""

    def per_point(x):
      predictions = model.predict(x)
      return jnp.stack([predictions[label] for label in model.output_labels])

    return per_point

  def test_matches_onnxruntime(self):
    for machine, version in GRAPH_VERSIONS:
      model = model_lib.TGLFNNukaeaGraphModel(machine, version)
      reference = tglfnn_ukaea.load_reference(machine, version)
      for dtype in (jnp.float32, jnp.float64):
        inputs = jnp.asarray(reference["inputs"], dtype=dtype)
        predictions = jax.jit(model.predict)(inputs)
        for label in model.output_labels:
          self.assertEqual(predictions[label].dtype, dtype)
        _assert_matches_reference(
            self, predictions, reference, model.output_labels, rtol=1e-5
        )

  def test_matches_onnxruntime_without_x64(self):
    jax.config.update("jax_enable_x64", False)
    model = model_lib.TGLFNNukaeaGraphModel("step", "4.1.2")
    reference = tglfnn_ukaea.load_reference("step", "4.1.2")
    predictions = jax.jit(model.predict)(reference["inputs"])
    _assert_matches_reference(
        self, predictions, reference, model.output_labels, rtol=1e-5
    )

  def test_batch_shapes(self):
    model = model_lib.TGLFNNukaeaGraphModel("step", "4.1.2")
    inputs = jnp.asarray(
        tglfnn_ukaea.load_reference("step", "4.1.2")["inputs"][:12]
    )
    flat = model.predict(inputs)
    for shape in ((12,), (3, 4), (2, 3, 2)):
      predictions = model.predict(inputs.reshape(shape + (inputs.shape[-1],)))
      for label in model.output_labels:
        self.assertEqual(predictions[label].shape, shape + (2,))
        np.testing.assert_array_equal(
            predictions[label].reshape(12, 2), flat[label]
        )
    single = model.predict(inputs[0])
    vmapped = jax.vmap(model.predict)(inputs)
    for label in model.output_labels:
      self.assertEqual(single[label].shape, (2,))
      np.testing.assert_allclose(vmapped[label], flat[label], rtol=1e-6)

  def test_jacobian_matches_finite_differences(self):
    for machine, version in GRAPH_VERSIONS:
      model = model_lib.TGLFNNukaeaGraphModel(machine, version)
      reference = tglfnn_ukaea.load_reference(machine, version)
      per_point = self._per_point(model)
      inputs = jnp.asarray(reference["inputs"], dtype=jnp.float64)
      for name, transform in (("jacfwd", jax.jacfwd), ("jacrev", jax.jacrev)):
        jacobian = np.asarray(jax.jit(jax.vmap(transform(per_point)))(inputs))
        for i, label in enumerate(model.output_labels):
          expected = reference[f"{label}_jacobian"]
          valid = reference[f"{label}_jacobian_valid"]
          # Relative to the largest derivative of the same output; where the
          # output is flat (a heat flux floored at zero), absolute.
          scale = np.abs(expected).max(axis=-1, keepdims=True)
          error = np.abs(jacobian[:, i] - expected) / np.maximum(
              np.abs(expected) + scale, 1e-12
          )
          with self.subTest(version=version, transform=name, flux=label):
            self.assertGreater(valid.mean(), 0.5)
            # The reference has ~1e-4 relative noise where it evaluates Erf in
            # float32 (pfi_gb), and is otherwise accurate to ~1e-7.
            self.assertLess(error[valid].max(), 5e-3)
            self.assertLess(np.median(error[valid]), 1e-4)

  def test_gradients_finite_far_from_training_data(self):
    model = model_lib.TGLFNNukaeaGraphModel("step", "4.1.2")
    arrays = tglfnn_ukaea.load("step", "4.1.2")
    mean = arrays["input_mean"].mean(axis=0)
    std = arrays["input_std"].max(axis=0)
    rng = np.random.default_rng(0)
    inputs = mean + std * rng.uniform(-6, 6, (256, len(mean)))
    # Inputs at zero, and positive quantities at zero or slightly negative.
    inputs[:64] = mean
    inputs[np.arange(64), rng.integers(0, len(mean), 64)] = 0.0
    inputs[64:128, model.input_labels.index("XNUE")] = rng.choice(
        [0.0, 1e-12, -1e-3], 64
    )
    inputs[128:192, model.input_labels.index("BETAE")] = rng.choice(
        [0.0, 1e-12, -1e-3], 64
    )
    per_point = self._per_point(model)
    inputs = jnp.asarray(inputs)
    self.assertTrue(np.isfinite(jax.vmap(per_point)(inputs)).all())
    for transform in (jax.jacfwd, jax.jacrev):
      jacobian = jax.jit(jax.vmap(transform(per_point)))(inputs)
      self.assertTrue(np.isfinite(jacobian).all())

  def test_hash_and_immutability(self):
    a = model_lib.TGLFNNukaeaGraphModel("step", "4.1.2")
    b = model_lib.TGLFNNukaeaGraphModel("step", "4.1.2")
    self.assertEqual(a, b)
    self.assertEqual(hash(a), hash(b))
    with self.assertRaises(AttributeError):
      a.version = "0"

  def test_parameter_format_rejected(self):
    with self.assertRaisesRegex(ValueError, "not stored as an ONNX graph"):
      model_lib.TGLFNNukaeaGraphModel("step", "1.0.1")


if __name__ == "__main__":
  unittest.main()
