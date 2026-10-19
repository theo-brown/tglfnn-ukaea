"""Interpreter for the ONNX graphs of the TGLFNN-ukaea networks.

Networks exported as a graph (see scripts/onnx_to_graph_npz.py) are stored as
the ONNX node list plus every tensor the nodes use, and evaluated here node by
node with the ONNX operator semantics. Nothing about the architecture is
assumed: the interpreter runs whatever the export contains, and fails on any
operator it does not implement.

The same code runs with NumPy (no dependencies beyond the package's own) or
with JAX, where the result is differentiable and can be jitted and vmapped:

    graph = Graph.from_arrays(tglfnn_ukaea.load("step", "4.1.2"), "efe_gb")
    mean, variance = graph(inputs)                    # NumPy
    mean, variance = graph(jnp.asarray(inputs))       # JAX

Values computed only from the stored tensors and the input *shape* (shapes,
axes, indices, clip bounds, ...) are always NumPy arrays, so they stay concrete
under jax.jit. Only values that depend on the input are JAX arrays.

Dtypes follow the graph exactly: constants keep their stored dtype, and each
Cast casts to the dtype it names. The exports run in float32 but compare some
inputs against thresholds in float64; under JAX this needs jax_enable_x64
(otherwise those comparisons are made in float32).
"""

from __future__ import annotations

import dataclasses
import json
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

# Prefixes of the graph entries in the npz, per output: f"{flux}/graph" holds
# the JSON node list and f"{flux}/tensor/{name}" each tensor.
GRAPH_KEY = "{flux}/graph"
TENSOR_PREFIX = "{flux}/tensor/"

# ONNX TensorProto data types used by Cast.
_ONNX_DTYPES = {
    1: np.float32,
    2: np.uint8,
    3: np.int8,
    5: np.int16,
    6: np.int32,
    7: np.int64,
    9: np.bool_,
    10: np.float16,
    11: np.float64,
}


@dataclasses.dataclass(frozen=True, eq=False)
class Graph:
  """An ONNX graph with one input, as stored by onnx_to_graph_npz.py.

  Attributes:
    nodes: The nodes in topological order, each a dict with keys "op",
      "inputs", "outputs" and "attrs". A tensor-valued attribute is stored as
      {"tensor": name}, referring to `tensors`.
    tensors: The initializers, Constant node outputs and tensor attributes.
    input_name: Name of the graph input.
    output_names: Names of the graph outputs.
    opset: Version of the default ONNX operator set.
  """

  nodes: tuple[Mapping[str, Any], ...]
  tensors: Mapping[str, np.ndarray]
  input_name: str
  output_names: tuple[str, ...]
  opset: int

  @classmethod
  def from_arrays(cls, arrays: Mapping[str, np.ndarray], flux: str) -> Graph:
    """The graph of output `flux` from the arrays of a graph npz."""
    spec = json.loads(str(arrays[GRAPH_KEY.format(flux=flux)]))
    prefix = TENSOR_PREFIX.format(flux=flux)
    tensors = {
        key[len(prefix):]: value
        for key, value in arrays.items()
        if key.startswith(prefix)
    }
    graph = cls(
        nodes=tuple(spec["nodes"]),
        tensors=tensors,
        input_name=spec["input"],
        output_names=tuple(spec["outputs"]),
        opset=int(spec["opset"]),
    )
    graph.check()
    return graph

  def check(self) -> None:
    """Checks that every op is implemented and every value is defined."""
    unsupported = sorted({n["op"] for n in self.nodes} - set(OPS))
    if unsupported:
      raise NotImplementedError(f"Unsupported ONNX ops: {unsupported}")
    defined = {self.input_name, "", *self.tensors}
    for node in self.nodes:
      missing = [name for name in node["inputs"] if name not in defined]
      if missing:
        raise ValueError(
            f"{node['op']} node producing {node['outputs']} uses undefined"
            f" values {missing} (nodes must be in topological order)"
        )
      defined.update(node["outputs"])
    missing = [name for name in self.output_names if name not in defined]
    if missing:
      raise ValueError(f"Graph outputs {missing} are never computed")

  def __call__(self, inputs) -> tuple[Any, ...]:
    """Evaluates the graph, returning its outputs in order.

    Args:
      inputs: The graph input, a NumPy or JAX array. It is not cast, so it
        should already have the dtype the graph expects (float32).
    """
    values: dict[str, Any] = dict(self.tensors)
    values[self.input_name] = inputs
    for node in self.nodes:
      args = [values[name] if name else None for name in node["inputs"]]
      attrs = {
          key: self.tensors[value["tensor"]]
          if isinstance(value, dict) and "tensor" in value
          else value
          for key, value in node["attrs"].items()
      }
      xp = _namespace(args)
      outputs = OPS[node["op"]](xp, args, attrs, self.opset)
      if not isinstance(outputs, tuple):
        outputs = (outputs,)
      if len(outputs) < len(node["outputs"]):
        raise NotImplementedError(
            f"{node['op']} only computes {len(outputs)} output(s), but"
            f" {len(node['outputs'])} are requested"
        )
      values.update(zip(node["outputs"], outputs))
    return tuple(values[name] for name in self.output_names)


# Array namespaces. Each op is written once against `xp`, which is NumPy
# unless one of its inputs is a JAX array.


class _NumpyOps:
  """NumPy versions of the functions NumPy does not provide."""

  @staticmethod
  def softplus(x):
    return np.logaddexp(x, np.zeros((), x.dtype))

  @staticmethod
  def erf(x):
    return np.vectorize(math.erf, otypes=[np.float64])(x).astype(x.dtype)

  @staticmethod
  def logsumexp(x, axis, keepdims):
    x_max = np.max(x, axis=axis, keepdims=True)
    x_max = np.where(np.isfinite(x_max), x_max, np.zeros((), x.dtype))
    out = np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    return out if keepdims else np.squeeze(out, axis=axis)

  @staticmethod
  def integer_pow(x, n):
    return x**n


class _Namespace:

  def __init__(self, array_module, special):
    self._array_module = array_module
    self._special = special

  def __getattr__(self, name):
    if hasattr(self._special, name):
      return getattr(self._special, name)
    return getattr(self._array_module, name)


_NUMPY = _Namespace(np, _NumpyOps)
_JAX = None


def _jax_namespace():
  """The JAX namespace, imported on first use so JAX stays optional."""
  global _JAX
  if _JAX is None:
    import jax  # pylint: disable=g-import-not-at-top
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top

    class _JaxOps:
      softplus = staticmethod(jax.nn.softplus)
      erf = staticmethod(jax.scipy.special.erf)

      @staticmethod
      def logsumexp(x, axis, keepdims):
        return jax.nn.logsumexp(x, axis=axis, keepdims=keepdims)

      @staticmethod
      def integer_pow(x, n):
        return jax.lax.integer_pow(x, n)

    _JAX = _Namespace(jnp, _JaxOps)
  return _JAX


def _namespace(args: Sequence[Any]):
  for arg in args:
    if arg is not None and not isinstance(arg, (np.ndarray, np.generic)):
      return _jax_namespace()
  return _NUMPY


def _static(value) -> np.ndarray:
  """A value that must be known when the graph is traced (axes, shapes)."""
  if not isinstance(value, (np.ndarray, np.generic)):
    raise TypeError(
        "Expected a constant (shape, axes or indices), got a value that"
        " depends on the graph input"
    )
  return np.asarray(value)


def _axes(args, attrs, opset, input_from_opset):
  """Reduction/squeeze axes: an attribute before `input_from_opset`, else input 1."""
  if opset >= input_from_opset:
    if len(args) > 1 and args[1] is not None:
      return tuple(int(a) for a in _static(args[1]).ravel())
    return None
  if "axes" in attrs:
    return tuple(int(a) for a in attrs["axes"])
  return None


# Operators. Each takes (xp, args, attrs, opset) and returns the output(s).


def _unary(name):
  return lambda xp, args, attrs, opset: getattr(xp, name)(args[0])


def _binary(fn):
  return lambda xp, args, attrs, opset: fn(xp, args[0], args[1])


def _div(xp, a, b):
  if np.issubdtype(a.dtype, np.integer):
    return xp.trunc(xp.divide(a, b)).astype(a.dtype)
  return xp.divide(a, b)


def _pow(xp, args, attrs, opset):
  base, exponent = args
  if isinstance(exponent, (np.ndarray, np.generic)) and exponent.size == 1:
    n = float(np.asarray(exponent).ravel()[0])
    if n.is_integer():
      # Exact, and with a gradient that is finite for negative bases.
      return xp.integer_pow(base, int(n))
  return xp.power(base, exponent.astype(base.dtype))


def _gemm(xp, args, attrs, opset):
  a, b = args[0], args[1]
  if attrs.get("transA", 0):
    a = a.T
  if attrs.get("transB", 0):
    b = b.T
  out = xp.matmul(a, b)
  if attrs.get("alpha", 1.0) != 1.0:
    out = xp.multiply(out, np.asarray(attrs["alpha"], out.dtype))
  if len(args) > 2 and args[2] is not None:
    c = args[2]
    if attrs.get("beta", 1.0) != 1.0:
      c = xp.multiply(c, np.asarray(attrs["beta"], c.dtype))
    out = xp.add(out, c)
  return out


def _layer_normalization(xp, args, attrs, opset):
  x, scale = args[0], args[1]
  bias = args[2] if len(args) > 2 else None
  if attrs.get("stash_type", 1) != 1:
    raise NotImplementedError("LayerNormalization with stash_type != 1")
  axis = attrs.get("axis", -1)
  axes = tuple(range(axis % x.ndim, x.ndim))
  centred = xp.subtract(x, xp.mean(x, axis=axes, keepdims=True))
  variance = xp.mean(xp.multiply(centred, centred), axis=axes, keepdims=True)
  epsilon = np.asarray(attrs.get("epsilon", 1e-5), dtype=x.dtype)
  out = xp.multiply(xp.divide(centred, xp.sqrt(xp.add(variance, epsilon))), scale)
  return xp.add(out, bias) if bias is not None else out


def _cast(xp, args, attrs, opset):
  return args[0].astype(_ONNX_DTYPES[attrs["to"]])


def _clip(xp, args, attrs, opset):
  x = args[0]
  if len(args) > 1 and args[1] is not None:
    x = xp.maximum(x, args[1])
  if len(args) > 2 and args[2] is not None:
    x = xp.minimum(x, args[2])
  return x


def _concat(xp, args, attrs, opset):
  return xp.concatenate(args, axis=attrs["axis"])


def _gather(xp, args, attrs, opset):
  data, indices = args[0], _static(args[1])
  axis = attrs.get("axis", 0)
  indices = np.where(indices < 0, indices + data.shape[axis], indices)
  return xp.take(data, indices, axis=axis)


def _squeeze(xp, args, attrs, opset):
  axes = _axes(args, attrs, opset, input_from_opset=13)
  return xp.squeeze(args[0], axis=axes)


def _unsqueeze(xp, args, attrs, opset):
  axes = _axes(args, attrs, opset, input_from_opset=13)
  return xp.expand_dims(args[0], axis=axes)


def _reduce(fn, input_from_opset):
  def reduce(xp, args, attrs, opset):
    axes = _axes(args, attrs, opset, input_from_opset)
    if axes is None and attrs.get("noop_with_empty_axes", 0):
      return args[0]
    return fn(xp, args[0], axes, bool(attrs.get("keepdims", 1)))

  return reduce


def _shape(xp, args, attrs, opset):
  shape = np.asarray(args[0].shape, dtype=np.int64)
  return shape[attrs.get("start", 0):attrs.get("end", None)]


def _constant_of_shape(xp, args, attrs, opset):
  value = np.asarray(attrs.get("value", np.zeros(1, np.float32)))
  return np.full(tuple(_static(args[0])), value.ravel()[0], dtype=value.dtype)


def _where(xp, args, attrs, opset):
  return xp.where(*args)


OPS: Mapping[str, Callable[..., Any]] = {
    "Identity": lambda xp, args, attrs, opset: args[0],
    "Abs": _unary("abs"),
    "Neg": _unary("negative"),
    "Exp": _unary("exp"),
    "Log": _unary("log"),
    "Sqrt": _unary("sqrt"),
    "Tanh": _unary("tanh"),
    "Sign": _unary("sign"),
    "Erf": _unary("erf"),
    "Softplus": _unary("softplus"),
    "Relu": lambda xp, args, attrs, opset: xp.maximum(
        args[0], np.zeros((), args[0].dtype)
    ),
    "Add": _binary(lambda xp, a, b: xp.add(a, b)),
    "Sub": _binary(lambda xp, a, b: xp.subtract(a, b)),
    "Mul": _binary(lambda xp, a, b: xp.multiply(a, b)),
    "Div": _binary(_div),
    "Pow": _pow,
    "Less": _binary(lambda xp, a, b: xp.less(a, b)),
    "LessOrEqual": _binary(lambda xp, a, b: xp.less_equal(a, b)),
    "Greater": _binary(lambda xp, a, b: xp.greater(a, b)),
    "GreaterOrEqual": _binary(lambda xp, a, b: xp.greater_equal(a, b)),
    "Equal": _binary(lambda xp, a, b: xp.equal(a, b)),
    "And": _binary(lambda xp, a, b: xp.logical_and(a, b)),
    "Or": _binary(lambda xp, a, b: xp.logical_or(a, b)),
    "Not": lambda xp, args, attrs, opset: xp.logical_not(args[0]),
    "Where": _where,
    "Gemm": _gemm,
    "MatMul": _binary(lambda xp, a, b: xp.matmul(a, b)),
    "LayerNormalization": _layer_normalization,
    "Cast": _cast,
    "Clip": _clip,
    "Concat": _concat,
    "Gather": _gather,
    "Squeeze": _squeeze,
    "Unsqueeze": _unsqueeze,
    "ReduceSum": _reduce(
        lambda xp, x, axes, keepdims: xp.sum(x, axis=axes, keepdims=keepdims),
        input_from_opset=13,
    ),
    "ReduceMean": _reduce(
        lambda xp, x, axes, keepdims: xp.mean(x, axis=axes, keepdims=keepdims),
        input_from_opset=18,
    ),
    "ReduceMax": _reduce(
        lambda xp, x, axes, keepdims: xp.max(x, axis=axes, keepdims=keepdims),
        input_from_opset=18,
    ),
    "ReduceLogSumExp": _reduce(
        lambda xp, x, axes, keepdims: xp.logsumexp(
            x, axis=axes, keepdims=keepdims
        ),
        input_from_opset=18,
    ),
    "Shape": _shape,
    "ConstantOfShape": _constant_of_shape,
}
