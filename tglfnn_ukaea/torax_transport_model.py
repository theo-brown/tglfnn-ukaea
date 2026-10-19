"""TORAX transport model for the TGLFNN-ukaea networks stored as ONNX graphs.

Based on TORAX's `tglfnn-ukaea` transport model, but evaluates the networks
with tglfnn_ukaea.model.TGLFNNukaeaGraphModel, i.e. exactly as exported, with
JAX derivatives for TORAX's solvers. The network inputs are assembled from
TORAX's TGLF inputs by label, so any graph network with the inputs in
`_INPUTS` can be used.

Written against TORAX 1.4.x. Register the model before building the config:

    import tglfnn_ukaea.torax_transport_model as tglfnn_graph
    tglfnn_graph.register()

and select it with

    "transport": {"model_name": "tglfnn-ukaea-graph", "version": "4.1.2", ...}

or `"path": <graph npz>` for a network not distributed with the package.

Gyro-Bohm reference mass: TORAX normalises its TGLF inputs and fluxes with the
deuterium mass (c_s = sqrt(T_e / m_D)), but the networks may be trained with a
different reference mass m_ref (the 4.1.x networks use the proton mass), which
the export records as TGLF's MASS_1 = m_e / m_ref. XNUE and VEXB_SHEAR
(proportional to 1 / c_s) and the gyro-Bohm flux unit Q_GB (proportional to
c_s rho_s^2, i.e. sqrt(m_ref)) all scale with sqrt(m_ref), so the inputs are
converted to the network's normalisation, and the predicted fluxes back to
TORAX's, with the factor sqrt(m_ref / m_D) = sqrt(MASS_1_TORAX / MASS_1_net).
"""

from __future__ import annotations

import dataclasses
from typing import Annotated, Callable, Literal

import chex
import jax
import jax.numpy as jnp
import pydantic
from torax._src import state
from torax._src.config import runtime_params as runtime_params_lib
from torax._src.geometry import geometry
from torax._src.pedestal_model import pedestal_model_output as pedestal_model_output_lib
from torax._src.torax_pydantic import torax_pydantic
from torax._src.transport_model import pydantic_model_base
from torax._src.transport_model import register_model
from torax._src.transport_model import tglf_based_transport_model
from torax._src.transport_model import transport_model as transport_model_lib

from tglfnn_ukaea import model as model_lib

MODEL_NAME = "tglfnn-ukaea-graph"

_TGLFInputs = tglf_based_transport_model.TGLFInputs

# Network inputs, by label, computed from TORAX's TGLF inputs (in TORAX's
# deuterium gyro-Bohm normalisation; see _MASS_DEPENDENT_INPUTS).
_INPUTS: dict[str, Callable[[_TGLFInputs], jax.Array]] = {
    "RLNS_1": lambda t: t.RLNS_1,
    "RLTS_1": lambda t: t.RLTS_1,
    "RLTS_2": lambda t: t.RLTS_2,
    "TAUS_2": lambda t: t.TAUS_2,
    "RMIN_LOC": lambda t: t.RMIN_LOC,
    "DRMAJDX_LOC": lambda t: t.DRMAJDX_LOC,
    # The networks are trained on |q| (as in the JETTO TCI adapter).
    "Q_LOC": lambda t: jnp.abs(t.Q_LOC),
    # TGLFNN-ukaea's magnetic shear r/q dq/dr. This is not S_HAT_LOC in s-alpha
    # geometry.
    "SHAT": lambda t: (t.RMIN_LOC / t.Q_LOC) ** 2 * t.Q_PRIME_LOC,
    "XNUE": lambda t: t.XNUE,
    "KAPPA_LOC": lambda t: t.KAPPA_LOC,
    "S_KAPPA_LOC": lambda t: t.S_KAPPA_LOC,
    "DELTA_LOC": lambda t: t.DELTA_LOC,
    "S_DELTA_LOC": lambda t: t.S_DELTA_LOC,
    "ZEFF": lambda t: t.ZEFF,
    # Includes rotation_multiplier (TGLF's ALPHA_E); zero unless use_rotation.
    "VEXB_SHEAR": lambda t: t.VEXB_SHEAR,
    "BETAE": lambda t: t.BETAE,
}

# Inputs proportional to sqrt(reference mass), via 1 / c_s.
_MASS_DEPENDENT_INPUTS = ("XNUE", "VEXB_SHEAR")


def gyrobohm_mass_factor(
    tglf_inputs: _TGLFInputs, network_MASS_1: float
) -> jax.Array:
  """sqrt(m_ref / m_TORAX): network-normalised = factor * TORAX-normalised.

  Applies to XNUE and VEXB_SHEAR, and to the gyro-Bohm fluxes in the reverse
  direction (flux in TORAX's unit = factor * flux in the network's unit).
  """
  return jnp.sqrt(tglf_inputs.MASS_1 / network_MASS_1)


@dataclasses.dataclass(frozen=True, eq=False)
class TGLFNNukaeaGraphTransportModel(
    tglf_based_transport_model.TGLFBasedTransportModel
):
  """TGLFNN-ukaea transport model, evaluating the exported ONNX graphs."""

  machine: Literal["step", "multimachine"] = "step"
  version: str | None = "4.1.2"
  path: str | None = None
  # Zero fluxes where sqrt(variance)/|mean| exceeds this threshold (as JETTO
  # TCI's TRTGLFFILTER). <= 0 disables. Note that for networks whose variance
  # is a calibrated prediction error (e.g. 4.1.x) rather than an ensemble
  # spread, the threshold has a different meaning than for older networks.
  uncertainty_filter: float = 0.0

  # Set by __post_init__.
  model: model_lib.TGLFNNukaeaGraphModel = dataclasses.field(init=False)

  def __post_init__(self):
    model = model_lib.TGLFNNukaeaGraphModel(
        self.machine, self.version, path=self.path
    )
    missing = [label for label in model.input_labels if label not in _INPUTS]
    if missing:
      raise ValueError(f"No TORAX equivalent for network inputs {missing}")
    for flux in ("efe_gb", "efi_gb", "pfi_gb"):
      if flux not in model.output_labels:
        raise ValueError(f"Network does not predict {flux}")
    if model.tglf_MASS_1 is None:
      raise ValueError(
          "The network export does not record its gyro-Bohm reference mass"
          " (tglf_MASS_1); re-export it with scripts/onnx_to_graph_npz.py from"
          " a bundle with tglf_settings.json"
      )
    object.__setattr__(self, "model", model)
    super().__post_init__()

  def _prepare_network_inputs(self, tglf_inputs: _TGLFInputs) -> jax.Array:
    """The network inputs, in the network's gyro-Bohm normalisation."""
    factor = gyrobohm_mass_factor(tglf_inputs, self.model.tglf_MASS_1)
    inputs = []
    for label in self.model.input_labels:
      value = _INPUTS[label](tglf_inputs)
      if label in _MASS_DEPENDENT_INPUTS:
        value = value * factor
      inputs.append(value)
    return jnp.stack(inputs, axis=-1)

  def call_implementation(
      self,
      transport: tglf_based_transport_model.RuntimeParams,
      runtime_params: runtime_params_lib.RuntimeParams,
      geo: geometry.Geometry,
      core_profiles: state.CoreProfiles,
      pedestal_model_output: pedestal_model_output_lib.PedestalModelOutput,
  ) -> transport_model_lib.TurbulentTransport:
    del pedestal_model_output  # unused
    tglf_inputs = self._prepare_tglf_inputs(
        transport=transport,
        geo=geo,
        core_profiles=core_profiles,
        poloidal_velocity_multiplier=runtime_params.neoclassical.poloidal_velocity_multiplier,
    )
    predictions = self.model.predict(self._prepare_network_inputs(tglf_inputs))

    # Converts the fluxes from the network's gyro-Bohm unit to TORAX's.
    factor = gyrobohm_mass_factor(tglf_inputs, self.model.tglf_MASS_1)

    def _filtered_mean(prediction: jax.Array) -> jax.Array:
      mean = prediction[..., 0]
      if self.uncertainty_filter > 0.0:
        std = jnp.sqrt(jnp.abs(prediction[..., 1]))
        relative_uncertainty = std / jnp.clip(jnp.abs(mean), 1e-10, None)
        mean = jnp.where(
            relative_uncertainty > self.uncertainty_filter, 0.0, mean
        )
      return mean * factor

    return self._make_core_transport(
        ion_heat_flux_GB=_filtered_mean(predictions["efi_gb"]),
        electron_heat_flux_GB=_filtered_mean(predictions["efe_gb"]),
        # The networks predict the (effective) ion particle flux, which equals
        # the electron particle flux for a single effective ion.
        electron_particle_flux_GB=_filtered_mean(predictions["pfi_gb"]),
        tglf_inputs=tglf_inputs,
        transport=transport,
        geo=geo,
        core_profiles=core_profiles,
    )


class TGLFNNukaeaGraphTransportConfig(pydantic_model_base.TransportBase):
  """Config for the TGLFNN-ukaea ONNX graph transport model.

  Attributes:
    model_name: Hardcoded to 'tglfnn-ukaea-graph'.
    machine: The machine the network was trained for.
    version: The network version distributed with tglfnn_ukaea.
    path: A graph npz (scripts/onnx_to_graph_npz.py) used instead of `version`.
    uncertainty_filter: Zero fluxes where sqrt(variance)/|mean| exceeds this.
      0 disables.
  """

  model_name: Annotated[
      Literal["tglfnn-ukaea-graph"], torax_pydantic.JAX_STATIC
  ] = "tglfnn-ukaea-graph"
  machine: Annotated[
      Literal["step", "multimachine"], torax_pydantic.JAX_STATIC
  ] = "step"
  version: Annotated[str | None, torax_pydantic.JAX_STATIC] = "4.1.2"
  path: Annotated[str | None, torax_pydantic.JAX_STATIC] = None
  uncertainty_filter: Annotated[
      pydantic.NonNegativeFloat, torax_pydantic.JAX_STATIC
  ] = 0.0
  rotation_multiplier: pydantic.NonNegativeFloat = 1.0
  use_rotation: Annotated[bool, torax_pydantic.JAX_STATIC] = False
  # Quasilinear transport options
  DV_effective: bool = False
  An_min: pydantic.PositiveFloat = 0.05
  collisionality_multiplier: float = 1.0

  def build_transport_model(self) -> TGLFNNukaeaGraphTransportModel:
    return TGLFNNukaeaGraphTransportModel(
        machine=self.machine,
        version=self.version,
        path=self.path,
        uncertainty_filter=self.uncertainty_filter,
    )

  def build_runtime_params(
      self, t: chex.Numeric
  ) -> tglf_based_transport_model.RuntimeParams:
    base_kwargs = dataclasses.asdict(super().build_runtime_params(t))
    return tglf_based_transport_model.RuntimeParams(
        DV_effective=self.DV_effective,
        An_min=self.An_min,
        rotation_multiplier=self.rotation_multiplier,
        use_rotation=self.use_rotation,
        collisionality_multiplier=self.collisionality_multiplier,
        **base_kwargs,
    )


_registered = False


def register() -> None:
  """Adds the 'tglfnn-ukaea-graph' transport model to TORAX's config.

  Safe to call more than once.
  """
  global _registered
  if not _registered:
    register_model.register_transport_model(TGLFNNukaeaGraphTransportConfig)
    _registered = True
