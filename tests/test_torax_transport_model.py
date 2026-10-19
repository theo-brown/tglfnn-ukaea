"""Smoke test of the TORAX transport model (needs the `torax` extra)."""

import copy
import unittest

import numpy as np

try:
  import dataclasses
  import jax.numpy as jnp
  import torax
  from torax._src import constants
  from torax._src.transport_model import tglf_based_transport_model
  from torax.examples import iterhybrid_rampup
  import tglfnn_ukaea.torax_transport_model as tglfnn_graph
except ImportError:
  torax = None

# Transport settings of the example kept for the TGLFNN model.
_KEPT_TRANSPORT_SETTINGS = (
    "apply_inner_patch", "D_e_inner", "V_e_inner", "chi_i_inner",
    "chi_e_inner", "rho_inner", "apply_outer_patch", "D_e_outer", "V_e_outer",
    "chi_i_outer", "chi_e_outer", "rho_outer", "chi_min", "chi_max",
    "D_e_min", "D_e_max", "V_e_min", "V_e_max", "smoothing_width", "An_min",
)


@unittest.skipIf(torax is None, "TORAX is not installed")
class ToraxTransportModelTest(unittest.TestCase):

  def test_gyrobohm_mass_conversion(self):
    """XNUE and VEXB_SHEAR are converted from TORAX's deuterium normalisation."""
    model = tglfnn_graph.TGLFNNukaeaGraphTransportModel(version="4.1.2")
    m_e_over_m_D = constants.CONSTANTS.m_e / (
        constants.CONSTANTS.m_amu * constants.ION_PROPERTIES_DICT["D"].A
    )
    fields = dataclasses.fields(tglf_based_transport_model.TGLFInputs)
    tglf_inputs = tglf_based_transport_model.TGLFInputs(
        **{field.name: jnp.full(3, 2.0) for field in fields}
    )
    tglf_inputs = dataclasses.replace(
        tglf_inputs, MASS_1=jnp.full(3, m_e_over_m_D)
    )
    factor = tglfnn_graph.gyrobohm_mass_factor(
        tglf_inputs, model.model.tglf_MASS_1
    )
    # 4.1.x is normalised to the proton mass: sqrt(m_p / m_D).
    np.testing.assert_allclose(factor, np.sqrt(1.007276 / 2.014102), rtol=1e-3)
    inputs = model._prepare_network_inputs(tglf_inputs)  # pylint: disable=protected-access
    labels = model.model.input_labels
    for i, label in enumerate(labels):
      if label in ("XNUE", "VEXB_SHEAR"):
        np.testing.assert_allclose(inputs[:, i], 2.0 * factor, rtol=1e-12)
    np.testing.assert_allclose(inputs[:, labels.index("RLTS_1")], 2.0)

  def test_newton_raphson_simulation(self):
    """A short ITER hybrid run, whose Newton-Raphson solver needs gradients."""
    tglfnn_graph.register()
    tglfnn_graph.register()  # Registering twice is harmless.
    config = copy.deepcopy(iterhybrid_rampup.CONFIG)
    config["numerics"]["t_final"] = 2.0
    config["numerics"]["fixed_dt"] = 1.0
    config["transport"] = {
        key: config["transport"][key] for key in _KEPT_TRANSPORT_SETTINGS
    } | {
        "model_name": "tglfnn-ukaea-graph",
        "version": "4.1.2",
        "DV_effective": True,
    }
    config["solver"]["solver_type"] = "newton_raphson"
    torax_config = torax.ToraxConfig.from_dict(config)
    self.assertIsInstance(
        torax_config.transport, tglfnn_graph.TGLFNNukaeaGraphTransportConfig
    )

    data_tree, state_history = torax.run_simulation(
        torax_config, log_timestep_info=False, progress_bar=False
    )
    self.assertEqual(state_history.sim_error, torax.SimError.NO_ERROR)
    np.testing.assert_allclose(data_tree.time.values, [0.0, 1.0, 2.0])
    for name in ("T_e", "T_i", "n_e", "chi_turb_e", "chi_turb_i"):
      self.assertTrue(np.isfinite(data_tree.profiles[name]).all(), name)
    # The network sets the transport between the inner and outer patches.
    rho = data_tree.profiles["chi_turb_e"].coords["rho_face_norm"].values
    mid = (rho > 0.35) & (rho < 0.85)
    chi_e = data_tree.profiles["chi_turb_e"].values[-1, mid]
    self.assertTrue((chi_e > config["transport"]["chi_min"]).any())


if __name__ == "__main__":
  unittest.main()
