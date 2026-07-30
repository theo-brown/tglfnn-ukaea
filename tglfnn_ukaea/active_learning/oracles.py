"""Labels input points by running TGLF through TORAX's f2py wrapper.

The only TORAX dependency is ``torax._src.transport_model.tglf.tglf2py``,
which exposes ``run_tglf(**gacode_params)`` and requires the compiled
``tglf2py_lib`` extension (see the TORAX installation docs). Everything else
here is independent of TORAX.

The fixed TGLF settings mirror ``MultiMachineHyperES_11Mar26/common.tglf``,
the configuration used to generate the training data of the multimachine
surrogate: SAT2, electrostatic, 2 species (electrons + deuterium),
Miller geometry with R/a = 3.
"""

import multiprocessing
import os
from typing import Callable, Mapping, Sequence

import numpy as np

# Output labels, in the order used throughout this package (matching the
# ``params`` keys of the pickled checkpoints).
OUTPUT_LABELS = ("efe_gb", "efi_gb", "pfi_gb")

# An oracle maps a batch of inputs (n_points, n_inputs) to fluxes
# (n_points, len(OUTPUT_LABELS)) in gyro-Bohm units.
Oracle = Callable[[np.ndarray, Sequence[str]], np.ndarray]

# Fixed GACODE settings from MultiMachineHyperES_11Mar26/common.tglf,
# excluding the parameters varied by the surrogate inputs.
COMMON_TGLF_SETTINGS: Mapping[str, str] = {
    "UNITS": "GYRO",
    "USE_TRANSPORT_MODEL": "T",
    "GEOMETRY_FLAG": "1",
    "WRITE_WAVEFUNCTION_FLAG": "0",
    "SIGN_BT": "1.0",
    "SIGN_IT": "1.0",
    "THETA_TRAPPED": "0.7",
    "WDIA_TRAPPED": "0.0",
    "PARK": "1.0",
    "GHAT": "1.0",
    "GCHAT": "1.0",
    "WD_ZERO": "0.1",
    "LINSKER_FACTOR": "0.0",
    "GRADB_FACTOR": "0.0",
    "FILTER": "2.0",
    "DAMP_PSI": "0.0",
    "DAMP_SIG": "0.0",
    "IFLUX": ".true.",
    "USE_BPER": ".false.",
    "USE_BPAR": ".false.",
    "USE_MHD_RULE": "F",
    "USE_BISECTION": ".true.",
    "USE_INBOARD_DETRAPPED": ".false.",
    "IBRANCH": "-1",
    "NMODES": "2",
    "NBASIS_MAX": "6",
    "NBASIS_MIN": "2",
    "NXGRID": "16",
    "NKY": "12",
    "USE_AVE_ION_GRID": ".false.",
    "ADIABATIC_ELEC": ".false.",
    "ALPHA_MACH": "0.0",
    "ALPHA_E": "1.0",
    "ALPHA_P": "1.0",
    "ALPHA_QUENCH": "0.0",
    "ALPHA_ZF": "-1",
    "XNU_FACTOR": "1.0",
    "DEBYE_FACTOR": "1.0",
    "ETG_FACTOR": "1.25",
    "RLNP_CUTOFF": "18.0",
    "SAT_RULE": "2",
    "KYGRID_MODEL": "4",
    "XNU_MODEL": "2",
    "VPAR_MODEL": "0",
    "VPAR_SHEAR_MODEL": "1",
    "NS": "2",
    "MASS_1": "2.723e-4",
    "MASS_2": "1.0",
    "ZS_1": "-1.0",
    "ZS_2": "1.0",
    "KY": "0.3",
    "WIDTH": "1.65",
    "WIDTH_MIN": "0.3",
    "NWIDTH": "21",
    "FIND_WIDTH": ".true.",
    "TAUS_1": "1.0",
    "AS_1": "1.0",
    "AS_2": "1.0",
    "VEXB": "0.0",
    "BETAE": "0.0",
    "DEBYE": "0.0",
    "NEW_EIKONAL": ".true.",
    "RMAJ_LOC": "3.0",
    "ZMAJ_LOC": "0.0",
    "DRMINDX_LOC": "1.0",
    "DZMAJDX_LOC": "0.0",
    "S_KAPPA_LOC": "0.0",
    "S_DELTA_LOC": "0.0",
    "ZETA_LOC": "0.0",
    "S_ZETA_LOC": "0.0",
    "P_PRIME_LOC": "0.0",
    "BETA_LOC": "0.0",
    "KX0_LOC": "0.0",
    "NN_MAX_ERROR": "-1.0",
}


def _gacode_params(named_inputs: Mapping[str, float]) -> dict:
    """Maps the 13 surrogate inputs to GACODE parameters for one TGLF run."""
    params = dict(COMMON_TGLF_SETTINGS)
    for label, value in named_inputs.items():
        if label == "SHAT":
            continue  # Handled below.
        params[label] = float(value)
    # Pure deuterium plasma with quasineutrality: equal electron and ion
    # density gradients.
    params["RLNS_2"] = float(named_inputs["RLNS_1"])
    # The surrogate's SHAT input is defined via s = (r/q)^2 q', which enters
    # TGLF Miller geometry as Q_PRIME_LOC = (q/r)^2 s.
    params["Q_PRIME_LOC"] = float(
        named_inputs["SHAT"]
        * (named_inputs["Q_LOC"] / named_inputs["RMIN_LOC"]) ** 2
    )
    return params


def _run_tglf_point(
    point: Sequence[float], input_labels: Sequence[str]
) -> tuple:
    """Runs TGLF on one point; returns an (efe, efi, pfi) tuple (NaN on failure)."""
    from torax._src.transport_model.tglf import tglf2py

    named_inputs = dict(zip(input_labels, point))
    try:
        _, ion_pflux, elec_eflux, ion_eflux = tglf2py.run_tglf(
            **_gacode_params(named_inputs)
        )
    except Exception as error:  # TGLF crashes on some corner points.
        print(f"TGLF failed on point {dict(named_inputs)}: {error}")
        return (float("nan"),) * len(OUTPUT_LABELS)
    return (
        float(np.sum(elec_eflux)),
        float(np.sum(ion_eflux)),
        float(np.sum(ion_pflux)),
    )


def tglf_oracle(x: np.ndarray, input_labels: Sequence[str]) -> np.ndarray:
    """Runs TGLF (via the TORAX wrapper) serially on each row of ``x``.

    Args:
        x: Input points, shape ``(n_points, n_inputs)``, in physical units,
            with columns ordered as ``input_labels``.
        input_labels: GACODE-style names of the input columns, e.g.
            ``("RLNS_1", "RLTS_1", ..., "VEXB_SHEAR")``.

    Returns:
        Fluxes in gyro-Bohm units, shape ``(n_points, 3)``, with columns
        ``(efe_gb, efi_gb, pfi_gb)``. Rows where TGLF fails are NaN.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    return np.array(
        [_run_tglf_point(tuple(point), tuple(input_labels)) for point in x]
    )


def _init_tglf_worker() -> None:
    """Initialises one TGLF worker process."""
    # TGLF points are distributed one per process; don't let each worker's
    # OpenMP runtime also fan out over all cores.
    os.environ["OMP_NUM_THREADS"] = "1"
    # Import once so the per-point calls don't pay the TORAX import cost.
    from torax._src.transport_model.tglf import tglf2py  # noqa: F401


class ParallelTGLFOracle:
    """Runs TGLF points in parallel across worker processes.

    TGLF's state lives in Fortran module memory, so each worker process
    hosts its own independent TGLF instance. Workers are spawned once (each
    paying the TORAX import cost at startup) and reused across calls, and
    points are handed out one at a time since TGLF runtimes vary per point.

    Use as a drop-in replacement for :func:`tglf_oracle`::

        oracle = ParallelTGLFOracle(n_workers=8)
        run_active_learning(config, oracle=oracle)
    """

    def __init__(self, n_workers: int | None = None):
        self.n_workers = n_workers or os.cpu_count() or 1
        self._pool = None

    def __call__(self, x: np.ndarray, input_labels: Sequence[str]) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        if self._pool is None:
            # Spawn (rather than fork) to keep the workers free of the
            # parent's JAX/OpenMP thread state.
            context = multiprocessing.get_context("spawn")
            self._pool = context.Pool(
                self.n_workers, initializer=_init_tglf_worker
            )
        results = self._pool.starmap(
            _run_tglf_point,
            [(tuple(point), tuple(input_labels)) for point in x],
            chunksize=1,
        )
        return np.array(results)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None


def mock_oracle(x: np.ndarray, input_labels: Sequence[str]) -> np.ndarray:
    """Cheap analytic stand-in for TGLF, for tests and dry runs.

    Mimics critical-gradient behaviour: fluxes switch on above a threshold
    in the driving gradients and grow nonlinearly, which gives the active
    learning loop a nontrivial function to resolve.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    named = {label: x[:, i] for i, label in enumerate(input_labels)}
    rlts_1, rlts_2 = named["RLTS_1"], named["RLTS_2"]
    rlns_1, q_loc = named["RLNS_1"], named["Q_LOC"]
    drive_e = np.maximum(0.0, rlts_1 - 4.0 + 0.5 * rlns_1)
    drive_i = np.maximum(0.0, rlts_2 - 4.0 + 0.5 * rlns_1)
    efe = drive_e**1.5 * (1.0 + 0.2 * q_loc)
    efi = drive_i**1.5 * (1.0 + 0.3 * q_loc)
    pfi = 0.3 * np.sign(rlns_1) * np.abs(rlns_1) ** 1.2 * (drive_i > 0.0)
    return np.stack([efe, efi, pfi], axis=-1)
