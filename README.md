# tglfnn-ukaea
Neural network surrogate models of the [TGLF](https://gafusion.github.io/doc/tglf.html) quasilinear plasma turbulent transport simulator in various parameter spaces.

<!-- STEP-4.1-SERIES -->
## STEP EM SAT1 bundles

| Version | Model | Runtime files |
| --- | --- | --- |
| [4.1.0](STEP-effectiveion-EM-SAT1-14D-asinh1-large-ensemble5-9482b564f803/README.md) | Large five-member ensemble | Optional LFS download |
| [4.1.1](STEP-effectiveion-EM-SAT1-14D-asinh1-arch512x8-ensemble5-6922c92623ea/README.md) | Small 512×8 five-member ensemble | Default LFS download |
| [4.1.2](STEP-effectiveion-EM-SAT1-14D-asinh1-arch512x8-single-7368dc3ea876/README.md) | Best 512×8 single per channel | Default LFS download |

Version names are relative symlinks to descriptive, content-identified directories.
The descriptive directories contain TorchScript/ONNX models, per-flux configs,
physical reference settings, validation/distribution plots, metrics and checksums.
Consult each card's uncertainty limitations. Detailed provenance remains in the
linked tagged surrogate-builder repository. 4.1.2 is also distributed in the
Python package as an ONNX graph, with a JAX model and a TORAX transport model
(see [section 4](#onnx-graph-networks-jax-and-torax)).

Git LFS is required for runtime files. The committed `.lfsconfig` excludes
`STEP-effectiveion-EM-SAT1-14D-asinh1-large-ensemble5-9482b564f803/**` from default clone/pull downloads; its metadata remains available.
Git LFS include/exclude patterns use the descriptive paths, where files are tracked.
To obtain the large model explicitly, override that exclusion:

```bash
git lfs pull --include='STEP-effectiveion-EM-SAT1-14D-asinh1-large-ensemble5-9482b564f803/**' --exclude=''
cd 4.1.0
sha256sum -c SHA256SUMS
```

Normal clones and `git lfs pull` download 4.1.1/4.1.2. The exclusion controls
downloads; all model payloads remain tracked in Git LFS. Per-user/local Git
configuration can override the repository defaults.
<!-- /STEP-4.1-SERIES -->

## Paper for acknowledgment

If you use these models within your work, we request you cite the following paper:

* L. Zanisi et al., [Data efficent digital twinning strategies and surrogate models of quasilinear turbulence in JET and STEP](https://conferences.iaea.org/event/392/contributions/36059/), International Atomic Energy Agency - Fusion Energy Conference, Chengdu, China, 2025

# Usage

Various different methods exist for using the models:

1. Loading from PyTorch checkpoint
2. Loading traced ONNX model
3. Loading traced TorchScript model
4. Loading the parameters directly into pure Python

Loading the traced TorchScript model allows the model to be used in Fortran (see below).
Loading the parameters directly is a minimal-dependency method designed for use with other machine learning frameworks.

## 1. Loading from PyTorch checkpoint

```python
import torch

# Load the model
efe_gb_model = torch.load('MultiMachineHyper_1Aug25/regressor_efe_gb.pt')

# Call the model
input_tensor = torch.tensor([[...]], dtype=torch.float32)  # Replace with appropriate input
output_tensor = efe_gb_model(input_tensor)
```

## 2. Loading traced ONNX model

```python
import onnxruntime as ort

# Load the model
ort_session = ort.InferenceSession('MultiMachineHyper_1Aug25/regressor_efe_gb.onnx')

# Call the model
input_tensor = np.array([[...]], dtype=np.float32)  # Replace with appropriate input
outputs = ort_session.run(None, {'input': input_tensor})
```

## 3. Loading traced TorchScript model

```python
import torch

# Load the model
torchscript_model = torch.jit.load('MultiMachineHyper_1Aug25/regressor_efe_gb_torchscript.pt')

# Call the model
input_tensor = torch.tensor([[...]], dtype=torch.float32)  # Replace with appropriate input
output_tensor = torchscript_model(input_tensor)
```

### Using the traced TorchScript models in Fortran

The traced PyTorch models can be used in Fortran with [FTorch](https://github.com/Cambridge-ICCS/FTorch), which provides Fortran bindings for LibTorch (the C++ backend of PyTorch).  Please [cite the Ftorch publication](https://github.com/Cambridge-ICCS/FTorch#authors-and-acknowledgment) if using these models from Fortran.

Further details on the FTorch Implementation of these networks can be found in a [related project](https://github.com/ProjectTorreyPines/TurbulentTransport.jl/blob/master/utilities/README_onnx_to_pytorch_fortran.md).

#### Prerequisites

- **LibTorch**: Download the appropriate version (CPU or GPU) from the [PyTorch website](https://pytorch.org/get-started/locally/) and ensure it is accessible in your environment. CPU versions of the `LibTorch` and `Pip` packages have been tested. The `LibTorch` version requires no Python to install or run. It is suggested to look at the `FTorch` instructions below first.
- **FTorch**: Install the FTorch library following the instructions in the [FTorch repository](https://github.com/Cambridge-ICCS/FTorch). This also provides a compiler specific module (`ftorch.mod`).
- **Fortran Compiler**: Use a modern Fortran compiler (e.g., `gfortran` or `ifort`) compatible with FTorch.
- **CMake**: Version >= 3.1 required to build FTorch. Not essential, but helpful for building final Fortran code. 


## 4. Loading the parameters directly into pure Python

The `tglfnn_ukaea` Python package distributes network parameters as plain NumPy arrays, with numpy as the only dependency. They are stored as `tglfnn_ukaea/weights/{machine}/{version}.npz`:

| Machine | Versions | Default |
|---|---|---|
| `multimachine` | 3.2.1 ([MultiMachineHyper_1Aug25](MultiMachineHyper_1Aug25)) | 3.2.1 |
| `step` | 1.0.1 ([JETTO-TGLF-EM-17Feb25](JETTO-TGLF-EM-17Feb25)), 4.1.2 ([STEP EM SAT1](4.1.2/README.md), ONNX graph) | 1.0.1 |

```python
import tglfnn_ukaea

params = tglfnn_ukaea.load("multimachine")  # default version, or e.g. version="3.2.1"
tglfnn_ukaea.available_versions("multimachine")  # ["3.2.1"]
params["version"]        # network version, e.g. "3.2.1"
params["input_labels"]   # input order expected by the network
params["efe_gb/layer_0/weight"]  # (num_estimators, out, in) float32
```

The parameters, normalisations and variance epsilon are exported directly from the ONNX models by [scripts/onnx_to_npz.py](scripts/onnx_to_npz.py), so they reproduce the ONNX outputs. That script documents the array layout and the forward pass. The normalisations come from the ONNX models, not `stats.json`, which differs from them. To regenerate the weights:

```sh
git lfs pull
pip install -e ".[export]"
python scripts/onnx_to_npz.py MultiMachineHyper_1Aug25 --version 3.2.1 \
    -o tglfnn_ukaea/weights/multimachine/3.2.1.npz --verify \
    --input-labels RLNS_1 RLTS_1 RLTS_2 TAUS_2 RMIN_LOC DRMAJDX_LOC Q_LOC SHAT XNUE KAPPA_LOC DELTA_LOC ZEFF VEXB_SHEAR
python scripts/onnx_to_npz.py JETTO-TGLF-EM-17Feb25 --version 1.0.1 \
    -o tglfnn_ukaea/weights/step/1.0.1.npz --verify \
    --input-labels RLNS_1 RLTS_1 RLTS_2 TAUS_2 RMIN_LOC DRMAJDX_LOC Q_LOC SHAT XNUE KAPPA_LOC S_KAPPA_LOC DELTA_LOC S_DELTA_LOC BETAE ZEFF
```

Reference outputs of the ONNX models are distributed as test data, for testing implementations of the networks. `tglfnn_ukaea.load_reference("multimachine")` returns random `inputs` and the ONNX models' `(mean, variance)` for each flux. They are generated by [scripts/make_onnx_reference.py](scripts/make_onnx_reference.py), which should be rerun whenever the weights are regenerated:

```sh
python scripts/make_onnx_reference.py
```

### ONNX graph networks (JAX and TORAX)

Networks whose exports are not a plain MLP ensemble (4.1.x: residual blocks, layer norms, asinh output transform and an embedded uncertainty correction) are distributed as their ONNX graph instead: the node list and every tensor it uses, as plain arrays (`format = "onnx-graph"` in the npz). [tglfnn_ukaea/graph.py](tglfnn_ukaea/graph.py) evaluates the graph node by node with the ONNX operator semantics, with NumPy or with JAX. Nothing about the architecture is assumed, and an unsupported operator is an error.

```python
import tglfnn_ukaea
from tglfnn_ukaea.graph import Graph

arrays = tglfnn_ukaea.load("step", "4.1.2")
mean, variance = Graph.from_arrays(arrays, "efe_gb")(inputs)  # float32 (batch, 14), raw inputs
```

With JAX (`pip install ".[jax]"`), `TGLFNNukaeaGraphModel` predicts all fluxes, and is jittable, vmappable and differentiable in forward and reverse mode:

```python
import jax
from tglfnn_ukaea.model import TGLFNNukaeaGraphModel

jax.config.update("jax_enable_x64", True)  # the exports compare some inputs in float64
model = TGLFNNukaeaGraphModel("step", "4.1.2")  # or path="<graph npz>"
predictions = model.predict(inputs)  # {"efe_gb": (..., 2) [mean, variance], ...}
```

The TORAX transport model (`pip install ".[torax]"`, TORAX 1.4.x) is registered as `tglfnn-ukaea-graph`:

```python
import tglfnn_ukaea.torax_transport_model as tglfnn_graph

tglfnn_graph.register()
config["transport"] = {"model_name": "tglfnn-ukaea-graph", "version": "4.1.2", ...}
```

The TORAX model converts between TORAX's gyro-Bohm normalisation (deuterium mass) and the network's, which the export records from the bundle's `tglf_settings.json` (`MASS_1`; 4.1.x uses the proton mass): XNUE and VEXB_SHEAR are multiplied by sqrt(m_ref/m_D) before the network, and the gyro-Bohm fluxes by the same factor after it. The 4.1.x networks are effective-ion models trained at fixed `RMAJ_LOC = 2.0` (aspect ratio 2), with `MASS_2 = 2.5`; neither is a network input, so other aspect ratios or ion masses are outside the training data.

The graphs are exported by [scripts/onnx_to_graph_npz.py](scripts/onnx_to_graph_npz.py), which checks the result against onnxruntime. Larger networks (e.g. 4.1.1, 270 MB) can be exported locally and used with `path=`:

```sh
python scripts/onnx_to_graph_npz.py 4.1.2 -o tglfnn_ukaea/weights/step/4.1.2.npz
python scripts/onnx_to_graph_npz.py 4.1.1 -o step-4.1.1.npz
```

For graph networks, the references also hold the derivatives of each output with respect to the inputs, from finite differences of the graph converted to float64 and run with onnxruntime. The tests in [tests/](tests/) compare the NumPy and JAX evaluations and the JAX derivatives against them:

```sh
python -m unittest discover tests
```

## 5. Compare TGLF and TGLFNN in JETTO production runs

- TGLFNN is only an approximation of TGLF and it will make mistakes
- [scripts/tglf_vs_nn_jetto_trajectories.py](scripts/tglf_vs_nn_jetto_trajectories.py) shows how to plot the inputs and outputs spanned by TGLF and TGLFNN in a JETTO production run. **NOTE**: Available only in the following build on the JDC `/home/tn2395/jintrac-devel`