# tglfnn-ukaea
Neural network surrogate models of the [TGLF](https://gafusion.github.io/doc/tglf.html) quasilinear plasma turbulent transport simulator in various parameter spaces.

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


## 4. Compare TGLF and TGLFNN in JETTO production runs

- TGLFNN is only an approximation of TGLF and it will make mistakes
- [scripts/tglf_vs_nn_jetto_trajectories.py](scripts/tglf_vs_nn_jetto_trajectories.py) shows how to plot the inputs and outputs spanned by TGLF and TGLFNN in a JETTO production run. **NOTE**: Available only in the following build on the JDC `/home/tn2395/jintrac-devel`
# Active learning

`tglfnn_ukaea.active_learning` contains a simple JAX active learning loop for
improving the surrogates. Each round it samples candidate points from the
model's training hypercube, selects the points where the ensemble members
disagree most (epistemic uncertainty), labels them by running TGLF through
[TORAX](https://github.com/google-deepmind/torax)'s `tglf2py` wrapper (the
only TORAX component used), fine-tunes the ensembles on the accumulated data,
and saves checkpoints in the same pickle format as the shipped weights.

```bash
pip install -e .[active-learning]
# Requires TORAX with its compiled TGLF wrapper for real labelling:
# https://torax.readthedocs.io/en/latest/installation.html#optional-install-tglf
python scripts/run_active_learning.py --n-rounds 10 --acquisition-batch 64

# Dry run without TGLF installed:
python scripts/run_active_learning.py --mock-oracle
```

Or from Python:

```python
from tglfnn_ukaea import active_learning

config = active_learning.ActiveLearningConfig(n_rounds=5, acquisition_batch=32)
result = active_learning.run_active_learning(config)
```

Note: as of July 2026, TORAX's `tglf2py` wrapper has two runtime bugs that
make it report the compiled extension as missing and crash on assigning the
`SHAPE_*` defaults. [patches/torax-tglf2py-runtime-fixes.patch](patches/torax-tglf2py-runtime-fixes.patch)
fixes both (apply with `git am` in your TORAX checkout).
