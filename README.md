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

A distilled single-network "student" variant of the multimachine model is
also provided for speed-critical applications (see section 5).

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

```python
import tglfnn_ukaea

model = tglfnn_ukaea.load("multimachine")  # or "step", "multimachine_student"
# model["params"]        - raw weights per flux, per ensemble member, per layer
# model["stats"]         - per-variable mean/std for input/output normalisation
# model["input_labels"]  - the required input ordering
# model["config"]        - architecture and training metadata
```

The reference JAX inference implementation for these dicts is
[`google-deepmind/fusion_surrogates`](https://github.com/google-deepmind/fusion_surrogates)
(`fusion_surrogates.tglfnn_ukaea.TGLFNNukaeaModel`), which is also the code
path used by [TORAX](https://github.com/google-deepmind/torax).

## 5. Distilled student model (fast single-network variant)

`tglfnn_ukaea/weights/multimachine_student.pkl` is a single-network
distillation of the `multimachine` deep ensemble, produced by
[scripts/distill_student.py](scripts/distill_student.py):

- **Teacher**: the released `multimachine` checkpoint (per flux: 5-member
  deep ensemble, 5x512 hidden layers, NLL-trained).
- **Student**: per flux, a single Gaussian MLP with 4x256 hidden layers and
  `tanh` activations (~26x fewer FLOPs per evaluation), trained to
  reproduce the teacher's ensemble mean and total variance
  (aleatoric + epistemic) on inputs sampled uniformly from the training
  hypercube recorded in the teacher checkpoint. `tanh` is used instead of
  `relu` so the surrogate is smooth, which improves the convergence of
  Newton-type transport solvers.
- **Schema**: identical pickle schema to the released checkpoints, with
  `num_estimators: 1` and the reduced architecture recorded in `config`.
  Since `fusion_surrogates` builds its network from those config entries,
  the student loads through the existing `TGLFNNukaeaModel` path unchanged:

```python
from fusion_surrogates.tglfnn_ukaea import tglfnn_ukaea_model

model = tglfnn_ukaea_model.TGLFNNukaeaModel("multimachine_student")
predictions = model.predict(inputs)  # same API and conventions as the teacher
```

Distillation quality metrics (student vs teacher on a held-out sample of the
training hypercube) are stored in `config["distillation"]["metrics"]` inside
the student checkpoint.

Caveats: the student approximates the *teacher*, adding a small additional
error on top of the teacher's TGLF approximation error; its variance output
is the teacher's total (aleatoric + epistemic) uncertainty folded into a
single channel, and the ensemble-spread decomposition is no longer
available.

## 6. Compare TGLF and TGLFNN in JETTO production runs

- TGLFNN is only an approximation of TGLF and it will make mistakes
- [scripts/tglf_vs_nn_jetto_trajectories.py](scripts/tglf_vs_nn_jetto_trajectories.py) shows how to plot the inputs and outputs spanned by TGLF and TGLFNN in a JETTO production run. **NOTE**: Available only in the following build on the JDC `/home/tn2395/jintrac-devel`