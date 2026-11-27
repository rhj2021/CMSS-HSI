# CMSS

Code for **Change-Masked Scanline Synchronization for Misregistration-Tolerant Hyperspectral Change Detection**.

CMSS is a training-free preprocessor for small residual misregistration in hyperspectral image pairs.

## Environment

Python 3.12, PyTorch 2.8.0 and torchvision 0.23.0. The full experiment suite uses Linux and CUDA.

```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python install_cuda_extensions.py
```

## Data

```text
data/
  River/
  Farmland/
  Hermiston/
```

Each folder contains the original MATLAB dataset files.

## Experiments

Settings are in `configs/benchmark.yaml`; outputs are saved under `results/`.

```bash
python run_benchmark.py --stage all
python run_controls.py --stage all
python run_timing.py --stage all
```

The three commands run the main comparisons and ablations, weight and spectral controls, and timing experiments, respectively.

RCDIFO uses `configs/rcdifo.yaml`:

```bash
python prepare_rcdifo.py --download
python run_rcdifo.py --stage all --main-output results/benchmark
```

GETNET, GLAFormer and IR-MAD are included. RoMa is installed by the extension script and downloads pretrained weights on first use. The RCDIFO preparation command downloads its source; detector checkpoints are trained locally.

## Python API

```python
from cmss import synchronize

result = synchronize(reference, moving, device="cuda:0")
aligned = result.corrected
flow = result.flow
```

Inputs are finite `H × W × B` arrays with matching shapes. Flow stores sampling offsets in `(dy, dx)` order. CPU execution is available with `device="cpu"`.

Third-party sources and licenses: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
