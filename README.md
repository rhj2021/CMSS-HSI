# CMSS-HSI

Code for **Change-Masked Scanline Synchronization (CMSS)**, a preprocessing method for hyperspectral image change detection under residual spatial misregistration.

CMSS estimates a global translation and a smooth row-wise horizontal displacement in a shared PCA space. Likely changed pixels are downweighted during matching, and a detector-free selector chooses among identity, PCA3 translation, and full CMSS.

## Requirements

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For GPU experiments, install the PyTorch build that matches your CUDA version.

## Data

Place the River, Farmland, and Hermiston datasets under `data/`. The expected file layout is listed in [`data/README.md`](data/README.md). The datasets are not redistributed here.

## Run the paper protocol

```bash
python run_cmss.py --config configs/paper.yaml --stage all --device cuda:0
```

The same command is available as:

```bash
bash scripts/run_paper.sh cuda:0
```

Stages can also be run separately:

```bash
python run_cmss.py --config configs/paper.yaml --stage inspect   --device cuda:0
python run_cmss.py --config configs/paper.yaml --stage prepare   --device cuda:0
python run_cmss.py --config configs/paper.yaml --stage train     --device cuda:0
python run_cmss.py --config configs/paper.yaml --stage evaluate  --device cuda:0
python run_cmss.py --config configs/paper.yaml --stage summarize --device cuda:0
```

Add `--force` to overwrite cached outputs for the selected stage.

## Quick test

The smoke test creates small synthetic datasets and runs the full pipeline on CPU.

```bash
bash scripts/run_smoke_test.sh
```

Optional unit tests:

```bash
pip install -r requirements-dev.txt
bash scripts/run_unit_tests.sh
```

## Main outputs

Results are written to `results/cmss_hsi/`.

```text
metrics/condition_metrics.csv
metrics/correction_metrics.csv
summary/main_gap_recovery.csv
summary/ablation_summary.csv
summary/clean_noharm_summary.csv
summary/cmss_selector_summary.csv
resolved_config.yaml
feedback_bundle.zip
```

## Detector code

GETNET and GLAFormer are based on their official released implementations, with minor changes for data loading, spatial splitting, training, and evaluation in the unified CMSS pipeline. The detector and validation-selected threshold are fixed across clean, misaligned, and corrected inputs for each seed.

## Notes

- Controlled perturbations are applied only to the second date.
- CMSS does not use ground-truth change labels or downstream detector scores for geometric correction.
- Dataset files, checkpoints, and generated results are ignored by Git.

## Citation

The citation entry will be added after the paper is accepted.

## License

This repository is prepared for public reproducibility. A repository-wide software license is not included because the detector code is derived from released baseline implementations. Check the upstream terms before redistributing or relicensing those files.
