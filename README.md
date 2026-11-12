# MGMT Run 4

This project contains the multi-cohort EfficientNet-B0 and EfficientNet-B2 pipeline for predicting MGMT promoter methylation from MRI. It includes the recovered v9 training script and a CPU evaluator that rebuilds results from the saved Run 4 predictions.

## Project structure

| Path | Purpose |
|---|---|
| `train.py` | Launch the recovered training script |
| `evaluate.py` | Rebuild and verify the reported metrics from saved predictions |
| `check_inputs.py` | Inspect the Kaggle training inputs |
| `training/` | Recovered v9 training implementation |
| `analysis/` | Run 4 evaluator and original plotting script |
| `notebooks/` | Self-contained Kaggle recovery notebook |
| `data/` | Input instructions and the UPenn-GBM labels from the recovered source archive |
| `checkpoints/` | Guidance for saving weights and resuming jobs |
| `results/verified_run4/` | Aggregate tables regenerated from the actual Run 4 archive |

## Reproduce the Run 4 results

Use Python 3.12 for the pinned analysis environment. From the project folder, install the CPU dependencies and pass the original Run 4 archive to the evaluator.

```bash
python -m pip install -r requirements.txt
python evaluate.py "/path/to/outputs_merged_run4.zip"
```

The evaluator accepts the outer Run 4 ZIP as well. It writes fresh tables to `results/rebuilt_run4/` by default. Use `--output` to choose another directory. See `verification_summary.json` for the input archive checksum and the recalculated headline AUCs.

For Kaggle, import `notebooks/MGMT_Run4_Recovery.ipynb`, attach the original Run 4 ZIP using **Add Input**, and select **Run All**. Use the CPU accelerator setting. The notebook writes a downloadable ZIP containing the regenerated tables and the source scripts.

## Train with the original datasets

Prepare the inputs listed in [data/README.md](data/README.md). Use a CUDA-enabled environment such as Kaggle with two T4 GPUs. Make the training dependencies available from `requirements-training.txt` and enable Internet on the first run if the ImageNet weights need to be downloaded.

```bash
python train.py --input-root "/path/to/datasets" --work-dir "./merged_work" --quick
python train.py --input-root "/path/to/datasets" --work-dir "./merged_work"
```

The full run writes to `merged_work/outputs_merged/`. The quick check uses a separate output directory. The recovered plotting script accepts a completed training output directory.

```bash
python analysis/paper_figures.py "./merged_work/outputs_merged"
```

## Verification and provenance

The saved Run 4 jobs reproduced **73.80%** checkpoint-selected AUC, **66.16%** fixed-epoch AUC, **386** duplicate exclusions, and **89.78%** specificity when the high-specificity threshold was selected on the other folds. The original plotting script selects one threshold on the pooled cohort and reports a different specificity. The evaluator reports both procedures explicitly.

The aggregate tables in `results/verified_run4/` come from the archived predictions. They omit patient-level prediction rows. The evaluator's selective-prediction intervals are recalculated using 2,000 patient bootstrap resamples. Some interval endpoints and held-out AUCs differ slightly from the manuscript at the reported precision.

The recovered training script is byte-identical to the v9 file supplied in `mgmt-merged-code.zip`. The Run 4 archive does not contain an executed notebook or source checksum that would establish the precise historical training file. This package was checked through the CPU evaluation path using the original Run 4 archive. Fresh GPU training was not rerun.
