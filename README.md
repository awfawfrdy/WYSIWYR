# WYSIWYR

**What You Segment Is What You Report: Segmentation-Grounded Observational Report
Generation for Endoscopic Lesion Understanding**

Anonymous reproducibility release for peer review (*Expert Systems with Applications*).

This repository contains a **minimal, frozen** implementation sufficient to reproduce
the core experiments of the submitted manuscript. Datasets, model weights, third-party
source trees and generated outputs are intentionally **not** redistributed.

## Repository layout

```
.
├── wysiwyr_autodl_allinone.py     # Core methods (self-contained)
├── seig.py                        # Structured Evidence Inference Graph (frozen)
├── run_real_medsam_2x2.py         # Main segmentation experiment (2x2 factorial)
├── run_reviewer2_experiments.py   # 2x2 evaluation / statistics / USR accounting
├── stage6_seig_mllm_end2end.py    # Main report-generation experiment
├── configs/protocol.json          # Fixed protocol & hyperparameters
├── manifests/split_seed_2023.csv  # Fixed train/val split manifest (870 / 580)
└── requirements.txt
```

## Components

- **`wysiwyr_autodl_allinone.py`** — the method modules plus audit metrics:
  - **ABLoss** — ambiguity-aware boundary loss (FP/FN boundary penalties on top of
    Dice + BCE);
  - **USR** — uncertainty-driven structured rectifier (test-time-transform
    perturbations, soft aggregation, prune/repair with area / component / hole guards);
  - **SEIG** — structured evidence inference graph over symbolic segmentation evidence;
  - segmentation / uncertainty / calibration metrics and the 2x2 factorial effect
    estimator.

- **`run_real_medsam_2x2.py`** — one-command segmentation experiment: bootstraps MedSAM,
  fine-tunes {Dice+BCE, +ABLoss} from the same initialisation, performs **GT-free** test
  inference (full-image box -> coarse mask -> prediction-derived tight box -> refined
  mask), then applies USR to build the full 2x2 factorial
  (MedSAM / MedSAM+ABLoss / MedSAM+USR / MedSAM+ABLoss+USR).

- **`run_reviewer2_experiments.py`** — evaluation of the 2x2 factorial: case-level
  Dice / IoU / Precision / Recall / Boundary-Dice / HD95 / ASSD, ABLoss and USR main and
  interaction effects, paired Wilcoxon tests with bootstrap 95% CIs, and the USR
  pixel-accounting breakdown.

- **`stage6_seig_mllm_end2end.py`** — end-to-end report generation: for each test image,
  five mask sources (MedSAM / +ABLoss / +USR / +ABLoss+USR / GT-oracle control) are
  converted by the frozen SEIG into structured evidence and fed to a fixed MLLM with
  deterministic decoding; raw (R0) and checker-filtered (R*) reports and paired
  contrasts are saved.

## Data and dependencies (not redistributed)

- **Datasets (public).** Experiments use the PraNet-style endoscopic suite
  (Kvasir-SEG, CVC-ClinicDB, CVC-ColonDB, CVC-300, ETIS-LaribPolypDB). Download them
  from their official sources and point the scripts at the local roots.
- **MedSAM.** Used as an external segmentation backbone (`segment_anything`); the source
  tree and the ViT-B checkpoint are **not** vendored here — see
  <https://github.com/bowang-lab/MedSAM>.
- **MLLM backends.** Report generation uses instruction-tuned vision-language models
  (e.g. Qwen2.5-VL) loaded from their official releases.

Install Python dependencies with:

```bash
pip install -r requirements.txt
```

## Reproducibility notes

- **Prompt protocol.** Training prompts are GT-derived boxes (training annotations are
  supervised data); **test prompts never use GT** (full-image box -> prediction-derived
  refinement). This is recorded in `configs/protocol.json`.
- **Recovered hyperparameters.** The original training code was lost and the manuscript
  did not disclose several numeric main-training hyperparameters; `configs/protocol.json`
  records the explicit revision-reproducibility settings (split seed 2023, ABLoss/USR
  hyperparameters, optimiser). These are reported as revision settings, not as the
  original values.
- **Absolute paths.** Some scripts and `configs/protocol.json` keep the development
  server's absolute data paths as defaults; override them via the corresponding CLI
  arguments / config fields before running.

## License

Released for peer review. Third-party components (MedSAM, MLLMs, datasets) remain under
their own respective licenses.
