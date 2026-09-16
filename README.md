# WYSIWYR

**What You Segment Is What You Report: Segmentation-Grounded Observational Report Generation for Endoscopic Lesion Understanding**

Minimal, frozen reproducibility release for peer review (*Expert Systems with Applications*).
This repository contains the code and frozen configuration used for the revised experiments.
Datasets, model weights, third-party source trees and generated outputs are **not** redistributed.

## Repository structure

```
.
├── wysiwyr_autodl_allinone.py            # Core methods (ABLoss / USR / SEIG + metrics)
├── seig.py                               # Frozen SEIG + Checker-v3 rule/lexicon engine
├── run_real_medsam_2x2.py                # Main MedSAM segmentation training + 2x2 factorial
├── stage4_promptfree.py                  # Prompt-free coarse proposer g_psi (SmallUNet)
├── stage5_calibrated.py                  # Validation-calibrated USR refinement
├── run_reviewer2_experiments.py          # 2x2 evaluation / statistics / USR accounting
├── stage6_seig_mllm_end2end.py           # Main report-generation experiment
├── stage7_seig_controls.py               # Information-source controls / mismatched evidence
├── stage8_multimllm.py                   # Multi-MLLM robustness control
├── stage9_seig_threshold_sensitivity.py  # SEIG threshold sensitivity
├── checker_v3_reaudit.py                 # Frozen Checker v3 rule-compliance re-audit
├── score_human_validation.py             # Checker human-validation scoring
├── analysis/                             # Reviewer-requested USR-failure + BH-FDR analyses
├── configs/
│   ├── protocol.json                     # Fixed protocol & hyperparameters
│   ├── checker_rulebook_v3.json          # Frozen rules: priority / negation / rewrite templates
│   ├── lexicons.json                     # Checker lexicons (forbidden / safety / boundary / calibration)
│   └── prompt_templates.json             # Deterministic SEIG evidence prompt (Table 5)
├── manifests/
│   └── split_seed_2023.csv               # Fixed train/val split manifest (870 / 580)
├── tools/
│   └── export_checker_spec.py            # Regenerates lexicons.json + prompt_templates.json from seig.py
├── requirements.txt
└── .gitignore
```

## Main MedSAM segmentation training

Baseline (Dice+BCE) and ABLoss are fine-tuned from the **same official MedSAM ViT-B
checkpoint** with an **identical** protocol: AdamW (`lr=1e-4`, `weight_decay=0.01`),
batch size 1, up to 50 epochs, CosineAnnealingLR, AMP, validation every 5 epochs,
early-stopping patience 4 checks, seed 2023. The main MedSAM checkpoint is selected by
**lowest validation loss** (the corresponding validation Dice is recorded). Test sets are
never used for checkpoint selection or hyperparameter tuning. See `configs/protocol.json`.

## Automatic prompt-free MedSAM initialization (coarse proposer g_psi)

Test-time MedSAM prompting does **not** use ground truth and does **not** use a full-image-box
bootstrap. An independent prompt-free coarse proposer `g_psi` predicts a coarse foreground
probability map, whose largest component is converted into a validation-calibrated box:

`P_coarse = g_psi(X)`, `M_init = I[P_coarse > tau_coarse]`, `b = Expand(Box(M_init), margin)`.

Final `g_psi`: lightweight four-level U-Net (SmallUNet, base width 24, input 320x320),
trained from scratch on the fixed 870-image split only (BCE + soft Dice, AdamW `lr=3e-4`,
`weight_decay=1e-4`, batch 8, up to 30 epochs, CosineAnnealingLR, patience 6), checkpoint by
**highest validation Dice**. Prompt thresholds/margins are calibrated on the validation split
only; `tau_coarse = 0.35`, `margin_fraction = 0.02`. See `stage4_promptfree.py`.

## USR inference

Final validation-calibrated USR: `T=8`, reference resolution 1024, prompt jitter 10 px at 1024,
aggregated foreground threshold 0.50, boundary kernels 3/9, low-probability pruning 0.65,
high-uncertainty pruning 0.35, local-support pruning 0.80, boundary-repair 0.70, area-drop 0.35,
area-rise 0.25, soft probability averaging, no per-image min-max normalization, resolution-dependent
spatial scaling. Prompt-driven models use box perturbations; non-prompt models use deterministic
TTA (identity / h-flip / v-flip / h+v) with inverse transform. See `stage5_calibrated.py` and
`configs/protocol.json`.

## SEIG + Checker

The frozen, deterministic SEIG and Checker-v3 rule engine lives in `seig.py`; the complete
rulebook (priority / negation / claim decomposition / rewrite templates), the 14 lexicons
(forbidden / safety / boundary / calibration), and the deterministic evidence prompt template
are materialised in `configs/`. Rules are frozen before independent human validation and are
not modified afterwards. `tools/export_checker_spec.py` regenerates the JSON specs from `seig.py`.

## Cross-MLLM evaluation

`stage8_multimllm.py` implements Qwen2.5-VL-3B, InternVL2.5-2B and MiniCPM-V-2.6 backends with
deterministic decoding. **HuatuoGPT-Vision is reported in the manuscript but its adapter is not
present in this release** (see "Reproducibility gaps" below). Model checkpoints are not
redistributed and must be obtained from their original providers.

## Statistical analysis

`analysis/finalize_usr_failure_analysis.py` implements Benjamini-Hochberg FDR correction
(`statsmodels.stats.multitest.multipletests(method="fdr_bh")`) over the reviewer-requested
USR-failure mechanism tests. Segmentation-to-report Spearman associations are in
`run_reviewer2_experiments.py`.

## Environment

```bash
pip install -r requirements.txt
```

## Data preparation

Experiments use the public PraNet-style endoscopic suite (Kvasir-SEG, CVC-ClinicDB,
CVC-ColonDB, CVC-300, ETIS-LaribPolypDB). Download them from their official sources; they are
**not redistributed**. Set the data/artefact root before running:

```bash
export WYSIWYR_DATA_ROOT=/path/to/your/data_root
```

## Reproduction commands

```bash
# Prompt-free coarse proposer + USR (final test-time prompting)
python stage4_promptfree.py --root "$WYSIWYR_DATA_ROOT/wysiwyr_real"
python stage5_calibrated.py  --root "$WYSIWYR_DATA_ROOT/wysiwyr_real"

# Main MedSAM segmentation training + 2x2 factorial
python run_real_medsam_2x2.py --auto --root "$WYSIWYR_DATA_ROOT/wysiwyr_real"

# Reviewer-requested experiments (Stages 6-9, checker)
python stage6_seig_mllm_end2end.py --root "$WYSIWYR_DATA_ROOT/wysiwyr_real" --model "$WYSIWYR_DATA_ROOT/models/Qwen2.5-VL-3B-Instruct"
python checker_v3_reaudit.py --source "$WYSIWYR_DATA_ROOT/wysiwyr_real/stage6_seig_mllm_end2end" --output "$WYSIWYR_DATA_ROOT/wysiwyr_real/stage6_checker_v3"
```

## Reproducibility statements

- Datasets and pretrained model weights are **not redistributed**.
- **No GT mask or GT box is used** for automatic test-time MedSAM prompting.
- Test sets are **not used** for hyperparameter tuning or checkpoint selection.
- SEIG / Checker rules are **deterministic and frozen**.
- MLLMs are **frozen** and use **deterministic decoding** in the main experiments.

## Reproducibility gaps

- **HuatuoGPT-Vision**: reported in the manuscript, but no adapter exists in the released code;
  reproduce Qwen2.5-VL-3B / InternVL2.5-2B / MiniCPM-V-2.6 only.
- **18 prespecified association tests**: the manuscript reports BH-FDR-adjusted q-values for 18
  prespecified segmentation-to-report association tests; the released scripts implement BH-FDR
  for the USR-failure mechanism tests and unadjusted Spearman associations separately.

## License

Released for peer review. Third-party components (MedSAM, MLLMs, datasets) remain under their
own respective licenses.
