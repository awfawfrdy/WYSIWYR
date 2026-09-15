# WYSIWYR

**What You Segment Is What You Report: Segmentation-Grounded Observational Report
Generation for Endoscopic Lesion Understanding**

Anonymous reproducibility release for peer review (*Expert Systems with Applications*).

This repository contains a **minimal, frozen** implementation sufficient to reproduce
the experiments reported in the submitted manuscript. Datasets, model weights,
third-party source trees, generated outputs and any future/prototype modules are
intentionally **not** redistributed.

## Repository layout

```
.
├── wysiwyr_autodl_allinone.py            # Core methods (self-contained)
├── seig.py                               # Frozen SEIG + Checker-v3 rule/lexicon engine
├── run_real_medsam_2x2.py                # Main segmentation experiment (2x2 factorial)
├── run_reviewer2_experiments.py          # 2x2 evaluation / statistics / USR accounting
├── stage6_seig_mllm_end2end.py           # Main report-generation experiment
├── stage7_seig_controls.py               # Information-source controls / mismatched evidence
├── stage8_multimllm.py                   # Multi-MLLM robustness control
├── stage9_seig_threshold_sensitivity.py  # SEIG threshold sensitivity
├── checker_v3_reaudit.py                 # Frozen Checker v3 rule-compliance re-audit
├── score_human_validation.py             # Checker human-validation scoring
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

## Components

- **`wysiwyr_autodl_allinone.py`** — the method modules plus audit metrics:
  **ABLoss** (ambiguity-aware boundary loss: FP/FN boundary penalties on top of
  Dice+BCE), **USR** (uncertainty-driven structured rectifier), **SEIG** (structured
  evidence inference graph), plus segmentation / uncertainty / calibration metrics and
  the 2x2 factorial effect estimator.
- **`seig.py`** — the frozen SEIG module and the Checker-v3 rule/lexicon engine
  (imported by the Stage 6-9 scripts and the checker).
- **`run_real_medsam_2x2.py`** — one-command segmentation experiment: bootstraps MedSAM,
  fine-tunes {Dice+BCE, +ABLoss} from the same initialisation, performs **GT-free** test
  inference (full-image box -> coarse mask -> prediction-derived tight box -> refined
  mask), then applies USR to build the full 2x2 factorial
  (MedSAM / +ABLoss / +USR / +ABLoss+USR).
- **`run_reviewer2_experiments.py`** — 2x2 evaluation: case-level
  Dice / IoU / Precision / Recall / Boundary-Dice / HD95 / ASSD, ABLoss and USR main and
  interaction effects, paired Wilcoxon with bootstrap 95% CIs, and USR pixel accounting.
- **`stage6_seig_mllm_end2end.py`** — end-to-end report generation: five mask sources
  (MedSAM / +ABLoss / +USR / +ABLoss+USR / GT-oracle control) -> frozen SEIG -> fixed
  MLLM with deterministic decoding; raw (R0) and checker-filtered (R*) reports.

## Data and dependencies (not redistributed)

- **Datasets (public).** Experiments use the PraNet-style endoscopic suite
  (Kvasir-SEG, CVC-ClinicDB, CVC-ColonDB, CVC-300, ETIS-LaribPolypDB). Download them
  from their official sources.
- **MedSAM.** Used as an external segmentation backbone (`segment_anything`); the source
  tree and the ViT-B checkpoint are **not** vendored — see
  <https://github.com/bowang-lab/MedSAM>.
- **MLLM backends.** Report generation uses instruction-tuned vision-language models
  (e.g. Qwen2.5-VL) loaded from their official releases.

```bash
pip install -r requirements.txt
```

## Data root (no server paths)

All scripts read the data/artefact root from the **`WYSIWYR_DATA_ROOT`** environment
variable (default: the current directory `.`). Set it once before running:

```bash
export WYSIWYR_DATA_ROOT=/path/to/your/data_root
```

Every path in the code is either derived from `WYSIWYR_DATA_ROOT` or exposed as a CLI
argument, so the scripts are no longer tied to any specific machine.

## Reproducing reviewer-requested experiments

Set `WYSIWYR_DATA_ROOT` first (see above). Commands below use its default-relative
paths; override the CLI arguments if your layout differs.

**Stage 7 — information-source controls / mismatched evidence**
(conditions: image-only, SEIG-only, image+SEIG, image+mismatched-SEIG)
```bash
python stage7_seig_controls.py \
    --root  "$WYSIWYR_DATA_ROOT/wysiwyr_real" \
    --model "$WYSIWYR_DATA_ROOT/models/Qwen2.5-VL-3B-Instruct"
```

**Stage 8 — multi-MLLM robustness control**
(40 cases per dataset = 200; four controls per case; Qwen2.5-VL reused from Stage 7,
plus InternVL2.5-2B and MiniCPM-V-2.6; run each backend then aggregate)
```bash
python stage8_multimllm.py \
    --stage7 "$WYSIWYR_DATA_ROOT/wysiwyr_real/stage7_seig_controls" \
    --output "$WYSIWYR_DATA_ROOT/wysiwyr_real/stage8_multimllm"
```

**Stage 9 — SEIG systematic threshold sensitivity**
(symbolic analysis on all Stage 6 cases; 11 threshold configurations; deterministic
stratified 200-case report subset, 40 per dataset; no test-set tuning)
```bash
python stage9_seig_threshold_sensitivity.py \
    --root  "$WYSIWYR_DATA_ROOT/wysiwyr_real" \
    --model "$WYSIWYR_DATA_ROOT/models/Qwen2.5-VL-3B-Instruct" \
    --cases-per-dataset 40 --max-new-tokens 384 --seed 2023
```

**Checker v3 — frozen rule-compliance re-audit** (produces R* from the Stage 6 raw R0)
```bash
python checker_v3_reaudit.py \
    --source       "$WYSIWYR_DATA_ROOT/wysiwyr_real/stage6_seig_mllm_end2end" \
    --output       "$WYSIWYR_DATA_ROOT/wysiwyr_real/stage6_checker_v3" \
    --dev-template "$WYSIWYR_DATA_ROOT/wysiwyr_real/stage6_checker_v2/checker_v2_human_validation_template.csv"
```
Required inputs: the Stage 6 case outputs and the prior v2 development-sample template
(used only to *exclude* the dev sample from the final validation set).

**Checker human-validation scoring** (after annotators fill the blinded CSV)
```bash
python score_human_validation.py --human <completed_blinded_csv> \
    --key <blinded_key_csv> --output <scores.json>
```

### Frozen Checker v3 specification (rules / lexicons / negation / priority / rewrites / prompts)

The complete, frozen specification used in the manuscript is materialised as plain JSON
so it can be reviewed without reading code:

| Artefact | Contents |
|---|---|
| `configs/checker_rulebook_v3.json` | rule **priority** (R1–R5 / R0), **negation policy**, deterministic **rewrite templates**, claim decomposition, freeze policy |
| `configs/lexicons.json` | the 14 checker **lexicons** — forbidden diagnosis / procedure / recommendation / anatomy vocabulary, safety-disclaimer exemptions, boundary descriptors, hedging and certainty lexicons |
| `configs/prompt_templates.json` | the deterministic **SEIG evidence prompt** used for report generation (Table 5), including wording rules and the five-section output format |

All three are generated from the executable source of truth (`seig.py` —
`SEIG.checker_rulebook()` and `SEIG.render_prompt`) via `tools/export_checker_spec.py`:

```bash
python tools/export_checker_spec.py
```

The v3 rules are **frozen** before the independent human validation and must not be
modified afterwards.

## Reproducibility notes

- **Prompt protocol.** Training prompts are GT-derived boxes (training annotations are
  supervised data); **test prompts never use GT** (full-image box -> prediction-derived
  refinement). Recorded in `configs/protocol.json`.
- **Recovered hyperparameters.** The original training code was lost and the manuscript
  did not disclose several numeric main-training hyperparameters; `configs/protocol.json`
  records the explicit revision-reproducibility settings (split seed 2023, ABLoss/USR
  hyperparameters, optimiser). These are reported as revision settings, not as the
  original values.

## License

Released for peer review. Third-party components (MedSAM, MLLMs, datasets) remain under
their own respective licenses.
