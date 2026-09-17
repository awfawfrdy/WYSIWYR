# WYSIWYR

**What You Segment Is What You Report: Segmentation-Grounded Observational Report Generation for Endoscopic Lesion Understanding**

Minimal, frozen reproducibility release for peer review (*Expert Systems with Applications*).
This repository contains the code and frozen configuration used for the revised experiments.
Datasets, model weights, third-party source trees and generated outputs are **not** redistributed.

> **Scope of this release (read first).** This is a *minimal frozen* release, not a
> one-command reproduction bundle. See
> [Reproducibility tiers](#reproducibility-tiers) for an explicit statement of what can be
> reproduced with public resources only, what needs model weights, what needs
> author-provided frozen artifacts, and what is not publicly reproducible at all.

## Repository structure

```
.
├── wysiwyr_autodl_allinone.py            # Core methods (ABLoss / USR / SEIG + metrics)
├── seig.py                               # Frozen SEIG + Checker-v3 rule/lexicon engine
├── run_real_medsam_2x2.py                # MedSAM training (baseline / ABLoss) + 2x2 inference driver
├── stage4_promptfree.py                  # Prompt-free coarse proposer g_psi (SmallUNet)
├── stage5_calibrated.py                  # Validation-calibrated prompt box + frozen USR (manuscript path)
├── run_reviewer2_experiments.py          # 2x2 evaluation / statistics / USR pixel accounting
├── stage6_seig_mllm_end2end.py           # Main end-to-end report-generation experiment
├── stage7_seig_controls.py               # Information-source controls / mismatched evidence
├── stage8_multimllm.py                   # Multi-MLLM robustness control
├── stage9_seig_threshold_sensitivity.py  # SEIG threshold sensitivity
├── checker_v3_reaudit.py                 # Frozen Checker v3 rule-compliance re-audit + validation sample
├── score_human_validation.py             # Checker-vs-human scoring (P/R/F1, macro-F1 + bootstrap CI, kappa, confusion matrix, raw agreement)
├── analysis/                             # Reviewer-requested analyses (see below), incl.
│   ├── reviewer2_r1_associations.py      #   18 prespecified case-level associations (BH-FDR + stratified bootstrap CI)
│   ├── reviewer2_r4_usr_pixel_audit.py   #   USR semantic pixel audit + case-level worsening
│   ├── reviewer2_r4_mask_verify.py       #   independent mask-level re-derivation of the audit
│   ├── analyze_usr_failure.py            #   USR failure-mechanism analysis
│   ├── finalize_usr_failure_analysis.py  #   BH-FDR for the failure-mechanism tests
│   ├── build_blinded_annotation.py       #   blinded rating-sheet construction
│   ├── clinician_mixed_effects.R         #   clinician CLMM (ordinal::clmm), OR / CI / p
│   └── clinician_agreement.py            #   ICC / Fleiss' kappa / raw agreement
├── configs/
│   ├── protocol.json                     # Fixed protocol & hyperparameters
│   ├── checker_rulebook_v3.json          # Frozen rules: priority / negation / rewrite templates
│   ├── lexicons.json                     # Checker lexicons (forbidden / safety / boundary / calibration)
│   ├── prompt_templates.json             # Deterministic SEIG evidence prompt (Table 5)
│   └── clinician_evaluation_protocol.json# Clinician study design (frozen before evaluation)
├── manifests/
│   └── split_seed_2023.csv               # Fixed train/val split manifest (870 / 580)
├── tools/
│   └── export_checker_spec.py            # Regenerates (--check: verifies) lexicons.json + prompt_templates.json
├── MANUSCRIPT_REPRODUCIBILITY.md          # Manuscript item -> repo entry -> command -> output
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

ABLoss is implemented as `ABLoss` / `ABLossConfig` in `wysiwyr_autodl_allinone.py`:

`L = Dice + BCE + lambda_fp * L_fp + lambda_fn * L_fn`

with inner/outer boundary bands built by `ABLoss.boundary_bands()` (dilate−erode at
`boundary_kernel_size=7`), asymmetry weights `alpha=2.0, beta=1.0, gamma=1.0, delta=0.3`
and `lambda_fp=0.8, lambda_fn=0.3` (identical to `configs/protocol.json["abloss"]`).

## Automatic prompt-free MedSAM initialization (coarse proposer g_psi)

Test-time MedSAM prompting does **not** use ground truth and does **not** use a full-image-box
bootstrap. An independent prompt-free coarse proposer `g_psi` predicts a coarse foreground
probability map, whose largest component is converted into a validation-calibrated box:

`P_coarse = g_psi(X)`, `M_init = I[P_coarse > tau_coarse]`, `b = Expand(Box(M_init), margin)`.

Final `g_psi`: lightweight four-level U-Net (SmallUNet, base width 24, input 320x320),
trained from scratch on the fixed 870-image split only (BCE + soft Dice, AdamW `lr=3e-4`,
`weight_decay=1e-4`, batch 8, up to 30 epochs, CosineAnnealingLR, patience 6), checkpoint by
**highest validation Dice**. Prompt thresholds/margins are calibrated on the validation split
only; `tau_coarse = 0.35`, `margin_fraction = 0.02`. See `stage4_promptfree.py` and
`configs/protocol.json["coarse_proposer_g_psi"]`.

Ground-truth boxes are used **only during training** (`configs/protocol.json ->
"training_prompt_protocol"`, `bbox_shift_at_1024 = 20`), never at test time. The frozen
`test_prompt_protocol` records `gt_used: false`, and the same prediction-derived box is shared
by the baseline and ABLoss branches.

## USR inference

Final validation-calibrated USR: `T=8`, reference resolution 1024, prompt jitter 10 px at 1024,
aggregated foreground threshold 0.50, boundary kernels 3/9, low-probability pruning 0.65,
high-uncertainty pruning 0.35, local-support pruning 0.80, boundary-repair 0.70, area-drop 0.35,
area-rise 0.25, soft probability averaging, no per-image min-max normalization, resolution-dependent
spatial scaling. Prompt-driven models use box perturbations; non-prompt models use deterministic
TTA (identity / h-flip / v-flip / h+v) with inverse transform. See `stage5_calibrated.py` and
`configs/protocol.json`.

**USR parameters are calibrated on validation data and frozen before test evaluation.**
Resolution-dependent scaling is implemented in code (`USR._scale_len` for length-based
parameters and `USR._scale_area` for area-based parameters, both referenced to 1024 px), not
only described here. Topology-guard and post-processing parameters are the `USRConfig` fields
`max_area_drop` / `max_area_rise` / `max_component_increase` / `max_hole_increase` /
`min_lcc_ratio` / `min_object_area_ref` / `max_hole_area_ref`.

## SEIG + Checker

The frozen, deterministic SEIG and Checker-v3 rule engine lives in `seig.py`; the complete
rulebook (priority / negation / claim decomposition / rewrite templates), the lexicons
(forbidden / safety / boundary / calibration), and the deterministic evidence prompt template
are materialised in `configs/`. Rules are frozen before independent human validation and are
not modified afterwards. `tools/export_checker_spec.py` regenerates the JSON specs from `seig.py`.

### SEIG reproducibility map

| Component | File / path | Purpose |
|---|---|---|
| Structured evidence extraction | `seig.py::SEIG.extract_evidence` | area / centroid / compactness / confidence / uncertainty from `Mr`, `p_bar`, `u` |
| Evidence vector dataclass | `seig.py::EvidenceVector` | typed container of the numeric evidence |
| Symbolic tuple construction | `seig.py::SEIG.symbolize` (+ `_area_label`, `_confidence_label`, `_uncertainty_label`, `_shape_label`, `_location_label`, `_quality_label`) | discretises evidence into symbolic labels |
| Discretisation thresholds | `seig.py::SEIGConfig`; mirrored in `stage6_seig_mllm_end2end.py` (`seig_config`) | area 0.01/0.05/0.15, compactness 1.30/1.80, confidence 0.60/0.80, uncertainty 0.25/0.50 |
| Node definitions | `seig.py::SEIG.build_graph` (`nodes`) | complete node set of the evidence interaction graph |
| Edge / relation rules | `seig.py::SEIG.build_graph` (`edges`) | complete edge set (`*-claim` relations) |
| Claim permission rules (allowed / cautious / prohibited) | `seig.py::SEIG.claim_permissions` → `seig.py::ClaimPermissionPlan` | per-evidence claim permissions |
| Full structured evidence prompt template | `seig.py::SEIG.render_prompt`; materialised in `configs/prompt_templates.json` | deterministic prompt T_ev (Table 5) |
| Claim decomposition | `seig.py::SEIG._split_claims`, `_normalize_heading_text`, `_is_section_heading` | markdown-aware claim splitting |
| Term lexicons | `configs/lexicons.json` (`lexicons`) | forbidden / safety / boundary / calibration term patterns |
| Synonym coverage | same file — alternation groups inside each `pattern` (e.g. `malignan(?:t|cy)\|benign\|dysplasia\|…`) | synonyms are encoded as regex alternations rather than a separate synonym list |
| Negation handling | `configs/checker_rulebook_v3.json::negation_policy`; `seig.py::SEIG._is_safe_diagnostic_disclaimer`, `_is_safe_treatment_disclaimer`, `_is_safe_anatomy_disclaimer`; `_negated_certainty` lexicon | safe vs unsafe negation |
| Conflicting-rule precedence | `configs/checker_rulebook_v3.json::priority` (R1 > R2 > R3 > R4 > R5 > R0); `seig.py::SEIG.checker_rulebook` | deterministic rule ordering |
| Deterministic rewrite templates | `configs/checker_rulebook_v3.json::rewrite_templates`; `seig.py::SEIG._rewrite_boundary` | text rewrites for unsupported anatomy / boundary calibration / prohibited claims |
| Unsupported-claim handling | rule R4 in `priority`; `seig.py::SEIG.claim_permissions` | no lesion-specific claims without valid lesion evidence |
| Anatomy-specific handling | rule R3; `_anatomy_terms` lexicon; `_is_safe_anatomy_disclaimer` | fine-grained subsites prohibited unless externally supplied |
| Diagnosis / pathology handling | rule R2; `_diagnosis_terms` lexicon; `_is_safe_diagnostic_disclaimer` | diagnostic vocabulary prohibited |
| Treatment / follow-up handling | rule R1; `_procedure_terms`, `_recommendation_cues`; `_is_safe_treatment_disclaimer` | treatment / follow-up vocabulary prohibited |
| Boundary-overconfidence handling | rule R5; `configs/lexicons.json::_boundary_terms`, `_boundary_descriptor`; `seig.py::SEIG._boundary_needs_calibration` | boundary calibration under moderate/high boundary uncertainty |
| Confidence-overstatement handling | `configs/lexicons.json::_strong_certainty`, `_negated_certainty`, `_caution_terms` | certainty vs cautious wording |
| Safety-note handling | `claim_permissions` → `"conservative safety note"`; `rewrite_templates::prohibited_diagnosis_or_treatment` | conservative safety note appended once |
| Idempotence requirement | `configs/checker_rulebook_v3.json::idempotence_requirement`; verified in `checker_v3_reaudit.py` | `verify(R*)` must add no new violation |

> **Reconstruction note (disclosed in code).** `seig.py::SEIGConfig` states explicitly that the
> manuscript does not fully specify a numeric rule for the final evidence-quality state `q`, nor
> an independent boundary-status threshold; this release makes those two mappings explicit and
> configurable. All other thresholds match the values reported in the manuscript.

## Cross-MLLM evaluation

`stage8_multimllm.py` implements Qwen2.5-VL-3B, InternVL2.5-2B, MiniCPM-V-2.6 and
HuatuoGPT-Vision backends with deterministic decoding (greedy; `do_sample=False`).
Model checkpoints are not redistributed and must be obtained from their original providers.

The HuatuoGPT-Vision backend uses the explicit model identifier
`FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL` (HuggingFace). Example:

    python stage8_multimllm.py --backend huatuo \
      --model "$WYSIWYR_DATA_ROOT/models/HuatuoGPT-Vision-7B-Qwen2.5VL" \
      --stage7 "$WYSIWYR_DATA_ROOT/wysiwyr_real/stage7_seig_controls" \
      --output "$WYSIWYR_DATA_ROOT/wysiwyr_real/stage8_multimllm" \
      --max-new-tokens 512

Add `--smoke` for a 1-case-per-dataset smoke run (Image-only / SEIG-only / Image + SEIG /
mismatched SEIG); omit it for the full cross-MLLM evaluation.

**Provenance.** The historical HuatuoGPT-Vision model-specific adapter that generated the
manuscript Table-30 numbers was **not retained**. The adapter released here is provided as a
**current reproducibility interface** only: it reuses the same Stage-8 image loading, SEIG
evidence, structured prompt template, frozen Checker-v3 and output schema, and adds only the
Huatuo-specific loading/inference interface. It has been **smoke-tested** (model loads; image
input is accepted; the Image-only and SEIG-conditioned conditions both produce valid text; output
files follow the repository schema). It should **not** be interpreted as the archived code that
originally produced the reported Table-30 values.

## Statistical analysis

The 18 prespecified segmentation-to-report association tests (**3 settings × 6 prespecified
pairs = 18**, computed in `stage6_seig_mllm_end2end.py`) receive a **single
Benjamini–Hochberg false-discovery-rate correction applied jointly across the full prespecified
family of 18 tests**; raw Spearman p-values and BH-adjusted q-values are written to
`segmentation_to_report_associations.csv`. Benjamini–Hochberg controls the false discovery
rate, **not** the family-wise error rate.
`analysis/finalize_usr_failure_analysis.py` applies BH-FDR to the reviewer-requested
USR-failure mechanism tests.

The six prespecified pairs are (ΔDice, ΔSymbolic match), (ΔBoundary Dice, ΔSymbolic match),
(ΔDice, ΔRaw violations), (ΔBoundary Dice, ΔRaw violations), (ΔHD95, ΔRaw violations) and
(ΔBoundary Dice, ΔBoundary overconfidence) — a prespecified subset of the nine conceivable
segmentation-metric × downstream-metric combinations, not a full 3 × 3 grid.
`analysis/reviewer2_r1_associations.py` recomputes these 18 tests with a 1,000-resample
dataset-stratified bootstrap CI and reproduces the stored ρ values to `max |Δρ| = 9.0e-17`.

## Environment

```bash
pip install -r requirements.txt
```

`requirements.txt` lists the minimal set of packages actually imported by the released code
(verified by scanning every import on 2026-09-17). MedSAM (`segment_anything`) and the MLLM
checkpoints are **not** installed by it; see the notes at the bottom of that file.

## Data preparation

Experiments use the public PraNet-style endoscopic suite (Kvasir-SEG, CVC-ClinicDB,
CVC-ColonDB, CVC-300, ETIS-LaribPolypDB). Download them from their official sources; they are
**not redistributed**. Set the data/artefact root before running:

```bash
export WYSIWYR_DATA_ROOT=/path/to/your/data_root
```

## Reproducing the manuscript

Every command below uses `$WYSIWYR_DATA_ROOT` as the root that contains `wysiwyr_real/`,
`models/` and `manifests/`. Commands marked *(author artifacts)* additionally require the
frozen artefacts described in [Expected artifact layout](#expected-artifact-layout); those
artefacts are not redistributed with the repository.

### 1. Environment

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/bowang-lab/MedSAM && pip install -e MedSAM
export WYSIWYR_DATA_ROOT=/path/to/your/data_root
```

INPUT: none.  OUTPUT: an environment able to import `torch`, `cv2`, `segment_anything`.

### 2. Data preparation

Place the public PraNet-style datasets so that the training pool is at
`$WYSIWYR_DATA_ROOT/wysiwyr_real/data/TrainDataset/{image,masks}` and each test set at
`$WYSIWYR_DATA_ROOT/wysiwyr_real/data/TestDataset/<name>/{image,mask}`.

INPUT: dataset downloads.  OUTPUT: the directory tree above.

### 3. Dataset splits

```bash
python run_real_medsam_2x2.py --prepare-only --root "$WYSIWYR_DATA_ROOT/wysiwyr_real"
```

INPUT: `data/TrainDataset`.  OUTPUT: `manifests/split_seed_<seed>.csv` (870 train / 580 val);
the released `manifests/split_seed_2023.csv` is the frozen manifest.

### 4. Train segmentation baseline

```bash
python run_real_medsam_2x2.py --root "$WYSIWYR_DATA_ROOT/wysiwyr_real" --seed 2023 \
  --epochs 50 --batch-size 1 --lr 1e-4 --weight-decay 0.01 --val-every 5 --patience 4 --amp
```

INPUT: split manifest + MedSAM ViT-B checkpoint.  OUTPUT: baseline checkpoint
(selected by lowest validation loss).

### 5. Train with ABLoss

Same driver, ABLoss branch (see `run_real_medsam_2x2.py` for the branch switch). Identical
optimiser/schedule/seed; only the loss differs.

INPUT/OUTPUT: as step 4, for the ABLoss checkpoint.

> **Note.** Steps 4–5 and the built-in 2x2 inference inside `run_real_medsam_2x2.py` are the
> *original* runner. Its `USRConfig` call site now **loads the frozen USR parameters from
> `configs/protocol.json -> "usr"` by default** (fixed 2026-09-17), so it no longer inherits the
> legacy class defaults (224 / 0.50 / 0.01) unless you explicitly pass `--legacy-usr-defaults`.
> Artefacts written by the old configuration (`<root>/predictions`, `<root>/reviewer2_results`)
> are **superseded** and must not be quoted as the manuscript's results; the reported 2x2 / USR
> numbers come from steps 6–8 below (`stage4_promptfree.py` + `stage5_calibrated.py`). See
> [Known limitations and legacy code](#known-limitations-and-legacy-code).

### 6. Train automatic coarse proposer

```bash
python stage4_promptfree.py --root "$WYSIWYR_DATA_ROOT/wysiwyr_real" --seed 2023 \
  --proposal-size 320 --proposal-epochs 30 --proposal-batch-size 8 \
  --proposal-lr 3e-4 --proposal-patience 6 --amp
```

INPUT: 870-image training split.  OUTPUT: `stage4_promptfree/proposal/checkpoints/best.pt`
(highest validation Dice) + prompt threshold/margin calibration.

### 7. Fully automatic MedSAM inference

```bash
python stage5_calibrated.py --root "$WYSIWYR_DATA_ROOT/wysiwyr_real" --seed 2023 --calibrate-only
python stage5_calibrated.py --root "$WYSIWYR_DATA_ROOT/wysiwyr_real" --seed 2023
```

INPUT: baseline/ABLoss checkpoints, proposer checkpoint, test sets.
OUTPUT: `stage5_calibrated/protocol_stage5.json` (frozen prompt+USR selection),
`stage5_calibrated/predictions/<dataset>/{baseline,abloss,usr,both}/{masks,prob,unc}`.

### 8. USR inference

USR is applied inside step 7 (`usr_run` in `stage5_calibrated.py`) using the validation-selected
configuration recorded in `protocol_stage5.json` (`reference_resolution 1024`, `T=8`,
`prune_prob_threshold 0.65`, `repair_prob_threshold 0.70`).

### 9. SEIG construction

```bash
python stage6_seig_mllm_end2end.py --root "$WYSIWYR_DATA_ROOT/wysiwyr_real" \
  --model "$WYSIWYR_DATA_ROOT/models/Qwen2.5-VL-3B-Instruct" --smoke
```

INPUT: `predictions/` masks/prob/unc.  OUTPUT: per-case SEIG evidence, symbolic tuples,
interaction graph, claim permissions and prompt (one JSON per case × source).

### 10. Report generation

Omit `--smoke` (and optionally set `--max-cases-per-dataset`) in the step-9 command.

INPUT: SEIG evidence + image + frozen MLLM.  OUTPUT:
`stage6_seig_mllm_end2end/case_level_results.csv`, `cases/<dataset>/<case>/<variant>.json`
(raw R0 and checker output R\*), `summary_by_dataset_variant.csv`,
`paired_report_contrasts_vs_medsam.csv`, `segmentation_to_report_associations.csv`.

### 11. Checker v3

```bash
python checker_v3_reaudit.py --source "$WYSIWYR_DATA_ROOT/wysiwyr_real/stage6_seig_mllm_end2end" \
  --output "$WYSIWYR_DATA_ROOT/wysiwyr_real/stage6_checker_v3"
```

INPUT: stored Stage-6 reports.  OUTPUT: `checker_v3_idempotence_failures.csv` (must be empty),
`checker_v3_human_validation_BLINDED_FOR_ANNOTATION.csv` (400-claim validation sample, blinded),
and the separate checker key file.

Scoring against human labels:

```bash
python score_human_validation.py --human <annotated_blinded.csv> --key <checker_key.csv> \
  --output <scores.json>
```

INPUT: blinded annotation sheet + checker key.  OUTPUT: precision / recall / F1 per class,
macro-F1 and overall accuracy (JSON).

### 12. Evaluation

```bash
python run_reviewer2_experiments.py \
  --gt     "$WYSIWYR_DATA_ROOT/wysiwyr_real/data/TestDataset/<ds>/mask" \
  --baseline "$WYSIWYR_DATA_ROOT/wysiwyr_real/stage5_calibrated/predictions/<ds>/baseline/masks" \
  --abloss   "$WYSIWYR_DATA_ROOT/wysiwyr_real/stage5_calibrated/predictions/<ds>/abloss/masks" \
  --usr      "$WYSIWYR_DATA_ROOT/wysiwyr_real/stage5_calibrated/predictions/<ds>/usr/masks" \
  --both     "$WYSIWYR_DATA_ROOT/wysiwyr_real/stage5_calibrated/predictions/<ds>/both/masks" \
  --output "$WYSIWYR_DATA_ROOT/wysiwyr_real/stage5_calibrated/reviewer2_results/<ds>" \
  --bootstrap 2000 --seed 2023
```

INPUT: GT + four prediction mask folders (+ optional `--prob-*` / `--unc-*`).
OUTPUT: `case_segmentation_metrics.csv`, `summary_segmentation_metrics.csv`,
`factorial_2x2_effects.csv`, `paired_tests.csv`, `usr_pixel_audit.csv`,
`usr_worsen_summary.csv`, `risk_coverage_*.csv`, `uncertainty_summary.csv`.

Self-test without any data:

```bash
python run_reviewer2_experiments.py --smoke-test --output /tmp/wysiwyr_smoke
```

### 13. Reviewer-requested analyses

Each entry lists the command, its inputs and its outputs. All analyses are deterministic and
read frozen artefacts; none re-trains or re-runs the MLLM.

| Analysis | Command | INPUT | OUTPUT |
|---|---|---|---|
| 2×2 ABLoss × USR factorial (main effects + interaction, dataset-stratified bootstrap CI) | `python run_reviewer2_experiments.py --gt ... --baseline ... --abloss ... --usr ... --both ... --output <out> --bootstrap 2000 --seed 2023` | GT + 4 mask folders | `factorial_2x2_effects.csv`, `paired_tests.csv` |
| Uncertainty calibration / pixel error detection | same command with `--prob-<variant>` and `--unc-<variant>` | probability + uncertainty maps | `uncertainty_case_metrics.csv`, `uncertainty_summary.csv`, `risk_coverage_*.csv` |
| USR semantic pixel audit + case-level worsening | `python analysis/reviewer2_r4_usr_pixel_audit.py` | `usr_failure_analysis_20260913/all_usr_pixel_audit.csv` (+ mask re-derivation CSV) | `results/reviewer2/reviewer2_r4_usr_case_audit.csv`, `reviewer2_r4_usr_pixel_case_summary.csv/.tex` |
| USR pixel-accounting independent re-derivation | `python analysis/reviewer2_r4_mask_verify.py` | prediction masks + GT masks | `results/reviewer2/reviewer2_r4_mask_recompute_verification.csv/.json` |
| End-to-end propagation | `python stage6_seig_mllm_end2end.py --root ... --model ...` | SEIG evidence + frozen MLLM | `case_level_results.csv`, `summary_macro_variant.csv`, `paired_report_contrasts_vs_medsam.csv` |
| 18 case-level associations (Spearman + 1,000 stratified bootstrap CI + joint BH-FDR) | `python analysis/reviewer2_r1_associations.py` | `stage6_seig_mllm_end2end/case_level_results.csv` | `results/reviewer2/reviewer2_r1_spearman_18.csv/.tex`, `r1_case_level_change_scores.csv` |
| SEIG / Checker validation | `python checker_v3_reaudit.py --source ... --output ...` then `python score_human_validation.py --human ... --key ...` | Stage-6 reports + human annotations | `checker_v3_idempotence_failures.csv`, blinded sample, scores JSON |
| Visual-vs-SEIG controls (image-only / SEIG-only / image+SEIG / mismatched) | `python stage7_seig_controls.py --root ... --model ... --seed 2023` | Stage-6 evidence + images | `stage7_seig_controls/` per-case JSON, `mismatch_pairs.csv`, `protocol_stage7.json` |
| Cross-MLLM replication | `python stage8_multimllm.py --backend <reuse-qwen\|internvl\|minicpm\|huatuo> --model <ckpt> --stage7 ... --output ...` (then `--backend aggregate`) | Stage-7 controls + MLLM checkpoints | `stage8_multimllm/` per-case JSON + `protocol_stage8.json` |
| Threshold sensitivity (11 configurations) | `python stage9_seig_threshold_sensitivity.py --root ... --model ... --seed 2023` | Stage-6 evidence + frozen MLLM | `stage9_seig_threshold_sensitivity/` configuration sweep tables |
| Clinician mixed-effects analysis (cumulative-link mixed model, crossed clinician/case random intercepts) | `analysis/clinician_mixed_effects.R` (`ordinal::clmm`) | completed clinician rating table — **not located in the current reproducibility archive**; schema in `configs/clinician_evaluation_protocol.json` | OR / 95% CI / p per condition, `primary_condition_or.csv`, `experienced_only_condition_or.csv`, `condition_by_experience_interaction.csv` |
| Clinician inter-rater agreement (ICC / Fleiss' kappa / raw agreement) | `analysis/clinician_agreement.py` | same rating table (long format: subject, rater, category) | `clinician_agreement.json`; self-test: `python analysis/clinician_agreement.py --smoke-test` |
| Checker-v3 validation metrics + bootstrap CI + Cohen's kappa | `score_human_validation.py` | blinded sheet with `human_status` filled + `checker_v3_human_validation_KEY_DO_NOT_SHOW_ANNOTATOR.csv` | accuracy, macro-F1 (with 95 % bootstrap CI), per-class P/R/F1, confusion matrix, Cohen's kappa, raw pre-adjudication agreement; self-test: `python score_human_validation.py --smoke-test` |
| Runtime analysis | from `stage5_calibrated/predictions/<ds>/latency.csv` and the per-case `generation_latency_s.json` fields | latency CSV / JSON | aggregated runtime tables |
| SEIG spec regeneration | `python tools/export_checker_spec.py` | `seig.py` | `configs/lexicons.json`, `configs/prompt_templates.json` |

> The scripts under `analysis/` accept optional `--data-root` / `--out-dir` overrides and fall back
> to `$WYSIWYR_DATA_ROOT`; they write into `results/reviewer2/`. `--help` is safe — it prints usage
> and exits without running anything. Their code-path self-tests are
> `python analysis/clinician_agreement.py --smoke-test` and
> `python score_human_validation.py --smoke-test`.

### Frozen generation length (please read before quoting a number)

Every reported report-generation experiment was produced with **`max_new_tokens = 384`,
`do_sample = False`**. This is recorded in the frozen protocol files *and* in the per-case JSON
of every generated report:

| Experiment | rows with a `decode` field | frozen `max_new_tokens` | evidence |
|---|---|---|---|
| Stage 6 — main end-to-end report generation | 3,990 | **384** | `stage6_seig_mllm_end2end/protocol_stage6.json` + all 3,990 per-case JSON `decode` |
| Stage 7 — visual-vs-SEIG controls | 3,193 | **384** | `protocol_stage7.json` + per-case JSON |
| Stage 8 — cross-MLLM (Qwen2.5-VL-3B / InternVL2.5-2B / MiniCPM-V-2.6) | 2,401 | **384** | `protocol_stage8.json` + per-case JSON |
| Stage 9 — SEIG threshold sensitivity | 2,201 | **384** | `protocol_stage9.json` + per-case JSON |
| Stage 8 smoke — HuatuoGPT-Vision-7B | — | **512** | `stage8_smoke/protocol_stage8.json` (`HUATUO_MAX_NEW_TOKENS = 512`) |

`512` appears **only** for the HuatuoGPT-Vision backbone, which needed a longer budget; the code
default is 192 (`stage6`) / 384 (`stage7`–`stage9`). If the manuscript states a single generation
length of 512 for the main report generation, that conflicts with the frozen artefacts and should
be corrected to 384 (or scoped explicitly to the HuatuoGPT-Vision setting). Nothing in this
repository was changed to match a different manuscript number.

### 14. Expected artifact layout

```
$WYSIWYR_DATA_ROOT/
├── models/
│   ├── Qwen2.5-VL-3B-Instruct/            # required for stages 6-9
│   └── HuatuoGPT-Vision-7B-Qwen2.5VL/     # optional, cross-MLLM
└── wysiwyr_real/
    ├── data/TrainDataset/{image,masks}
    ├── data/TestDataset/<name>/{image,mask}
    ├── manifests/split_seed_2023.csv
    ├── checkpoints/{baseline,abloss}/best.pt
    ├── stage4_promptfree/proposal/checkpoints/best.pt
    ├── stage5_calibrated/
    │   ├── protocol_stage5.json           # frozen prompt + USR selection
    │   ├── predictions/<ds>/{baseline,abloss,usr,both}/{masks,prob,unc}
    │   └── reviewer2_results/<ds>/        # canonical 2x2 / USR pixel accounting
    ├── usr_failure_analysis_20260913/     # frozen USR-failure analysis (author artifacts)
    ├── stage6_seig_mllm_end2end/          # case_level_results.csv, cases/, associations
    ├── stage6_checker_v3/
    ├── stage7_seig_controls/
    ├── stage8_multimllm/
    └── stage9_seig_threshold_sensitivity/
```

### 15. Reproducibility notes and limitations

See [Reproducibility tiers](#reproducibility-tiers),
[Known limitations and legacy code](#known-limitations-and-legacy-code) and
[Data availability and restrictions](#data-availability-and-restrictions).

## Reproducibility tiers

| Tier | What | Status |
|---|---|---|
| **A. Fully reproducible from public datasets** | SEIG construction rules, lexicons, prompt template, Checker-v3 rule engine and its idempotence test, SEIG threshold sensitivity (symbolic part), 2×2 / association *statistics* given masks | **Yes** — code + frozen configs are in this repository |
| **B. Reproducible given model weights** | MedSAM baseline/ABLoss training, prompt-free proposer training, fully automatic inference, USR inference, report generation, cross-MLLM evaluation | **Yes, if** you obtain the official MedSAM ViT-B checkpoint and the MLLM checkpoints yourself |
| **C. Analysis reproducible from released frozen artifacts** | USR semantic pixel audit, USR mask re-derivation, 18 case-level associations, end-to-end propagation tables, Checker-v3 validation scoring | **Yes, given the author-provided artefacts** (`stage5_calibrated/`, `usr_failure_analysis_20260913/`, `stage6_*`). These artefacts are **not** redistributed with the repository — request them from the authors, or regenerate them via tiers A/B. |
| **D. Code complete, records not in archive** | Clinician evaluation (completed rating records not located); Checker-v3 human validation (completed 400-claim labels not located); Tumour-30 / internal subset (not distributed — institutional privacy and ethical restrictions) | **Code: yes** (analysis scripts released and self-testable). **Numbers: no** — the completed human-evaluation records are not part of the current archive, so those manuscript values cannot be independently recomputed from it. |

We deliberately make **no** "one command reproduces every result" claim: tiers B–D require
external weights or non-public data.

## Known limitations and legacy code

1. **`run_real_medsam_2x2.py` uses legacy USR defaults.** Its `core.USRConfig(...)` call site
   does not pass `reference_resolution`, `prune_prob_threshold` or `repair_prob_threshold`, so it
   inherits the original class defaults (224 px / 0.50 / 0.01) rather than the frozen
   manuscript values (1024 px / 0.65 / 0.70). Artefacts written by that runner
   (`<root>/predictions`, `<root>/reviewer2_results`) are **superseded** and must not be quoted
   as the manuscript's results. **The current manuscript-reproduction pathway explicitly loads the
   frozen USR configuration from `configs/protocol.json`** (`--protocol` overrides the file), so it
   can no longer inherit the legacy values, **and historical superseded artefacts may have been
   produced with earlier legacy defaults.** The old behaviour is still reachable via the explicit
   `--legacy-usr-defaults` flag, which `--help` documents as being for re-deriving the superseded
   artefacts only. No historical artefact was created or modified by this change.
2. **`USRConfig` class defaults are legacy.** They are kept for backwards compatibility and are
   labelled `LEGACY` in the source. Read the frozen values from `configs/protocol.json -> "usr"`.
3. **`USRConfig.initial_prompt_mode = "full_image_box"` is dead code for the reported results.**
   It is only used by the `run_prompt()` fallback when no `coarse_mask` is supplied; the frozen
   test-time pathway always supplies a prompt-free-proposer-derived box. No reported result uses
   a full-image-box bootstrap.
4. **`run_real_medsam_2x2.py` records `baseline_init_prompt="full_image_box"` in its per-case
   `prompt_protocol.csv`.** This describes the *internal coarse+refine initialisation*, not the
   test-time prompt source; the prediction-derived box is recorded alongside it in the
   `*_prediction_box` columns. Do not read that column as evidence of full-image-box prompting.
   (The column label is intentionally left unchanged so that existing artefacts stay readable.)
5. **`analysis/*` scripts now take optional `--data-root` / `--out-dir` arguments and are
   side-effect free under `--help`.** *(Fixed 2026-09-17; previously `--help` executed the
   analysis.*) `analysis/step0_audit.py` is an internal audit helper and is deliberately **not**
   part of the release set.
6. **`tools/export_checker_spec.py` is line-ending safe.** *(Fixed 2026-09-17; it previously
   rewrote `configs/*.json` with different line endings and left the working tree dirty.)*
   The writers now preserve the on-disk line ending, and `--check` verifies drift without writing.
   Verified: regenerating produces byte-identical files (`md5` unchanged) and the tree stays clean.
7. **`results/` is git-ignored.** Generated analysis outputs are not part of the committed tree.

## Data availability and restrictions

* **Public datasets** (Kvasir-SEG, CVC-ClinicDB, CVC-ColonDB, CVC-300, ETIS-LaribPolypDB) are
  downloaded from their official sources and are **not redistributed** here.
* **No GT mask or GT box is used** for automatic test-time MedSAM prompting.
* **Test sets are not used** for hyperparameter tuning or checkpoint selection.
* **Internal tumour data are not distributed.** Internal Tumour-30 data are not distributed due
  to institutional privacy and ethical restrictions. The code, the expected directory layout and
  the evaluation pipeline for that subset are released, but the images themselves are not, and
  the corresponding experiments therefore cannot be reproduced from public resources alone.
  `run_real_medsam_2x2.py` keeps `--skip-tumor30` enabled by default and refuses to treat the
  archive's generic `labeled/` subset (435 pairs) as the manuscript's Tumour-30 subset (162
  pairs) without a verified mapping.
* **Clinician rating records are not included in this archive.** The original *completed*
  clinician-rating records were searched for on the analysis server and in the local project
  archives and were **not located**. The blinded rating sheets that do exist are empty templates
  (all rating columns blank). The evaluation design (`configs/clinician_evaluation_protocol.json`),
  the analysis code (`analysis/clinician_mixed_effects.R`, `analysis/clinician_agreement.py`) and
  the expected input schema are released. Any future sharing of human-evaluation data must comply
  with the applicable ethical and privacy requirements.
* **Checker-v3 human labels are not included in this archive.** The released blinded 400-claim
  sheet (`checker_v3_human_validation_BLINDED_FOR_ANNOTATION.csv`) is an **un-annotated template**
  (`human_status` and the other annotation columns are empty), and no completed annotation file
  was located in the archive. The complete scoring code is released
  (`score_human_validation.py`). The same ethical/privacy requirements apply to any future
  sharing of these annotations.
* **No patient identifiers, API keys, tokens or credentials are present in this repository.**
  All data/artefact roots are resolved from the `WYSIWYR_DATA_ROOT` environment variable.

## Reproducibility statements

- Datasets and pretrained model weights are **not redistributed**.
- **No GT mask or GT box is used** for automatic test-time MedSAM prompting.
- Test sets are **not used** for hyperparameter tuning or checkpoint selection.
- SEIG / Checker rules are **deterministic and frozen**.
- MLLMs are **frozen** and use **deterministic decoding** in the main experiments.
- USR parameters are calibrated on validation data and frozen before test evaluation.

## Not released / records not in the current archive

The following are **code-complete but record-blocked**: the analysis code is released, but the
corresponding *completed* human-evaluation records were **not located in the current
reproducibility archive** (searched on the analysis server and in the local project archives on
2026-09-17). Consequently the affected manuscript numbers cannot be independently recomputed from
what is published here. This is a statement about what the archive contains — not a claim about
why any particular record is absent.

* **Clinician rating records.** `analysis/clinician_mixed_effects.R` (cumulative-link mixed model
  via `ordinal::clmm`, condition fixed effect, crossed clinician and case random intercepts,
  experienced-only sensitivity subset, condition × experience interaction, OR / 95 % CI / p) and
  `analysis/clinician_agreement.py` (ICC(2,1)/ICC(2,k)/ICC(3,1), Fleiss' kappa, raw agreement)
  are released. The original completed clinician-rating records are **not included in this
  archive**; the blinded rating sheets found on disk are empty templates. The frozen design is
  recorded in `configs/clinician_evaluation_protocol.json` and the expected input schema is
  documented in both scripts. **The manuscript's clinician OR / CI / p / ICC / kappa values
  therefore cannot be independently recomputed from the current public archive.** Any future
  sharing of human-evaluation data must comply with the applicable ethical and privacy
  requirements.
* **Checker-v3 human labels.** `score_human_validation.py` computes per-class precision/recall/F1,
  macro-F1, accuracy, a **case-level bootstrap 95 % CI for macro-F1 and accuracy**, Cohen's kappa
  (4-class and binary-violation), the **4×4 confusion matrix** and the **raw pre-adjudication
  agreement** between independent annotators — verified on synthetic labels via `--smoke-test`.
  The repository ships the blinded 400-claim sample
  (`checker_v3_human_validation_BLINDED_FOR_ANNOTATION.csv`) and the checker key, but the released
  blinded sheet is an **un-annotated template** (`human_status` and the other annotation columns
  are empty) and no completed annotation file was located. The manuscript's
  accuracy / macro-F1 / CI / raw-agreement / kappa figures **cannot be independently recomputed
  from the current public archive**; they become reproducible once the completed labels are
  available. The manuscript values themselves are unchanged and none of them is hardcoded in the
  scoring script.
* **The 400-claim annotation sheets** themselves are not included in this archive (free-text model
  output plus human labels). `checker_v3_reaudit.py` releases the *sample construction* and the
  blinded schema.
* See also `MANUSCRIPT_REPRODUCIBILITY.md` for the manuscript-item → code mapping.

## Reproducibility notes

The HuatuoGPT-Vision cross-MLLM adapter is now included as a current reproducibility interface
(see "Cross-MLLM evaluation"). The historical adapter used to generate the manuscript Table-30
HuatuoGPT-Vision numbers was not retained and is not claimed to be reproduced here.

## License

Released for peer review. Third-party components (MedSAM, MLLMs, datasets) remain under their
own respective licenses.
