# MANUSCRIPT_REPRODUCIBILITY.md

Mapping from manuscript items to repository entry points, for
*What You Segment Is What You Report* (ESWA).

> **How to read this table.** The "Manuscript item" column names the table/figure **by content**,
> because the submitted and revised versions use different table numbers (in the submitted PDF,
> for example, Table 19 is the average report length and Table 20 is the cross-MLLM evaluation).
> Reconcile the numbers against the version you are reviewing; the *content* and the *repository
> entry* are the load-bearing columns.
>
> `$R` = repository root, `$D` = `$WYSIWYR_DATA_ROOT`, `$W` = `$D/wysiwyr_real`.
> Cells marked *(artifacts)* additionally need author-provided frozen artefacts — see
> [Reproducibility tiers](README.md#reproducibility-tiers) in the README.
> **No command below re-trains a model unless it is a `--*train*`/`--auto` command.**

## Segregation front end and 2×2 factorial

| Manuscript item | Repository entry | Command | Expected output |
|---|---|---|---|
| Segmentation protocol / hyperparameters | `configs/protocol.json`; `run_real_medsam_2x2.py`; `wysiwyr_autodl_allinone.py::ABLoss` | — (configuration) | — |
| Split definition (870 / 580) | `manifests/split_seed_2023.csv`; `run_real_medsam_2x2.py::make_split_manifest` | `python run_real_medsam_2x2.py --prepare-only --root $W` | `manifests/split_seed_2023.csv` |
| Baseline / ABLoss training curves + val Dice | `run_real_medsam_2x2.py` | `python run_real_medsam_2x2.py --root $W --seed 2023 --epochs 50 --batch-size 1 --lr 1e-4 --weight-decay 0.01 --val-every 5 --patience 4 --amp` | per-epoch logs, `checkpoints/{baseline,abloss}/best.pt` |
| **Table 7 — segmentation results (MedSAM / +ABLoss)** *(artifacts)* | `run_reviewer2_experiments.py` (dataset-level 2×2) | `python run_reviewer2_experiments.py --gt $W/data/TestDataset/<ds>/mask --baseline $W/stage5_calibrated/predictions/<ds>/baseline/masks --abloss …/abloss/masks --usr …/usr/masks --both …/both/masks --output $W/stage5_calibrated/reviewer2_results/<ds>` | `summary_segmentation_metrics.csv`, `case_segmentation_metrics.csv` |
| **Tables 10–11 — ABLoss / USR main effects and interaction** *(artifacts)* | `run_reviewer2_experiments.py::_factorial_effects`; `wysiwyr_autodl_allinone.py::factorial_2x2_effects` | same command as Table 7 | `factorial_2x2_effects.csv` (main effects + A×B interaction) |
| Paired significance for the 2×2 | `run_reviewer2_experiments.py::_paired_wilcoxon`, `_bootstrap_mean_diff` | same command with `--bootstrap 2000 --seed 2023` | `paired_tests.csv` (paired Wilcoxon p, bootstrap 95% CI) |
| Segmentation significance family + one joint BH-FDR per declared family scope | `analysis/recompute_table8_fdr.py` | `python analysis/recompute_table8_fdr.py --data-root $W --out-dir reproducibility_outputs` | `reproducibility_outputs/table8_wilcoxon_bh_fdr.csv` / `.json` (raw p and BH q per family scope; `table8` = 125 tests), `table8_reproduction_check.csv` |
| Recent-method comparison (SAM2-UNet / SEPNet / MVSNet / multi-view simulation) | **NOT RETAINED** — no code, checkpoint reference, configuration, prediction, log or provenance record exists in this repository (verified 2026-09-17) | — | — (see README "Recent-method comparison provenance") |
| **Table 12 — fully automatic prompting / proposer** *(artifacts)* | `stage4_promptfree.py`; `configs/protocol.json -> coarse_proposer_g_psi` | `python stage4_promptfree.py --root $W --seed 2023 --proposal-size 320 --proposal-epochs 30 --proposal-batch-size 8 --proposal-lr 3e-4 --proposal-patience 6 --amp` | `stage4_promptfree/proposal/checkpoints/best.pt`, `proposal/test_summary_<ds>.csv`, `prompt_protocol.csv` |
| **Tables 13–14 — USR before/after aggregate metrics** *(artifacts)* | `stage5_calibrated.py::usr_run`; `wysiwyr_autodl_allinone.py::USR.rectify` | `python stage5_calibrated.py --root $W --seed 2023` | `protocol_stage5.json`, `predictions/<ds>/{usr,both}/{masks,prob,unc}`, `reviewer2_results/<ds>/summary_segmentation_metrics.csv` |
| USR validation-only calibration grid | `stage5_calibrated.py::calibrate_usr` | `python stage5_calibrated.py --root $W --seed 2023 --calibrate-only` | `stage5_calibrated/calibration/usr_grid.csv`, `selected_usr_params.json` |
| **USR semantic audit table** *(artifacts)* | `analysis/reviewer2_r4_usr_pixel_audit.py`; `analysis/reviewer2_r4_mask_verify.py` | `python $R/analysis/reviewer2_r4_mask_verify.py` then `python $R/analysis/reviewer2_r4_usr_pixel_audit.py` | `results/reviewer2/reviewer2_r4_usr_case_audit.csv`, `reviewer2_r4_usr_pixel_case_summary.csv/.tex`, `reviewer2_r4_mask_recompute_verification.csv/.json` |
| USR failure-mechanism analysis (logistic / group comparison) *(artifacts)* | `analysis/analyze_usr_failure.py`; `analysis/finalize_usr_failure_analysis.py` | `python $R/analysis/analyze_usr_failure.py` then `cd $W/usr_failure_analysis_20260913 && python finalize_usr_failure_analysis.py` | `all_case_segmentation_metrics.csv`, `all_usr_pixel_audit.csv`, `usr_case_level_failure.csv`, `usr_failure_logistic_regression.csv`, `usr_semantic_mechanism_final.csv` |

## Reliability / uncertainty

| Manuscript item | Repository entry | Command | Expected output |
|---|---|---|---|
| **Uncertainty reliability table** (AUROC / Brier / ECE / risk–coverage) *(artifacts)* | `run_reviewer2_experiments.py` steps 5–6 (requires `--prob-*`, `--unc-*`) | `python run_reviewer2_experiments.py … --prob-baseline $W/stage5_calibrated/predictions/<ds>/baseline/prob --unc-baseline …/baseline/unc --prob-abloss … --unc-abloss … --prob-usr … --unc-usr … --prob-both … --unc-both … --output <out>` | `uncertainty_case_metrics.csv`, `uncertainty_summary.csv`, `risk_coverage_{baseline,abloss,usr,both}.csv` |
| Bernoulli-entropy uncertainty definition | `wysiwyr_autodl_allinone.py::USR.bernoulli_entropy`; `USR.aggregate` | — (code) | — |
| CVC-300 ASSD increase after USR (reviewer query) | `analysis/reviewer2_r4_usr_pixel_audit.py` (per-case metrics) | as above | `reviewer2_r4_usr_case_audit.csv` (`assd`-related columns via `all_case_segmentation_metrics.csv`) |

## End-to-end propagation and associations

| Manuscript item | Repository entry | Command | Expected output |
|---|---|---|---|
| Five mask sources by the same SEIG + frozen MLLM | `stage6_seig_mllm_end2end.py` (`VARIANT_LABELS`: baseline / abloss / usr / both / gt-oracle) | `python stage6_seig_mllm_end2end.py --root $W --model $D/models/Qwen2.5-VL-3B-Instruct` | `cases/<ds>/<case>/<variant>.json`, `case_level_results.csv` |
| **End-to-end propagation table** (paired contrasts vs MedSAM) *(artifacts)* | `stage6_seig_mllm_end2end.py::stratified_bootstrap_delta` | same command (aggregation runs automatically) | `paired_report_contrasts_vs_medsam.csv`, `summary_macro_variant.csv` |
| Symbolic match / raw violations / forbidden / unsupported / boundary overconfidence per case | `case_level_results.csv` columns `symbolic_match_fraction`, `raw_violation_count`, `raw_forbidden_count`, `raw_anatomy_unsupported_count`, `raw_boundary_overconfidence_count`; produced by `seig.py` + `SEIG.verify_report` | same command | `case_level_results.csv` |
| **18-association table** (3 settings × 6 prespecified pairs, Spearman + 1,000 stratified bootstrap CI + joint BH-FDR) *(artifacts)* | `stage6_seig_mllm_end2end.py::aggregate` (writes ρ + BH q); `analysis/reviewer2_r1_associations.py` (adds bootstrap CI, reproduces ρ to 9.0e-17) | `python $R/analysis/reviewer2_r1_associations.py` | `results/reviewer2/reviewer2_r1_spearman_18.csv/.tex`, `results/reviewer2/r1_case_level_change_scores.csv` |
| SEIG component ablation | `stage6_seig_mllm_end2end.py` (A1–A8 settings via CLI `--max-cases-per-dataset` / variant selection); see the Stage-6 protocol file | as above | `summary_by_dataset_variant.csv`, `summary_macro_variant.csv` |

## Controls and robustness

| Manuscript item | Repository entry | Command | Expected output |
|---|---|---|---|
| **Visual-vs-SEIG controls** (image-only / SEIG-only / image+SEIG / mismatched SEIG) *(artifacts)* | `stage7_seig_controls.py` (`CONDITIONS`) | `python stage7_seig_controls.py --root $W --model $D/models/Qwen2.5-VL-3B-Instruct --seed 2023` | `stage7_seig_controls/` per-case JSON, `mismatch_pairs.csv`, `protocol_stage7.json` |
| Mismatched-evidence donor pairing (target image unchanged, within-dataset donor, differing location+area, deterministic seed, one-to-one derangement) | `stage7_seig_controls.py::build_mismatch_pairs`; `mismatch_policy` in `protocol_stage7.json` | same command (`--seed 2023`) | `mismatch_pairs.csv` (donor id, target/donor location+area, permutation flag) |
| **Cross-MLLM controls** (Qwen2.5-VL-3B / InternVL2.5-2B / MiniCPM-V-2.6 / HuatuoGPT-Vision) | `stage8_multimllm.py` (`--backend`) | `python stage8_multimllm.py --backend reuse-qwen --stage7 $W/stage7_seig_controls --output $W/stage8_multimllm` then `--backend internvl|minicpm|huatuo --model <ckpt>` then `--backend aggregate` | `stage8_multimllm/` per-case JSON, `protocol_stage8.json` |
| **Threshold sensitivity** (11 configurations) | `stage9_seig_threshold_sensitivity.py` (default, global lenient/strict, area ±20%, compactness ±10%, confidence ±0.05, uncertainty ±0.05) | `python stage9_seig_threshold_sensitivity.py --root $W --model $D/models/Qwen2.5-VL-3B-Instruct --seed 2023` | `stage9_seig_threshold_sensitivity/` sweep tables (symbolic-tuple change %, permission change %, violation counts) |
| Symbolic-only threshold sensitivity (no MLLM) | same script, `--symbolic-only` | `python stage9_seig_threshold_sensitivity.py --root $W --symbolic-only` | sweep tables without report-level metrics |
| **Runtime table** *(artifacts)* | `predictions/<ds>/latency.csv` (`stage5_calibrated.py`); per-case `generation_latency_s.json` / `generated_tokens` in Stage-6+ records | aggregation from those files | aggregated per-stage latency / throughput |

## Checker validation

| Manuscript item | Repository entry | Command | Expected output |
|---|---|---|---|
| **Checker v3 validation table** (precision / recall / F1 / macro-F1 / accuracy) *(artifacts)* | `checker_v3_reaudit.py` (sample construction + idempotence); `score_human_validation.py` (scores) | `python checker_v3_reaudit.py --source $W/stage6_seig_mllm_end2end --output $W/stage6_checker_v3` then `python score_human_validation.py --human <annotated_blinded.csv> --key $W/stage6_checker_v3/checker_v3_human_validation_KEY_DO_NOT_SHOW_ANNOTATOR.csv --output <scores.json>` | `checker_v3_idempotence_failures.csv` (empty), `checker_v3_human_validation_BLINDED_FOR_ANNOTATION.csv`, human scores JSON |
| Checker idempotence | `checker_v3_reaudit.py` (`idempotent` flag; exits non-zero on failure) | same command | `checker_v3_idempotence_failures.csv` |
| Checker rulebook / lexicons / prompt template | `configs/checker_rulebook_v3.json`, `configs/lexicons.json`, `configs/prompt_templates.json`; regenerate with `tools/export_checker_spec.py` | `python tools/export_checker_spec.py` | regenerated JSON specs |
| Bootstrap CI of the validation metrics | `score_human_validation.py` (adds case-level bootstrap CI for macro-F1 and accuracy, Cohen's kappa, confusion matrix, raw pre-adjudication agreement) | `python score_human_validation.py --human <annotated_blinded.csv> --key <checker_key.csv> --output <scores.json> [--bootstrap 2000 --seed 2023]` | `human_validation_scores.json` (see README "Not released / data-only restrictions" for why the manuscript numbers cannot be re-derived yet) |

## Human / clinician study

| Manuscript item | Repository entry | Command | Expected output |
|---|---|---|---|
| Blinded rating sheet construction | `analysis/build_blinded_annotation.py` (blinded package + private key, seed 20260913) | `python $R/analysis/build_blinded_annotation.py` | `difficulty_annotation_20260913/blinded_package`, `private_key` |
| Clinician cumulative-link mixed-effects model (clinician random intercept, case random intercept) | `analysis/clinician_mixed_effects.R` (`ordinal::clmm`); design in `configs/clinician_evaluation_protocol.json` | `Rscript analysis/clinician_mixed_effects.R --data <ratings.csv> --out <outdir>` | `primary_condition_or.csv` (OR / 95 % CI / p), `primary_random_effects.csv` |
| Experience-stratified sensitivity analysis (18 experienced clinicians) | same script, `experience == "experienced"` subset | same command | `experienced_only_condition_or.csv` |
| Method × experience interaction analysis | same script, `condition * experience` model | same command | `condition_by_experience_interaction.csv` |
| ICC / Fleiss kappa / raw agreement | `analysis/clinician_agreement.py` (pure Python; no R needed) | `python analysis/clinician_agreement.py --data <ratings_long.csv> --out <outdir>` | `clinician_agreement.json` (ICC(2,1), ICC(2,k), ICC(3,1), Fleiss' kappa, raw agreement) |

> **Records not in this archive.** The original *completed* clinician-rating records were searched
> for on the analysis server and in the local project archives and were **not located**; the
> blinded rating sheets found on disk are empty templates. Both scripts are released and
> self-testable (`--smoke-test` for the Python one), but the manuscript's
> OR / CI / p and ICC / kappa values **cannot be independently recomputed from the current public
> archive** and are not claimed to be reproduced here. Input schema:
> `clinician_id, case_id, dataset, condition, score, experience` (long format), documented in
> `configs/clinician_evaluation_protocol.json`. Any future sharing of human-evaluation data must
> comply with the applicable ethical and privacy requirements.

## Not released / records not in the current archive

Code is released for every item below, but the corresponding *completed* human-evaluation records
were **not located in the current reproducibility archive**, so the affected manuscript numbers
cannot be independently recomputed from it. This states what the archive contains, not why a
particular record is absent.

* The original **completed clinician-rating records** (searched 2026-09-17 on the analysis server
  and in the local project archives; not found). Design recorded in
  `configs/clinician_evaluation_protocol.json`.
* The **completed Checker-v3 human labels**: the released
  `checker_v3_human_validation_BLINDED_FOR_ANNOTATION.csv` is an un-annotated template
  (`human_status` and the other annotation columns are empty) and no completed annotation file was
  located, so accuracy / macro-F1 / CI / raw pre-adjudication agreement / Cohen's kappa cannot be
  recomputed. The manuscript values are unchanged and none is hardcoded in the scoring script.
* Internal **Tumour-30** images are **not distributed due to institutional privacy and ethical
  restrictions**.
* Model checkpoints (MedSAM ViT-B, Qwen2.5-VL-3B, InternVL2.5-2B, MiniCPM-V-2.6,
  HuatuoGPT-Vision-7B) and the public endoscopic datasets.
* Frozen prediction / artefact trees under `$W` (12 GB-scale); available from the authors on
  request, or regenerable via the tier A/B commands in the README.

## Frozen decoding length

All reported report-generation experiments used `do_sample = False` and
`max_new_tokens = 384` (Stage 6: 3,990/3,990 per-case JSONs; Stage 7: 3,193; Stage 8: 2,401;
Stage 9: 2,201). `512` appears only for the HuatuoGPT-Vision backbone
(`HUATUO_MAX_NEW_TOKENS = 512`). See the README section "Frozen generation length".
