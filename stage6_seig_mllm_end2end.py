#!/usr/bin/env python3
"""
WYSIWYR Stage 6: mask-source -> fixed SEIG -> fixed Qwen2.5-VL end-to-end experiment.

Reviewer-facing design
----------------------
For every case, the original endoscopic image and all downstream settings are fixed.
Only the segmentation evidence source changes among:
  1) MedSAM
  2) MedSAM+ABLoss
  3) MedSAM+USR
  4) MedSAM+ABLoss+USR
  5) GT-mask oracle

The same deterministic SEIG rules and the same local Qwen2.5-VL-3B-Instruct model
(do_sample=False) are used for every condition. Raw report R0 and checker-filtered R*
are both saved. Checker-derived counts are explicitly treated as rule-compliance metrics,
not independent clinical-validity labels; independent human validation is a separate
reviewer-requested experiment.

GT oracle convention
--------------------
The GT condition is an oracle upper-bound control. Its binary mask is used as geometry,
with p_bar = GT and uncertainty = 0. This intentionally represents perfect segmentation
confidence and is never presented as a deployable setting.
"""
from __future__ import annotations
import os
DATA_ROOT = os.environ.get("WYSIWYR_DATA_ROOT", ".")


import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch

from seig import SEIG, SEIGConfig

VARIANTS = ("baseline", "abloss", "usr", "both", "gt")
VARIANT_LABELS = {
    "baseline": "MedSAM",
    "abloss": "MedSAM+ABLoss",
    "usr": "MedSAM+USR",
    "both": "MedSAM+ABLoss+USR",
    "gt": "GT-mask oracle",
}
DATASETS_DEFAULT = ("CVC-300", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def log(msg: str = "") -> None:
    print(msg, flush=True)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                keys.append(k); seen.add(k)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def read_mask(path: Path) -> np.ndarray:
    x = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if x is None:
        raise FileNotFoundError(path)
    return (x > 127).astype(np.uint8)


def find_by_stem(folder: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not folder.exists():
        return out
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            out[p.stem] = p
    return out


def discover_cases(root: Path, dataset: str) -> List[Tuple[str, Path, Path]]:
    base = root / "data" / "TestDataset" / dataset
    images = find_by_stem(base / "image")
    masks = find_by_stem(base / "mask")
    common = sorted(set(images) & set(masks))
    if not common:
        raise RuntimeError(f"No paired cases found for {dataset}: {base}")
    return [(s, images[s], masks[s]) for s in common]


def load_stage5_metric_index(root: Path, dataset: str) -> Dict[Tuple[str, str], Dict[str, float]]:
    path = root / "stage5_calibrated" / "reviewer2_results" / dataset / "case_segmentation_metrics.csv"
    rows = read_csv(path)
    if not rows:
        raise FileNotFoundError(f"Missing Stage5 case metrics: {path}")
    idx: Dict[Tuple[str, str], Dict[str, float]] = {}
    for r in rows:
        key = (str(r["case_id"]), str(r["variant_key"]))
        idx[key] = {k: float(r[k]) for k in ("dice", "iou", "precision", "recall", "boundary_dice", "hd95", "assd")}
    return idx


def load_evidence_inputs(root: Path, dataset: str, stem: str, variant: str, gt_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if variant == "gt":
        m = read_mask(gt_path)
        return m, m.astype(np.float32), np.zeros_like(m, dtype=np.float32)
    base = root / "stage5_calibrated" / "predictions" / dataset / variant
    mp = base / "masks" / f"{stem}.png"
    pp = base / "prob" / f"{stem}.npy"
    up = base / "unc" / f"{stem}.npy"
    missing = [str(p) for p in (mp, pp, up) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing Stage5 prediction asset(s):\n" + "\n".join(missing))
    m = read_mask(mp)
    p = np.asarray(np.load(pp), dtype=np.float32)
    u = np.asarray(np.load(up), dtype=np.float32)
    if m.shape != p.shape or m.shape != u.shape:
        raise ValueError(f"Shape mismatch {dataset}/{stem}/{variant}: mask={m.shape} prob={p.shape} unc={u.shape}")
    return m, p, u


def model_environment(model_path: Path) -> Dict[str, Any]:
    try:
        import transformers
        tr_ver = transformers.__version__
    except Exception:
        tr_ver = "unknown"
    try:
        import qwen_vl_utils
        qvu_ver = getattr(qwen_vl_utils, "__version__", "unknown")
    except Exception:
        qvu_ver = "unknown"
    files = []
    for p in sorted(model_path.glob("*")):
        if p.is_file() and p.suffix in {".json", ".safetensors"}:
            files.append({"name": p.name, "size_bytes": p.stat().st_size})
    return {
        "model_path": str(model_path),
        "model_files": files,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": tr_ver,
        "qwen_vl_utils": qvu_ver,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    }


def load_qwen(model_path: Path):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    log("Loading frozen Qwen2.5-VL model (local files only) ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    # Avoid a harmless warning inherited from some generation_config files.
    try:
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    except Exception:
        pass
    return model, processor


def generate_report(model, processor, image_path: Path, prompt: str, max_new_tokens: int) -> Tuple[str, float, int]:
    from qwen_vl_utils import process_vision_info
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": "file://" + str(image_path.resolve())},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")
    if torch.cuda.is_available(): torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    latency = time.perf_counter() - t0
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    text_out = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    n_out = int(trimmed[0].numel()) if trimmed else 0
    del inputs, generated, trimmed
    return text_out, latency, n_out


def audit_stats(audit: List[Dict[str, str]], raw_report: str) -> Dict[str, Any]:
    status_counts = defaultdict(int)
    action_counts = defaultdict(int)
    for a in audit:
        status_counts[a.get("status", "unknown")] += 1
        action_counts[a.get("action", "unknown")] += 1
    forbidden = status_counts["prohibited"]
    unsupported = status_counts["unsupported"]
    calibrated = status_counts["calibrated"]
    violations = forbidden + unsupported + calibrated
    anatomy = action_counts["generalize"]
    boundary_over = action_counts["rewrite_cautiously"]
    removed = action_counts["remove"]
    headings = [
        "visual finding", "location and approximate size", "morphology and boundary",
        "confidence and uncertainty", "evidence-supported safety note"
    ]
    section_hits = sum(bool(re.search(re.escape(h), raw_report, flags=re.I)) for h in headings)
    return {
        "raw_violation_count": int(violations),
        "raw_forbidden_count": int(forbidden),
        "raw_unsupported_count": int(unsupported),
        "raw_calibrated_count": int(calibrated),
        "raw_anatomy_unsupported_count": int(anatomy),
        "raw_boundary_overconfidence_count": int(boundary_over),
        "raw_removed_claim_count": int(removed),
        "raw_any_violation": int(violations > 0),
        "raw_section_heading_hits": int(section_hits),
        "raw_char_count": len(raw_report),
    }


def checked_residual_stats(seig: SEIG, checked_report: str, result) -> Dict[str, Any]:
    _checked2, audit2 = seig.verify_report(checked_report, result, append_safety_note=False)
    stats = audit_stats(audit2, checked_report)
    return {"checked_" + k.removeprefix("raw_"): v for k, v in stats.items() if k.startswith("raw_")}


def symbolic_agreement(symbolic: Dict[str, Any], gt_symbolic: Dict[str, Any]) -> Dict[str, Any]:
    fields = ["location", "area_level", "shape", "boundary_status"]
    vals = {f"symbolic_{k}_match": int(symbolic.get(k) == gt_symbolic.get(k)) for k in fields}
    vals["symbolic_match_fraction"] = float(np.mean(list(vals.values())))
    return vals


def flatten_record(record: Dict[str, Any]) -> Dict[str, Any]:
    e = record["seig"]["vector"]
    s = record["seig"]["symbolic"]
    row = {
        "dataset": record["dataset"], "case_id": record["case_id"], "variant_key": record["variant_key"],
        "variant": record["variant"], "is_gt_oracle": int(record["variant_key"] == "gt"),
        **record["segmentation_metrics"],
        "area_ratio": e["area_ratio"], "centroid_x": e["centroid_x"], "centroid_y": e["centroid_y"],
        "compactness": e["compactness"], "lesion_confidence": e["lesion_confidence"],
        "lesion_uncertainty": e["lesion_uncertainty"], "boundary_uncertainty": e["boundary_uncertainty"],
        "location_label": s["location"], "area_level": s["area_level"], "shape_label": s["shape"],
        "boundary_status": s["boundary_status"], "confidence_label": s["confidence"],
        "internal_uncertainty_label": s["internal_uncertainty"], "boundary_uncertainty_label": s["boundary_uncertainty"],
        **record["symbolic_agreement"], **record["raw_audit_stats"], **record["checked_residual_stats"],
        "generation_latency_s": record["generation_latency_s"], "generated_tokens": record["generated_tokens"],
    }
    return row


def average_ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        r = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = r
        i = j
    return ranks


def spearman(x: Iterable[float], y: Iterable[float]) -> float:
    a = np.asarray(list(x), dtype=float); b = np.asarray(list(y), dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    a = a[ok]; b = b[ok]
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    ra = average_ranks(a); rb = average_ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def mean_finite(vals: Iterable[Any]) -> float:
    a = np.asarray([float(x) for x in vals], dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def stratified_bootstrap_delta(rows: List[Dict[str, Any]], metric: str, target: str, ref: str = "baseline", n_boot: int = 2000, seed: int = 2023) -> Tuple[float, float, float, int]:
    by_ds_case: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        by_ds_case[r["dataset"]][r["case_id"]][r["variant_key"]] = r
    diffs_by_ds: Dict[str, np.ndarray] = {}
    for ds, cases in by_ds_case.items():
        ds_diffs = []
        for _, vv in cases.items():
            if target in vv and ref in vv:
                a = float(vv[target][metric]); b = float(vv[ref][metric])
                if np.isfinite(a) and np.isfinite(b): ds_diffs.append(a - b)
        if ds_diffs: diffs_by_ds[ds] = np.asarray(ds_diffs, dtype=float)
    pooled = np.concatenate(list(diffs_by_ds.values())) if diffs_by_ds else np.asarray([], float)
    point = float(pooled.mean()) if len(pooled) else float("nan")
    rng = np.random.default_rng(seed); boots = []
    for _ in range(n_boot):
        z = []
        for arr in diffs_by_ds.values():
            z.append(rng.choice(arr, size=len(arr), replace=True))
        if z: boots.append(float(np.concatenate(z).mean()))
    if boots:
        lo, hi = np.percentile(boots, [2.5, 97.5])
    else:
        lo = hi = float("nan")
    return point, float(lo), float(hi), int(len(pooled))


def aggregate(out_root: Path, seed: int) -> None:
    records = []
    for p in sorted((out_root / "cases").glob("*/*/*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            if obj.get("complete") is True:
                records.append(flatten_record(obj))
        except Exception as e:
            log(f"[WARN] could not parse {p}: {e}")
    if not records:
        raise RuntimeError("No completed Stage6 records found")
    write_csv(out_root / "case_level_results.csv", records)

    # Dataset x variant summary.
    summary = []
    grouped = defaultdict(list)
    for r in records: grouped[(r["dataset"], r["variant_key"])].append(r)
    summary_metrics = [
        "dice", "boundary_dice", "hd95", "assd", "symbolic_match_fraction",
        "raw_violation_count", "raw_forbidden_count", "raw_anatomy_unsupported_count",
        "raw_boundary_overconfidence_count", "raw_any_violation", "raw_section_heading_hits",
        "checked_violation_count", "generation_latency_s", "generated_tokens"
    ]
    for (ds, v), rs in sorted(grouped.items()):
        row = {"dataset": ds, "variant_key": v, "variant": VARIANT_LABELS[v], "n": len(rs)}
        for m in summary_metrics: row[f"mean_{m}"] = mean_finite(r[m] for r in rs)
        summary.append(row)
    write_csv(out_root / "summary_by_dataset_variant.csv", summary)

    # Macro summary (equal weight per dataset).
    macro = []
    for v in VARIANTS:
        dsrows = [r for r in summary if r["variant_key"] == v]
        if not dsrows: continue
        row = {"variant_key": v, "variant": VARIANT_LABELS[v], "n_datasets": len(dsrows)}
        for m in summary_metrics: row[f"macro_{m}"] = mean_finite(r[f"mean_{m}"] for r in dsrows)
        macro.append(row)
    write_csv(out_root / "summary_macro_variant.csv", macro)

    # Paired report/evidence contrasts against baseline, stratified bootstrap.
    contrasts = []
    for target in ("abloss", "usr", "both", "gt"):
        for metric in ("symbolic_match_fraction", "raw_violation_count", "raw_any_violation", "raw_boundary_overconfidence_count", "raw_anatomy_unsupported_count"):
            point, lo, hi, n = stratified_bootstrap_delta(records, metric, target, "baseline", 2000, seed)
            contrasts.append({"target": target, "target_label": VARIANT_LABELS[target], "reference": "baseline", "metric": metric,
                              "paired_mean_delta_target_minus_reference": point, "bootstrap95_low": lo, "bootstrap95_high": hi, "n_paired": n})
    write_csv(out_root / "paired_report_contrasts_vs_medsam.csv", contrasts)

    # Descriptive case-level segmentation -> evidence/report associations (model masks only).
    associations = []
    model_rows = [r for r in records if r["variant_key"] != "gt"]
    pairs = [
        ("dice", "symbolic_match_fraction"), ("boundary_dice", "symbolic_match_fraction"),
        ("dice", "raw_violation_count"), ("boundary_dice", "raw_violation_count"),
        ("hd95", "raw_violation_count"), ("assd", "raw_violation_count"),
        ("boundary_dice", "raw_boundary_overconfidence_count"), ("hd95", "raw_boundary_overconfidence_count"),
    ]
    scopes = [("pooled_model_conditions", model_rows)]
    for v in ("baseline", "abloss", "usr", "both"):
        scopes.append((v, [r for r in model_rows if r["variant_key"] == v]))
    for scope, rs in scopes:
        for x, y in pairs:
            rho = spearman((r[x] for r in rs), (r[y] for r in rs))
            associations.append({"scope": scope, "x": x, "y": y, "spearman_rho": rho, "n": len(rs),
                                 "note": "descriptive association; repeated conditions within case are not treated as independent inferential units"})

    # Within-case deltas vs MedSAM: stronger propagation analysis.
    by_case = defaultdict(dict)
    for r in records:
        if r["variant_key"] != "gt": by_case[(r["dataset"], r["case_id"])][r["variant_key"]] = r
    for target in ("abloss", "usr", "both"):
        drows = []
        for _, vv in by_case.items():
            if "baseline" in vv and target in vv:
                b, t = vv["baseline"], vv[target]
                drows.append({
                    "delta_dice": t["dice"] - b["dice"],
                    "delta_boundary_dice": t["boundary_dice"] - b["boundary_dice"],
                    "delta_hd95": t["hd95"] - b["hd95"],
                    "delta_symbolic_match": t["symbolic_match_fraction"] - b["symbolic_match_fraction"],
                    "delta_raw_violations": t["raw_violation_count"] - b["raw_violation_count"],
                    "delta_boundary_overconfidence": t["raw_boundary_overconfidence_count"] - b["raw_boundary_overconfidence_count"],
                })
        for x, y in [
            ("delta_dice", "delta_symbolic_match"), ("delta_boundary_dice", "delta_symbolic_match"),
            ("delta_dice", "delta_raw_violations"), ("delta_boundary_dice", "delta_raw_violations"),
            ("delta_hd95", "delta_raw_violations"), ("delta_boundary_dice", "delta_boundary_overconfidence"),
        ]:
            associations.append({"scope": f"within_case_{target}_minus_baseline", "x": x, "y": y,
                                 "spearman_rho": spearman((r[x] for r in drows), (r[y] for r in drows)), "n": len(drows),
                                 "note": "within-case delta association relative to MedSAM baseline"})
    write_csv(out_root / "segmentation_to_report_associations.csv", associations)

    readme = f"""# WYSIWYR Stage 6 end-to-end mask-source -> SEIG -> Qwen2.5-VL results\n\nCompleted records: {len(records)}.\n\nPrimary conditions: MedSAM, MedSAM+ABLoss, MedSAM+USR, MedSAM+ABLoss+USR, GT-mask oracle.\n\n## Interpretation guardrails\n- All conditions use the same original image, the same SEIG implementation, the same Qwen2.5-VL checkpoint, and deterministic decoding.\n- The GT condition is an oracle upper-bound control with p_bar=GT and uncertainty=0; it is not deployable.\n- `raw_*` checker metrics quantify deterministic rule compliance of R0. They are **not** independent clinical-validity labels.\n- `checked_*` metrics concern R* after the same deterministic checker and therefore should not be used as independent evidence of clinical correctness.\n- Independent human/evaluator validation of the checker is a separate reviewer-requested experiment and remains necessary.\n\n## Main files\n- `case_level_results.csv`: one row per case x mask source.\n- `summary_by_dataset_variant.csv`: dataset-level summary.\n- `summary_macro_variant.csv`: equal-dataset macro summary.\n- `paired_report_contrasts_vs_medsam.csv`: paired, dataset-stratified bootstrap contrasts.\n- `segmentation_to_report_associations.csv`: descriptive and within-case delta propagation analyses.\n- `cases/<dataset>/<case>/<variant>.json`: full SEIG evidence, prompt, raw report R0, checked report R*, and audit log.\n"""
    (out_root / "README_RESULTS.md").write_text(readme, encoding="utf-8")


def preflight(root: Path, model_path: Path, datasets: List[str]) -> Dict[str, Any]:
    errors = []
    if not model_path.exists(): errors.append(f"Missing model directory: {model_path}")
    required_model = ["config.json", "preprocessor_config.json", "model.safetensors.index.json"]
    for fn in required_model:
        if not (model_path / fn).exists(): errors.append(f"Missing model file: {model_path/fn}")
    counts = {}
    for ds in datasets:
        try:
            cases = discover_cases(root, ds); counts[ds] = len(cases)
            load_stage5_metric_index(root, ds)
            # Spot-check first case for all model variants.
            stem, _, gt = cases[0]
            for v in ("baseline", "abloss", "usr", "both"):
                load_evidence_inputs(root, ds, stem, v, gt)
        except Exception as e:
            errors.append(f"{ds}: {e}")
    if not torch.cuda.is_available(): errors.append("CUDA GPU is not available")
    return {"ok": not errors, "errors": errors, "dataset_counts": counts}


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root", default=DATA_ROOT + "/wysiwyr_real")
    p.add_argument("--model", default=DATA_ROOT + "/models/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--output", default="")
    p.add_argument("--datasets", default=",".join(DATASETS_DEFAULT))
    p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--max-cases-per-dataset", type=int, default=0, help="0 = all cases")
    p.add_argument("--smoke", action="store_true", help="Run one CVC-300 case x five mask sources")
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--no-package", action="store_true")
    return p.parse_args()


def main():
    args = parse_args(); seed_all(args.seed)
    root = Path(args.root).resolve(); model_path = Path(args.model).resolve()
    out_root = Path(args.output).resolve() if args.output else root / "stage6_seig_mllm_end2end"
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    if args.smoke:
        datasets = ["CVC-300"]; args.max_cases_per_dataset = 1
        out_root = root / "stage6_smoke"

    log("\n" + "="*80)
    log("WYSIWYR STAGE6 - MASK SOURCE -> FIXED SEIG -> FIXED QWEN2.5-VL")
    log("="*80)
    pf = preflight(root, model_path, datasets)
    log(f"Preflight dataset counts: {pf['dataset_counts']}")
    if not pf["ok"]:
        for e in pf["errors"]: log("[ERROR] " + e)
        raise SystemExit("Preflight failed; no experiment was started.")
    log("PRECHECK PASSED")
    if args.preflight_only: return

    model, processor = load_qwen(model_path)
    seig = SEIG(SEIGConfig())
    env = model_environment(model_path)
    protocol = {
        "stage": "Stage6 end-to-end mask-source -> SEIG -> Qwen2.5-VL",
        "seed": args.seed, "datasets": datasets, "variants": list(VARIANTS), "variant_labels": VARIANT_LABELS,
        "model": env, "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens, "use_cache": True},
        "seig_config": vars(seig.config),
        "gt_oracle": {"mask": "ground truth binary mask", "p_bar": "GT binary probability", "uncertainty": 0.0,
                      "interpretation": "oracle upper-bound control; not deployable"},
        "primary_comparison_rule": "same original image, same SEIG rules, same MLLM checkpoint, same deterministic decoding; only segmentation evidence source varies",
        "evaluation_guardrail": "checker-derived metrics are rule-compliance metrics; independent human validation remains separate",
    }
    out_root.mkdir(parents=True, exist_ok=True); write_json(out_root / "protocol_stage6.json", protocol)

    total_expected = 0
    dataset_cases = {}
    metric_indexes = {}
    for ds in datasets:
        cases = discover_cases(root, ds)
        if args.max_cases_per_dataset > 0: cases = cases[:args.max_cases_per_dataset]
        dataset_cases[ds] = cases; metric_indexes[ds] = load_stage5_metric_index(root, ds)
        total_expected += len(cases) * len(VARIANTS)
    log(f"Expected report generations: {total_expected}")

    completed = 0; generated_now = 0
    for ds in datasets:
        cases = dataset_cases[ds]; metrics_idx = metric_indexes[ds]
        log(f"\n===== {ds}: {len(cases)} cases x {len(VARIANTS)} mask sources =====")
        for ci, (stem, image_path, gt_path) in enumerate(cases, 1):
            # Build GT SEIG first so every model-derived symbolic tuple can be compared to it.
            gm, gp, gu = load_evidence_inputs(root, ds, stem, "gt", gt_path)
            gt_res = seig.build(gm, gp, gu)
            gt_symbolic = gt_res.as_dict()["symbolic"]
            for vi, variant in enumerate(VARIANTS, 1):
                rec_path = out_root / "cases" / ds / stem / f"{variant}.json"
                if rec_path.exists():
                    try:
                        old = json.loads(rec_path.read_text(encoding="utf-8"))
                        if old.get("complete") is True:
                            completed += 1; continue
                    except Exception:
                        pass
                m, pbar, unc = load_evidence_inputs(root, ds, stem, variant, gt_path)
                result = gt_res if variant == "gt" else seig.build(m, pbar, unc)
                raw, latency, ntok = generate_report(model, processor, image_path, result.prompt, args.max_new_tokens)
                checked, audit = seig.verify_report(raw, result)
                rstats = audit_stats(audit, raw)
                cstats = checked_residual_stats(seig, checked, result)
                if variant == "gt":
                    segm = {"dice":1.0,"iou":1.0,"precision":1.0,"recall":1.0,"boundary_dice":1.0,"hd95":0.0,"assd":0.0}
                else:
                    key = (stem, variant)
                    if key not in metrics_idx:
                        raise KeyError(f"Missing Stage5 metrics for {ds}/{stem}/{variant}")
                    segm = metrics_idx[key]
                rdict = result.as_dict()
                record = {
                    "complete": True, "dataset": ds, "case_id": stem, "variant_key": variant, "variant": VARIANT_LABELS[variant],
                    "image_path": str(image_path), "gt_mask_path": str(gt_path),
                    "segmentation_metrics": segm,
                    "seig": rdict,
                    "symbolic_agreement": symbolic_agreement(rdict["symbolic"], gt_symbolic),
                    "raw_report_R0": raw, "checked_report_Rstar": checked, "checker_audit": audit,
                    "raw_audit_stats": rstats, "checked_residual_stats": cstats,
                    "generation_latency_s": latency, "generated_tokens": ntok,
                    "decode": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
                    "gt_oracle_condition": variant == "gt",
                }
                write_json(rec_path, record)
                generated_now += 1; completed += 1
                log(f"{ds} {ci}/{len(cases)} [{vi}/5 {variant}] tokens={ntok} gen={latency:.2f}s violations={rstats['raw_violation_count']}")
                if torch.cuda.is_available() and generated_now % 20 == 0: torch.cuda.empty_cache()

    log("\nAggregating reviewer-facing tables and associations ...")
    aggregate(out_root, args.seed)
    write_json(out_root / "RUN_COMPLETE.json", {
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"), "completed_records": completed,
        "generated_this_run": generated_now, "expected_records": total_expected, "datasets": datasets,
    })

    if not args.no_package and not args.smoke:
        zip_base = Path(DATA_ROOT + "/stage6_results_for_review")
        if zip_base.with_suffix(".zip").exists(): zip_base.with_suffix(".zip").unlink()
        shutil.make_archive(str(zip_base), "zip", out_root)
        log(f"Packaged results: {zip_base.with_suffix('.zip')}")

    log("\n" + "="*80)
    log("STAGE6 COMPLETE")
    log(f"Results: {out_root}")
    if not args.no_package and not args.smoke: log("Review ZIP: <DATA_ROOT>/stage6_results_for_review.zip")
    log("="*80)


if __name__ == "__main__":
    main()
