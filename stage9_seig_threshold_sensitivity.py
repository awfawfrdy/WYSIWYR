#!/usr/bin/env python3
from __future__ import annotations
import os
DATA_ROOT = os.environ.get("WYSIWYR_DATA_ROOT", ".")


import argparse
import csv
import json
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from seig import (
    SEIG,
    SEIGConfig,
    SEIGResult,
    EvidenceVector,
    SymbolicEvidence,
    ClaimPermissionPlan,
)

DATASETS = ("CVC-300", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir")
DEFAULT_CFG = SEIGConfig()


def log(x=""):
    print(x, flush=True)


def write_json(p: Path, obj: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + ".tmp")
    q.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    q.replace(p)


def write_csv(p: Path, rows: List[Dict[str, Any]]):
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with p.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def mean(xs):
    a = np.asarray([float(x) for x in xs], dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def threshold_configs() -> Dict[str, Tuple[str, SEIGConfig, str]]:
    """Pre-specified reviewer-grade threshold grid.

    Default values are the manuscript/reconstruction defaults. Perturbations are
    fixed BEFORE test evaluation and are not selected using test performance.

    Global lenient/strict changes confidence + uncertainty only, matching the
    manuscript's original global strictness concept. One-at-a-time (OAT) runs
    separately vary area, compactness, confidence, and uncertainty.
    """
    d = DEFAULT_CFG
    return {
        "default": (
            "Default",
            d,
            "Reference configuration",
        ),
        "global_lenient": (
            "Global lenient",
            replace(
                d,
                confidence_moderate=0.55,
                confidence_high=0.75,
                uncertainty_low=0.30,
                uncertainty_high=0.55,
            ),
            "Combined confidence/uncertainty relaxation; area and compactness unchanged",
        ),
        "global_strict": (
            "Global strict",
            replace(
                d,
                confidence_moderate=0.65,
                confidence_high=0.85,
                uncertainty_low=0.20,
                uncertainty_high=0.45,
            ),
            "Combined confidence/uncertainty tightening; area and compactness unchanged",
        ),
        "area_low": (
            "Area thresholds -20%",
            replace(d, area_tiny=0.008, area_small=0.040, area_medium=0.120),
            "OAT: all area cut-points reduced by 20% relative to default",
        ),
        "area_high": (
            "Area thresholds +20%",
            replace(d, area_tiny=0.012, area_small=0.060, area_medium=0.180),
            "OAT: all area cut-points increased by 20% relative to default",
        ),
        "compactness_low": (
            "Compactness thresholds -10%",
            replace(d, compactness_regular=1.17, compactness_irregular=1.62),
            "OAT: compactness cut-points reduced by 10% relative to default",
        ),
        "compactness_high": (
            "Compactness thresholds +10%",
            replace(d, compactness_regular=1.43, compactness_irregular=1.98),
            "OAT: compactness cut-points increased by 10% relative to default",
        ),
        "confidence_low": (
            "Confidence thresholds -0.05",
            replace(d, confidence_moderate=0.55, confidence_high=0.75),
            "OAT: both confidence cut-points reduced by 0.05",
        ),
        "confidence_high": (
            "Confidence thresholds +0.05",
            replace(d, confidence_moderate=0.65, confidence_high=0.85),
            "OAT: both confidence cut-points increased by 0.05",
        ),
        "uncertainty_low": (
            "Uncertainty thresholds -0.05",
            replace(d, uncertainty_low=0.20, uncertainty_high=0.45),
            "OAT: both uncertainty cut-points reduced by 0.05",
        ),
        "uncertainty_high": (
            "Uncertainty thresholds +0.05",
            replace(d, uncertainty_low=0.30, uncertainty_high=0.55),
            "OAT: both uncertainty cut-points increased by 0.05",
        ),
    }


def cfg_row(key: str, label: str, cfg: SEIGConfig, rationale: str) -> Dict[str, Any]:
    return {
        "config_key": key,
        "config_label": label,
        "area_tiny": cfg.area_tiny,
        "area_small": cfg.area_small,
        "area_medium": cfg.area_medium,
        "compactness_regular": cfg.compactness_regular,
        "compactness_irregular": cfg.compactness_irregular,
        "confidence_moderate": cfg.confidence_moderate,
        "confidence_high": cfg.confidence_high,
        "uncertainty_low": cfg.uncertainty_low,
        "uncertainty_high": cfg.uncertainty_high,
        "rationale": rationale,
    }


def result_from_vector(vec: EvidenceVector, cfg: SEIGConfig) -> SEIGResult:
    s = SEIG(cfg)
    sym = s.symbolize(vec)
    graph = s.build_graph(sym)
    plan = s.claim_permissions(sym)
    prompt = s.render_prompt(vec, sym, graph, plan)
    return SEIGResult(vec, sym, graph, plan, prompt)


def vector_from_stage6(o: Dict[str, Any]) -> EvidenceVector:
    return EvidenceVector(**o["seig"]["vector"])


def discover_stage6(stage6: Path) -> Dict[str, List[Tuple[str, Path, Dict[str, Any]]]]:
    out = {}
    for ds in DATASETS:
        rows = []
        for p in sorted((stage6 / "cases" / ds).glob("*/both.json")):
            try:
                o = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            if o.get("complete") is True:
                rows.append((str(o["case_id"]), p, o))
        if not rows:
            raise RuntimeError(f"No Stage6 both.json records found for {ds}: {stage6}")
        out[ds] = rows
    return out


def plan_signature(plan: ClaimPermissionPlan) -> Dict[str, Tuple[str, ...]]:
    return {
        "allowed": tuple(plan.allowed),
        "cautious": tuple(plan.cautious),
        "prohibited": tuple(plan.prohibited),
    }


def symbolic_signature(sym: SymbolicEvidence) -> Dict[str, str]:
    return asdict(sym)


def run_symbolic_full(
    dsrows: Dict[str, List[Tuple[str, Path, Dict[str, Any]]]],
    configs: Dict[str, Tuple[str, SEIGConfig, str]],
    out: Path,
):
    log("\n===== PHASE A: FULL 798-CASE SYMBOLIC / PERMISSION SENSITIVITY =====")
    rows = []
    dist = Counter()
    for ds, cases in dsrows.items():
        for i, (cid, _, o) in enumerate(cases, 1):
            vec = vector_from_stage6(o)
            dres = result_from_vector(vec, configs["default"][1])
            dsym = symbolic_signature(dres.symbolic)
            dplan = plan_signature(dres.claim_plan)
            for key, (label, cfg, rationale) in configs.items():
                res = result_from_vector(vec, cfg)
                sym = symbolic_signature(res.symbolic)
                plan = plan_signature(res.claim_plan)
                changed_fields = [f for f in dsym if sym[f] != dsym[f]]
                row = {
                    "dataset": ds,
                    "case_id": cid,
                    "config_key": key,
                    "config_label": label,
                    "tuple_changed_any": int(bool(changed_fields)),
                    "n_symbolic_fields_changed": len(changed_fields),
                    "changed_fields": "|".join(changed_fields),
                    "permission_changed_any": int(plan != dplan),
                    "allowed_changed": int(plan["allowed"] != dplan["allowed"]),
                    "cautious_changed": int(plan["cautious"] != dplan["cautious"]),
                    "prohibited_changed": int(plan["prohibited"] != dplan["prohibited"]),
                    "allowed_count": len(plan["allowed"]),
                    "cautious_count": len(plan["cautious"]),
                    "prohibited_count": len(plan["prohibited"]),
                    "area_level": sym["area_level"],
                    "shape": sym["shape"],
                    "confidence": sym["confidence"],
                    "internal_uncertainty": sym["internal_uncertainty"],
                    "boundary_uncertainty": sym["boundary_uncertainty"],
                    "quality": sym["quality"],
                    "area_ratio": vec.area_ratio,
                    "compactness": vec.compactness,
                    "lesion_confidence": vec.lesion_confidence,
                    "lesion_uncertainty": vec.lesion_uncertainty,
                    "boundary_uncertainty_numeric": vec.boundary_uncertainty,
                }
                for f in ("area_level", "shape", "confidence", "internal_uncertainty", "boundary_uncertainty", "quality"):
                    row[f"changed_{f}"] = int(sym[f] != dsym[f])
                    dist[(key, f, sym[f])] += 1
                rows.append(row)
            if i == 1 or i % 100 == 0 or i == len(cases):
                log(f"{ds}: {i}/{len(cases)}")

    write_csv(out / "symbolic_full_case_results.csv", rows)

    summaries = []
    bycfg = defaultdict(list)
    for r in rows:
        bycfg[r["config_key"]].append(r)
    for key, rr in bycfg.items():
        label = configs[key][0]
        z = {"config_key": key, "config_label": label, "n_cases": len(rr)}
        for m in (
            "tuple_changed_any", "n_symbolic_fields_changed", "permission_changed_any",
            "allowed_changed", "cautious_changed", "prohibited_changed",
            "changed_area_level", "changed_shape", "changed_confidence",
            "changed_internal_uncertainty", "changed_boundary_uncertainty", "changed_quality",
            "allowed_count", "cautious_count", "prohibited_count",
        ):
            z["mean_" + m] = mean(r[m] for r in rr)
        summaries.append(z)
    write_csv(out / "symbolic_full_summary.csv", summaries)

    dist_rows = []
    for (key, field, label), n in sorted(dist.items()):
        dist_rows.append({
            "config_key": key,
            "config_label": configs[key][0],
            "field": field,
            "symbolic_label": label,
            "count": n,
            "fraction": n / sum(1 for r in rows if r["config_key"] == key),
        })
    write_csv(out / "symbolic_label_distribution.csv", dist_rows)
    return rows, summaries


def make_subset(dsrows, cases_per_dataset: int, seed: int, smoke: bool = False):
    subset = {}
    for di, (ds, rows) in enumerate(dsrows.items()):
        if smoke:
            subset[ds] = rows[:1] if ds == "CVC-300" else []
            continue
        n = min(cases_per_dataset, len(rows))
        rng = random.Random(seed + 7919 * (di + 1))
        chosen = rng.sample(rows, n)
        chosen.sort(key=lambda x: x[0])
        subset[ds] = chosen
    return {k: v for k, v in subset.items() if v}


def load_qwen(model_path: Path):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    log("Loading frozen Qwen2.5-VL model (local files only) ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path), dtype=torch.bfloat16, device_map="auto", local_files_only=True
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    try:
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    except Exception:
        pass
    return model, processor


def generate(model, processor, prompt: str, image_path: Path, max_new_tokens: int):
    from qwen_vl_utils import process_vision_info
    messages = [{"role": "user", "content": [
        {"type": "image", "image": "file://" + str(image_path.resolve())},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to("cuda")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    trim = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    txt = processor.batch_decode(trim, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    ntok = int(trim[0].numel())
    del inputs, out, trim
    return txt, dt, ntok


def audit_stats(audit, raw_report):
    status, action = defaultdict(int), defaultdict(int)
    for a in audit:
        status[a.get("status", "unknown")] += 1
        action[a.get("action", "unknown")] += 1
    forbidden = status["prohibited"]
    unsupported = status["unsupported"]
    calibrated = status["calibrated"]
    v = forbidden + unsupported + calibrated
    return {
        "violation_count": int(v),
        "forbidden_count": int(forbidden),
        "unsupported_count": int(unsupported),
        "boundary_overconfidence_count": int(action["rewrite_cautiously"]),
        "anatomy_unsupported_count": int(action["generalize"]),
        "any_violation": int(v > 0),
        "char_count": len(raw_report),
    }


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().replace("–", "-").replace("—", "-")).strip()


def phrase_present(text: str, phrase: str) -> bool:
    t, p = norm(text), norm(phrase)
    p_re = re.escape(p).replace(r"\-", r"[- ]")
    return bool(re.search(r"(?<!\w)" + p_re + r"(?!\w)", t, re.I))


def safe_section(report: str, name: str) -> str:
    names = [
        "visual finding", "location and approximate size", "morphology and boundary",
        "confidence and uncertainty", "evidence-supported safety note",
    ]
    low = report.lower()
    pos = low.find(name)
    if pos < 0:
        return report
    start = pos + len(name)
    ends = [low.find(n, start) for n in names if low.find(n, start) >= 0]
    return report[start:(min(ends) if ends else len(report))]


def loc_area_alignment(report: str, sym: SymbolicEvidence):
    sec = safe_section(report, "location and approximate size")
    return {
        "location_match": int(phrase_present(sec, sym.location)),
        "area_match": int(phrase_present(sec, sym.area_level)),
    }


def paired_wilcoxon(a, b):
    x = np.asarray(a, float)
    y = np.asarray(b, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) == 0:
        return float("nan")
    if np.allclose(x, y):
        return 1.0
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(x, y, alternative="two-sided", zero_method="wilcox").pvalue)
    except Exception:
        return float("nan")


def bootstrap_delta(a, b, n_boot=3000, seed=2023):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    mask = np.isfinite(a) & np.isfinite(b)
    d = a[mask] - b[mask]
    if not len(d):
        return float("nan"), float("nan"), float("nan"), 0
    point = float(d.mean())
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        vals.append(float(rng.choice(d, size=len(d), replace=True).mean()))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi), int(len(d))


def run_reports(
    subset,
    configs,
    model_path: Path,
    out: Path,
    max_new_tokens: int,
    seed: int,
):
    log("\n===== PHASE B: 200-CASE FINAL-REPORT THRESHOLD SENSITIVITY =====")
    manifest = []
    for ds, cases in subset.items():
        for cid, p, o in cases:
            manifest.append({"dataset": ds, "case_id": cid, "stage6_case_json": str(p), "image_path": o["image_path"]})
    write_csv(out / "report_subset_manifest.csv", manifest)

    model, processor = load_qwen(model_path)
    generated_now, reused = 0, 0
    for ds, cases in subset.items():
        log(f"\n--- {ds}: {len(cases)} cases x {len(configs)} configs ---")
        for ci, (cid, _, o) in enumerate(cases, 1):
            vec = vector_from_stage6(o)
            image = Path(o["image_path"])
            if not image.exists():
                raise FileNotFoundError(image)
            for ki, (key, (label, cfg, rationale)) in enumerate(configs.items(), 1):
                recp = out / "cases" / ds / cid / f"{key}.json"
                if recp.exists():
                    try:
                        old = json.loads(recp.read_text(encoding="utf-8"))
                        if old.get("complete") is True:
                            continue
                    except Exception:
                        pass
                res = result_from_vector(vec, cfg)
                if key == "default":
                    dec = o.get("decode", {})
                    can_reuse = (
                        dec.get("do_sample") is False
                        and int(dec.get("max_new_tokens", -1)) == max_new_tokens
                        and bool(o.get("raw_report_R0"))
                    )
                    if can_reuse:
                        raw = o["raw_report_R0"]
                        latency = float(o.get("generation_latency_s", np.nan))
                        ntok = int(o.get("generated_tokens", 0))
                        source = "reused_exact_stage6_default"
                        reused += 1
                    else:
                        raw, latency, ntok = generate(model, processor, res.prompt, image, max_new_tokens)
                        source = "generated_stage9_default"
                        generated_now += 1
                else:
                    raw, latency, ntok = generate(model, processor, res.prompt, image, max_new_tokens)
                    source = "generated_stage9"
                    generated_now += 1
                checker = SEIG(cfg)
                checked, audit = checker.verify_report(raw, res)
                stats = audit_stats(audit, raw)
                align = loc_area_alignment(raw, res.symbolic)
                rec = {
                    "complete": True,
                    "dataset": ds,
                    "case_id": cid,
                    "config_key": key,
                    "config_label": label,
                    "thresholds": cfg_row(key, label, cfg, rationale),
                    "symbolic": asdict(res.symbolic),
                    "claim_plan": asdict(res.claim_plan),
                    "prompt": res.prompt,
                    "raw_report_R0": raw,
                    "checked_report_Rstar": checked,
                    "checker_audit": audit,
                    "stats": stats,
                    "alignment": align,
                    "generation_latency_s": latency,
                    "generated_tokens": ntok,
                    "generation_source": source,
                    "decode": {"do_sample": False, "max_new_tokens": max_new_tokens},
                }
                write_json(recp, rec)
                log(
                    f"{ds} {ci}/{len(cases)} [{ki}/{len(configs)} {key}] "
                    f"V={stats['violation_count']} loc={align['location_match']} area={align['area_match']} "
                    f"tok={ntok} gen={latency:.2f}s {source}"
                )
                if torch.cuda.is_available() and generated_now % 20 == 0:
                    torch.cuda.empty_cache()

    rows = []
    for p in sorted((out / "cases").glob("*/*/*.json")):
        o = json.loads(p.read_text(encoding="utf-8"))
        if not o.get("complete"):
            continue
        r = {
            "dataset": o["dataset"],
            "case_id": o["case_id"],
            "config_key": o["config_key"],
            "config_label": o["config_label"],
            **o["stats"],
            **o["alignment"],
            "generation_latency_s": o["generation_latency_s"],
            "generated_tokens": o["generated_tokens"],
            "generation_source": o["generation_source"],
        }
        rows.append(r)
    write_csv(out / "report_case_results.csv", rows)

    metrics = [
        "violation_count", "forbidden_count", "unsupported_count",
        "boundary_overconfidence_count", "anatomy_unsupported_count", "any_violation",
        "location_match", "area_match", "generation_latency_s", "generated_tokens",
    ]
    summary = []
    grp = defaultdict(list)
    for r in rows:
        grp[(r["dataset"], r["config_key"])].append(r)
    for (ds, key), rr in sorted(grp.items()):
        z = {"dataset": ds, "config_key": key, "config_label": configs[key][0], "n": len(rr)}
        for m in metrics:
            z["mean_" + m] = mean(r[m] for r in rr)
        summary.append(z)
    write_csv(out / "report_summary_by_dataset_config.csv", summary)

    macro = []
    for key in configs:
        ss = [z for z in summary if z["config_key"] == key]
        z = {"config_key": key, "config_label": configs[key][0], "n_datasets": len(ss)}
        for m in metrics:
            z["macro_" + m] = mean(x["mean_" + m] for x in ss)
        macro.append(z)
    write_csv(out / "report_macro_summary.csv", macro)

    bycase = defaultdict(dict)
    for r in rows:
        bycase[(r["dataset"], r["case_id"])][r["config_key"]] = r
    contrasts = []
    for key in configs:
        if key == "default":
            continue
        for m in (
            "violation_count", "forbidden_count", "unsupported_count",
            "boundary_overconfidence_count", "any_violation", "location_match", "area_match",
        ):
            aa, bb = [], []
            for vv in bycase.values():
                if key in vv and "default" in vv:
                    aa.append(float(vv[key][m]))
                    bb.append(float(vv["default"][m]))
            pt, lo, hi, n = bootstrap_delta(aa, bb, n_boot=3000, seed=seed + len(contrasts))
            pval = paired_wilcoxon(aa, bb)
            contrasts.append({
                "config_key": key,
                "config_label": configs[key][0],
                "reference": "default",
                "metric": m,
                "paired_mean_delta": pt,
                "bootstrap95_low": lo,
                "bootstrap95_high": hi,
                "wilcoxon_p": pval,
                "n_paired": n,
            })
    write_csv(out / "paired_report_contrasts_vs_default.csv", contrasts)
    return rows, summary, macro, contrasts, generated_now, reused


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root", default=DATA_ROOT + "/wysiwyr_real")
    p.add_argument("--stage6", default="")
    p.add_argument("--model", default=DATA_ROOT + "/models/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--output", default="")
    p.add_argument("--cases-per-dataset", type=int, default=40)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--symbolic-only", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--no-package", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(args.root).resolve()
    stage6 = Path(args.stage6).resolve() if args.stage6 else root / "stage6_seig_mllm_end2end"
    model_path = Path(args.model).resolve()
    out = Path(args.output).resolve() if args.output else root / "stage9_seig_threshold_sensitivity"
    if args.smoke:
        out = root / "stage9_smoke"

    configs = threshold_configs()
    log("\n" + "=" * 86)
    log("WYSIWYR STAGE9 - REVIEWER-GRADE SEIG SYSTEMATIC THRESHOLD SENSITIVITY")
    log("=" * 86)
    log(f"Stage6 source: {stage6}")
    log(f"Qwen model:    {model_path}")
    log(f"Configs:       {len(configs)} (default + global lenient/strict + 8 OAT variants)")
    log(f"Report subset: {args.cases_per_dataset} per dataset = {args.cases_per_dataset * 5} cases")

    if not stage6.exists():
        raise SystemExit(f"Missing Stage6 source: {stage6}")
    dsrows = discover_stage6(stage6)
    counts = {d: len(v) for d, v in dsrows.items()}
    log(f"Stage6 case counts: {counts}; total={sum(counts.values())}")
    if sum(counts.values()) != 798:
        log("WARNING: total Stage6 count differs from 798; proceeding with discovered cases and recording counts.")

    grid_rows = [cfg_row(k, label, cfg, rationale) for k, (label, cfg, rationale) in configs.items()]
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "threshold_grid.csv", grid_rows)
    protocol = {
        "stage": "Stage9 SEIG systematic threshold sensitivity",
        "seed": args.seed,
        "stage6_source": str(stage6),
        "datasets": counts,
        "default_thresholds": cfg_row("default", configs["default"][0], configs["default"][1], configs["default"][2]),
        "grid": grid_rows,
        "design": {
            "global_settings": "Lenient/default/strict vary confidence and uncertainty jointly; area and compactness remain at default in global settings, matching the manuscript's original strictness concept.",
            "OAT_settings": "Area, compactness, confidence, and uncertainty are each varied separately while every other threshold is held fixed.",
            "pre_specified_perturbations": {
                "area": "+/-20% relative at all three cut-points",
                "compactness": "+/-10% relative at both cut-points",
                "confidence": "+/-0.05 absolute at both cut-points",
                "uncertainty": "+/-0.05 absolute at both cut-points",
            },
            "test_tuning": "None. Grid is fixed before reading test performance.",
            "symbolic_analysis": "All discovered Stage6 cases (expected 798).",
            "report_analysis": f"Fixed stratified subset of {args.cases_per_dataset} cases per dataset; same cases across all threshold configurations.",
            "default_report_reuse": "Exact Stage6 Image+SEIG report reused only when do_sample=False and max_new_tokens match; all non-default prompts are regenerated.",
            "checker": "Frozen Checker v3 applied under each threshold-specific SEIG result; this measures deterministic rule compliance, not independent clinical validity.",
        },
        "decoding": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "checker_version": SEIG.checker_rulebook().get("version"),
    }
    write_json(out / "protocol_stage9.json", protocol)

    log("\n===== THRESHOLD GRID =====")
    for r in grid_rows:
        log(
            f"{r['config_key']:<18} area=({r['area_tiny']:.3f},{r['area_small']:.3f},{r['area_medium']:.3f}) "
            f"comp=({r['compactness_regular']:.2f},{r['compactness_irregular']:.2f}) "
            f"conf=({r['confidence_moderate']:.2f},{r['confidence_high']:.2f}) "
            f"unc=({r['uncertainty_low']:.2f},{r['uncertainty_high']:.2f})"
        )

    if args.preflight_only:
        if not model_path.exists():
            log("NOTE: Qwen model missing, but preflight-only symbolic source check passed.")
        log("\nPRECHECK PASSED")
        return

    run_symbolic_full(dsrows, configs, out)
    if args.symbolic_only:
        log("SYMBOLIC-ONLY COMPLETE")
        return

    if not model_path.exists():
        raise SystemExit(f"Missing Qwen model: {model_path}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is not available")

    subset = make_subset(dsrows, args.cases_per_dataset, args.seed, smoke=args.smoke)
    expected_cases = sum(len(v) for v in subset.values())
    expected_records = expected_cases * len(configs)
    rows, summary, macro, contrasts, generated_now, reused = run_reports(
        subset, configs, model_path, out, args.max_new_tokens, args.seed
    )

    run_complete = {
        "symbolic_full_cases": sum(counts.values()),
        "n_threshold_configs": len(configs),
        "report_subset_cases": expected_cases,
        "expected_report_records": expected_records,
        "completed_report_records": len(rows),
        "generated_this_run": generated_now,
        "reused_stage6_default": reused,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(out / "RUN_COMPLETE.json", run_complete)
    readme = (
        f"# WYSIWYR Stage9 SEIG systematic threshold sensitivity\n\n"
        f"Full symbolic/permission sensitivity: {sum(counts.values())} cases x {len(configs)} configurations.\n\n"
        f"Final-report sensitivity: {expected_cases} fixed stratified cases x {len(configs)} configurations = {len(rows)} records.\n\n"
        "Design: global lenient/default/strict confidence+uncertainty settings plus one-at-a-time area, "
        "compactness, confidence, and uncertainty perturbations. Threshold grid was fixed before test evaluation.\n\n"
        "Default reports are reused from Stage6 only under identical deterministic decoding. Non-default threshold "
        "prompts are regenerated with frozen Qwen2.5-VL. Frozen Checker v3 is used for threshold-specific "
        "rule-compliance analysis.\n"
    )
    (out / "README_RESULTS.md").write_text(readme, encoding="utf-8")

    if not args.no_package and not args.smoke:
        zipbase = Path(DATA_ROOT + "/stage9_threshold_sensitivity_results_for_review")
        if zipbase.with_suffix(".zip").exists():
            zipbase.with_suffix(".zip").unlink()
        shutil.make_archive(str(zipbase), "zip", out)
        log(f"RESULT ZIP: {zipbase.with_suffix('.zip')}")

    log("\n" + "=" * 86)
    log("STAGE9 COMPLETE")
    log(f"Full symbolic cases: {sum(counts.values())} x {len(configs)} configs")
    log(f"Report records: {len(rows)} / {expected_records}")
    log(f"Generated now: {generated_now}; reused exact Stage6 default: {reused}")
    log("=" * 86)


if __name__ == "__main__":
    main()
