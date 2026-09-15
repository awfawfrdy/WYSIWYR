#!/usr/bin/env python3
"""
WYSIWYR Reviewer #2 experiment/evaluation runner for AutoDL.

Designed to be used with `wysiwyr_autodl_allinone.py` in the same folder.
No nano/vim/editing is required. Point it at ground-truth and prediction folders.

What this script produces in ONE run
------------------------------------
1) Full 2x2 factorial segmentation table:
   MedSAM / MedSAM+ABLoss / MedSAM+USR / MedSAM+ABLoss+USR
2) Case-level Dice, IoU, Precision, Recall, Boundary Dice, HD95, ASSD
3) ABLoss and USR main effects + interaction effects
4) Paired Wilcoxon comparisons + bootstrap 95% CI of paired differences
5) Reviewer-requested USR semantic pixel accounting:
   - correctly recovered lesion pixels
   - incorrectly added background pixels
   - correctly removed false-positive pixels
   - incorrectly removed lesion pixels
   - probability USR worsens Dice / Boundary Dice
6) Optional uncertainty validation if probability + uncertainty maps are supplied:
   - pixel error-detection AUROC
   - Brier score
   - ECE
   - AURC / mean risk-coverage curve
7) Optional case-level association between segmentation quality and report errors
   if `--report-metrics-csv` is supplied.
8) Machine-readable CSVs + a paper-ready Markdown summary.

Folder contract
---------------
Each folder may contain PNG/JPG/TIF/BMP masks or .npy/.npz arrays. Files are paired
by filename stem. Example:

  /root/autodl-tmp/exp/gt/001.png
  /root/autodl-tmp/exp/medsam/001.png
  /root/autodl-tmp/exp/abloss/001.png
  /root/autodl-tmp/exp/usr/001.png
  /root/autodl-tmp/exp/both/001.png

Probability/uncertainty maps are optional and use the same case stems.

Important scientific note
-------------------------
This script does NOT invent experimental numbers. It only analyzes actual predictions.
For HD95/ASSD, empty-vs-nonempty cases are retained as inf in per-case output and are
excluded from mean/SD summaries, with the number of excluded non-finite cases reported.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# AutoDL images sometimes carry malformed thread env vars. Sanitize before numpy/scipy.
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _v = os.environ.get(_k)
    if _v is not None:
        try:
            if int(_v) <= 0:
                raise ValueError
        except Exception:
            os.environ[_k] = "1"

import numpy as np

try:
    import cv2
except Exception as e:
    raise SystemExit("opencv-python is required. Install with: pip install opencv-python-headless") from e

try:
    from scipy.stats import wilcoxon, spearmanr
except Exception as e:
    raise SystemExit("scipy is required. Install with: pip install scipy") from e

try:
    import wysiwyr_autodl_allinone as core
except Exception as e:
    raise SystemExit(
        "Cannot import wysiwyr_autodl_allinone.py. Put this script and "
        "wysiwyr_autodl_allinone.py in the SAME directory, then run again.\n"
        f"Original error: {e}"
    ) from e


VARIANTS = ["baseline", "abloss", "usr", "both"]
VARIANT_LABELS = {
    "baseline": "MedSAM",
    "abloss": "MedSAM+ABLoss",
    "usr": "MedSAM+USR",
    "both": "MedSAM+ABLoss+USR",
}
METRICS = ["dice", "iou", "precision", "recall", "boundary_dice", "hd95", "assd"]
HIGHER_IS_BETTER = {"dice", "iou", "precision", "recall", "boundary_dice"}
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy", ".npz"}


def _case_stem(p: Path) -> str:
    return p.stem


def _find_files(folder: Path, recursive: bool) -> Dict[str, Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    it = folder.rglob("*") if recursive else folder.glob("*")
    out: Dict[str, Path] = {}
    duplicates: Dict[str, List[str]] = {}
    for p in it:
        if not p.is_file() or p.suffix.lower() not in SUPPORTED_EXTS:
            continue
        stem = _case_stem(p)
        if stem in out:
            duplicates.setdefault(stem, [str(out[stem])]).append(str(p))
        else:
            out[stem] = p
    if duplicates:
        example = next(iter(duplicates.items()))
        raise RuntimeError(
            f"Duplicate case stem '{example[0]}' in {folder}. Rename files so every case stem is unique. "
            f"Examples: {example[1][:3]}"
        )
    return out


def _load_array(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext == ".npy":
        a = np.load(path)
    elif ext == ".npz":
        z = np.load(path)
        keys = list(z.keys())
        if not keys:
            raise ValueError(f"Empty npz: {path}")
        # Prefer common names; otherwise use first array.
        key = next((k for k in ("prob", "probability", "mask", "pred", "uncertainty", "arr_0") if k in z), keys[0])
        a = z[key]
    else:
        a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if a is None:
            raise ValueError(f"Failed to read image: {path}")
        if a.ndim == 3:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    a = np.asarray(a)
    a = np.squeeze(a)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D map, got {a.shape} from {path}")
    return a


def _to_unit_float(a: np.ndarray) -> np.ndarray:
    x = np.asarray(a, dtype=np.float32)
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    mn, mx = float(x.min()), float(x.max())
    if mx <= 1.0 and mn >= 0.0:
        return x
    # Common 8/16-bit probability or mask storage.
    if mn >= 0.0 and mx <= 255.0:
        return x / 255.0
    if mn >= 0.0 and mx <= 65535.0:
        return x / 65535.0
    # For arbitrary logits/scores, use sigmoid only when values cross outside [0,1].
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _resize_to(a: np.ndarray, shape_hw: Tuple[int, int], continuous: bool) -> np.ndarray:
    h, w = shape_hw
    if a.shape == (h, w):
        return a
    interp = cv2.INTER_LINEAR if continuous else cv2.INTER_NEAREST
    return cv2.resize(a, (w, h), interpolation=interp)


def _load_mask(path: Path, shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    a = _to_unit_float(_load_array(path))
    if shape is not None:
        a = _resize_to(a, shape, continuous=False)
    return (a > 0.5).astype(np.uint8)


def _load_prob(path: Path, shape: Tuple[int, int]) -> np.ndarray:
    a = _to_unit_float(_load_array(path))
    a = _resize_to(a, shape, continuous=True)
    return np.clip(a.astype(np.float32), 0.0, 1.0)


def _json_safe(v: Any) -> Any:
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None if math.isnan(v) else ("inf" if v > 0 else "-inf")
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    return v


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        keys: List[str] = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    keys.append(k); seen.add(k)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            rr = {}
            for k in fieldnames:
                v = r.get(k, "")
                if isinstance(v, (dict, list, tuple)):
                    v = json.dumps(_json_safe(v), ensure_ascii=False)
                rr[k] = v
            w.writerow(rr)


def _finite_stats(values: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(values, dtype=float)
    finite = x[np.isfinite(x)]
    if len(finite) == 0:
        return {"mean": float("nan"), "sd": float("nan"), "median": float("nan"), "n_finite": 0, "n_nonfinite": int(len(x))}
    return {
        "mean": float(np.mean(finite)),
        "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
        "median": float(np.median(finite)),
        "n_finite": int(len(finite)),
        "n_nonfinite": int(len(x) - len(finite)),
    }


def _bootstrap_mean_diff(a: Sequence[float], b: Sequence[float], n_boot: int, seed: int) -> Tuple[float, float, float, int]:
    """Paired diff = b-a. Non-finite pairs are dropped."""
    x = np.asarray(a, float); y = np.asarray(b, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    d = y - x
    est = float(np.mean(d))
    if n == 1:
        return est, est, est, 1
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = np.mean(d[idx])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return est, float(lo), float(hi), n


def _paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> Tuple[float, int]:
    x = np.asarray(a, float); y = np.asarray(b, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) == 0:
        return float("nan"), 0
    d = y - x
    if np.allclose(d, 0):
        return 1.0, int(len(x))
    try:
        res = wilcoxon(y, x, alternative="two-sided", zero_method="wilcox", method="auto")
        return float(res.pvalue), int(len(x))
    except Exception:
        return float("nan"), int(len(x))


def _factorial_effects(cells: Dict[str, Sequence[float]], metric: str) -> Dict[str, Any]:
    b = np.asarray(cells["baseline"], float)
    a = np.asarray(cells["abloss"], float)
    u = np.asarray(cells["usr"], float)
    ab = np.asarray(cells["both"], float)
    # Use the same complete finite cases in all four cells.
    ok = np.isfinite(b) & np.isfinite(a) & np.isfinite(u) & np.isfinite(ab)
    b, a, u, ab = b[ok], a[ok], u[ok], ab[ok]
    if len(b) == 0:
        return {"metric": metric, "n_complete": 0}
    m = lambda z: float(np.mean(z))
    return {
        "metric": metric,
        "direction": "higher_better" if metric in HIGHER_IS_BETTER else "lower_better",
        "n_complete": int(len(b)),
        "mean_baseline": m(b),
        "mean_abloss": m(a),
        "mean_usr": m(u),
        "mean_both": m(ab),
        "main_effect_abloss_raw": 0.5 * ((m(a)-m(b)) + (m(ab)-m(u))),
        "main_effect_usr_raw": 0.5 * ((m(u)-m(b)) + (m(ab)-m(a))),
        "interaction_raw": (m(ab)-m(a)) - (m(u)-m(b)),
    }


def _read_report_metrics(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _detect_case_col(fields: Sequence[str]) -> Optional[str]:
    low = {f.lower(): f for f in fields}
    for k in ("case_id", "case", "id", "image_id", "image", "filename", "file"):
        if k in low:
            return low[k]
    return None


def _numeric_columns(rows: Sequence[Dict[str, str]], exclude: Iterable[str]) -> List[str]:
    if not rows:
        return []
    exc = set(exclude)
    cols = []
    for k in rows[0].keys():
        if k in exc:
            continue
        vals = []
        for r in rows:
            s = str(r.get(k, "")).strip()
            if s == "":
                continue
            try:
                vals.append(float(s))
            except Exception:
                vals = []
                break
        if vals:
            cols.append(k)
    return cols


def _report_associations(case_rows: Sequence[Dict[str, Any]], report_csv: Path) -> List[Dict[str, Any]]:
    rr = _read_report_metrics(report_csv)
    if not rr:
        return []
    case_col = _detect_case_col(list(rr[0].keys()))
    if case_col is None:
        raise ValueError("Report metrics CSV needs one case identifier column: case_id/case/id/image_id/image/filename/file")
    rep_by_case: Dict[str, Dict[str, str]] = {}
    for r in rr:
        cid = Path(str(r[case_col])).stem
        rep_by_case[cid] = r
    report_cols = _numeric_columns(rr, exclude=[case_col])

    out: List[Dict[str, Any]] = []
    for variant in VARIANTS:
        cr = [r for r in case_rows if r["variant"] == variant and r["case_id"] in rep_by_case]
        for sm in METRICS:
            for rm in report_cols:
                x, y = [], []
                for r in cr:
                    try:
                        xv = float(r[sm]); yv = float(rep_by_case[r["case_id"]][rm])
                    except Exception:
                        continue
                    if np.isfinite(xv) and np.isfinite(yv):
                        x.append(xv); y.append(yv)
                if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
                    rho = p = float("nan")
                else:
                    s = spearmanr(x, y, nan_policy="omit")
                    rho, p = float(s.statistic), float(s.pvalue)
                out.append({
                    "variant": VARIANT_LABELS[variant], "segmentation_metric": sm,
                    "report_metric": rm, "n": len(x), "spearman_rho": rho, "p_value": p,
                })
    return out


def _mean_risk_coverage(curves: Sequence[Dict[str, Any]]) -> List[Dict[str, float]]:
    if not curves:
        return []
    cov = np.asarray(curves[0]["coverage"], float)
    risks = []
    for c in curves:
        c_cov = np.asarray(c["coverage"], float)
        c_risk = np.asarray(c["risk"], float)
        if len(c_cov) == len(cov) and np.allclose(c_cov, cov):
            risks.append(c_risk)
    if not risks:
        return []
    R = np.stack(risks, axis=0)
    return [{"coverage": float(c), "mean_risk": float(np.nanmean(R[:, i])), "sd_risk": float(np.nanstd(R[:, i], ddof=1)) if R.shape[0] > 1 else 0.0}
            for i, c in enumerate(cov)]


def _make_markdown(outdir: Path, n_cases: int, summary_rows: Sequence[Dict[str, Any]],
                   factorial_rows: Sequence[Dict[str, Any]], paired_rows: Sequence[Dict[str, Any]],
                   worsen_rows: Sequence[Dict[str, Any]], uncertainty_summary: Sequence[Dict[str, Any]],
                   assoc_rows: Sequence[Dict[str, Any]]) -> None:
    def fmt(x: Any, nd: int = 4) -> str:
        try:
            v = float(x)
            if not np.isfinite(v): return "NA"
            return f"{v:.{nd}f}"
        except Exception:
            return str(x)

    lines = [
        "# WYSIWYR Reviewer #2 experiment summary",
        "",
        f"Common paired cases: **{n_cases}**",
        "",
        "## 1. Full 2x2 segmentation results",
        "",
        "| Variant | Metric | Mean | SD | Median | Finite n | Non-finite n |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(f"| {r['variant']} | {r['metric']} | {fmt(r['mean'])} | {fmt(r['sd'])} | {fmt(r['median'])} | {r['n_finite']} | {r['n_nonfinite']} |")

    lines += ["", "## 2. 2x2 main and interaction effects", "",
              "Raw effects are differences on the original metric scale. For HD95/ASSD, negative is improvement because lower is better.", "",
              "| Metric | Direction | n | ABLoss main | USR main | Interaction |",
              "|---|---|---:|---:|---:|---:|"]
    for r in factorial_rows:
        lines.append(f"| {r['metric']} | {r.get('direction','')} | {r.get('n_complete',0)} | {fmt(r.get('main_effect_abloss_raw'))} | {fmt(r.get('main_effect_usr_raw'))} | {fmt(r.get('interaction_raw'))} |")

    lines += ["", "## 3. Paired comparisons", "",
              "| Metric | Comparison | Mean paired diff (second-first) | 95% bootstrap CI | Wilcoxon p | n |",
              "|---|---|---:|---:|---:|---:|"]
    for r in paired_rows:
        ci = f"[{fmt(r['ci95_low'])}, {fmt(r['ci95_high'])}]"
        lines.append(f"| {r['metric']} | {r['comparison']} | {fmt(r['mean_diff_second_minus_first'])} | {ci} | {fmt(r['wilcoxon_p'],6)} | {r['n_paired']} |")

    lines += ["", "## 4. USR case-level worsening probability", "",
              "| USR comparison | n | Dice worsened | Boundary Dice worsened |",
              "|---|---:|---:|---:|"]
    for r in worsen_rows:
        lines.append(f"| {r['comparison']} | {r['n']} | {fmt(r['dice_worsen_rate'])} | {fmt(r['boundary_dice_worsen_rate'])} |")

    if uncertainty_summary:
        lines += ["", "## 5. Uncertainty validation", "",
                  "| Variant | Metric | Mean | SD | n |",
                  "|---|---|---:|---:|---:|"]
        for r in uncertainty_summary:
            lines.append(f"| {r['variant']} | {r['metric']} | {fmt(r['mean'])} | {fmt(r['sd'])} | {r['n_finite']} |")

    if assoc_rows:
        lines += ["", "## 6. Segmentation-report associations", "",
                  "Full Spearman results are in `segmentation_report_associations.csv`.",
                  "This table shows the largest absolute associations with at least 3 cases.", "",
                  "| Variant | Seg metric | Report metric | n | rho | p |",
                  "|---|---|---|---:|---:|---:|"]
        good = [r for r in assoc_rows if np.isfinite(float(r.get("spearman_rho", np.nan)))]
        good.sort(key=lambda r: abs(float(r["spearman_rho"])), reverse=True)
        for r in good[:20]:
            lines.append(f"| {r['variant']} | {r['segmentation_metric']} | {r['report_metric']} | {r['n']} | {fmt(r['spearman_rho'])} | {fmt(r['p_value'],6)} |")

    lines += [
        "", "## Output files", "",
        "- `case_segmentation_metrics.csv`: all per-case segmentation metrics.",
        "- `summary_segmentation_metrics.csv`: variant-level summary.",
        "- `factorial_2x2_effects.csv`: ABLoss/USR main and interaction effects.",
        "- `paired_tests.csv`: paired Wilcoxon + bootstrap confidence intervals.",
        "- `usr_pixel_audit.csv`: semantic pixel accounting requested by Reviewer #2.",
        "- `usr_worsen_summary.csv`: probability that USR worsens case-level Dice/Boundary Dice.",
        "- `uncertainty_case_metrics.csv` / `uncertainty_summary.csv`: written when prob+unc maps exist.",
        "- `risk_coverage_*.csv`: mean risk-coverage curve for each uncertainty-enabled variant.",
        "- `segmentation_report_associations.csv`: written when report-metrics CSV is provided.",
        "", "## Interpretation guardrails", "",
        "- Do not treat automated SEIG rule compliance as clinical correctness.",
        "- Report human validation precision/recall/F1 separately when those annotations are available.",
        "- Do not replace non-finite HD95/ASSD cases with arbitrary constants; this runner reports their counts explicitly.",
        "- New revision results must come from actual reruns; this script does not copy manuscript numbers into outputs.",
    ]
    (outdir / "README_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    paths = {
        "gt": Path(args.gt),
        "baseline": Path(args.baseline),
        "abloss": Path(args.abloss),
        "usr": Path(args.usr),
        "both": Path(args.both),
    }
    maps = {k: _find_files(p, args.recursive) for k, p in paths.items()}

    common = set(maps["gt"])
    for v in VARIANTS:
        common &= set(maps[v])
    case_ids = sorted(common)
    if not case_ids:
        counts = {k: len(v) for k, v in maps.items()}
        raise RuntimeError(f"No common case stems across GT + four variants. Counts={counts}")

    missing_notes = []
    for k, mp in maps.items():
        if len(mp) != len(case_ids):
            missing_notes.append(f"{k}: found {len(mp)}, common paired {len(case_ids)}")

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    # Optional uncertainty folders.
    prob_dirs = {v: getattr(args, f"prob_{v}") for v in VARIANTS}
    unc_dirs = {v: getattr(args, f"unc_{v}") for v in VARIANTS}
    prob_maps: Dict[str, Optional[Dict[str, Path]]] = {}
    unc_maps: Dict[str, Optional[Dict[str, Path]]] = {}
    for v in VARIANTS:
        prob_maps[v] = _find_files(Path(prob_dirs[v]), args.recursive) if prob_dirs[v] else None
        unc_maps[v] = _find_files(Path(unc_dirs[v]), args.recursive) if unc_dirs[v] else None

    case_rows: List[Dict[str, Any]] = []
    by_variant_metric: Dict[str, Dict[str, List[float]]] = {v: {m: [] for m in METRICS} for v in VARIANTS}
    masks_cache: Dict[str, Dict[str, np.ndarray]] = {v: {} for v in VARIANTS}
    gt_cache: Dict[str, np.ndarray] = {}

    unc_case_rows: List[Dict[str, Any]] = []
    unc_curves: Dict[str, List[Dict[str, Any]]] = {v: [] for v in VARIANTS}

    print(f"[1/6] Evaluating {len(case_ids)} common cases ...")
    for idx, cid in enumerate(case_ids, 1):
        gt = _load_mask(maps["gt"][cid])
        gt_cache[cid] = gt
        shape = gt.shape
        for v in VARIANTS:
            pred = _load_mask(maps[v][cid], shape)
            masks_cache[v][cid] = pred
            met = core.segmentation_metrics(pred, gt)
            row = {"case_id": cid, "variant_key": v, "variant": VARIANT_LABELS[v]}
            row.update(met)
            case_rows.append(row)
            for m in METRICS:
                by_variant_metric[v][m].append(float(met[m]))

            pm, um = prob_maps[v], unc_maps[v]
            if pm is not None and um is not None and cid in pm and cid in um:
                prob = _load_prob(pm[cid], shape)
                unc = _load_prob(um[cid], shape)
                umat = core.uncertainty_metrics(prob, unc, gt, pred_mask=pred)
                urow = {
                    "case_id": cid, "variant_key": v, "variant": VARIANT_LABELS[v],
                    "error_detection_auroc": umat["error_detection_auroc"],
                    "brier": umat["brier"], "ece_15bin": umat["ece_15bin"], "aurc": umat["aurc"],
                }
                unc_case_rows.append(urow)
                unc_curves[v].append(umat["risk_coverage"])
        if idx % max(1, len(case_ids)//10) == 0 or idx == len(case_ids):
            print(f"      {idx}/{len(case_ids)}")

    _write_csv(outdir / "case_segmentation_metrics.csv", case_rows)

    print("[2/6] Summaries + 2x2 factorial effects ...")
    summary_rows: List[Dict[str, Any]] = []
    for v in VARIANTS:
        for m in METRICS:
            s = _finite_stats(by_variant_metric[v][m])
            summary_rows.append({"variant_key": v, "variant": VARIANT_LABELS[v], "metric": m, **s})
    _write_csv(outdir / "summary_segmentation_metrics.csv", summary_rows)

    factorial_rows = []
    for m in METRICS:
        cells = {v: by_variant_metric[v][m] for v in VARIANTS}
        factorial_rows.append(_factorial_effects(cells, m))
    _write_csv(outdir / "factorial_2x2_effects.csv", factorial_rows)

    print("[3/6] Paired Wilcoxon + bootstrap 95% CIs ...")
    comparisons = [
        ("baseline", "abloss"),
        ("baseline", "usr"),
        ("baseline", "both"),
        ("abloss", "both"),
        ("usr", "both"),
    ]
    paired_rows = []
    for m in METRICS:
        for a, b in comparisons:
            xa, xb = by_variant_metric[a][m], by_variant_metric[b][m]
            p, n = _paired_wilcoxon(xa, xb)
            est, lo, hi, nb = _bootstrap_mean_diff(xa, xb, args.bootstrap, args.seed)
            paired_rows.append({
                "metric": m,
                "direction": "higher_better" if m in HIGHER_IS_BETTER else "lower_better",
                "first": VARIANT_LABELS[a], "second": VARIANT_LABELS[b],
                "comparison": f"{VARIANT_LABELS[a]} -> {VARIANT_LABELS[b]}",
                "mean_diff_second_minus_first": est,
                "ci95_low": lo, "ci95_high": hi,
                "wilcoxon_p": p, "n_paired": min(n, nb),
            })
    _write_csv(outdir / "paired_tests.csv", paired_rows)

    print("[4/6] Reviewer-requested USR semantic pixel audit ...")
    audit_rows = []
    worsen_rows = []
    usr_pairs = [
        ("baseline", "usr", "MedSAM -> MedSAM+USR"),
        ("abloss", "both", "MedSAM+ABLoss -> MedSAM+ABLoss+USR"),
    ]
    for before_key, after_key, label in usr_pairs:
        dice_worse, bd_worse = [], []
        for cid in case_ids:
            audit = core.usr_correction_audit(masks_cache[before_key][cid], masks_cache[after_key][cid], gt_cache[cid])
            audit_rows.append({"case_id": cid, "comparison": label, **audit})
            dice_worse.append(float(audit["dice_worsened"]))
            bd_worse.append(float(audit["boundary_dice_worsened"]))
        worsen_rows.append({
            "comparison": label, "n": len(case_ids),
            "dice_worsen_count": int(np.sum(dice_worse)),
            "dice_worsen_rate": float(np.mean(dice_worse)),
            "boundary_dice_worsen_count": int(np.sum(bd_worse)),
            "boundary_dice_worsen_rate": float(np.mean(bd_worse)),
        })
    _write_csv(outdir / "usr_pixel_audit.csv", audit_rows)
    _write_csv(outdir / "usr_worsen_summary.csv", worsen_rows)

    print("[5/6] Uncertainty validation (when maps are available) ...")
    uncertainty_summary: List[Dict[str, Any]] = []
    if unc_case_rows:
        _write_csv(outdir / "uncertainty_case_metrics.csv", unc_case_rows)
        for v in VARIANTS:
            vv = [r for r in unc_case_rows if r["variant_key"] == v]
            for m in ("error_detection_auroc", "brier", "ece_15bin", "aurc"):
                s = _finite_stats([float(r[m]) for r in vv])
                uncertainty_summary.append({"variant_key": v, "variant": VARIANT_LABELS[v], "metric": m, **s})
            rc = _mean_risk_coverage(unc_curves[v])
            if rc:
                _write_csv(outdir / f"risk_coverage_{v}.csv", rc)
        _write_csv(outdir / "uncertainty_summary.csv", uncertainty_summary)
    else:
        (outdir / "UNCERTAINTY_NOT_RUN.txt").write_text(
            "No variant had BOTH probability and uncertainty folders with matching case stems.\n"
            "Provide --prob-* and --unc-* arguments to run AUROC/Brier/ECE/risk-coverage.\n",
            encoding="utf-8",
        )

    print("[6/6] Optional segmentation-report association analysis ...")
    assoc_rows: List[Dict[str, Any]] = []
    if args.report_metrics_csv:
        assoc_rows = _report_associations(case_rows, Path(args.report_metrics_csv))
        _write_csv(outdir / "segmentation_report_associations.csv", assoc_rows)

    # Provenance/config record.
    run_config = vars(args).copy()
    run_config.update({
        "n_common_cases": len(case_ids),
        "case_ids": case_ids,
        "pairing_notes": missing_notes,
        "variant_labels": VARIANT_LABELS,
        "metric_directions": {m: ("higher_better" if m in HIGHER_IS_BETTER else "lower_better") for m in METRICS},
    })
    (outdir / "run_config.json").write_text(json.dumps(_json_safe(run_config), indent=2, ensure_ascii=False), encoding="utf-8")
    _make_markdown(outdir, len(case_ids), summary_rows, factorial_rows, paired_rows, worsen_rows, uncertainty_summary, assoc_rows)

    print("\n==================== REVIEWER #2 ANALYSIS COMPLETE ====================")
    print(f"Common paired cases : {len(case_ids)}")
    if missing_notes:
        print("Pairing notes        : " + " | ".join(missing_notes))
    print(f"Results directory   : {outdir.resolve()}")
    print(f"Paper-ready summary : {(outdir / 'README_RESULTS.md').resolve()}")
    print("======================================================================")


def _synthetic_circle(h: int, w: int, cy: int, cx: int, r: int) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    return (((yy-cy)**2 + (xx-cx)**2) <= r*r).astype(np.uint8)


def smoke_test() -> None:
    import tempfile
    rng = np.random.default_rng(2026)
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        dirs = {k: root/k for k in ["gt", "baseline", "abloss", "usr", "both", "prob_usr", "unc_usr", "prob_both", "unc_both"]}
        for d in dirs.values(): d.mkdir(parents=True, exist_ok=True)
        for i in range(8):
            gt = _synthetic_circle(96, 112, 48+rng.integers(-3,4), 56+rng.integers(-3,4), 20+rng.integers(-2,3))
            # Deliberately create a hierarchy of masks; USR occasionally worsens one edge.
            b = np.roll(gt, shift=(2, -2), axis=(0,1))
            a = np.roll(gt, shift=(1, -1), axis=(0,1))
            u = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
            ab = cv2.morphologyEx(a, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
            if i == 2:
                u = cv2.erode(u, np.ones((5,5), np.uint8), iterations=1)
            for key, arr in (("gt",gt),("baseline",b),("abloss",a),("usr",u),("both",ab)):
                cv2.imwrite(str(dirs[key]/f"case_{i:03d}.png"), arr*255)
            # Soft probability / entropy maps for USR variants.
            for key, arr in (("usr",u),("both",ab)):
                p = cv2.GaussianBlur(arr.astype(np.float32), (9,9), 2.0)
                p = np.clip(0.05 + 0.9*p, 0, 1)
                ent = -(p*np.log2(p+1e-8) + (1-p)*np.log2(1-p+1e-8))
                np.save(dirs[f"prob_{key}"]/f"case_{i:03d}.npy", p)
                np.save(dirs[f"unc_{key}"]/f"case_{i:03d}.npy", ent)
        args = argparse.Namespace(
            gt=str(dirs["gt"]), baseline=str(dirs["baseline"]), abloss=str(dirs["abloss"]), usr=str(dirs["usr"]), both=str(dirs["both"]),
            prob_baseline=None, unc_baseline=None, prob_abloss=None, unc_abloss=None,
            prob_usr=str(dirs["prob_usr"]), unc_usr=str(dirs["unc_usr"]), prob_both=str(dirs["prob_both"]), unc_both=str(dirs["unc_both"]),
            output=str(root/"results"), report_metrics_csv=None, recursive=False, bootstrap=200, seed=2026,
        )
        run(args)
        must = [
            "case_segmentation_metrics.csv", "summary_segmentation_metrics.csv", "factorial_2x2_effects.csv",
            "paired_tests.csv", "usr_pixel_audit.csv", "usr_worsen_summary.csv", "uncertainty_summary.csv", "README_RESULTS.md",
        ]
        for f in must:
            assert (root/"results"/f).exists(), f"Missing smoke-test output: {f}"
    print("REVIEWER2 EXPERIMENT RUNNER SMOKE TEST PASSED")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="WYSIWYR Reviewer #2 2x2 + USR + uncertainty analysis runner",
    )
    p.add_argument("--smoke-test", action="store_true", help="Run synthetic self-test; no data needed")
    p.add_argument("--gt", help="Ground-truth mask folder")
    p.add_argument("--baseline", help="MedSAM prediction mask folder")
    p.add_argument("--abloss", help="MedSAM+ABLoss prediction mask folder")
    p.add_argument("--usr", help="MedSAM+USR prediction mask folder")
    p.add_argument("--both", help="MedSAM+ABLoss+USR prediction mask folder")

    for v in VARIANTS:
        p.add_argument(f"--prob-{v}", dest=f"prob_{v}", help=f"Optional {VARIANT_LABELS[v]} foreground probability-map folder")
        p.add_argument(f"--unc-{v}", dest=f"unc_{v}", help=f"Optional {VARIANT_LABELS[v]} uncertainty-map folder")

    p.add_argument("--report-metrics-csv", help="Optional case-level report-error CSV for Spearman association analysis")
    p.add_argument("--output", default="reviewer2_results", help="Output directory (default: reviewer2_results)")
    p.add_argument("--recursive", action="store_true", help="Search folders recursively")
    p.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples for paired mean-difference CI (default: 2000)")
    p.add_argument("--seed", type=int, default=2026, help="Random seed (default: 2026)")
    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()
    if args.smoke_test:
        smoke_test(); return
    required = ["gt", "baseline", "abloss", "usr", "both"]
    missing = [x for x in required if not getattr(args, x)]
    if missing:
        p.error("Missing required arguments: " + ", ".join("--"+m for m in missing))
    run(args)


if __name__ == "__main__":
    main()
