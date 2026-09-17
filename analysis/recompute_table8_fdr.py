#!/usr/bin/env python3
"""Independent reproduction of the paired-Wilcoxon family used by the manuscript's
segmentation significance table ("Table 8"), with a single joint
Benjamini-Hochberg FDR correction over the declared family.

What it does
------------
1. Reads ONLY the frozen per-case segmentation metrics
   (`stage5_calibrated/reviewer2_results/<dataset>/case_segmentation_metrics.csv`).
2. Recomputes the two-sided paired Wilcoxon signed-rank p-value for every
   (dataset, metric, comparison) cell using EXACTLY the implementation in
   `run_reviewer2_experiments.py::_paired_wilcoxon`:
       - non-finite pairs dropped
       - d = second - first; if allclose(d, 0) -> p = 1.0
       - else scipy.stats.wilcoxon(second, first, alternative="two-sided",
                                   zero_method="wilcox", method="auto")
   No other statistical method is substituted.
3. Cross-checks every recomputed raw p against the frozen
   `paired_tests.csv::wilcoxon_p` of the same repository, and reports the
   maximum absolute deviation. A non-zero deviation is reported, never hidden,
   and the frozen artifacts are never modified.
4. Applies ONE joint Benjamini-Hochberg FDR correction per declared family scope
   (see --family). No per-cell correction, no star-based back-solving.

Family scopes
-------------
The frozen runner emits 5 datasets x 7 metrics x 5 prespecified comparisons.
The manuscript's Table 8 uses five metrics (dice, iou, boundary_dice, hd95,
assd). Because the manuscript source is not part of this repository, the family
is declared explicitly rather than guessed; every emitted row carries the
`family` it belongs to.

    table8           5 metrics x 5 datasets x 5 comparisons      = 125  (default)
    table8_main      5 metrics x 5 datasets x 1 main comparison  =  25
    table8_per_dataset   BH computed within each dataset         =  25 each
    all7             7 metrics x 5 datasets x 5 comparisons      = 175

Outputs (>= 8 significant digits kept in the CSV/JSON):
    <out>/table8_wilcoxon_bh_fdr.csv
    <out>/table8_wilcoxon_bh_fdr.json
    <out>/table8_reproduction_check.csv     (recomputed vs frozen raw p)

Usage
-----
    python analysis/recompute_table8_fdr.py
    python analysis/recompute_table8_fdr.py --family table8
    python analysis/recompute_table8_fdr.py --data-root <wysiwyr_real> --out-dir <dir>
    python analysis/recompute_table8_fdr.py --smoke-test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

DATA_ROOT = Path(os.environ.get("WYSIWYR_DATA_ROOT", ".")).expanduser() / "wysiwyr_real"
RESULTS_SUBDIR = Path("stage5_calibrated") / "reviewer2_results"
OUT_DEFAULT = Path(__file__).resolve().parent.parent / "reproducibility_outputs"

DATASETS = ["CVC-300", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir"]
# same order and names as run_reviewer2_experiments.py
METRICS_ALL = ["dice", "iou", "precision", "recall", "boundary_dice", "hd95", "assd"]
METRICS_TABLE8 = ["dice", "iou", "boundary_dice", "hd95", "assd"]
HIGHER_IS_BETTER = {"dice", "iou", "precision", "recall", "boundary_dice"}
VARIANTS = ["baseline", "abloss", "usr", "both"]
VARIANT_LABELS = {"baseline": "MedSAM", "abloss": "MedSAM+ABLoss",
                  "usr": "MedSAM+USR", "both": "MedSAM+ABLoss+USR"}
COMPARISONS = [("baseline", "abloss"), ("baseline", "usr"), ("baseline", "both"),
               ("abloss", "both"), ("usr", "both")]
MAIN_COMPARISON = ("baseline", "both")


# --------------------------------------------------------------------------
# statistics — identical to run_reviewer2_experiments.py::_paired_wilcoxon
# --------------------------------------------------------------------------
def paired_wilcoxon(a, b):
    """Two-sided paired Wilcoxon; returns (p, n). Mirrors the frozen runner."""
    x = np.asarray(a, float)
    y = np.asarray(b, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) == 0:
        return float("nan"), 0
    d = y - x
    if np.allclose(d, 0):
        return 1.0, int(len(x))
    try:
        res = wilcoxon(y, x, alternative="two-sided", zero_method="wilcox",
                       method="auto")
        return float(res.pvalue), int(len(x))
    except Exception:
        return float("nan"), int(len(x))


def bh_fdr(pvals):
    """Benjamini-Hochberg step-up FDR. Pure Python; no statsmodels required.

    Returns q-values in the input order; NaN p-values are passed through as NaN
    and excluded from the correction.
    """
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    fin = np.isfinite(p)
    m = int(fin.sum())
    if m == 0:
        return out
    idx = np.flatnonzero(fin)
    order = idx[np.argsort(p[idx], kind="mergesort")]
    ranked = p[order]
    q = np.empty(m, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        val = ranked[i] * m / (i + 1)
        prev = min(prev, val)
        q[i] = prev
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def _bh_sanity_check():
    """Verify bh_fdr with exact hand-checkable cases + statsmodels (when available).

    The hand-worked cases are small enough to verify by arithmetic:
      m=1                 -> q == p
      [0.01, 0.04]        -> [0.02, 0.04]
      [0.04, 0.01]        -> [0.04, 0.02]   (input order preserved)
      [0.5, 0.5, 0.5]     -> [0.5, 0.5, 0.5] (clip at 1.0 then enforce monotonicity)
    """
    assert np.isclose(bh_fdr([0.03])[0], 0.03)
    assert np.allclose(bh_fdr([0.01, 0.04]), [0.02, 0.04])
    assert np.allclose(bh_fdr([0.04, 0.01]), [0.04, 0.02])
    assert np.allclose(bh_fdr([0.5, 0.5, 0.5]), [0.5, 0.5, 0.5])
    # BH q is never smaller than the raw p, and never exceeds 1
    rng = np.random.default_rng(0)
    pv = rng.uniform(0, 1, size=200)
    q = bh_fdr(pv)
    assert np.all(q >= pv - 1e-12) and np.all(q <= 1.0)
    # NaN passthrough
    g2 = bh_fdr([0.01, float("nan"), 0.04])
    assert np.isnan(g2[1]) and np.isfinite(g2[0]) and np.isfinite(g2[2])
    # authoritative cross-check
    try:
        from statsmodels.stats.multitest import multipletests
        _, q_sm, _, _ = multipletests(pv, method="fdr_bh")
        assert np.allclose(q, q_sm, atol=1e-12), "bh_fdr disagrees with statsmodels"
        cross = "statsmodels multipletests(fdr_bh) agreement verified on 200 random p-values"
    except ImportError:
        cross = "statsmodels not installed; hand-worked cases + property checks only"
    return cross


# --------------------------------------------------------------------------
# data loading
# --------------------------------------------------------------------------
def load_frozen(results_root: Path, datasets):
    """Return ({dataset: per-case DataFrame}, frozen paired_tests DataFrame)."""
    frames, frozen_rows = {}, []
    label_to_key = {v: k for k, v in VARIANT_LABELS.items()}
    for ds in datasets:
        case_csv = results_root / ds / "case_segmentation_metrics.csv"
        if not case_csv.exists():
            raise SystemExit(f"missing frozen per-case metrics: {case_csv}")
        df = pd.read_csv(case_csv, encoding="utf-8-sig")
        df["case_id"] = df["case_id"].astype(str)
        if "variant_key" not in df.columns:
            # older layout uses the human-readable label column
            if "variant" not in df.columns:
                raise SystemExit(f"no variant column in {case_csv}")
            df["variant_key"] = df["variant"].map(label_to_key)
            if df["variant_key"].isna().any():
                raise SystemExit(f"unmappable variant labels in {case_csv}: "
                                 f"{sorted(df.loc[df['variant_key'].isna(),'variant'].unique())}")
        frames[ds] = df

        pt = results_root / ds / "paired_tests.csv"
        if pt.exists():
            f = pd.read_csv(pt, encoding="utf-8-sig")
            f["dataset"] = ds
            f["comparison_key"] = f["comparison"].map(
                lambda s: "->".join(label_to_key.get(p.strip(), p.strip())
                                    for p in str(s).split("->")))
            frozen_rows.append(f)
    frozen = pd.concat(frozen_rows, ignore_index=True) if frozen_rows else pd.DataFrame()
    return frames, frozen


def per_case_metric(frame, variant, metric):
    sub = frame[frame["variant_key"] == variant]
    return dict(zip(sub["case_id"], sub[metric]))


# --------------------------------------------------------------------------
# main computation
# --------------------------------------------------------------------------
def build_rows(frames, datasets, metrics, comparisons):
    rows = []
    for ds in datasets:
        frame = frames[ds]
        for metric in metrics:
            per_variant = {v: per_case_metric(frame, v, metric) for v in VARIANTS}
            for v1, v2 in comparisons:
                m1, m2 = per_variant.get(v1, {}), per_variant.get(v2, {})
                common = sorted(set(m1) & set(m2))       # explicit case_id pairing
                a = [m1[c] for c in common]
                b = [m2[c] for c in common]
                p, n = paired_wilcoxon(a, b)
                fa = np.asarray(a, float)
                fb = np.asarray(b, float)
                ok = np.isfinite(fa) & np.isfinite(fb)
                diff = fb[ok] - fa[ok]
                rows.append({
                    "dataset": ds,
                    "metric": metric,
                    "direction": "higher_better" if metric in HIGHER_IS_BETTER else "lower_better",
                    "first": VARIANT_LABELS[v1],
                    "second": VARIANT_LABELS[v2],
                    "comparison": f"{VARIANT_LABELS[v1]} -> {VARIANT_LABELS[v2]}",
                    "comparison_key": f"{v1}->{v2}",
                    "n_paired": n,
                    "mean_paired_diff_second_minus_first": float(np.mean(diff)) if len(diff) else float("nan"),
                    "median_paired_diff_second_minus_first": float(np.median(diff)) if len(diff) else float("nan"),
                    "raw_p": p,
                })
    return pd.DataFrame(rows)


def annotate_families(df, frames, datasets, comparisons):
    """Attach BH q-values for every declared family scope."""
    df["family"] = ""
    df["bh_q"] = np.nan
    df["raw_sig"] = df["raw_p"] < 0.05

    out = []
    scopes = []
    # 1) table8 : 5 metrics x 5 datasets x 5 comparisons
    scopes.append(("table8", (df.metric.isin(METRICS_TABLE8)), True))
    # 2) table8_main : 5 metrics x 5 datasets x main comparison only
    scopes.append(("table8_main", (df.metric.isin(METRICS_TABLE8)
                                   & (df.comparison_key == f"{MAIN_COMPARISON[0]}->{MAIN_COMPARISON[1]}")), True))
    # 3) table8_per_dataset : within-dataset BH over 5 metrics x 5 comparisons
    for ds in datasets:
        scopes.append((f"table8_per_dataset:{ds}",
                       (df.metric.isin(METRICS_TABLE8) & (df.dataset == ds)), True))
    # 4) all7 : every metric/cohort/comparison the frozen runner emitted
    scopes.append(("all7", pd.Series(True, index=df.index), True))

    for name, mask, _ in scopes:
        sub = df[mask].copy()
        if sub.empty:
            continue
        sub["family"] = name
        sub["bh_q"] = bh_fdr(sub["raw_p"].to_numpy())
        sub["raw_sig"] = sub["raw_p"] < 0.05
        sub["bh_sig"] = sub["bh_q"] < 0.05
        out.append(sub)
    res = pd.concat(out, ignore_index=True)
    res["bh_q_lt_0.001"] = res["bh_q"] < 0.001
    res["bh_q_lt_0.01"] = res["bh_q"] < 0.01
    res["bh_q_lt_0.05"] = res["bh_q"] < 0.05
    return res


def verify_against_frozen(rowdf, frozen):
    """Recomputed raw p vs the frozen paired_tests.csv::wilcoxon_p."""
    if frozen is None or frozen.empty:
        return pd.DataFrame(), {"status": "NO_FROZEN_PAIRED_TESTS_FOUND"}
    base = rowdf[rowdf["comparison_key"].notna()][
        ["dataset", "metric", "comparison_key", "n_paired", "raw_p"]]
    m = base.merge(frozen[["dataset", "metric", "comparison_key", "wilcoxon_p", "n_paired"]],
                   on=["dataset", "metric", "comparison_key"], how="outer",
                   suffixes=("_recomputed", "_frozen"))
    m["abs_p_diff"] = (m["raw_p"] - m["wilcoxon_p"]).abs()
    m["n_match"] = m["n_paired_recomputed"] == m["n_paired_frozen"]
    finite = m["abs_p_diff"].dropna()
    summary = {
        "cells_compared": int(len(m)),
        "cells_missing_on_either_side": int(m["raw_p"].isna().sum() + m["wilcoxon_p"].isna().sum()),
        "max_abs_p_diff": float(finite.max()) if len(finite) else None,
        "n_cells_with_p_diff_gt_1e-9": int((finite > 1e-9).sum()) if len(finite) else 0,
        "n_cells_with_n_mismatch": int((~m["n_match"].fillna(False)).sum()),
        "status": "MATCH" if (len(finite) and finite.max() <= 1e-9) else "DEVIATION",
    }
    return m, summary


# --------------------------------------------------------------------------
def smoke_test():
    cross = _bh_sanity_check()
    # identical inputs -> p == 1
    p, n = paired_wilcoxon([1, 2, 3, 4], [1, 2, 3, 4])
    assert p == 1.0 and n == 4
    # known two-sided value for a small hand-checkable pair set
    p2, n2 = paired_wilcoxon([1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                             [2, 2, 4, 4, 6, 6, 8, 8, 10, 12])
    assert 0.0 < p2 < 1.0 and n2 == 10
    print("BH sanity check:", cross)
    print("zero-difference handling:", p)
    print("example paired p:", p2)
    print("RECOMPUTE_TABLE8_FDR SMOKE TEST PASSED")


def main():
    ap = argparse.ArgumentParser(
        description="Independently recompute the paired-Wilcoxon family behind the "
                    "segmentation significance table and apply one joint BH-FDR "
                    "correction per declared family. Read-only w.r.t. frozen artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data-root", default=str(DATA_ROOT),
                    help="directory that contains stage5_calibrated/reviewer2_results/ "
                         "(i.e. <root>/wysiwyr_real)")
    ap.add_argument("--out-dir", default=str(OUT_DEFAULT), help="output directory")
    ap.add_argument("--family", default="all", choices=["all", "table8", "table8_main"],
                    help="which family scope(s) to emit; 'all' writes every scope")
    ap.add_argument("--smoke-test", action="store_true",
                    help="run the code-path/BH self test (no data needed)")
    args = ap.parse_args()

    if args.smoke_test:
        smoke_test()
        return

    results_root = Path(args.data_root).expanduser() / RESULTS_SUBDIR
    if not results_root.exists():
        raise SystemExit(f"missing frozen results root: {results_root}")
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("Table-8 paired-Wilcoxon + joint BH-FDR reproduction")
    print("=" * 78)
    print("input :", results_root)
    print("output:", out_dir)

    frames, frozen = load_frozen(results_root, DATASETS)
    n_cases = {ds: frames[ds]["case_id"].nunique() for ds in DATASETS}
    print("cohort cases per dataset:", n_cases, "total:", sum(n_cases.values()))

    rowdf = build_rows(frames, DATASETS, METRICS_ALL, COMPARISONS)
    print(f"recomputed cells (5 datasets x 7 metrics x 5 comparisons): {len(rowdf)}")

    chk, chk_summary = verify_against_frozen(rowdf, frozen)
    print("frozen cross-check:", json.dumps(chk_summary, indent=2))

    res = annotate_families(rowdf, frames, DATASETS, COMPARISONS)
    if args.family != "all":
        res = res[res["family"] == args.family].copy()

    cols = ["family", "dataset", "metric", "direction", "first", "second", "comparison",
            "n_paired", "mean_paired_diff_second_minus_first",
            "median_paired_diff_second_minus_first", "raw_p", "bh_q",
            "raw_sig", "bh_sig", "bh_q_lt_0.001", "bh_q_lt_0.01", "bh_q_lt_0.05"]
    res = res[cols].sort_values(["family", "dataset", "metric", "comparison"])

    csv_path = out_dir / "table8_wilcoxon_bh_fdr.csv"
    json_path = out_dir / "table8_wilcoxon_bh_fdr.json"
    chk_path = out_dir / "table8_reproduction_check.csv"
    # >= 8 significant digits
    res.to_csv(csv_path, index=False, float_format="%.10g")
    if len(chk):
        chk.to_csv(chk_path, index=False, float_format="%.10g")

    fam_summary = {}
    for fam, g in res.groupby("family"):
        fam_summary[fam] = {
            "n_tests": int(len(g)),
            "n_raw_p_lt_0.05": int(g["raw_sig"].sum()),
            "n_bh_q_lt_0.05": int(g["bh_sig"].sum()),
            "n_bh_q_lt_0.01": int(g["bh_q_lt_0.01"].sum()),
            "n_bh_q_lt_0.001": int(g["bh_q_lt_0.001"].sum()),
            "n_lost_significance_after_bh": int((g["raw_sig"] & ~g["bh_sig"]).sum()),
        }
    payload = {
        "generator": "analysis/recompute_table8_fdr.py",
        "input_root": str(results_root),
        "cohort_case_counts": n_cases,
        "statistical_test": ("scipy.stats.wilcoxon(second, first, alternative='two-sided', "
                             "zero_method='wilcox', method='auto')"),
        "zero_handling": "zero_method='wilcox'; all-zero difference vector -> p = 1.0",
        "correction": "Benjamini-Hochberg (fdr_bh), applied once per family scope",
        "bh_implementation": "in-repo pure-Python bh_fdr() "
                             "(cross-checked against statsmodels when available)",
        "families": fam_summary,
        "frozen_cross_check": chk_summary,
        "notes": [
            "Raw p-values are recomputed from frozen per-case metrics only; the frozen "
            "paired_tests.csv is used solely for verification and is never modified.",
            "The manuscript source is not part of this repository, so the family scope is "
            "declared explicitly in the 'family' column rather than inferred from the "
            "manuscript table layout.",
            "No significance star is reverse-engineered into a p-value.",
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nfamily summary:")
    for fam, s in fam_summary.items():
        print(f"  {fam:26s} n={s['n_tests']:4d} raw<0.05={s['n_raw_p_lt_0.05']:4d} "
              f"BH<0.05={s['n_bh_q_lt_0.05']:4d} BH<0.01={s['n_bh_q_lt_0.01']:4d} "
              f"BH<0.001={s['n_bh_q_lt_0.001']:4d} lost={s['n_lost_significance_after_bh']}")
    print("\nWROTE", csv_path)
    print("WROTE", json_path)
    if len(chk):
        print("WROTE", chk_path)
    if chk_summary.get("status") != "MATCH":
        print("\n[WARN] recomputed raw p does not exactly match the frozen artifact; "
              "see table8_reproduction_check.csv", file=sys.stderr)
    print("\nTABLE8 FDR REPRODUCTION DONE")


if __name__ == "__main__":
    main()
