#!/usr/bin/env python
"""Reviewer #2 Comment #1 -- case-level associations between segmentation
changes and downstream evidence/report changes.

Read-only w.r.t. all existing artifacts. CPU only. No GPU, no re-inference.

FROZEN DEFINITION (recovered from stage6_seig_mllm_end2end.py::aggregate and
README.md "3 settings x 6 associations"):

    3 settings (within-case contrast vs the MedSAM baseline)
        abloss - baseline   (MedSAM+ABLoss   - MedSAM)
        usr    - baseline   (MedSAM+USR      - MedSAM)
        both   - baseline   (MedSAM+ABLoss+USR - MedSAM)
    x
    6 prespecified metric pairs
        (delta_dice,              delta_symbolic_match)
        (delta_boundary_dice,     delta_symbolic_match)
        (delta_dice,              delta_raw_violations)
        (delta_boundary_dice,     delta_raw_violations)
        (delta_hd95,              delta_raw_violations)
        (delta_boundary_dice,     delta_boundary_overconfidence)
    = 18 prespecified association tests.

BH-FDR is applied ONCE jointly over all 18 raw p-values.

NEW in this script (additive, requested by the authors for the response letter):
    1,000 dataset-stratified case-level bootstrap resamples ->
    percentile 95% CI of Spearman rho. Seed 20260916.

An auxiliary NON-FROZEN 2-contrast x 9-metric scheme is also written to a
separately named diagnostic CSV only, clearly flagged, to resolve the
definition ambiguity. It must not be used unless explicitly confirmed.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# Data/artefact root. Set WYSIWYR_DATA_ROOT to the directory that contains
# `wysiwyr_real` (e.g. export WYSIWYR_DATA_ROOT=/path/to/data_root).
DATA_ROOT = Path(os.environ.get("WYSIWYR_DATA_ROOT", ".")).expanduser() / "wysiwyr_real"
S6 = DATA_ROOT / "stage6_seig_mllm_end2end"
OUT = ROOT / "results" / "reviewer2"


def build_parser():
    ap = argparse.ArgumentParser(
        description="Reviewer #2 Comment #1 -- 18 prespecified case-level associations "
                    "(two-sided Spearman + 1,000 dataset-stratified bootstrap CI + joint BH-FDR). "
                    "Reads frozen artefacts only; never re-runs any model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data-root", default=str(DATA_ROOT),
                    help="directory that contains stage6_seig_mllm_end2end/ (i.e. <root>/wysiwyr_real)")
    ap.add_argument("--out-dir", default=str(OUT), help="output directory")
    ap.add_argument("--seed", type=int, default=SEED, help="bootstrap RNG seed")
    ap.add_argument("--n-boot", type=int, default=N_BOOT, help="bootstrap resamples")
    return ap


def apply_args(args):
    """Rebind the module-level roots/knobs from parsed CLI arguments."""
    global DATA_ROOT, S6, OUT, SEED, N_BOOT
    DATA_ROOT = Path(args.data_root).expanduser()
    S6 = DATA_ROOT / "stage6_seig_mllm_end2end"
    OUT = Path(args.out_dir).expanduser()
    SEED = int(args.seed)
    N_BOOT = int(args.n_boot)
    OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ["CVC-300", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir"]
VARIANTS = ["baseline", "abloss", "usr", "both"]
TARGETS = ["abloss", "usr", "both"]
LABEL = {"baseline": "MedSAM", "abloss": "MedSAM+ABLoss", "usr": "MedSAM+USR",
         "both": "MedSAM+ABLoss+USR"}

SEED = 20260916
N_BOOT = 1000

# the 6 prespecified pairs (segmentation change, downstream change)
PAIRS = [
    ("d_dice", "d_symbolic_match"),
    ("d_boundary_dice", "d_symbolic_match"),
    ("d_dice", "d_raw_violations"),
    ("d_boundary_dice", "d_raw_violations"),
    ("d_hd95", "d_raw_violations"),
    ("d_boundary_dice", "d_boundary_oc"),
]
PRETTY = {
    "d_dice": r"$\Delta$Dice",
    "d_boundary_dice": r"$\Delta$Boundary Dice",
    "d_hd95": r"$\Delta$HD95",
    "d_symbolic_match": r"$\Delta$Symbolic match",
    "d_raw_violations": r"$\Delta$Raw violations",
    "d_boundary_oc": r"$\Delta$Boundary overconfidence",
}
CONTRAST_PRETTY = {
    "abloss": r"MedSAM+ABLoss $-$ MedSAM",
    "usr": r"MedSAM+USR $-$ MedSAM",
    "both": r"MedSAM+ABLoss+USR $-$ MedSAM",
}


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fmt(x: float, nd: int = 3) -> str:
    return "NA" if not np.isfinite(x) else f"{x:.{nd}f}"


def bh(pvals):
    """Benjamini-Hochberg, identical to statsmodels fdr_bh."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    fin = np.isfinite(p)
    if fin.sum() == 0:
        return out
    _, q, _, _ = multipletests(p[fin], method="fdr_bh")
    out[fin] = q
    return out


def main():
    apply_args(build_parser().parse_args())
    clr = pd.read_csv(S6 / "case_level_results.csv", encoding="utf-8-sig")
    clr["case_id"] = clr["case_id"].astype(str)

    # ---- explicit merge by (dataset, case_id); NEVER by row order ----------
    by_case = defaultdict(dict)
    for r in clr[clr["variant_key"].isin(VARIANTS)].itertuples(index=False):
        by_case[(r.dataset, r.case_id)][r.variant_key] = r
    assert len(by_case) == 798, f"expected 798 model cases, got {len(by_case)}"

    # ---- per-case change scores -------------------------------------------
    change_rows, drows_by_target = [], {t: [] for t in TARGETS}
    for (ds, cid), vv in sorted(by_case.items()):
        if "baseline" not in vv:
            continue
        b = vv["baseline"]
        for t in TARGETS:
            if t not in vv:
                continue
            a = vv[t]
            row = {
                "case_id": cid, "dataset": ds, "contrast": f"{t}_minus_baseline",
                "d_dice": a.dice - b.dice,
                "d_boundary_dice": a.boundary_dice - b.boundary_dice,
                "d_hd95": a.hd95 - b.hd95,
                "d_symbolic_match": a.symbolic_match_fraction - b.symbolic_match_fraction,
                "d_raw_violations": a.raw_violation_count - b.raw_violation_count,
                "d_boundary_oc": (a.raw_boundary_overconfidence_count
                                  - b.raw_boundary_overconfidence_count),
            }
            change_rows.append(row)
            drows_by_target[t].append(row)

    change = pd.DataFrame(change_rows)
    change.to_csv(OUT / "r1_case_level_change_scores.csv", index=False)
    print(f"[write] r1_case_level_change_scores.csv  rows={len(change)}")
    print(change.groupby("contrast").size().to_string())

    # ---- 18 association tests ---------------------------------------------
    rng = np.random.default_rng(SEED)
    results = []
    for t in TARGETS:
        rows = drows_by_target[t]
        ds_all = np.asarray([r["dataset"] for r in rows], dtype=object)
        for x, y in PAIRS:
            xv = np.asarray([r[x] for r in rows], dtype=float)
            yv = np.asarray([r[y] for r in rows], dtype=float)
            ok = np.isfinite(xv) & np.isfinite(yv)
            xv, yv, dsv = xv[ok], yv[ok], ds_all[ok]
            n = int(len(xv))

            # exact frozen guard
            if n >= 3 and np.any(xv != xv[0]) and np.any(yv != yv[0]):
                rho, p = spearmanr(xv, yv)
                rho, p = float(rho), float(p)
            else:
                rho, p = float("nan"), float("nan")

            # ---- dataset-stratified case-level bootstrap of rho ----
            if n > 0 and np.isfinite(rho):
                idx_by_ds = {d: np.flatnonzero(dsv == d) for d in np.unique(dsv)}
                boots = []
                for _ in range(N_BOOT):
                    parts = [rng.choice(ix, size=len(ix), replace=True)
                             for ix in idx_by_ds.values()]
                    sel = np.concatenate(parts)
                    bx, by_ = xv[sel], yv[sel]
                    if np.any(bx != bx[0]) and np.any(by_ != by_[0]):
                        boots.append(spearmanr(bx, by_)[0])
                if boots:
                    lo, hi = np.percentile(boots, [2.5, 97.5])
                else:
                    lo = hi = float("nan")
                n_boot = len(boots)
            else:
                lo = hi = float("nan")
                n_boot = 0

            results.append({
                "contrast": f"{t}_minus_baseline", "target": t,
                "segmentation_change": x, "downstream_change": y,
                "n_paired": n, "rho": rho, "ci_low": float(lo), "ci_high": float(hi),
                "p_raw": p, "n_bootstrap_finite": n_boot,
            })

    res = pd.DataFrame(results)
    assert len(res) == 18, f"expected 18 tests, got {len(res)}"

    # joint BH-FDR over all 18 raw p-values
    res["q_bh"] = bh(res["p_raw"].to_numpy())
    res["fdr_significant"] = res["q_bh"] < 0.05
    res["ci_excludes_zero"] = (res["ci_low"] > 0) | (res["ci_high"] < 0)

    # ---- VERIFY rho against the frozen stored artifact ---------------------
    stored = pd.read_csv(S6 / "segmentation_to_report_associations.csv",
                         encoding="utf-8-sig")
    stored = stored[stored["scope"].str.startswith("within_case_")].copy()
    NAME_MAP = {
        "delta_dice": "d_dice", "delta_boundary_dice": "d_boundary_dice",
        "delta_hd95": "d_hd95", "delta_symbolic_match": "d_symbolic_match",
        "delta_raw_violations": "d_raw_violations",
        "delta_boundary_overconfidence": "d_boundary_oc",
    }
    stored["x_m"] = stored["x"].map(NAME_MAP)
    stored["y_m"] = stored["y"].map(NAME_MAP)
    stored["contrast_m"] = stored["scope"].str.replace("within_case_", "", regex=False)
    assert stored["x_m"].notna().all() and stored["y_m"].notna().all()
    chk = res.merge(
        stored.rename(columns={"spearman_rho": "rho_stored", "n": "n_stored"}),
        left_on=["contrast", "segmentation_change", "downstream_change"],
        right_on=["contrast_m", "x_m", "y_m"], how="left")
    assert chk["rho_stored"].notna().all(), "some frozen rows not matched"
    d_rho = np.abs(chk["rho"] - chk["rho_stored"]).max()
    d_n = int(np.abs(chk["n_paired"] - chk["n_stored"]).max())
    print(f"\n[verify] max |rho - rho_stored| = {d_rho:.3e}   max |N - N_stored| = {d_n}")
    assert d_rho < 1e-9 and d_n == 0, "REPRODUCTION MISMATCH vs frozen artifact"

    res.to_csv(OUT / "reviewer2_r1_spearman_18.csv", index=False)
    print(f"[write] reviewer2_r1_spearman_18.csv  rows={len(res)}")

    # ---- LaTeX table -------------------------------------------------------
    tex = []
    tex.append(r"\begin{table*}[t]")
    tex.append(r"\centering")
    tex.append(r"\small")
    tex.append(r"\caption{Case-level associations between segmentation changes and "
               r"downstream evidence/report changes ($N=798$ paired cases per contrast; "
               r"fully automatic cohort). Values are two-sided Spearman $\rho$ with a "
               r"95\% dataset-stratified case-level bootstrap CI "
               r"($1{,}000$ resamples), the raw $p$-value, and the "
               r"Benjamini--Hochberg adjusted $q$-value from a single "
               r"false-discovery-rate correction applied jointly across the full "
               r"prespecified family of $18$ tests (Benjamini--Hochberg controls the "
               r"false discovery rate, not the family-wise error rate).}")
    tex.append(r"\label{tab:r2_case_level_associations}")
    tex.append(r"\begin{tabular}{lllrrrrr}")
    tex.append(r"\toprule")
    tex.append(r"Contrast & Segmentation change & Report/evidence change & $N$ & "
               r"Spearman $\rho$ & 95\% CI & Raw $p$ & BH $q$ \\")
    tex.append(r"\midrule")
    prev = None
    for r in res.itertuples(index=False):
        lab = CONTRAST_PRETTY[r.target]
        if prev is not None and prev != r.target:
            tex.append(r"\addlinespace")
        prev = r.target
        tex.append(
            f"{lab} & {PRETTY[r.segmentation_change]} & {PRETTY[r.downstream_change]} & "
            f"{r.n_paired} & {fmt(r.rho)} & [{fmt(r.ci_low)}, {fmt(r.ci_high)}] & "
            f"{fmt_p(r.p_raw)} & {fmt_p(r.q_bh)} \\\\")
    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(r"\vspace{2pt}")
    tex.append(r"\parbox{\textwidth}{\footnotesize\textit{Note.} Higher Dice, Boundary "
               r"Dice and symbolic-evidence match indicate better performance, whereas "
               r"higher HD95, raw violation count and boundary overconfidence indicate "
               r"worse performance; all change scores are reported with their natural "
               r"sign and are \emph{not} sign-flipped. $N$ is the number of paired cases "
               r"with finite values for both variables. No association in this table was "
               r"undefined; where a variable is constant, $\rho$ is reported as NA rather "
               r"than substituted by 0.}")
    tex.append(r"\end{table*}")
    (OUT / "reviewer2_r1_spearman_18.tex").write_text("\n".join(tex) + "\n",
                                                       encoding="utf-8")
    print("[write] reviewer2_r1_spearman_18.tex")

    # ---- auxiliary NON-FROZEN diagnostic: 2 contrasts x 9 metric pairs ------
    aux_pairs = [(x, y) for x in ("d_dice", "d_boundary_dice", "d_hd95")
                 for y in ("d_symbolic_match", "d_raw_violations",
                           "d_boundary_oc")]
    aux = []
    for t in ("abloss", "both"):
        rows = drows_by_target[t]
        for x, y in aux_pairs:
            xv = np.asarray([r[x] for r in rows], float)
            yv = np.asarray([r[y] for r in rows], float)
            ok = np.isfinite(xv) & np.isfinite(yv)
            xv, yv = xv[ok], yv[ok]
            if len(xv) >= 3 and np.any(xv != xv[0]) and np.any(yv != yv[0]):
                rho, p = spearmanr(xv, yv)
            else:
                rho, p = float("nan"), float("nan")
            aux.append({"contrast": f"{t}_minus_baseline", "target": t,
                        "segmentation_change": x, "downstream_change": y,
                        "n_paired": int(len(xv)), "rho": float(rho),
                        "p_raw": float(p),
                        "WARNING": "NON-FROZEN DEFINITION - diagnostic only, "
                                   "do not report without explicit confirmation"})
    aux = pd.DataFrame(aux)
    aux["q_bh_within_this_aux_set"] = bh(aux["p_raw"].to_numpy())
    aux.to_csv(OUT / "NON_FROZEN_alt_2contrast_x_9_metric_DIAGNOSTIC.csv", index=False)
    print("[write] NON_FROZEN_alt_2contrast_x_9_metric_DIAGNOSTIC.csv")

    # ---- machine-readable summary -----------------------------------------
    summary = {
        "frozen_definition": "3 settings x 6 metric pairs = 18 prespecified tests",
        "source_of_definition": "stage6_seig_mllm_end2end.py::aggregate + README.md",
        "cohort": "fully automatic 798-case test cohort",
        "datasets": {d: int((clr[clr.variant_key == 'baseline'].dataset == d).sum())
                     for d in DATASETS},
        "paired_n_per_contrast": {t: int(sum(1 for vv in by_case.values()
                                             if "baseline" in vv and t in vv))
                                  for t in TARGETS},
        "bootstrap": {"n_resamples": N_BOOT, "stratified_by": "dataset",
                      "seed": SEED, "ci": "percentile 2.5/97.5"},
        "fdr": "Benjamini-Hochberg, joint over all 18 raw p-values",
        "n_fdr_significant_q_lt_0.05": int(res["fdr_significant"].sum()),
        "n_ci_excludes_zero": int(res["ci_excludes_zero"].sum()),
        "rho_max_abs_diff_vs_frozen_artifact": float(d_rho),
        "note_pq_not_persisted_in_frozen_csv": True,
    }
    (OUT / "reviewer2_r1_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 100)
    print(res[["target", "segmentation_change", "downstream_change", "n_paired",
               "rho", "ci_low", "ci_high", "p_raw", "q_bh", "fdr_significant"]]
          .to_string(index=False))
    print("=" * 100)
    print("FDR-significant (q<0.05):", int(res["fdr_significant"].sum()), "/ 18")
    print(json.dumps(summary, indent=2))
    print("\nPART A DONE")


if __name__ == "__main__":
    main()
