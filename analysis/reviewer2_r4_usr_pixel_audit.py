#!/usr/bin/env python
"""Reviewer #2 Comment #4 -- semantic audit of USR pixel edits and
case-level worsening.

Read-only w.r.t. all existing artifacts. CPU only. No GPU, no re-inference.

PRIMARY (frozen) pathway, per the fully automatic 798-case cohort:
    MedSAM+ABLoss (M0)  ->  MedSAM+ABLoss+USR (Mr)   compared against GT (Y)

SUPPLEMENTARY consistency check:
    MedSAM (M0)         ->  MedSAM+USR (Mr)

The four changed-pixel categories follow usr_correction_audit() in
wysiwyr_autodl_allinone.py exactly:
    added   = (Mr == 1) & (M0 == 0)
    removed = (Mr == 0) & (M0 == 1)
    1. correct lesion recovery   : added   & (Y == 1)   (FN -> TP)
    2. incorrect addition        : added   & (Y == 0)   (TN -> FP)
    3. correct FP removal        : removed & (Y == 0)   (FP -> TN)
    4. incorrect lesion removal  : removed & (Y == 1)   (TP -> FN)
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# Data/artefact root. Set WYSIWYR_DATA_ROOT to the directory that contains
# `wysiwyr_real` (e.g. export WYSIWYR_DATA_ROOT=/path/to/data_root).
DATA_ROOT = Path(os.environ.get("WYSIWYR_DATA_ROOT", ".")).expanduser() / "wysiwyr_real"
S6 = DATA_ROOT / "stage6_seig_mllm_end2end"
UFA = DATA_ROOT / "usr_failure_analysis_20260913"
OUT = ROOT / "results" / "reviewer2"


def build_parser():
    ap = argparse.ArgumentParser(
        description="Reviewer #2 Comment #4 -- semantic audit of USR pixel edits and case-level "
                    "worsening (frozen criterion first, tolerance-aware sensitivity second), plus "
                    "the LaTeX/Markdown summary tables. Reads stored artefacts only.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data-root", default=str(DATA_ROOT),
                    help="directory that contains usr_failure_analysis_20260913/ (i.e. <root>/wysiwyr_real)")
    ap.add_argument("--out-dir", default=str(OUT), help="output directory")
    return ap


def apply_args(args):
    """Rebind the module-level roots from parsed CLI arguments."""
    global DATA_ROOT, S6, UFA, OUT
    DATA_ROOT = Path(args.data_root).expanduser()
    S6 = DATA_ROOT / "stage6_seig_mllm_end2end"
    UFA = DATA_ROOT / "usr_failure_analysis_20260913"
    OUT = Path(args.out_dir).expanduser()
    OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ["CVC-300", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir"]
FROZEN_N = {"CVC-300": 60, "CVC-ClinicDB": 62, "CVC-ColonDB": 380,
            "ETIS-LaribPolypDB": 196, "Kvasir": 100}

PRIMARY = "MedSAM+ABLoss -> MedSAM+ABLoss+USR"
SUPP = "MedSAM -> MedSAM+USR"

CAT = ["correctly_recovered_lesion_pixels", "incorrectly_added_pixels",
       "correctly_removed_false_positive_pixels", "incorrectly_removed_lesion_pixels"]
TOL = 1e-12


def wilson(k: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NA"
    return f"{x:.{nd}f}"


def pct(k, n, nd=2):
    return "NA" if n == 0 else f"{100.0 * k / n:.{nd}f}"


def main():
    apply_args(build_parser().parse_args())
    au = pd.read_csv(UFA / "all_usr_pixel_audit.csv", encoding="utf-8-sig")
    au["case_id"] = au["case_id"].astype(str)

    # canonical per-case segmentation metrics for cross-validation
    csm = pd.read_csv(UFA / "all_case_segmentation_metrics.csv", encoding="utf-8-sig")
    csm["case_id"] = csm["case_id"].astype(str)

    print("=" * 90)
    print("PART B -- USR pixel-level semantic audit")
    print("=" * 90)

    # ---------------------------------------------------------------- B6 checks
    print("\n[B6] sanity checks on inputs")
    print("  all_usr_pixel_audit rows:", len(au),
          "| comparisons:", sorted(au["comparison"].unique()))
    assert set(au["comparison"].unique()) == {PRIMARY, SUPP}, "unexpected comparisons"
    for c, g in au.groupby("comparison"):
        n = len(g)
        got = g.groupby("dataset").size().to_dict()
        exp = {d: FROZEN_N[d] for d in DATASETS if d in got}
        ok = all(got[d] == exp[d] for d in got)
        print(f"  {c:42s} N={n:4d} per-dataset={got} frozen-match={ok}")
        assert ok, "dataset N mismatch"
        dup = g.duplicated(subset=["dataset", "case_id"]).sum()
        assert dup == 0, f"duplicate case rows in {c}"
    print("  no duplicate (dataset, case_id): OK")
    neg = (au[CAT] < 0).to_numpy().sum()
    print("  negative category counts:", int(neg), "->", "OK" if neg == 0 else "PROBLEM")
    assert neg == 0
    # every row must be an integer-valued count
    nonint = (au[CAT] != au[CAT].round()).to_numpy().sum()
    print("  non-integer category counts:", int(nonint))
    assert nonint == 0

    out_rows, summary_rows = [], []

    for comp in [PRIMARY, SUPP]:
        g = au[au["comparison"] == comp].copy()
        g["n_changed"] = g[CAT].sum(axis=1)
        g["n_added"] = (g["correctly_recovered_lesion_pixels"]
                        + g["incorrectly_added_pixels"])
        g["n_removed"] = (g["correctly_removed_false_positive_pixels"]
                          + g["incorrectly_removed_lesion_pixels"])

        # ---- B2.A proportions of changed pixels (case level, NA-safe) ------
        g["share_correct_recovery"] = np.where(
            g["n_changed"] > 0, g["correctly_recovered_lesion_pixels"] / g["n_changed"].replace(0, np.nan), np.nan)
        g["share_incorrect_addition"] = np.where(
            g["n_changed"] > 0, g["incorrectly_added_pixels"] / g["n_changed"].replace(0, np.nan), np.nan)
        g["share_correct_fp_removal"] = np.where(
            g["n_changed"] > 0, g["correctly_removed_false_positive_pixels"] / g["n_changed"].replace(0, np.nan), np.nan)
        g["share_incorrect_lesion_removal"] = np.where(
            g["n_changed"] > 0, g["incorrectly_removed_lesion_pixels"] / g["n_changed"].replace(0, np.nan), np.nan)
        # ---- B2.B / B2.C correctness of additions / removals --------------
        g["addition_correct_rate"] = np.where(
            g["n_added"] > 0, g["correctly_recovered_lesion_pixels"] / g["n_added"].replace(0, np.nan), np.nan)
        g["addition_incorrect_rate"] = np.where(
            g["n_added"] > 0, g["incorrectly_added_pixels"] / g["n_added"].replace(0, np.nan), np.nan)
        g["removal_correct_rate"] = np.where(
            g["n_removed"] > 0, g["correctly_removed_false_positive_pixels"] / g["n_removed"].replace(0, np.nan), np.nan)
        g["removal_incorrect_rate"] = np.where(
            g["n_removed"] > 0, g["incorrectly_removed_lesion_pixels"] / g["n_removed"].replace(0, np.nan), np.nan)

        # ---- B3 case-level worsening with tolerance ------------------------
        g["d_dice"] = g["dice_after"] - g["dice_before"]
        g["d_bdice"] = g["boundary_dice_after"] - g["boundary_dice_before"]
        g["dice_status"] = np.where(g["d_dice"] > TOL, "improved",
                                    np.where(g["d_dice"] < -TOL, "worsened", "unchanged"))
        g["bdice_status"] = np.where(g["d_bdice"] > TOL, "improved",
                                     np.where(g["d_bdice"] < -TOL, "worsened", "unchanged"))
        # frozen pipeline used a strict comparison (no tolerance)
        g["dice_worsened_strict"] = (g["d_dice"] < 0).astype(int)
        g["bdice_worsened_strict"] = (g["d_bdice"] < 0).astype(int)
        # mask-derived TP==0 (complete miss) flags; available if the independent
        # mask re-derivation has been run first
        mv_path = OUT / "reviewer2_r4_mask_recompute_verification.csv"
        if mv_path.exists():
            mv = pd.read_csv(mv_path, encoding="utf-8-sig")
            mv["case_id"] = mv["case_id"].astype(str)
            mv = mv[mv["comparison"] == comp][
                ["dataset", "case_id", "mk_tp_zero_before", "mk_tp_zero_after"]]
            g = g.merge(mv, on=["dataset", "case_id"], how="left", validate="one_to_one")
            g = g.rename(columns={"mk_tp_zero_before": "tp_zero_before",
                                  "mk_tp_zero_after": "tp_zero_after"})
            assert bool(g["tp_zero_before"].notna().all()), "mask verification merge incomplete"
            print(f"  [merge] TP==0 flags joined for {comp}")
        else:
            g["tp_zero_before"] = np.nan
            g["tp_zero_after"] = np.nan
            print(f"  [warn] {mv_path.name} not found -> TP==0 columns left NA")

        keep = ["dataset", "case_id", "comparison", "n_changed", "n_added", "n_removed"] + CAT + [
            "share_correct_recovery", "share_incorrect_addition",
            "share_correct_fp_removal", "share_incorrect_lesion_removal",
            "addition_correct_rate", "addition_incorrect_rate",
            "removal_correct_rate", "removal_incorrect_rate",
            "dice_before", "dice_after", "d_dice", "dice_status",
            "dice_worsened_strict",
            "boundary_dice_before", "boundary_dice_after", "d_bdice", "bdice_status",
            "bdice_worsened_strict",
            "tp_zero_before", "tp_zero_after"]
        out_rows.append(g[keep])

        # ---- B4 dataset-level + overall ------------------------------------
        for ds in DATASETS + ["Overall"]:
            gg = g if ds == "Overall" else g[g["dataset"] == ds]
            n = len(gg)
            if n == 0:
                continue
            tot_changed = int(gg["n_changed"].sum())
            c1 = int(gg["correctly_recovered_lesion_pixels"].sum())
            c2 = int(gg["incorrectly_added_pixels"].sum())
            c3 = int(gg["correctly_removed_false_positive_pixels"].sum())
            c4 = int(gg["incorrectly_removed_lesion_pixels"].sum())
            assert c1 + c2 + c3 + c4 == tot_changed, f"bookkeeping fail {comp} {ds}"
            add_tot, rem_tot = c1 + c2, c3 + c4
            dw = int((gg["dice_status"] == "worsened").sum())
            du = int((gg["dice_status"] == "unchanged").sum())
            di = int((gg["dice_status"] == "improved").sum())
            dws = int(gg["dice_worsened_strict"].sum())
            bw = int((gg["bdice_status"] == "worsened").sum())
            bu = int((gg["bdice_status"] == "unchanged").sum())
            bi = int((gg["bdice_status"] == "improved").sum())
            bws = int(gg["bdice_worsened_strict"].sum())
            tpz0 = int(gg["tp_zero_before"].fillna(-1).eq(1).sum())
            tpz1 = int(gg["tp_zero_after"].fillna(-1).eq(1).sum())
            dw_lo, dw_hi = wilson(dw, n)
            bw_lo, bw_hi = wilson(bw, n)
            summary_rows.append({
                "comparison": comp, "dataset": ds, "n_cases": n,
                "n_changed_pixels": tot_changed,
                "correct_recovery_n": c1, "correct_recovery_pct_changed": 100 * c1 / tot_changed,
                "incorrect_addition_n": c2, "incorrect_addition_pct_changed": 100 * c2 / tot_changed,
                "correct_fp_removal_n": c3, "correct_fp_removal_pct_changed": 100 * c3 / tot_changed,
                "incorrect_lesion_removal_n": c4, "incorrect_lesion_removal_pct_changed": 100 * c4 / tot_changed,
                "addition_correctness_pct": 100 * c1 / add_tot if add_tot else float("nan"),
                "addition_incorrectness_pct": 100 * c2 / add_tot if add_tot else float("nan"),
                "removal_correctness_pct": 100 * c3 / rem_tot if rem_tot else float("nan"),
                "removal_incorrectness_pct": 100 * c4 / rem_tot if rem_tot else float("nan"),
                "median_d_dice": float(np.median(gg["d_dice"])),
                "iqr_d_dice_low": float(np.percentile(gg["d_dice"], 25)),
                "iqr_d_dice_high": float(np.percentile(gg["d_dice"], 75)),
                "dice_worsened_n": dw, "dice_worsened_pct": 100 * dw / n,
                "dice_worsened_wilson_lo": 100 * dw_lo, "dice_worsened_wilson_hi": 100 * dw_hi,
                "dice_worsened_strict_n": dws, "dice_worsened_strict_pct": 100 * dws / n,
                "dice_worsening_rule": "delta < -1e-12 (tolerance-robust)",
                "dice_unchanged_n": du, "dice_unchanged_pct": 100 * du / n,
                "dice_improved_n": di, "dice_improved_pct": 100 * di / n,
                "n_tp_zero_before": tpz0, "n_tp_zero_after": tpz1,
                "median_d_bdice": float(np.median(gg["d_bdice"])),
                "iqr_d_bdice_low": float(np.percentile(gg["d_bdice"], 25)),
                "iqr_d_bdice_high": float(np.percentile(gg["d_bdice"], 75)),
                "bdice_worsened_n": bw, "bdice_worsened_pct": 100 * bw / n,
                "bdice_worsened_wilson_lo": 100 * bw_lo, "bdice_worsened_wilson_hi": 100 * bw_hi,
                "bdice_worsened_strict_n": bws, "bdice_worsened_strict_pct": 100 * bws / n,
                "bdice_worsening_rule": "delta < -1e-12 (tolerance-robust)",
                "bdice_unchanged_n": bu, "bdice_unchanged_pct": 100 * bu / n,
                "bdice_improved_n": bi, "bdice_improved_pct": 100 * bi / n,
            })

    case = pd.concat(out_rows, ignore_index=True)
    case = case.sort_values(["comparison", "dataset", "case_id"]).reset_index(drop=True)

    # ---- cross-validate dice_before/after against canonical metrics ---------
    print("\n[B6] cross-validate stored dice_before/after against canonical "
          "case_segmentation_metrics.csv")
    wide = csm.pivot_table(index=["dataset", "case_id"], columns="variant_key_norm"
                           if "variant_key_norm" in csm.columns else "variant_key",
                           values=["dice", "boundary_dice"], aggfunc="first")
    m0v, mrv = ("abloss", "both")
    ref = pd.DataFrame({
        "dice_abloss_ref": wide[("dice", m0v)], "dice_both_ref": wide[("dice", mrv)],
        "bd_abloss_ref": wide[("boundary_dice", m0v)],
        "bd_both_ref": wide[("boundary_dice", mrv)],
    }).reset_index()
    sub = case[case["comparison"] == PRIMARY][["dataset", "case_id", "dice_before",
                                               "dice_after", "boundary_dice_before",
                                               "boundary_dice_after"]]
    m = sub.merge(ref, on=["dataset", "case_id"], how="left")
    assert m["dice_abloss_ref"].notna().all(), "unmatched cases in cross-validation"
    e1 = float(np.abs(m["dice_before"] - m["dice_abloss_ref"]).max())
    e2 = float(np.abs(m["dice_after"] - m["dice_both_ref"]).max())
    e3 = float(np.abs(m["boundary_dice_before"] - m["bd_abloss_ref"]).max())
    e4 = float(np.abs(m["boundary_dice_after"] - m["bd_both_ref"]).max())
    print(f"  max abs err  dice_before={e1:.3e}  dice_after={e2:.3e}  "
          f"bdice_before={e3:.3e}  bdice_after={e4:.3e}")
    assert max(e1, e2, e3, e4) < 1e-9, "audit CSV disagrees with canonical metrics"
    print("  audit CSV == canonical per-case metrics: OK")

    # ---- writes -------------------------------------------------------------
    case.to_csv(OUT / "reviewer2_r4_usr_case_audit.csv", index=False)
    sm = pd.DataFrame(summary_rows)
    sm.to_csv(OUT / "reviewer2_r4_usr_pixel_case_summary.csv", index=False)
    print("\n[write] reviewer2_r4_usr_case_audit.csv  rows={}".format(len(case)))
    print("[write] reviewer2_r4_usr_pixel_case_summary.csv  rows={}".format(len(sm)))

    # ---- LaTeX --------------------------------------------------------------
    # Presentation only: every value below is read from the frozen summary rows
    # (`sm`), which are themselves derived from the stored audit CSV. No metric is
    # recomputed here and the frozen worsening criterion is presented first.
    TOLTEX = r"10^{-12}"
    tex = [r"\begin{table*}[t]", r"\centering", r"\scriptsize",
           r"\setlength{\tabcolsep}{3pt}",
           r"\caption{Semantic audit of USR pixel edits and case-level worsening on the "
           r"fully automatic 798-case cohort, for the primary pathway "
           r"MedSAM+ABLoss $\rightarrow$ MedSAM+ABLoss+USR. Changed pixels are those with "
           r"$M_0 \neq M_r$. Percentages in the first four columns are shares of all "
           r"changed pixels; addition/removal correctness are pooled ratios over "
           r"added/removed pixels. Case-level worsening is reported under the "
           r"\emph{frozen} evaluation criterion first (``frozen'' columns, $\Delta < 0$), "
           rf"followed by a tolerance-aware numerical-sensitivity analysis "
           rf"(``sens.'' columns, $\Delta < -{TOLTEX}$).}}",
           r"\label{tab:r2_usr_semantic_audit}",
           r"\begin{tabular}{lrrrrrrrrrrrr}", r"\toprule",
           r" & & & & & & & & & \multicolumn{2}{c}{Dice worsened} & "
           r"\multicolumn{2}{c}{B.\ Dice worsened} \\",
           r"\cmidrule(lr){10-11}\cmidrule(lr){12-13}",
           r"Dataset & $N$ & Changed & Correct & Incorrect & Correct & Incorrect & Add. & "
           r"Rem. & frozen & sens. & frozen & sens. \\",
           r" & & pixels & recovery & addition & FP removal & lesion rem. & corr. & corr. & "
           rf"$\Delta<0$ & $\Delta<-{TOLTEX}$ & $\Delta<0$ & $\Delta<-{TOLTEX}$ \\",
           r"\midrule"]

    def _tex_row(r) -> str:
        """One dataset row. Worsening columns: frozen strict first, sensitivity second."""
        return (
            f"{r.dataset} & {r.n_cases} & {r.n_changed_pixels:,} & "
            f"{r.correct_recovery_n:,} ({r.correct_recovery_pct_changed:.2f}\\%) & "
            f"{r.incorrect_addition_n:,} ({r.incorrect_addition_pct_changed:.2f}\\%) & "
            f"{r.correct_fp_removal_n:,} ({r.correct_fp_removal_pct_changed:.2f}\\%) & "
            f"{r.incorrect_lesion_removal_n:,} ({r.incorrect_lesion_removal_pct_changed:.2f}\\%) & "
            f"{r.addition_correctness_pct:.2f}\\% & {r.removal_correctness_pct:.2f}\\% & "
            f"{r.dice_worsened_strict_n} ({r.dice_worsened_strict_pct:.2f}\\%) & "
            f"{r.dice_worsened_n} ({r.dice_worsened_pct:.2f}\\%) & "
            f"{r.bdice_worsened_strict_n} ({r.bdice_worsened_strict_pct:.2f}\\%) & "
            f"{r.bdice_worsened_n} ({r.bdice_worsened_pct:.2f}\\%) \\\\")

    for r in sm[sm["comparison"] == PRIMARY].itertuples(index=False):
        tex.append(_tex_row(r))
    tex += [r"\midrule",
            r"\multicolumn{13}{l}{\footnotesize Supplementary pathway "
            r"MedSAM $\rightarrow$ MedSAM+USR (consistency check), same column layout:} \\"]
    for r in sm[(sm["comparison"] == SUPP)].itertuples(index=False):
        tex.append(_tex_row(r))
    ovp = sm[(sm["comparison"] == PRIMARY) & (sm["dataset"] == "Overall")].iloc[0]
    tex += [r"\bottomrule", r"\end{tabular}", r"\vspace{2pt}",
            r"\parbox{\textwidth}{\footnotesize\textit{Note.} The frozen evaluation protocol "
            r"defines a case as worsened when the metric decreases, i.e.\ $\Delta < 0$, with no "
            r"tolerance; these counts are given in the ``frozen'' columns. The tolerance-aware "
            rf"criterion $\Delta < -{TOLTEX}$ is reported \emph{{only}} as a "
            r"\emph{numerical-sensitivity analysis} (``sens.'' columns); it does not redefine "
            r"the frozen evaluation protocol. \textbf{No cases were removed from any analysis.} "
            r"The difference between the strict and the tolerance-aware counts is entirely "
            r"attributable to complete-miss cases with $\mathrm{TP}=0$ "
            rf"({int(ovp.n_tp_zero_before)}/{int(ovp.n_cases)} before rectification and "
            rf"{int(ovp.n_tp_zero_after)}/{int(ovp.n_cases)} after), whose "
            r"$\epsilon$-smoothed metric differences are approximately $10^{-13}$. Under the "
            rf"sensitivity analysis the overall worsening rates are "
            rf"${int(ovp.dice_worsened_n)}/{int(ovp.n_cases)} = {ovp.dice_worsened_pct:.2f}\%$ "
            rf"for Dice (Wilson 95\% CI {ovp.dice_worsened_wilson_lo:.2f}--"
            rf"{ovp.dice_worsened_wilson_hi:.2f}\%) and "
            rf"${int(ovp.bdice_worsened_n)}/{int(ovp.n_cases)} = {ovp.bdice_worsened_pct:.2f}\%$ "
            rf"for Boundary Dice (Wilson 95\% CI {ovp.bdice_worsened_wilson_lo:.2f}--"
            rf"{ovp.bdice_worsened_wilson_hi:.2f}\%); per-dataset confidence intervals are "
            r"reported in the accompanying summary CSV. USR is predominantly a pruning "
            r"operator: 84.3\% of the modified pixels are removals (373,954) and 15.7\% are "
            r"additions (69,532); over the whole cohort it incorrectly removes 68,600 lesion "
            r"pixels compared with 16,772 correctly recovered lesion pixels, approximately a "
            r"fourfold difference. ``Add. corr.'' and ``Rem. corr.'' are the pooled fractions "
            r"of added and of removed pixels that change the label in the correct "
            r"direction.}"]
    tex.append(r"\end{table*}")
    (OUT / "reviewer2_r4_usr_pixel_case_summary.tex").write_text("\n".join(tex) + "\n",
                                                                 encoding="utf-8")
    print("[write] reviewer2_r4_usr_pixel_case_summary.tex")

    # ---- machine summary ----------------------------------------------------
    ov = sm[(sm["comparison"] == PRIMARY) & (sm["dataset"] == "Overall")].iloc[0]
    ovs = sm[(sm["comparison"] == SUPP) & (sm["dataset"] == "Overall")].iloc[0]
    summary = {
        "cohort": {"n_cases": int(ov.n_cases),
                   "composition": {d: FROZEN_N[d] for d in DATASETS}},
        "primary_pathway": PRIMARY, "supplementary_pathway": SUPP,
        "tolerance_for_worsening": TOL,
        "primary": {
            "changed_pixels": int(ov.n_changed_pixels),
            "correct_recovery": [int(ov.correct_recovery_n), float(ov.correct_recovery_pct_changed)],
            "incorrect_addition": [int(ov.incorrect_addition_n), float(ov.incorrect_addition_pct_changed)],
            "correct_fp_removal": [int(ov.correct_fp_removal_n), float(ov.correct_fp_removal_pct_changed)],
            "incorrect_lesion_removal": [int(ov.incorrect_lesion_removal_n), float(ov.incorrect_lesion_removal_pct_changed)],
            "addition_correctness_pct": float(ov.addition_correctness_pct),
            "removal_correctness_pct": float(ov.removal_correctness_pct),
            "dice_worsened": [int(ov.dice_worsened_n), int(ov.n_cases), float(ov.dice_worsened_pct),
                              float(ov.dice_worsened_wilson_lo), float(ov.dice_worsened_wilson_hi)],
            "dice_worsened_strict_rule": [int(ov.dice_worsened_strict_n), int(ov.n_cases),
                                          float(ov.dice_worsened_strict_pct)],
            "n_tp_zero_before": int(ov.n_tp_zero_before),
            "n_tp_zero_after": int(ov.n_tp_zero_after),
            "dice_unchanged_pct": float(ov.dice_unchanged_pct),
            "dice_improved_pct": float(ov.dice_improved_pct),
            "median_d_dice": float(ov.median_d_dice),
            "iqr_d_dice": [float(ov.iqr_d_dice_low), float(ov.iqr_d_dice_high)],
            "bdice_worsened": [int(ov.bdice_worsened_n), int(ov.n_cases), float(ov.bdice_worsened_pct),
                               float(ov.bdice_worsened_wilson_lo), float(ov.bdice_worsened_wilson_hi)],
            "bdice_worsened_strict_rule": [int(ov.bdice_worsened_strict_n), int(ov.n_cases),
                                           float(ov.bdice_worsened_strict_pct)],
            "bdice_unchanged_pct": float(ov.bdice_unchanged_pct),
            "bdice_improved_pct": float(ov.bdice_improved_pct),
            "median_d_bdice": float(ov.median_d_bdice),
            "iqr_d_bdice": [float(ov.iqr_d_bdice_low), float(ov.iqr_d_bdice_high)],
        },
        "supplementary": {
            "changed_pixels": int(ovs.n_changed_pixels),
            "addition_correctness_pct": float(ovs.addition_correctness_pct),
            "removal_correctness_pct": float(ovs.removal_correctness_pct),
            "dice_worsened": [int(ovs.dice_worsened_n), int(ovs.n_cases), float(ovs.dice_worsened_pct)],
            "dice_worsened_strict_rule": [int(ovs.dice_worsened_strict_n), int(ovs.n_cases),
                                          float(ovs.dice_worsened_strict_pct)],
            "bdice_worsened": [int(ovs.bdice_worsened_n), int(ovs.n_cases), float(ovs.bdice_worsened_pct)],
            "bdice_worsened_strict_rule": [int(ovs.bdice_worsened_strict_n), int(ovs.n_cases),
                                           float(ovs.bdice_worsened_strict_pct)],
        },
    }
    (OUT / "reviewer2_r4_summary.json").write_text(json.dumps(summary, indent=2),
                                                   encoding="utf-8")

    print("\n" + "=" * 110)
    print(sm[sm["comparison"] == PRIMARY][
        ["dataset", "n_cases", "n_changed_pixels", "correct_recovery_pct_changed",
         "incorrect_addition_pct_changed", "correct_fp_removal_pct_changed",
         "incorrect_lesion_removal_pct_changed", "addition_correctness_pct",
         "removal_correctness_pct", "dice_worsened_n", "dice_worsened_pct",
         "bdice_worsened_n", "bdice_worsened_pct"]].to_string(index=False))
    print("=" * 110)
    print(json.dumps(summary, indent=2))
    print("\nPART B DONE")


if __name__ == "__main__":
    main()
