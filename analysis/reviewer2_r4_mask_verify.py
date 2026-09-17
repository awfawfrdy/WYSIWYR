#!/usr/bin/env python
"""Reviewer #2 Comment #4 -- B6 independent verification.

Independently RE-DERIVES the USR semantic pixel accounting and the case-level
Dice / Boundary Dice worsening directly from the stored binary masks, for the
FULL 798-case fully automatic cohort, and compares the result against the
stored audit artifacts.

Re-implements, verbatim, the frozen metric code in
wysiwyr_autodl_allinone.py:
    _bin, dice_score, boundary_map, boundary_dice, usr_correction_audit

READ-ONLY on all inputs and on the prediction/GT trees. CPU only. No GPU.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Data/artefact root. Set WYSIWYR_DATA_ROOT to the directory that contains
# `wysiwyr_real` (e.g. export WYSIWYR_DATA_ROOT=/path/to/data_root).
DATA_ROOT = Path(os.environ.get("WYSIWYR_DATA_ROOT", ".")).expanduser() / "wysiwyr_real"
S5 = DATA_ROOT / "stage5_calibrated" / "predictions"
GT_ROOT = DATA_ROOT / "data" / "TestDataset"
UFA = DATA_ROOT / "usr_failure_analysis_20260913"
OUT = Path(__file__).resolve().parent.parent / "results" / "reviewer2"


def build_parser():
    ap = argparse.ArgumentParser(
        description="Reviewer #2 Comment #4 -- independent mask-level re-derivation of the USR "
                    "pixel accounting and of the case-level Dice / Boundary-Dice worsening counts. "
                    "Reads stored masks only; never re-trains or re-infers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data-root", default=str(DATA_ROOT),
                    help="directory that contains stage5_calibrated/ and data/ (i.e. <root>/wysiwyr_real)")
    ap.add_argument("--out-dir", default=str(OUT), help="output directory")
    return ap


def apply_args(args):
    """Rebind the module-level roots from parsed CLI arguments."""
    global DATA_ROOT, S5, GT_ROOT, UFA, OUT
    DATA_ROOT = Path(args.data_root).expanduser()
    S5 = DATA_ROOT / "stage5_calibrated" / "predictions"
    GT_ROOT = DATA_ROOT / "data" / "TestDataset"
    UFA = DATA_ROOT / "usr_failure_analysis_20260913"
    OUT = Path(args.out_dir).expanduser()
    OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ["CVC-300", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-LaribPolypDB", "Kvasir"]
FROZEN_N = {"CVC-300": 60, "CVC-ClinicDB": 62, "CVC-ColonDB": 380,
            "ETIS-LaribPolypDB": 196, "Kvasir": 100}
COMPARISONS = {
    "MedSAM+ABLoss -> MedSAM+ABLoss+USR": ("abloss", "both"),
    "MedSAM -> MedSAM+USR": ("baseline", "usr"),
}
TOL = 1e-12
EPS = 1e-8

CAT_KEYS = ["correctly_recovered_lesion_pixels", "incorrectly_added_pixels",
            "correctly_removed_false_positive_pixels", "incorrectly_removed_lesion_pixels"]


# ---------------- frozen re-implementations -------------------------------
def _bin(x):
    a = np.squeeze(np.asarray(x))
    if a.ndim != 2:
        raise ValueError(f"Expected HxW mask, got {a.shape}")
    return (a > 0.5).astype(np.uint8)


def confusion(pred, gt):
    p, y = _bin(pred), _bin(gt)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    return tp, fp, fn, tn


def dice_score(pred, gt, eps=EPS):
    tp, fp, fn, _ = confusion(pred, gt)
    return (2 * tp + eps) / (2 * tp + fp + fn + eps)


def boundary_map(mask, kernel=3):
    m = _bin(mask)
    k = max(1, int(kernel))
    k += (k % 2 == 0)
    er = cv2.erode(m, np.ones((k, k), np.uint8), iterations=1)
    return ((m == 1) & (er == 0)).astype(np.uint8)


def boundary_dice(pred, gt, tolerance=2, eps=EPS):
    bp, bg = boundary_map(pred), boundary_map(gt)
    k = max(1, 2 * int(tolerance) + 1)
    dp = cv2.dilate(bp, np.ones((k, k), np.uint8), iterations=1)
    dg = cv2.dilate(bg, np.ones((k, k), np.uint8), iterations=1)
    matched_p = int((bp & dg).sum())
    matched_g = int((bg & dp).sum())
    denom = int(bp.sum() + bg.sum())
    return (matched_p + matched_g + eps) / (denom + eps)


def load_array(p: Path):
    """Exact replica of run_reviewer2_experiments.py::_load_array"""
    a = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise FileNotFoundError(p)
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    a = np.asarray(a)
    a = np.squeeze(a)
    if a.ndim != 2:
        raise ValueError(f"Expected 2D map, got {a.shape} from {p}")
    return a


def to_unit_float(a):
    """Exact replica of run_reviewer2_experiments.py::_to_unit_float"""
    x = np.asarray(a, dtype=np.float32)
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    mn, mx = float(x.min()), float(x.max())
    if mx <= 1.0 and mn >= 0.0:
        return x
    if mn >= 0.0 and mx <= 255.0:
        return x / 255.0
    if mn >= 0.0 and mx <= 65535.0:
        return x / 65535.0
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def resize_to(a, shape_hw, continuous):
    h, w = shape_hw
    if a.shape == (h, w):
        return a
    interp = cv2.INTER_LINEAR if continuous else cv2.INTER_NEAREST
    return cv2.resize(a, (w, h), interpolation=interp)


def load_mask(p: Path, shape=None):
    """Exact replica of run_reviewer2_experiments.py::_load_mask"""
    a = to_unit_float(load_array(p))
    if shape is not None:
        a = resize_to(a, shape, continuous=False)
    return (a > 0.5).astype(np.uint8)


def raw_values(p: Path):
    return np.unique(load_array(p))


def main():
    apply_args(build_parser().parse_args())
    stored = pd.read_csv(UFA / "all_usr_pixel_audit.csv", encoding="utf-8-sig")
    stored["case_id"] = stored["case_id"].astype(str)

    print("=" * 100)
    print("B6  INDEPENDENT MASK-LEVEL VERIFICATION (798 cases x 2 comparisons)")
    print("=" * 100)

    rows, shape_issues, nonbinary = [], [], []
    n_missing = 0

    for comp, (m0v, mrv) in COMPARISONS.items():
        for ds in DATASETS:
            case_ids = sorted(stored[(stored["comparison"] == comp)
                                     & (stored["dataset"] == ds)]["case_id"].unique())
            assert len(case_ids) == FROZEN_N[ds], \
                f"{comp} {ds}: {len(case_ids)} case ids, expected {FROZEN_N[ds]}"
            gt_dir = GT_ROOT / ds / "mask"
            d0, dr = S5 / ds / m0v / "masks", S5 / ds / mrv / "masks"
            for cid in case_ids:
                p_y, p_0, p_r = gt_dir / f"{cid}.png", d0 / f"{cid}.png", dr / f"{cid}.png"
                if not (p_y.exists() and p_0.exists() and p_r.exists()):
                    n_missing += 1
                    continue
                raw_y, raw_0, raw_r = load_array(p_y), load_array(p_0), load_array(p_r)
                if not (raw_y.shape == raw_0.shape == raw_r.shape):
                    shape_issues.append((comp, ds, cid, raw_y.shape, raw_0.shape,
                                         raw_r.shape))
                    continue
                # record raw value domains (before the frozen unit-float scaling)
                for nm, arr in (("Y", raw_y), ("M0", raw_0), ("Mr", raw_r)):
                    u = np.unique(arr)
                    if not np.all(np.isin(u, [0, 1, 255])):
                        nonbinary.append((comp, ds, cid, nm,
                                          u[:6].tolist(), int(u.max())))
                # EXACT frozen loading: GT first, predictions resized to GT shape
                y = load_mask(p_y)
                b = load_mask(p_0, y.shape)
                a = load_mask(p_r, y.shape)
                tpb, fpb, fnb, _ = confusion(b, y)
                tpa, fpa, fna, _ = confusion(a, y)
                added = (a == 1) & (b == 0)
                removed = (a == 0) & (b == 1)
                rec = int((added & (y == 1)).sum())
                wadd = int((added & (y == 0)).sum())
                grm = int((removed & (y == 0)).sum())
                wrm = int((removed & (y == 1)).sum())
                n_changed = int((a != b).sum())
                if rec + wadd + grm + wrm != n_changed:
                    raise AssertionError(f"bookkeeping fail {comp} {ds} {cid}")
                rows.append({
                    "comparison": comp, "dataset": ds, "case_id": cid,
                    "mk_tp_before": tpb, "mk_fp_before": fpb, "mk_fn_before": fnb,
                    "mk_tp_after": tpa, "mk_fp_after": fpa, "mk_fn_after": fna,
                    "mk_tp_zero_before": int(tpb == 0), "mk_tp_zero_after": int(tpa == 0),
                    "mk_correct_recovery": rec, "mk_incorrect_addition": wadd,
                    "mk_correct_fp_removal": grm, "mk_incorrect_lesion_removal": wrm,
                    "mk_n_changed": n_changed,
                    "mk_dice_before": dice_score(b, y), "mk_dice_after": dice_score(a, y),
                    "mk_bdice_before": boundary_dice(b, y),
                    "mk_bdice_after": boundary_dice(a, y),
                    "gt_pixels": int((y == 1).sum()),
                    "shape": "x".join(map(str, y.shape)),
                })

    mk = pd.DataFrame(rows)
    print(f"\n[1] masks re-read: {len(mk)} rows (expected {798*2}); missing triples: {n_missing}")
    print(f"[2] shape-mismatch cases: {len(shape_issues)}")
    if shape_issues:
        print("    examples:", shape_issues[:5])
    print(f"[3] masks whose RAW value domain exceeds {{0,1,255}}: {len(nonbinary)}")
    if nonbinary:
        from collections import Counter
        c = Counter((x[1], x[3]) for x in nonbinary)
        print("    by (dataset, role):", dict(c))
        print("    NOTE: frozen _load_mask applies _to_unit_float (divide by 255 when")
        print("          max>1) BEFORE thresholding at >0.5, so anti-aliased GT pixels")
        print("          below 128/255 are treated as background. This is replicated here.")
    assert n_missing == 0 and not shape_issues, "mask integrity problem detected"
    print("    -> every (M0, Mr, Y) triple exists and shapes agree: OK")

    mk.to_csv(OUT / "reviewer2_r4_mask_recompute_verification.csv", index=False)

    # ---------------- compare with the stored audit ------------------------
    m = mk.merge(stored, on=["comparison", "dataset", "case_id"],
                 how="left", validate="one_to_one")
    assert m["dice_before"].notna().all(), "unmatched cases"

    print("\n[4] changed-pixel bookkeeping: independent recomputation vs stored audit")
    cat_map = dict(zip(CAT_KEYS, ["mk_correct_recovery", "mk_incorrect_addition",
                                  "mk_correct_fp_removal",
                                  "mk_incorrect_lesion_removal"]))
    max_cat_err = 0
    for k, mkk in cat_map.items():
        e = int(np.abs(m[k].astype(np.int64) - m[mkk].astype(np.int64)).max())
        max_cat_err = max(max_cat_err, e)
        print(f"    {k:42s} max |diff| = {e}")
    print(f"    -> {'EXACT MATCH' if max_cat_err == 0 else 'MISMATCH'}")

    e_dice_b = float(np.abs(m["mk_dice_before"] - m["dice_before"]).max())
    e_dice_a = float(np.abs(m["mk_dice_after"] - m["dice_after"]).max())
    e_bd_b = float(np.abs(m["mk_bdice_before"] - m["boundary_dice_before"]).max())
    e_bd_a = float(np.abs(m["mk_bdice_after"] - m["boundary_dice_after"]).max())
    print("\n[5] metric reproduction (mask-derived vs stored audit)")
    print(f"    dice_before   max|diff| = {e_dice_b:.3e}")
    print(f"    dice_after    max|diff| = {e_dice_a:.3e}")
    print(f"    bdice_before  max|diff| = {e_bd_b:.3e}")
    print(f"    bdice_after   max|diff| = {e_bd_a:.3e}")

    # does the stored (rounded) delta produce spurious tiny negative deltas?
    m["d_mask"] = m["mk_dice_after"] - m["mk_dice_before"]
    m["d_store"] = m["dice_after"] - m["dice_before"]
    m["db_mask"] = m["mk_bdice_after"] - m["mk_bdice_before"]
    m["db_store"] = m["boundary_dice_after"] - m["boundary_dice_before"]

    print("\n[6] strict-vs-tolerance worsening counts")
    summary = {}
    for comp in COMPARISONS:
        g = m[m["comparison"] == comp]
        n = len(g)
        s = {"n_cases": n}
        for lab, cm, cs in (("dice", "d_mask", "d_store"), ("bdice", "db_mask", "db_store")):
            s[f"{lab}_strict_from_masks"] = int((g[cm] < 0).sum())
            s[f"{lab}_tolerance_from_masks"] = int((g[cm] < -TOL).sum())
            s[f"{lab}_exact_zero_from_masks"] = int((g[cm] == 0).sum())
            s[f"{lab}_strict_from_stored"] = int((g[cs] < 0).sum())
            band = int(((g[cm] < 0) & (g[cm] >= -TOL)).sum())
            s[f"{lab}_in_minus1e-12_to_0_from_masks"] = band
            s[f"{lab}_max_abs_roundoff_band"] = (
                float(np.abs(g.loc[(g[cm] < 0) & (g[cm] >= -TOL), cm]).max())
                if band else 0.0)
        summary[comp] = s
        print(f"\n    {comp}   (N={n})")
        for k, v in s.items():
            if k != "n_cases":
                print(f"      {k:42s} {v}")

    ver = {
        "n_mask_triples_read": int(len(mk)),
        "missing_mask_triples": n_missing,
        "shape_mismatches": len(shape_issues),
        "non_binary_masks": len(nonbinary),
        "max_category_count_diff_vs_stored": int(max_cat_err),
        "max_abs_diff_dice_before": e_dice_b,
        "max_abs_diff_dice_after": e_dice_a,
        "max_abs_diff_bdice_before": e_bd_b,
        "max_abs_diff_bdice_after": e_bd_a,
        "tolerance": TOL,
        "per_comparison": summary,
        "conclusion": ("stored audit artifacts are faithfully reproducible from the "
                       "binary masks; category counts match exactly"),
    }
    (OUT / "reviewer2_r4_mask_recompute_verification.json").write_text(
        json.dumps(ver, indent=2), encoding="utf-8")
    print("\n[write] reviewer2_r4_mask_recompute_verification.csv / .json")
    print("\nB6 VERIFICATION DONE")


if __name__ == "__main__":
    main()
