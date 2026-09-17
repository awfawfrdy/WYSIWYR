#!/usr/bin/env python3
"""Checker-v3 human-validation scoring.

Scores the frozen Checker v3 against blinded human labels and reports:

  * per-class precision / recall / F1 (4 claim-status classes)
  * macro-F1 over the declared classes
  * overall accuracy
  * macro-F1 bootstrap 95% percentile CI (case-level resampling)
  * raw pre-adjudication agreement between independent annotators
  * Cohen's kappa (chance-corrected agreement), 4-class and binary-violation
  * the full 4x4 confusion matrix

Nothing is hard-coded: every number printed is computed from the input files.

--------------------------------------------------
INPUT SCHEMA
--------------------------------------------------
--human : the blinded sheet returned by the annotators. It may contain, per
          claim (one row per annotation_id):
            annotation_id            (required; joins to --key)
            human_status             (adjudicated / final label)
            human_action, human_rule_id, notes   (optional, not scored)
          Plus OPTIONAL per-annotator columns for the pre-adjudication
          agreement statistic, in any of these forms:
            annotator_1_status, annotator_2_status, ...   (wide format)
            or a long-format file with  annotation_id, annotator_id, human_status
          If only one label exists per claim, pre-adjudication agreement is
          reported as null (it is undefined with a single annotator).

--key   : the checker key file (`checker_v3_human_validation_KEY_DO_NOT_SHOW_ANNOTATOR.csv`)
          with columns annotation_id, checker_v3_status.

Label domain: supported | calibrated | unsupported | prohibited
Binary violation endpoint: anything != "supported" is a violation.

--------------------------------------------------
USAGE
--------------------------------------------------
    python score_human_validation.py --human <annotated_blinded.csv> \
        --key <checker_key.csv> --output <scores.json>

    # code-path self test on synthetic labels (no data needed):
    python score_human_validation.py --smoke-test
"""
from __future__ import annotations

import os

DATA_ROOT = os.environ.get("WYSIWYR_DATA_ROOT", ".")

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ["supported", "calibrated", "unsupported", "prohibited"]


# --------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------
def read(p):
    with open(p, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _norm(x):
    return (x or "").strip().lower()


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def prf(y_true, y_pred, positive):
    """One-vs-rest precision/recall/F1 for `positive`."""
    tp = sum(t == positive and p == positive for t, p in zip(y_true, y_pred))
    fp = sum(t != positive and p == positive for t, p in zip(y_true, y_pred))
    fn = sum(t == positive and p != positive for t, p in zip(y_true, y_pred))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}


def macro_f1(y_true, y_pred, classes=CLASSES):
    """Unweighted mean of per-class F1 over the declared classes."""
    return sum(prf(y_true, y_pred, c)["f1"] for c in classes) / len(classes)


def accuracy(y_true, y_pred):
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0


def confusion_matrix(y_true, y_pred, classes=CLASSES):
    """4x4 confusion matrix; rows = human, cols = checker."""
    idx = {c: i for i, c in enumerate(classes)}
    m = [[0] * len(classes) for _ in classes]
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            m[idx[t]][idx[p]] += 1
    return {"classes": classes, "rows_human_cols_checker": m}


def cohen_kappa(y_true, y_pred):
    """Cohen's kappa for two label sequences (any label domain)."""
    n = len(y_true)
    if n == 0:
        return float("nan")
    labels = sorted(set(y_true) | set(y_pred))
    po = sum(t == p for t, p in zip(y_true, y_pred)) / n
    ct, cp = Counter(y_true), Counter(y_pred)
    pe = sum((ct[l] / n) * (cp[l] / n) for l in labels)
    return (po - pe) / (1 - pe) if (1 - pe) else float("nan")


def percentile(sorted_vals, q):
    """Linear-interpolation percentile on a pre-sorted list (q in [0,100])."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * (q / 100.0)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(sorted_vals[lo])
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo))


def bootstrap_ci(y_true, y_pred, stat_fn, n_boot=2000, seed=2023, alpha=0.05):
    """Case-level (row-level) bootstrap percentile CI of `stat_fn`."""
    n = len(y_true)
    if n == 0:
        return (float("nan"), float("nan"), 0)
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        bt = [y_true[i] for i in idx]
        bp = [y_pred[i] for i in idx]
        if len(set(bt)) < 2 and len(set(bp)) < 2:
            continue  # degenerate replicate
        v = stat_fn(bt, bp)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    vals.sort()
    return (percentile(vals, 100 * alpha / 2), percentile(vals, 100 * (1 - alpha / 2)), len(vals))


def pre_adjudication_agreement(human_rows, annotator_cols):
    """Mean pairwise agreement between independent annotators, per claim.

    Returns (mean pairwise agreement, n_claims_with_>=2 annotations) or
    (None, 0) when fewer than two annotator labels are present anywhere.
    """
    usable = 0
    per_claim = []
    for r in human_rows:
        labs = [_norm(r.get(c, "")) for c in annotator_cols]
        labs = [x for x in labs if x]
        if len(labs) < 2:
            continue
        pairs = tot = 0
        for i in range(len(labs)):
            for j in range(i + 1, len(labs)):
                tot += 1
                pairs += int(labs[i] == labs[j])
        if tot:
            per_claim.append(pairs / tot)
            usable += 1
    if not per_claim:
        return (None, 0)
    return (sum(per_claim) / len(per_claim), usable)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def score(pairs, n_boot, seed):
    """`pairs` = list of (human_row, checker_row)."""
    yt = [_norm(r["human_status"]) for r, _ in pairs]
    yp = [_norm(q["checker_v3_status"]) for _, q in pairs]
    tv = ["violation" if x != "supported" else "supported" for x in yt]
    pv = ["violation" if x != "supported" else "supported" for x in yp]

    per = {c: prf(yt, yp, c) for c in CLASSES}
    macro = {m: sum(per[c][m] for c in CLASSES) / len(CLASSES)
             for m in ("precision", "recall", "f1")}
    acc = accuracy(yt, yp)
    mf1 = macro_f1(yt, yp)
    k4 = cohen_kappa(yt, yp)
    kb = cohen_kappa(tv, pv)
    accb = accuracy(tv, pv)

    mf1_lo, mf1_hi, n_ok = bootstrap_ci(yt, yp, macro_f1, n_boot, seed)
    acc_lo, acc_hi, _ = bootstrap_ci(yt, yp, accuracy, n_boot, seed)
    k4_lo, k4_hi, _ = bootstrap_ci(yt, yp, cohen_kappa, n_boot, seed)

    ann_cols = sorted({k for r, _ in pairs for k in r
                       if k.startswith("annotator") and k.endswith("status")})
    raw_agr, n_multi = pre_adjudication_agreement([r for r, _ in pairs], ann_cols)

    return {
        "n": len(pairs),
        "accuracy": acc,
        "accuracy_ci95": [acc_lo, acc_hi],
        "macro_f1": mf1,
        "macro_f1_ci95": [mf1_lo, mf1_hi],
        "bootstrap": {"n_resamples": n_boot, "seed": seed,
                      "n_finite_replicates": n_ok, "unit": "claim (row-level)"},
        "per_class": per,
        "macro_over_classes": macro,
        "confusion_matrix": confusion_matrix(yt, yp),
        "binary_violation": {
            **prf(tv, pv, "violation"),
            "accuracy": accb,
            "cohen_kappa": kb,
            "n_true_violation": sum(x == "violation" for x in tv),
            "n_pred_violation": sum(x == "violation" for x in pv),
        },
        "cohen_kappa_status4": k4,
        "cohen_kappa_status4_ci95": [k4_lo, k4_hi],
        "raw_pre_adjudication_agreement": raw_agr,
        "raw_pre_adjudication_agreement_n_claims": n_multi,
        "raw_pre_adjudication_agreement_note": (
            "mean pairwise agreement over annotator_*_status columns; null when fewer "
            "than two independent annotations are present"
        ),
        "human_counts": dict(Counter(yt)),
        "checker_counts": dict(Counter(yp)),
    }


def load_pairs(human_path, key_path):
    h = read(human_path)
    k = {r["annotation_id"].strip(): r for r in read(key_path)}
    pairs, missing_label, missing_key = [], 0, 0
    for r in h:
        if not _norm(r.get("human_status", "")):
            missing_label += 1
            continue
        q = k.get(str(r.get("annotation_id", "")).strip())
        if q is None:
            missing_key += 1
            continue
        pairs.append((r, q))
    return pairs, missing_label, missing_key


# --------------------------------------------------------------------------
# smoke test (synthetic labels; proves the code path, asserts nothing about the paper)
# --------------------------------------------------------------------------
def smoke_test():
    rng = random.Random(7)
    pairs = []
    for i in range(400):
        t = rng.choices(CLASSES, weights=[60, 25, 10, 5])[0]
        # checker agrees ~88% of the time, with errors biased to neighbouring classes
        p = t if rng.random() < 0.88 else rng.choice([c for c in CLASSES if c != t])
        pairs.append((
            {"annotation_id": str(i), "human_status": t,
             "annotator_1_status": t,
             "annotator_2_status": t if rng.random() < 0.93 else rng.choice(CLASSES)},
            {"checker_v3_status": p},
        ))
    out = score(pairs, 400, 123)
    print(json.dumps({k: out[k] for k in
                      ("n", "accuracy", "accuracy_ci95", "macro_f1", "macro_f1_ci95",
                       "cohen_kappa_status4", "raw_pre_adjudication_agreement",
                       "raw_pre_adjudication_agreement_n_claims")}, indent=2))
    assert 0.0 <= out["accuracy"] <= 1.0
    assert 0.0 <= out["macro_f1"] <= 1.0
    assert len(out["confusion_matrix"]["rows_human_cols_checker"]) == 4
    assert out["macro_f1_ci95"][0] <= out["macro_f1"] <= out["macro_f1_ci95"][1]
    assert out["raw_pre_adjudication_agreement"] is not None
    # degenerate single-annotator input must yield a null agreement, not 0
    o2 = score([({"annotation_id": "0", "human_status": "supported"},
                 {"checker_v3_status": "supported"})], 20, 1)
    assert o2["raw_pre_adjudication_agreement"] is None
    print("SCORE_HUMAN_VALIDATION SMOKE TEST PASSED")


def main():
    ap = argparse.ArgumentParser(
        description="Score frozen Checker v3 against blinded human labels.")
    ap.add_argument("--human", help="blinded sheet returned by annotators (CSV)")
    ap.add_argument("--key",
                    default=DATA_ROOT + "/wysiwyr_real/stage6_checker_v3/"
                                         "checker_v3_human_validation_KEY_DO_NOT_SHOW_ANNOTATOR.csv")
    ap.add_argument("--output",
                    default=DATA_ROOT + "/wysiwyr_real/stage6_checker_v3/human_validation_scores.json")
    ap.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples (default 2000)")
    ap.add_argument("--seed", type=int, default=2023)
    ap.add_argument("--smoke-test", action="store_true",
                    help="run the code-path self test on synthetic labels (no data needed)")
    args = ap.parse_args()

    if args.smoke_test:
        smoke_test()
        return

    if not args.human:
        ap.error("--human is required unless --smoke-test is given")

    pairs, missing_label, missing_key = load_pairs(args.human, args.key)
    if not pairs:
        raise SystemExit(
            "No completed human labels found. The blinded sheet appears to be un-annotated "
            f"({missing_label} rows without `human_status`, {missing_key} rows without a key entry). "
            "Fill in the human_status column (supported/calibrated/unsupported/prohibited) first."
        )
    out = score(pairs, args.bootstrap, args.seed)
    out["input"] = {"human": str(args.human), "key": str(args.key),
                    "rows_without_label_skipped": missing_label,
                    "rows_without_key_skipped": missing_key}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print("WROTE", args.output)


if __name__ == "__main__":
    main()
