#!/usr/bin/env python3
"""WYSIWYR clinician evaluation -- inter-rater agreement statistics (pure Python).

Computes, from an anonymised long-format rating table:

  * ICC(2,1) and ICC(2,k) -- two-way random-effects, absolute agreement
    (plus ICC(3,1) consistency for completeness)
  * Fleiss' kappa (general, arbitrary number of categories and raters per subject)
  * raw agreement (fraction of subjects whose raters all agree)

STATUS / HONESTY NOTE
---------------------
The original completed clinician-rating records are NOT included in the current
reproducibility archive: they were searched for on the analysis server and in the
local project archives and were not located. This script has been written to the
documented input schema and self-tested on synthetic data, but it has NOT been
executed against the manuscript data and contains no hardcoded or assumed result.
Run it once the completed rating records are available.

Expected input (long format, one row per rating):
    subject   -- case identifier (or case x condition, depending on the design)
    rater     -- anonymised clinician identifier
    category  -- ordinal rating (1..5) or any discrete label

Usage:
    python analysis/clinician_agreement.py --data <ratings.csv> --out <outdir>
    python analysis/clinician_agreement.py --smoke-test
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path


def read_long(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def to_matrix(rows, subject_key, rater_key, category_key):
    """Return (subjects, raters, {subject: {rater: category}})."""
    table = defaultdict(dict)
    raters = set()
    for r in rows:
        s = str(r[subject_key]).strip()
        t = str(r[rater_key]).strip()
        c = str(r[category_key]).strip()
        if not s or not t or not c:
            continue
        table[s][t] = c
        raters.add(t)
    raters = sorted(raters)
    return sorted(table), raters, table


def icc_twoway(table, subjects, raters):
    """ICC(2,1), ICC(2,k) absolute agreement and ICC(3,1) consistency.

    Uses a complete-case subset: only subjects rated by ALL raters are used,
    which is the standard balanced-design requirement for these ICC forms.
    """
    k = len(raters)
    complete = [s for s in subjects if all(t in table[s] for t in raters)]
    n = len(complete)
    if n < 2 or k < 2:
        return None
    # numeric encoding (category -> rank); require numeric-ish ratings
    def val(c):
        try:
            return float(c)
        except ValueError:
            return None
    Y = []
    for s in complete:
        row = [val(table[s][t]) for t in raters]
        if any(v is None for v in row):
            return None
        Y.append(row)

    grand = sum(sum(r) for r in Y) / (n * k)
    row_means = [sum(r) / k for r in Y]
    col_means = [sum(Y[i][j] for i in range(n)) / n for j in range(k)]

    ss_rows = k * sum((m - grand) ** 2 for m in row_means)
    ss_cols = n * sum((m - grand) ** 2 for m in col_means)
    ss_tot = sum((Y[i][j] - grand) ** 2 for i in range(n) for j in range(k))
    ss_err = ss_tot - ss_rows - ss_cols

    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))

    def safe(num, den):
        return num / den if den else float("nan")

    icc21 = safe(msr - mse, msr + (k - 1) * mse + k * (msc - mse) / n)
    icc2k = safe(msr - mse, msr + (msc - mse) / n)
    icc31 = safe(msr - mse, msr + (k - 1) * mse)
    return {
        "n_subjects_complete": n,
        "n_raters": k,
        "ms_rows": msr, "ms_cols": msc, "ms_error": mse,
        "ICC_2_1_absolute_agreement": icc21,
        "ICC_2_k_absolute_agreement": icc2k,
        "ICC_3_1_consistency": icc31,
        "note": "complete-case balanced design (subjects rated by all raters)",
    }


def fleiss_kappa(table, subjects, raters):
    """General (Fleiss) kappa with an arbitrary, possibly varying number of raters."""
    usable = {s: v for s, v in ((s, table[s]) for s in subjects) if len(v) >= 2}
    n = len(usable)
    if n == 0:
        return None
    cats = sorted({c for v in usable.values() for c in v.values()})
    n_ij = {s: Counter(v.values()) for s, v in usable.items()}
    P_i = []
    for s in usable:
        k = len(usable[s])
        tot = sum(c * c for c in n_ij[s].values())
        P_i.append((tot - k) / (k * (k - 1)))
    P_bar = sum(P_i) / n
    total_ratings = sum(len(v) for v in usable.values())
    p_j = {c: sum(n_ij[s][c] for s in usable) / total_ratings for c in cats}
    P_e = sum(p * p for p in p_j.values())
    kappa = (P_bar - P_e) / (1 - P_e) if (1 - P_e) else float("nan")
    return {
        "n_subjects": n,
        "categories": cats,
        "P_bar_observed": P_bar,
        "P_e_expected": P_e,
        "fleiss_kappa": kappa,
        "category_marginals": p_j,
    }


def raw_agreement(table, subjects):
    """Fraction of subjects on which every rater agrees."""
    usable = [v for v in (table[s] for s in subjects) if len(v) >= 2]
    if not usable:
        return None
    agree = sum(1 for v in usable if len(set(v.values())) == 1)
    return {"n_subjects": len(usable), "n_unanimous": agree,
            "raw_agreement": agree / len(usable)}


def analyse(rows, subject_key="subject", rater_key="rater", category_key="category"):
    subjects, raters, table = to_matrix(rows, subject_key, rater_key, category_key)
    return {
        "n_rows": len(rows),
        "n_subjects": len(subjects),
        "n_raters": len(raters),
        "ratings_per_subject": dict(Counter(len(table[s]) for s in subjects)),
        "raw_agreement": raw_agreement(table, subjects),
        "fleiss_kappa": fleiss_kappa(table, subjects, raters),
        "icc": icc_twoway(table, subjects, raters),
    }


def smoke_test():
    rng = random.Random(11)
    rows = []
    for s in range(40):
        latent = rng.choice([1, 2, 3, 4, 5])
        for t in range(4):  # 4 raters
            v = latent if rng.random() < 0.7 else rng.choice([1, 2, 3, 4, 5])
            rows.append({"subject": str(s), "rater": f"R{t}", "category": str(v)})
    out = analyse(rows)
    print(json.dumps(out, indent=2, default=str))
    assert out["icc"] is not None and -1 <= out["icc"]["ICC_2_1_absolute_agreement"] <= 1
    assert out["fleiss_kappa"] is not None
    assert 0 <= out["raw_agreement"]["raw_agreement"] <= 1
    # unbalanced input must still work
    out2 = analyse(rows[:-7])
    assert out2["fleiss_kappa"] is not None
    print("CLINICIAN_AGREEMENT SMOKE TEST PASSED")


def main():
    ap = argparse.ArgumentParser(
        description="Inter-rater agreement (ICC / Fleiss' kappa / raw agreement) for the "
                    "clinician evaluation. Reads an anonymised long-format rating table.")
    ap.add_argument("--data", help="long-format CSV with subject, rater, category columns")
    ap.add_argument("--subject-key", default="subject")
    ap.add_argument("--rater-key", default="rater")
    ap.add_argument("--category-key", default="category")
    ap.add_argument("--out", default="clinician_results")
    ap.add_argument("--smoke-test", action="store_true",
                    help="run the code-path self test on synthetic ratings (no data needed)")
    args = ap.parse_args()

    if args.smoke_test:
        smoke_test()
        return
    if not args.data:
        ap.error("--data is required unless --smoke-test is given")

    rows = read_long(args.data)
    out = analyse(rows, args.subject_key, args.rater_key, args.category_key)
    out["input"] = {"data": args.data}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    p = Path(args.out) / "clinician_agreement.json"
    p.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))
    print("WROTE", p)


if __name__ == "__main__":
    main()
