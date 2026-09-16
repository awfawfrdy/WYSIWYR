from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from scipy.stats import (
    mannwhitneyu,
    spearmanr,
    wilcoxon
)

ROOT = Path(
    "/root/autodl-tmp/wysiwyr_real/stage5_calibrated/reviewer2_results"
)
OUT = Path(
    "/root/autodl-tmp/wysiwyr_real/usr_failure_analysis_20260913"
)
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("USR FAILURE-CASE QUANTITATIVE ANALYSIS")
print("=" * 80)
print("Input :", ROOT)
print("Output:", OUT)
print()

# ------------------------------------------------------------
# 1. Discover datasets
# ------------------------------------------------------------
datasets = []

for d in sorted(ROOT.iterdir()):
    if not d.is_dir():
        continue

    case_csv = d / "case_segmentation_metrics.csv"
    audit_csv = d / "usr_pixel_audit.csv"

    if case_csv.exists() and audit_csv.exists():
        datasets.append(d)

print("Datasets discovered:")
for d in datasets:
    print(" -", d.name)

if not datasets:
    raise RuntimeError("No valid Stage-5 dataset directories found.")

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------
def norm_variant_key(x):
    x = str(x).strip().lower()

    mapping = {
        "baseline": "baseline",
        "medsam": "baseline",

        "abloss": "abloss",
        "medsam+abloss": "abloss",
        "medsam + abloss": "abloss",

        "usr": "usr",
        "medsam+usr": "usr",
        "medsam + usr": "usr",

        "both": "both",
        "medsam+abloss+usr": "both",
        "medsam + abloss + usr": "both",
    }
    return mapping.get(x, x)


def safe_mwu(a, b):
    a = pd.to_numeric(pd.Series(a), errors="coerce").dropna().values
    b = pd.to_numeric(pd.Series(b), errors="coerce").dropna().values
    if len(a) == 0 or len(b) == 0:
        return np.nan
    try:
        return mannwhitneyu(a, b, alternative="two-sided").pvalue
    except Exception:
        return np.nan


def safe_spearman(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    z = pd.DataFrame({"x": x, "y": y}).dropna()

    if len(z) < 3:
        return np.nan, np.nan

    if z["x"].nunique() < 2 or z["y"].nunique() < 2:
        return np.nan, np.nan

    try:
        r, p = spearmanr(z["x"], z["y"])
        return float(r), float(p)
    except Exception:
        return np.nan, np.nan


def bootstrap_ci(values, func=np.mean, n_boot=3000, seed=2026):
    v = pd.to_numeric(pd.Series(values), errors="coerce").dropna().values
    if len(v) == 0:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    boots = []

    for _ in range(n_boot):
        s = rng.choice(v, size=len(v), replace=True)
        boots.append(func(s))

    return (
        float(np.quantile(boots, 0.025)),
        float(np.quantile(boots, 0.975))
    )


# ------------------------------------------------------------
# 2. Load case segmentation metrics
# ------------------------------------------------------------
seg_all = []
audit_all = []

for d in datasets:
    dataset = d.name

    seg = pd.read_csv(d / "case_segmentation_metrics.csv")
    seg["dataset"] = dataset
    seg["case_id"] = seg["case_id"].astype(str)

    if "variant_key" in seg.columns:
        seg["variant_key_norm"] = seg["variant_key"].map(norm_variant_key)
    else:
        seg["variant_key_norm"] = seg["variant"].map(norm_variant_key)

    seg_all.append(seg)

    au = pd.read_csv(d / "usr_pixel_audit.csv")
    au["dataset"] = dataset
    au["case_id"] = au["case_id"].astype(str)
    audit_all.append(au)

seg = pd.concat(seg_all, ignore_index=True)
audit = pd.concat(audit_all, ignore_index=True)

seg.to_csv(OUT / "all_case_segmentation_metrics.csv", index=False)
audit.to_csv(OUT / "all_usr_pixel_audit.csv", index=False)

print()
print("Segmentation rows:", len(seg))
print("Pixel-audit rows :", len(audit))
print()

print("Variant keys:")
print(seg["variant_key_norm"].value_counts(dropna=False))
print()

# ------------------------------------------------------------
# 3. Convert Stage5 4 variants to wide case table
# ------------------------------------------------------------
metrics = [
    "dice",
    "iou",
    "precision",
    "recall",
    "boundary_dice",
    "hd95",
    "assd"
]

keep_metrics = [m for m in metrics if m in seg.columns]

wide = seg.pivot_table(
    index=["dataset", "case_id"],
    columns="variant_key_norm",
    values=keep_metrics,
    aggfunc="first"
)

wide.columns = [
    f"{metric}_{variant}"
    for metric, variant in wide.columns
]
wide = wide.reset_index()

required = [
    "dice_baseline",
    "dice_usr",
    "dice_abloss",
    "dice_both",
    "boundary_dice_baseline",
    "boundary_dice_usr",
    "boundary_dice_abloss",
    "boundary_dice_both"
]

missing = [x for x in required if x not in wide.columns]
if missing:
    print("WARNING: missing expected columns:", missing)

# ------------------------------------------------------------
# 4. Compute paired USR changes
# ------------------------------------------------------------
if {"dice_baseline", "dice_usr"}.issubset(wide.columns):
    wide["delta_dice_baseline_to_usr"] = (
        wide["dice_usr"] - wide["dice_baseline"]
    )
    wide["dice_worsen_baseline_to_usr"] = (
        wide["delta_dice_baseline_to_usr"] < 0
    ).astype(int)

if {"boundary_dice_baseline", "boundary_dice_usr"}.issubset(wide.columns):
    wide["delta_bdice_baseline_to_usr"] = (
        wide["boundary_dice_usr"] - wide["boundary_dice_baseline"]
    )
    wide["bdice_worsen_baseline_to_usr"] = (
        wide["delta_bdice_baseline_to_usr"] < 0
    ).astype(int)

if {"dice_abloss", "dice_both"}.issubset(wide.columns):
    wide["delta_dice_abloss_to_both"] = (
        wide["dice_both"] - wide["dice_abloss"]
    )
    wide["dice_worsen_abloss_to_both"] = (
        wide["delta_dice_abloss_to_both"] < 0
    ).astype(int)

if {"boundary_dice_abloss", "boundary_dice_both"}.issubset(wide.columns):
    wide["delta_bdice_abloss_to_both"] = (
        wide["boundary_dice_both"] - wide["boundary_dice_abloss"]
    )
    wide["bdice_worsen_abloss_to_both"] = (
        wide["delta_bdice_abloss_to_both"] < 0
    ).astype(int)

wide.to_csv(
    OUT / "usr_case_level_failure.csv",
    index=False
)

print("=" * 80)
print("CASE COUNT")
print("=" * 80)
print("Unique cases:", len(wide))
print(wide["dataset"].value_counts())
print()

# ------------------------------------------------------------
# 5. Overall + per-dataset worsen summary
# ------------------------------------------------------------
comparisons = [
    (
        "Baseline_to_USR",
        "delta_dice_baseline_to_usr",
        "dice_worsen_baseline_to_usr",
        "delta_bdice_baseline_to_usr",
        "bdice_worsen_baseline_to_usr"
    ),
    (
        "ABLoss_to_Both",
        "delta_dice_abloss_to_both",
        "dice_worsen_abloss_to_both",
        "delta_bdice_abloss_to_both",
        "bdice_worsen_abloss_to_both"
    )
]

summary_rows = []

for comp, dcol, wcol, bdcol, bwcol in comparisons:
    if dcol not in wide.columns:
        continue

    for dataset_name, g in [
        ("ALL_798", wide),
        *[(x, y) for x, y in wide.groupby("dataset")]
    ]:
        row = {
            "comparison": comp,
            "dataset": dataset_name,
            "n": len(g),
            "mean_delta_dice": g[dcol].mean(),
            "median_delta_dice": g[dcol].median(),
            "dice_worsen_count": int(g[wcol].sum()),
            "dice_worsen_rate": g[wcol].mean(),
        }

        lo, hi = bootstrap_ci(g[dcol])
        row["mean_delta_dice_ci95_low"] = lo
        row["mean_delta_dice_ci95_high"] = hi

        try:
            x = g[dcol].dropna()
            if len(x) > 0 and np.any(x != 0):
                row["wilcoxon_delta_dice_p"] = wilcoxon(x).pvalue
            else:
                row["wilcoxon_delta_dice_p"] = np.nan
        except Exception:
            row["wilcoxon_delta_dice_p"] = np.nan

        if bdcol in g.columns:
            row["mean_delta_bdice"] = g[bdcol].mean()
            row["median_delta_bdice"] = g[bdcol].median()
            row["bdice_worsen_count"] = int(g[bwcol].sum())
            row["bdice_worsen_rate"] = g[bwcol].mean()

            lo, hi = bootstrap_ci(g[bdcol])
            row["mean_delta_bdice_ci95_low"] = lo
            row["mean_delta_bdice_ci95_high"] = hi

            try:
                x = g[bdcol].dropna()
                if len(x) > 0 and np.any(x != 0):
                    row["wilcoxon_delta_bdice_p"] = wilcoxon(x).pvalue
                else:
                    row["wilcoxon_delta_bdice_p"] = np.nan
            except Exception:
                row["wilcoxon_delta_bdice_p"] = np.nan

        summary_rows.append(row)

summary = pd.DataFrame(summary_rows)
summary.to_csv(
    OUT / "usr_failure_group_summary.csv",
    index=False
)

print("=" * 80)
print("WORSEN SUMMARY")
print("=" * 80)

show_cols = [
    "comparison",
    "dataset",
    "n",
    "mean_delta_dice",
    "dice_worsen_count",
    "dice_worsen_rate",
    "mean_delta_bdice",
    "bdice_worsen_count",
    "bdice_worsen_rate"
]

show_cols = [x for x in show_cols if x in summary.columns]

print(summary[show_cols].to_string(index=False))
print()

# ------------------------------------------------------------
# 6. Merge semantic pixel audit
# ------------------------------------------------------------
# comparison strings observed in current Stage5:
# MedSAM -> MedSAM+USR
# MedSAM+ABLoss -> MedSAM+ABLoss+USR

def classify_comparison(x):
    s = str(x).replace(" ", "").lower()

    if "abloss" in s:
        return "ABLoss_to_Both"

    if "usr" in s:
        return "Baseline_to_USR"

    return s

audit["comparison_norm"] = audit["comparison"].map(classify_comparison)

# ------------------------------------------------------------
# 7. Semantic edit summary
# ------------------------------------------------------------
semantic_cols = [
    "correctly_recovered_lesion_pixels",
    "incorrectly_added_pixels",
    "correctly_removed_false_positive_pixels",
    "incorrectly_removed_lesion_pixels",
    "correctly_recovered_rate_vs_gt_lesion",
    "incorrectly_removed_rate_vs_gt_lesion",
    "incorrectly_added_rate_vs_gt_background",
    "correctly_removed_fp_rate_vs_gt_background",
    "dice_before",
    "dice_after",
    "dice_worsened",
    "boundary_dice_before",
    "boundary_dice_after",
    "boundary_dice_worsened",
]

semantic_cols = [c for c in semantic_cols if c in audit.columns]

semantic_summary_rows = []

for comp, g0 in audit.groupby("comparison_norm"):

    for dataset_name, g in [
        ("ALL_798", g0),
        *[(x, y) for x, y in g0.groupby("dataset")]
    ]:

        row = {
            "comparison": comp,
            "dataset": dataset_name,
            "n": len(g)
        }

        for col in semantic_cols:
            if col in [
                "dice_worsened",
                "boundary_dice_worsened"
            ]:
                row[f"{col}_rate"] = pd.to_numeric(
                    g[col], errors="coerce"
                ).mean()
            else:
                z = pd.to_numeric(g[col], errors="coerce")
                row[f"{col}_mean"] = z.mean()
                row[f"{col}_median"] = z.median()

        semantic_summary_rows.append(row)

semantic_summary = pd.DataFrame(semantic_summary_rows)

semantic_summary.to_csv(
    OUT / "usr_semantic_edit_summary.csv",
    index=False
)

# ------------------------------------------------------------
# 8. Worsen vs non-worsen characteristics
# ------------------------------------------------------------
analysis_rows = []

pair_specs = [
    (
        "Baseline_to_USR",
        "dice_worsen_baseline_to_usr",
        "delta_dice_baseline_to_usr",
        [
            "dice_baseline",
            "boundary_dice_baseline",
            "hd95_baseline",
            "assd_baseline",
            "precision_baseline",
            "recall_baseline",
        ],
    ),
    (
        "ABLoss_to_Both",
        "dice_worsen_abloss_to_both",
        "delta_dice_abloss_to_both",
        [
            "dice_abloss",
            "boundary_dice_abloss",
            "hd95_abloss",
            "assd_abloss",
            "precision_abloss",
            "recall_abloss",
        ],
    ),
]

for comp, flag_col, delta_col, candidate_predictors in pair_specs:

    if flag_col not in wide.columns:
        continue

    for var in candidate_predictors:
        if var not in wide.columns:
            continue

        w = wide.loc[wide[flag_col] == 1, var]
        nw = wide.loc[wide[flag_col] == 0, var]

        r, rp = safe_spearman(wide[var], wide[delta_col])

        analysis_rows.append({
            "comparison": comp,
            "variable": var,
            "n_worsen": int(pd.to_numeric(w, errors="coerce").notna().sum()),
            "n_nonworsen": int(pd.to_numeric(nw, errors="coerce").notna().sum()),
            "worsen_mean": pd.to_numeric(w, errors="coerce").mean(),
            "worsen_median": pd.to_numeric(w, errors="coerce").median(),
            "nonworsen_mean": pd.to_numeric(nw, errors="coerce").mean(),
            "nonworsen_median": pd.to_numeric(nw, errors="coerce").median(),
            "mannwhitney_p": safe_mwu(w, nw),
            "spearman_vs_delta": r,
            "spearman_p": rp
        })

predictor_stats = pd.DataFrame(analysis_rows)

predictor_stats.to_csv(
    OUT / "usr_failure_predictor_stats.csv",
    index=False
)

# ------------------------------------------------------------
# 9. Merge audit with wide outcomes for semantic failure groups
# ------------------------------------------------------------
audit_merge = audit.merge(
    wide,
    on=["dataset", "case_id"],
    how="left"
)

audit_merge.to_csv(
    OUT / "usr_pixel_audit_with_case_metrics.csv",
    index=False
)

semantic_predictor_cols = [
    "correctly_recovered_rate_vs_gt_lesion",
    "incorrectly_removed_rate_vs_gt_lesion",
    "incorrectly_added_rate_vs_gt_background",
    "correctly_removed_fp_rate_vs_gt_background",
]

sem_rows = []

for comp, g in audit_merge.groupby("comparison_norm"):

    if comp == "Baseline_to_USR":
        flag_col = "dice_worsen_baseline_to_usr"
        delta_col = "delta_dice_baseline_to_usr"

    elif comp == "ABLoss_to_Both":
        flag_col = "dice_worsen_abloss_to_both"
        delta_col = "delta_dice_abloss_to_both"

    else:
        continue

    if flag_col not in g.columns:
        continue

    for var in semantic_predictor_cols:
        if var not in g.columns:
            continue

        w = g.loc[g[flag_col] == 1, var]
        nw = g.loc[g[flag_col] == 0, var]
        r, rp = safe_spearman(g[var], g[delta_col])

        sem_rows.append({
            "comparison": comp,
            "variable": var,
            "n_worsen": pd.to_numeric(
                w, errors="coerce"
            ).notna().sum(),
            "n_nonworsen": pd.to_numeric(
                nw, errors="coerce"
            ).notna().sum(),
            "worsen_mean": pd.to_numeric(
                w, errors="coerce"
            ).mean(),
            "worsen_median": pd.to_numeric(
                w, errors="coerce"
            ).median(),
            "nonworsen_mean": pd.to_numeric(
                nw, errors="coerce"
            ).mean(),
            "nonworsen_median": pd.to_numeric(
                nw, errors="coerce"
            ).median(),
            "mannwhitney_p": safe_mwu(w, nw),
            "spearman_vs_delta": r,
            "spearman_p": rp
        })

semantic_predictors = pd.DataFrame(sem_rows)

semantic_predictors.to_csv(
    OUT / "usr_semantic_failure_predictor_stats.csv",
    index=False
)

# ------------------------------------------------------------
# 10. Dataset-specific risk table
# ------------------------------------------------------------
risk_rows = []

for comp, _, flag_col, _, bd_flag_col in comparisons:

    if flag_col not in wide.columns:
        continue

    for dataset_name, g in wide.groupby("dataset"):

        risk_rows.append({
            "comparison": comp,
            "dataset": dataset_name,
            "n": len(g),
            "dice_worsen_n": int(g[flag_col].sum()),
            "dice_worsen_pct": 100 * g[flag_col].mean(),
            "bdice_worsen_n": (
                int(g[bd_flag_col].sum())
                if bd_flag_col in g.columns else np.nan
            ),
            "bdice_worsen_pct": (
                100 * g[bd_flag_col].mean()
                if bd_flag_col in g.columns else np.nan
            )
        })

risk = pd.DataFrame(risk_rows)

risk.to_csv(
    OUT / "usr_dataset_failure_risk.csv",
    index=False
)

# ------------------------------------------------------------
# 11. Worst / best cases for later qualitative inspection
# ------------------------------------------------------------
case_rank_cols = [
    "dataset",
    "case_id",
    "dice_baseline",
    "dice_usr",
    "delta_dice_baseline_to_usr",
    "boundary_dice_baseline",
    "boundary_dice_usr",
    "delta_bdice_baseline_to_usr",
    "dice_abloss",
    "dice_both",
    "delta_dice_abloss_to_both",
    "boundary_dice_abloss",
    "boundary_dice_both",
    "delta_bdice_abloss_to_both",
]

case_rank_cols = [
    c for c in case_rank_cols if c in wide.columns
]

if "delta_dice_baseline_to_usr" in wide.columns:

    worst = wide.sort_values(
        "delta_dice_baseline_to_usr"
    ).head(50)

    best = wide.sort_values(
        "delta_dice_baseline_to_usr",
        ascending=False
    ).head(50)

    worst[case_rank_cols].to_csv(
        OUT / "usr_top50_worst_cases.csv",
        index=False
    )

    best[case_rank_cols].to_csv(
        OUT / "usr_top50_best_cases.csv",
        index=False
    )

# ------------------------------------------------------------
# 12. Optional multivariable logistic regression
# ------------------------------------------------------------
logit_rows = []

try:
    import statsmodels.formula.api as smf

    for comp, flag_col, _, candidate_predictors in pair_specs:

        if flag_col not in wide.columns:
            continue

        # Use only pre-USR segmentation quality + dataset,
        # not post-USR semantic edit outcomes.
        predictors = [
            x for x in candidate_predictors
            if x in wide.columns
        ]

        # avoid too many correlated predictors
        selected = []

        for preferred in [
            "dice_baseline",
            "boundary_dice_baseline",
            "hd95_baseline",
            "dice_abloss",
            "boundary_dice_abloss",
            "hd95_abloss",
        ]:
            if preferred in predictors:
                selected.append(preferred)

        # max 3 numerical predictors
        selected = selected[:3]

        cols = [flag_col, "dataset"] + selected
        dat = wide[cols].dropna().copy()

        if len(dat) < 100 or dat[flag_col].nunique() < 2:
            continue

        rhs = " + ".join(selected + ["C(dataset)"])
        formula = f"{flag_col} ~ {rhs}"

        print()
        print("Logistic model:", formula)

        model = smf.logit(
            formula=formula,
            data=dat
        ).fit(disp=False, maxiter=500)

        conf = model.conf_int()

        for term in model.params.index:
            beta = model.params[term]
            lo = conf.loc[term, 0]
            hi = conf.loc[term, 1]

            logit_rows.append({
                "comparison": comp,
                "term": term,
                "beta": beta,
                "odds_ratio": np.exp(beta),
                "ci95_low": np.exp(lo),
                "ci95_high": np.exp(hi),
                "p_value": model.pvalues[term],
                "n": len(dat)
            })

except Exception as e:
    print()
    print("NOTE: statsmodels logistic regression skipped:")
    print(repr(e))

pd.DataFrame(logit_rows).to_csv(
    OUT / "usr_failure_logistic_regression.csv",
    index=False
)

# ------------------------------------------------------------
# 13. Text summary
# ------------------------------------------------------------
lines = []

lines.append("USR FAILURE-CASE ANALYSIS")
lines.append("=" * 70)
lines.append(f"Unique cases: {len(wide)}")
lines.append("")

for comp, dcol, wcol, bdcol, bwcol in comparisons:

    if dcol not in wide.columns:
        continue

    lines.append(comp)
    lines.append("-" * 70)

    lines.append(
        f"Mean Delta Dice: {wide[dcol].mean():.6f}"
    )

    lines.append(
        f"Dice worsen: {int(wide[wcol].sum())}/{len(wide)} "
        f"({100 * wide[wcol].mean():.2f}%)"
    )

    if bdcol in wide.columns:
        lines.append(
            f"Mean Delta Boundary Dice: {wide[bdcol].mean():.6f}"
        )

        lines.append(
            f"Boundary-Dice worsen: "
            f"{int(wide[bwcol].sum())}/{len(wide)} "
            f"({100 * wide[bwcol].mean():.2f}%)"
        )

    lines.append("")

(OUT / "SUMMARY.txt").write_text(
    "\n".join(lines),
    encoding="utf-8"
)

print()
print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print("\n".join(lines))

print()
print("=" * 80)
print("OUTPUT FILES")
print("=" * 80)

for p in sorted(OUT.iterdir()):
    print(p.name)

print()
print("USR FAILURE ANALYSIS COMPLETED SUCCESSFULLY")
