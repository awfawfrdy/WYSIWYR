import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu, spearmanr
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

OUT = Path(".")
wide = pd.read_csv("usr_case_level_failure.csv")
audit = pd.read_csv("usr_pixel_audit_with_case_metrics.csv")

# ============================================================
# 1. Build semantic edit balances
# ============================================================
audit["lesion_edit_balance"] = (
    audit["correctly_recovered_rate_vs_gt_lesion"]
    - audit["incorrectly_removed_rate_vs_gt_lesion"]
)

audit["background_edit_balance"] = (
    audit["correctly_removed_fp_rate_vs_gt_background"]
    - audit["incorrectly_added_rate_vs_gt_background"]
)

specs = {
    "Baseline_to_USR": {
        "flag": "dice_worsen_baseline_to_usr",
        "delta": "delta_dice_baseline_to_usr",
        "predictors": [
            "dice_baseline",
            "boundary_dice_baseline",
            "hd95_baseline"
        ]
    },
    "ABLoss_to_Both": {
        "flag": "dice_worsen_abloss_to_both",
        "delta": "delta_dice_abloss_to_both",
        "predictors": [
            "dice_abloss",
            "boundary_dice_abloss",
            "hd95_abloss"
        ]
    }
}

# ============================================================
# 2. Semantic mechanism tests + effect size + FDR
# ============================================================
semantic_vars = [
    "correctly_recovered_rate_vs_gt_lesion",
    "incorrectly_removed_rate_vs_gt_lesion",
    "incorrectly_added_rate_vs_gt_background",
    "correctly_removed_fp_rate_vs_gt_background",
    "lesion_edit_balance",
    "background_edit_balance",
]

rows = []

for comp, sp in specs.items():
    g = audit[audit["comparison_norm"] == comp].copy()
    flag = sp["flag"]
    delta = sp["delta"]

    for var in semantic_vars:
        d = g[[flag, delta, var]].dropna()
        w = d.loc[d[flag] == 1, var].values
        nw = d.loc[d[flag] == 0, var].values

        if len(w) == 0 or len(nw) == 0:
            continue

        U, p = mannwhitneyu(w, nw, alternative="two-sided")

        # Rank-biserial / Mann-Whitney effect orientation:
        # positive = greater in worsen group
        rbc = 2 * U / (len(w) * len(nw)) - 1

        rho, rho_p = spearmanr(d[var], d[delta])

        rows.append({
            "comparison": comp,
            "variable": var,
            "n_worsen": len(w),
            "n_nonworsen": len(nw),
            "worsen_mean": np.mean(w),
            "worsen_median": np.median(w),
            "nonworsen_mean": np.mean(nw),
            "nonworsen_median": np.median(nw),
            "rank_biserial_effect": rbc,
            "mannwhitney_p": p,
            "spearman_vs_delta": rho,
            "spearman_p": rho_p,
        })

sem = pd.DataFrame(rows)

sem["mannwhitney_fdr"] = np.nan
sem["spearman_fdr"] = np.nan

for comp in sem["comparison"].unique():
    ix = sem["comparison"] == comp
    sem.loc[ix, "mannwhitney_fdr"] = multipletests(
        sem.loc[ix, "mannwhitney_p"],
        method="fdr_bh"
    )[1]
    sem.loc[ix, "spearman_fdr"] = multipletests(
        sem.loc[ix, "spearman_p"],
        method="fdr_bh"
    )[1]

sem.to_csv(
    "usr_semantic_mechanism_final.csv",
    index=False
)

# ============================================================
# 3. Standardized dataset-adjusted logistic models
#    One predictor at a time = primary exploratory analysis
# ============================================================
single_rows = []

for comp, sp in specs.items():
    flag = sp["flag"]

    for pred in sp["predictors"]:
        d = wide[[flag, "dataset", pred]].dropna().copy()

        sd = d[pred].std(ddof=0)
        if sd == 0:
            continue

        zname = pred + "_z"
        d[zname] = (d[pred] - d[pred].mean()) / sd

        formula = f"{flag} ~ {zname} + C(dataset)"
        model = smf.logit(formula, data=d).fit(
            disp=False,
            maxiter=500
        )

        term = zname
        ci = model.conf_int().loc[term]

        single_rows.append({
            "comparison": comp,
            "predictor": pred,
            "n": len(d),
            "odds_ratio_per_1SD": np.exp(model.params[term]),
            "ci95_low": np.exp(ci[0]),
            "ci95_high": np.exp(ci[1]),
            "p_value": model.pvalues[term],
        })

single = pd.DataFrame(single_rows)

single["fdr_p"] = np.nan
for comp in single["comparison"].unique():
    ix = single["comparison"] == comp
    single.loc[ix, "fdr_p"] = multipletests(
        single.loc[ix, "p_value"],
        method="fdr_bh"
    )[1]

single.to_csv(
    "usr_dataset_adjusted_single_predictor_logistic.csv",
    index=False
)

# ============================================================
# 4. Standardized multivariable model
# ============================================================
multi_rows = []
vif_rows = []

for comp, sp in specs.items():
    flag = sp["flag"]
    preds = sp["predictors"]

    d = wide[[flag, "dataset"] + preds].dropna().copy()
    znames = []

    for pred in preds:
        sd = d[pred].std(ddof=0)
        z = pred + "_z"
        d[z] = (d[pred] - d[pred].mean()) / sd
        znames.append(z)

    # VIF for continuous predictors
    X = d[znames].copy()
    X["intercept"] = 1.0

    for i, z in enumerate(znames):
        vif_rows.append({
            "comparison": comp,
            "variable": z,
            "VIF": variance_inflation_factor(X.values, i)
        })

    formula = (
        f"{flag} ~ "
        + " + ".join(znames)
        + " + C(dataset)"
    )

    model = smf.logit(
        formula,
        data=d
    ).fit(disp=False, maxiter=500)

    ci = model.conf_int()

    for term in znames:
        multi_rows.append({
            "comparison": comp,
            "predictor": term.replace("_z", ""),
            "n": len(d),
            "odds_ratio_per_1SD": np.exp(model.params[term]),
            "ci95_low": np.exp(ci.loc[term, 0]),
            "ci95_high": np.exp(ci.loc[term, 1]),
            "p_value": model.pvalues[term],
        })

multi = pd.DataFrame(multi_rows)
vif = pd.DataFrame(vif_rows)

multi.to_csv(
    "usr_standardized_multivariable_logistic.csv",
    index=False
)

vif.to_csv(
    "usr_predictor_vif.csv",
    index=False
)

# ============================================================
# 5. Print publication-grade summary
# ============================================================
print("\n" + "=" * 88)
print("A. SEMANTIC FAILURE MECHANISM")
print("=" * 88)

cols = [
    "comparison",
    "variable",
    "worsen_mean",
    "nonworsen_mean",
    "rank_biserial_effect",
    "mannwhitney_p",
    "mannwhitney_fdr",
    "spearman_vs_delta",
    "spearman_fdr",
]

print(sem[cols].to_string(index=False))

print("\n" + "=" * 88)
print("B. DATASET-ADJUSTED SINGLE-PREDICTOR LOGISTIC MODELS")
print("   OR is per 1-SD increase")
print("=" * 88)

print(single.to_string(index=False))

print("\n" + "=" * 88)
print("C. MULTIVARIABLE STANDARDIZED LOGISTIC MODEL")
print("   OR is per 1-SD increase")
print("=" * 88)

print(multi.to_string(index=False))

print("\n" + "=" * 88)
print("D. VIF")
print("=" * 88)

print(vif.to_string(index=False))

print("\nFINALIZATION COMPLETED")
