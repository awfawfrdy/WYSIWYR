#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# WYSIWYR clinician evaluation -- primary statistical analysis (R)
#
# The manuscript's primary clinician analysis is a cumulative-link mixed-effects
# model of the ordinal 1-5 quality rating with
#     * report condition        as the fixed effect of interest,
#     * clinician               as a crossed random intercept,
#     * case                    as a crossed random intercept,
# and reports odds ratios with 95% confidence intervals and p-values, plus
#     * an experienced-only sensitivity analysis (the 18 experienced clinicians),
#     * an exploratory method x experience interaction.
#
# Python has no equivalent crossed-random-effects ordinal model, hence R
# `ordinal::clmm` (as permitted by the release plan).
#
# ---------------------------------------------------------------------------
# STATUS / HONESTY NOTE
# ---------------------------------------------------------------------------
# The original completed clinician-rating records are NOT included in the current
# reproducibility archive: they were searched for on the analysis server and in
# the local project archives and were not located. This script has therefore been
# written to the documented input schema and syntax-checked, but it has NOT been
# executed against the manuscript data and it does NOT contain, hardcode or assume
# any reported number. Run it once the completed rating records are available and
# compare the output against the manuscript tables.
#
# Expected input (long format, one row per evaluation):
#   clinician_id, case_id, dataset, condition, score, experience
#   - condition : 3 levels (report condition; identity concealed during rating)
#   - score     : ordinal integer 1..5
#   - experience: "experienced" for the 18 experienced clinicians
#
# Usage:
#   Rscript analysis/clinician_mixed_effects.R --data <ratings.csv> --out <outdir>
# ---------------------------------------------------------------------------

suppressWarnings(suppressMessages({
  ok <- requireNamespace("ordinal", quietly = TRUE)
}))

args <- commandArgs(trailingOnly = TRUE)
getopt <- function(flag, default = NULL) {
  i <- match(flag, args)
  if (is.na(i)) return(default)
  if (i == length(args)) stop(sprintf("Missing value for %s", flag))
  args[i + 1]
}

data_path <- getopt("--data")
out_dir   <- getopt("--out", "clinician_results")

if (is.null(data_path)) {
  stop("--data <ratings.csv> is required.\n",
       "The original completed clinician-rating records are not included in the current ",
       "reproducibility archive; see configs/clinician_evaluation_protocol.json for the ",
       "expected schema.")
}
if (!file.exists(data_path)) stop(sprintf("Rating file not found: %s", data_path))
if (!ok) {
  stop("Package 'ordinal' is required. install.packages('ordinal')")
}

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

d <- read.csv(data_path, stringsAsFactors = FALSE)

required <- c("clinician_id", "case_id", "dataset", "condition", "score", "experience")
missing <- setdiff(required, names(d))
if (length(missing)) {
  stop(sprintf("Rating file is missing required columns: %s",
               paste(missing, collapse = ", ")))
}

d$score        <- as.ordered(d$score)
d$condition    <- factor(d$condition)
d$clinician_id <- factor(d$clinician_id)
d$case_id      <- factor(d$case_id)
d$experience   <- factor(d$experience)

cat("== input ==\n")
cat("rows              :", nrow(d), "\n")
cat("clinicians        :", nlevels(d$clinician_id), "\n")
cat("  experienced     :", nlevels(droplevels(d$clinician_id[d$experience == "experienced"])), "\n")
cat("cases             :", nlevels(d$case_id), "\n")
cat("conditions        :", nlevels(d$condition), "\n")
cat("evaluations/cond  :", paste(tapply(d$score, d$condition, length), collapse = ", "), "\n\n")

# --- report a model's fixed effects as OR / 95% CI / p ---------------------
report_or <- function(model, label, path) {
  co   <- coef(summary(model))
  est  <- co$Estimate
  se   <- co$`Std. Error`
  z    <- co$`z value`
  p    <- co$`Pr(>|z|)`
  # Wald 95% CI on the log-odds scale (fast, deterministic)
  lo <- est - 1.96 * se
  hi <- est + 1.96 * se
  tab <- data.frame(
    term     = rownames(co),
    log_odds = est,
    odds_ratio = exp(est),
    or_ci_low  = exp(lo),
    or_ci_high = exp(hi),
    z = z, p_value = p,
    row.names = NULL
  )
  write.csv(tab, path, row.names = FALSE)
  cat("==", label, "==\n")
  print(tab, digits = 4)
  cat("\n")
  tab
}

# --- 1. primary model: condition + crossed random intercepts ---------------
m_primary <- ordinal::clmm(
  score ~ condition + (1 | clinician_id) + (1 | case_id),
  data = d, link = "logit", Hess = TRUE
)
cat("== primary model ==\n")
print(summary(m_primary))
vc <- ordinal::VarCorr(m_primary)
cat("\nrandom-effect SDs:\n"); print(vc)
write.csv(as.data.frame(vc), file.path(out_dir, "primary_random_effects.csv"), row.names = FALSE)
report_or(m_primary, "primary model (condition fixed effect)",
          file.path(out_dir, "primary_condition_or.csv"))

# --- 2. experienced-only sensitivity analysis (18 experienced clinicians) --
d_exp <- droplevels(d[d$experience == "experienced", ])
cat("== experienced-only subset ==\n")
cat("rows:", nrow(d_exp), " clinicians:", nlevels(d_exp$clinician_id), "\n\n")
if (nlevels(d_exp$condition) >= 2 && nlevels(d_exp$clinician_id) >= 2) {
  m_exp <- ordinal::clmm(
    score ~ condition + (1 | clinician_id) + (1 | case_id),
    data = d_exp, link = "logit", Hess = TRUE
  )
  report_or(m_exp, "experienced-only sensitivity",
            file.path(out_dir, "experienced_only_condition_or.csv"))
} else {
  cat("SKIPPED: insufficient levels for the experienced-only subset.\n\n")
}

# --- 3. exploratory method x experience interaction ------------------------
if (nlevels(d$experience) >= 2) {
  m_int <- ordinal::clmm(
    score ~ condition * experience + (1 | clinician_id) + (1 | case_id),
    data = d, link = "logit", Hess = TRUE
  )
  report_or(m_int, "condition x experience interaction (exploratory)",
            file.path(out_dir, "condition_by_experience_interaction.csv"))
} else {
  cat("SKIPPED: only one experience level present.\n")
}

cat("Wrote outputs to:", normalizePath(out_dir), "\n")
