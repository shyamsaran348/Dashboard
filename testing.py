# hypothesis_test_suite_with_statements.py
"""
Comprehensive hypothesis-testing suite for EEG features with explicit hypothesis statements.

Reads:  /mnt/data/a1821912-e2bf-4094-b16e-2c0ce80e6eb6.csv  (replace DATA_FILE if needed)
Writes: ./artifacts/eda/*.csv
        ./artifacts/eda/stat_test_summary.txt
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# ---------------------------
# SETTINGS
# ---------------------------
# <-- I've set DATA_FILE to the uploaded file path found in your session history.
DATA_FILE = "/Users/shyam/Desktop/DAV/dashboard/artifacts/features_table.csv"
ARTIFACT_DIR = "/Users/shyam/Desktop/DAV/dashboard/artifacts/eda"  # keep your artifact dir
ALPHA = 0.05
TOP_K = 10
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# ---------------------------
# HELPERS
# ---------------------------
def is_constant(series):
    return series.dropna().nunique() <= 1

def bh_fdr(pvals, alpha=0.05):
    p = np.array(pvals)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresholds = (np.arange(1, n+1) / n) * alpha
    below = ranked <= thresholds
    if not np.any(below):
        return np.zeros(n, dtype=bool)
    max_i = np.max(np.where(below)[0])
    reject = np.zeros(n, dtype=bool)
    reject[:max_i+1] = True
    mask = np.zeros(n, dtype=bool)
    mask[order] = reject
    return mask

def decision_from_p(p, alpha=ALPHA):
    return "Reject H0" if (p is not None and (not np.isnan(p)) and p < alpha) else "Fail to reject H0"

# ---------------------------
# LOAD
# ---------------------------
print("Loading:", DATA_FILE)
df = pd.read_csv(DATA_FILE)

if "label_enc" not in df.columns:
    if "semantic_relevance" in df.columns:
        df["label_enc"] = df["semantic_relevance"].astype(int)
    else:
        raise ValueError("No 'label_enc' column found in dataset.")

exclude = {"label_enc", "semantic_relevance", "participant", "topic", "word"}
numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number) and c not in exclude]

constant_feats = [c for c in numeric_cols if is_constant(df[c])]
active_feats = [c for c in numeric_cols if c not in constant_feats]

print("Numeric features:", numeric_cols)
print("Constant (skipped):", constant_feats)
print("Active features:", active_feats)

# ---------------------------
# HYPOTHESIS TEXT TEMPLATES
# ---------------------------
def ttest_hypothesis(feat):
    H0 = f"Null: The mean of '{feat}' is equal for label_enc=0 and label_enc=1 (Mean_{feat}|0 = Mean_{feat}|1)."
    H1 = f"Alternative: The mean of '{feat}' differs between label_enc=0 and label_enc=1."
    return H0, H1

def mw_hypothesis(feat):
    H0 = f"Null: The distribution of '{feat}' is the same for label_enc=0 and label_enc=1."
    H1 = f"Alternative: The distribution of '{feat}' differs between label_enc=0 and label_enc=1."
    return H0, H1

def anova_hypothesis(feat, group_name):
    H0 = f"Null: All group means of '{feat}' across {group_name} are equal."
    H1 = f"Alternative: At least one group's mean of '{feat}' across {group_name} differs."
    return H0, H1

def corr_hypothesis(feat):
    H0 = f"Null: There is no linear correlation between '{feat}' and label_enc (correlation = 0)."
    H1 = f"Alternative: There is a non-zero linear correlation between '{feat}' and label_enc."
    return H0, H1

def logistic_hypothesis():
    H0 = "Null: All regression coefficients (except intercept) = 0 (no relationship between features and label)."
    H1 = "Alternative: At least one coefficient != 0 (some feature predicts label)."
    return H0, H1

# ---------------------------
# 1) T-TESTS (Welch)
# ---------------------------
t_results = []
t_pvals = []

print("\n=== Welch Independent T-Tests ===\n")
for feat in active_feats:
    g0 = df.loc[df.label_enc == 0, feat].dropna()
    g1 = df.loc[df.label_enc == 1, feat].dropna()
    try:
        stat, p = stats.ttest_ind(g0, g1, equal_var=False)
    except Exception:
        stat, p = np.nan, np.nan

    H0_text, H1_text = ttest_hypothesis(feat)
    row = {
        "feature": feat,
        "H0_statement": H0_text,
        "H1_statement": H1_text,
        "t_stat": stat,
        "p": p,
        "decision": decision_from_p(p)
    }
    t_results.append(row)
    t_pvals.append(p)
    print(f"{feat:12} | t={stat:.4f}, p={p:.6f} -> {row['decision']}")

reject_mask = bh_fdr([p if not np.isnan(p) else 1.0 for p in t_pvals])
for i, r in enumerate(reject_mask):
    t_results[i]["FDR_reject"] = bool(r)

pd.DataFrame(t_results).to_csv(os.path.join(ARTIFACT_DIR, "t_tests_with_hypotheses.csv"), index=False)

# ---------------------------
# 2) MANN-WHITNEY U
# ---------------------------
mw_results = []
mw_pvals = []

print("\n=== Mann-Whitney U Tests (Non-Parametric) ===\n")
for feat in active_feats:
    g0 = df.loc[df.label_enc == 0, feat].dropna()
    g1 = df.loc[df.label_enc == 1, feat].dropna()
    try:
        stat, p = stats.mannwhitneyu(g0, g1, alternative="two-sided")
    except Exception:
        stat, p = np.nan, np.nan

    H0_text, H1_text = mw_hypothesis(feat)
    row = {
        "feature": feat,
        "H0_statement": H0_text,
        "H1_statement": H1_text,
        "U_stat": stat,
        "p": p,
        "decision": decision_from_p(p)
    }
    mw_results.append(row)
    mw_pvals.append(p)
    print(f"{feat:12} | U={stat}, p={p:.6f} -> {row['decision']}")

reject_mask = bh_fdr([p if not np.isnan(p) else 1.0 for p in mw_pvals])
for i, r in enumerate(reject_mask):
    mw_results[i]["FDR_reject"] = bool(r)

pd.DataFrame(mw_results).to_csv(os.path.join(ARTIFACT_DIR, "mannwhitney_with_hypotheses.csv"), index=False)

# ---------------------------
# 3) ANOVA – PARTICIPANTS
# ---------------------------
anova_part = []

print("\n=== ANOVA Across Participants ===\n")
if "participant" in df.columns:
    top_part = df.participant.value_counts().head(TOP_K).index
    for feat in active_feats:
        groups = [df.loc[df.participant == p, feat].dropna() for p in top_part]
        try:
            stat, p = stats.f_oneway(*groups)
        except Exception:
            stat, p = np.nan, np.nan

        H0_text, H1_text = anova_hypothesis(feat, group_name="participants (top {})".format(TOP_K))
        row = {
            "feature": feat,
            "H0_statement": H0_text,
            "H1_statement": H1_text,
            "F_stat": stat,
            "p": p,
            "decision": decision_from_p(p)
        }
        anova_part.append(row)
        print(f"{feat:12} | F={stat:.4f}, p={p:.6f} -> {row['decision']}")

pd.DataFrame(anova_part).to_csv(os.path.join(ARTIFACT_DIR, "anova_participants_with_hypotheses.csv"), index=False)

# ---------------------------
# 4) ANOVA – TOPICS
# ---------------------------
anova_topic = []

print("\n=== ANOVA Across Topics ===\n")
if "topic" in df.columns:
    top_topic = df.topic.value_counts().head(TOP_K).index
    for feat in active_feats:
        groups = [df.loc[df.topic == t, feat].dropna() for t in top_topic]
        try:
            stat, p = stats.f_oneway(*groups)
        except Exception:
            stat, p = np.nan, np.nan

        H0_text, H1_text = anova_hypothesis(feat, group_name="topics (top {})".format(TOP_K))
        row = {
            "feature": feat,
            "H0_statement": H0_text,
            "H1_statement": H1_text,
            "F_stat": stat,
            "p": p,
            "decision": decision_from_p(p)
        }
        anova_topic.append(row)
        print(f"{feat:12} | F={stat:.4f}, p={p:.6f} -> {row['decision']}")

pd.DataFrame(anova_topic).to_csv(os.path.join(ARTIFACT_DIR, "anova_topics_with_hypotheses.csv"), index=False)

# ---------------------------
# 5) LOGISTIC REGRESSION (Multivariate)
# ---------------------------
print("\n=== Logistic Regression (Multivariate) ===\n")
import statsmodels.api as sm

lr_df = df[["label_enc"] + active_feats].dropna()
formula = "label_enc ~ " + " + ".join(active_feats)

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        lr_model = smf.logit(formula, data=lr_df).fit(disp=False)

    coef_table = lr_model.summary2().tables[1].reset_index().rename(columns={"index":"feature"})
    H0_text, H1_text = logistic_hypothesis()
    coef_table["H0_statement"] = H0_text
    coef_table["H1_statement"] = H1_text
    coef_table["decision"] = coef_table["P>|z|"].apply(decision_from_p)
    coef_table.to_csv(os.path.join(ARTIFACT_DIR, "logistic_regression_with_hypotheses.csv"), index=False)

    print(coef_table[["feature", "Coef.", "P>|z|", "decision"]])

except Exception as e:
    print("Logistic regression failed:", e)

# ---------------------------
# 6) CORRELATIONS
# ---------------------------
print("\n=== Correlations (Pearson + Point-Biserial) ===\n")
corr_results = []

for feat in active_feats:
    try:
        r1, p1 = stats.pearsonr(df[feat].fillna(0), df.label_enc.fillna(0))
    except Exception:
        r1, p1 = np.nan, np.nan

    try:
        # pointbiserialr expects (binary, continuous) or (continuous, binary) depending on scipy version;
        # here we compute with (label, feature).
        r2, p2 = stats.pointbiserialr(df.label_enc.fillna(0), df[feat].fillna(0))
    except Exception:
        r2, p2 = np.nan, np.nan

    H0_text, H1_text = corr_hypothesis(feat)
    row = {
        "feature": feat,
        "H0_statement": H0_text,
        "H1_statement": H1_text,
        "pearson_r": r1,
        "pearson_p": p1,
        "pearson_decision": decision_from_p(p1),
        "pbiserial_r": r2,
        "pbiserial_p": p2,
        "pbiserial_decision": decision_from_p(p2),
    }
    corr_results.append(row)
    print(f"{feat:12} | Pearson p={p1:.6f} -> {decision_from_p(p1)} | PB p={p2:.6f} -> {decision_from_p(p2)}")

pd.DataFrame(corr_results).to_csv(os.path.join(ARTIFACT_DIR, "correlations_with_hypotheses.csv"), index=False)

# ---------------------------
# WRITE HUMAN SUMMARY
# ---------------------------
summary_path = os.path.join(ARTIFACT_DIR, "stat_test_summary.txt")
with open(summary_path, "w") as f:
    f.write("Statistical Test Summary for EEG Dataset\n")
    f.write("=======================================\n\n")
    f.write(f"Run on file: {os.path.basename(DATA_FILE)}\n\n")
    f.write(f"Significance level (alpha): {ALPHA}\n\n")
    f.write("This suite runs the following tests and records the full hypothesis statements with results.\n\n")

    # Top-level descriptions for each test type
    f.write("T-TEST (Welch's independent t-test):\n")
    f.write("  - H0: equal means between the two label groups (label_enc=0 vs label_enc=1).\n")
    f.write("  - H1: means differ.\n\n")

    f.write("Mann-Whitney U (non-parametric):\n")
    f.write("  - H0: distributions identical between the two label groups.\n")
    f.write("  - H1: distributions differ.\n\n")

    f.write("ANOVA (one-way):\n")
    f.write("  - H0: all group means are equal across the grouping variable (participants or topics).\n")
    f.write("  - H1: at least one group mean differs.\n\n")

    f.write("Pearson correlation / Point-biserial:\n")
    f.write("  - H0: correlation = 0 (no linear association).\n")
    f.write("  - H1: correlation != 0.\n\n")

    f.write("Logistic regression (multivariate):\n")
    f.write("  - H0: all coefficients (except intercept) are zero (no predictive relationship).\n")
    f.write("  - H1: at least one coefficient != 0.\n\n")

    # Write per-test outcomes (using the already-built results)
    def write_table_summary(table, title):
        f.write(f"\n--- {title} ---\n")
        if not table:
            f.write("No results.\n")
            return
        for r in table:
            # many tables store p under 'p' or 'pearson_p' etc. handle gracefully
            p_val = r.get("p", r.get("pearson_p", r.get("pbiserial_p", r.get("P>|z|", np.nan))))
            decision = r.get("decision", r.get("pearson_decision", r.get("pbiserial_decision", "")))
            f.write(f"{r['feature']}:\n")
            f.write(f"  H0: {r.get('H0_statement')}\n")
            f.write(f"  H1: {r.get('H1_statement')}\n")
            f.write(f"  p-value: {p_val}\n")
            f.write(f"  Decision: {decision}\n\n")

    write_table_summary(t_results, "T-TESTS")
    write_table_summary(mw_results, "MANN-WHITNEY")
    write_table_summary(anova_part, "ANOVA – PARTICIPANTS")
    write_table_summary(anova_topic, "ANOVA – TOPICS")

    f.write("\nNotes:\n- Results with NaN p-values indicate the test couldn't be computed (e.g., insufficient data or constant series).\n")
    f.write("- FDR flags present in the CSV for t-tests and Mann-Whitney indicate BH-FDR adjusted rejections.\n")

print("\nAll results saved inside:", ARTIFACT_DIR)
print("Summary file:", summary_path)
print("DONE.")
