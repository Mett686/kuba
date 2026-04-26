# =========================
# IMPORT LIBRARIES
# =========================
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# LOAD DATA
# =========================
df = pd.read_csv("final_data hotovo uplne.csv")

# =========================
# DATA CLEANING
# =========================
# přejmenování sloupců (bez mezer)
df.columns = df.columns.str.strip().str.replace(" ", "_")

# numerické převody
numeric_cols = [
    'Snatch', 'Clean_Jerk', 'Result', 'Sinclair',
    'CMJ_Height', 'SJ_Height',
    'Handgrip_P', 'Handgrip_L',
    'Biceps_(cm)', 'Stehno_(cm)',
    'Délka_ruky_(cm)', 'Délka_nohy_(cm)', 'Výška_(cm)'
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# =========================
# DERIVED VARIABLES
# =========================

# nejlepší handgrip
df["Handgrip_max"] = df[["Handgrip_P", "Handgrip_L"]].max(axis=1)

# EUR
df["EUR"] = df["CMJ_Height"] / df["SJ_Height"]

# relativní délky
df["RAL"] = df["Délka_ruky_(cm)"] / df["Výška_(cm)"] * 100
df["RLL"] = df["Délka_nohy_(cm)"] / df["Výška_(cm)"] * 100

# výkonové skupiny (quartily)
q75 = df["Sinclair"].quantile(0.75)
df["Performance_group"] = np.where(df["Sinclair"] >= q75, "High", "Low")

# =========================
# DESCRIPTIVE STATISTICS
# =========================
desc = df.describe()
desc.to_csv("descriptive_stats.csv")

# =========================
# NORMALITY TEST
# =========================
def normality_test(series):
    series = series.dropna()
    if len(series) < 3:
        return np.nan
    return stats.shapiro(series)[1]

normality_results = {
    col: normality_test(df[col])
    for col in ["CMJ_Height", "SJ_Height", "Sinclair", "Handgrip_max"]
}

print("Normality p-values:", normality_results)

# =========================
# CORRELATION FUNCTION
# =========================
def corr_analysis(x, y):
    data = df[[x, y]].dropna()
    if len(data) < 5:
        return None
    
    pearson = stats.pearsonr(data[x], data[y])
    spearman = stats.spearmanr(data[x], data[y])
    
    return {
        "Pearson_r": pearson.statistic,
        "Pearson_p": pearson.pvalue,
        "Spearman_r": spearman.correlation,
        "Spearman_p": spearman.pvalue
    }

# =========================
# H1: CMJ, SJ vs Sinclair
# =========================
H1_CMJ = corr_analysis("CMJ_Height", "Sinclair")
H1_SJ = corr_analysis("SJ_Height", "Sinclair")

# =========================
# H2: Handgrip vs performance
# =========================
H2_snatch = corr_analysis("Handgrip_max", "Snatch")
H2_cj = corr_analysis("Handgrip_max", "Clean_Jerk")
H2_total = corr_analysis("Handgrip_max", "Result")
H2_sinclair = corr_analysis("Handgrip_max", "Sinclair")

# =========================
# H3: Circumference vs absolute performance
# =========================
H3_biceps = corr_analysis("Biceps_(cm)", "Result")
H3_thigh = corr_analysis("Stehno_(cm)", "Result")

# =========================
# H5: EUR vs Sinclair
# =========================
H5 = corr_analysis("EUR", "Sinclair")

# =========================
# GROUP COMPARISON (H4)
# =========================
def cohens_d(x1, x2):
    nx = len(x1)
    ny = len(x2)
    pooled_sd = np.sqrt(((nx - 1)*np.var(x1) + (ny - 1)*np.var(x2)) / (nx + ny - 2))
    return (np.mean(x1) - np.mean(x2)) / pooled_sd

def group_test(variable):
    high = df[df["Performance_group"] == "High"][variable].dropna()
    low = df[df["Performance_group"] == "Low"][variable].dropna()
    
    if len(high) < 5 or len(low) < 5:
        return None
    
    # normalita
    p_high = stats.shapiro(high).pvalue
    p_low = stats.shapiro(low).pvalue
    
    if p_high > 0.05 and p_low > 0.05:
        test = stats.ttest_ind(high, low)
        test_name = "t-test"
    else:
        test = stats.mannwhitneyu(high, low)
        test_name = "Mann-Whitney"
    
    d = cohens_d(high, low)
    
    return {
        "test": test_name,
        "p_value": test.pvalue,
        "cohens_d": d
    }

H4_vars = ["RAL", "RLL", "CMJ_Height", "SJ_Height", "Handgrip_max"]
H4_results = {var: group_test(var) for var in H4_vars}

# =========================
# EXPORT RESULTS
# =========================
results = {
    "H1_CMJ": H1_CMJ,
    "H1_SJ": H1_SJ,
    "H2_snatch": H2_snatch,
    "H2_cj": H2_cj,
    "H2_total": H2_total,
    "H2_sinclair": H2_sinclair,
    "H3_biceps": H3_biceps,
    "H3_thigh": H3_thigh,
    "H5": H5,
    "H4": H4_results
}

pd.DataFrame(results).to_csv("hypothesis_results.csv")

# =========================
# BASIC PLOTS (for thesis)
# =========================

# CMJ vs Sinclair
sns.regplot(data=df, x="CMJ_Height", y="Sinclair")
plt.title("CMJ vs Sinclair")
plt.savefig("H1_CMJ_plot.png")
plt.clf()

# Handgrip vs Total
sns.regplot(data=df, x="Handgrip_max", y="Result")
plt.title("Handgrip vs Total")
plt.savefig("H2_plot.png")
plt.clf()

# EUR vs Sinclair
sns.regplot(data=df, x="EUR", y="Sinclair")
plt.title("EUR vs Sinclair")
plt.savefig("H5_plot.png")
plt.clf()

print("Analysis DONE")