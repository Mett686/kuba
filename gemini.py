import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, pearsonr, spearmanr, ttest_ind, mannwhitneyu

# Setup plotting style
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})

if not os.path.exists('graphs_english'):
    os.makedirs('graphs_english')

# 1. Load Data
df = pd.read_csv('weightlifting_dataset_2025.csv')

# Clean columns
cols_to_clean = [
    'Snatch', 'Clean Jerk', 'Result', 'B.weight', 'Sinclair',
    'Délka ruky (cm)', 'Délka nohy (cm)', 'Biceps (cm)', 'Stehno (cm)', 
    'Výška (cm)', 'Handgrip P', 'Handgrip L', 'CMJ Height', 'SJ Height'
]

for col in cols_to_clean:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(',', '.').str.replace(' Kg', '', case=False)
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Map Czech terms to English for plotting and groups
df['Gender'] = df['Pohlaví'].map({'Muži': 'Men', 'Ženy': 'Women'})
df['Age Group'] = df['Věková skupina'].map({'Junior': 'Junior', 'U23': 'U23'})
df['Group'] = df['Gender'] + " " + df['Age Group']

# 2. Derived Parameters
df['Sinclair_Coeff'] = df['Sinclair'] / df['Result']
df['Sinclair_Snatch'] = df['Snatch'] * df['Sinclair_Coeff']
df['Sinclair_CJ'] = df['Clean Jerk'] * df['Sinclair_Coeff']
df['EUR'] = df['CMJ Height'] / df['SJ Height']
df['RLL'] = (df['Délka nohy (cm)'] / df['Výška (cm)']) * 100
df['RAL'] = (df['Délka ruky (cm)'] / df['Výška (cm)']) * 100
df['Handgrip_Max'] = df[['Handgrip P', 'Handgrip L']].max(axis=1)

# 3. Helper Functions
def get_correlation(data, var1, var2):
    subset = data[[var1, var2]].dropna()
    if len(subset) < 3: return np.nan, np.nan, "N/A"
    p1, p2 = shapiro(subset[var1])[1], shapiro(subset[var2])[1]
    if p1 > 0.05 and p2 > 0.05:
        r, p = pearsonr(subset[var1], subset[var2])
        return r, p, "Pearson"
    else:
        r, p = spearmanr(subset[var1], subset[var2])
        return r, p, "Spearman"

def plot_scatter(data, x_col, y_col, title, filename, x_label, y_label):
    subset = data[[x_col, y_col]].dropna()
    if len(subset) < 3: return
    r, p, method = get_correlation(data, x_col, y_col)
    
    plt.figure(figsize=(8, 6))
    sns.regplot(data=subset, x=x_col, y=y_col, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
    plt.title(f"{title}\n({method}: r = {r:.3f}, p = {p:.3f})")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.savefig(f'graphs_english/{filename}.png')
    plt.close()

def plot_boxplot(data, x_col, y_col, title, filename, y_label):
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=data, x=x_col, y=y_col, palette="Set2")
    sns.stripplot(data=data, x=x_col, y=y_col, color=".3", alpha=0.5, jitter=True)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(f'graphs_english/{filename}.png')
    plt.close()

# 4. Analysis H1, H2, H3
results_h123 = []
perf_vars = [
    ('Snatch', 'Absolute (kg)', 'Snatch (kg)'), 
    ('Sinclair_Snatch', 'Sinclair (pts)', 'Snatch Sinclair Points'),
    ('Clean Jerk', 'Absolute (kg)', 'Clean & Jerk (kg)'), 
    ('Sinclair_CJ', 'Sinclair (pts)', 'Clean & Jerk Sinclair Points'),
    ('Result', 'Absolute (kg)', 'Total (kg)'), 
    ('Sinclair', 'Sinclair (pts)', 'Total Sinclair Points')
]
groups = ['Men Junior', 'Men U23', 'Women Junior', 'Women U23']

for grp in groups:
    grp_data = df[df['Group'] == grp]
    for p_var, p_type, _ in perf_vars:
        r_cmj, p_cmj, _ = get_correlation(grp_data, 'CMJ Height', p_var)
        r_sj, p_sj, _ = get_correlation(grp_data, 'SJ Height', p_var)
        r_hg, p_hg, _ = get_correlation(grp_data, 'Handgrip_Max', p_var)
        r_bic, p_bic, _ = get_correlation(grp_data, 'Biceps (cm)', p_var)
        r_th, p_th, _ = get_correlation(grp_data, 'Stehno (cm)', p_var)
        
        results_h123.append({
            'Group': grp, 'Parameter': p_type, 'Discipline': p_var,
            'CMJ_r': r_cmj, 'CMJ_p': p_cmj, 'SJ_r': r_sj, 'SJ_p': p_sj,
            'Handgrip_r': r_hg, 'Handgrip_p': p_hg,
            'Biceps_r': r_bic, 'Biceps_p': p_bic, 'Thigh_r': r_th, 'Thigh_p': p_th
        })

# Generate English Scatter Plots
plot_scatter(df, 'CMJ Height', 'Sinclair', 'H1: CMJ Height vs Sinclair Coefficient (Total)', 'H1_CMJ_vs_Sinclair_EN', 'CMJ Height (cm)', 'Sinclair Points')
plot_scatter(df, 'Handgrip_Max', 'Snatch', 'H2: Max Handgrip vs Snatch (Total)', 'H2_Handgrip_vs_Snatch_EN', 'Max Handgrip (kg)', 'Snatch (kg)')
plot_scatter(df, 'Handgrip_Max', 'Clean Jerk', 'H2: Max Handgrip vs Clean & Jerk (Total)', 'H2_Handgrip_vs_CJ_EN', 'Max Handgrip (kg)', 'Clean & Jerk (kg)')
plot_scatter(df, 'Stehno (cm)', 'Result', 'H3: Thigh Circumference vs Total (Total)', 'H3_Thigh_vs_Total_EN', 'Thigh Circumference (cm)', 'Total Result (kg)')

# 5. Analysis H4
h4_results = []
anthro_vars = [('RLL', 'Relative Leg Length (RLL)'), ('RAL', 'Relative Arm Length (RAL)')]
df_h4_plot = pd.DataFrame()

for gender in ['Men', 'Women']:
    gender_data = df[df['Gender'] == gender].dropna(subset=['Sinclair', 'RLL', 'RAL'])
    if gender_data.empty: continue
        
    threshold = gender_data['Sinclair'].quantile(0.75)
    gender_data['Category_H4'] = np.where(gender_data['Sinclair'] >= threshold, 'Elite (TOP 25%)', 'Others')
    df_h4_plot = pd.concat([df_h4_plot, gender_data])
    
    elite = gender_data[gender_data['Category_H4'] == 'Elite (TOP 25%)']
    rest = gender_data[gender_data['Category_H4'] == 'Others']
    
    for var, var_name in anthro_vars:
        m_e, s_e = elite[var].mean(), elite[var].std()
        m_r, s_r = rest[var].mean(), rest[var].std()
        
        p_n = min(shapiro(elite[var])[1], shapiro(rest[var])[1])
        p_val = ttest_ind(elite[var], rest[var])[1] if p_n > 0.05 else mannwhitneyu(elite[var], rest[var])[1]
            
        h4_results.append({'Gender': gender, 'Variable': var_name, 'Elite_M_SD': f"{m_e:.2f}±{s_e:.2f}", 
                           'Rest_M_SD': f"{m_r:.2f}±{s_r:.2f}", 'p_value': p_val})

# Generate English Boxplots
if not df_h4_plot.empty:
    plot_boxplot(df_h4_plot[df_h4_plot['Gender']=='Men'], 'Category_H4', 'RLL', 'H4: Relative Leg Length (RLL) - Men', 'H4_RLL_Men_EN', 'RLL (%)')
    plot_boxplot(df_h4_plot[df_h4_plot['Gender']=='Women'], 'Category_H4', 'RLL', 'H4: Relative Leg Length (RLL) - Women', 'H4_RLL_Women_EN', 'RLL (%)')

# 6. Analysis H5
h5_results = []
for grp in groups:
    grp_data = df[df['Group'] == grp]
    r_eur, p_eur, _ = get_correlation(grp_data, 'EUR', 'Sinclair')
    h5_results.append({'Group': grp, 'EUR_r': r_eur, 'EUR_p': p_eur})

plot_scatter(df, 'EUR', 'Sinclair', 'H5: Eccentric Utilization Ratio (EUR) vs Sinclair (Total)', 'H5_EUR_vs_Sinclair_EN', 'EUR Index', 'Sinclair Points')

# 7. Save Excel
with pd.ExcelWriter('Hypotheses_Results_English.xlsx') as writer:
    pd.DataFrame(results_h123).to_excel(writer, sheet_name='H1_H2_H3_Correlations', index=False)
    pd.DataFrame(h4_results).to_excel(writer, sheet_name='H4_Anthropometry', index=False)
    pd.DataFrame(h5_results).to_excel(writer, sheet_name='H5_EUR', index=False)