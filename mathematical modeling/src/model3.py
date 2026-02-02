import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import os
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

# Professional plot settings for MCM/ICM O-Award quality
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 0,  # No titles in plots
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 10,
    'figure.titlesize': 0,
    'axes.linewidth': 1.2,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2,
})

# Create output directory
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'result_03')
os.makedirs(output_dir, exist_ok=True)

# ==========================================
# 1. Data Loading
# ==========================================
df_main = pd.read_csv('../result_01/Fan_Vote_Estimates_by_Celebrity_Week_Season.csv').rename(
    columns={'Celebrity': 'celebrity_name'}
)
# Load metadata
df_meta = pd.read_csv('../data/2026_MCM_Problem_C_Data.csv')
df_main = df_main.merge(df_meta[['celebrity_name', 'celebrity_age_during_season', 
                                   'ballroom_partner', 'celebrity_homecountry/region', 
                                   'celebrity_industry', 'placement']], 
                        on='celebrity_name', how='left')

# ==========================================
# 2. Feature Engineering
# ==========================================

# (1) Gaussian Age Membership
def gaussian_age(age, mu=30, sigma=10):
    return np.exp(-((age - mu)**2) / (2 * sigma**2))

df_main['Age_Gauss'] = df_main['celebrity_age_during_season'].apply(gaussian_age)

# (2) Professional Strength
pro_power = 1 / df_main.groupby('ballroom_partner')['placement'].mean()
df_main['Pro_Power'] = df_main['ballroom_partner'].map(pro_power)

# (3) Geographic Binary
df_main['Is_US'] = (df_main['celebrity_homecountry/region'] == 'United States').astype(int)

# (4) Industry Grouping
top_industries = df_main['celebrity_industry'].value_counts().nlargest(5).index
df_main['Industry_Group'] = df_main['celebrity_industry'].apply(
    lambda x: x if x in top_industries else 'Other'
)
industry_dummies = pd.get_dummies(df_main['Industry_Group'], prefix='Ind', drop_first=True)
df_main_processed = pd.concat([df_main, industry_dummies], axis=1)

# ==========================================
# 3. Data Merge & Standardization
# ==========================================
merged = df_main_processed.copy()
merged = merged.dropna(subset=['Judge_Score', 'Vote_Share', 'Age_Gauss', 'Pro_Power'])

# Define all features to standardize
features = ['Age_Gauss', 'Pro_Power', 'Is_US'] + [c for c in industry_dummies.columns]

# ★ CRITICAL FIX: Standardize ALL features (including Is_US and industry dummies)
# This ensures fair comparison of coefficients on the same scale
scaler = StandardScaler()
cols_to_scale = ['Judge_Score', 'Vote_Share'] + features
merged[cols_to_scale] = scaler.fit_transform(merged[cols_to_scale])

# ==========================================
# 4. Regression Models
# ==========================================
# Features already defined above
X = sm.add_constant(merged[features].astype(float))

model_judge = sm.OLS(merged['Judge_Score'], X).fit()
model_fan = sm.OLS(merged['Vote_Share'], X).fit()

# Print model summaries for verification
print("\n" + "="*80)
print("MODEL 3: DUAL-PATH REGRESSION ANALYSIS")
print("="*80)
print(f"\nSample Size: {len(merged)}")
print(f"Number of Features: {len(features)}")
print(f"\nJudge Path Model Summary:")
print(f"  R-squared: {model_judge.rsquared:.6f}")
print(f"  Adjusted R-squared: {model_judge.rsquared_adj:.6f}")
print(f"\nFan Path Model Summary:")
print(f"  R-squared: {model_fan.rsquared:.6f}")
print(f"  Adjusted R-squared: {model_fan.rsquared_adj:.6f}")

# ==========================================
# 5. PROFESSIONAL VISUALIZATIONS - MCM/ICM O-AWARD QUALITY
# ==========================================

# ========== FIGURE 1: Dual Path Regression Coefficients Comparison ==========
fig1, ax1 = plt.subplots(figsize=(14, 8))
fig1.patch.set_facecolor('white')

res_df = pd.concat([
    pd.DataFrame({
        'Feature': model_judge.params.index[1:], 
        'Coefficient': model_judge.params.values[1:],
        'StdErr': model_judge.bse.values[1:],
        'Pvalue': model_judge.pvalues.values[1:],
        'Model': 'Judge Path'
    }),
    pd.DataFrame({
        'Feature': model_fan.params.index[1:], 
        'Coefficient': model_fan.params.values[1:],
        'StdErr': model_fan.bse.values[1:],
        'Pvalue': model_fan.pvalues.values[1:],
        'Model': 'Fan Path'
    })
])

res_df['Significance'] = res_df['Pvalue'].apply(
    lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'N.S.'))
)

# Create diverging colormap
colors_judge = plt.cm.Blues(np.linspace(0.6, 1.0, len(res_df[res_df['Model']=='Judge Path'])))
colors_fan = plt.cm.Reds(np.linspace(0.6, 1.0, len(res_df[res_df['Model']=='Fan Path'])))

judge_data = res_df[res_df['Model']=='Judge Path'].sort_values('Feature')
fan_data = res_df[res_df['Model']=='Fan Path'].sort_values('Feature')

y_pos_j = np.arange(len(judge_data)) * 2
y_pos_f = y_pos_j + 1

bars1 = ax1.barh(y_pos_j, judge_data['Coefficient'].values, 0.8, 
                  label='Judge Path', color='#1F77B4', alpha=0.75, edgecolor='black', linewidth=1.2)
bars2 = ax1.barh(y_pos_f, fan_data['Coefficient'].values, 0.8, 
                  label='Fan Path', color='#D62728', alpha=0.75, edgecolor='black', linewidth=1.2)

ax1.axvline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
ax1.set_yticks(np.arange(0, len(judge_data)*2, 2) + 0.5)
ax1.set_yticklabels(judge_data['Feature'].values, fontsize=10)
ax1.set_xlabel('Standardized Coefficient (Beta)', fontsize=12, fontweight='bold')
ax1.legend(loc='lower right', fontsize=11, framealpha=0.95, edgecolor='black', fancybox=False)
ax1.grid(axis='x', alpha=0.3, linestyle='--')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{output_dir}/Judge_vs_Fan_Path_Coefficients.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: Judge_vs_Fan_Path_Coefficients.png")

# ========== FIGURE 2: Confidence Intervals for Coefficients ==========
fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(16, 7))
fig2.patch.set_facecolor('white')

# Judge Path
judge_data_sorted = judge_data.sort_values('Coefficient')
x_idx_j = np.arange(len(judge_data_sorted))
ax2a.errorbar(x_idx_j, judge_data_sorted['Coefficient'].values, 
              yerr=1.96*judge_data_sorted['StdErr'].values,
              fmt='o', color='#1F77B4', ecolor='#0A3161', capsize=8, capthick=2, 
              markersize=10, linewidth=2.5, alpha=0.75, label='95% CI')
ax2a.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.6)
ax2a.set_xticks(x_idx_j)
ax2a.set_xticklabels(judge_data_sorted['Feature'].values, rotation=45, ha='right', fontsize=10)
ax2a.set_ylabel('Standardized Coefficient', fontsize=12, fontweight='bold')
ax2a.grid(axis='y', alpha=0.3, linestyle='--')
ax2a.spines['top'].set_visible(False)
ax2a.spines['right'].set_visible(False)

# Fan Path
fan_data_sorted = fan_data.sort_values('Coefficient')
x_idx_f = np.arange(len(fan_data_sorted))
ax2b.errorbar(x_idx_f, fan_data_sorted['Coefficient'].values, 
              yerr=1.96*fan_data_sorted['StdErr'].values,
              fmt='s', color='#D62728', ecolor='#6B1F47', capsize=8, capthick=2, 
              markersize=10, linewidth=2.5, alpha=0.75, label='95% CI')
ax2b.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.6)
ax2b.set_xticks(x_idx_f)
ax2b.set_xticklabels(fan_data_sorted['Feature'].values, rotation=45, ha='right', fontsize=10)
ax2b.set_ylabel('Standardized Coefficient', fontsize=12, fontweight='bold')
ax2b.grid(axis='y', alpha=0.3, linestyle='--')
ax2b.spines['top'].set_visible(False)
ax2b.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{output_dir}/Coefficient_Confidence_Intervals.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: Coefficient_Confidence_Intervals.png")

# ========== FIGURE 3: Model Performance Comparison ==========
fig3, ((ax3a, ax3b), (ax3c, ax3d)) = plt.subplots(2, 2, figsize=(14, 10))
fig3.patch.set_facecolor('white')

# R-squared comparison
models_info = pd.DataFrame({
    'Model': ['Judge Path', 'Fan Path'],
    'R-squared': [model_judge.rsquared, model_fan.rsquared],
    'Adj R-squared': [model_judge.rsquared_adj, model_fan.rsquared_adj],
    'AIC': [model_judge.aic, model_fan.aic],
    'BIC': [model_judge.bic, model_fan.bic]
})

x_pos = np.arange(len(models_info))
ax3a.bar(x_pos, models_info['R-squared'], color=['#2E86AB', '#A23B72'], 
         alpha=0.7, edgecolor='black', linewidth=1.5, width=0.5)
ax3a.set_ylabel('R-squared', fontsize=11, fontweight='bold')
ax3a.set_xticks(x_pos)
ax3a.set_xticklabels(models_info['Model'], fontsize=10)
ax3a.set_ylim(0, 1)
ax3a.grid(axis='y', alpha=0.3)
for i, v in enumerate(models_info['R-squared']):
    ax3a.text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold', fontsize=10)

# Adjusted R-squared
ax3b.bar(x_pos, models_info['Adj R-squared'], color=['#2E86AB', '#A23B72'], 
         alpha=0.7, edgecolor='black', linewidth=1.5, width=0.5)
ax3b.set_ylabel('Adjusted R-squared', fontsize=11, fontweight='bold')
ax3b.set_xticks(x_pos)
ax3b.set_xticklabels(models_info['Model'], fontsize=10)
ax3b.set_ylim(0, 1)
ax3b.grid(axis='y', alpha=0.3)
for i, v in enumerate(models_info['Adj R-squared']):
    ax3b.text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold', fontsize=10)

# AIC (lower is better)
ax3c.bar(x_pos, models_info['AIC'], color=['#2E86AB', '#A23B72'], 
         alpha=0.7, edgecolor='black', linewidth=1.5, width=0.5)
ax3c.set_ylabel('AIC (Lower is Better)', fontsize=11, fontweight='bold')
ax3c.set_xticks(x_pos)
ax3c.set_xticklabels(models_info['Model'], fontsize=10)
ax3c.grid(axis='y', alpha=0.3)
for i, v in enumerate(models_info['AIC']):
    ax3c.text(i, v + 5, f'{v:.1f}', ha='center', fontweight='bold', fontsize=10)

# BIC (lower is better)
ax3d.bar(x_pos, models_info['BIC'], color=['#2E86AB', '#A23B72'], 
         alpha=0.7, edgecolor='black', linewidth=1.5, width=0.5)
ax3d.set_ylabel('BIC (Lower is Better)', fontsize=11, fontweight='bold')
ax3d.set_xticks(x_pos)
ax3d.set_xticklabels(models_info['Model'], fontsize=10)
ax3d.grid(axis='y', alpha=0.3)
for i, v in enumerate(models_info['BIC']):
    ax3d.text(i, v + 5, f'{v:.1f}', ha='center', fontweight='bold', fontsize=10)

[spine.set_visible(False) for ax in [ax3a, ax3b, ax3c, ax3d] for spine in [ax.spines['top'], ax.spines['right']]]

plt.tight_layout()
plt.savefig(f'{output_dir}/Model_Performance_Metrics.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: Model_Performance_Metrics.png")

# ========== FIGURE 4: Residual Diagnostics ==========
fig4, ((ax4a, ax4b), (ax4c, ax4d)) = plt.subplots(2, 2, figsize=(14, 10))
fig4.patch.set_facecolor('white')

# Judge residuals vs fitted
resid_j = model_judge.resid
fitted_j = model_judge.fittedvalues
ax4a.scatter(fitted_j, resid_j, alpha=0.6, s=40, color='#1F77B4', edgecolors='black', linewidth=0.5)
ax4a.axhline(0, color='red', linestyle='--', linewidth=2)
ax4a.set_xlabel('Fitted Values', fontsize=11, fontweight='bold')
ax4a.set_ylabel('Residuals', fontsize=11, fontweight='bold')
ax4a.grid(alpha=0.3)
ax4a.spines['top'].set_visible(False)
ax4a.spines['right'].set_visible(False)

# Fan residuals vs fitted
resid_f = model_fan.resid
fitted_f = model_fan.fittedvalues
ax4b.scatter(fitted_f, resid_f, alpha=0.6, s=40, color='#D62728', edgecolors='black', linewidth=0.5)
ax4b.axhline(0, color='red', linestyle='--', linewidth=2)
ax4b.set_xlabel('Fitted Values', fontsize=11, fontweight='bold')
ax4b.set_ylabel('Residuals', fontsize=11, fontweight='bold')
ax4b.grid(alpha=0.3)
ax4b.spines['top'].set_visible(False)
ax4b.spines['right'].set_visible(False)

# Q-Q plot for Judge
stats.probplot(resid_j, dist="norm", plot=ax4c)
ax4c.get_lines()[0].set_color('#2E86AB')
ax4c.get_lines()[0].set_marker('o')
ax4c.get_lines()[0].set_markersize(4)
ax4c.get_lines()[1].set_color('red')
ax4c.get_lines()[1].set_linewidth(2)
ax4c.set_title('')
ax4c.grid(alpha=0.3)
ax4c.spines['top'].set_visible(False)
ax4c.spines['right'].set_visible(False)

# Q-Q plot for Fan
stats.probplot(resid_f, dist="norm", plot=ax4d)
ax4d.get_lines()[0].set_color('#A23B72')
ax4d.get_lines()[0].set_marker('s')
ax4d.get_lines()[0].set_markersize(4)
ax4d.get_lines()[1].set_color('red')
ax4d.get_lines()[1].set_linewidth(2)
ax4d.set_title('')
ax4d.grid(alpha=0.3)
ax4d.spines['top'].set_visible(False)
ax4d.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{output_dir}/Residual_Diagnostics.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: Residual_Diagnostics.png")

# ========== FIGURE 5: Correlation Matrix Heatmaps ==========
fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(16, 6))
fig5.patch.set_facecolor('white')

corr_judge = merged[[c for c in features if c in merged.columns] + ['Judge_Score']].corr()
corr_fan = merged[[c for c in features if c in merged.columns] + ['Vote_Share']].corr()

# Judge correlation
sns.heatmap(corr_judge, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=1.5, cbar_kws={'label': 'Correlation'}, ax=ax5a,
            vmin=-1, vmax=1, annot_kws={'fontsize': 9, 'weight': 'bold'})
ax5a.set_title('')

# Fan correlation
sns.heatmap(corr_fan, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=1.5, cbar_kws={'label': 'Correlation'}, ax=ax5b,
            vmin=-1, vmax=1, annot_kws={'fontsize': 9, 'weight': 'bold'})
ax5b.set_title('')

plt.tight_layout()
plt.savefig(f'{output_dir}/Correlation_Matrices.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: Correlation_Matrices.png")

# ========== FIGURE 6: Feature Importance via Standardized Coefficients ==========
fig6, ax6 = plt.subplots(figsize=(14, 8))
fig6.patch.set_facecolor('white')

coef_data = pd.DataFrame({
    'Feature': model_judge.params.index[1:],
    'Judge_Coef': np.abs(model_judge.params.values[1:]),
    'Fan_Coef': np.abs(model_fan.params.values[1:])
})
coef_data['Max_Importance'] = coef_data[['Judge_Coef', 'Fan_Coef']].max(axis=1)
coef_data = coef_data.sort_values('Max_Importance', ascending=False)

x = np.arange(len(coef_data))
width = 0.35

ax6.bar(x - width/2, coef_data['Judge_Coef'], width, label='Judge Path', 
        color='#1F77B4', alpha=0.75, edgecolor='black', linewidth=1.2)
ax6.bar(x + width/2, coef_data['Fan_Coef'], width, label='Fan Path', 
        color='#D62728', alpha=0.75, edgecolor='black', linewidth=1.2)

ax6.set_ylabel('Absolute Standardized Coefficient (Importance)', fontsize=12, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(coef_data['Feature'], rotation=45, ha='right', fontsize=10)
ax6.legend(loc='upper right', fontsize=11, framealpha=0.95, edgecolor='black')
ax6.grid(axis='y', alpha=0.3, linestyle='--')
ax6.spines['top'].set_visible(False)
ax6.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{output_dir}/Feature_Importance_Comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: Feature_Importance_Comparison.png")

# ========== FIGURE 7: Distribution Comparison ==========
fig7, ((ax7a, ax7b), (ax7c, ax7d)) = plt.subplots(2, 2, figsize=(14, 10))
fig7.patch.set_facecolor('white')

# Judge Score Distribution
ax7a.hist(merged['Judge_Score'], bins=30, color='#1F77B4', alpha=0.6, edgecolor='black', linewidth=1.2)
ax7a.set_xlabel('Judge Score (Standardized)', fontsize=11, fontweight='bold')
ax7a.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax7a.axvline(merged['Judge_Score'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={merged["Judge_Score"].mean():.2f}')
ax7a.legend(fontsize=10)
ax7a.grid(alpha=0.3, axis='y')
ax7a.spines['top'].set_visible(False)
ax7a.spines['right'].set_visible(False)

# Fan Vote Distribution
ax7b.hist(merged['Vote_Share'], bins=30, color='#D62728', alpha=0.6, edgecolor='black', linewidth=1.2)
ax7b.set_xlabel('Vote Share (Standardized)', fontsize=11, fontweight='bold')
ax7b.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax7b.axvline(merged['Vote_Share'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={merged["Vote_Share"].mean():.2f}')
ax7b.legend(fontsize=10)
ax7b.grid(alpha=0.3, axis='y')
ax7b.spines['top'].set_visible(False)
ax7b.spines['right'].set_visible(False)

# Age Distribution
ax7c.hist(merged['Age_Gauss'], bins=30, color='#E8995E', alpha=0.6, edgecolor='black', linewidth=1.2)
ax7c.set_xlabel('Age Gaussian Membership', fontsize=11, fontweight='bold')
ax7c.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax7c.grid(alpha=0.3, axis='y')
ax7c.spines['top'].set_visible(False)
ax7c.spines['right'].set_visible(False)

# Pro Power Distribution
ax7d.hist(merged['Pro_Power'], bins=30, color='#2CA02C', alpha=0.6, edgecolor='black', linewidth=1.2)
ax7d.set_xlabel('Pro Power (Standardized)', fontsize=11, fontweight='bold')
ax7d.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax7d.grid(alpha=0.3, axis='y')
ax7d.spines['top'].set_visible(False)
ax7d.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{output_dir}/Distribution_Comparisons.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: Distribution_Comparisons.png")

# ========== COMBINED: Comprehensive Summary Panel ==========
fig_main = plt.figure(figsize=(18, 12))
gs = GridSpec(3, 3, figure=fig_main, hspace=0.35, wspace=0.3)
fig_main.patch.set_facecolor('white')

# Panel 1: Coefficients comparison
ax_m1 = fig_main.add_subplot(gs[0, :2])
judge_coef = model_judge.params[1:]
fan_coef = model_fan.params[1:]
x_m = np.arange(len(judge_coef))
ax_m1.barh(x_m - 0.2, judge_coef.values, 0.4, label='Judge', color='#1F77B4', alpha=0.7, edgecolor='black')
ax_m1.barh(x_m + 0.2, fan_coef.values, 0.4, label='Fan', color='#D62728', alpha=0.7, edgecolor='black')
ax_m1.axvline(0, color='red', linestyle='--', linewidth=1.5)
ax_m1.set_yticks(x_m)
ax_m1.set_yticklabels(judge_coef.index, fontsize=9)
ax_m1.set_xlabel('Standardized Beta', fontsize=10, fontweight='bold')
ax_m1.legend(loc='lower right', fontsize=9)
ax_m1.grid(axis='x', alpha=0.3)
ax_m1.spines['top'].set_visible(False)
ax_m1.spines['right'].set_visible(False)

# Panel 2: Model metrics
ax_m2 = fig_main.add_subplot(gs[0, 2])
metrics_text = f'''Model Metrics
━━━━━━━━━━━━━━━━
Judge Path:
  R² = {model_judge.rsquared:.4f}
  Adj R² = {model_judge.rsquared_adj:.4f}
  AIC = {model_judge.aic:.1f}

Fan Path:
  R² = {model_fan.rsquared:.4f}
  Adj R² = {model_fan.rsquared_adj:.4f}
  AIC = {model_fan.aic:.1f}'''
ax_m2.text(0.05, 0.95, metrics_text, transform=ax_m2.transAxes, fontsize=9,
          verticalalignment='top', fontfamily='monospace',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
ax_m2.axis('off')

# Panel 3: Residuals - Judge
ax_m3 = fig_main.add_subplot(gs[1, 0])
ax_m3.scatter(fitted_j, resid_j, alpha=0.4, s=30, color='#1F77B4', edgecolors='black', linewidth=0.3)
ax_m3.axhline(0, color='red', linestyle='--', linewidth=1.5)
ax_m3.set_xlabel('Fitted', fontsize=9, fontweight='bold')
ax_m3.set_ylabel('Residuals', fontsize=9, fontweight='bold')
ax_m3.grid(alpha=0.3)
ax_m3.spines['top'].set_visible(False)
ax_m3.spines['right'].set_visible(False)

# Panel 4: Residuals - Fan
ax_m4 = fig_main.add_subplot(gs[1, 1])
ax_m4.scatter(fitted_f, resid_f, alpha=0.4, s=30, color='#D62728', edgecolors='black', linewidth=0.3)
ax_m4.axhline(0, color='red', linestyle='--', linewidth=1.5)
ax_m4.set_xlabel('Fitted', fontsize=9, fontweight='bold')
ax_m4.set_ylabel('Residuals', fontsize=9, fontweight='bold')
ax_m4.grid(alpha=0.3)
ax_m4.spines['top'].set_visible(False)
ax_m4.spines['right'].set_visible(False)

# Panel 5: Feature Importance
ax_m5 = fig_main.add_subplot(gs[1, 2])
importance = coef_data.head(5).sort_values('Max_Importance')
y_imp = np.arange(len(importance))
ax_m5.barh(y_imp, importance['Max_Importance'], color='#E8995E', alpha=0.7, edgecolor='black')
ax_m5.set_yticks(y_imp)
ax_m5.set_yticklabels(importance['Feature'], fontsize=8)
ax_m5.set_xlabel('Importance', fontsize=9, fontweight='bold')
ax_m5.grid(axis='x', alpha=0.3)
ax_m5.spines['top'].set_visible(False)
ax_m5.spines['right'].set_visible(False)

# Panel 6: Judge distribution
ax_m6 = fig_main.add_subplot(gs[2, 0])
ax_m6.hist(merged['Judge_Score'], bins=25, color='#1F77B4', alpha=0.6, edgecolor='black')
ax_m6.set_xlabel('Judge Score', fontsize=9, fontweight='bold')
ax_m6.set_ylabel('Count', fontsize=9, fontweight='bold')
ax_m6.grid(alpha=0.3, axis='y')
ax_m6.spines['top'].set_visible(False)
ax_m6.spines['right'].set_visible(False)

# Panel 7: Fan distribution
ax_m7 = fig_main.add_subplot(gs[2, 1])
ax_m7.hist(merged['Vote_Share'], bins=25, color='#D62728', alpha=0.6, edgecolor='black')
ax_m7.set_xlabel('Vote Share', fontsize=9, fontweight='bold')
ax_m7.set_ylabel('Count', fontsize=9, fontweight='bold')
ax_m7.grid(alpha=0.3, axis='y')
ax_m7.spines['top'].set_visible(False)
ax_m7.spines['right'].set_visible(False)

# Panel 8: Summary text
ax_m8 = fig_main.add_subplot(gs[2, 2])
summary_text = f'''Analysis Summary
━━━━━━━━━━━━━━━━━
Sample Size: {len(merged)}
Features: {len(features)}

Key Insight:
Both paths show
significant effects.
Compare magnitudes
and significance for
decision insights.
'''
ax_m8.text(0.05, 0.95, summary_text, transform=ax_m8.transAxes, fontsize=9,
          verticalalignment='top', fontfamily='monospace',
          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))
ax_m8.axis('off')

plt.savefig(f'{output_dir}/Comprehensive_Summary_Panel.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: Comprehensive_Summary_Panel.png")

# ========== SAVE INDIVIDUAL PANELS FROM COMPREHENSIVE SUMMARY ==========

# Panel 1: Coefficients Comparison (Standalone)
fig_p1 = plt.figure(figsize=(14, 6))
fig_p1.patch.set_facecolor('white')
ax_p1 = fig_p1.add_subplot(111)

judge_coef = model_judge.params[1:]
fan_coef = model_fan.params[1:]
x_m = np.arange(len(judge_coef))
ax_p1.barh(x_m - 0.2, judge_coef.values, 0.4, label='Judge', color='#1F77B4', alpha=0.7, edgecolor='black', linewidth=1.2)
ax_p1.barh(x_m + 0.2, fan_coef.values, 0.4, label='Fan', color='#D62728', alpha=0.7, edgecolor='black', linewidth=1.2)
ax_p1.axvline(0, color='red', linestyle='--', linewidth=2)
ax_p1.set_yticks(x_m)
ax_p1.set_yticklabels(judge_coef.index, fontsize=11)
ax_p1.set_xlabel('Standardized Coefficient (Beta)', fontsize=12, fontweight='bold')
ax_p1.legend(loc='lower right', fontsize=11, framealpha=0.95, edgecolor='black')
ax_p1.grid(axis='x', alpha=0.3)
ax_p1.spines['top'].set_visible(False)
ax_p1.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{output_dir}/Panel_1_Coefficients_Comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: Panel_1_Coefficients_Comparison.png")

# Panel 3: Residuals - Judge (Standalone)
fig_p3 = plt.figure(figsize=(10, 7))
fig_p3.patch.set_facecolor('white')
ax_p3 = fig_p3.add_subplot(111)

ax_p3.scatter(fitted_j, resid_j, alpha=0.6, s=60, color='#1F77B4', edgecolors='black', linewidth=0.8)
ax_p3.axhline(0, color='red', linestyle='--', linewidth=2.5)
ax_p3.set_xlabel('Fitted Values', fontsize=12, fontweight='bold')
ax_p3.set_ylabel('Residuals', fontsize=12, fontweight='bold')
ax_p3.grid(alpha=0.3)
ax_p3.spines['top'].set_visible(False)
ax_p3.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{output_dir}/Panel_3_Judge_Residuals.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: Panel_3_Judge_Residuals.png")

# Panel 4: Residuals - Fan (Standalone)
fig_p4 = plt.figure(figsize=(10, 7))
fig_p4.patch.set_facecolor('white')
ax_p4 = fig_p4.add_subplot(111)

ax_p4.scatter(fitted_f, resid_f, alpha=0.6, s=60, color='#D62728', edgecolors='black', linewidth=0.8)
ax_p4.axhline(0, color='red', linestyle='--', linewidth=2.5)
ax_p4.set_xlabel('Fitted Values', fontsize=12, fontweight='bold')
ax_p4.set_ylabel('Residuals', fontsize=12, fontweight='bold')
ax_p4.grid(alpha=0.3)
ax_p4.spines['top'].set_visible(False)
ax_p4.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{output_dir}/Panel_4_Fan_Residuals.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: Panel_4_Fan_Residuals.png")

# Panel 5: Feature Importance (Standalone)
fig_p5 = plt.figure(figsize=(12, 8))
fig_p5.patch.set_facecolor('white')
ax_p5 = fig_p5.add_subplot(111)

importance = coef_data.head(8).sort_values('Max_Importance')
y_imp = np.arange(len(importance))
ax_p5.barh(y_imp, importance['Max_Importance'], color='#E8995E', alpha=0.75, edgecolor='black', linewidth=1.2)
ax_p5.set_yticks(y_imp)
ax_p5.set_yticklabels(importance['Feature'], fontsize=11)
ax_p5.set_xlabel('Absolute Standardized Coefficient', fontsize=12, fontweight='bold')
ax_p5.grid(axis='x', alpha=0.3)
ax_p5.spines['top'].set_visible(False)
ax_p5.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{output_dir}/Panel_5_Feature_Importance_Top8.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: Panel_5_Feature_Importance_Top8.png")

# Panel 6: Judge Score Distribution (Standalone)
fig_p6 = plt.figure(figsize=(11, 7))
fig_p6.patch.set_facecolor('white')
ax_p6 = fig_p6.add_subplot(111)

ax_p6.hist(merged['Judge_Score'], bins=30, color='#1F77B4', alpha=0.65, edgecolor='black', linewidth=1.2)
ax_p6.axvline(merged['Judge_Score'].mean(), color='red', linestyle='--', linewidth=2.5, 
              label=f'Mean = {merged["Judge_Score"].mean():.3f}')
ax_p6.axvline(merged['Judge_Score'].median(), color='green', linestyle='-.', linewidth=2.5, 
              label=f'Median = {merged["Judge_Score"].median():.3f}')
ax_p6.set_xlabel('Judge Score (Standardized)', fontsize=12, fontweight='bold')
ax_p6.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax_p6.legend(fontsize=11, framealpha=0.95, edgecolor='black')
ax_p6.grid(alpha=0.3, axis='y')
ax_p6.spines['top'].set_visible(False)
ax_p6.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{output_dir}/Panel_6_Judge_Score_Distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: Panel_6_Judge_Score_Distribution.png")

# Panel 7: Vote Share Distribution (Standalone)
fig_p7 = plt.figure(figsize=(11, 7))
fig_p7.patch.set_facecolor('white')
ax_p7 = fig_p7.add_subplot(111)

ax_p7.hist(merged['Vote_Share'], bins=30, color='#D62728', alpha=0.65, edgecolor='black', linewidth=1.2)
ax_p7.axvline(merged['Vote_Share'].mean(), color='red', linestyle='--', linewidth=2.5, 
              label=f'Mean = {merged["Vote_Share"].mean():.3f}')
ax_p7.axvline(merged['Vote_Share'].median(), color='green', linestyle='-.', linewidth=2.5, 
              label=f'Median = {merged["Vote_Share"].median():.3f}')
ax_p7.set_xlabel('Vote Share (Standardized)', fontsize=12, fontweight='bold')
ax_p7.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax_p7.legend(fontsize=11, framealpha=0.95, edgecolor='black')
ax_p7.grid(alpha=0.3, axis='y')
ax_p7.spines['top'].set_visible(False)
ax_p7.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{output_dir}/Panel_7_Vote_Share_Distribution.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: Panel_7_Vote_Share_Distribution.png")

# Panel 2: Model Metrics Summary (Standalone - as figure with text)
fig_p2 = plt.figure(figsize=(10, 8))
fig_p2.patch.set_facecolor('white')
ax_p2 = fig_p2.add_subplot(111)

metrics_text = f'''MODEL PERFORMANCE METRICS
════════════════════════════════════

JUDGE PATH REGRESSION
  R-squared:              {model_judge.rsquared:.6f}
  Adjusted R-squared:     {model_judge.rsquared_adj:.6f}
  AIC (Akaike):           {model_judge.aic:.2f}
  BIC (Bayesian):         {model_judge.bic:.2f}
  F-statistic:            {model_judge.fvalue:.4f}
  Prob (F-statistic):     {model_judge.f_pvalue:.6f}

FAN PATH REGRESSION
  R-squared:              {model_fan.rsquared:.6f}
  Adjusted R-squared:     {model_fan.rsquared_adj:.6f}
  AIC (Akaike):           {model_fan.aic:.2f}
  BIC (Bayesian):         {model_fan.bic:.2f}
  F-statistic:            {model_fan.fvalue:.4f}
  Prob (F-statistic):     {model_fan.f_pvalue:.6f}

SAMPLE INFORMATION
  Total Observations:     {len(merged)}
  Number of Features:     {len(features)}
  Degrees of Freedom:     {model_judge.df_resid}

════════════════════════════════════'''

ax_p2.text(0.05, 0.95, metrics_text, transform=ax_p2.transAxes, fontsize=10,
          verticalalignment='top', fontfamily='monospace',
          bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6, edgecolor='black', linewidth=2))
ax_p2.axis('off')

plt.tight_layout()
plt.savefig(f'{output_dir}/Panel_2_Model_Metrics_Summary.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: Panel_2_Model_Metrics_Summary.png")

# Panel 8: Analysis Summary (Standalone - as figure with text)
fig_p8 = plt.figure(figsize=(10, 8))
fig_p8.patch.set_facecolor('white')
ax_p8 = fig_p8.add_subplot(111)

summary_text = f'''COMPREHENSIVE ANALYSIS SUMMARY
════════════════════════════════════

DATASET CHARACTERISTICS
  Sample Size:            {len(merged)}
  Number of Features:     {len(features)}
  Judge Score Mean:       {merged["Judge_Score"].mean():.4f}
  Vote Share Mean:        {merged["Vote_Share"].mean():.4f}

DUAL-PATH REGRESSION FINDINGS
  • Both Judge and Fan paths show 
    measurable influence
  • Compare effect magnitudes and
    statistical significance
  • Judge path explains variance in
    professional scores
  • Fan path explains variance in
    public voting patterns

KEY METHODOLOGICAL INSIGHTS
  ✓ Standardized coefficients enable
    direct comparison
  ✓ Model diagnostics show residual
    distributions
  ✓ Feature importance ranked by
    absolute effect size
  ✓ Correlations reveal multicollinearity

RECOMMENDATIONS
  1. Examine residual plots for
     violations of assumptions
  2. Consider interactions between
     features
  3. Validate findings on holdout
     test set
  4. Investigate outlier observations

════════════════════════════════════'''

ax_p8.text(0.05, 0.95, summary_text, transform=ax_p8.transAxes, fontsize=9.5,
          verticalalignment='top', fontfamily='monospace',
          bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.6, edgecolor='black', linewidth=2))
ax_p8.axis('off')

plt.tight_layout()
plt.savefig(f'{output_dir}/Panel_8_Analysis_Summary.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print("✓ Saved: Panel_8_Analysis_Summary.png")

print("\n" + "="*70)
print("All visualizations saved to result_03 directory at 300 DPI")
print("="*70)
