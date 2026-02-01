import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.colors import LinearSegmentedColormap

# Professional aesthetic settings for MCM O-award quality
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 14,
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8,
})

# Create custom colormap: soft blue (0) -> deep green (0.5) -> soft red (1)
custom_cmap = LinearSegmentedColormap.from_list(
    'custom_conflict', 
    ['#C5DFF0', '#1a4d1a', '#E0A0A0'],  # very light blue, deep green, very light red
    N=256
)

# Create result directories
script_dir = os.path.dirname(os.path.abspath(__file__))
result_01_dir = os.path.join(script_dir, '../result_01')
result_02_p1_dir = os.path.join(script_dir, '../result_02_p1')

# 1. Load Data
csv_path = os.path.join(result_01_dir, 'Fan_Vote_Estimates_by_Celebrity_Week_Season.csv')
df_fan = pd.read_csv(csv_path)


def advanced_modeling(df):
    results = []
    # Filter out weeks with too few contestants to avoid statistical bias
    grouped = df.groupby(['Season', 'Week']).filter(lambda x: len(x) > 3)

    for (s, w), group in grouped.groupby(['Season', 'Week']):
        n = len(group)
        j_scores = group['Judge_Score'].values
        f_shares = group['Vote_Share'].values

        # --- Method A: Rank-Based (Ordinal Logic) ---
        # We use 'average' method for ranks to handle ties realistically
        j_rank = stats.rankdata(-j_scores, method='average')
        f_rank = stats.rankdata(-f_shares, method='average')
        v_rank_score = j_rank + f_rank
        final_rank_rank = stats.rankdata(v_rank_score, method='min')

        # --- Method B: Percentage-Based (Cardinal Logic) ---
        # Normalize judge scores to percentage to match fan shares
        j_pct = j_scores / (np.sum(j_scores) + 1e-9)
        v_pct_score = j_pct + f_shares
        final_rank_pct = stats.rankdata(-v_pct_score, method='min')

        # --- Optimized Metrics ---
        # 1. ARD: Aggregate Rank Displacement
        ard = np.mean(np.abs(final_rank_rank - final_rank_pct))

        # 2. Optimized FEI (Fan Impact Score)
        # Measuring how much the Fan Rank improves the contestant's outcome compared to Judge Rank
        rescue_rank = np.mean(j_rank - final_rank_rank)
        rescue_pct = np.mean(j_rank - final_rank_pct)

        # 3. Decision Conflict (DC): Does the eliminated person change?
        elim_rank = np.argmax(final_rank_rank)
        elim_pct = np.argmax(final_rank_pct)
        conflict = 1 if elim_rank != elim_pct else 0

        results.append({
            'Season': s,
            'Week': w,
            'ARD': ard,
            'FEI_Rank': rescue_rank,
            'FEI_Percent': rescue_pct,
            'Conflict': conflict,
            'J_Variance': np.var(j_pct)
        })

    return pd.DataFrame(results)


# 2. Execute Analysis
analysis_df = advanced_modeling(df_fan)

# 3. Enhanced Professional Visualization Suite
# ============================================================

# FIGURE 1: Comprehensive Two-Panel Analysis (300 DPI)
fig1, ax = plt.subplots(2, 2, figsize=(16, 12))
fig1.patch.set_facecolor('white')

# Panel A: Rank Displacement Over Seasons with Conflict Rate
ax1_main = ax[0, 0]
season_ard = analysis_df.groupby('Season')['ARD'].mean()
ax1_main.plot(season_ard.index, season_ard.values, color='#2C3E50', 
              linewidth=2.5, marker='o', markersize=7, label='Mean ARD', zorder=3)
ax1_main.fill_between(season_ard.index, season_ard.values, alpha=0.2, color='#3498DB')
ax1_main.set_xlabel('Season', fontsize=12, fontweight='bold')
ax1_main.set_ylabel('Aggregate Rank Displacement (ARD)', fontsize=12, fontweight='bold', color='#2C3E50')
ax1_main.tick_params(axis='y', labelcolor='#2C3E50')
ax1_main.grid(True, alpha=0.3, linestyle='--')
ax1_main.set_xlim(0.5, 34.5)

# Twin axis for Conflict Rate
ax1_twin = ax1_main.twinx()
conflict_rate = analysis_df.groupby('Season')['Conflict'].mean() * 100
ax1_twin.bar(conflict_rate.index, conflict_rate.values, alpha=0.25, 
             color='#E74C3C', width=0.8, label='Conflict Rate (%)')
ax1_twin.set_ylabel('Elimination Conflict Rate (%)', fontsize=12, fontweight='bold', color='#E74C3C')
ax1_twin.tick_params(axis='y', labelcolor='#E74C3C')

# Panel B: Fan Rescue Potential Analysis
ax[0, 1].scatter(analysis_df['J_Variance'], analysis_df['FEI_Percent'], 
                 alpha=0.5, s=60, color='#E67E22', label='Percentage-Based', edgecolors='black', linewidth=0.5)
ax[0, 1].scatter(analysis_df['J_Variance'], analysis_df['FEI_Rank'], 
                 alpha=0.5, s=60, color='#3498DB', label='Rank-Based', edgecolors='black', linewidth=0.5)
z_pct = np.polyfit(analysis_df['J_Variance'], analysis_df['FEI_Percent'], 2)
p_pct = np.poly1d(z_pct)
x_smooth = np.linspace(analysis_df['J_Variance'].min(), analysis_df['J_Variance'].max(), 100)
ax[0, 1].plot(x_smooth, p_pct(x_smooth), color='#E67E22', linewidth=2.5, linestyle='--', alpha=0.8)
z_rank = np.polyfit(analysis_df['J_Variance'], analysis_df['FEI_Rank'], 2)
p_rank = np.poly1d(z_rank)
ax[0, 1].plot(x_smooth, p_rank(x_smooth), color='#3498DB', linewidth=2.5, linestyle='--', alpha=0.8)
ax[0, 1].set_xlabel('Judge Score Variance', fontsize=12, fontweight='bold')
ax[0, 1].set_ylabel('Fan Rescue Magnitude (FEI)', fontsize=12, fontweight='bold')
ax[0, 1].legend(loc='best', framealpha=0.95, edgecolor='black')
ax[0, 1].grid(True, alpha=0.3, linestyle='--')

# Panel C: ARD Distribution by Season (Box Plot)
ax[1, 0].boxplot([analysis_df[analysis_df['Season']==s]['ARD'].values for s in range(1, 35)],
                  positions=range(1, 35), widths=0.6, patch_artist=True,
                  boxprops=dict(facecolor='#3498DB', alpha=0.7),
                  medianprops=dict(color='#E74C3C', linewidth=2),
                  whiskerprops=dict(linewidth=1.5, color='#2C3E50'),
                  capprops=dict(linewidth=1.5, color='#2C3E50'))
ax[1, 0].set_xlabel('Season', fontsize=12, fontweight='bold')
ax[1, 0].set_ylabel('Rank Displacement Distribution', fontsize=12, fontweight='bold')
ax[1, 0].set_xlim(0.5, 34.5)
ax[1, 0].grid(True, alpha=0.3, linestyle='--', axis='y')

# Panel D: Conflict Rate Heatmap
conflict_matrix = []
for s in range(1, 35):
    season_data = analysis_df[analysis_df['Season'] == s]['Conflict'].values
    conflict_matrix.append(season_data)
max_week = max(len(cm) for cm in conflict_matrix)
conflict_matrix_padded = np.full((34, max_week), np.nan)
for i, cm in enumerate(conflict_matrix):
    conflict_matrix_padded[i, :len(cm)] = cm

im = ax[1, 1].imshow(conflict_matrix_padded, cmap=custom_cmap, aspect='auto', interpolation='nearest')
ax[1, 1].set_xlabel('Week', fontsize=12, fontweight='bold')
ax[1, 1].set_ylabel('Season', fontsize=12, fontweight='bold')
cbar = plt.colorbar(im, ax=ax[1, 1], label='Conflict (1=Different Elimination)')
ax[1, 1].set_yticks(range(0, 34, 2))
ax[1, 1].set_yticklabels(range(1, 35, 2))

plt.tight_layout()
plt.savefig(os.path.join(result_02_p1_dir, 'Figure_4_Comprehensive_Q2_Analysis.png'), 
            dpi=300, bbox_inches='tight', facecolor='white')
print("? Saved: Figure_4_Comprehensive_Q2_Analysis.png")
plt.close()

# ============================================================
# FIGURE 2: System Divergence & Decision Conflict (Main)
fig2, ax2_main = plt.subplots(figsize=(14, 7))
fig2.patch.set_facecolor('white')

season_ard = analysis_df.groupby('Season')['ARD'].mean()
ax2_main.plot(season_ard.index, season_ard.values, color='#2C3E50', 
              linewidth=3, marker='o', markersize=8, label='Mean ARD', zorder=3)
ax2_main.fill_between(season_ard.index, season_ard.values, alpha=0.15, color='#3498DB')
ax2_main.set_xlabel('Season', fontsize=13, fontweight='bold')
ax2_main.set_ylabel('Aggregate Rank Displacement (ARD)', fontsize=13, fontweight='bold', color='#2C3E50')
ax2_main.tick_params(axis='y', labelcolor='#2C3E50', labelsize=11)
ax2_main.tick_params(axis='x', labelsize=11)
ax2_main.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax2_main.set_xlim(0.5, 34.5)

ax2_twin = ax2_main.twinx()
conflict_rate = analysis_df.groupby('Season')['Conflict'].mean() * 100
bars = ax2_twin.bar(conflict_rate.index, conflict_rate.values, alpha=0.25, 
                     color='#E74C3C', width=0.85, label='Conflict Rate (%)')
ax2_twin.set_ylabel('Elimination Conflict Rate (%)', fontsize=13, fontweight='bold', color='#E74C3C')
ax2_twin.tick_params(axis='y', labelcolor='#E74C3C', labelsize=11)

lines1, labels1 = ax2_main.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2_main.legend(lines1 + [bars], labels1 + labels2, loc='upper left', fontsize=11, framealpha=0.95, edgecolor='black')

plt.tight_layout()
plt.savefig(os.path.join(result_02_p1_dir, 'Figure_4.1_System_Divergence_Rank_Displacement_ARD_Conflict_Analysis.png'), 
            dpi=300, bbox_inches='tight', facecolor='white')
print("? Saved: Figure_4.1_System_Divergence_Rank_Displacement_ARD_Conflict_Analysis.png")
plt.close()

# ============================================================
# FIGURE 3: Fan Bias vs Judge Variance (Main)
fig3, ax3 = plt.subplots(figsize=(14, 7))
fig3.patch.set_facecolor('white')

ax3.scatter(analysis_df['J_Variance'], analysis_df['FEI_Percent'], 
            alpha=0.6, s=80, color='#E67E22', label='Percentage-Based FEI', 
            edgecolors='black', linewidth=0.7, zorder=2)
ax3.scatter(analysis_df['J_Variance'], analysis_df['FEI_Rank'], 
            alpha=0.6, s=80, color='#3498DB', label='Rank-Based FEI', 
            edgecolors='black', linewidth=0.7, zorder=2)

z_pct = np.polyfit(analysis_df['J_Variance'], analysis_df['FEI_Percent'], 2)
p_pct = np.poly1d(z_pct)
x_smooth = np.linspace(analysis_df['J_Variance'].min(), analysis_df['J_Variance'].max(), 100)
ax3.plot(x_smooth, p_pct(x_smooth), color='#E67E22', linewidth=3, linestyle='--', alpha=0.9, label='Pct Trend')

z_rank = np.polyfit(analysis_df['J_Variance'], analysis_df['FEI_Rank'], 2)
p_rank = np.poly1d(z_rank)
ax3.plot(x_smooth, p_rank(x_smooth), color='#3498DB', linewidth=3, linestyle='--', alpha=0.9, label='Rank Trend')

ax3.set_xlabel('Judge Score Variance (Heterogeneity)', fontsize=13, fontweight='bold')
ax3.set_ylabel('Fan Rescue Magnitude (FEI)', fontsize=13, fontweight='bold')
ax3.legend(loc='best', fontsize=11, framealpha=0.95, edgecolor='black')
ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax3.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig(os.path.join(result_02_p1_dir, 'Figure_4.2_Fan_Bias_Analysis_Judge_Variance_vs_Fan_Rescue_Impact.png'), 
            dpi=300, bbox_inches='tight', facecolor='white')
print("? Saved: Figure_4.2_Fan_Bias_Analysis_Judge_Variance_vs_Fan_Rescue_Impact.png")
plt.close()

# ============================================================
# FIGURE 4: ARD Distribution Analysis
fig4, ax4 = plt.subplots(figsize=(14, 7))
fig4.patch.set_facecolor('white')

bp = ax4.boxplot([analysis_df[analysis_df['Season']==s]['ARD'].values for s in range(1, 35)],
                  positions=range(1, 35), widths=0.65, patch_artist=True,
                  boxprops=dict(facecolor='#3498DB', alpha=0.7, linewidth=1.5),
                  medianprops=dict(color='#E74C3C', linewidth=2.5),
                  whiskerprops=dict(linewidth=1.5, color='#2C3E50'),
                  capprops=dict(linewidth=1.5, color='#2C3E50'),
                  flierprops=dict(marker='D', markerfacecolor='#E74C3C', markersize=5, alpha=0.5))

ax4.set_xlabel('Season', fontsize=13, fontweight='bold')
ax4.set_ylabel('Rank Displacement Distribution', fontsize=13, fontweight='bold')
ax4.set_xlim(0.5, 34.5)
ax4.grid(True, alpha=0.3, linestyle='--', axis='y')
ax4.tick_params(labelsize=11)

plt.tight_layout()
plt.savefig(os.path.join(result_02_p1_dir, 'Figure_4.3_ARD_Seasonality_Distribution_Pattern_Analysis.png'), 
            dpi=300, bbox_inches='tight', facecolor='white')
print("? Saved: Figure_4.3_ARD_Seasonality_Distribution_Pattern_Analysis.png")
plt.close()

# ============================================================
# FIGURE 5: Conflict Heatmap with Season-Week Resolution
fig5, ax5 = plt.subplots(figsize=(16, 8))
fig5.patch.set_facecolor('white')

conflict_matrix = []
for s in range(1, 35):
    season_data = analysis_df[analysis_df['Season'] == s]['Conflict'].values
    conflict_matrix.append(season_data)
max_week = max(len(cm) for cm in conflict_matrix)
conflict_matrix_padded = np.full((34, max_week), np.nan)
for i, cm in enumerate(conflict_matrix):
    conflict_matrix_padded[i, :len(cm)] = cm

im = ax5.imshow(conflict_matrix_padded, cmap=custom_cmap, aspect='auto', interpolation='nearest')
ax5.set_xlabel('Week Number', fontsize=13, fontweight='bold')
ax5.set_ylabel('Season', fontsize=13, fontweight='bold')
cbar = plt.colorbar(im, ax=ax5, label='Decision Conflict Indicator', pad=0.02)
cbar.ax.tick_params(labelsize=11)
ax5.set_yticks(range(0, 34, 1))
ax5.set_yticklabels(range(1, 35))
ax5.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig(os.path.join(result_02_p1_dir, 'Figure_4.4_Temporal_Conflict_Heatmap_Judge_vs_Fan_Decision_Patterns.png'), 
            dpi=300, bbox_inches='tight', facecolor='white')
print("? Saved: Figure_4.4_Temporal_Conflict_Heatmap_Judge_vs_Fan_Decision_Patterns.png")
plt.close()

print("\n? All Q2 visualizations saved to result_02/ directory with 300 DPI resolution")

# 4. Comprehensive Diagnostic Output Report
print("\n" + "="*80)
print("Q2 PART 1: JUDGE-FAN INTERACTION QUANTITATIVE ANALYSIS REPORT")
print("="*80)
print(f"\n? MACRO-LEVEL SYSTEM DIVERGENCE METRICS")
print(f"   ? Mean Rank Displacement (ARD):        {analysis_df['ARD'].mean():.4f}")
print(f"     - Std Dev: {analysis_df['ARD'].std():.4f}")
print(f"     - Min: {analysis_df['ARD'].min():.4f}, Max: {analysis_df['ARD'].max():.4f}")
print(f"\n   ? Elimination Conflict Rate:           {analysis_df['Conflict'].mean() * 100:.2f}%")
print(f"     (Proportion of weeks where judge-only vs fan-balanced rankings differ)")

print(f"\n? FAN RESCUE POTENTIAL (Impact Elasticity)")
print(f"   ? Average Fan Rescue (Rank-Based):    {analysis_df['FEI_Rank'].mean():.4f}")
print(f"     - Interpretation: Fan votes shift rankings by {analysis_df['FEI_Rank'].mean():.2f} positions on average")
print(f"\n   ? Average Fan Rescue (Percentage):    {analysis_df['FEI_Percent'].mean():.4f}")
print(f"     - Interpretation: Fan votes provide {analysis_df['FEI_Percent'].mean():.2f} strength units")

print(f"\n??  JUDGE SCORE VARIANCE ELASTICITY")
high_var_threshold = analysis_df['J_Variance'].median()
high_var = analysis_df[analysis_df['J_Variance'] > high_var_threshold]
low_var = analysis_df[analysis_df['J_Variance'] <= high_var_threshold]
print(f"   ? High Variance Seasons (Disagreement > Median):")
print(f"     - Rescue Magnitude (Pct): {high_var['FEI_Percent'].mean():.4f}")
print(f"     - Rescue Magnitude (Rank): {high_var['FEI_Rank'].mean():.4f}")
print(f"\n   ? Low Variance Seasons (Consensus):")
print(f"     - Rescue Magnitude (Pct): {low_var['FEI_Percent'].mean():.4f}")
print(f"     - Rescue Magnitude (Rank): {low_var['FEI_Rank'].mean():.4f}")
print(f"\n   ? Elasticity Ratio (High/Low Variance): {high_var['FEI_Percent'].mean() / (low_var['FEI_Percent'].mean() + 1e-9):.3f}x")

print(f"\n? SEASONAL TRENDS")
print(f"   ? Most Volatile Season (Highest ARD): Season {analysis_df.groupby('Season')['ARD'].mean().idxmax()}")
print(f"     - ARD Value: {analysis_df.groupby('Season')['ARD'].mean().max():.4f}")
print(f"\n   ? Most Stable Season (Lowest ARD): Season {analysis_df.groupby('Season')['ARD'].mean().idxmin()}")
print(f"     - ARD Value: {analysis_df.groupby('Season')['ARD'].mean().min():.4f}")

print(f"\n? VISUALIZATION OUTPUTS")
print(f"   ? Figure_4_Comprehensive_Q2_Analysis.png (4-panel overview)")
print(f"   ? Figure_4.1_System_Divergence_Rank_Displacement_ARD_Conflict_Analysis.png")
print(f"   ? Figure_4.2_Fan_Bias_Analysis_Judge_Variance_vs_Fan_Rescue_Impact.png")
print(f"   ? Figure_4.3_ARD_Seasonality_Distribution_Pattern_Analysis.png")
print(f"   ? Figure_4.4_Temporal_Conflict_Heatmap_Judge_vs_Fan_Decision_Patterns.png")
print("="*80)