import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata
import os
import seaborn as sns

# Set up matplotlib style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_palette("husl")

# Create output directory
os.makedirs('../result_02_p2', exist_ok=True)

# Load data
df_fan = pd.read_csv('../result_01/Fan_Vote_Estimates_by_Celebrity_Week_Season.csv')
# Filter seasons 28-34 for empirical analysis
df_study = df_fan[df_fan['Season'] >= 28].copy()


def calculate_mri_analysis(df):
    mri_results = []

    for (season, week), group in df.groupby(['Season', 'Week']):
        if len(group) < 3: continue  # Finals week typically not applicable

        # Calculate Rank-based composite score (lower is safer, higher is riskier)
        # Note: Rank 1 is the highest score
        j_rank = rankdata(-group['Judge_Score'], method='min')
        f_rank = rankdata(-group['Vote_Share'], method='min')
        total_rank = j_rank + f_rank

        # Identify Bottom 2 (lowest combined rankings)
        bottom_indices = np.argsort(-total_rank)[:2]
        bottom_couples = group.iloc[bottom_indices]

        # Simulate judge's save: select the person with higher Judge_Score
        # saved_couple has higher score, eliminated_couple has lower score
        if bottom_couples.iloc[0]['Judge_Score'] >= bottom_couples.iloc[1]['Judge_Score']:
            saved = bottom_couples.iloc[0]
            elim = bottom_couples.iloc[1]
        else:
            saved = bottom_couples.iloc[1]
            elim = bottom_couples.iloc[0]

        # Calculate MRI (normalized denominator)
        j_range = group['Judge_Score'].max() - group['Judge_Score'].min()
        if j_range == 0: j_range = 1  # Prevent division by zero

        mri = (saved['Judge_Score'] - elim['Judge_Score']) / j_range

        mri_results.append({
            'Season': season,
            'Week': week,
            'MRI': mri,
            'J_Saved': saved['Judge_Score'],
            'J_Elim': elim['Judge_Score']
        })

    return pd.DataFrame(mri_results)


mri_df = calculate_mri_analysis(df_study)

# Visualize MRI trends - Enhanced version
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
fig_title = 'Meritocratic Return Index (MRI) Analysis: Seasons 28-34'
# No suptitle in the figure

# Main plot: MRI bar chart
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(mri_df)))
bars = ax1.bar(range(len(mri_df)), mri_df['MRI'], 
               color=colors, alpha=0.85, edgecolor='navy', linewidth=1.2)

# Add average line
avg_mri = mri_df['MRI'].mean()
ax1.axhline(y=avg_mri, color='crimson', linestyle='--', linewidth=2.5, 
            label=f'Average MRI: {avg_mri:.3f}', alpha=0.8)

# Highlight highest and lowest MRI values
max_idx = mri_df['MRI'].idxmax()
min_idx = mri_df['MRI'].idxmin()
ax1.bar(max_idx, mri_df.loc[max_idx, 'MRI'], 
        color='gold', alpha=0.95, edgecolor='darkorange', linewidth=2)
ax1.bar(min_idx, mri_df.loc[min_idx, 'MRI'], 
        color='lightcoral', alpha=0.95, edgecolor='darkred', linewidth=2)

# Add value labels (show subset to avoid crowding)
for i in range(0, len(mri_df), max(1, len(mri_df)//15)):
    ax1.text(i, mri_df.iloc[i]['MRI'] + 0.01, f'{mri_df.iloc[i]["MRI"]:.3f}', 
             ha='center', va='bottom', fontsize=8, fontweight='bold', alpha=0.7)

ax1.set_xlabel('Elimination Events (Chronological Order)', fontsize=13, fontweight='bold')
ax1.set_ylabel('MRI Value (Merit Correction Strength)', fontsize=13, fontweight='bold')
ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax1.set_xlim(-1, len(mri_df))

# Style axes
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_linewidth(1.5)
ax1.spines['bottom'].set_linewidth(1.5)

# Subplot: MRI distribution histogram
ax2.hist(mri_df['MRI'], bins=20, color='steelblue', alpha=0.7, 
         edgecolor='black', linewidth=1.2)
ax2.axvline(x=avg_mri, color='crimson', linestyle='--', linewidth=2.5, 
            label=f'Mean: {avg_mri:.3f}')
ax2.axvline(x=mri_df['MRI'].median(), color='orange', linestyle=':', linewidth=2.5, 
            label=f'Median: {mri_df["MRI"].median():.3f}')

ax2.set_xlabel('MRI Value', fontsize=12, fontweight='bold')
ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax2.set_title('')  # Remove any title
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)

# Style axes
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_linewidth(1.5)
ax2.spines['bottom'].set_linewidth(1.5)

# Save combined figure
plt.tight_layout()
fig_filename = fig_title.replace(': ', '_').replace(' ', '_').lower()
plt.savefig(f'../result_02_p2/{fig_filename}.png', dpi=300, bbox_inches='tight')

# Save individual subplots
fig1, ax1_single = plt.subplots(figsize=(14, 6))
colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(mri_df)))
ax1_single.bar(range(len(mri_df)), mri_df['MRI'], 
               color=colors, alpha=0.85, edgecolor='navy', linewidth=1.2)
ax1_single.axhline(y=avg_mri, color='crimson', linestyle='--', linewidth=2.5, 
                   label=f'Average MRI: {avg_mri:.3f}', alpha=0.8)
ax1_single.bar(max_idx, mri_df.loc[max_idx, 'MRI'], 
               color='gold', alpha=0.95, edgecolor='darkorange', linewidth=2)
ax1_single.bar(min_idx, mri_df.loc[min_idx, 'MRI'], 
               color='lightcoral', alpha=0.95, edgecolor='darkred', linewidth=2)
for i in range(0, len(mri_df), max(1, len(mri_df)//15)):
    ax1_single.text(i, mri_df.iloc[i]['MRI'] + 0.01, f'{mri_df.iloc[i]["MRI"]:.3f}', 
                    ha='center', va='bottom', fontsize=8, fontweight='bold', alpha=0.7)
ax1_single.set_xlabel('Elimination Events (Chronological Order)', fontsize=13, fontweight='bold')
ax1_single.set_ylabel('MRI Value (Merit Correction Strength)', fontsize=13, fontweight='bold')
ax1_single.legend(loc='upper right', fontsize=11, framealpha=0.9)
ax1_single.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax1_single.spines['top'].set_visible(False)
ax1_single.spines['right'].set_visible(False)
ax1_single.spines['left'].set_linewidth(1.5)
ax1_single.spines['bottom'].set_linewidth(1.5)
plt.tight_layout()
plt.savefig('../result_02_p2/MRI_Bar_Chart.png', dpi=300, bbox_inches='tight')
plt.show()

# Save distribution histogram separately
fig2_dist, ax2_single = plt.subplots(figsize=(10, 6))
ax2_single.hist(mri_df['MRI'], bins=20, color='steelblue', alpha=0.7, 
                edgecolor='black', linewidth=1.2)
ax2_single.axvline(x=avg_mri, color='crimson', linestyle='--', linewidth=2.5, 
                   label=f'Mean: {avg_mri:.3f}')
ax2_single.axvline(x=mri_df['MRI'].median(), color='orange', linestyle=':', linewidth=2.5, 
                   label=f'Median: {mri_df["MRI"].median():.3f}')
ax2_single.set_xlabel('MRI Value', fontsize=12, fontweight='bold')
ax2_single.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax2_single.set_title('')  # Remove any title
ax2_single.legend(loc='upper right', fontsize=10)
ax2_single.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax2_single.spines['top'].set_visible(False)
ax2_single.spines['right'].set_visible(False)
ax2_single.spines['left'].set_linewidth(1.5)
ax2_single.spines['bottom'].set_linewidth(1.5)
plt.tight_layout()
plt.savefig('../result_02_p2/MRI_Distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional analysis plot: MRI trends by season
fig3, ax3 = plt.subplots(figsize=(14, 7))
fig3_title = 'MRI Trends Across Seasons 28-34'

mri_df['Season_Week'] = mri_df['Season'].astype(str) + '-W' + mri_df['Week'].astype(str)
seasons = mri_df['Season'].unique()
season_colors = plt.cm.tab10(np.linspace(0, 1, len(seasons)))

for idx, season in enumerate(sorted(seasons)):
    season_data = mri_df[mri_df['Season'] == season]
    ax3.plot(season_data.index, season_data['MRI'], 
             marker='o', markersize=8, linewidth=2.5, 
             label=f'Season {int(season)}', color=season_colors[idx],
             alpha=0.8)

ax3.axhline(y=avg_mri, color='crimson', linestyle='--', linewidth=2, 
            label=f'Overall Average: {avg_mri:.3f}', alpha=0.7)

ax3.set_xlabel('Elimination Event Index', fontsize=13, fontweight='bold')
ax3.set_ylabel('MRI Value', fontsize=13, fontweight='bold')
ax3.legend(loc='best', fontsize=10, ncol=2, framealpha=0.9)
ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

# Style axes
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_linewidth(1.5)
ax3.spines['bottom'].set_linewidth(1.5)

# Save trends plot
plt.tight_layout()
fig3_filename = fig3_title.replace(' ', '_').lower()
plt.savefig(f'../result_02_p2/{fig3_filename}.png', dpi=300, bbox_inches='tight')
plt.show()

# Statistical Summary
summary_stats = {
    'Average MRI': mri_df['MRI'].mean(),
    'Median MRI': mri_df['MRI'].median(),
    'Std Dev': mri_df['MRI'].std(),
    'Min MRI': mri_df['MRI'].min(),
    'Max MRI': mri_df['MRI'].max(),
    'Total Events': len(mri_df)
}

print("\n" + "="*60)
print("MERITOCRATIC RETURN INDEX (MRI) STATISTICAL SUMMARY")
print("="*60)
for key, value in summary_stats.items():
    print(f"{key:.<40} {value:.4f}")
print("="*60)

# Save statistical data
mri_df.to_csv('../result_02_p2/MRI_Analysis_Data.csv', index=False)
pd.DataFrame([summary_stats]).to_csv('../result_02_p2/MRI_Summary_Statistics.csv', index=False)

print(f"\n✓ Plots saved to result_02_p2 folder")
print(f"  - meritocratic_return_index_(mri)_analysis:_seasons_28-34.png (Combined)")
print(f"  - MRI_Bar_Chart.png")
print(f"  - MRI_Distribution.png")
print(f"  - mri_trends_across_seasons_28-34.png (Separated by Season)")
print(f"  - MRI_Analysis_Data.csv")
print(f"  - MRI_Summary_Statistics.csv")