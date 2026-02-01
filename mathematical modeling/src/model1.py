import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


class DWTSFinalSubmissionModel:
    def __init__(self, data_path):
        # 加载数据并清洗
        self.df = pd.read_csv(data_path)
        score_cols = [c for c in self.df.columns if 'score' in c]
        for col in score_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)

    def get_season_context(self, s_id):
        """解析赛季上下文：活跃选手、分数、淘汰者"""
        s_data = self.df[self.df['season'] == s_id].reset_index(drop=True)
        names = s_data['celebrity_name'].tolist()
        weeks_ctx = {}

        for w in range(1, 13):
            j_cols = [f'week{w}_judge{j}_score' for j in range(1, 5)]
            if not all(c in s_data.columns for c in j_cols): continue

            scores = s_data[j_cols].sum(axis=1).values
            active_mask = scores > 0
            if not any(active_mask): continue

            elim_idx = -1
            target_str = f'Eliminated Week {w}'
            for i, res in enumerate(s_data['results']):
                if isinstance(res, str) and target_str in res:
                    elim_idx = i;
                    break

            weeks_ctx[w] = {'scores': scores, 'active_mask': active_mask, 'elim_idx': elim_idx}
        return names, weeks_ctx

    def calculate_survival_v(self, f_t, j_scores, active_mask, season_id):
        """核心公式实现：根据赛季规则计算生存值 V"""
        # 转换人气强度为粉丝份额 (Softmax)
        shares = np.zeros_like(f_t)
        exp_f = np.exp(f_t)
        shares[active_mask] = exp_f[active_mask] / (exp_f[active_mask].sum() + 1e-9)

        v = np.zeros_like(f_t)
        # 阶段 A (S1-2) & 阶段 C (S28-34): 排名制
        if season_id <= 2 or season_id >= 28:
            # 排名第一记为1，因此加负号使分值越大越安全
            r_j = pd.Series(j_scores).rank(ascending=False, method='min').values
            r_s = pd.Series(shares).rank(ascending=False, method='min').values
            v[active_mask] = -(r_j[active_mask] + r_s[active_mask])
        # 阶段 B (S3-27): 百分比制
        else:
            j_pct = j_scores / (j_scores[active_mask].sum() + 1e-9)
            v[active_mask] = j_pct[active_mask] + shares[active_mask]

        return v, shares

    def objective(self, f_flat, n_con, weeks_list, weeks_ctx, season_id, lam=0.2):
        """目标函数：平滑项 + 逻辑一致性项 (Hinge Loss)"""
        f = f_flat.reshape((len(weeks_list), n_con))
        smooth_loss = np.sum(np.diff(f, axis=0) ** 2)
        elim_loss = 0
        delta = 0.05  # 判别边际

        for i, w in enumerate(weeks_list):
            ctx = weeks_ctx[w]
            if ctx['elim_idx'] == -1: continue

            v, _ = self.calculate_survival_v(f[i], ctx['scores'], ctx['active_mask'], season_id)
            v_elim = v[ctx['elim_idx']]

            if season_id < 28:
                # 确定性规则：淘汰者必须是最低生存值
                survivors = ctx['active_mask'] & (np.arange(n_con) != ctx['elim_idx'])
                elim_loss += np.sum(np.maximum(0, delta - (v[survivors] - v_elim)))
            else:
                # S28+ 规则：淘汰者必须进入 Bottom Two (倒数前二)
                v_active = np.sort(v[ctx['active_mask']])
                if len(v_active) >= 3:
                    safe_line = v_active[2]  # 倒数第三名分值
                    elim_loss += np.maximum(0, delta - (safe_line - v_elim))
        return lam * smooth_loss + elim_loss

    def solve_all(self):
        seasons = sorted(self.df['season'].unique())
        summary, cert_details = [], []

        print("Optimizing 34 Seasons based on evolving rules...")
        for s_id in seasons:
            names, weeks_ctx = self.get_season_context(s_id)
            w_list = sorted(weeks_ctx.keys())
            if not w_list: continue

            n_con = len(names)
            res = minimize(self.objective, np.zeros(len(w_list) * n_con),
                           args=(n_con, w_list, weeks_ctx, s_id), method='L-BFGS-B')

            f_opt = res.x.reshape((len(w_list), n_con))
            correct_w, total_w = 0, 0

            for i, w in enumerate(w_list):
                ctx = weeks_ctx[w]
                if ctx['elim_idx'] == -1: continue
                total_w += 1
                v, _ = self.calculate_survival_v(f_opt[i], ctx['scores'], ctx['active_mask'], s_id)
                v_inf = np.where(ctx['active_mask'], v, np.inf)

                # EAA 逻辑判断
                if s_id >= 28:
                    bottom_two = v_inf.argsort()[:2]
                    if ctx['elim_idx'] in bottom_two: correct_w += 1
                else:
                    if v_inf.argmin() == ctx['elim_idx']: correct_w += 1

                # EMC 确定性量化
                v_active = np.sort(v[ctx['active_mask']])
                idx = 2 if (s_id >= 28 and len(v_active) > 2) else 1
                thresh = v_active[idx] if len(v_active) > idx else v_active[0]

                for c_idx in range(n_con):
                    if ctx['active_mask'][c_idx]:
                        emc = np.exp(-abs(v[c_idx] - thresh) * 2.5)
                        cert_details.append({'Season': s_id, 'Week': w, 'EMC': emc})

            summary.append({'Season': s_id, 'EAA': correct_w / total_w, 'SCS': np.exp(-res.fun / (total_w + 1))})

        return pd.DataFrame(summary), pd.DataFrame(cert_details)


# --- 美赛O奖级可视化部分 ---
import os
from scipy.stats import gaussian_kde

# 获取脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, '../data/2026_MCM_Problem_C_Data.csv')
result_dir = os.path.join(script_dir, '../result')
os.makedirs(result_dir, exist_ok=True)

model = DWTSFinalSubmissionModel(data_path)
sum_df, cert_df = model.solve_all()

# 设置全局美观风格
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 15
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

# ============ 图 1: 综合性能分析 (4合1布局) ============
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.28, top=0.96, bottom=0.08, left=0.08, right=0.96)

# 1.1 EAA 准确度柱状图
ax1 = fig.add_subplot(gs[0, :])
eaa_vals = sum_df['EAA'].values
seasons = sum_df['Season'].values
colors_eaa = plt.cm.RdYlGn(eaa_vals)
bars = ax1.bar(seasons, eaa_vals, color=colors_eaa, edgecolor='#2c3e50', linewidth=1.2, alpha=0.88, width=0.75)
ax1.axhline(eaa_vals.mean(), color='#e74c3c', linestyle='--', linewidth=2.5, label=f'Mean: {eaa_vals.mean():.4f}', alpha=0.9)
ax1.fill_between(range(len(seasons)), eaa_vals.mean()-eaa_vals.std(), eaa_vals.mean()+eaa_vals.std(), 
                  alpha=0.15, color='#e74c3c', label=f'±1 Std: {eaa_vals.std():.4f}')
ax1.set_ylabel('Accuracy Rate', fontsize=12, fontweight='bold', color='#2c3e50')
ax1.set_ylim(0, 1.15)
ax1.set_xlim(0, 35)
ax1.grid(axis='y', alpha=0.25, linestyle=':', linewidth=0.8)
ax1.legend(fontsize=10, loc='lower left', framealpha=0.95, edgecolor='black', fancybox=True)
for i, v in enumerate(eaa_vals):
    if v < 0.95:
        ax1.text(seasons[i], v+0.02, f'{v:.3f}', ha='center', fontsize=8, fontweight='bold', color='#c0392b')

# 1.2 SCS 稳定性曲线
ax2 = fig.add_subplot(gs[1, :])
scs_vals = sum_df['SCS'].values
ax2.fill_between(range(len(seasons)), scs_vals, alpha=0.25, color='#3498db', label='Confidence Region')
line = ax2.plot(range(len(seasons)), scs_vals, marker='o', markersize=7, 
                linewidth=2.8, color='#2980b9', label='System Confidence Score (SCS)',
                markerfacecolor='#e67e22', markeredgewidth=2, markeredgecolor='#2980b9', zorder=5)
ax2.axhline(scs_vals.mean(), color='#16a085', linestyle='--', linewidth=2.2, 
            label=f'Mean: {scs_vals.mean():.4f}', alpha=0.85, zorder=3)
ax2.fill_between(range(len(seasons)), scs_vals.mean()-scs_vals.std(), scs_vals.mean()+scs_vals.std(),
                  alpha=0.12, color='#16a085', zorder=2)
ax2.set_ylabel('Stability Score', fontsize=12, fontweight='bold', color='#2c3e50')
ax2.set_ylim(0.5, 1.05)
ax2.set_xlim(-1, len(seasons))
ax2.grid(axis='y', alpha=0.25, linestyle=':', linewidth=0.8)
ax2.legend(fontsize=10, loc='lower right', framealpha=0.95, edgecolor='black', fancybox=True)
ax2.set_xticks(range(0, len(seasons), 3))
ax2.set_xticklabels(seasons[::3], fontsize=9)

# 1.3 EAA vs SCS 相关性分析 (散点 + 拟合)
ax3 = fig.add_subplot(gs[2, 0])
scatter = ax3.scatter(eaa_vals, scs_vals, c=seasons, cmap='viridis', s=180, 
                      alpha=0.75, edgecolors='#2c3e50', linewidth=1.5, zorder=4)
z = np.polyfit(eaa_vals, scs_vals, 2)
p = np.poly1d(z)
x_smooth = np.linspace(eaa_vals.min()-0.02, eaa_vals.max()+0.02, 300)
ax3.plot(x_smooth, p(x_smooth), "r-", linewidth=3, alpha=0.7, label='Polynomial Fit (degree=2)', zorder=3)
ax3.set_xlabel('Elimination Accuracy (EAA)', fontsize=11, fontweight='bold', color='#2c3e50')
ax3.set_ylabel('System Stability (SCS)', fontsize=11, fontweight='bold', color='#2c3e50')
ax3.grid(True, alpha=0.2, linestyle='--', linewidth=0.7)
ax3.legend(fontsize=9, loc='lower right', framealpha=0.95, edgecolor='black', fancybox=True)
cbar = plt.colorbar(scatter, ax=ax3, pad=0.02)
cbar.set_label('Season', fontsize=10, fontweight='bold')
cbar.ax.tick_params(labelsize=9)

# 1.4 性能分布 KDE + 直方图
ax4 = fig.add_subplot(gs[2, 1])
n_bins = 10
ax4.hist(eaa_vals, bins=n_bins, alpha=0.65, color='#3498db', edgecolor='#2c3e50', linewidth=1.2, label='Frequency Distribution')
ax4_twin = ax4.twinx()
kde_eaa = gaussian_kde(eaa_vals)
x_range = np.linspace(eaa_vals.min()-0.05, eaa_vals.max()+0.05, 200)
ax4_twin.plot(x_range, kde_eaa(x_range), 'r-', linewidth=3, label='KDE Density', zorder=5)
ax4.set_xlabel('Accuracy (EAA)', fontsize=11, fontweight='bold', color='#2c3e50')
ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold', color='#2c3e50')
ax4_twin.set_ylabel('Density', fontsize=11, fontweight='bold', color='#c0392b')
ax4.grid(axis='y', alpha=0.2)
ax4.tick_params(axis='y', labelcolor='#2c3e50')
ax4_twin.tick_params(axis='y', labelcolor='#c0392b')
lines1, labels1 = ax4.get_legend_handles_labels()
lines2, labels2 = ax4_twin.get_legend_handles_labels()
ax4.legend(lines1+lines2, labels1+labels2, fontsize=9, loc='upper left', framealpha=0.95, edgecolor='black', fancybox=True)

plt.savefig(os.path.join(result_dir, 'Figure_1_Comprehensive_Analysis.png'), dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✓ Saved: Figure_1_Comprehensive_Analysis.png")

# ============ 保存四个子图 ============
# 1.1 EAA 单独
fig_1_1 = plt.figure(figsize=(16, 5))
ax_1_1 = fig_1_1.add_subplot(111)
colors_eaa = plt.cm.RdYlGn(eaa_vals)
ax_1_1.bar(seasons, eaa_vals, color=colors_eaa, edgecolor='#2c3e50', linewidth=1.2, alpha=0.88, width=0.75)
ax_1_1.axhline(eaa_vals.mean(), color='#e74c3c', linestyle='--', linewidth=2.5, label=f'Mean: {eaa_vals.mean():.4f}', alpha=0.9)
ax_1_1.fill_between(range(len(seasons)), eaa_vals.mean()-eaa_vals.std(), eaa_vals.mean()+eaa_vals.std(), 
                     alpha=0.15, color='#e74c3c', label=f'±1 Std: {eaa_vals.std():.4f}')
ax_1_1.set_ylabel('Accuracy Rate', fontsize=12, fontweight='bold', color='#2c3e50')
ax_1_1.set_xlabel('Season', fontsize=11, fontweight='bold')
ax_1_1.set_ylim(0, 1.15)
ax_1_1.set_xlim(0, 35)
ax_1_1.grid(axis='y', alpha=0.25, linestyle=':', linewidth=0.8)
ax_1_1.legend(fontsize=10, loc='lower left', framealpha=0.95, edgecolor='black', fancybox=True)
for i, v in enumerate(eaa_vals):
    if v < 0.95:
        ax_1_1.text(seasons[i], v+0.02, f'{v:.3f}', ha='center', fontsize=8, fontweight='bold', color='#c0392b')
fig_1_1.tight_layout()
filename_1_1 = 'Figure_1.1_Elimination_Prediction_Accuracy_EAA_34_Seasons_Analysis.png'
plt.savefig(os.path.join(result_dir, filename_1_1), dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✓ Saved: {filename_1_1}")
plt.close(fig_1_1)

# 1.2 SCS 单独
fig_1_2 = plt.figure(figsize=(16, 5))
ax_1_2 = fig_1_2.add_subplot(111)
ax_1_2.fill_between(range(len(seasons)), scs_vals, alpha=0.25, color='#3498db', label='Confidence Region')
ax_1_2.plot(range(len(seasons)), scs_vals, marker='o', markersize=7, 
            linewidth=2.8, color='#2980b9', label='System Confidence Score (SCS)',
            markerfacecolor='#e67e22', markeredgewidth=2, markeredgecolor='#2980b9', zorder=5)
ax_1_2.axhline(scs_vals.mean(), color='#16a085', linestyle='--', linewidth=2.2, 
               label=f'Mean: {scs_vals.mean():.4f}', alpha=0.85, zorder=3)
ax_1_2.fill_between(range(len(seasons)), scs_vals.mean()-scs_vals.std(), scs_vals.mean()+scs_vals.std(),
                     alpha=0.12, color='#16a085', zorder=2)
ax_1_2.set_ylabel('Stability Score', fontsize=12, fontweight='bold', color='#2c3e50')
ax_1_2.set_xlabel('Season', fontsize=11, fontweight='bold')
ax_1_2.set_ylim(0.5, 1.05)
ax_1_2.set_xlim(-1, len(seasons))
ax_1_2.grid(axis='y', alpha=0.25, linestyle=':', linewidth=0.8)
ax_1_2.legend(fontsize=10, loc='lower right', framealpha=0.95, edgecolor='black', fancybox=True)
ax_1_2.set_xticks(range(0, len(seasons), 3))
ax_1_2.set_xticklabels(seasons[::3], fontsize=9)
fig_1_2.tight_layout()
filename_1_2 = 'Figure_1.2_System_Stability_Model_Confidence_Trajectory.png'
plt.savefig(os.path.join(result_dir, filename_1_2), dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✓ Saved: {filename_1_2}")
plt.close(fig_1_2)

# 1.3 相关性 单独
fig_1_3 = plt.figure(figsize=(8, 7))
ax_1_3 = fig_1_3.add_subplot(111)
scatter_1_3 = ax_1_3.scatter(eaa_vals, scs_vals, c=seasons, cmap='viridis', s=180, 
                             alpha=0.75, edgecolors='#2c3e50', linewidth=1.5, zorder=4)
z = np.polyfit(eaa_vals, scs_vals, 2)
p = np.poly1d(z)
x_smooth = np.linspace(eaa_vals.min()-0.02, eaa_vals.max()+0.02, 300)
ax_1_3.plot(x_smooth, p(x_smooth), "r-", linewidth=3, alpha=0.7, label='Polynomial Fit (degree=2)', zorder=3)
ax_1_3.set_xlabel('Elimination Accuracy (EAA)', fontsize=11, fontweight='bold', color='#2c3e50')
ax_1_3.set_ylabel('System Stability (SCS)', fontsize=11, fontweight='bold', color='#2c3e50')
ax_1_3.grid(True, alpha=0.2, linestyle='--', linewidth=0.7)
ax_1_3.legend(fontsize=9, loc='lower right', framealpha=0.95, edgecolor='black', fancybox=True)
cbar_1_3 = plt.colorbar(scatter_1_3, ax=ax_1_3, pad=0.02)
cbar_1_3.set_label('Season', fontsize=10, fontweight='bold')
cbar_1_3.ax.tick_params(labelsize=9)
fig_1_3.tight_layout()
filename_1_3 = 'Figure_1.3_EAA_SCS_Correlation.png'
plt.savefig(os.path.join(result_dir, filename_1_3), dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✓ Saved: {filename_1_3}")
plt.close(fig_1_3)

# 1.4 分布 单独
fig_1_4 = plt.figure(figsize=(8, 6))
ax_1_4 = fig_1_4.add_subplot(111)
kde_eaa = gaussian_kde(eaa_vals)
x_range = np.linspace(eaa_vals.min()-0.05, eaa_vals.max()+0.05, 200)
ax_1_4.hist(eaa_vals, bins=10, alpha=0.65, color='#3498db', edgecolor='#2c3e50', linewidth=1.2, label='Frequency Distribution')
ax_1_4_twin = ax_1_4.twinx()
ax_1_4_twin.plot(x_range, kde_eaa(x_range), 'r-', linewidth=3, label='KDE Density', zorder=5)
ax_1_4.set_xlabel('Accuracy (EAA)', fontsize=11, fontweight='bold', color='#2c3e50')
ax_1_4.set_ylabel('Frequency', fontsize=11, fontweight='bold', color='#2c3e50')
ax_1_4_twin.set_ylabel('Density', fontsize=11, fontweight='bold', color='#c0392b')
ax_1_4.grid(axis='y', alpha=0.2)
ax_1_4.tick_params(axis='y', labelcolor='#2c3e50')
ax_1_4_twin.tick_params(axis='y', labelcolor='#c0392b')
lines1_4, labels1_4 = ax_1_4.get_legend_handles_labels()
lines2_4, labels2_4 = ax_1_4_twin.get_legend_handles_labels()
ax_1_4.legend(lines1_4+lines2_4, labels1_4+labels2_4, fontsize=9, loc='upper left', framealpha=0.95, edgecolor='black', fancybox=True)
fig_1_4.tight_layout()
filename_1_4 = 'Figure_1.4_Distribution_Analysis.png'
plt.savefig(os.path.join(result_dir, filename_1_4), dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✓ Saved: {filename_1_4}")
plt.close(fig_1_4)

# ============ 图 2: EMC 微观确定性分布 ============
fig, ax = plt.subplots(figsize=(16, 8))
season_sample = sorted(cert_df['Season'].unique())[::3]
cert_data_parts = [cert_df[cert_df['Season']==s]['EMC'].values for s in season_sample]

parts = ax.violinplot(cert_data_parts, positions=range(len(season_sample)), widths=0.7,
                       showmeans=True, showmedians=True, showextrema=True)

# 美化小提琴图
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(plt.cm.Spectral(i/len(season_sample)))
    pc.set_alpha(0.75)
    pc.set_edgecolor('#2c3e50')
    pc.set_linewidth(1.5)

for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
    if partname in parts:
        parts[partname].set_edgecolor('#2c3e50')
        parts[partname].set_linewidth(2)

ax.set_xticks(range(len(season_sample)))
ax.set_xticklabels([f'Season {s}' for s in season_sample], fontsize=10, fontweight='bold')
ax.set_ylabel('Certainty Score (EMC)', fontsize=12, fontweight='bold', color='#2c3e50')
ax.set_xlabel('Representative Seasons', fontsize=12, fontweight='bold', color='#2c3e50')
ax.set_ylim(-0.05, 1.15)
ax.grid(axis='y', alpha=0.25, linestyle=':', linewidth=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
filename_2 = 'Figure_2_Granular_Certainty_Distribution_EMC_Micro_Level_Model_Confidence.png'
plt.savefig(os.path.join(result_dir, filename_2), dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✓ Saved: {filename_2}")

# ============ 图 3: Season×Week 热力图 ============
fig, ax = plt.subplots(figsize=(16, 10))
pivot_data = cert_df.pivot_table(values='EMC', index='Season', columns='Week', aggfunc='mean')

# 使用科学级别的色彩映射 - 改用nearest插值使边界清晰
im = ax.imshow(pivot_data.fillna(pivot_data.mean().mean()), cmap='coolwarm', aspect='auto', vmin=0, vmax=1, interpolation='nearest')

ax.set_xticks(range(len(pivot_data.columns)))
ax.set_xticklabels([f'Wk{int(c)}' for c in pivot_data.columns], fontsize=9, fontweight='bold')
ax.set_yticks(range(0, len(pivot_data), 2))
ax.set_yticklabels([f'S{int(pivot_data.index[i])}' for i in range(0, len(pivot_data), 2)], fontsize=9, fontweight='bold')

ax.set_xlabel('Week Number', fontsize=12, fontweight='bold', color='#2c3e50')
ax.set_ylabel('Season', fontsize=12, fontweight='bold', color='#2c3e50')

cbar = plt.colorbar(im, ax=ax, pad=0.02)
cbar.set_label('Mean EMC Score', fontsize=11, fontweight='bold')
cbar.ax.tick_params(labelsize=9)

# 添加网格线使读取更清晰
ax.set_xticks(np.arange(-.5, len(pivot_data.columns), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(pivot_data), 1), minor=True)
ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.2)

plt.tight_layout()
filename_3 = 'Figure_3_Certainty_Heatmap_Season_Week_Analysis_Mean_EMC_Score_Distribution.png'
plt.savefig(os.path.join(result_dir, filename_3), dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f"✓ Saved: {filename_3}")

# ============ 图 4: 统计总结表 ============
summary_stats = {
    'Performance Metric': [
        'EAA Mean', 'EAA Std Dev', 'EAA Min', 'EAA Max',
        'SCS Mean', 'SCS Std Dev', 'SCS Min', 'SCS Max',
        'EMC Mean', 'EMC Std Dev', 'Total Seasons', 'Total Weeks Analyzed'
    ],
    'Value': [
        f'{eaa_vals.mean():.6f}',
        f'{eaa_vals.std():.6f}',
        f'{eaa_vals.min():.6f}',
        f'{eaa_vals.max():.6f}',
        f'{scs_vals.mean():.6f}',
        f'{scs_vals.std():.6f}',
        f'{scs_vals.min():.6f}',
        f'{scs_vals.max():.6f}',
        f'{cert_df["EMC"].mean():.6f}',
        f'{cert_df["EMC"].std():.6f}',
        str(len(sum_df)),
        str(len(cert_df))
    ]
}
stats_df = pd.DataFrame(summary_stats)
stats_df.to_csv(os.path.join(result_dir, 'Statistics_Summary.csv'), index=False)
print(f"✓ Saved: Statistics_Summary.csv")

# 输出模型完整报告
print("\n" + "="*80)
print("DANCING WITH THE STARS - COMPREHENSIVE MODEL REPORT")
print("="*80)
print(f"\n📊 MACRO-LEVEL METRICS (Accuracy & Stability)")
print(f"   • Elimination Alignment Accuracy (EAA):")
print(f"     - Mean: {eaa_vals.mean():.6f} ({eaa_vals.mean()*100:.2f}%)")
print(f"     - Std Dev: {eaa_vals.std():.6f}")
print(f"     - Range: [{eaa_vals.min():.6f}, {eaa_vals.max():.6f}]")
print(f"\n   • System Confidence Score (SCS):")
print(f"     - Mean: {scs_vals.mean():.6f} ({scs_vals.mean()*100:.2f}%)")
print(f"     - Std Dev: {scs_vals.std():.6f}")
print(f"     - Range: [{scs_vals.min():.6f}, {scs_vals.max():.6f}]")
print(f"\n🔬 MICRO-LEVEL METRICS (Certainty Distribution)")
print(f"   • Elimination Micro-level Certainty (EMC):")
print(f"     - Mean: {cert_df['EMC'].mean():.6f}")
print(f"     - Std Dev: {cert_df['EMC'].std():.6f}")
print(f"     - Coverage: {len(cert_df)} data points across {cert_df['Season'].nunique()} seasons")
print(f"\n📈 ANALYSIS SCOPE")
print(f"   • Total Seasons Analyzed: {len(sum_df)}")
print(f"   • Total Week-Season Combinations: {len(cert_df)}")
print(f"\n✅ VISUALIZATION OUTPUTS")
print(f"   ✓ Figure_1_Comprehensive_Analysis.png (4-panel analysis)")
print(f"   ✓ Figure_2_EMC_Distribution.png (violin plot)")
print(f"   ✓ Figure_3_Heatmap_Analysis.png (temporal heatmap)")
print(f"   ✓ Statistics_Summary.csv (numerical report)")
print("="*80 + "\n")