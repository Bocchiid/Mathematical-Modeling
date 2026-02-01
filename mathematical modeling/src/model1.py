import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


class DWTSComprehensiveModel:
    def __init__(self, data_path):
        # 1. 加载并清洗数据
        self.df = pd.read_csv(data_path)
        score_cols = [c for c in self.df.columns if 'score' in c]
        for col in score_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)

    def get_season_data(self, season_id):
        s_data = self.df[self.df['season'] == season_id].copy().reset_index(drop=True)
        contestants = s_data['celebrity_name'].tolist()
        weeks_context = {}

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
                    elim_idx = i
                    break

            weeks_context[w] = {'scores': scores, 'active_mask': active_mask, 'elim_idx': elim_idx}
        return contestants, weeks_context

    def calculate_survival_v(self, f_t, j_scores, active_mask, season_id):
        """计算生存值 V。逻辑：越小越危险"""
        shares = np.zeros_like(f_t)
        if any(active_mask):
            exp_f = np.exp(f_t)
            shares[active_mask] = exp_f[active_mask] / (exp_f[active_mask].sum() + 1e-9)

        v = np.zeros_like(f_t)
        if season_id <= 2 or season_id >= 28:  # 排名制阶段
            r_j = pd.Series(j_scores).rank(ascending=False, method='min').values
            r_s = pd.Series(shares).rank(ascending=False, method='min').values
            v[active_mask] = -(r_j[active_mask] + r_s[active_mask])
        else:  # 百分比制阶段
            j_pct = j_scores / (j_scores[active_mask].sum() + 1e-9)
            v[active_mask] = j_pct[active_mask] + shares[active_mask]

        return v

    def objective(self, f_flat, n_con, weeks_list, weeks_context, season_id, lam=0.2):
        f = f_flat.reshape((len(weeks_list), n_con))
        smooth_loss = np.sum(np.diff(f, axis=0) ** 2)
        elim_loss = 0
        delta = 0.05  # Hinge Loss 阈值

        for i, w in enumerate(weeks_list):
            ctx = weeks_context[w]
            if ctx['elim_idx'] == -1: continue
            v = self.calculate_survival_v(f[i], ctx['scores'], ctx['active_mask'], season_id)
            elim_v = v[ctx['elim_idx']]
            survivors_mask = ctx['active_mask'] & (np.arange(n_con) != ctx['elim_idx'])
            if any(survivors_mask):
                survivor_vs = v[survivors_mask]
                # Hinge Loss 确保淘汰事实被 100% 解释
                elim_loss += np.sum(np.maximum(0, delta - (survivor_vs - elim_v)))
        return lam * smooth_loss + elim_loss

    def solve_all(self):
        seasons = sorted(self.df['season'].unique())
        summary_results = []
        detailed_certainty = []

        print("Optimizing 34 Seasons...")
        for s_id in seasons:
            names, weeks_ctx = self.get_season_data(s_id)
            w_list = sorted(weeks_ctx.keys())
            if not w_list: continue

            n_con = len(names)
            res = minimize(self.objective, np.zeros(len(w_list) * n_con),
                           args=(n_con, w_list, weeks_ctx, s_id), method='L-BFGS-B')

            f_opt = res.x.reshape((len(w_list), n_con))

            # --- 计算 EAA 和 SCS ---
            correct_count, elim_weeks = 0, 0
            for i, w in enumerate(w_list):
                ctx = weeks_ctx[w]
                if ctx['elim_idx'] == -1: continue
                elim_weeks += 1
                v = self.calculate_survival_v(f_opt[i], ctx['scores'], ctx['active_mask'], s_id)
                v_filtered = np.where(ctx['active_mask'], v, np.inf)

                # 针对 S28+ 规则的 Bottom Two 容错
                if s_id >= 28:
                    bottom_two = v_filtered.argsort()[:2]
                    if ctx['elim_idx'] in bottom_two: correct_count += 1
                else:
                    if v_filtered.argmin() == ctx['elim_idx']: correct_count += 1

                # --- 计算颗粒化确定性 (EMC) ---
                active_vs = np.sort(v[ctx['active_mask']])
                threshold = active_vs[1] if len(active_vs) > 1 else active_vs[0]
                for c_idx in range(n_con):
                    if ctx['active_mask'][c_idx]:
                        margin = abs(v[c_idx] - threshold)
                        emc = np.exp(-margin * 3.0)
                        detailed_certainty.append({'Season': s_id, 'Week': w, 'EMC': emc})

            eaa = correct_count / elim_weeks if elim_weeks > 0 else 1.0
            scs = np.exp(-res.fun / (elim_weeks + 1))
            summary_results.append({'Season': s_id, 'EAA': eaa, 'SCS': scs})

        return pd.DataFrame(summary_results), pd.DataFrame(detailed_certainty)


# --- 运行与多维可视化 ---
if __name__ == "__main__":
    import os
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建相对路径
    data_path = os.path.join(script_dir, '../data/2026_MCM_Problem_C_Data.csv')
    result_dir = os.path.join(script_dir, '../result')
    
    # 确保result目录存在
    os.makedirs(result_dir, exist_ok=True)
    
    engine = DWTSComprehensiveModel(data_path)
    final_df, cert_df = engine.solve_all()

    # ===== 美赛级别的优化可视化 =====
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. 主要指标对比图 (EAA vs SCS) - 使用高级设计
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # 1.1 EAA 分布 - 柱状图
    ax1 = fig.add_subplot(gs[0, :])
    seasons = final_df['Season'].values
    eaa_vals = final_df['EAA'].values
    colors_eaa = plt.cm.RdYlGn(eaa_vals)
    bars = ax1.bar(seasons, eaa_vals, color=colors_eaa, edgecolor='black', linewidth=1.5, alpha=0.85)
    ax1.axhline(eaa_vals.mean(), color='navy', linestyle='--', linewidth=2.5, label=f'Mean EAA: {eaa_vals.mean():.3f}')
    ax1.axhline(eaa_vals.std(), color='red', linestyle=':', linewidth=2, alpha=0.7, label=f'Std Dev: {eaa_vals.std():.3f}')
    ax1.set_ylabel('Elimination Alignment Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Model Performance: Elimination Prediction Accuracy Across 34 Seasons', 
                   fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylim(0, 1.15)
    ax1.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.legend(fontsize=11, loc='upper left')
    ax1.set_xlabel('Season', fontsize=11, fontweight='bold')
    
    # 1.2 SCS 趋势图 - 高级曲线
    ax2 = fig.add_subplot(gs[1, :])
    scs_vals = final_df['SCS'].values
    ax2.fill_between(range(len(seasons)), scs_vals, alpha=0.3, color='#2E86AB', label='Confidence Region')
    line = ax2.plot(range(len(seasons)), scs_vals, marker='o', markersize=8, 
                    linewidth=2.5, color='#2E86AB', label='System Confidence Score', 
                    markerfacecolor='#A23B72', markeredgewidth=2, markeredgecolor='#2E86AB')
    ax2.axhline(scs_vals.mean(), color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Mean SCS: {scs_vals.mean():.3f}')
    ax2.set_ylabel('Stability & Confidence Score', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Season', fontsize=11, fontweight='bold')
    ax2.set_title('Model Stability: System Confidence Score Across Seasons', 
                   fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylim(0, 1.05)
    ax2.grid(axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.legend(fontsize=11, loc='lower right')
    ax2.set_xticks(range(0, len(seasons), 3))
    ax2.set_xticklabels(seasons[::3], rotation=0)
    
    # 1.3 EAA vs SCS 散点图 - 相关性分析
    ax3 = fig.add_subplot(gs[2, 0])
    scatter = ax3.scatter(eaa_vals, scs_vals, c=seasons, cmap='viridis', s=150, 
                         alpha=0.7, edgecolors='black', linewidth=1.5)
    # 添加拟合线
    z = np.polyfit(eaa_vals, scs_vals, 1)
    p = np.poly1d(z)
    ax3.plot(eaa_vals, p(eaa_vals), "r--", linewidth=2.5, alpha=0.8, label='Trend Line')
    ax3.set_xlabel('Elimination Alignment Accuracy (EAA)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('System Confidence Score (SCS)', fontsize=11, fontweight='bold')
    ax3.set_title('EAA vs SCS Correlation Analysis', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax3.legend(fontsize=10)
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Season', fontsize=10, fontweight='bold')
    
    # 1.4 统计分布 - KDE图
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.hist(eaa_vals, bins=8, alpha=0.6, color='#2E86AB', edgecolor='black', label='EAA Distribution')
    ax4_twin = ax4.twinx()
    from scipy.stats import gaussian_kde
    kde_eaa = gaussian_kde(eaa_vals)
    x_range = np.linspace(eaa_vals.min()-0.05, eaa_vals.max()+0.05, 200)
    ax4_twin.plot(x_range, kde_eaa(x_range), 'r-', linewidth=2.5, label='KDE Density')
    ax4.set_xlabel('EAA Value', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold', color='#2E86AB')
    ax4_twin.set_ylabel('Density', fontsize=11, fontweight='bold', color='red')
    ax4.set_title('EAA Distribution Analysis', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    ax4.tick_params(axis='y', labelcolor='#2E86AB')
    ax4_twin.tick_params(axis='y', labelcolor='red')
    
    plt.savefig(os.path.join(result_dir, 'comprehensive_model_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(result_dir, 'comprehensive_model_analysis.png')}")
    
    # 2. EMC 微观确定性分布 - 小提琴图 (高级)
    fig, ax = plt.subplots(figsize=(16, 7))
    
    # 选择有代表性的季节
    season_sample = sorted(cert_df['Season'].unique())[::3]  # 每3季选1个
    cert_df_sample = cert_df[cert_df['Season'].isin(season_sample)]
    
    parts = ax.violinplot([cert_df[cert_df['Season']==s]['EMC'].values for s in season_sample],
                           positions=range(len(season_sample)), widths=0.7,
                           showmeans=True, showmedians=True)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(plt.cm.coolwarm(i/len(season_sample)))
        pc.set_alpha(0.7)
        pc.set_edgecolor('black')
        pc.set_linewidth(1.5)
    
    ax.set_xticks(range(len(season_sample)))
    ax.set_xticklabels([f'S{s}' for s in season_sample], fontsize=11, fontweight='bold')
    ax.set_ylabel('Certainty Score (EMC)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Season', fontsize=12, fontweight='bold')
    ax.set_title('Granular Certainty Distribution (EMC): Micro-Level Model Confidence', 
                  fontsize=14, fontweight='bold', pad=15)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'emc_certainty_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(result_dir, 'emc_certainty_distribution.png')}")
    
    # 3. 热力图 - Season x Week 分析
    fig, ax = plt.subplots(figsize=(14, 10))
    
    pivot_data = cert_df.pivot_table(values='EMC', index='Season', columns='Week', aggfunc='mean')
    im = ax.imshow(pivot_data.fillna(0), cmap='Pastel1', aspect='auto', vmin=0, vmax=1, alpha=0.85)
    
    ax.set_xticks(range(len(pivot_data.columns)))
    ax.set_xticklabels(pivot_data.columns, fontsize=10)
    ax.set_yticks(range(0, len(pivot_data), 2))
    ax.set_yticklabels(pivot_data.index[::2], fontsize=10)
    
    ax.set_xlabel('Week', fontsize=12, fontweight='bold')
    ax.set_ylabel('Season', fontsize=12, fontweight='bold')
    ax.set_title('Certainty Heatmap: EMC Scores by Season and Week', 
                  fontsize=14, fontweight='bold', pad=15)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean EMC Score', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'emc_heatmap.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(result_dir, 'emc_heatmap.png')}")
    
    # 4. 导出统计摘要
    summary_stats = {
        'Metric': ['EAA Mean', 'EAA Std', 'EAA Min', 'EAA Max',
                   'SCS Mean', 'SCS Std', 'SCS Min', 'SCS Max',
                   'EMC Mean', 'EMC Std'],
        'Value': [
            eaa_vals.mean(), eaa_vals.std(), eaa_vals.min(), eaa_vals.max(),
            scs_vals.mean(), scs_vals.std(), scs_vals.min(), scs_vals.max(),
            cert_df['EMC'].mean(), cert_df['EMC'].std()
        ]
    }
    summary_df = pd.DataFrame(summary_stats)
    summary_df.to_csv(os.path.join(result_dir, 'model_statistics.csv'), index=False)
    print(f"Saved: {os.path.join(result_dir, 'model_statistics.csv')}")
    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))
    print("="*60)