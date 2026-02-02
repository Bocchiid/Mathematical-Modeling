import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
import os
from matplotlib.gridspec import GridSpec

# 设置绘图风格 - O奖质量
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 10,
    'axes.linewidth': 1.2,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.8,
    'lines.linewidth': 2,
})

# 创建输出目录
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'sensitivity_result')
os.makedirs(output_dir, exist_ok=True)


# ==============================================================================
# 1. 数据加载与预处理 (Data Loading & Preprocessing)
# ==============================================================================
def load_and_prep_data():
    """
    加载并合并评委数据和粉丝数据，准备用于两个模型的测试集。
    """
    # 读取数据
    try:
        judge_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', '2026_MCM_Problem_C_Data.csv')
        vote_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'result_01', 'Fan_Vote_Estimates_by_Celebrity_Week_Season.csv')
        
        df_raw = pd.read_csv(judge_path)
        df_votes = pd.read_csv(vote_path)
    except FileNotFoundError as e:
        print(f"Error: 找不到数据文件 - {e}")
        return None

    # 1.1 清洗评委数据 (Wide to Long)
    # 我们需要保留 celebrity_age_during_season 用于 Model 3
    judge_cols = ['celebrity_name', 'celebrity_age_during_season', 'season', 'ballroom_partner',
                  'celebrity_homecountry/region']
    # 提取分数列
    score_cols = [c for c in df_raw.columns if 'judge' in c and 'score' in c]

    # 将评委分数宽表转长表
    df_melt = df_raw.melt(id_vars=judge_cols, value_vars=score_cols,
                          var_name='Week_Judge', value_name='Raw_Score')

    # 解析 Week 和 Judge 信息
    df_melt['Week'] = df_melt['Week_Judge'].str.extract(r'week(\d+)_').astype(int)
    df_melt['Raw_Score'] = pd.to_numeric(df_melt['Raw_Score'], errors='coerce')
    df_melt.dropna(subset=['Raw_Score'], inplace=True)

    # 清洗名字以进行合并
    df_melt['celebrity_name'] = df_melt['celebrity_name'].str.strip().str.lower()
    df_votes['Celebrity'] = df_votes['Celebrity'].str.strip().str.lower()

    # 聚合评委分数：计算每周的平均分和变异系数(CV) - 用于 Model 4
    df_judge_agg = df_melt.groupby(['celebrity_name', 'season', 'Week']).agg(
        Avg_Judge_Score=('Raw_Score', 'mean'),
        Std_Judge_Score=('Raw_Score', 'std'),
        Age=('celebrity_age_during_season', 'first'),
        Partner=('ballroom_partner', 'first'),
        HomeCountry=('celebrity_homecountry/region', 'first')
    ).reset_index()

    # 计算 CV (Coefficient of Variation)
    df_judge_agg['Judge_CV'] = df_judge_agg['Std_Judge_Score'] / (df_judge_agg['Avg_Judge_Score'] + 1e-9)

    # 1.2 合并粉丝数据
    # 粉丝数据列名: Season, Week, Celebrity, Vote_Share
    merged = pd.merge(df_judge_agg, df_votes,
                      left_on=['celebrity_name', 'season', 'Week'],
                      right_on=['Celebrity', 'Season', 'Week'],
                      how='inner')

    # 1.3 特征工程准备
    # 专业导师能力 (Pro_Power)
    pro_means = df_raw.groupby('ballroom_partner')['placement'].mean()
    merged['Pro_Power'] = merged['Partner'].map(lambda x: 1 / pro_means.get(x, 10))  # 倒数，排名越小能力越强

    # 是否美国 (Is_US)
    merged['Is_US'] = merged['HomeCountry'].apply(lambda x: 1 if str(x).strip() == 'United States' else 0)

    return merged


# ==============================================================================
# 2. Model 3 稳健性测试: 年龄高斯模型 (Age Gaussian Robustness)
# ==============================================================================
def run_model3_sensitivity(df):
    print("\n" + "=" * 60)
    print(">>> Running Model 3 Sensitivity Analysis (Age Parameters)...")
    print("=" * 60)

    # 参数范围
    mu_range = range(20, 41, 1)  # 20 到 40
    sigma_range = range(5, 16, 1)  # 5 到 15

    results_pvalue = np.zeros((len(sigma_range), len(mu_range)))

    # 准备回归数据 (除去NaN)
    reg_df = df.dropna(subset=['Avg_Judge_Score', 'Vote_Share', 'Age', 'Pro_Power', 'Is_US']).copy()

    # 标准化非年龄变量，方便比较系数
    for col in ['Pro_Power', 'Is_US']:
        reg_df[col] = (reg_df[col] - reg_df[col].mean()) / reg_df[col].std()

    # 循环测试
    for i, sigma in enumerate(sigma_range):
        for j, mu in enumerate(mu_range):
            # 1. 计算当前参数下的 Age_Gauss
            age_gauss = np.exp(-((reg_df['Age'] - mu) ** 2) / (2 * sigma ** 2))
            reg_df['Age_Feature'] = (age_gauss - age_gauss.mean()) / age_gauss.std()  # 标准化

            X = reg_df[['Age_Feature', 'Pro_Power', 'Is_US']]
            X = sm.add_constant(X)

            # 2. 双路径回归
            # 路径 A: 评委
            y_judge = (reg_df['Avg_Judge_Score'] - reg_df['Avg_Judge_Score'].mean()) / reg_df['Avg_Judge_Score'].std()
            model_j = sm.OLS(y_judge, X).fit()

            # 路径 B: 粉丝
            y_fan = (reg_df['Vote_Share'] - reg_df['Vote_Share'].mean()) / reg_df['Vote_Share'].std()
            model_f = sm.OLS(y_fan, X).fit()

            # 3. 提取系数与标准误
            beta_j = model_j.params['Age_Feature']
            se_j = model_j.bse['Age_Feature']
            beta_f = model_f.params['Age_Feature']
            se_f = model_f.bse['Age_Feature']

            # 4. Z-Test (检验评委系数是否显著大于粉丝系数，或者两者差异显著)
            # 这里检验差异显著性 H0: beta_j = beta_f
            z_score = (beta_j - beta_f) / np.sqrt(se_j ** 2 + se_f ** 2)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # 双尾检验

            results_pvalue[i, j] = p_value

    # --- 可视化: Contour Map - O奖质量 ---
    fig, ax = plt.subplots(figsize=(12, 9), dpi=100)
    X_grid, Y_grid = np.meshgrid(mu_range, sigma_range)
    
    # 绘制填充等高线
    cp = ax.contourf(X_grid, Y_grid, results_pvalue, levels=20, cmap='RdYlGn_r')
    cbar = plt.colorbar(cp, ax=ax)
    cbar.set_label('Z-Test P-Value (Difference Significance)', fontsize=10, fontweight='bold')

    # 标记显著性阈值线
    contour_lines = ax.contour(X_grid, Y_grid, results_pvalue, levels=[0.01, 0.05], 
                                colors='black', linestyles='--', linewidths=1.5, alpha=0.7)
    ax.clabel(contour_lines, inline=True, fmt='p=%g', fontsize=9)

    ax.set_xlabel('Peak Age (μ)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Decay Scale (σ)', fontsize=11, fontweight='bold')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Model3_Age_Robustness_Contour.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Saved: Model3_Age_Robustness_Contour.png")


# ==============================================================================
# 3. Model 4 敏感性测试: DQW 决策阈值 (Threshold Analysis)
# ==============================================================================
def calculate_jain_index(scores):
    """Jain's Fairness Index"""
    n = len(scores)
    if n == 0: return 0
    numerator = np.sum(scores) ** 2
    denominator = n * np.sum(scores ** 2)
    return numerator / denominator


def run_model4_sensitivity(df):
    print("\n" + "=" * 60)
    print(">>> Running Model 4 Sensitivity Analysis (DQW Threshold H)...")
    print("=" * 60)

    # 固定参数
    k_steepness = 20
    theta_cv = 0.20  # 争议阈值

    # 测试变量: 权重上限 H
    H_values = [0.70, 0.725, 0.75, 0.775, 0.80]

    results = []

    # 归一化原始分数 (Min-Max per week) - 模拟 Model 4 的输入
    df_proc = df.copy()
    # 简单按周归一化
    df_proc['Norm_Judge'] = df_proc.groupby(['season', 'Week'])['Avg_Judge_Score'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9))
    df_proc['Norm_Fan'] = df_proc.groupby(['season', 'Week'])['Vote_Share'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9))

    # 基准排名 (传统平均: 50/50 或 纯加法)
    df_proc['Score_Trad'] = 0.5 * df_proc['Norm_Judge'] + 0.5 * df_proc['Norm_Fan']
    df_proc['Rank_Trad'] = df_proc.groupby(['season', 'Week'])['Score_Trad'].rank(ascending=False)

    print(f"{'H-Value':<10} | {'Jain Index':<12} | {'Rank Shift Std':<15} | {'Description'}")
    print("-" * 60)

    plot_data = {'H': [], 'Fairness': [], 'Shift_Std': []}

    for H in H_values:
        # 1. 计算 DQW 权重
        # W_pro = H / (1 + exp(-k * (CV - theta)))
        # sigmoid part
        sigmoid_part = 1 / (1 + np.exp(-k_steepness * (df_proc['Judge_CV'] - theta_cv)))
        df_proc['W_pro'] = H * sigmoid_part

        # 2. 计算 DQW 最终得分
        # 当 CV 很低时，W_pro 接近 0，此时权重如何分配？
        # DQW 逻辑通常是：有争议(高CV) -> 听专家的(W_pro高)。无争议 -> 混合。
        # 这里为了敏感性测试，简化为动态加权公式:
        # Score = W_pro * Judge + (1 - W_pro) * Fan
        # 注意：W_pro 的基础值在无争议时应该回归到 0.5 还是由公式决定？
        # 根据论文逻辑，无争议时 CV低 -> sigmoid -> 0 -> W_pro -> 0。这不对，无争议不应该全听粉丝。
        # 修正逻辑：W_pro 是"额外给予评委的权重" 或者 W_pro 是"最终评委权重"。
        # 假设 W_pro 是最终评委权重。当 CV 低时，我们希望 W_pro = 0.5。
        # 修正公式用于测试：W_final = 0.5 + (H - 0.5) * sigmoid(CV)
        # 这样：CV低 -> W=0.5; CV高 -> W=H。

        w_base = 0.5
        w_dynamic = w_base + (H - w_base) * sigmoid_part

        df_proc['Score_DQW'] = w_dynamic * df_proc['Norm_Judge'] + (1 - w_dynamic) * df_proc['Norm_Fan']

        # 3. 计算新排名
        df_proc['Rank_DQW'] = df_proc.groupby(['season', 'Week'])['Score_DQW'].rank(ascending=False)

        # 4. 计算指标
        # 4.1 公平性 (Jain Index 对 Score_DQW)
        jain = calculate_jain_index(df_proc['Score_DQW'].values)

        # 4.2 稳定性 (排名波动的标准差)
        rank_shift = np.abs(df_proc['Rank_DQW'] - df_proc['Rank_Trad'])
        shift_std = rank_shift.std()

        # 记录
        desc = "Balanced"
        if H >= 0.79:
            desc = "High Correction"
        elif H >= 0.74:
            desc = "Pro Favored"

        results.append({
            'H': H,
            'Jain': jain,
            'Shift_Std': shift_std,
            'Desc': desc
        })

        plot_data['H'].append(H)
        plot_data['Fairness'].append(jain)
        plot_data['Shift_Std'].append(shift_std)

        print(f"{H:<10.3f} | {jain:<12.4f} | {shift_std:<15.4f} | {desc}")

    print("\n" + "=" * 60)
    print("Model 4 Sensitivity Analysis Complete")
    print("=" * 60)

    # --- 可视化: 合并双轴图 (Combined) ---
    fig, ax1 = plt.subplots(figsize=(12, 7), dpi=100)
    fig.patch.set_facecolor('white')

    color_fairness = '#1f77b4'  # 蓝色
    ax1.set_xlabel('Max Professional Weight (H)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Jain Fairness Index', color=color_fairness, fontsize=11, fontweight='bold')
    line1 = ax1.plot(plot_data['H'], plot_data['Fairness'], color=color_fairness, marker='o', 
                     linewidth=2.5, markersize=8, label='Fairness Index', zorder=3)
    ax1.tick_params(axis='y', labelcolor=color_fairness, labelsize=10)
    ax1.set_ylim(min(plot_data['Fairness']) * 0.98, max(plot_data['Fairness']) * 1.02)
    ax1.grid(True, alpha=0.3, linestyle=':')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax2 = ax1.twinx()
    color_volatility = '#d62728'  # 红色
    ax2.set_ylabel('Rank Volatility (Std Dev)', color=color_volatility, fontsize=11, fontweight='bold')
    line2 = ax2.plot(plot_data['H'], plot_data['Shift_Std'], color=color_volatility, marker='s', 
                     linestyle='--', linewidth=2.5, markersize=8, label='Volatility', zorder=2)
    ax2.tick_params(axis='y', labelcolor=color_volatility, labelsize=10)
    ax2.spines['top'].set_visible(False)

    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center left', fontsize=10, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/Model4_DQW_Fairness_Stability_Tradeoff.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: Model4_DQW_Fairness_Stability_Tradeoff.png (Combined)")

    # --- 可视化: 单独图1 - 公平性 (Fairness Only) ---
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    fig.patch.set_facecolor('white')
    ax.plot(plot_data['H'], plot_data['Fairness'], color='#1f77b4', marker='o', 
            linewidth=2.5, markersize=9, label='Jain Fairness Index')
    ax.fill_between(plot_data['H'], plot_data['Fairness'], alpha=0.3, color='#1f77b4')
    ax.set_xlabel('Max Professional Weight (H)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Jain Fairness Index', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Model4_DQW_Fairness_Index_Analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: Model4_DQW_Fairness_Index_Analysis.png")

    # --- 可视化: 单独图2 - 排名波动性 (Volatility Only) ---
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    fig.patch.set_facecolor('white')
    ax.plot(plot_data['H'], plot_data['Shift_Std'], color='#d62728', marker='s', 
            linestyle='--', linewidth=2.5, markersize=9, label='Rank Volatility')
    ax.fill_between(plot_data['H'], plot_data['Shift_Std'], alpha=0.3, color='#d62728')
    ax.set_xlabel('Max Professional Weight (H)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Rank Volatility (Std Dev of Shifts)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/Model4_DQW_Rank_Volatility_Analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: Model4_DQW_Rank_Volatility_Analysis.png")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("    SENSITIVITY ANALYSIS FOR DQW-SYSTEM (Model 3 & Model 4)")
    print("=" * 80)
    
    # 1. 准备数据
    df_main = load_and_prep_data()

    if df_main is not None:
        print(f"✓ Data loaded: {len(df_main)} records")
        
        # 2. 运行 Model 3 (年龄) 敏感性分析
        print("\n[1/2] Running Model 3 (Age Robustness) Sensitivity Analysis...")
        run_model3_sensitivity(df_main)

        # 3. 运行 Model 4 (DQW) 敏感性分析
        print("\n[2/2] Running Model 4 (DQW Threshold) Sensitivity Analysis...")
        run_model4_sensitivity(df_main)

        print("\n" + "=" * 80)
        print(f"✓ All sensitivity tests completed!")
        print(f"✓ Figures saved to: {output_dir}")
        print(f"✓ Total files: 5 (1 Model3 contour + 1 Model4 combined + 2 Model4 individual)")
        print("=" * 80 + "\n")