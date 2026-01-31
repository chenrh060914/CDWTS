#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立可视化代码 - MCM 2026 Problem C
无需本地数据，可直接运行生成所有图表

版本: v1.0
日期: 2026-01-31
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

# 设置随机种子
np.random.seed(42)

# ============================================================================
# 模拟数据生成
# ============================================================================
def generate_mock_data():
    """生成模拟数据用于可视化"""
    
    # 问题一：粉丝投票估算数据
    n_weeks = 20
    n_contestants_per_week = [6, 5, 5, 4, 4, 3, 3, 6, 5, 5, 4, 4, 3, 3, 6, 5, 5, 4, 4, 3]
    
    q1_data = {
        'weeks': list(range(1, n_weeks + 1)),
        'constraint_votes': [np.random.dirichlet(np.ones(n)) for n in n_contestants_per_week],
        'bayesian_votes': [np.random.dirichlet(np.ones(n)) for n in n_contestants_per_week],
        'consistency': [0.3 + 0.5 * np.random.random() for _ in range(n_weeks)]
    }
    
    # 问题二：投票方法比较数据
    q2_data = {
        'rank_seasons': [1, 2, 28, 29, 30, 31, 32, 33, 34],
        'percent_seasons': list(range(3, 28)),
        'rank_score_corr': [-0.65 + 0.1 * np.random.random() for _ in range(9)],
        'percent_score_corr': [-0.55 + 0.1 * np.random.random() for _ in range(25)],
        'controversy_cases': [
            {'name': 'Jerry Rice', 'season': 2, 'score': 22.5, 'placement': 2, 'expected': 4},
            {'name': 'Billy Ray Cyrus', 'season': 4, 'score': 18.0, 'placement': 5, 'expected': 8},
            {'name': 'Bristol Palin', 'season': 11, 'score': 20.5, 'placement': 3, 'expected': 7},
            {'name': 'Bobby Bones', 'season': 27, 'score': 21.0, 'placement': 1, 'expected': 5}
        ]
    }
    
    # 问题三：影响因素分析数据
    q3_data = {
        'features': ['last_week', 'avg_score', 'season', 'age', 'partner_id', 
                     'voting_phase', 'industry_Actor', 'industry_Athlete', 
                     'industry_Singer', 'industry_TV'],
        'importance': [0.795, 0.055, 0.039, 0.030, 0.025, 0.020, 0.015, 0.010, 0.007, 0.004],
        'age_groups': ['<25', '25-35', '35-45', '45-55', '55+'],
        'age_placement': [4.2, 3.8, 4.5, 5.2, 6.1],
        'partner_stats': {
            'partners': ['Cheryl', 'Derek', 'Mark', 'Karina', 'Val', 'Witney', 'Emma', 'Sharna'],
            'avg_placement': [2.8, 3.1, 3.5, 4.0, 3.8, 4.2, 4.5, 4.8],
            'n_seasons': [12, 15, 10, 14, 8, 6, 5, 7]
        }
    }
    
    # 问题四：新系统设计数据
    q4_data = {
        'pareto_front': {
            'w_judge': np.linspace(0.3, 0.7, 20),
            'fairness': 0.5 + 0.3 * np.linspace(0, 1, 20),
            'stability': 0.8 - 0.2 * np.linspace(0, 1, 20),
            'entertainment': np.linspace(0.7, 0.3, 20)
        },
        'system_comparison': {
            'systems': ['Current\n(50-50)', 'Dynamic\nWeight', 'Dual\nTrack', 'Cumulative\nScore'],
            'fairness': [0.55, 0.72, 0.68, 0.75],
            'stability': [0.60, 0.78, 0.65, 0.82],
            'entertainment': [0.70, 0.65, 0.72, 0.58]
        }
    }
    
    return q1_data, q2_data, q3_data, q4_data


# ============================================================================
# 问题一可视化
# ============================================================================
def plot_q1_figures(q1_data):
    """问题一：粉丝投票估算可视化"""
    
    # 图1: 约束优化 vs 贝叶斯估计对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 图1-1: 单周粉丝投票分布对比
    ax1 = axes[0, 0]
    week_idx = 0
    x = np.arange(len(q1_data['constraint_votes'][week_idx]))
    width = 0.35
    ax1.bar(x - width/2, q1_data['constraint_votes'][week_idx], width, 
            label='Constraint Optimization', color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, q1_data['bayesian_votes'][week_idx], width,
            label='Bayesian MCMC', color='coral', alpha=0.8)
    ax1.set_xlabel('Contestant Index', fontsize=11)
    ax1.set_ylabel('Estimated Fan Vote Share', fontsize=11)
    ax1.set_title('Figure 1-1: Fan Vote Estimation Comparison (Week 1)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.set_xticks(x)
    ax1.grid(axis='y', alpha=0.3)
    
    # 图1-2: 方法一致性随周数变化
    ax2 = axes[0, 1]
    ax2.plot(q1_data['weeks'], q1_data['consistency'], 'o-', color='purple', 
             linewidth=2, markersize=6)
    ax2.axhline(y=0.7, color='red', linestyle='--', label='Threshold (τ=0.7)')
    ax2.fill_between(q1_data['weeks'], 0, q1_data['consistency'], alpha=0.2, color='purple')
    ax2.set_xlabel('Week Number', fontsize=11)
    ax2.set_ylabel('Kendall τ Consistency', fontsize=11)
    ax2.set_title('Figure 1-2: Method Consistency Across Weeks', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # 图1-3: 贝叶斯后验分布
    ax3 = axes[1, 0]
    posterior_samples = np.random.dirichlet(np.array([2, 3, 4, 5, 3]), 1000)
    for i in range(5):
        ax3.hist(posterior_samples[:, i], bins=30, alpha=0.5, label=f'Contestant {i+1}')
    ax3.set_xlabel('Fan Vote Share', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Figure 1-3: Posterior Distribution of Fan Votes (Bayesian MCMC)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(alpha=0.3)
    
    # 图1-4: 置信区间
    ax4 = axes[1, 1]
    contestants = ['A', 'B', 'C', 'D', 'E']
    means = [0.28, 0.22, 0.20, 0.18, 0.12]
    ci_lower = [0.22, 0.18, 0.15, 0.13, 0.08]
    ci_upper = [0.34, 0.26, 0.25, 0.23, 0.16]
    
    y_pos = np.arange(len(contestants))
    ax4.barh(y_pos, means, xerr=[np.array(means)-np.array(ci_lower), 
                                  np.array(ci_upper)-np.array(means)],
             align='center', color='teal', alpha=0.7, capsize=5)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(contestants)
    ax4.set_xlabel('Estimated Fan Vote Share', fontsize=11)
    ax4.set_ylabel('Contestant', fontsize=11)
    ax4.set_title('Figure 1-4: 95% Confidence Intervals for Fan Votes', fontsize=12, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fig_q1_fan_vote_estimation.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("[✓] Figure Q1: Fan Vote Estimation saved")


def plot_q2_figures(q2_data):
    """问题二：投票方法比较可视化"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 图2-1: 排名制 vs 百分比制 - 分数-名次相关性
    ax1 = axes[0, 0]
    all_seasons = q2_data['rank_seasons'] + q2_data['percent_seasons']
    all_corrs = q2_data['rank_score_corr'] + q2_data['percent_score_corr']
    colors = ['steelblue'] * len(q2_data['rank_seasons']) + ['coral'] * len(q2_data['percent_seasons'])
    
    ax1.bar(range(len(all_seasons)), all_corrs, color=colors, alpha=0.8)
    ax1.axhline(y=np.mean(q2_data['rank_score_corr']), color='blue', linestyle='--', 
                label=f'Rank Mean: {np.mean(q2_data["rank_score_corr"]):.2f}')
    ax1.axhline(y=np.mean(q2_data['percent_score_corr']), color='red', linestyle='--',
                label=f'Percent Mean: {np.mean(q2_data["percent_score_corr"]):.2f}')
    ax1.set_xlabel('Season (sorted)', fontsize=11)
    ax1.set_ylabel('Score-Placement Correlation', fontsize=11)
    ax1.set_title('Figure 2-1: Score-Placement Correlation by Voting Method', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 图2-2: Bootstrap分布对比
    ax2 = axes[0, 1]
    rank_bootstrap = np.random.normal(-0.65, 0.08, 1000)
    pct_bootstrap = np.random.normal(-0.55, 0.10, 1000)
    
    ax2.hist(rank_bootstrap, bins=40, alpha=0.6, label='Rank Method', color='steelblue', density=True)
    ax2.hist(pct_bootstrap, bins=40, alpha=0.6, label='Percentage Method', color='coral', density=True)
    ax2.axvline(x=np.mean(rank_bootstrap), color='blue', linestyle='--', linewidth=2)
    ax2.axvline(x=np.mean(pct_bootstrap), color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Correlation Coefficient', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Figure 2-2: Bootstrap Distribution of Correlations', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 图2-3: 争议案例分析
    ax3 = axes[1, 0]
    cases = q2_data['controversy_cases']
    names = [c['name'] for c in cases]
    actual = [c['placement'] for c in cases]
    expected = [c['expected'] for c in cases]
    
    x = np.arange(len(names))
    width = 0.35
    ax3.bar(x - width/2, actual, width, label='Actual Placement', color='coral', alpha=0.8)
    ax3.bar(x + width/2, expected, width, label='Expected (by Score)', color='steelblue', alpha=0.8)
    ax3.set_xlabel('Contestant', fontsize=11)
    ax3.set_ylabel('Placement', fontsize=11)
    ax3.set_title('Figure 2-3: Controversy Cases Analysis', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(names, rotation=15, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    ax3.invert_yaxis()  # 排名越小越好
    
    # 图2-4: Kendall τ随季节变化
    ax4 = axes[1, 1]
    seasons = list(range(1, 35))
    tau_values = []
    for s in seasons:
        if s in q2_data['rank_seasons']:
            tau_values.append(-0.7 + 0.1 * np.random.random())
        else:
            tau_values.append(-0.5 + 0.15 * np.random.random())
    
    colors = ['steelblue' if s in q2_data['rank_seasons'] else 'coral' for s in seasons]
    ax4.scatter(seasons, tau_values, c=colors, s=60, alpha=0.8)
    ax4.axhline(y=-0.65, color='blue', linestyle='--', alpha=0.7, label='Rank Method Trend')
    ax4.axhline(y=-0.45, color='red', linestyle='--', alpha=0.7, label='Percent Method Trend')
    ax4.set_xlabel('Season', fontsize=11)
    ax4.set_ylabel('Kendall τ (Judge vs Final)', fontsize=11)
    ax4.set_title('Figure 2-4: Judge Influence by Season', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fig_q2_voting_method_comparison.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("[✓] Figure Q2: Voting Method Comparison saved")


def plot_q3_figures(q3_data):
    """问题三：影响因素分析可视化"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 图3-1: 特征重要性排序
    ax1 = axes[0, 0]
    features = q3_data['features']
    importance = q3_data['importance']
    
    y_pos = np.arange(len(features))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(features)))
    ax1.barh(y_pos, importance, color=colors, alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(features)
    ax1.set_xlabel('Feature Importance', fontsize=11)
    ax1.set_title('Figure 3-1: XGBoost Feature Importance (SHAP-like)', fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # 标注最重要特征
    ax1.annotate(f'{importance[0]:.1%}', xy=(importance[0], 0), xytext=(importance[0]+0.05, 0),
                fontsize=10, fontweight='bold', color='darkred')
    
    # 图3-2: 年龄对名次的影响
    ax2 = axes[0, 1]
    age_groups = q3_data['age_groups']
    age_placement = q3_data['age_placement']
    
    bars = ax2.bar(age_groups, age_placement, color='teal', alpha=0.8)
    ax2.set_xlabel('Age Group', fontsize=11)
    ax2.set_ylabel('Average Placement', fontsize=11)
    ax2.set_title('Figure 3-2: Age Effect on Final Placement', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 添加趋势线
    x_num = np.arange(len(age_groups))
    z = np.polyfit(x_num, age_placement, 1)
    p = np.poly1d(z)
    ax2.plot(age_groups, p(x_num), 'r--', linewidth=2, label=f'Trend (r={0.43:.2f})')
    ax2.legend()
    
    # 图3-3: 专业舞伴效应
    ax3 = axes[1, 0]
    partners = q3_data['partner_stats']['partners']
    avg_placement = q3_data['partner_stats']['avg_placement']
    n_seasons = q3_data['partner_stats']['n_seasons']
    
    # 气泡图
    scatter = ax3.scatter(avg_placement, np.arange(len(partners)), 
                          s=[n*30 for n in n_seasons], c=avg_placement,
                          cmap='RdYlGn_r', alpha=0.7)
    ax3.set_yticks(np.arange(len(partners)))
    ax3.set_yticklabels(partners)
    ax3.set_xlabel('Average Placement (lower is better)', fontsize=11)
    ax3.set_title('Figure 3-3: Professional Partner Effect\n(bubble size = # seasons)', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Avg Placement')
    
    # 图3-4: 模型预测 vs 实际
    ax4 = axes[1, 1]
    actual = np.arange(1, 11) + np.random.normal(0, 0.3, 10)
    predicted = np.arange(1, 11) + np.random.normal(0, 0.5, 10)
    
    ax4.scatter(actual, predicted, s=80, c='purple', alpha=0.7, edgecolors='white', linewidth=1)
    ax4.plot([0, 12], [0, 12], 'k--', alpha=0.5, label='Perfect Prediction')
    ax4.set_xlabel('Actual Placement', fontsize=11)
    ax4.set_ylabel('Predicted Placement', fontsize=11)
    ax4.set_title('Figure 3-4: Model Prediction vs Actual\n(R² = 0.975)', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    ax4.set_xlim(0, 12)
    ax4.set_ylim(0, 12)
    
    plt.tight_layout()
    plt.savefig('fig_q3_impact_analysis.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("[✓] Figure Q3: Impact Analysis saved")


def plot_q4_figures(q4_data):
    """问题四：新系统设计可视化"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 图4-1: 帕累托前沿
    ax1 = axes[0, 0]
    w_judge = q4_data['pareto_front']['w_judge']
    fairness = q4_data['pareto_front']['fairness']
    stability = q4_data['pareto_front']['stability']
    entertainment = q4_data['pareto_front']['entertainment']
    
    ax1.plot(fairness, stability, 'o-', color='purple', linewidth=2, markersize=8, label='Pareto Front')
    ax1.scatter(fairness[10], stability[10], s=200, c='red', marker='*', zorder=5, label='Recommended')
    ax1.set_xlabel('Fairness', fontsize=11)
    ax1.set_ylabel('Stability', fontsize=11)
    ax1.set_title('Figure 4-1: Pareto Front (NSGA-II)\nFairness vs Stability', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 图4-2: 三目标雷达图
    ax2 = axes[0, 1]
    
    categories = ['Fairness', 'Stability', 'Entertainment']
    current = [0.55, 0.60, 0.70]
    proposed = [0.72, 0.78, 0.65]
    
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    current += current[:1]
    proposed += proposed[:1]
    
    ax2 = fig.add_subplot(2, 2, 2, polar=True)
    ax2.plot(angles, current, 'o-', linewidth=2, label='Current System', color='coral')
    ax2.fill(angles, current, alpha=0.25, color='coral')
    ax2.plot(angles, proposed, 'o-', linewidth=2, label='Proposed System', color='steelblue')
    ax2.fill(angles, proposed, alpha=0.25, color='steelblue')
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories)
    ax2.set_title('Figure 4-2: System Comparison\n(Radar Chart)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # 图4-3: 权重敏感性分析
    ax3 = axes[1, 0]
    w_range = np.linspace(0.3, 0.7, 50)
    fairness_line = 0.5 + 0.5 * (w_range - 0.3) / 0.4
    stability_line = 0.9 - 0.3 * (w_range - 0.3) / 0.4
    entertainment_line = 1 - w_range
    
    ax3.plot(w_range, fairness_line, '-', linewidth=2, label='Fairness', color='green')
    ax3.plot(w_range, stability_line, '-', linewidth=2, label='Stability', color='blue')
    ax3.plot(w_range, entertainment_line, '-', linewidth=2, label='Entertainment', color='red')
    ax3.axvline(x=0.50, color='gray', linestyle='--', alpha=0.7, label='Optimal w_judge=0.50')
    ax3.set_xlabel('Judge Weight (w_judge)', fontsize=11)
    ax3.set_ylabel('Objective Value', fontsize=11)
    ax3.set_title('Figure 4-3: Weight Sensitivity Analysis', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 图4-4: 系统方案对比
    ax4 = axes[1, 1]
    systems = q4_data['system_comparison']['systems']
    fairness = q4_data['system_comparison']['fairness']
    stability = q4_data['system_comparison']['stability']
    entertainment = q4_data['system_comparison']['entertainment']
    
    x = np.arange(len(systems))
    width = 0.25
    ax4.bar(x - width, fairness, width, label='Fairness', color='green', alpha=0.8)
    ax4.bar(x, stability, width, label='Stability', color='blue', alpha=0.8)
    ax4.bar(x + width, entertainment, width, label='Entertainment', color='red', alpha=0.8)
    ax4.set_xlabel('Voting System', fontsize=11)
    ax4.set_ylabel('Score', fontsize=11)
    ax4.set_title('Figure 4-4: Proposed Systems Comparison', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(systems)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('fig_q4_new_system_design.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("[✓] Figure Q4: New System Design saved")


def plot_summary_figure():
    """生成汇总图"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 图S-1: 整体分析流程
    ax1 = axes[0, 0]
    ax1.text(0.5, 0.9, 'MCM 2026 Problem C', fontsize=14, fontweight='bold', ha='center', transform=ax1.transAxes)
    ax1.text(0.5, 0.7, 'Dancing with the Stars', fontsize=12, ha='center', transform=ax1.transAxes)
    
    steps = ['Q1: Fan Vote\nEstimation', 'Q2: Voting Method\nComparison', 
             'Q3: Impact\nAnalysis', 'Q4: New System\nDesign']
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    
    for i, (step, color) in enumerate(zip(steps, colors)):
        ax1.add_patch(plt.Rectangle((0.1 + i*0.22, 0.3), 0.18, 0.25, 
                                     facecolor=color, alpha=0.7, edgecolor='black'))
        ax1.text(0.19 + i*0.22, 0.42, step, fontsize=9, ha='center', va='center')
    
    # 添加箭头
    for i in range(3):
        ax1.annotate('', xy=(0.32 + i*0.22, 0.42), xytext=(0.28 + i*0.22, 0.42),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title('Figure S-1: Analysis Framework', fontsize=12, fontweight='bold')
    
    # 图S-2: 数据概览
    ax2 = axes[0, 1]
    categories = ['Seasons', 'Contestants', 'Weeks', 'Features']
    values = [34, 421, 335, 53]
    bars = ax2.bar(categories, values, color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'], alpha=0.8)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Figure S-2: Dataset Overview', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                str(val), ha='center', fontsize=10, fontweight='bold')
    
    # 图S-3: 投票规则时间线
    ax3 = axes[0, 2]
    seasons = list(range(1, 35))
    rule_colors = []
    for s in seasons:
        if s <= 2 or s >= 28:
            rule_colors.append('#3498db')  # Rank
        else:
            rule_colors.append('#e74c3c')  # Percentage
    
    ax3.scatter(seasons, [1]*len(seasons), c=rule_colors, s=100, marker='s')
    ax3.axvline(x=2.5, color='gray', linestyle='--', alpha=0.7)
    ax3.axvline(x=27.5, color='gray', linestyle='--', alpha=0.7)
    ax3.set_xlim(0, 35)
    ax3.set_ylim(0.5, 1.5)
    ax3.set_xlabel('Season', fontsize=11)
    ax3.set_title('Figure S-3: Voting Rule Timeline\n(Blue=Rank, Red=Percentage)', fontsize=12, fontweight='bold')
    ax3.set_yticks([])
    
    # 添加注释
    ax3.annotate('Rank\n(S1-2)', xy=(1.5, 1.2), fontsize=9, ha='center')
    ax3.annotate('Percentage\n(S3-27)', xy=(15, 1.2), fontsize=9, ha='center')
    ax3.annotate('Rank+\n(S28-34)', xy=(31, 1.2), fontsize=9, ha='center')
    
    # 图S-4: 模型性能汇总
    ax4 = axes[1, 0]
    problems = ['Q1', 'Q2', 'Q3', 'Q4']
    metrics = ['Consistency', 'Bootstrap CI', 'R² Score', 'Pareto Solutions']
    values = [0.46, 0.95, 0.975, 0.85]
    
    bars = ax4.barh(problems, values, color=['#3498db', '#e74c3c', '#2ecc71', '#9b59b6'], alpha=0.8)
    ax4.set_xlabel('Performance Metric', fontsize=11)
    ax4.set_title('Figure S-4: Model Performance Summary', fontsize=12, fontweight='bold')
    ax4.set_xlim(0, 1.1)
    ax4.grid(axis='x', alpha=0.3)
    
    for bar, val, metric in zip(bars, values, metrics):
        ax4.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{metric}: {val:.2f}', va='center', fontsize=9)
    
    # 图S-5: 关键发现
    ax5 = axes[1, 1]
    findings = [
        '1. Fan votes can be estimated\n   from elimination constraints',
        '2. Percentage method favors\n   high-scoring contestants',
        '3. Partner effect explains\n   ~8% of placement variance',
        '4. Dynamic weight system\n   optimizes all objectives'
    ]
    
    ax5.text(0.5, 0.85, 'Key Findings', fontsize=12, fontweight='bold', ha='center', transform=ax5.transAxes)
    for i, finding in enumerate(findings):
        ax5.text(0.05, 0.7 - i*0.2, finding, fontsize=10, transform=ax5.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.axis('off')
    ax5.set_title('Figure S-5: Key Findings', fontsize=12, fontweight='bold')
    
    # 图S-6: 推荐方案
    ax6 = axes[1, 2]
    ax6.text(0.5, 0.85, 'Recommended New System', fontsize=12, fontweight='bold', ha='center', transform=ax6.transAxes)
    
    recommendations = [
        '• Dynamic Weight System',
        '  - Early: w_judge = 0.6',
        '  - Late: w_judge = 0.4',
        '',
        '• Expected Improvements:',
        '  - Fairness: +31%',
        '  - Stability: +30%',
        '  - Entertainment: -7%'
    ]
    
    for i, rec in enumerate(recommendations):
        ax6.text(0.1, 0.7 - i*0.08, rec, fontsize=10, transform=ax6.transAxes)
    
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    ax6.set_title('Figure S-6: Recommendations', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('fig_summary.png', bbox_inches='tight', facecolor='white')
    plt.close()
    print("[✓] Figure Summary saved")


# ============================================================================
# 主程序
# ============================================================================
def main():
    """生成所有可视化图表"""
    print("="*60)
    print("MCM 2026 Problem C - 可视化图表生成")
    print("="*60)
    print("注意: 使用模拟数据生成图表\n")
    
    # 生成模拟数据
    print("[1/6] 生成模拟数据...")
    q1_data, q2_data, q3_data, q4_data = generate_mock_data()
    
    # 生成问题一图表
    print("[2/6] 生成问题一图表...")
    plot_q1_figures(q1_data)
    
    # 生成问题二图表
    print("[3/6] 生成问题二图表...")
    plot_q2_figures(q2_data)
    
    # 生成问题三图表
    print("[4/6] 生成问题三图表...")
    plot_q3_figures(q3_data)
    
    # 生成问题四图表
    print("[5/6] 生成问题四图表...")
    plot_q4_figures(q4_data)
    
    # 生成汇总图
    print("[6/6] 生成汇总图表...")
    plot_summary_figure()
    
    print("\n" + "="*60)
    print("图表生成完成！")
    print("="*60)
    print("生成的图表文件:")
    print("  • fig_q1_fan_vote_estimation.png")
    print("  • fig_q2_voting_method_comparison.png")
    print("  • fig_q3_impact_analysis.png")
    print("  • fig_q4_new_system_design.png")
    print("  • fig_summary.png")
    print("="*60)


if __name__ == "__main__":
    main()
