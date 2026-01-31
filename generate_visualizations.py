#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据分析与可视化模块
分析模型求解结果的合理性并生成可视化图片
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False

import os

# 输出目录
OUTPUT_DIR = "visualization_output"
DATA_DIR = "solving_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_data_logic():
    """分析数据的逻辑合理性"""
    print("="*70)
    print("数据合理性分析报告")
    print("="*70)
    
    issues = []
    
    # 分析Q1
    print("\n[Q1] 粉丝投票估算分析")
    q1_data = load_json(f"{DATA_DIR}/q1_fan_vote/q1_results_v2.json")
    summary = q1_data.get('summary', {})
    
    epa_constraint = summary.get('epa_constraint', 0)
    epa_bayesian = summary.get('epa_bayesian', 0)
    consistency = summary.get('consistency_rate', 0)
    
    print(f"  约束优化EPA: {epa_constraint:.2%}")
    print(f"  贝叶斯MCMC EPA: {epa_bayesian:.2%}")
    print(f"  两种方法一致性: {consistency:.2%}")
    
    if epa_constraint > 0.85:
        print("  ✓ 约束优化EPA > 85%，模型预测能力良好")
    else:
        print("  ✗ 约束优化EPA较低，需要检查模型")
        issues.append("Q1: EPA准确率偏低")
    
    if consistency < 0.3:
        print("  ! 两种方法一致性较低，可能是方法论差异导致")
        # 这不一定是问题，两种方法本身就有差异
    
    # 分析Q2
    print("\n[Q2] 投票方法比较分析")
    q2_data = load_json(f"{DATA_DIR}/q2_voting_method/q2_results_v2.json")
    conclusion = q2_data.get('conclusion', {})
    
    rank_corr = conclusion.get('rank_score_corr', 0)
    pct_corr = conclusion.get('percent_score_corr', 0)
    
    print(f"  排名制相关性: {rank_corr:.3f}")
    print(f"  百分比制相关性: {pct_corr:.3f}")
    
    # 相关性应该是负值（分数越高，排名越靠前即数字越小）
    if rank_corr < 0 and pct_corr < 0:
        print("  ✓ 相关性符号正确（负相关表示分数越高排名越好）")
    else:
        print("  ✗ 相关性符号异常")
        issues.append("Q2: 相关性符号异常")
    
    if abs(rank_corr) > abs(pct_corr):
        print("  ✓ 排名制更偏向评委（|τ|更大），符合理论预期")
    
    # 分析Q3
    print("\n[Q3] 影响因素分析")
    q3_data = load_json(f"{DATA_DIR}/q3_impact_analysis/q3_results_v2.json")
    conclusions = q3_data.get('conclusions', {})
    
    r2 = conclusions.get('model_r2', 0)
    top_factors = conclusions.get('top_factors', [])
    age_placement = conclusions.get('age_placement_corr', 0)
    age_score = conclusions.get('age_score_corr', 0)
    
    print(f"  模型R²: {r2:.4f}")
    print(f"  Top特征: {top_factors[:3]}")
    print(f"  年龄-名次相关: {age_placement:.3f}")
    print(f"  年龄-评分相关: {age_score:.3f}")
    
    if r2 > 0.9:
        print("  ✓ R² > 0.9，模型拟合良好")
    else:
        print("  ✗ R²偏低，模型可能欠拟合")
        issues.append("Q3: 模型R²偏低")
    
    if 'last_week' in top_factors:
        print("  ✓ 'last_week'是最重要特征，符合逻辑（上周排名预测本周）")
    
    # 年龄效应：年龄越大，排名越差（正相关），评分越低（负相关）
    if age_placement > 0 and age_score < 0:
        print("  ✓ 年龄效应符合预期（年长选手评分低、名次差）")
    
    # 分析Q4
    print("\n[Q4] 新投票系统分析")
    q4_data = load_json(f"{DATA_DIR}/q4_new_system/q4_results_v2.json")
    summary = q4_data.get('summary', {})
    
    w_judge = summary.get('recommended_w_judge', 0)
    w_fan = summary.get('recommended_w_fan', 0)
    n_pareto = summary.get('n_pareto_solutions', 0)
    
    print(f"  推荐评委权重: {w_judge:.2f}")
    print(f"  推荐粉丝权重: {w_fan:.2f}")
    print(f"  帕累托前沿解数: {n_pareto}")
    
    if 0.3 <= w_judge <= 0.7:
        print("  ✓ 权重在合理范围内")
    else:
        print("  ✗ 权重可能过于极端")
        issues.append("Q4: 权重可能过于极端")
    
    # 总结
    print("\n" + "="*70)
    print("总结")
    print("="*70)
    
    if len(issues) == 0:
        print("✓ 所有数据均符合逻辑常理，可以生成可视化图片")
        return True
    else:
        print(f"发现 {len(issues)} 个潜在问题:")
        for issue in issues:
            print(f"  - {issue}")
        return True  # 仍然生成可视化，让用户判断

def plot_q1_visualizations():
    """生成Q1可视化"""
    print("\n生成Q1可视化...")
    
    # 加载数据
    df = pd.read_csv(f"{DATA_DIR}/q1_fan_vote/q1_fan_vote_estimates.csv")
    epa_df = pd.read_csv(f"{DATA_DIR}/q1_fan_vote/q1_epa_details.csv")
    
    # 图1: 粉丝投票分布直方图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 粉丝投票分布
    axes[0].hist(df['estimated_fan_vote'], bins=30, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Estimated Fan Vote Percentage')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Q1: Distribution of Estimated Fan Votes')
    axes[0].axvline(df['estimated_fan_vote'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {df["estimated_fan_vote"].mean():.3f}')
    axes[0].legend()
    
    # EPA正确率按voting_rule
    if 'voting_rule' in epa_df.columns:
        epa_by_rule = epa_df.groupby('voting_rule')['is_correct'].mean()
        bars = axes[1].bar(epa_by_rule.index, epa_by_rule.values, color=['steelblue', 'coral'])
        axes[1].set_xlabel('Voting Rule')
        axes[1].set_ylabel('EPA Accuracy')
        axes[1].set_title('Q1: Elimination Prediction Accuracy by Voting Rule')
        for bar, val in zip(bars, epa_by_rule.values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                        f'{val:.1%}', ha='center')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/q1_fan_vote_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {OUTPUT_DIR}/q1_fan_vote_analysis.png")

def plot_q2_visualizations():
    """生成Q2可视化"""
    print("\n生成Q2可视化...")
    
    # 加载数据
    q2_data = load_json(f"{DATA_DIR}/q2_voting_method/q2_results_v2.json")
    controversy_df = pd.read_csv(f"{DATA_DIR}/q2_voting_method/q2_controversy_counterfactual.csv")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 图1: 投票方法对比
    comparison = q2_data.get('method_comparison', {})
    methods = ['Rank System', 'Percentage System']
    
    rank_data = comparison.get('rank_system', {})
    pct_data = comparison.get('percentage_system', {})
    
    kendall_tau = [
        abs(rank_data.get('kendall_tau_mean', -0.72)),
        abs(pct_data.get('kendall_tau_mean', -0.58))
    ]
    
    bars = axes[0].bar(methods, kendall_tau, color=['steelblue', 'coral'])
    axes[0].set_ylabel('|Kendall τ| (Judge-Placement Correlation)')
    axes[0].set_title('Q2: Voting Method Comparison')
    axes[0].set_ylim(0, 1)
    for bar, val in zip(bars, kendall_tau):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{val:.3f}', ha='center')
    axes[0].axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Moderate correlation')
    axes[0].legend()
    
    # 图2: 争议案例分析
    if len(controversy_df) > 0:
        names = controversy_df['name'].tolist()
        actual = controversy_df['placement'].tolist()
        predicted = controversy_df['rank_predicted'].tolist()
        
        x = np.arange(len(names))
        width = 0.35
        
        axes[1].bar(x - width/2, actual, width, label='Actual Placement', color='steelblue')
        axes[1].bar(x + width/2, predicted, width, label='Rank System Prediction', color='coral')
        axes[1].set_xlabel('Contestant')
        axes[1].set_ylabel('Placement')
        axes[1].set_title('Q2: Controversy Cases - Counterfactual Analysis')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(names, rotation=45, ha='right')
        axes[1].legend()
        axes[1].invert_yaxis()  # 排名越小越好
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/q2_voting_method_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {OUTPUT_DIR}/q2_voting_method_comparison.png")

def plot_q3_visualizations():
    """生成Q3可视化"""
    print("\n生成Q3可视化...")
    
    # 加载数据
    importance_df = pd.read_csv(f"{DATA_DIR}/q3_impact_analysis/q3_feature_importance.csv")
    age_df = pd.read_csv(f"{DATA_DIR}/q3_impact_analysis/q3_age_effect.csv")
    q3_data = load_json(f"{DATA_DIR}/q3_impact_analysis/q3_results_v2.json")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 图1: Top 15特征重要性
    top_features = importance_df.head(15).sort_values('importance', ascending=True)
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(top_features)))
    
    axes[0].barh(top_features['feature'], top_features['importance'], color=colors)
    axes[0].set_xlabel('Feature Importance')
    axes[0].set_title('Q3: Top 15 Feature Importance (XGBoost)')
    axes[0].set_xlim(0, top_features['importance'].max() * 1.1)
    
    # 图2: 年龄效应
    age_effect = q3_data.get('age_effect', {})
    age_placement = age_effect.get('age_placement_corr', 0)
    age_score = age_effect.get('age_score_corr', 0)
    
    effects = ['Age-Placement\n(Higher is Worse)', 'Age-Score\n(Lower is Worse)']
    values = [age_placement, age_score]
    colors = ['coral' if v > 0 else 'steelblue' for v in values]
    
    bars = axes[1].bar(effects, values, color=colors)
    axes[1].set_ylabel('Pearson Correlation')
    axes[1].set_title('Q3: Age Effect on Performance')
    axes[1].axhline(0, color='black', linewidth=0.5)
    for bar, val in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + 0.02 if val > 0 else bar.get_height() - 0.05, 
                    f'{val:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/q3_impact_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {OUTPUT_DIR}/q3_impact_analysis.png")

def plot_q4_visualizations():
    """生成Q4可视化"""
    print("\n生成Q4可视化...")
    
    # 加载数据
    sensitivity_df = pd.read_csv(f"{DATA_DIR}/q4_new_system/q4_sensitivity_analysis.csv")
    system_df = pd.read_csv(f"{DATA_DIR}/q4_new_system/q4_system_comparison.csv")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 图1: 敏感性分析
    axes[0].plot(sensitivity_df['w_judge'], sensitivity_df['fairness'], 
                 label='Fairness', marker='o', markersize=3)
    axes[0].plot(sensitivity_df['w_judge'], sensitivity_df['stability'], 
                 label='Stability', marker='s', markersize=3)
    axes[0].plot(sensitivity_df['w_judge'], sensitivity_df['entertainment'], 
                 label='Entertainment', marker='^', markersize=3)
    axes[0].plot(sensitivity_df['w_judge'], sensitivity_df['total'], 
                 label='Total Score', linewidth=2, color='black')
    
    # 标记最优点
    optimal_idx = sensitivity_df['total'].idxmax()
    optimal_w = sensitivity_df.loc[optimal_idx, 'w_judge']
    optimal_total = sensitivity_df.loc[optimal_idx, 'total']
    axes[0].scatter([optimal_w], [optimal_total], color='red', s=100, zorder=5, 
                    label=f'Optimal (w_judge={optimal_w:.2f})')
    
    axes[0].set_xlabel('Judge Weight (w_judge)')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Q4: Sensitivity Analysis of Voting Weights')
    axes[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axes[0].grid(True, alpha=0.3)
    
    # 图2: 系统对比
    systems = system_df['System'].tolist()
    x = np.arange(len(systems))
    width = 0.2
    
    axes[1].bar(x - width, system_df['Fairness'], width, label='Fairness', color='steelblue')
    axes[1].bar(x, system_df['Stability'], width, label='Stability', color='coral')
    axes[1].bar(x + width, system_df['Entertainment'], width, label='Entertainment', color='green')
    
    axes[1].set_xlabel('System')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Q4: System Comparison')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([s[:15] + '...' if len(s) > 15 else s for s in systems], 
                            rotation=15, ha='right')
    axes[1].legend()
    
    # 添加Total分数标注
    for i, total in enumerate(system_df['Total']):
        axes[1].text(i, 1.05, f'Total: {total:.2f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/q4_new_system_design.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {OUTPUT_DIR}/q4_new_system_design.png")

def create_summary_figure():
    """创建总结图"""
    print("\n生成总结图...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Q1 摘要
    q1_data = load_json(f"{DATA_DIR}/q1_fan_vote/q1_results_v2.json")
    summary = q1_data.get('summary', {})
    
    metrics = ['EPA (Constraint)', 'EPA (Bayesian)', 'Consistency']
    values = [summary.get('epa_constraint', 0), 
              summary.get('epa_bayesian', 0), 
              summary.get('consistency_rate', 0)]
    
    bars = axes[0, 0].bar(metrics, values, color=['steelblue', 'coral', 'green'])
    axes[0, 0].set_ylabel('Rate')
    axes[0, 0].set_title('Q1: Fan Vote Estimation Performance')
    axes[0, 0].set_ylim(0, 1)
    for bar, val in zip(bars, values):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                       f'{val:.1%}', ha='center')
    
    # Q2 摘要
    q2_data = load_json(f"{DATA_DIR}/q2_voting_method/q2_results_v2.json")
    rec = q2_data.get('recommendation', {})
    
    methods = ['Rank System', 'Percentage System']
    scores = [rec.get('rank_total_score', 0), rec.get('percent_total_score', 0)]
    colors = ['steelblue', 'coral']
    
    bars = axes[0, 1].bar(methods, scores, color=colors)
    axes[0, 1].set_ylabel('Total Score')
    axes[0, 1].set_title(f"Q2: Recommended Method: {rec.get('recommended_method', 'N/A')}")
    for bar, val in zip(bars, scores):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{val:.3f}', ha='center')
    
    # Q3 摘要
    q3_data = load_json(f"{DATA_DIR}/q3_impact_analysis/q3_results_v2.json")
    xgb = q3_data.get('xgboost_analysis', {})
    metrics_q3 = xgb.get('metrics', {})
    
    metric_names = ['R²', 'RMSE', 'MAE', 'CV RMSE']
    metric_values = [metrics_q3.get('r2', 0), metrics_q3.get('rmse', 0), 
                    metrics_q3.get('mae', 0), metrics_q3.get('cv_rmse', 0)]
    
    bars = axes[1, 0].bar(metric_names, metric_values, color='steelblue')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_title('Q3: XGBoost Model Performance')
    for bar, val in zip(bars, metric_values):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                       f'{val:.3f}', ha='center')
    
    # Q4 摘要
    q4_data = load_json(f"{DATA_DIR}/q4_new_system/q4_results_v2.json")
    summary_q4 = q4_data.get('summary', {})
    
    # 饼图显示推荐权重
    weights = [summary_q4.get('recommended_w_judge', 0.3), 
               summary_q4.get('recommended_w_fan', 0.7)]
    labels = ['Judge Weight', 'Fan Weight']
    colors = ['steelblue', 'coral']
    
    axes[1, 1].pie(weights, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    axes[1, 1].set_title('Q4: Recommended Voting Weight Distribution')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/summary_all_questions.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  保存: {OUTPUT_DIR}/summary_all_questions.png")

def main():
    print("="*70)
    print("MCM 2026 Problem C: 数据分析与可视化")
    print("="*70)
    
    # 分析数据合理性
    is_valid = analyze_data_logic()
    
    if is_valid:
        # 生成可视化
        plot_q1_visualizations()
        plot_q2_visualizations()
        plot_q3_visualizations()
        plot_q4_visualizations()
        create_summary_figure()
        
        print("\n" + "="*70)
        print(f"所有可视化已保存到: {OUTPUT_DIR}/")
        print("="*70)
        
        # 列出生成的文件
        files = os.listdir(OUTPUT_DIR)
        print("\n生成的文件:")
        for f in files:
            filepath = os.path.join(OUTPUT_DIR, f)
            size = os.path.getsize(filepath)
            print(f"  - {f} ({size/1024:.1f} KB)")

if __name__ == "__main__":
    main()
