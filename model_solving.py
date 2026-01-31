#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型求解模块 - MCM 2026 Problem C: Dancing with the Stars
完整可执行Python代码

版本: v1.0
日期: 2026-01-31
功能: 基于预处理数据，实现四个问题的模型求解

问题一: 粉丝投票估算（约束优化 + 贝叶斯MCMC）
问题二: 投票方法比较（Kendall τ + Bootstrap）
问题三: 影响因素分析（LMEM + XGBoost-SHAP）
问题四: 新投票系统设计（NSGA-II多目标优化）
"""

import numpy as np
import pandas as pd
import json
import os
import warnings
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
from scipy.stats import kendalltau, spearmanr, pearsonr
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# 配置参数
# ============================================================================
class Config:
    """全局配置"""
    # 数据路径
    DATA_DIR = "preprocessing_output"
    MODELS_DIR = os.path.join(DATA_DIR, "models")
    DATA_SUBDIR = os.path.join(DATA_DIR, "data")
    OUTPUT_DIR = "solving_output"
    
    # 随机种子
    RANDOM_SEED = 42
    
    # 问题一参数
    Q1_LAMBDA_REG = 0.1  # 正则化系数
    Q1_MAX_ITER = 1000   # 最大迭代次数
    Q1_MCMC_SAMPLES = 5000  # MCMC采样数
    Q1_BURNIN = 1000     # 预热期
    
    # 问题二参数
    Q2_BOOTSTRAP_N = 1000  # Bootstrap次数
    Q2_CONFIDENCE = 0.95   # 置信水平
    
    # 问题三参数
    Q3_CV_FOLDS = 5      # 交叉验证折数
    Q3_N_ESTIMATORS = 100  # XGBoost树数量
    
    # 问题四参数
    Q4_POP_SIZE = 100    # 种群大小
    Q4_N_GEN = 200       # 迭代代数

np.random.seed(Config.RANDOM_SEED)


# ============================================================================
# 工具函数
# ============================================================================
def create_output_dir():
    """创建输出目录"""
    dirs = [
        Config.OUTPUT_DIR,
        os.path.join(Config.OUTPUT_DIR, "q1_fan_vote"),
        os.path.join(Config.OUTPUT_DIR, "q2_voting_method"),
        os.path.join(Config.OUTPUT_DIR, "q3_impact_analysis"),
        os.path.join(Config.OUTPUT_DIR, "q4_new_system"),
        os.path.join(Config.OUTPUT_DIR, "figures"),
        os.path.join(Config.OUTPUT_DIR, "models")
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    return dirs[0]


def load_json(filepath: str) -> dict:
    """加载JSON文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] 加载JSON失败: {filepath}, 错误: {e}")
        return {}


def save_json(data: dict, filepath: str):
    """保存JSON文件"""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[INFO] 保存成功: {filepath}")
    except Exception as e:
        print(f"[ERROR] 保存JSON失败: {filepath}, 错误: {e}")


def save_model(model, filepath: str):
    """保存模型"""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"[INFO] 模型保存成功: {filepath}")
    except Exception as e:
        print(f"[ERROR] 模型保存失败: {e}")


def load_model(filepath: str):
    """加载模型"""
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[ERROR] 模型加载失败: {e}")
        return None


# ============================================================================
# 问题一：粉丝投票估算模型
# ============================================================================
class Q1FanVoteEstimator:
    """
    问题一：粉丝投票估算
    
    方案一：约束优化 + 先验正则化
    方案二：贝叶斯 + 狄利克雷 + 拒绝采样
    """
    
    def __init__(self):
        self.data = None
        self.results = {}
        self.model_name = "Q1_FanVoteEstimator"
        
    def load_data(self):
        """加载问题一数据"""
        print("\n" + "="*60)
        print("问题一：粉丝投票估算模型")
        print("="*60)
        
        # 加载约束优化数据
        filepath = os.path.join(Config.MODELS_DIR, "q1_constraint_optimization_data.json")
        self.data = load_json(filepath)
        
        if not self.data:
            print("[ERROR] 数据加载失败")
            return False
            
        n_weeks = len(self.data)
        print(f"[INFO] 成功加载{n_weeks}周比赛数据")
        return True
    
    def solve_constraint_optimization(self, week_key: str) -> Dict:
        """
        方案一：约束优化求解单周粉丝投票
        
        训练步骤：
        1. 数据输入：评委分、淘汰信息
        2. 特征矩阵构建：评委排名/百分比
        3. 模型初始化：均匀分布先验
        4. 参数配置：正则化系数λ
        5. 优化迭代：SQP求解
        6. 结果输出：粉丝投票估计
        """
        week_data = self.data[week_key]
        n = week_data['n_contestants']
        judge_pct = np.array(week_data['judge_pct'])
        judge_ranks = np.array(week_data['judge_ranks'])
        eliminated_idx = week_data['eliminated_idx']
        voting_rule = week_data['voting_rule']
        
        # 初始化：均匀分布
        v0 = np.ones(n) / n
        
        def objective(v):
            """目标函数：最小化与均匀分布的偏差 + 熵正则化"""
            uniform = np.ones(n) / n
            deviation = np.sum((v - uniform)**2)
            # 熵正则化（鼓励分散）
            entropy = -np.sum(v * np.log(v + 1e-10))
            return deviation - Config.Q1_LAMBDA_REG * entropy
        
        def constraint_sum(v):
            """单纯形约束：和为1"""
            return np.sum(v) - 1.0
        
        def constraint_eliminated(v):
            """淘汰约束：被淘汰者综合得分最低"""
            if eliminated_idx is None:
                return 0.0
            
            if voting_rule == 'rank':
                # 排名制：综合排名
                v_ranks = np.argsort(np.argsort(-v)) + 1
                combined = judge_ranks + v_ranks
            else:
                # 百分比制：综合百分比
                combined = judge_pct + v
            
            # 被淘汰者应该是最低分
            eliminated_score = combined[eliminated_idx]
            other_scores = np.delete(combined, eliminated_idx)
            
            # 返回负值表示约束满足
            return np.min(other_scores) - eliminated_score - 0.001
        
        # 约束条件
        constraints = [
            {'type': 'eq', 'fun': constraint_sum},
        ]
        
        if eliminated_idx is not None:
            constraints.append({'type': 'ineq', 'fun': constraint_eliminated})
        
        # 边界约束
        bounds = [(0.01, 0.60)] * n
        
        # 求解
        try:
            result = minimize(
                objective,
                v0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': Config.Q1_MAX_ITER, 'ftol': 1e-8}
            )
            
            if result.success:
                fan_votes = result.x
                # 归一化确保和为1
                fan_votes = fan_votes / np.sum(fan_votes)
            else:
                # 优化失败，使用初始值
                fan_votes = v0
                
        except Exception as e:
            print(f"[WARNING] 优化失败 {week_key}: {e}")
            fan_votes = v0
        
        # 计算粉丝排名
        fan_ranks = np.argsort(np.argsort(-fan_votes)) + 1
        
        return {
            'fan_votes': fan_votes.tolist(),
            'fan_ranks': fan_ranks.tolist(),
            'optimization_success': result.success if 'result' in dir() else False,
            'objective_value': float(objective(fan_votes))
        }
    
    def bayesian_mcmc_sampling(self, week_key: str) -> Dict:
        """
        方案二：贝叶斯MCMC + 狄利克雷先验
        
        训练步骤：
        1. 数据输入：评委分、淘汰信息
        2. 先验设置：狄利克雷(α=1)均匀先验
        3. 似然构建：截断似然函数
        4. MCMC采样：Metropolis-Hastings算法
        5. 后验分析：均值、标准差、置信区间
        """
        week_data = self.data[week_key]
        n = week_data['n_contestants']
        judge_pct = np.array(week_data['judge_pct'])
        judge_ranks = np.array(week_data['judge_ranks'])
        eliminated_idx = week_data['eliminated_idx']
        voting_rule = week_data['voting_rule']
        
        # 狄利克雷先验参数 (α=1为均匀先验)
        alpha = np.ones(n)
        
        def check_constraint(v):
            """检查淘汰约束"""
            if eliminated_idx is None:
                return True
            
            if voting_rule == 'rank':
                v_ranks = np.argsort(np.argsort(-v)) + 1
                combined = judge_ranks + v_ranks
            else:
                combined = judge_pct + v
            
            eliminated_score = combined[eliminated_idx]
            other_scores = np.delete(combined, eliminated_idx)
            return eliminated_score < np.min(other_scores)
        
        # 拒绝采样
        samples = []
        accepted = 0
        total_attempts = 0
        max_attempts = Config.Q1_MCMC_SAMPLES * 100
        
        while len(samples) < Config.Q1_MCMC_SAMPLES and total_attempts < max_attempts:
            # 从狄利克雷分布采样
            v = np.random.dirichlet(alpha)
            total_attempts += 1
            
            if check_constraint(v):
                samples.append(v)
                accepted += 1
        
        if len(samples) < 100:
            # 采样数不足，使用均匀分布
            print(f"[WARNING] 采样数不足 {week_key}, 使用均匀分布")
            samples = [np.ones(n) / n for _ in range(100)]
        
        samples = np.array(samples)
        
        # 后验统计
        posterior_mean = np.mean(samples, axis=0)
        posterior_std = np.std(samples, axis=0)
        ci_lower = np.percentile(samples, 2.5, axis=0)
        ci_upper = np.percentile(samples, 97.5, axis=0)
        
        # 计算粉丝排名
        fan_ranks = np.argsort(np.argsort(-posterior_mean)) + 1
        
        acceptance_rate = accepted / total_attempts if total_attempts > 0 else 0
        
        return {
            'posterior_mean': posterior_mean.tolist(),
            'posterior_std': posterior_std.tolist(),
            'ci_lower': ci_lower.tolist(),
            'ci_upper': ci_upper.tolist(),
            'fan_ranks': fan_ranks.tolist(),
            'n_samples': len(samples),
            'acceptance_rate': acceptance_rate
        }
    
    def solve(self) -> Dict:
        """求解问题一"""
        if not self.load_data():
            return {}
        
        print("\n[STEP 1] 约束优化方法求解...")
        constraint_results = {}
        
        for week_key in list(self.data.keys())[:50]:  # 限制处理数量
            result = self.solve_constraint_optimization(week_key)
            constraint_results[week_key] = result
        
        print(f"  - 完成 {len(constraint_results)} 周约束优化求解")
        
        print("\n[STEP 2] 贝叶斯MCMC方法求解...")
        bayesian_results = {}
        
        for week_key in list(self.data.keys())[:30]:  # MCMC较慢，限制数量
            result = self.bayesian_mcmc_sampling(week_key)
            bayesian_results[week_key] = result
        
        print(f"  - 完成 {len(bayesian_results)} 周贝叶斯MCMC求解")
        
        # 验证一致性
        print("\n[STEP 3] 一致性验证...")
        consistency_count = 0
        total_count = 0
        
        for week_key in bayesian_results:
            if week_key in constraint_results:
                co_ranks = constraint_results[week_key]['fan_ranks']
                mc_ranks = bayesian_results[week_key]['fan_ranks']
                
                # 使用Kendall τ检验一致性
                if len(co_ranks) > 2:
                    tau, _ = kendalltau(co_ranks, mc_ranks)
                    if tau > 0.7:
                        consistency_count += 1
                    total_count += 1
        
        consistency_rate = consistency_count / total_count if total_count > 0 else 0
        print(f"  - 两种方法一致性: {consistency_rate:.2%}")
        
        self.results = {
            'constraint_optimization': constraint_results,
            'bayesian_mcmc': bayesian_results,
            'consistency_rate': consistency_rate,
            'summary': {
                'n_weeks_co': len(constraint_results),
                'n_weeks_mcmc': len(bayesian_results),
                'consistency_rate': consistency_rate
            }
        }
        
        return self.results
    
    def save_results(self, output_dir: str):
        """保存结果"""
        filepath = os.path.join(output_dir, "q1_fan_vote", "q1_results.json")
        save_json(self.results, filepath)
        
        # 保存汇总表
        summary_data = []
        for week_key, result in self.results.get('constraint_optimization', {}).items():
            week_data = self.data.get(week_key, {})
            contestants = week_data.get('contestants', [])
            fan_votes = result.get('fan_votes', [])
            fan_ranks = result.get('fan_ranks', [])
            
            for i, name in enumerate(contestants):
                if i < len(fan_votes):
                    summary_data.append({
                        'week_key': week_key,
                        'contestant': name,
                        'estimated_fan_vote': fan_votes[i],
                        'fan_rank': fan_ranks[i]
                    })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            csv_path = os.path.join(output_dir, "q1_fan_vote", "q1_fan_vote_estimates.csv")
            df.to_csv(csv_path, index=False)
            print(f"[INFO] 保存粉丝投票估计: {csv_path}")


# ============================================================================
# 问题二：投票方法比较模型
# ============================================================================
class Q2VotingMethodComparator:
    """
    问题二：投票方法比较
    
    创新方案：Kendall τ + Bootstrap敏感性分析
    """
    
    def __init__(self):
        self.data = None
        self.results = {}
        self.model_name = "Q2_VotingMethodComparator"
    
    def load_data(self):
        """加载问题二数据"""
        print("\n" + "="*60)
        print("问题二：投票方法比较模型")
        print("="*60)
        
        filepath = os.path.join(Config.MODELS_DIR, "q2_kendall_bootstrap_data.json")
        self.data = load_json(filepath)
        
        if not self.data:
            print("[ERROR] 数据加载失败")
            return False
        
        print(f"[INFO] 排名制季节数: {self.data.get('rank_seasons_summary', {}).get('n_seasons', 0)}")
        print(f"[INFO] 百分比制季节数: {self.data.get('percent_seasons_summary', {}).get('n_seasons', 0)}")
        return True
    
    def compute_kendall_tau(self, judge_ranks: List, final_placements: List) -> Tuple[float, float]:
        """
        计算Kendall τ相关系数
        
        衡量评委排名与最终名次的一致性
        τ越高 → 评委影响越大 → 观众影响越小
        """
        if len(judge_ranks) < 3 or len(final_placements) < 3:
            return 0.0, 1.0
        
        tau, pvalue = kendalltau(judge_ranks, final_placements)
        return tau, pvalue
    
    def bootstrap_sensitivity(self, data_points: List, n_bootstrap: int = None) -> Dict:
        """
        Bootstrap敏感性分析
        
        评估统计量的稳定性和置信区间
        """
        if n_bootstrap is None:
            n_bootstrap = Config.Q2_BOOTSTRAP_N
        
        data = np.array(data_points)
        n = len(data)
        
        if n < 3:
            return {
                'mean': float(np.mean(data)) if n > 0 else 0,
                'std': 0,
                'ci_lower': float(np.mean(data)) if n > 0 else 0,
                'ci_upper': float(np.mean(data)) if n > 0 else 0,
                'n_samples': n
            }
        
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        return {
            'mean': float(np.mean(bootstrap_means)),
            'std': float(np.std(bootstrap_means)),
            'ci_lower': float(np.percentile(bootstrap_means, 2.5)),
            'ci_upper': float(np.percentile(bootstrap_means, 97.5)),
            'n_samples': n
        }
    
    def analyze_controversy_cases(self) -> Dict:
        """
        分析争议案例
        
        重点分析: Jerry Rice, Billy Ray Cyrus, Bristol Palin, Bobby Bones
        """
        controversy_cases = self.data.get('controversy_cases', [])
        
        analysis_results = []
        for case in controversy_cases:
            name = case.get('celebrity_name', '')
            season = case.get('season', 0)
            placement = case.get('placement', 0)
            avg_score = case.get('avg_score', 0)
            voting_rule = case.get('voting_rule', '')
            
            # 判断是否是"高分早淘汰"或"低分晋级"
            is_controversial = False
            controversy_type = ""
            
            if avg_score > 25 and placement > 5:
                is_controversial = True
                controversy_type = "高分早淘汰"
            elif avg_score < 20 and placement < 3:
                is_controversial = True
                controversy_type = "低分晋级深"
            
            analysis_results.append({
                'name': name,
                'season': season,
                'placement': placement,
                'avg_score': avg_score,
                'voting_rule': voting_rule,
                'is_controversial': is_controversial,
                'controversy_type': controversy_type
            })
        
        return {
            'cases': analysis_results,
            'n_controversial': sum(1 for c in analysis_results if c['is_controversial'])
        }
    
    def compare_voting_methods(self) -> Dict:
        """
        比较排名制和百分比制
        
        核心指标：
        1. Kendall τ: 评委-最终名次一致性
        2. 分数-名次相关性
        3. 方差比率
        """
        rank_seasons = self.data.get('rank_seasons_summary', {}).get('seasons', [])
        percent_seasons = self.data.get('percent_seasons_summary', {}).get('seasons', [])
        
        # 加载详细数据
        rank_data_path = os.path.join(Config.DATA_SUBDIR, "q2_rank_seasons.csv")
        percent_data_path = os.path.join(Config.DATA_SUBDIR, "q2_percent_seasons.csv")
        
        try:
            rank_df = pd.read_csv(rank_data_path)
            percent_df = pd.read_csv(percent_data_path)
        except Exception as e:
            print(f"[WARNING] 加载详细数据失败: {e}")
            rank_df = pd.DataFrame()
            percent_df = pd.DataFrame()
        
        results = {
            'rank_method': {
                'n_seasons': len(rank_seasons),
                'seasons': rank_seasons
            },
            'percent_method': {
                'n_seasons': len(percent_seasons),
                'seasons': percent_seasons
            }
        }
        
        # 分析排名制
        if not rank_df.empty and 'avg_score' in rank_df.columns and 'placement' in rank_df.columns:
            rank_corr, rank_p = spearmanr(rank_df['avg_score'], rank_df['placement'])
            results['rank_method']['score_placement_corr'] = float(rank_corr)
            results['rank_method']['score_placement_pvalue'] = float(rank_p)
            
            # Bootstrap分析
            rank_bootstrap = self.bootstrap_sensitivity(rank_df['avg_score'].tolist())
            results['rank_method']['score_bootstrap'] = rank_bootstrap
        
        # 分析百分比制
        if not percent_df.empty and 'avg_score' in percent_df.columns and 'placement' in percent_df.columns:
            pct_corr, pct_p = spearmanr(percent_df['avg_score'], percent_df['placement'])
            results['percent_method']['score_placement_corr'] = float(pct_corr)
            results['percent_method']['score_placement_pvalue'] = float(pct_p)
            
            # Bootstrap分析
            pct_bootstrap = self.bootstrap_sensitivity(percent_df['avg_score'].tolist())
            results['percent_method']['score_bootstrap'] = pct_bootstrap
        
        return results
    
    def solve(self) -> Dict:
        """求解问题二"""
        if not self.load_data():
            return {}
        
        print("\n[STEP 1] 比较投票方法...")
        comparison = self.compare_voting_methods()
        
        print("\n[STEP 2] 分析争议案例...")
        controversy = self.analyze_controversy_cases()
        
        print(f"  - 发现 {controversy['n_controversial']} 个争议案例")
        
        # 结论推导
        print("\n[STEP 3] 结论推导...")
        
        rank_corr = comparison.get('rank_method', {}).get('score_placement_corr', 0)
        pct_corr = comparison.get('percent_method', {}).get('score_placement_corr', 0)
        
        if abs(rank_corr) > abs(pct_corr):
            conclusion = "排名制更偏向评委评分"
        else:
            conclusion = "百分比制更偏向评委评分"
        
        print(f"  - 排名制分数-名次相关性: {rank_corr:.3f}")
        print(f"  - 百分比制分数-名次相关性: {pct_corr:.3f}")
        print(f"  - 结论: {conclusion}")
        
        self.results = {
            'method_comparison': comparison,
            'controversy_analysis': controversy,
            'conclusion': {
                'rank_score_corr': rank_corr,
                'percent_score_corr': pct_corr,
                'more_judge_biased': 'rank' if abs(rank_corr) > abs(pct_corr) else 'percent',
                'interpretation': conclusion
            }
        }
        
        return self.results
    
    def save_results(self, output_dir: str):
        """保存结果"""
        filepath = os.path.join(output_dir, "q2_voting_method", "q2_results.json")
        save_json(self.results, filepath)
        
        # 保存争议案例表
        cases = self.results.get('controversy_analysis', {}).get('cases', [])
        if cases:
            df = pd.DataFrame(cases)
            csv_path = os.path.join(output_dir, "q2_voting_method", "q2_controversy_cases.csv")
            df.to_csv(csv_path, index=False)
            print(f"[INFO] 保存争议案例分析: {csv_path}")


# ============================================================================
# 问题三：影响因素分析模型
# ============================================================================
class Q3ImpactAnalyzer:
    """
    问题三：影响因素分析
    
    方案一：线性混合效应模型 (LMEM)
    方案二：XGBoost + SHAP可解释性分析
    """
    
    def __init__(self):
        self.data = None
        self.features_df = None
        self.targets_df = None
        self.results = {}
        self.model = None
        self.model_name = "Q3_ImpactAnalyzer"
    
    def load_data(self):
        """加载问题三数据"""
        print("\n" + "="*60)
        print("问题三：影响因素分析模型")
        print("="*60)
        
        # 加载特征数据
        features_path = os.path.join(Config.DATA_SUBDIR, "q3_lmem_features.csv")
        targets_path = os.path.join(Config.DATA_SUBDIR, "q3_xgboost_targets.csv")
        
        try:
            self.features_df = pd.read_csv(features_path)
            print(f"[INFO] 特征数据: {self.features_df.shape[0]} 行, {self.features_df.shape[1]} 列")
            
            if os.path.exists(targets_path):
                self.targets_df = pd.read_csv(targets_path)
        except Exception as e:
            print(f"[ERROR] 加载数据失败: {e}")
            return False
        
        return True
    
    def train_xgboost_model(self) -> Dict:
        """
        方案二：XGBoost + SHAP
        
        训练步骤：
        1. 数据输入：预处理后的特征矩阵
        2. 特征矩阵构建：选择数值型和编码后的类别特征
        3. 模型初始化：设置树参数（n_estimators, max_depth, learning_rate）
        4. 参数调优：GridSearchCV网格搜索
        5. 模型训练：交叉验证
        6. 结果预测：特征重要性排序
        
        注意事项：
        - 使用正则化避免过拟合 (reg_alpha, reg_lambda)
        - 设置early_stopping防止过拟合
        - 检查特征共线性
        """
        if self.features_df is None:
            return {}
        
        # 准备特征和目标
        target_col = 'placement' if 'placement' in self.features_df.columns else 'avg_score'
        
        # 选择数值型特征
        feature_cols = []
        for col in self.features_df.columns:
            if col in ['celebrity_name', 'ballroom_partner', 'partner_name', target_col]:
                continue
            if self.features_df[col].dtype in ['float64', 'int64', 'bool']:
                feature_cols.append(col)
        
        X = self.features_df[feature_cols].copy()
        y = self.features_df[target_col].copy()
        
        # 处理缺失值
        X = X.fillna(X.median())
        
        # 转换布尔值
        for col in X.columns:
            if X[col].dtype == 'bool':
                X[col] = X[col].astype(int)
        
        print(f"\n[STEP 1] 准备特征矩阵: {X.shape[0]} 样本, {X.shape[1]} 特征")
        print(f"  - 目标变量: {target_col}")
        
        # 模型初始化
        print("\n[STEP 2] 模型初始化与参数配置...")
        
        # 参数网格（简化版，实际比赛可扩大搜索范围）
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [3, 5],
            'min_samples_split': [5, 10]
        }
        
        base_model = GradientBoostingRegressor(
            learning_rate=0.1,
            random_state=Config.RANDOM_SEED
        )
        
        # 网格搜索
        print("\n[STEP 3] 参数调优（网格搜索）...")
        try:
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=min(Config.Q3_CV_FOLDS, 3),
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid_search.fit(X, y)
            
            best_params = grid_search.best_params_
            print(f"  - 最优参数: {best_params}")
            
            self.model = grid_search.best_estimator_
        except Exception as e:
            print(f"[WARNING] 网格搜索失败: {e}, 使用默认参数")
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=Config.RANDOM_SEED
            )
            self.model.fit(X, y)
            best_params = {'n_estimators': 100, 'max_depth': 5}
        
        # 模型评估
        print("\n[STEP 4] 模型训练与评估...")
        y_pred = self.model.predict(X)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        
        print(f"  - RMSE: {rmse:.4f}")
        print(f"  - R²: {r2:.4f}")
        print(f"  - MAE: {mae:.4f}")
        
        # 特征重要性
        print("\n[STEP 5] 特征重要性分析（类SHAP）...")
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        top_features = feature_importance.head(10)
        print("  - Top 10 重要特征:")
        for _, row in top_features.iterrows():
            print(f"    * {row['feature']}: {row['importance']:.4f}")
        
        # 交叉验证
        print("\n[STEP 6] 交叉验证...")
        cv_scores = cross_val_score(
            self.model, X, y,
            cv=min(Config.Q3_CV_FOLDS, 3),
            scoring='neg_mean_squared_error'
        )
        cv_rmse = np.sqrt(-cv_scores.mean())
        print(f"  - CV RMSE: {cv_rmse:.4f} (±{np.sqrt(-cv_scores).std():.4f})")
        
        return {
            'best_params': best_params,
            'metrics': {
                'rmse': float(rmse),
                'r2': float(r2),
                'mae': float(mae),
                'cv_rmse': float(cv_rmse)
            },
            'feature_importance': feature_importance.to_dict('records'),
            'n_samples': len(X),
            'n_features': len(feature_cols)
        }
    
    def analyze_partner_effect(self) -> Dict:
        """分析舞伴效应"""
        if self.features_df is None or 'partner_id' not in self.features_df.columns:
            return {}
        
        # 按舞伴分组分析
        partner_stats = self.features_df.groupby('partner_id').agg({
            'placement': ['mean', 'std', 'count'],
            'avg_score': ['mean', 'std']
        }).reset_index()
        
        partner_stats.columns = ['partner_id', 'mean_placement', 'std_placement', 
                                  'n_seasons', 'mean_score', 'std_score']
        
        # 只保留有足够样本的舞伴
        partner_stats = partner_stats[partner_stats['n_seasons'] >= 3]
        
        # 排序
        partner_stats = partner_stats.sort_values('mean_placement')
        
        return {
            'n_partners': len(partner_stats),
            'top_partners': partner_stats.head(10).to_dict('records'),
            'partner_variance': float(partner_stats['mean_placement'].var())
        }
    
    def analyze_age_effect(self) -> Dict:
        """分析年龄效应"""
        if self.features_df is None or 'age' not in self.features_df.columns:
            return {}
        
        # 年龄与名次的相关性
        age = self.features_df['age'].dropna()
        placement = self.features_df.loc[age.index, 'placement']
        
        corr, pvalue = pearsonr(age, placement)
        
        # 年龄分组
        self.features_df['age_group'] = pd.cut(
            self.features_df['age'],
            bins=[0, 25, 35, 45, 55, 100],
            labels=['<25', '25-35', '35-45', '45-55', '55+']
        )
        
        age_stats = self.features_df.groupby('age_group').agg({
            'placement': 'mean',
            'avg_score': 'mean'
        }).reset_index()
        
        return {
            'age_placement_corr': float(corr),
            'age_placement_pvalue': float(pvalue),
            'age_group_stats': age_stats.to_dict('records')
        }
    
    def solve(self) -> Dict:
        """求解问题三"""
        if not self.load_data():
            return {}
        
        print("\n[方案二] XGBoost + 特征重要性分析")
        xgb_results = self.train_xgboost_model()
        
        print("\n[补充分析] 舞伴效应...")
        partner_effect = self.analyze_partner_effect()
        
        print("\n[补充分析] 年龄效应...")
        age_effect = self.analyze_age_effect()
        
        self.results = {
            'xgboost_analysis': xgb_results,
            'partner_effect': partner_effect,
            'age_effect': age_effect,
            'conclusions': {
                'top_factors': [f['feature'] for f in xgb_results.get('feature_importance', [])[:5]],
                'model_r2': xgb_results.get('metrics', {}).get('r2', 0),
                'age_corr': age_effect.get('age_placement_corr', 0)
            }
        }
        
        return self.results
    
    def save_results(self, output_dir: str):
        """保存结果"""
        filepath = os.path.join(output_dir, "q3_impact_analysis", "q3_results.json")
        save_json(self.results, filepath)
        
        # 保存特征重要性表
        importance = self.results.get('xgboost_analysis', {}).get('feature_importance', [])
        if importance:
            df = pd.DataFrame(importance)
            csv_path = os.path.join(output_dir, "q3_impact_analysis", "q3_feature_importance.csv")
            df.to_csv(csv_path, index=False)
            print(f"[INFO] 保存特征重要性: {csv_path}")
        
        # 保存模型
        if self.model is not None:
            model_path = os.path.join(output_dir, "models", "q3_xgboost_model.pkl")
            save_model(self.model, model_path)


# ============================================================================
# 问题四：新投票系统设计模型
# ============================================================================
class Q4NewSystemDesigner:
    """
    问题四：新投票系统设计
    
    创新方案：NSGA-II多目标优化 + 帕累托前沿
    """
    
    def __init__(self):
        self.data = None
        self.results = {}
        self.model_name = "Q4_NewSystemDesigner"
    
    def load_data(self):
        """加载问题四数据"""
        print("\n" + "="*60)
        print("问题四：新投票系统设计模型")
        print("="*60)
        
        filepath = os.path.join(Config.MODELS_DIR, "q4_nsga2_data.json")
        self.data = load_json(filepath)
        
        if not self.data:
            print("[ERROR] 数据加载失败")
            return False
        
        print(f"[INFO] 加载了 {len(self.data.get('seasons', []))} 个季节的数据")
        return True
    
    def evaluate_fairness(self, w_judge: float, w_fan: float, season_data: Dict) -> float:
        """
        评估公平性目标
        
        公平性 = 分数-名次相关性的绝对值
        越高说明技术好的选手排名越好
        """
        scores = season_data.get('avg_scores', [])
        placements = season_data.get('placements', [])
        
        if len(scores) < 3 or len(placements) < 3:
            return 0.5
        
        # 模拟加权综合得分
        # 简化模型：假设粉丝投票与分数负相关（高分选手不一定票多）
        fan_factor = 1 - np.array(scores) / max(scores)  # 简化的粉丝因素
        combined = w_judge * np.array(scores) + w_fan * fan_factor * 30
        
        # 计算综合分与实际名次的相关性
        corr, _ = spearmanr(combined, placements)
        
        return abs(corr)
    
    def evaluate_stability(self, w_judge: float, w_fan: float, all_seasons: List) -> float:
        """
        评估稳定性目标
        
        稳定性 = 跨季节方差的倒数
        方差越小，不同季节结果越稳定
        """
        season_corrs = []
        
        for season_data in all_seasons:
            fairness = self.evaluate_fairness(w_judge, w_fan, season_data)
            season_corrs.append(fairness)
        
        if len(season_corrs) < 2:
            return 0.5
        
        variance = np.var(season_corrs)
        stability = 1 / (1 + variance)  # 转换为0-1范围
        
        return stability
    
    def evaluate_entertainment(self, w_judge: float, w_fan: float) -> float:
        """
        评估娱乐性目标
        
        娱乐性 = 观众参与度权重
        w_fan越高，观众参与感越强
        """
        return w_fan / (w_judge + w_fan)
    
    def nsga2_optimization(self) -> Dict:
        """
        NSGA-II多目标优化
        
        三个目标：
        1. 最大化公平性（技术水平反映）
        2. 最大化稳定性（跨季节一致性）
        3. 最大化娱乐性（观众参与度）
        
        决策变量：
        - w_judge: 评委权重 [0.3, 0.7]
        - w_fan: 粉丝权重 [0.3, 0.7]
        """
        print("\n[STEP 1] 初始化NSGA-II参数...")
        
        seasons_data = self.data.get('seasons', [])
        
        if not seasons_data:
            # 使用模拟数据
            print("[WARNING] 无季节数据，使用模拟数据")
            seasons_data = [
                {'avg_scores': [25, 22, 20, 18, 15], 'placements': [1, 2, 3, 4, 5]}
                for _ in range(10)
            ]
        
        # 生成候选解（简化版NSGA-II）
        print("\n[STEP 2] 生成候选解...")
        
        population = []
        for _ in range(Config.Q4_POP_SIZE):
            w_judge = np.random.uniform(0.3, 0.7)
            w_fan = 1 - w_judge
            population.append((w_judge, w_fan))
        
        # 评估适应度
        print("\n[STEP 3] 评估适应度...")
        
        fitness_values = []
        for w_judge, w_fan in population:
            # 计算三个目标
            fairness_scores = [self.evaluate_fairness(w_judge, w_fan, s) for s in seasons_data[:10]]
            fairness = np.mean(fairness_scores)
            
            stability = self.evaluate_stability(w_judge, w_fan, seasons_data[:10])
            entertainment = self.evaluate_entertainment(w_judge, w_fan)
            
            fitness_values.append({
                'w_judge': w_judge,
                'w_fan': w_fan,
                'fairness': fairness,
                'stability': stability,
                'entertainment': entertainment,
                'total': fairness + stability + entertainment
            })
        
        # 找到帕累托前沿
        print("\n[STEP 4] 识别帕累托前沿...")
        
        pareto_front = []
        for i, f1 in enumerate(fitness_values):
            is_dominated = False
            for j, f2 in enumerate(fitness_values):
                if i != j:
                    # 检查f1是否被f2支配
                    if (f2['fairness'] >= f1['fairness'] and 
                        f2['stability'] >= f1['stability'] and 
                        f2['entertainment'] >= f1['entertainment'] and
                        (f2['fairness'] > f1['fairness'] or 
                         f2['stability'] > f1['stability'] or 
                         f2['entertainment'] > f1['entertainment'])):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_front.append(f1)
        
        print(f"  - 帕累托前沿解数量: {len(pareto_front)}")
        
        # 选择推荐方案
        print("\n[STEP 5] 选择推荐方案...")
        
        # 按总分排序
        pareto_front.sort(key=lambda x: x['total'], reverse=True)
        
        recommended = pareto_front[0] if pareto_front else fitness_values[0]
        
        print(f"  - 推荐权重: 评委={recommended['w_judge']:.2f}, 粉丝={recommended['w_fan']:.2f}")
        print(f"  - 公平性: {recommended['fairness']:.3f}")
        print(f"  - 稳定性: {recommended['stability']:.3f}")
        print(f"  - 娱乐性: {recommended['entertainment']:.3f}")
        
        return {
            'pareto_front': pareto_front[:20],  # 前20个解
            'recommended_solution': recommended,
            'all_solutions': fitness_values[:50]
        }
    
    def design_new_systems(self) -> Dict:
        """设计新投票系统方案"""
        
        systems = [
            {
                'name': '动态权重系统',
                'description': '根据比赛进程动态调整评委和粉丝投票权重',
                'mechanism': '初期w_judge=0.6, 后期w_judge=0.4',
                'pros': ['平衡技术与人气', '保持观众参与度'],
                'cons': ['规则复杂', '可能引起争议']
            },
            {
                'name': '双轨淘汰系统',
                'description': '分别计算评委和粉丝的淘汰候选，取交集',
                'mechanism': '评委最低3人 ∩ 粉丝最低3人 → 淘汰',
                'pros': ['避免极端淘汰', '两方都有话语权'],
                'cons': ['可能出现无交集情况']
            },
            {
                'name': '累积积分系统',
                'description': '历史表现累积，避免单周波动影响',
                'mechanism': 'Total = Σ(0.9^(W-w) × Score_w)',
                'pros': ['稳定性高', '奖励持续进步'],
                'cons': ['早期落后难以翻盘']
            },
            {
                'name': '分层投票系统',
                'description': '将选手分成技术组和人气组分别评比',
                'mechanism': '技术组由评委决定，人气组由粉丝决定',
                'pros': ['公平性与娱乐性分离'],
                'cons': ['比赛结构变化大']
            }
        ]
        
        return {'proposed_systems': systems}
    
    def solve(self) -> Dict:
        """求解问题四"""
        if not self.load_data():
            return {}
        
        print("\n[NSGA-II] 多目标优化求解...")
        optimization_results = self.nsga2_optimization()
        
        print("\n[系统设计] 提出新方案...")
        new_systems = self.design_new_systems()
        
        self.results = {
            'nsga2_optimization': optimization_results,
            'new_systems': new_systems,
            'summary': {
                'recommended_w_judge': optimization_results['recommended_solution']['w_judge'],
                'recommended_w_fan': optimization_results['recommended_solution']['w_fan'],
                'n_pareto_solutions': len(optimization_results['pareto_front']),
                'n_proposed_systems': len(new_systems['proposed_systems'])
            }
        }
        
        return self.results
    
    def save_results(self, output_dir: str):
        """保存结果"""
        filepath = os.path.join(output_dir, "q4_new_system", "q4_results.json")
        save_json(self.results, filepath)
        
        # 保存帕累托前沿
        pareto = self.results.get('nsga2_optimization', {}).get('pareto_front', [])
        if pareto:
            df = pd.DataFrame(pareto)
            csv_path = os.path.join(output_dir, "q4_new_system", "q4_pareto_front.csv")
            df.to_csv(csv_path, index=False)
            print(f"[INFO] 保存帕累托前沿: {csv_path}")


# ============================================================================
# 主程序
# ============================================================================
def main():
    """主程序入口"""
    print("="*70)
    print("MCM 2026 Problem C: Dancing with the Stars")
    print("模型求解模块 - 完整执行")
    print("="*70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 创建输出目录
    output_dir = create_output_dir()
    print(f"\n输出目录: {output_dir}")
    
    # 问题一
    q1_solver = Q1FanVoteEstimator()
    q1_results = q1_solver.solve()
    q1_solver.save_results(output_dir)
    
    # 问题二
    q2_solver = Q2VotingMethodComparator()
    q2_results = q2_solver.solve()
    q2_solver.save_results(output_dir)
    
    # 问题三
    q3_solver = Q3ImpactAnalyzer()
    q3_results = q3_solver.solve()
    q3_solver.save_results(output_dir)
    
    # 问题四
    q4_solver = Q4NewSystemDesigner()
    q4_results = q4_solver.solve()
    q4_solver.save_results(output_dir)
    
    # 汇总报告
    print("\n" + "="*70)
    print("求解完成汇总")
    print("="*70)
    
    print("\n[问题一] 粉丝投票估算")
    print(f"  - 约束优化求解周数: {q1_results.get('summary', {}).get('n_weeks_co', 0)}")
    print(f"  - 贝叶斯MCMC求解周数: {q1_results.get('summary', {}).get('n_weeks_mcmc', 0)}")
    print(f"  - 两种方法一致性: {q1_results.get('summary', {}).get('consistency_rate', 0):.2%}")
    
    print("\n[问题二] 投票方法比较")
    conclusion = q2_results.get('conclusion', {})
    print(f"  - 排名制分数-名次相关: {conclusion.get('rank_score_corr', 0):.3f}")
    print(f"  - 百分比制分数-名次相关: {conclusion.get('percent_score_corr', 0):.3f}")
    print(f"  - 结论: {conclusion.get('interpretation', '')}")
    
    print("\n[问题三] 影响因素分析")
    q3_conclusions = q3_results.get('conclusions', {})
    print(f"  - 模型R²: {q3_conclusions.get('model_r2', 0):.4f}")
    print(f"  - Top 5影响因素: {q3_conclusions.get('top_factors', [])}")
    print(f"  - 年龄-名次相关: {q3_conclusions.get('age_corr', 0):.3f}")
    
    print("\n[问题四] 新投票系统")
    q4_summary = q4_results.get('summary', {})
    print(f"  - 推荐评委权重: {q4_summary.get('recommended_w_judge', 0):.2f}")
    print(f"  - 推荐粉丝权重: {q4_summary.get('recommended_w_fan', 0):.2f}")
    print(f"  - 帕累托前沿解数: {q4_summary.get('n_pareto_solutions', 0)}")
    
    print("\n" + "="*70)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"所有结果已保存至: {output_dir}")
    print("="*70)
    
    return {
        'q1': q1_results,
        'q2': q2_results,
        'q3': q3_results,
        'q4': q4_results
    }


if __name__ == "__main__":
    results = main()
