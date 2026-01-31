#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
DWTS 数据预处理完整可执行代码
MCM 2026 Problem C: Data With The Stars
================================================================================
版本: 3.0
日期: 2026-01-30
说明: 本代码实现DWTS数据集的完整预处理流程，输出所有模型建立模块所需的数据

模型数据输出:
- 问题一: 约束优化数据 + 贝叶斯MCMC数据
- 问题二: Kendall τ + Bootstrap数据
- 问题三: LMEM特征矩阵 + XGBoost-SHAP特征矩阵
- 问题四: NSGA-II多目标优化数据
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import json
import pickle
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
import re

warnings.filterwarnings('ignore')

# ============================================================================
# ⚠️ 重要：请修改以下路径为您的本地数据路径
# ============================================================================
# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 输入路径设置 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
INPUT_DATA_PATH = "./2026_MCM_Problem_C_Data.csv"  # 原始数据文件路径
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ 输入路径设置 ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

# ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓ 输出路径设置 ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓
OUTPUT_DIR = "./preprocessing_output"              # 输出文件夹
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑ 输出路径设置 ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
# ============================================================================


class DWTSDataPreprocessor:
    """
    DWTS数据预处理类
    
    功能:
    1. 数据加载与清洗
    2. 特征工程
    3. 为各模型生成专用数据
    """
    
    def __init__(self, input_path: str, output_dir: str):
        """
        初始化预处理器
        
        Args:
            input_path: 原始数据文件路径
            output_dir: 输出目录路径
        """
        self.input_path = input_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        (self.output_dir / 'data').mkdir(exist_ok=True)
        (self.output_dir / 'figures').mkdir(exist_ok=True)
        (self.output_dir / 'models').mkdir(exist_ok=True)
        
        # 数据存储
        self.df_raw = None
        self.df_clean = None
        
        # 评委数量规则（根据季节）
        self.judge_count_rules = {
            range(1, 19): 3,    # S1-18: 3评委
            range(19, 21): 4,   # S19-20: 4评委
            range(21, 23): 3,   # S21-22: 3评委
            range(23, 25): 4,   # S23-24: 4评委
            range(25, 30): 3,   # S25-29: 3评委
            range(30, 32): 4,   # S30-31: 4评委
            range(32, 35): 3,   # S32-34: 3评委
        }
        
    def get_judge_count(self, season: int) -> int:
        """根据季节获取评委数量"""
        for season_range, count in self.judge_count_rules.items():
            if season in season_range:
                return count
        return 3  # 默认3评委
    
    def get_voting_rule(self, season: int) -> str:
        """
        根据季节获取投票规则
        
        Returns:
            'rank': 排名制 (S1-2, S28-34)
            'percent': 百分比制 (S3-27)
        """
        if season <= 2 or season >= 28:
            return 'rank'
        else:
            return 'percent'
    
    def get_voting_phase(self, season: int) -> int:
        """
        获取投票规则阶段
        
        Returns:
            1: 排名制第一阶段 (S1-2)
            2: 百分比制 (S3-27)
            3: 排名制+评委拯救 (S28-34)
        """
        if season <= 2:
            return 1
        elif season <= 27:
            return 2
        else:
            return 3
    
    # ========================================================================
    # 第1部分：数据加载
    # ========================================================================
    
    def load_data(self) -> pd.DataFrame:
        """加载原始数据"""
        print("=" * 60)
        print("第1步：加载原始数据")
        print("=" * 60)
        
        self.df_raw = pd.read_csv(self.input_path)
        
        print(f"✅ 数据加载成功!")
        print(f"   - 文件路径: {self.input_path}")
        print(f"   - 数据维度: {self.df_raw.shape[0]} 行 × {self.df_raw.shape[1]} 列")
        print(f"   - 季节范围: S{self.df_raw['season'].min()} - S{self.df_raw['season'].max()}")
        
        return self.df_raw
    
    # ========================================================================
    # 第2部分：数据清洗
    # ========================================================================
    
    def clean_data(self) -> pd.DataFrame:
        """
        数据清洗主流程
        
        处理内容:
        1. 0分标记识别
        2. 评委数量动态识别
        3. 淘汰周次提取
        4. 类别标准化
        """
        print("\n" + "=" * 60)
        print("第2步：数据清洗")
        print("=" * 60)
        
        df = self.df_raw.copy()
        
        # ----- 2.1 添加评委数量字段 -----
        print("  [2.1] 添加评委数量字段...")
        df['judge_count'] = df['season'].apply(self.get_judge_count)
        df['max_score'] = df['judge_count'] * 10  # 满分
        
        # ----- 2.2 添加投票规则字段 -----
        print("  [2.2] 添加投票规则字段...")
        df['voting_rule'] = df['season'].apply(self.get_voting_rule)
        df['voting_phase'] = df['season'].apply(self.get_voting_phase)
        
        # ----- 2.3 提取淘汰周次 -----
        print("  [2.3] 提取淘汰周次...")
        df['elimination_week'] = df['results'].apply(self._extract_elimination_week)
        
        # ----- 2.4 计算每周总分和最后有效周 -----
        print("  [2.4] 计算每周总分...")
        week_cols = []
        for week in range(1, 12):  # 最多11周
            judge_cols = [f'week{week}_judge{j}_score' for j in range(1, 5)]
            existing_cols = [c for c in judge_cols if c in df.columns]
            
            if existing_cols:
                # 计算每周总分（忽略NaN）
                total_col = f'week{week}_total'
                df[total_col] = df[existing_cols].sum(axis=1, skipna=True)
                week_cols.append(total_col)
                
                # 计算每周平均分
                avg_col = f'week{week}_avg'
                df[avg_col] = df[existing_cols].mean(axis=1, skipna=True)
        
        # ----- 2.5 计算最后有效周（分数>0的最后一周） -----
        print("  [2.5] 计算最后有效周...")
        df['last_valid_week'] = df.apply(self._get_last_valid_week, axis=1, week_cols=week_cols)
        
        # ----- 2.6 类别字段标准化 -----
        print("  [2.6] 类别字段标准化...")
        if 'celebrity_industry' in df.columns:
            df['celebrity_industry'] = df['celebrity_industry'].str.strip().str.title()
        
        # ----- 2.7 舞伴编码 -----
        print("  [2.7] 舞伴编码...")
        df['partner_id'] = pd.factorize(df['ballroom_partner'])[0]
        
        # ----- 2.8 计算整体表现指标 -----
        print("  [2.8] 计算整体表现指标...")
        df['avg_score_all_weeks'] = df[[c for c in week_cols if c in df.columns]].replace(0, np.nan).mean(axis=1)
        
        # 计算分数进步趋势
        df['score_trend'] = df.apply(self._calculate_score_trend, axis=1, week_cols=week_cols)
        
        self.df_clean = df
        
        print(f"\n✅ 数据清洗完成!")
        print(f"   - 清洗后维度: {df.shape[0]} 行 × {df.shape[1]} 列")
        print(f"   - 新增字段: judge_count, voting_rule, voting_phase, elimination_week, last_valid_week, partner_id, avg_score_all_weeks, score_trend")
        
        return df
    
    def _extract_elimination_week(self, result: str) -> int:
        """从results字段提取淘汰周次"""
        if pd.isna(result):
            return -1
        
        result = str(result).lower()
        
        # 冠军
        if 'winner' in result or '1st' in result:
            return 99  # 特殊标记：冠军
        
        # Eliminated Week X
        match = re.search(r'week\s*(\d+)', result)
        if match:
            return int(match.group(1))
        
        # 数字形式的名次
        match = re.search(r'(\d+)(st|nd|rd|th)', result)
        if match:
            placement = int(match.group(1))
            # 名次越靠前，淘汰周越晚（简化估算）
            return 12 - placement if placement < 12 else 1
        
        return -1
    
    def _get_last_valid_week(self, row, week_cols: List[str]) -> int:
        """获取选手最后一个有效比赛周（分数>0）"""
        last_week = 0
        for i, col in enumerate(week_cols, 1):
            if col in row.index and row[col] > 0:
                last_week = i
        return last_week
    
    def _calculate_score_trend(self, row, week_cols: List[str]) -> float:
        """计算分数变化趋势（斜率）"""
        scores = []
        for col in week_cols:
            if col in row.index and row[col] > 0:
                scores.append(row[col])
        
        if len(scores) < 2:
            return 0.0
        
        # 简单线性回归斜率
        x = np.arange(len(scores))
        slope, _ = np.polyfit(x, scores, 1)
        return slope
    
    # ========================================================================
    # 第3部分：问题一数据准备
    # ========================================================================
    
    def prepare_q1_constraint_optimization_data(self) -> Dict:
        """
        问题一方案一：约束优化模型数据准备
        
        输出结构:
        {
            (season, week): {
                'contestants': [name1, name2, ...],
                'judge_scores': [J1, J2, ...],
                'judge_pct': [pct1, pct2, ...],
                'judge_ranks': [R1, R2, ...],
                'eliminated_idx': idx,
                'eliminated_name': name,
                'voting_rule': 'rank' or 'percent',
                'n_contestants': N
            }
        }
        """
        print("\n" + "=" * 60)
        print("第3步：准备问题一（约束优化）数据")
        print("=" * 60)
        
        df = self.df_clean
        week_data = {}
        
        for season in sorted(df['season'].unique()):
            season_df = df[df['season'] == season].copy()
            voting_rule = self.get_voting_rule(season)
            judge_count = self.get_judge_count(season)
            max_score = judge_count * 10
            
            # 确定该季有多少周比赛
            max_week = int(season_df['last_valid_week'].max())
            
            for week in range(1, max_week + 1):
                week_col = f'week{week}_total'
                
                if week_col not in season_df.columns:
                    continue
                
                # 筛选当周仍在比赛的选手（分数>0）
                active_mask = season_df[week_col] > 0
                active_df = season_df[active_mask].copy()
                
                if len(active_df) < 2:
                    continue
                
                # 获取评委分
                judge_scores = active_df[week_col].values.astype(float)
                contestants = active_df['celebrity_name'].tolist()
                
                # 计算评委分占比
                total_judge = judge_scores.sum()
                judge_pct = judge_scores / total_judge if total_judge > 0 else np.ones(len(judge_scores)) / len(judge_scores)
                
                # 计算评委排名（分数越高排名越靠前=数字越小）
                judge_ranks = stats.rankdata(-judge_scores, method='min')
                
                # 找出被淘汰的选手
                eliminated_idx = None
                eliminated_name = None
                
                # 被淘汰者 = 下一周分数变为0的选手
                if week < max_week:
                    next_week_col = f'week{week+1}_total'
                    if next_week_col in season_df.columns:
                        for i, (idx, row) in enumerate(active_df.iterrows()):
                            if season_df.loc[idx, next_week_col] == 0:
                                eliminated_idx = i
                                eliminated_name = row['celebrity_name']
                                break
                
                # 存储数据
                week_data[(season, week)] = {
                    'contestants': contestants,
                    'judge_scores': [float(x) for x in judge_scores],
                    'judge_pct': [float(x) for x in judge_pct],
                    'judge_ranks': [int(x) for x in judge_ranks],
                    'eliminated_idx': int(eliminated_idx) if eliminated_idx is not None else None,
                    'eliminated_name': eliminated_name,
                    'voting_rule': voting_rule,
                    'judge_count': int(judge_count),
                    'max_score': int(max_score),
                    'n_contestants': len(contestants),
                    'season': int(season),
                    'week': int(week)
                }
        
        # 保存数据
        output_path = self.output_dir / 'models' / 'q1_constraint_optimization_data.json'
        
        # 转换key为字符串以便JSON序列化
        serializable_data = {f"{k[0]}_{k[1]}": v for k, v in week_data.items()}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 问题一（约束优化）数据准备完成!")
        print(f"   - 数据条数: {len(week_data)} 周")
        print(f"   - 输出路径: {output_path}")
        
        return week_data
    
    def prepare_q1_bayesian_mcmc_data(self) -> Dict:
        """
        问题一方案二：贝叶斯MCMC + 狄利克雷数据准备
        
        输出结构:
        {
            (season, week): {
                'n': 选手数量,
                'alpha_prior': 狄利克雷先验参数,
                'judge_pct': 评委分占比,
                'constraint_type': 'rank' or 'percent',
                'eliminated_idx': 被淘汰者索引
            }
        }
        """
        print("\n" + "=" * 60)
        print("第4步：准备问题一（贝叶斯MCMC）数据")
        print("=" * 60)
        
        # 首先获取约束优化数据
        constraint_data = self.prepare_q1_constraint_optimization_data() if not hasattr(self, '_q1_data') else self._q1_data
        
        mcmc_data = {}
        
        for key, data in constraint_data.items():
            n = data['n_contestants']
            judge_pct = np.array(data['judge_pct'])
            
            # 设计狄利克雷先验参数
            # 方法：以评委分占比为基础，加入不确定性
            # alpha_i = base_alpha * (1 + judge_pct_i)
            base_alpha = 2.0  # 先验强度
            alpha_prior = base_alpha * (1 + judge_pct * n)
            
            # 确保alpha > 0
            alpha_prior = np.maximum(alpha_prior, 0.5)
            
            mcmc_data[key] = {
                'n': int(n),
                'alpha_prior': [float(x) for x in alpha_prior],
                'judge_pct': data['judge_pct'],
                'judge_ranks': data['judge_ranks'],
                'constraint_type': data['voting_rule'],
                'eliminated_idx': data['eliminated_idx'],
                'contestants': data['contestants'],
                'season': int(data['season']),
                'week': int(data['week'])
            }
        
        # 保存数据
        output_path = self.output_dir / 'models' / 'q1_bayesian_mcmc_data.json'
        serializable_data = {f"{k[0]}_{k[1]}": v for k, v in mcmc_data.items()}
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 问题一（贝叶斯MCMC）数据准备完成!")
        print(f"   - 数据条数: {len(mcmc_data)} 周")
        print(f"   - 输出路径: {output_path}")
        
        return mcmc_data
    
    # ========================================================================
    # 第4部分：问题二数据准备
    # ========================================================================
    
    def prepare_q2_kendall_bootstrap_data(self) -> Dict:
        """
        问题二：Kendall τ + Bootstrap 敏感性分析数据准备
        
        输出结构:
        {
            'rank_seasons': 排名制季节数据列表,
            'percent_seasons': 百分比制季节数据列表,
            'controversy_cases': 争议案例数据,
            'cross_season_comparison': 跨季节对比数据
        }
        """
        print("\n" + "=" * 60)
        print("第5步：准备问题二（Kendall τ + Bootstrap）数据")
        print("=" * 60)
        
        df = self.df_clean
        
        # ----- 分离排名制和百分比制数据 -----
        rank_seasons = df[df['voting_rule'] == 'rank'].copy()
        percent_seasons = df[df['voting_rule'] == 'percent'].copy()
        
        # ----- 争议案例提取 -----
        controversy_names = ['Jerry Rice', 'Billy Ray Cyrus', 'Bristol Palin', 'Bobby Bones']
        controversy_cases = df[df['celebrity_name'].isin(controversy_names)].copy()
        
        # ----- 为每个季节计算Kendall τ所需数据 -----
        season_rankings = []
        
        for season in sorted(df['season'].unique()):
            season_df = df[df['season'] == season].copy()
            
            # 计算评委分排名
            season_df['judge_rank'] = season_df['avg_score_all_weeks'].rank(ascending=False, method='min')
            
            # 最终排名
            season_df['final_rank'] = season_df['placement']
            
            season_rankings.append({
                'season': int(season),
                'voting_rule': self.get_voting_rule(season),
                'voting_phase': self.get_voting_phase(season),
                'contestants': season_df['celebrity_name'].tolist(),
                'judge_ranks': season_df['judge_rank'].tolist(),
                'final_ranks': season_df['final_rank'].tolist(),
                'n_contestants': len(season_df)
            })
        
        # ----- Bootstrap敏感性分析所需数据 -----
        bootstrap_data = {
            'all_seasons': season_rankings,
            'rank_rule_seasons': [s for s in season_rankings if s['voting_rule'] == 'rank'],
            'percent_rule_seasons': [s for s in season_rankings if s['voting_rule'] == 'percent']
        }
        
        # ----- 汇总输出数据 -----
        output_data = {
            'rank_seasons_summary': {
                'n_seasons': len(rank_seasons['season'].unique()),
                'seasons': sorted(rank_seasons['season'].unique().tolist()),
                'total_contestants': len(rank_seasons)
            },
            'percent_seasons_summary': {
                'n_seasons': len(percent_seasons['season'].unique()),
                'seasons': sorted(percent_seasons['season'].unique().tolist()),
                'total_contestants': len(percent_seasons)
            },
            'controversy_cases': controversy_cases[['celebrity_name', 'season', 'placement', 'voting_rule', 'avg_score_all_weeks']].to_dict('records'),
            'season_rankings': season_rankings,
            'bootstrap_data': bootstrap_data
        }
        
        # 保存数据
        output_path = self.output_dir / 'models' / 'q2_kendall_bootstrap_data.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存详细CSV数据
        rank_seasons.to_csv(self.output_dir / 'data' / 'q2_rank_seasons.csv', index=False)
        percent_seasons.to_csv(self.output_dir / 'data' / 'q2_percent_seasons.csv', index=False)
        controversy_cases.to_csv(self.output_dir / 'data' / 'q2_controversy_cases.csv', index=False)
        
        print(f"✅ 问题二（Kendall τ + Bootstrap）数据准备完成!")
        print(f"   - 排名制季节: {len(rank_seasons['season'].unique())} 季")
        print(f"   - 百分比制季节: {len(percent_seasons['season'].unique())} 季")
        print(f"   - 争议案例: {len(controversy_cases)} 条")
        print(f"   - 输出路径: {output_path}")
        
        return output_data
    
    # ========================================================================
    # 第5部分：问题三数据准备
    # ========================================================================
    
    def prepare_q3_lmem_data(self) -> Tuple[pd.DataFrame, Dict]:
        """
        问题三方案一：线性混合效应模型(LMEM)数据准备
        
        输出:
        - 特征矩阵 X (固定效应 + 随机效应标识)
        - 目标变量 y (最终排名 / 平均评分)
        """
        print("\n" + "=" * 60)
        print("第6步：准备问题三（LMEM）数据")
        print("=" * 60)
        
        df = self.df_clean.copy()
        
        # ----- 固定效应特征 -----
        fixed_effects = pd.DataFrame()
        
        # 年龄（连续变量，标准化）
        fixed_effects['age'] = df['celebrity_age_during_season']
        fixed_effects['age_scaled'] = (fixed_effects['age'] - fixed_effects['age'].mean()) / fixed_effects['age'].std()
        
        # 性别（根据industry推断，如有专门字段则使用）
        # 这里使用行业作为代理
        
        # 行业（类别变量，独热编码）
        industry_dummies = pd.get_dummies(df['celebrity_industry'], prefix='industry')
        
        # 季节（控制变量）
        fixed_effects['season'] = df['season']
        fixed_effects['season_scaled'] = (df['season'] - df['season'].mean()) / df['season'].std()
        
        # 投票规则阶段
        fixed_effects['voting_phase'] = df['voting_phase']
        
        # ----- 随机效应标识 -----
        random_effects = pd.DataFrame()
        random_effects['partner_id'] = df['partner_id']
        random_effects['partner_name'] = df['ballroom_partner']
        random_effects['season_group'] = df['season']  # 季节作为随机效应分组
        
        # ----- 目标变量 -----
        targets = pd.DataFrame()
        targets['placement'] = df['placement']  # 最终排名
        targets['avg_score'] = df['avg_score_all_weeks']  # 平均评分
        targets['last_week'] = df['last_valid_week']  # 存活周数
        
        # ----- 合并数据 -----
        lmem_data = pd.concat([
            df[['celebrity_name', 'season', 'ballroom_partner']],
            fixed_effects,
            random_effects,
            industry_dummies,
            targets
        ], axis=1)
        
        # 移除缺失值
        lmem_data = lmem_data.dropna(subset=['avg_score', 'placement'])
        
        # 保存数据
        output_path = self.output_dir / 'data' / 'q3_lmem_features.csv'
        lmem_data.to_csv(output_path, index=False)
        
        # 保存元信息
        meta_info = {
            'fixed_effects': ['age_scaled', 'season_scaled', 'voting_phase'] + [c for c in industry_dummies.columns],
            'random_effects': ['partner_id', 'season_group'],
            'target_variables': ['placement', 'avg_score', 'last_week'],
            'n_samples': len(lmem_data),
            'n_partners': df['partner_id'].nunique(),
            'n_industries': df['celebrity_industry'].nunique()
        }
        
        with open(self.output_dir / 'models' / 'q3_lmem_meta.json', 'w') as f:
            json.dump(meta_info, f, indent=2)
        
        print(f"✅ 问题三（LMEM）数据准备完成!")
        print(f"   - 样本数: {len(lmem_data)}")
        print(f"   - 固定效应数: {len(meta_info['fixed_effects'])}")
        print(f"   - 随机效应组: partner_id ({meta_info['n_partners']}个), season")
        print(f"   - 输出路径: {output_path}")
        
        return lmem_data, meta_info
    
    def prepare_q3_xgboost_shap_data(self) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """
        问题三方案二：XGBoost + SHAP 可解释性分析数据准备
        
        输出:
        - X: 特征矩阵
        - y: 目标变量
        - feature_info: 特征说明
        """
        print("\n" + "=" * 60)
        print("第7步：准备问题三（XGBoost + SHAP）数据")
        print("=" * 60)
        
        df = self.df_clean.copy()
        
        # ----- 构建特征矩阵 -----
        features = pd.DataFrame()
        
        # 数值特征
        features['age'] = df['celebrity_age_during_season']
        features['season'] = df['season']
        features['voting_phase'] = df['voting_phase']
        features['avg_score'] = df['avg_score_all_weeks']
        features['score_trend'] = df['score_trend']
        features['partner_id'] = df['partner_id']
        features['judge_count'] = df['judge_count']
        
        # 类别特征编码
        # 行业编码
        features['industry_encoded'] = pd.factorize(df['celebrity_industry'])[0]
        
        # 国家编码
        if 'celebrity_homecountry/region' in df.columns:
            features['is_usa'] = (df['celebrity_homecountry/region'] == 'United States').astype(int)
        
        # 计算舞伴历史胜率
        partner_win_rate = df.groupby('ballroom_partner').apply(
            lambda x: (x['placement'] <= 3).sum() / len(x)
        ).to_dict()
        features['partner_win_rate'] = df['ballroom_partner'].map(partner_win_rate)
        
        # 计算同行业历史表现
        industry_avg_placement = df.groupby('celebrity_industry')['placement'].mean().to_dict()
        features['industry_avg_placement'] = df['celebrity_industry'].map(industry_avg_placement)
        
        # ----- 目标变量 -----
        # 二分类：是否进入前3
        y_binary = (df['placement'] <= 3).astype(int)
        
        # 回归：最终排名
        y_regression = df['placement']
        
        # 移除缺失值
        valid_mask = features.notna().all(axis=1)
        X = features[valid_mask].copy()
        y_bin = y_binary[valid_mask]
        y_reg = y_regression[valid_mask]
        
        # 特征说明
        feature_info = {
            'numerical_features': ['age', 'season', 'avg_score', 'score_trend', 'partner_win_rate', 'industry_avg_placement'],
            'categorical_features': ['voting_phase', 'partner_id', 'judge_count', 'industry_encoded', 'is_usa'],
            'target_binary': 'top3 (placement <= 3)',
            'target_regression': 'placement',
            'n_samples': len(X),
            'n_features': X.shape[1]
        }
        
        # 保存数据
        X.to_csv(self.output_dir / 'data' / 'q3_xgboost_features.csv', index=False)
        pd.DataFrame({'y_binary': y_bin, 'y_regression': y_reg}).to_csv(
            self.output_dir / 'data' / 'q3_xgboost_targets.csv', index=False
        )
        
        with open(self.output_dir / 'models' / 'q3_xgboost_meta.json', 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        print(f"✅ 问题三（XGBoost + SHAP）数据准备完成!")
        print(f"   - 样本数: {len(X)}")
        print(f"   - 特征数: {X.shape[1]}")
        print(f"   - 数值特征: {len(feature_info['numerical_features'])}")
        print(f"   - 类别特征: {len(feature_info['categorical_features'])}")
        print(f"   - 输出路径: {self.output_dir / 'data' / 'q3_xgboost_features.csv'}")
        
        return X, y_reg, feature_info
    
    # ========================================================================
    # 第6部分：问题四数据准备
    # ========================================================================
    
    def prepare_q4_nsga2_data(self) -> Dict:
        """
        问题四：NSGA-II + 帕累托前沿 多目标优化数据准备
        
        三个优化目标:
        1. 公平性（评委分与最终结果的相关性）
        2. 稳定性（结果对投票噪声的敏感度）
        3. 娱乐性（悬念程度/冷门概率）
        
        决策变量:
        - w: 评委分权重 [0, 1]
        - threshold: 淘汰阈值
        """
        print("\n" + "=" * 60)
        print("第8步：准备问题四（NSGA-II）数据")
        print("=" * 60)
        
        df = self.df_clean.copy()
        
        # ----- 历史数据统计（用于目标函数计算） -----
        
        # 1. 计算各季节评委分与最终排名的相关性
        season_correlations = []
        for season in df['season'].unique():
            season_df = df[df['season'] == season]
            if len(season_df) > 3:
                corr = season_df['avg_score_all_weeks'].corr(season_df['placement'])
                season_correlations.append({
                    'season': int(season),
                    'judge_placement_corr': corr,
                    'voting_rule': self.get_voting_rule(season)
                })
        
        # 2. 计算"冷门"频率（评委分高但排名差的选手）
        df['is_upset'] = ((df['avg_score_all_weeks'] > df['avg_score_all_weeks'].median()) & 
                         (df['placement'] > df['placement'].median())).astype(int)
        
        upset_rate_by_rule = df.groupby('voting_rule')['is_upset'].mean().to_dict()
        
        # 3. 准备模拟器所需数据
        simulation_data = []
        for season in sorted(df['season'].unique()):
            season_df = df[df['season'] == season].copy()
            
            simulation_data.append({
                'season': int(season),
                'voting_rule': self.get_voting_rule(season),
                'n_contestants': len(season_df),
                'judge_scores': season_df['avg_score_all_weeks'].tolist(),
                'final_placements': season_df['placement'].tolist(),
                'contestants': season_df['celebrity_name'].tolist()
            })
        
        # ----- 决策变量边界 -----
        decision_bounds = {
            'w_judge': [0.0, 1.0],      # 评委权重
            'w_fan': [0.0, 1.0],        # 粉丝权重（= 1 - w_judge）
            'threshold_low': [0.0, 0.3], # 低分淘汰阈值
            'save_probability': [0.0, 0.5]  # 评委拯救概率
        }
        
        # ----- 汇总输出 -----
        nsga2_data = {
            'objectives': {
                'fairness': '评委分与最终排名的Spearman相关系数',
                'stability': '结果对投票噪声的敏感度（标准差）',
                'entertainment': '悬念程度（冷门概率）'
            },
            'decision_variables': decision_bounds,
            'historical_data': {
                'season_correlations': season_correlations,
                'upset_rate_by_rule': upset_rate_by_rule,
                'simulation_data': simulation_data
            },
            'constraints': {
                'w_sum': 'w_judge + w_fan = 1',
                'elimination_rule': '每周至少淘汰1人（除决赛）'
            }
        }
        
        # 保存数据
        output_path = self.output_dir / 'models' / 'q4_nsga2_data.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(nsga2_data, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存详细模拟数据
        pd.DataFrame(season_correlations).to_csv(
            self.output_dir / 'data' / 'q4_season_correlations.csv', index=False
        )
        
        print(f"✅ 问题四（NSGA-II）数据准备完成!")
        print(f"   - 优化目标数: 3 (公平性/稳定性/娱乐性)")
        print(f"   - 决策变量数: 4")
        print(f"   - 历史季节数: {len(simulation_data)}")
        print(f"   - 输出路径: {output_path}")
        
        return nsga2_data
    
    # ========================================================================
    # 第7部分：通用数据输出
    # ========================================================================
    
    def save_general_data(self):
        """保存通用预处理后数据"""
        print("\n" + "=" * 60)
        print("第9步：保存通用预处理数据")
        print("=" * 60)
        
        df = self.df_clean
        
        # 保存完整数据
        df.to_csv(self.output_dir / 'data' / 'dwts_preprocessed_full.csv', index=False)
        
        # 按季节划分数据
        # 训练集: S3-24 (百分比制主体)
        # 验证集: S25-27 (百分比制末期)
        # 测试集: S28-34 (新规则)
        
        train_df = df[df['season'].between(3, 24)]
        val_df = df[df['season'].between(25, 27)]
        test_df = df[df['season'] >= 28]
        
        train_df.to_csv(self.output_dir / 'data' / 'dwts_train_by_season.csv', index=False)
        val_df.to_csv(self.output_dir / 'data' / 'dwts_val_by_season.csv', index=False)
        test_df.to_csv(self.output_dir / 'data' / 'dwts_test_by_season.csv', index=False)
        
        print(f"✅ 通用数据保存完成!")
        print(f"   - 完整数据: {len(df)} 条")
        print(f"   - 训练集(S3-24): {len(train_df)} 条")
        print(f"   - 验证集(S25-27): {len(val_df)} 条")
        print(f"   - 测试集(S28-34): {len(test_df)} 条")
    
    # ========================================================================
    # 第8部分：数据可视化
    # ========================================================================
    
    def generate_visualizations(self):
        """生成预处理可视化图表"""
        print("\n" + "=" * 60)
        print("第10步：生成可视化图表")
        print("=" * 60)
        
        df = self.df_clean
        fig_dir = self.output_dir / 'figures'
        
        # 设置图表风格
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # ----- 图1: 季节数据分布 -----
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 每季选手数
        ax1 = axes[0, 0]
        season_counts = df.groupby('season').size()
        colors = ['#3498db' if self.get_voting_rule(s) == 'percent' else '#e74c3c' 
                  for s in season_counts.index]
        season_counts.plot(kind='bar', ax=ax1, color=colors)
        ax1.set_title('Number of Contestants per Season', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Season')
        ax1.set_ylabel('Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # 评委数量分布
        ax2 = axes[0, 1]
        judge_counts = df.groupby('season')['judge_count'].first()
        judge_counts.plot(kind='bar', ax=ax2, color='#2ecc71')
        ax2.set_title('Judge Count per Season', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Season')
        ax2.set_ylabel('Number of Judges')
        ax2.tick_params(axis='x', rotation=45)
        
        # 投票规则分布
        ax3 = axes[1, 0]
        voting_dist = df['voting_rule'].value_counts()
        voting_dist.plot(kind='pie', ax=ax3, autopct='%1.1f%%', colors=['#3498db', '#e74c3c'])
        ax3.set_title('Distribution of Voting Rules', fontsize=12, fontweight='bold')
        
        # 年龄分布
        ax4 = axes[1, 1]
        df['celebrity_age_during_season'].hist(ax=ax4, bins=20, color='#9b59b6', edgecolor='white')
        ax4.set_title('Age Distribution of Contestants', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Age')
        ax4.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'data_overview.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # ----- 图2: 评委分与排名关系 -----
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 按投票规则分色
        for rule, color, ax in zip(['percent', 'rank'], ['#3498db', '#e74c3c'], axes):
            rule_df = df[df['voting_rule'] == rule]
            ax.scatter(rule_df['avg_score_all_weeks'], rule_df['placement'], 
                      alpha=0.6, c=color, s=50)
            ax.set_xlabel('Average Judge Score', fontsize=11)
            ax.set_ylabel('Final Placement', fontsize=11)
            ax.set_title(f'{rule.title()} System: Score vs Placement', fontsize=12, fontweight='bold')
            ax.invert_yaxis()  # 排名越小越好
            
            # 添加相关系数
            corr = rule_df['avg_score_all_weeks'].corr(rule_df['placement'])
            ax.annotate(f'r = {corr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                       fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'score_placement_correlation.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # ----- 图3: 舞伴影响分析 -----
        fig, ax = plt.subplots(figsize=(12, 6))
        
        partner_stats = df.groupby('ballroom_partner').agg({
            'placement': 'mean',
            'season': 'count'
        }).rename(columns={'season': 'appearances'})
        
        # 只显示出场次数>=3的舞伴
        top_partners = partner_stats[partner_stats['appearances'] >= 3].sort_values('placement')
        
        bars = ax.barh(range(len(top_partners)), top_partners['placement'], color='#1abc9c')
        ax.set_yticks(range(len(top_partners)))
        ax.set_yticklabels(top_partners.index)
        ax.set_xlabel('Average Placement (lower is better)', fontsize=11)
        ax.set_title('Professional Partners Performance (≥3 appearances)', fontsize=12, fontweight='bold')
        ax.invert_xaxis()
        
        plt.tight_layout()
        plt.savefig(fig_dir / 'partner_performance.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 可视化图表生成完成!")
        print(f"   - 图表数量: 3")
        print(f"   - 输出目录: {fig_dir}")
    
    # ========================================================================
    # 主流程
    # ========================================================================
    
    def run_all(self):
        """运行完整预处理流程"""
        print("\n" + "=" * 70)
        print("DWTS 数据预处理完整流程")
        print("=" * 70)
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 数据清洗
        self.clean_data()
        
        # 3. 问题一数据准备
        self.prepare_q1_constraint_optimization_data()
        self.prepare_q1_bayesian_mcmc_data()
        
        # 4. 问题二数据准备
        self.prepare_q2_kendall_bootstrap_data()
        
        # 5. 问题三数据准备
        self.prepare_q3_lmem_data()
        self.prepare_q3_xgboost_shap_data()
        
        # 6. 问题四数据准备
        self.prepare_q4_nsga2_data()
        
        # 7. 保存通用数据
        self.save_general_data()
        
        # 8. 生成可视化
        self.generate_visualizations()
        
        # 9. 生成数据报告
        self._generate_report()
        
        print("\n" + "=" * 70)
        print("✅✅✅ 所有数据预处理完成! ✅✅✅")
        print("=" * 70)
        print(f"\n输出目录: {self.output_dir}")
        print("\n输出文件清单:")
        self._print_output_files()
    
    def _generate_report(self):
        """生成数据预处理报告"""
        report = f"""
================================================================================
DWTS 数据预处理报告
MCM 2026 Problem C
================================================================================

一、数据概览
-----------
原始数据: {self.input_path}
数据维度: {self.df_raw.shape[0]} 行 × {self.df_raw.shape[1]} 列
季节范围: S{self.df_raw['season'].min()} - S{self.df_raw['season'].max()}

二、数据清洗结果
---------------
清洗后维度: {self.df_clean.shape[0]} 行 × {self.df_clean.shape[1]} 列
新增字段: judge_count, voting_rule, voting_phase, elimination_week, 
         last_valid_week, partner_id, avg_score_all_weeks, score_trend

三、投票规则分布
---------------
排名制(S1-2, S28-34): {len(self.df_clean[self.df_clean['voting_rule'] == 'rank'])} 条
百分比制(S3-27): {len(self.df_clean[self.df_clean['voting_rule'] == 'percent'])} 条

四、输出文件
-----------
详见 {self.output_dir} 目录

五、各模型数据说明
-----------------
问题一（约束优化）: models/q1_constraint_optimization_data.json
问题一（贝叶斯MCMC）: models/q1_bayesian_mcmc_data.json
问题二（Kendall τ）: models/q2_kendall_bootstrap_data.json
问题三（LMEM）: data/q3_lmem_features.csv
问题三（XGBoost）: data/q3_xgboost_features.csv
问题四（NSGA-II）: models/q4_nsga2_data.json
"""
        
        with open(self.output_dir / 'preprocessing_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
    
    def _print_output_files(self):
        """打印输出文件列表"""
        for subdir in ['data', 'models', 'figures']:
            dir_path = self.output_dir / subdir
            if dir_path.exists():
                print(f"\n  📁 {subdir}/")
                for file in sorted(dir_path.iterdir()):
                    size = file.stat().st_size / 1024  # KB
                    print(f"     📄 {file.name} ({size:.1f} KB)")


# ============================================================================
# 数据验证函数
# ============================================================================

def verify_preprocessed_data(output_dir: str):
    """
    验证预处理后数据的完整性
    
    Args:
        output_dir: 输出目录路径
    """
    print("\n" + "=" * 60)
    print("数据完整性验证")
    print("=" * 60)
    
    output_dir = Path(output_dir)
    
    required_files = [
        'data/dwts_preprocessed_full.csv',
        'data/q3_lmem_features.csv',
        'data/q3_xgboost_features.csv',
        'models/q1_constraint_optimization_data.json',
        'models/q1_bayesian_mcmc_data.json',
        'models/q2_kendall_bootstrap_data.json',
        'models/q4_nsga2_data.json'
    ]
    
    all_valid = True
    for file in required_files:
        file_path = output_dir / file
        if file_path.exists():
            size = file_path.stat().st_size / 1024
            print(f"  ✅ {file} ({size:.1f} KB)")
        else:
            print(f"  ❌ {file} - 文件缺失!")
            all_valid = False
    
    if all_valid:
        print("\n✅ 所有必需文件验证通过!")
    else:
        print("\n⚠️ 部分文件缺失，请检查预处理流程!")
    
    return all_valid


# ============================================================================
# 数据预览函数
# ============================================================================

def preview_data(output_dir: str):
    """
    预览处理后数据的前10行和后5行
    
    Args:
        output_dir: 输出目录路径
    """
    output_dir = Path(output_dir)
    
    print("\n" + "=" * 60)
    print("处理后数据预览")
    print("=" * 60)
    
    # 预览完整数据
    full_data_path = output_dir / 'data' / 'dwts_preprocessed_full.csv'
    if full_data_path.exists():
        df = pd.read_csv(full_data_path)
        
        print("\n📊 完整数据 (dwts_preprocessed_full.csv)")
        print("-" * 50)
        print(f"维度: {df.shape[0]} 行 × {df.shape[1]} 列")
        print("\n前10行:")
        print(df.head(10).to_string())
        print("\n后5行:")
        print(df.tail(5).to_string())


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║         DWTS 数据预处理程序                                    ║
    ║         MCM 2026 Problem C: Data With The Stars               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # ⚠️ 请确保已修改文件顶部的路径设置
    print(f"📂 输入文件: {INPUT_DATA_PATH}")
    print(f"📂 输出目录: {OUTPUT_DIR}")
    print("-" * 60)
    
    # 检查输入文件是否存在
    if not Path(INPUT_DATA_PATH).exists():
        print(f"\n❌ 错误: 找不到输入文件 '{INPUT_DATA_PATH}'")
        print("请修改文件顶部的 INPUT_DATA_PATH 变量为正确的数据文件路径。")
        exit(1)
    
    # 创建预处理器并运行
    preprocessor = DWTSDataPreprocessor(INPUT_DATA_PATH, OUTPUT_DIR)
    preprocessor.run_all()
    
    # 验证输出
    verify_preprocessed_data(OUTPUT_DIR)
    
    # 预览数据
    preview_data(OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("程序执行完毕!")
    print("=" * 60)
