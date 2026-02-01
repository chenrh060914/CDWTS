#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
DWTS 粉丝数据分析模块
MCM 2026 Problem C: Dancing with the Stars
================================================================================
版本: v1.0
日期: 2026-02-01
说明: 基于补充的社交媒体粉丝量数据进行增强分析

功能:
1. 一、估算粉丝投票 - 使用粉丝量作为代理变量
2. 二、对比两种投票组合方法 - 排名法 vs 百分比法
3. 三、分析争议选手案例 - Jerry Rice、Bristol Palin、Bobby Bones等
4. 四、评估选手/舞伴特征影响 - 舞伴社交媒体带动作用
5. 五、设计更公平的投票系统 - 基于粉丝量设定权重
================================================================================
"""

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Optional
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# 配置参数
# ============================================================================
class Config:
    """全局配置"""
    # 数据路径
    ORIGINAL_DATA_PATH = "./2026_MCM_Problem_C_Data.csv"
    FAN_DATA_PATH = "./2026美赛C题补充数据集！.xlsx"
    OUTPUT_DIR = "./fan_analysis_output"
    
    # 争议选手名单
    CONTROVERSIAL_CELEBRITIES = [
        "Jerry Rice",       # S2: 评委低分但获亚军
        "Billy Ray Cyrus",  # S4: 争议案例
        "Bristol Palin",    # S11, S15: 低分但晋级深
        "Bobby Bones",      # S27: 低分夺冠
    ]
    
    # 季节投票规则
    # S1-2: 排名法, S3-27: 百分比法, S28+: 排名法+评委二选一
    RANK_SEASONS = list(range(1, 3)) + list(range(28, 35))
    PERCENT_SEASONS = list(range(3, 28))
    
    # 随机种子
    RANDOM_SEED = 42

np.random.seed(Config.RANDOM_SEED)


# ============================================================================
# 数据加载与整合
# ============================================================================
class FanDataIntegrator:
    """粉丝数据整合器"""
    
    def __init__(self):
        self.df_original = None
        self.df_fan = None
        self.df_merged = None
        
    def load_data(self):
        """加载原始数据和粉丝数据"""
        print("=" * 60)
        print("加载数据...")
        
        # 加载原始数据
        self.df_original = pd.read_csv(Config.ORIGINAL_DATA_PATH)
        print(f"原始数据: {self.df_original.shape[0]} 行, {self.df_original.shape[1]} 列")
        
        # 加载粉丝数据
        self.df_fan = pd.read_excel(Config.FAN_DATA_PATH, sheet_name='enriched')
        print(f"粉丝数据: {self.df_fan.shape[0]} 行, {self.df_fan.shape[1]} 列")
        
        # 查看粉丝数据列
        print(f"\n粉丝数据列: {list(self.df_fan.columns)}")
        
        # 非空统计
        print("\n粉丝数据非空统计:")
        print(self.df_fan.notna().sum())
        
        return self
    
    def merge_data(self):
        """合并原始数据和粉丝数据"""
        print("\n" + "=" * 60)
        print("合并数据...")
        
        # 原始数据处理
        df_orig = self.df_original.copy()
        
        # 获取每位选手的唯一记录（去重）
        df_contestants = df_orig.drop_duplicates(subset=['celebrity_name']).copy()
        
        # 合并粉丝数据
        self.df_merged = df_contestants.merge(
            self.df_fan,
            left_on='celebrity_name',
            right_on='celebrity_name',
            how='left'
        )
        
        print(f"合并后数据: {self.df_merged.shape[0]} 行")
        print(f"有粉丝数据的选手: {self.df_merged['celebrity_total_followers_wikidata'].notna().sum()} 人")
        
        # 计算总粉丝量（选手+舞伴）
        self.df_merged['total_social_followers'] = (
            self.df_merged['celebrity_total_followers_wikidata'].fillna(0) +
            self.df_merged['partner_total_followers_wikidata'].fillna(0)
        )
        
        return self
    
    def get_season_data(self, season: int) -> pd.DataFrame:
        """获取特定季节的数据"""
        return self.df_original[self.df_original['season'] == season].copy()
    
    def get_contestant_followers(self, name: str) -> Dict:
        """获取特定选手的粉丝信息"""
        row = self.df_fan[self.df_fan['celebrity_name'] == name]
        if len(row) == 0:
            return {}
        row = row.iloc[0]
        return {
            'celebrity_instagram': row.get('celebrity_instagram_followers', np.nan),
            'celebrity_twitter': row.get('celebrity_twitter_followers', np.nan),
            'celebrity_tiktok': row.get('celebrity_tiktok_followers', np.nan),
            'celebrity_youtube': row.get('celebrity_youtube_subscribers', np.nan),
            'celebrity_total': row.get('celebrity_total_followers_wikidata', np.nan),
            'partner_instagram': row.get('partner_instagram_followers', np.nan),
            'partner_twitter': row.get('partner_twitter_followers', np.nan),
            'partner_tiktok': row.get('partner_tiktok_followers', np.nan),
            'partner_youtube': row.get('partner_youtube_subscribers', np.nan),
            'partner_total': row.get('partner_total_followers_wikidata', np.nan),
        }


# ============================================================================
# 一、估算粉丝投票
# ============================================================================
class FanVoteEstimator:
    """基于粉丝量估算粉丝投票"""
    
    def __init__(self, integrator: FanDataIntegrator):
        self.integrator = integrator
        self.results = {}
        
    def estimate_fan_vote_by_followers(self, season: int) -> pd.DataFrame:
        """
        基于粉丝量估算粉丝投票比例
        
        假设: 粉丝投票与社交媒体粉丝量成正比
        公式: FanVote_i = Followers_i / Σ Followers_j
        """
        df_season = self.integrator.get_season_data(season)
        contestants = df_season.drop_duplicates(subset=['celebrity_name'])['celebrity_name'].tolist()
        
        # 获取粉丝数据
        follower_data = []
        for name in contestants:
            info = self.integrator.get_contestant_followers(name)
            total = info.get('celebrity_total', 0) or 0
            partner_total = info.get('partner_total', 0) or 0
            follower_data.append({
                'celebrity_name': name,
                'celebrity_followers': total,
                'partner_followers': partner_total,
                'combined_followers': total + partner_total
            })
        
        df_followers = pd.DataFrame(follower_data)
        
        if len(df_followers) == 0:
            return df_followers
        
        # 计算粉丝投票比例
        total_followers = df_followers['combined_followers'].sum()
        if total_followers > 0:
            df_followers['estimated_fan_vote_pct'] = (
                df_followers['combined_followers'] / total_followers * 100
            )
        else:
            # 无粉丝数据时使用均匀分布
            df_followers['estimated_fan_vote_pct'] = 100.0 / len(df_followers)
        
        # 排名
        df_followers['fan_vote_rank'] = df_followers['estimated_fan_vote_pct'].rank(
            ascending=False, method='min', na_option='bottom'
        ).fillna(len(df_followers)).astype(int)
        
        return df_followers
    
    def compare_with_elimination(self, season: int) -> Dict:
        """
        对比粉丝投票估计与实际淘汰结果
        
        验证粉丝量是否能解释淘汰顺序
        """
        df_season = self.integrator.get_season_data(season)
        
        # 获取实际名次
        placements = df_season.drop_duplicates(subset=['celebrity_name']).set_index('celebrity_name')['placement'].to_dict()
        
        # 获取粉丝投票估计
        df_estimate = self.estimate_fan_vote_by_followers(season)
        
        # 合并
        results = []
        for _, row in df_estimate.iterrows():
            name = row['celebrity_name']
            actual_placement = placements.get(name, np.nan)
            results.append({
                'celebrity_name': name,
                'combined_followers': row['combined_followers'],
                'estimated_fan_vote_pct': row['estimated_fan_vote_pct'],
                'fan_vote_rank': row['fan_vote_rank'],
                'actual_placement': actual_placement
            })
        
        df_results = pd.DataFrame(results)
        
        # 计算相关性
        valid_data = df_results.dropna(subset=['actual_placement'])
        if len(valid_data) >= 3 and valid_data['combined_followers'].sum() > 0:
            # 粉丝多 -> 名次好(数值小)，所以应该是负相关
            corr, pval = spearmanr(
                valid_data['combined_followers'],
                valid_data['actual_placement']
            )
        else:
            corr, pval = np.nan, np.nan
        
        return {
            'season': season,
            'contestants': df_results.to_dict('records'),
            'follower_placement_correlation': corr,
            'correlation_pvalue': pval,
            'interpretation': '粉丝量与名次呈负相关(粉丝多->名次好)' if corr < 0 else '相关性为正或不显著'
        }
    
    def analyze_all_seasons(self) -> Dict:
        """分析所有季节的粉丝投票估计"""
        all_results = {}
        correlations = []
        
        for season in range(1, 35):
            result = self.compare_with_elimination(season)
            all_results[f'season_{season}'] = result
            if not np.isnan(result['follower_placement_correlation']):
                correlations.append(result['follower_placement_correlation'])
        
        summary = {
            'mean_correlation': np.mean(correlations) if correlations else np.nan,
            'std_correlation': np.std(correlations) if correlations else np.nan,
            'n_seasons_with_data': len(correlations),
            'seasons': all_results
        }
        
        self.results = summary
        return summary


# ============================================================================
# 二、对比两种投票组合方法
# ============================================================================
class VotingMethodComparator:
    """对比排名法和百分比法"""
    
    def __init__(self, integrator: FanDataIntegrator):
        self.integrator = integrator
        self.results = {}
        
    def simulate_voting_methods(self, season: int, df_week: pd.DataFrame) -> Dict:
        """
        模拟两种投票方法的结果
        
        排名法: 评委排名 + 粉丝排名 (数值小=排名高)
        百分比法: 评委百分比 + 粉丝百分比 (数值大=好)
        """
        # 获取粉丝数据
        contestants = df_week['celebrity_name'].unique().tolist()
        follower_data = {}
        for name in contestants:
            info = self.integrator.get_contestant_followers(name)
            total = (info.get('celebrity_total', 0) or 0) + (info.get('partner_total', 0) or 0)
            follower_data[name] = total
        
        # 计算评委分数（使用week1的分数）
        week1_judge_cols = ['week1_judge1_score', 'week1_judge2_score', 'week1_judge3_score', 'week1_judge4_score']
        df_week = df_week.copy()
        
        # 处理缺失值
        for col in week1_judge_cols:
            if col in df_week.columns:
                df_week[col] = pd.to_numeric(df_week[col], errors='coerce').fillna(0)
        
        available_judge_cols = [c for c in week1_judge_cols if c in df_week.columns]
        df_week['total_judge_score'] = df_week[available_judge_cols].sum(axis=1)
        
        # 排名法
        df_week['judge_rank'] = df_week['total_judge_score'].rank(ascending=False, method='min', na_option='bottom').fillna(len(df_week)).astype(int)
        
        # 粉丝排名（基于粉丝量）
        df_week['follower_count'] = df_week['celebrity_name'].map(follower_data).fillna(0)
        df_week['fan_rank'] = df_week['follower_count'].rank(ascending=False, method='min', na_option='bottom').fillna(len(df_week)).astype(int)
        
        # 排名法综合分数（越小越好）
        df_week['rank_combined'] = df_week['judge_rank'] + df_week['fan_rank']
        df_week['rank_method_result'] = df_week['rank_combined'].rank(ascending=True, method='min', na_option='bottom').fillna(len(df_week)).astype(int)
        
        # 百分比法
        total_judge = df_week['total_judge_score'].sum()
        total_followers = df_week['follower_count'].sum()
        
        if total_judge > 0:
            df_week['judge_pct'] = df_week['total_judge_score'] / total_judge * 50
        else:
            df_week['judge_pct'] = 50 / len(df_week)
            
        if total_followers > 0:
            df_week['fan_pct'] = df_week['follower_count'] / total_followers * 50
        else:
            df_week['fan_pct'] = 50 / len(df_week)
        
        # 百分比法综合分数（越大越好）
        df_week['pct_combined'] = df_week['judge_pct'] + df_week['fan_pct']
        df_week['pct_method_result'] = df_week['pct_combined'].rank(ascending=False, method='min', na_option='bottom').fillna(len(df_week)).astype(int)
        
        return df_week
    
    def analyze_method_bias(self) -> Dict:
        """
        分析两种方法对粉丝偏好的倾斜程度
        
        计算粉丝量高的选手在两种方法下的排名差异
        """
        rank_system_correlations = []
        pct_system_correlations = []
        
        for season in range(1, 35):
            df_season = self.integrator.get_season_data(season)
            
            # 使用第一周数据做分析
            if 'week' in df_season.columns:
                df_week1 = df_season[df_season['week'] == 1]
            else:
                df_week1 = df_season
            
            if len(df_week1) < 3:
                continue
            
            df_sim = self.simulate_voting_methods(season, df_week1)
            
            # 有粉丝数据的选手
            valid = df_sim[df_sim['follower_count'] > 0]
            if len(valid) < 3:
                continue
            
            # 粉丝量与最终排名的相关性
            # 排名法: 粉丝多 -> rank_method_result 小 -> 负相关
            corr_rank, _ = spearmanr(valid['follower_count'], valid['rank_method_result'])
            corr_pct, _ = spearmanr(valid['follower_count'], valid['pct_method_result'])
            
            rank_system_correlations.append(corr_rank)
            pct_system_correlations.append(corr_pct)
        
        # 排名法的相关性应该更负（粉丝量对结果影响大）
        results = {
            'rank_system': {
                'mean_follower_result_correlation': np.mean(rank_system_correlations),
                'std': np.std(rank_system_correlations),
                'n_seasons': len(rank_system_correlations)
            },
            'percentage_system': {
                'mean_follower_result_correlation': np.mean(pct_system_correlations),
                'std': np.std(pct_system_correlations),
                'n_seasons': len(pct_system_correlations)
            },
            'analysis': {
                'more_fan_biased': 'rank' if np.mean(rank_system_correlations) < np.mean(pct_system_correlations) else 'percentage',
                'difference': abs(np.mean(rank_system_correlations) - np.mean(pct_system_correlations)),
                'interpretation': '排名法更偏向粉丝投票' if np.mean(rank_system_correlations) < np.mean(pct_system_correlations) else '百分比法更偏向粉丝投票'
            }
        }
        
        self.results = results
        return results


# ============================================================================
# 三、分析争议选手案例
# ============================================================================
class ControversyCaseAnalyzer:
    """分析争议选手案例"""
    
    def __init__(self, integrator: FanDataIntegrator):
        self.integrator = integrator
        self.results = {}
    
    def _get_judge_score_cols(self) -> List[str]:
        """获取所有评委分数列"""
        cols = []
        for week in range(1, 12):
            for judge in range(1, 5):
                cols.append(f'week{week}_judge{judge}_score')
        return cols
    
    def _calculate_avg_score(self, df_row: pd.Series) -> float:
        """计算选手的平均评委分数"""
        score_cols = self._get_judge_score_cols()
        scores = []
        for col in score_cols:
            if col in df_row.index:
                val = pd.to_numeric(df_row[col], errors='coerce')
                if pd.notna(val) and val > 0:
                    scores.append(val)
        return np.mean(scores) if scores else 0
        
    def analyze_controversial_contestant(self, name: str) -> Dict:
        """
        分析单个争议选手
        
        量化其"人气优势"，解释为何评委分低仍能晋级
        """
        # 获取选手信息
        df = self.integrator.df_original
        df_contestant = df[df['celebrity_name'] == name]
        
        if len(df_contestant) == 0:
            return {'name': name, 'error': '未找到该选手'}
        
        season = df_contestant['season'].iloc[0]
        placement = df_contestant['placement'].iloc[0]
        
        # 获取粉丝数据
        fan_info = self.integrator.get_contestant_followers(name)
        celebrity_followers = fan_info.get('celebrity_total', 0) or 0
        partner_followers = fan_info.get('partner_total', 0) or 0
        total_followers = celebrity_followers + partner_followers
        
        # 计算评委评分
        avg_score = self._calculate_avg_score(df_contestant.iloc[0])
        
        # 获取同季选手进行对比
        df_season = df[df['season'] == season]
        all_contestants = df_season.drop_duplicates(subset=['celebrity_name'])
        
        # 计算相对位置
        relative_data = []
        for _, row in all_contestants.iterrows():
            other_name = row['celebrity_name']
            other_placement = row['placement']
            other_fan = self.integrator.get_contestant_followers(other_name)
            other_total = (other_fan.get('celebrity_total', 0) or 0) + (other_fan.get('partner_total', 0) or 0)
            
            # 评分
            other_avg = self._calculate_avg_score(row)
            
            relative_data.append({
                'name': other_name,
                'placement': other_placement,
                'total_followers': other_total,
                'avg_score': other_avg
            })
        
        df_relative = pd.DataFrame(relative_data)
        
        # 计算排名
        df_relative['score_rank'] = df_relative['avg_score'].rank(ascending=False, method='min')
        df_relative['follower_rank'] = df_relative['total_followers'].rank(ascending=False, method='min', na_option='bottom')
        
        n_contestants = len(df_relative)
        
        # 争议选手的排名
        contestant_row = df_relative[df_relative['name'] == name]
        if len(contestant_row) == 0:
            return {'name': name, 'error': '未找到该选手排名信息'}
        contestant_row = contestant_row.iloc[0]
        score_rank = contestant_row['score_rank'] if pd.notna(contestant_row['score_rank']) else n_contestants
        follower_rank = contestant_row['follower_rank'] if pd.notna(contestant_row['follower_rank']) else n_contestants
        
        # 粉丝优势计算
        if df_relative['total_followers'].sum() > 0 and total_followers > 0:
            follower_pct = total_followers / df_relative['total_followers'].sum() * 100
        else:
            follower_pct = 0
        
        # 是否争议性
        is_controversial = (
            (score_rank > n_contestants * 0.6) and  # 评分排名后40%
            (placement <= n_contestants * 0.4)       # 但名次在前40%
        )
        
        # 确保没有NaN
        total_followers_safe = int(total_followers) if pd.notna(total_followers) and total_followers > 0 else 0
        celebrity_followers_safe = int(celebrity_followers) if pd.notna(celebrity_followers) else 0
        partner_followers_safe = int(partner_followers) if pd.notna(partner_followers) else 0
        
        result = {
            'name': name,
            'season': int(season),
            'placement': int(placement),
            'n_contestants': n_contestants,
            'avg_judge_score': round(avg_score, 2) if pd.notna(avg_score) else 0,
            'score_rank': int(score_rank),
            'total_followers': total_followers_safe,
            'follower_rank': int(follower_rank),
            'follower_percentage': round(follower_pct, 2) if pd.notna(follower_pct) else 0,
            'is_controversial': bool(is_controversial),
            'explanation': '',
            'fan_advantage': {
                'celebrity_followers': celebrity_followers_safe,
                'partner_followers': partner_followers_safe,
                'total': total_followers_safe
            }
        }
        
        # 解释
        if is_controversial:
            result['explanation'] = (
                f"{name}虽然评委评分排名第{int(score_rank)}（共{n_contestants}人），"
                f"但凭借{follower_pct:.1f}%的粉丝占比（排名第{int(follower_rank)}），"
                f"最终获得第{int(placement)}名。粉丝量优势抵消了评分劣势。"
            )
        else:
            result['explanation'] = f"{name}的表现与评分排名基本一致，不存在明显争议。"
        
        return result
    
    def analyze_all_controversies(self) -> Dict:
        """分析所有争议选手"""
        cases = []
        
        for name in Config.CONTROVERSIAL_CELEBRITIES:
            result = self.analyze_controversial_contestant(name)
            cases.append(result)
        
        # 额外分析：找出所有潜在争议选手
        df = self.integrator.df_original
        potential_controversies = []
        
        for season in range(1, 35):
            df_season = df[df['season'] == season].drop_duplicates(subset=['celebrity_name'])
            n = len(df_season)
            
            for _, row in df_season.iterrows():
                name = row['celebrity_name']
                placement = row['placement']
                
                # 计算平均评分
                avg_score = self._calculate_avg_score(row)
                
                # 检查是否争议性
                fan_info = self.integrator.get_contestant_followers(name)
                total_followers = (fan_info.get('celebrity_total', 0) or 0) + (fan_info.get('partner_total', 0) or 0)
                
                if avg_score < 22 and placement <= 3 and total_followers > 1000000:
                    potential_controversies.append({
                        'name': name,
                        'season': season,
                        'placement': placement,
                        'avg_score': avg_score,
                        'total_followers': total_followers
                    })
        
        self.results = {
            'documented_cases': cases,
            'potential_controversies': potential_controversies,
            'summary': {
                'n_documented': len(cases),
                'n_potential': len(potential_controversies),
                'common_pattern': '高粉丝量抵消低评分'
            }
        }
        
        return self.results


# ============================================================================
# 四、评估选手/舞伴特征影响
# ============================================================================
class PartnerInfluenceAnalyzer:
    """分析舞伴社交媒体带动作用"""
    
    def __init__(self, integrator: FanDataIntegrator):
        self.integrator = integrator
        self.results = {}
        
    def analyze_partner_effect(self) -> Dict:
        """
        分析专业舞伴的社交媒体影响力
        
        量化舞伴粉丝量对选手成绩的带动作用
        """
        df = self.integrator.df_original
        
        partner_data = []
        
        for _, row in df.drop_duplicates(subset=['celebrity_name']).iterrows():
            name = row['celebrity_name']
            placement = row['placement']
            partner = row.get('ballroom_partner', '')
            season = row['season']
            
            fan_info = self.integrator.get_contestant_followers(name)
            celebrity_followers = fan_info.get('celebrity_total', 0) or 0
            partner_followers = fan_info.get('partner_total', 0) or 0
            
            # 舞伴贡献比例
            total = celebrity_followers + partner_followers
            if total > 0:
                partner_contribution_pct = partner_followers / total * 100
            else:
                partner_contribution_pct = 0
            
            partner_data.append({
                'celebrity_name': name,
                'partner': partner,
                'season': season,
                'placement': placement,
                'celebrity_followers': celebrity_followers,
                'partner_followers': partner_followers,
                'total_followers': total,
                'partner_contribution_pct': partner_contribution_pct
            })
        
        df_partners = pd.DataFrame(partner_data)
        
        # 分析舞伴影响
        valid_data = df_partners[df_partners['total_followers'] > 0]
        
        # 舞伴贡献与名次的关系
        if len(valid_data) >= 3:
            corr, pval = spearmanr(
                valid_data['partner_contribution_pct'],
                valid_data['placement']
            )
        else:
            corr, pval = np.nan, np.nan
        
        # Top舞伴分析
        partner_stats = df_partners.groupby('partner').agg({
            'placement': 'mean',
            'partner_followers': 'first',
            'celebrity_name': 'count'
        }).rename(columns={'celebrity_name': 'n_seasons'})
        
        partner_stats = partner_stats[partner_stats['n_seasons'] >= 3]
        partner_stats = partner_stats.sort_values('partner_followers', ascending=False)
        
        self.results = {
            'partner_placement_correlation': corr,
            'correlation_pvalue': pval,
            'interpretation': '舞伴粉丝占比与名次相关' if pval < 0.05 else '舞伴粉丝占比与名次无显著相关',
            'mean_partner_contribution': df_partners['partner_contribution_pct'].mean(),
            'top_partners': partner_stats.head(10).to_dict('index'),
            'all_data': df_partners.to_dict('records')
        }
        
        return self.results
    
    def analyze_age_industry_effect(self) -> Dict:
        """分析年龄和行业的影响"""
        df = self.integrator.df_original
        
        contestant_data = []
        
        for _, row in df.drop_duplicates(subset=['celebrity_name']).iterrows():
            name = row['celebrity_name']
            placement = row['placement']
            age = row.get('celebrity_age_during_season', np.nan)
            industry = row.get('celebrity_industry', '')
            season = row['season']
            
            fan_info = self.integrator.get_contestant_followers(name)
            total_followers = (fan_info.get('celebrity_total', 0) or 0) + (fan_info.get('partner_total', 0) or 0)
            
            contestant_data.append({
                'celebrity_name': name,
                'season': season,
                'placement': placement,
                'age': age,
                'industry': industry,
                'total_followers': total_followers
            })
        
        df_contestants = pd.DataFrame(contestant_data)
        
        # 年龄分析
        valid_age = df_contestants.dropna(subset=['age'])
        if len(valid_age) >= 3:
            age_follower_corr, _ = pearsonr(valid_age['age'], valid_age['total_followers'])
            age_placement_corr, _ = pearsonr(valid_age['age'], valid_age['placement'])
        else:
            age_follower_corr, age_placement_corr = np.nan, np.nan
        
        # 行业分析
        industry_stats = df_contestants.groupby('industry').agg({
            'placement': 'mean',
            'total_followers': 'mean',
            'celebrity_name': 'count'
        }).rename(columns={'celebrity_name': 'n_contestants'})
        
        industry_stats = industry_stats[industry_stats['n_contestants'] >= 3]
        industry_stats = industry_stats.sort_values('total_followers', ascending=False)
        
        return {
            'age_analysis': {
                'age_follower_correlation': age_follower_corr,
                'age_placement_correlation': age_placement_corr,
                'interpretation': '年轻选手往往有更多粉丝' if age_follower_corr < 0 else '年龄与粉丝量无明显关系'
            },
            'industry_analysis': {
                'top_industries_by_followers': industry_stats.head(10).to_dict('index'),
                'interpretation': '不同行业的粉丝量差异明显'
            }
        }


# ============================================================================
# 五、设计更公平的投票系统
# ============================================================================
class FairVotingSystemDesigner:
    """设计更公平的投票系统"""
    
    def __init__(self, integrator: FanDataIntegrator):
        self.integrator = integrator
        self.results = {}
        
    def _get_judge_score_cols(self) -> List[str]:
        """获取所有评委分数列"""
        cols = []
        for week in range(1, 12):
            for judge in range(1, 5):
                cols.append(f'week{week}_judge{judge}_score')
        return cols
    
    def _calculate_avg_score(self, df_row: pd.Series) -> float:
        """计算选手的平均评委分数"""
        score_cols = self._get_judge_score_cols()
        scores = []
        for col in score_cols:
            if col in df_row.index:
                val = pd.to_numeric(df_row[col], errors='coerce')
                if pd.notna(val) and val > 0:
                    scores.append(val)
        return np.mean(scores) if scores else 0
    
    def simulate_weighted_system(self, season: int, judge_weight: float) -> Dict:
        """
        模拟不同权重的投票系统
        
        Args:
            season: 季节
            judge_weight: 评委权重 (0-1), 粉丝权重 = 1 - judge_weight
        """
        df_season = self.integrator.get_season_data(season)
        contestants = df_season.drop_duplicates(subset=['celebrity_name'])
        
        results = []
        
        for _, row in contestants.iterrows():
            name = row['celebrity_name']
            actual_placement = row['placement']
            
            # 评委分数
            avg_score = self._calculate_avg_score(row)
            
            # 粉丝数据
            fan_info = self.integrator.get_contestant_followers(name)
            total_followers = (fan_info.get('celebrity_total', 0) or 0) + (fan_info.get('partner_total', 0) or 0)
            
            results.append({
                'name': name,
                'avg_score': avg_score,
                'total_followers': total_followers,
                'actual_placement': actual_placement
            })
        
        df_results = pd.DataFrame(results)
        
        if len(df_results) == 0:
            return {
                'judge_weight': judge_weight,
                'fan_weight': 1 - judge_weight,
                'correlation_with_actual': np.nan,
                'contestants': []
            }
        
        # 标准化
        max_score = df_results['avg_score'].max() or 1
        max_followers = df_results['total_followers'].max() or 1
        
        df_results['score_normalized'] = df_results['avg_score'] / max_score
        df_results['followers_normalized'] = df_results['total_followers'] / max_followers
        
        # 综合分数
        fan_weight = 1 - judge_weight
        df_results['weighted_score'] = (
            judge_weight * df_results['score_normalized'] +
            fan_weight * df_results['followers_normalized']
        )
        
        # 预测名次
        df_results['predicted_placement'] = df_results['weighted_score'].rank(
            ascending=False, method='min', na_option='bottom'
        ).fillna(len(df_results)).astype(int)
        
        # 计算与实际名次的相关性
        valid_data = df_results.dropna(subset=['predicted_placement', 'actual_placement'])
        if len(valid_data) >= 3:
            corr, _ = spearmanr(valid_data['predicted_placement'], valid_data['actual_placement'])
        else:
            corr = np.nan
        
        return {
            'judge_weight': judge_weight,
            'fan_weight': fan_weight,
            'correlation_with_actual': corr,
            'contestants': df_results.to_dict('records')
        }
    
    def find_optimal_weights(self) -> Dict:
        """
        找到最优的权重分配
        
        目标: 最大化预测准确性 + 保证公平性
        """
        weight_options = np.arange(0.2, 0.85, 0.05)
        
        results_by_weight = {w: [] for w in weight_options}
        
        for season in range(1, 35):
            for weight in weight_options:
                result = self.simulate_weighted_system(season, weight)
                results_by_weight[weight].append(result['correlation_with_actual'])
        
        # 计算每个权重的平均相关性
        weight_performance = []
        for weight, correlations in results_by_weight.items():
            valid_corrs = [c for c in correlations if not np.isnan(c)]
            if valid_corrs:
                weight_performance.append({
                    'judge_weight': weight,
                    'fan_weight': 1 - weight,
                    'mean_correlation': np.mean(valid_corrs),
                    'std_correlation': np.std(valid_corrs),
                    'n_seasons': len(valid_corrs)
                })
        
        df_performance = pd.DataFrame(weight_performance)
        
        # 最优权重
        if len(df_performance) > 0:
            best_idx = df_performance['mean_correlation'].idxmax()
            best_weight = df_performance.loc[best_idx]
        else:
            best_weight = None
        
        self.results = {
            'weight_analysis': df_performance.to_dict('records'),
            'optimal_weights': {
                'judge_weight': best_weight['judge_weight'] if best_weight is not None else 0.5,
                'fan_weight': best_weight['fan_weight'] if best_weight is not None else 0.5,
                'correlation': best_weight['mean_correlation'] if best_weight is not None else np.nan
            },
            'recommendations': self._generate_recommendations(df_performance)
        }
        
        return self.results
    
    def _generate_recommendations(self, df_performance: pd.DataFrame) -> List[str]:
        """生成投票系统设计建议"""
        recommendations = []
        
        if len(df_performance) > 0:
            best = df_performance.loc[df_performance['mean_correlation'].idxmax()]
            
            recommendations.append(
                f"推荐评委权重: {best['judge_weight']:.0%}, 粉丝权重: {best['fan_weight']:.0%}"
            )
            recommendations.append(
                f"该权重配置下预测-实际名次相关性: {best['mean_correlation']:.3f}"
            )
            
            # 限制粉丝投票建议
            if best['fan_weight'] > 0.6:
                recommendations.append(
                    "建议: 当前粉丝权重较高，可考虑设置粉丝投票上限以防止极端情况"
                )
            
            # 动态权重建议
            recommendations.append(
                "建议: 考虑使用动态权重系统，初期评委权重高（保证技术基础），后期粉丝权重高（增加互动性）"
            )
        
        return recommendations
    
    def design_fair_system(self) -> Dict:
        """
        设计综合公平投票系统
        """
        optimal = self.find_optimal_weights()
        
        fair_system = {
            'name': '基于粉丝量的公平投票系统',
            'description': '结合社交媒体粉丝量作为粉丝投票代理变量的投票系统',
            'components': {
                'judge_score': {
                    'weight': optimal['optimal_weights']['judge_weight'],
                    'source': '评委评分（技术层面）',
                    'normalization': '归一化到0-1区间'
                },
                'fan_vote': {
                    'weight': optimal['optimal_weights']['fan_weight'],
                    'source': '社交媒体粉丝量作为代理变量',
                    'normalization': '归一化到0-1区间',
                    'cap': '建议设置上限（如最高30%投票占比）防止极端人气主导'
                }
            },
            'fairness_measures': [
                '使用粉丝量数据减少建模主观假设',
                '设置粉丝投票上限防止极端情况',
                '保留评委二选一机制作为安全阀',
                '透明化权重计算公式'
            ],
            'expected_benefits': [
                '更准确反映实际人气分布',
                '减少\"低分高名次\"争议案例',
                '提高预测与实际结果的一致性'
            ],
            'optimal_weights': optimal['optimal_weights'],
            'recommendations': optimal['recommendations']
        }
        
        return fair_system


# ============================================================================
# 主执行函数
# ============================================================================
def main():
    """主执行函数"""
    print("=" * 80)
    print("DWTS 粉丝数据分析模块 - MCM 2026 Problem C")
    print("=" * 80)
    
    # 创建输出目录
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # 1. 数据加载与整合
    print("\n[1/5] 数据加载与整合...")
    integrator = FanDataIntegrator()
    integrator.load_data()
    integrator.merge_data()
    
    # 2. 一、估算粉丝投票
    print("\n[2/5] 一、估算粉丝投票...")
    fan_estimator = FanVoteEstimator(integrator)
    fan_vote_results = fan_estimator.analyze_all_seasons()
    print(f"平均粉丝量-名次相关性: {fan_vote_results['mean_correlation']:.3f}")
    
    # 3. 二、对比两种投票方法
    print("\n[3/5] 二、对比两种投票组合方法...")
    method_comparator = VotingMethodComparator(integrator)
    method_results = method_comparator.analyze_method_bias()
    print(f"更偏向粉丝投票的方法: {method_results['analysis']['more_fan_biased']}")
    
    # 4. 三、分析争议选手案例
    print("\n[4/5] 三、分析争议选手案例...")
    controversy_analyzer = ControversyCaseAnalyzer(integrator)
    controversy_results = controversy_analyzer.analyze_all_controversies()
    print(f"分析了 {len(controversy_results['documented_cases'])} 个记录在案的争议选手")
    
    for case in controversy_results['documented_cases']:
        if 'error' not in case:
            print(f"  - {case['name']}: 粉丝{case['total_followers']:,}, 评分排名第{case['score_rank']}, 最终第{case['placement']}名")
    
    # 5. 四、评估选手/舞伴特征影响
    print("\n[4/5] 四、评估选手/舞伴特征影响...")
    partner_analyzer = PartnerInfluenceAnalyzer(integrator)
    partner_results = partner_analyzer.analyze_partner_effect()
    age_industry_results = partner_analyzer.analyze_age_industry_effect()
    print(f"舞伴平均贡献比例: {partner_results['mean_partner_contribution']:.1f}%")
    
    # 6. 五、设计更公平的投票系统
    print("\n[5/5] 五、设计更公平的投票系统...")
    system_designer = FairVotingSystemDesigner(integrator)
    fair_system = system_designer.design_fair_system()
    print(f"推荐权重: 评委{fair_system['optimal_weights']['judge_weight']:.0%}, "
          f"粉丝{fair_system['optimal_weights']['fan_weight']:.0%}")
    
    # 保存结果
    print("\n" + "=" * 60)
    print("保存结果...")
    
    all_results = {
        'fan_vote_estimation': fan_vote_results,
        'voting_method_comparison': method_results,
        'controversy_analysis': controversy_results,
        'partner_influence': partner_results,
        'age_industry_effect': age_industry_results,
        'fair_system_design': fair_system
    }
    
    # 保存为JSON（处理numpy类型）
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            if np.isnan(obj):
                return None
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj
    
    clean_results = convert_numpy(all_results)
    
    with open(os.path.join(Config.OUTPUT_DIR, 'fan_analysis_results.json'), 'w', encoding='utf-8') as f:
        json.dump(clean_results, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到: {Config.OUTPUT_DIR}/fan_analysis_results.json")
    
    print("\n" + "=" * 80)
    print("分析完成!")
    print("=" * 80)
    
    return all_results


if __name__ == "__main__":
    results = main()
