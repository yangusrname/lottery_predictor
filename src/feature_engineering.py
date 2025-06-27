# -*- coding: utf-8 -*-
"""
时序特征工程模块
专门为时序预测模型设计的特征提取
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import Counter
import logging
from typing import Dict, Any, Tuple, List
from datetime import datetime

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """时序特征工程类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化特征工程器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.scaler = StandardScaler()
        self.feature_names = []
        self.sequence_length = config['model'].get('sequence_length', 30)
        
    def extract_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        从数据中提取时序特征
        
        Args:
            data: 清洗后的数据框
            
        Returns:
            (features, labels) 特征和标签数组
        """
        logger.info("开始时序特征提取...")
        
        features_list = []
        
        for idx in range(len(data)):
            row_features = []
            
            # 1. 当前期号码特征
            if 'numbers' in data.columns:
                number_features = self._extract_number_features_single(data.iloc[idx])
                row_features.extend(number_features)
            
            # 2. 历史统计特征（基于滑动窗口）
            hist_features = self._extract_historical_features(data, idx)
            row_features.extend(hist_features)
            
            # 3. 时间特征
            if 'date' in data.columns:
                time_features = self._extract_time_features_single(data.iloc[idx])
                row_features.extend(time_features)
            
            # 4. 趋势特征
            trend_features = self._extract_trend_features(data, idx)
            row_features.extend(trend_features)
            
            # 5. 周期性特征
            cycle_features = self._extract_cycle_features(data, idx)
            row_features.extend(cycle_features)
            
            features_list.append(row_features)
        
        features = np.array(features_list)
        
        # 准备标签（预测下一期的号码）
        labels = self._prepare_labels(data)
        
        # 特征标准化
        features = self._normalize_features(features)
        
        # 确保特征和标签数量匹配
        min_len = min(len(features), len(labels))
        features = features[:min_len]
        labels = labels[:min_len]
        
        logger.info(f"时序特征提取完成: {features.shape[0]} 个样本, {features.shape[1]} 个特征")
        
        return features, labels
    
    def _extract_number_features_single(self, row) -> List[float]:
        """
        提取单行的号码特征
        
        Args:
            row: 数据行
            
        Returns:
            特征列表
        """
        features = []
        
        if pd.isna(row.get('numbers', '')):
            return [0] * 60  # 返回默认特征
        
        numbers_str = str(row['numbers'])
        
        try:
            if '+' in numbers_str:
                red_part, blue_part = numbers_str.split('+')
                red_balls = [int(x) for x in red_part.split(',')]
                blue_ball = int(blue_part)
                
                # 1. 号码分布特征（使用概率表示）
                red_dist = np.zeros(33)
                for ball in red_balls:
                    if 1 <= ball <= 33:
                        red_dist[ball-1] = 1.0 / 6  # 概率表示
                features.extend(red_dist)
                
                blue_dist = np.zeros(16)
                if 1 <= blue_ball <= 16:
                    blue_dist[blue_ball-1] = 1.0
                features.extend(blue_dist)
                
                # 2. 统计特征
                features.extend([
                    np.mean(red_balls) / 33,  # 归一化均值
                    np.std(red_balls) / 33,   # 归一化标准差
                    (max(red_balls) - min(red_balls)) / 33,  # 归一化跨度
                    sum(ball % 2 for ball in red_balls) / 6,  # 奇数比例
                    sum(ball <= 16 for ball in red_balls) / 6,  # 小号比例
                    sum(red_balls) / (33 * 6),  # 归一化和值
                    blue_ball / 16  # 归一化蓝球
                ])
                
                # 3. 区间分布特征
                zones = [0, 0, 0]  # 1-11, 12-22, 23-33
                for ball in red_balls:
                    if ball <= 11:
                        zones[0] += 1
                    elif ball <= 22:
                        zones[1] += 1
                    else:
                        zones[2] += 1
                features.extend([z / 6 for z in zones])
                
                # 4. 连号特征
                sorted_reds = sorted(red_balls)
                consecutive_count = 0
                for i in range(len(sorted_reds) - 1):
                    if sorted_reds[i+1] - sorted_reds[i] == 1:
                        consecutive_count += 1
                features.append(consecutive_count / 5)  # 最多5个连号
                
            else:
                features = [0] * 60
                
        except Exception as e:
            logger.warning(f"处理号码特征时出错: {str(e)}")
            features = [0] * 60
        
        return features
    
    def _extract_historical_features(self, data: pd.DataFrame, current_idx: int) -> List[float]:
        """
        提取历史统计特征
        
        Args:
            data: 数据框
            current_idx: 当前索引
            
        Returns:
            历史特征列表
        """
        features = []
        
        # 获取历史窗口
        window_sizes = [10, 20, 30, 50]
        
        for window_size in window_sizes:
            start_idx = max(0, current_idx - window_size)
            window_data = data.iloc[start_idx:current_idx]
            
            if len(window_data) > 0 and 'numbers' in data.columns:
                # 统计各号码出现频率
                red_counter = Counter()
                blue_counter = Counter()
                
                for _, row in window_data.iterrows():
                    if pd.isna(row.get('numbers', '')):
                        continue
                    
                    numbers_str = str(row['numbers'])
                    try:
                        if '+' in numbers_str:
                            red_part, blue_part = numbers_str.split('+')
                            red_balls = [int(x) for x in red_part.split(',')]
                            blue_ball = int(blue_part)
                            
                            red_counter.update(red_balls)
                            blue_counter[blue_ball] += 1
                    except:
                        continue
                
                # 红球频率特征
                red_freq = np.zeros(33)
                total_reds = sum(red_counter.values())
                if total_reds > 0:
                    for ball, count in red_counter.items():
                        if 1 <= ball <= 33:
                            red_freq[ball-1] = count / total_reds
                features.extend(red_freq)
                
                # 蓝球频率特征
                blue_freq = np.zeros(16)
                total_blues = sum(blue_counter.values())
                if total_blues > 0:
                    for ball, count in blue_counter.items():
                        if 1 <= ball <= 16:
                            blue_freq[ball-1] = count / total_blues
                features.extend(blue_freq)
                
                # 冷热号特征
                hot_threshold = 0.15  # 出现频率阈值
                hot_reds = sum(1 for f in red_freq if f > hot_threshold) / 33
                cold_reds = sum(1 for f in red_freq if f < 0.05) / 33
                features.extend([hot_reds, cold_reds])
                
            else:
                # 如果没有历史数据，使用默认值
                features.extend([0] * 51)
        
        return features
    
    def _extract_time_features_single(self, row) -> List[float]:
        """
        提取时间特征
        
        Args:
            row: 数据行
            
        Returns:
            时间特征列表
        """
        features = []
        
        if 'date' in row.index and not pd.isna(row['date']):
            try:
                date = pd.to_datetime(row['date'])
                
                # 1. 基础时间特征
                features.extend([
                    date.dayofweek / 6,  # 星期几 (0-6) -> (0-1)
                    date.day / 31,       # 日期 (1-31) -> (0-1)
                    date.month / 12,     # 月份 (1-12) -> (0-1)
                    date.quarter / 4,    # 季度 (1-4) -> (0-1)
                ])
                
                # 2. 周期性编码（使用正弦余弦编码）
                # 星期周期
                features.extend([
                    np.sin(2 * np.pi * date.dayofweek / 7),
                    np.cos(2 * np.pi * date.dayofweek / 7)
                ])
                
                # 月份周期
                features.extend([
                    np.sin(2 * np.pi * date.month / 12),
                    np.cos(2 * np.pi * date.month / 12)
                ])
                
                # 3. 特殊时期标记
                features.extend([
                    1.0 if date.dayofweek >= 5 else 0.0,  # 是否周末
                    1.0 if date.day <= 7 else 0.0,        # 是否月初
                    1.0 if date.day >= 24 else 0.0,       # 是否月末
                ])
                
            except:
                features = [0] * 11
        else:
            features = [0] * 11
        
        return features
    
    def _extract_trend_features(self, data: pd.DataFrame, current_idx: int) -> List[float]:
        """
        提取趋势特征
        
        Args:
            data: 数据框
            current_idx: 当前索引
            
        Returns:
            趋势特征列表
        """
        features = []
        
        # 使用不同的回看期计算趋势
        lookback_periods = [5, 10, 20]
        
        for period in lookback_periods:
            start_idx = max(0, current_idx - period)
            window_data = data.iloc[start_idx:current_idx]
            
            if len(window_data) >= 2 and 'numbers' in data.columns:
                # 计算号码和值的趋势
                sums = []
                for _, row in window_data.iterrows():
                    if pd.isna(row.get('numbers', '')):
                        continue
                    
                    numbers_str = str(row['numbers'])
                    try:
                        if '+' in numbers_str:
                            red_part, _ = numbers_str.split('+')
                            red_balls = [int(x) for x in red_part.split(',')]
                            sums.append(sum(red_balls))
                    except:
                        continue
                
                if len(sums) >= 2:
                    # 计算趋势（斜率）
                    x = np.arange(len(sums))
                    slope, _ = np.polyfit(x, sums, 1)
                    features.append(slope / 100)  # 归一化
                    
                    # 计算变化率
                    change_rate = (sums[-1] - sums[0]) / (sums[0] + 1e-6)
                    features.append(np.tanh(change_rate))  # 使用tanh限制范围
                else:
                    features.extend([0, 0])
            else:
                features.extend([0, 0])
        
        return features
    
    def _extract_cycle_features(self, data: pd.DataFrame, current_idx: int) -> List[float]:
        """
        提取周期性特征
        
        Args:
            data: 数据框
            current_idx: 当前索引
            
        Returns:
            周期性特征列表
        """
        features = []
        
        # 分析号码出现的周期性
        if current_idx >= 30 and 'numbers' in data.columns:
            # 检查每个号码的出现周期
            period_features = []
            
            for num in range(1, 34):  # 红球1-33
                appearances = []
                
                for i in range(max(0, current_idx - 100), current_idx):
                    row = data.iloc[i]
                    if pd.isna(row.get('numbers', '')):
                        continue
                    
                    numbers_str = str(row['numbers'])
                    try:
                        if '+' in numbers_str:
                            red_part, _ = numbers_str.split('+')
                            red_balls = [int(x) for x in red_part.split(',')]
                            if num in red_balls:
                                appearances.append(i)
                    except:
                        continue
                
                # 计算平均出现间隔
                if len(appearances) >= 2:
                    intervals = [appearances[i] - appearances[i-1] for i in range(1, len(appearances))]
                    avg_interval = np.mean(intervals)
                    # 计算距离上次出现的期数
                    last_appearance = current_idx - appearances[-1] if appearances else 100
                    # 周期性得分
                    cycle_score = np.exp(-abs(last_appearance - avg_interval) / (avg_interval + 1))
                    period_features.append(cycle_score)
                else:
                    period_features.append(0)
            
            # 取前10个最有周期性的号码特征
            top_cycle_scores = sorted(period_features, reverse=True)[:10]
            features.extend(top_cycle_scores)
            
            # 平均周期性得分
            features.append(np.mean(period_features))
        else:
            features = [0] * 11
        
        return features
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        标准化特征
        
        Args:
            features: 特征数组
            
        Returns:
            标准化后的特征
        """
        # 使用标准化，保留特征的相对关系
        return self.scaler.fit_transform(features)
    
    def _prepare_labels(self, data: pd.DataFrame) -> np.ndarray:
        """
        准备标签数据（下一期的号码）
        
        Args:
            data: 数据框
            
        Returns:
            标签数组
        """
        labels = []
        
        # 对于每个样本，其标签是下一期的号码
        for i in range(len(data) - 1):
            next_row = data.iloc[i + 1]
            
            if 'numbers' in data.columns and not pd.isna(next_row.get('numbers', '')):
                numbers_str = str(next_row['numbers'])
                
                try:
                    if '+' in numbers_str:
                        # 双色球格式
                        red_part, blue_part = numbers_str.split('+')
                        red_balls = [int(x) for x in red_part.split(',')]
                        blue_ball = int(blue_part)
                        
                        # 创建多标签向量（每个号码是否出现）
                        label = np.zeros(49)  # 33个红球 + 16个蓝球
                        
                        for ball in red_balls:
                            if 1 <= ball <= 33:
                                label[ball-1] = 1
                        
                        if 1 <= blue_ball <= 16:
                            label[33 + blue_ball - 1] = 1
                            
                        labels.append(label)
                    else:
                        # 其他格式，使用默认标签
                        labels.append(np.zeros(49))
                except:
                    labels.append(np.zeros(49))
            else:
                labels.append(np.zeros(49))
        
        # 最后一个样本没有下一期，使用零标签
        labels.append(np.zeros(49))
        
        return np.array(labels)
    
    def transform_for_prediction(self, data: pd.DataFrame, last_n: int = 30) -> np.ndarray:
        """
        为预测准备时序数据
        
        Args:
            data: 数据框
            last_n: 使用最后n条记录
            
        Returns:
            特征序列
        """
        # 使用最后n条记录
        recent_data = data.tail(last_n)
        
        # 提取特征
        features, _ = self.extract_features(recent_data)
        
        # 返回序列形式的特征
        if len(features) >= self.sequence_length:
            return features[-self.sequence_length:]
        else:
            # 如果数据不足，进行填充
            pad_length = self.sequence_length - len(features)
            padding = np.zeros((pad_length, features.shape[1]))
            return np.vstack([padding, features])
    
    def get_feature_names(self) -> List[str]:
        """
        获取特征名称列表
        
        Returns:
            特征名称列表
        """
        names = []
        
        # 号码分布特征
        for i in range(1, 34):
            names.append(f'red_ball_{i}_prob')
        for i in range(1, 17):
            names.append(f'blue_ball_{i}_prob')
        
        # 统计特征
        names.extend(['mean_norm', 'std_norm', 'range_norm', 'odd_ratio', 
                     'small_ratio', 'sum_norm', 'blue_norm'])
        
        # 区间分布
        names.extend(['zone_1_ratio', 'zone_2_ratio', 'zone_3_ratio'])
        
        # 连号特征
        names.append('consecutive_ratio')
        
        # 历史特征（为每个窗口大小）
        for window in [10, 20, 30, 50]:
            for i in range(1, 34):
                names.append(f'red_{i}_freq_w{window}')
            for i in range(1, 17):
                names.append(f'blue_{i}_freq_w{window}')
            names.extend([f'hot_ratio_w{window}', f'cold_ratio_w{window}'])
        
        # 时间特征
        names.extend(['dayofweek_norm', 'day_norm', 'month_norm', 'quarter_norm',
                     'week_sin', 'week_cos', 'month_sin', 'month_cos',
                     'is_weekend', 'is_month_start', 'is_month_end'])
        
        # 趋势特征
        for period in [5, 10, 20]:
            names.extend([f'trend_slope_p{period}', f'trend_change_p{period}'])
        
        # 周期性特征
        for i in range(10):
            names.append(f'top_cycle_score_{i+1}')
        names.append('avg_cycle_score')
        
        return names