# -*- coding: utf-8 -*-
"""
特征工程模块
负责从原始数据中提取有用的特征
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from collections import Counter
import logging
from typing import Dict, Any, Tuple, List

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """特征工程类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化特征工程器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_names = []
        
    def extract_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        从数据中提取特征
        
        Args:
            data: 清洗后的数据框
            
        Returns:
            (features, labels) 特征和标签数组
        """
        logger.info("开始特征提取...")
        
        features_list = []
        
        # 1. 号码相关特征
        if 'numbers' in data.columns:
            number_features = self._extract_number_features(data)
            features_list.append(number_features)
        
        # 2. 时间相关特征
        if 'date' in data.columns:
            time_features = self._extract_time_features(data)
            features_list.append(time_features)
        
        # 3. 统计特征
        stat_features = self._extract_statistical_features(data)
        features_list.append(stat_features)
        
        # 合并所有特征
        features = np.hstack(features_list)
        
        # 准备标签（预测下一期的号码）
        labels = self._prepare_labels(data)
        
        # 确保特征和标签数量匹配
        min_len = min(len(features), len(labels))
        features = features[:min_len]
        labels = labels[:min_len]
        
        logger.info(f"特征提取完成: {features.shape[0]} 个样本, {features.shape[1]} 个特征")
        
        return features, labels
    
    def _extract_number_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        提取号码相关特征
        
        Args:
            data: 数据框
            
        Returns:
            号码特征数组
        """
        features = []
        
        for idx, row in data.iterrows():
            if pd.isna(row.get('numbers', '')):
                # 如果号码缺失，使用零向量
                features.append(np.zeros(50))  # 预设特征维度
                continue
                
            numbers_str = str(row['numbers'])
            row_features = []
            
            try:
                # 解析号码（以双色球为例）
                if '+' in numbers_str:
                    red_part, blue_part = numbers_str.split('+')
                    red_balls = [int(x) for x in red_part.split(',')]
                    blue_ball = int(blue_part)
                    
                    # 1. 红球的独热编码（1-33）
                    red_onehot = np.zeros(33)
                    for ball in red_balls:
                        if 1 <= ball <= 33:
                            red_onehot[ball-1] = 1
                    row_features.extend(red_onehot)
                    
                    # 2. 蓝球的独热编码（1-16）
                    blue_onehot = np.zeros(16)
                    if 1 <= blue_ball <= 16:
                        blue_onehot[blue_ball-1] = 1
                    row_features.extend(blue_onehot)
                    
                    # 3. 号码统计特征
                    row_features.append(np.mean(red_balls))  # 红球平均值
                    row_features.append(np.std(red_balls))   # 红球标准差
                    row_features.append(max(red_balls) - min(red_balls))  # 红球跨度
                    row_features.append(sum(ball % 2 for ball in red_balls))  # 奇数个数
                    row_features.append(sum(ball <= 16 for ball in red_balls))  # 小号个数
                    
                    # 4. 和值特征
                    row_features.append(sum(red_balls))  # 红球和值
                    row_features.append(sum(red_balls) + blue_ball)  # 总和值
                    
                else:
                    # 其他类型彩票的处理
                    row_features = np.zeros(56)  # 使用默认特征维度
                    
            except Exception as e:
                logger.warning(f"处理号码 {numbers_str} 时出错: {str(e)}")
                row_features = np.zeros(56)
            
            features.append(row_features)
        
        return np.array(features)
    
    def _extract_time_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        提取时间相关特征
        
        Args:
            data: 数据框
            
        Returns:
            时间特征数组
        """
        features = []
        
        for idx, row in data.iterrows():
            row_features = []
            
            if 'date' in data.columns and not pd.isna(row['date']):
                try:
                    date = pd.to_datetime(row['date'])
                    
                    # 1. 星期几（0-6）
                    row_features.append(date.dayofweek)
                    
                    # 2. 月份（1-12）
                    row_features.append(date.month)
                    
                    # 3. 季度（1-4）
                    row_features.append(date.quarter)
                    
                    # 4. 是否是月初/月末
                    row_features.append(1 if date.day <= 7 else 0)
                    row_features.append(1 if date.day >= 24 else 0)
                    
                    # 5. 是否是周末
                    row_features.append(1 if date.dayofweek >= 5 else 0)
                    
                except:
                    row_features = [0] * 6
            else:
                row_features = [0] * 6
            
            features.append(row_features)
        
        return np.array(features)
    
    def _extract_statistical_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        提取统计特征（冷热号等）
        
        Args:
            data: 数据框
            
        Returns:
            统计特征数组
        """
        features = []
        window_size = 30  # 统计窗口大小
        
        for i in range(len(data)):
            row_features = []
            
            # 获取历史窗口数据
            start_idx = max(0, i - window_size)
            window_data = data.iloc[start_idx:i+1]
            
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
                
                # 1. 最热的红球号码（出现次数最多的5个）
                hot_reds = [num for num, _ in red_counter.most_common(5)]
                hot_reds.extend([0] * (5 - len(hot_reds)))  # 补齐到5个
                row_features.extend(hot_reds)
                
                # 2. 最冷的红球号码（最近未出现的5个）
                all_reds = set(range(1, 34))
                appeared_reds = set(red_counter.keys())
                cold_reds = list(all_reds - appeared_reds)[:5]
                cold_reds.extend([0] * (5 - len(cold_reds)))
                row_features.extend(cold_reds)
                
                # 3. 连续未出现期数统计（前10个号码）
                for num in range(1, 11):
                    last_appear = -1
                    for j in range(len(window_data)-1, -1, -1):
                        row_data = window_data.iloc[j]
                        if pd.isna(row_data.get('numbers', '')):
                            continue
                        numbers_str = str(row_data['numbers'])
                        try:
                            if '+' in numbers_str:
                                red_part, _ = numbers_str.split('+')
                                red_balls = [int(x) for x in red_part.split(',')]
                                if num in red_balls:
                                    last_appear = j
                                    break
                        except:
                            continue
                    
                    missing_periods = len(window_data) - 1 - last_appear if last_appear >= 0 else len(window_data)
                    row_features.append(missing_periods)
                
            else:
                # 如果没有足够的历史数据，使用零特征
                row_features = [0] * 20
            
            features.append(row_features)
        
        return np.array(features)
    
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
        为预测转换数据（只提取特征，不需要标签）
        
        Args:
            data: 数据框
            last_n: 使用最后n条记录
            
        Returns:
            特征数组
        """
        # 使用最后n条记录
        recent_data = data.tail(last_n)
        
        # 提取特征但不需要标签
        features, _ = self.extract_features(recent_data)
        
        # 返回最后一条记录的特征
        return features[-1:] if len(features) > 0 else np.zeros((1, 82))