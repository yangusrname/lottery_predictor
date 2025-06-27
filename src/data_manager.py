# -*- coding: utf-8 -*-
"""
数据管理模块
负责数据的加载、清洗、切分等操作
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)

class DataManager:
    """数据管理类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据管理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.raw_data = None
        self.cleaned_data = None
        
    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """
        从CSV文件加载数据
        
        Args:
            filepath: CSV文件路径
            
        Returns:
            加载的数据框
        """
        try:
            logger.info(f"正在加载数据: {filepath}")
            self.raw_data = pd.read_csv(filepath, encoding='utf-8')
            logger.info(f"成功加载 {len(self.raw_data)} 条记录")
            return self.raw_data
        except Exception as e:
            logger.error(f"加载数据失败: {str(e)}")
            raise
    
    def clean_data(self) -> pd.DataFrame:
        """
        清洗数据
        - 去重
        - 处理缺失值
        - 统一日期格式
        - 验证号码格式
        
        Returns:
            清洗后的数据框
        """
        if self.raw_data is None:
            raise ValueError("请先加载数据")
        
        logger.info("开始清洗数据...")
        df = self.raw_data.copy()
        
        # 记录原始数据量
        original_count = len(df)
        
        # 1. 去除完全重复的行
        df = df.drop_duplicates()
        logger.info(f"去重后剩余 {len(df)} 条记录 (删除 {original_count - len(df)} 条)")
        
        # 2. 处理缺失值
        # 假设数据包含期号、日期、中奖号码等列
        # 这里需要根据实际数据格式调整列名
        if 'date' in df.columns:
            # 删除日期缺失的记录
            df = df.dropna(subset=['date'])
        
        # 3. 统一日期格式
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'])
            except:
                logger.warning("日期格式转换失败，保持原格式")
        
        # 4. 验证号码格式（假设福利彩票是双色球，红球6个1-33，蓝球1个1-16）
        # 这里需要根据实际彩票类型调整验证逻辑
        if 'numbers' in df.columns:
            # 假设号码以字符串形式存储，如 "01,02,03,04,05,06+07"
            valid_rows = []
            for idx, row in df.iterrows():
                if self._validate_lottery_numbers(row.get('numbers', '')):
                    valid_rows.append(idx)
            
            df = df.loc[valid_rows]
            logger.info(f"号码格式验证后剩余 {len(df)} 条记录")
        
        # 5. 按日期排序
        if 'date' in df.columns:
            df = df.sort_values('date')
        
        self.cleaned_data = df
        logger.info(f"数据清洗完成，最终保留 {len(df)} 条记录")
        
        return self.cleaned_data
    
    def _validate_lottery_numbers(self, numbers_str: str) -> bool:
        """
        验证彩票号码格式是否正确
        
        Args:
            numbers_str: 号码字符串
            
        Returns:
            是否有效
        """
        if not numbers_str or pd.isna(numbers_str):
            return False
        
        try:
            # 这里以双色球为例，实际需要根据具体彩票类型调整
            # 格式: "01,02,03,04,05,06+07"
            if '+' in str(numbers_str):
                red_part, blue_part = str(numbers_str).split('+')
                red_balls = [int(x) for x in red_part.split(',')]
                blue_ball = int(blue_part)
                
                # 验证红球数量和范围
                if len(red_balls) != 6:
                    return False
                if not all(1 <= ball <= 33 for ball in red_balls):
                    return False
                if len(set(red_balls)) != 6:  # 检查是否有重复
                    return False
                    
                # 验证蓝球范围
                if not (1 <= blue_ball <= 16):
                    return False
                    
                return True
            else:
                # 其他格式的彩票验证逻辑
                return True
                
        except:
            return False
    
    def split_data(self, features: np.ndarray, labels: np.ndarray) -> Tuple[Tuple[np.ndarray, np.ndarray], 
                                                                            Tuple[np.ndarray, np.ndarray], 
                                                                            Tuple[np.ndarray, np.ndarray]]:
        """
        切分数据集为训练集、验证集和测试集
        
        Args:
            features: 特征数组
            labels: 标签数组
            
        Returns:
            (train_data, val_data, test_data) 每个都是 (features, labels) 元组
        """
        # 获取切分比例
        train_ratio = self.config['data'].get('train_ratio', 0.99)
        val_ratio = self.config['data'].get('val_ratio', 0.005)
        test_ratio = self.config['data'].get('test_ratio', 0.005)
        
        # 确保比例和为1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "数据集切分比例之和必须为1"
        
        # 首先切分出训练集和临时集
        X_train, X_temp, y_train, y_temp = train_test_split(
            features, labels, 
            test_size=(val_ratio + test_ratio),
            random_state=42
        )
        
        # 再从临时集中切分验证集和测试集
        val_size = val_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_size),
            random_state=42
        )
        
        logger.info(f"数据集切分完成:")
        logger.info(f"  训练集: {len(X_train)} 样本")
        logger.info(f"  验证集: {len(X_val)} 样本")
        logger.info(f"  测试集: {len(X_test)} 样本")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def get_latest_records(self, n: int = 10) -> pd.DataFrame:
        """
        获取最新的n条记录
        
        Args:
            n: 记录数量
            
        Returns:
            最新的记录
        """
        if self.cleaned_data is None:
            raise ValueError("请先加载并清洗数据")
        
        return self.cleaned_data.tail(n)
    
    def save_processed_data(self, filepath: str):
        """
        保存处理后的数据
        
        Args:
            filepath: 保存路径
        """
        if self.cleaned_data is None:
            raise ValueError("没有可保存的数据")
        
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.cleaned_data.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"数据已保存至: {output_path}")