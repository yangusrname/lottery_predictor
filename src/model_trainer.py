# -*- coding: utf-8 -*-
"""
模型训练模块
支持多种机器学习和深度学习模型
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
import joblib
import json
from pathlib import Path
import logging
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelTrainer:
    """模型训练器类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化模型训练器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.model_type = config['model']['type']
        self.model_params = config['model'].get('params', {})
        self.model = None
        self.training_history = []
        
    def _create_model(self):
        """
        根据配置创建模型
        
        Returns:
            模型实例
        """
        logger.info(f"创建 {self.model_type} 模型...")
        
        if self.model_type == 'randomforest':
            base_model = RandomForestClassifier(
                n_estimators=self.model_params.get('n_estimators', 100),
                max_depth=self.model_params.get('max_depth', 10),
                min_samples_split=self.model_params.get('min_samples_split', 2),
                min_samples_leaf=self.model_params.get('min_samples_leaf', 1),
                random_state=42,
                n_jobs=-1
            )
            # 使用多输出分类器包装
            return MultiOutputClassifier(base_model)
            
        elif self.model_type == 'xgboost':
            base_model = xgb.XGBClassifier(
                n_estimators=self.model_params.get('n_estimators', 100),
                max_depth=self.model_params.get('max_depth', 6),
                learning_rate=self.model_params.get('learning_rate', 0.1),
                subsample=self.model_params.get('subsample', 0.8),
                colsample_bytree=self.model_params.get('colsample_bytree', 0.8),
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            return MultiOutputClassifier(base_model)
            
        elif self.model_type == 'lightgbm':
            base_model = lgb.LGBMClassifier(
                n_estimators=self.model_params.get('n_estimators', 100),
                max_depth=self.model_params.get('max_depth', -1),
                learning_rate=self.model_params.get('learning_rate', 0.1),
                num_leaves=self.model_params.get('num_leaves', 31),
                subsample=self.model_params.get('subsample', 0.8),
                colsample_bytree=self.model_params.get('colsample_bytree', 0.8),
                random_state=42,
                verbosity=-1
            )
            return MultiOutputClassifier(base_model)
            
        elif self.model_type == 'mlp':
            # 多层感知机
            hidden_layers = self.model_params.get('hidden_layers', (100, 50))
            return MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                activation=self.model_params.get('activation', 'relu'),
                solver=self.model_params.get('solver', 'adam'),
                learning_rate_init=self.model_params.get('learning_rate', 0.001),
                max_iter=self.model_params.get('max_iter', 500),
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
            
        elif self.model_type == 'lstm':
            # LSTM需要使用深度学习框架，这里简化为使用MLP
            logger.warning("LSTM模型需要TensorFlow/PyTorch，暂时使用MLP替代")
            return MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                max_iter=1000,
                random_state=42,
                early_stopping=True
            )
            
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def train(self, train_data: Tuple[np.ndarray, np.ndarray], 
              val_data: Tuple[np.ndarray, np.ndarray] = None):
        """
        训练模型
        
        Args:
            train_data: (X_train, y_train) 训练数据
            val_data: (X_val, y_val) 验证数据（可选）
        """
        X_train, y_train = train_data
        
        # 创建模型
        self.model = self._create_model()
        
        logger.info(f"开始训练，训练样本数: {len(X_train)}")
        
        # 训练模型
        if self.model_type in ['xgboost', 'lightgbm'] and val_data is not None:
            # XGBoost和LightGBM支持验证集
            X_val, y_val = val_data
            
            # 由于使用了MultiOutputClassifier，需要特殊处理
            self.model.fit(X_train, y_train)
            
            # 记录训练历史
            train_score = self.model.score(X_train, y_train)
            val_score = self.model.score(X_val, y_val) if val_data else None
            
            self.training_history.append({
                'train_score': train_score,
                'val_score': val_score
            })
            
            logger.info(f"训练完成! 训练集得分: {train_score:.4f}")
            if val_score:
                logger.info(f"验证集得分: {val_score:.4f}")
                
        else:
            # 其他模型直接训练
            self.model.fit(X_train, y_train)
            
            # 计算训练得分
            train_score = self.model.score(X_train, y_train)
            self.training_history.append({'train_score': train_score})
            
            logger.info(f"训练完成! 训练集得分: {train_score:.4f}")
            
            if val_data is not None:
                X_val, y_val = val_data
                val_score = self.model.score(X_val, y_val)
                logger.info(f"验证集得分: {val_score:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        使用模型进行预测
        
        Args:
            X: 特征数据
            
        Returns:
            预测结果
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
            
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率
        
        Args:
            X: 特征数据
            
        Returns:
            预测概率
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        if hasattr(self.model, 'predict_proba'):
            # 获取每个输出的概率
            probas = []
            
            if hasattr(self.model, 'estimators_'):
                # MultiOutputClassifier
                for estimator in self.model.estimators_:
                    proba = estimator.predict_proba(X)
                    # 只取正类的概率
                    if proba.shape[1] == 2:
                        probas.append(proba[:, 1])
                    else:
                        probas.append(proba[:, 0])
                        
                return np.column_stack(probas)
            else:
                # 单输出模型
                return self.model.predict_proba(X)
        else:
            # 如果模型不支持概率预测，返回二值预测
            return self.model.predict(X)
    
    def save_model(self, filepath: str):
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        if self.model is None:
            raise ValueError("没有可保存的模型")
        
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        joblib.dump(self.model, output_path)
        
        # 保存训练历史和配置
        meta_path = output_path.with_suffix('.meta.json')
        meta_data = {
            'model_type': self.model_type,
            'model_params': self.model_params,
            'training_history': self.training_history
        }
        
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"模型已保存至: {output_path}")
        logger.info(f"元数据已保存至: {meta_path}")
    
    def load_model(self, filepath: str):
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
        """
        model_path = Path(filepath)
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        # 加载模型
        self.model = joblib.load(model_path)
        
        # 尝试加载元数据
        meta_path = model_path.with_suffix('.meta.json')
        if meta_path.exists():
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
                self.model_type = meta_data.get('model_type', 'unknown')
                self.model_params = meta_data.get('model_params', {})
                self.training_history = meta_data.get('training_history', [])
        
        logger.info(f"模型加载成功: {filepath}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        获取特征重要性（仅适用于树模型）
        
        Returns:
            特征重要性数据框
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        if hasattr(self.model, 'estimators_'):
            # MultiOutputClassifier
            if hasattr(self.model.estimators_[0], 'feature_importances_'):
                # 计算所有估计器的平均特征重要性
                importances = np.mean([est.feature_importances_ 
                                     for est in self.model.estimators_], axis=0)
                
                # 创建特征名称
                feature_names = [f'feature_{i}' for i in range(len(importances))]
                
                # 创建数据框
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                })
                
                # 按重要性排序
                importance_df = importance_df.sort_values('importance', ascending=False)
                
                return importance_df
        
        logger.warning(f"{self.model_type} 模型不支持特征重要性分析")
        return pd.DataFrame()