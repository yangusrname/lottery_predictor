# -*- coding: utf-8 -*-
"""
中国福利彩票预测工具 - 核心模块包

该包包含以下核心模块：
- data_manager: 数据管理和处理
- feature_engineering: 特征工程
- model_trainer: 模型训练
- model_evaluator: 模型评估
- predictor: 预测功能
- utils: 工具函数
"""

__version__ = '1.0.0'
__author__ = 'Lottery Predictor Team'

# 导入主要类和函数，方便外部调用
from .data_manager import DataManager
from .feature_engineering import FeatureEngineer
from .model_trainer import ModelTrainer
from .model_evaluator import ModelEvaluator
from .predictor import Predictor
from .utils import (
    load_config,
    save_config,
    setup_logging,
    create_project_structure,
    generate_sample_data,
    validate_lottery_format,
    calculate_prize
)

__all__ = [
    'DataManager',
    'FeatureEngineer', 
    'ModelTrainer',
    'ModelEvaluator',
    'Predictor',
    'load_config',
    'save_config',
    'setup_logging',
    'create_project_structure',
    'generate_sample_data',
    'validate_lottery_format',
    'calculate_prize'
]

# 模块信息
MODULE_INFO = {
    'data_manager': '数据管理模块 - 负责数据的加载、清洗和切分',
    'feature_engineering': '特征工程模块 - 提取号码、时间和统计特征',
    'model_trainer': '模型训练模块 - 支持多种机器学习算法',
    'model_evaluator': '模型评估模块 - 评估模型性能并生成报告',
    'predictor': '预测模块 - 使用训练好的模型进行预测',
    'utils': '工具函数模块 - 提供配置加载、日志等通用功能'
}

# 支持的模型类型
SUPPORTED_MODELS = [
    'randomforest',
    'xgboost', 
    'lightgbm',
    'mlp',
    'lstm'
]

# 默认配置
DEFAULT_CONFIG = {
    'model': {
        'type': 'randomforest',
        'params': {
            'n_estimators': 100,
            'max_depth': 10
        }
    },
    'data': {
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15
    }
}