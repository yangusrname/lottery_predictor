# -*- coding: utf-8 -*-
"""
工具函数模块
提供配置加载、日志设置等通用功能
"""

import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any
import sys

def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        # 如果配置文件不存在，创建默认配置
        default_config = create_default_config()
        save_config(default_config, config_path)
        logging.info(f"创建默认配置文件: {config_path}")
        return default_config
    
    # 根据文件扩展名选择加载方式
    if config_file.suffix in ['.yml', '.yaml']:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    elif config_file.suffix == '.json':
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        raise ValueError(f"不支持的配置文件格式: {config_file.suffix}")
    
    return config

def save_config(config: Dict[str, Any], config_path: str):
    """
    保存配置文件
    
    Args:
        config: 配置字典
        config_path: 保存路径
    """
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    if config_file.suffix in ['.yml', '.yaml']:
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    elif config_file.suffix == '.json':
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

def create_default_config() -> Dict[str, Any]:
    """
    创建默认配置
    
    Returns:
        默认配置字典
    """
    return {
        'model': {
            'type': 'randomforest',
            'params': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            }
        },
        'data': {
            'source': 'data/raw/lottery.csv',
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15
        },
        'training': {
            'batch_size': 32,
            'epochs': 100,
            'early_stopping': True,
            'patience': 10
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    }

def setup_logging(log_level: str = 'INFO'):
    """
    设置日志配置
    
    Args:
        log_level: 日志级别
    """
    # 创建logs目录
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # 日志格式
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 配置日志
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                log_dir / f'lottery_predictor_{Path.cwd().name}.log',
                encoding='utf-8'
            )
        ]
    )
    
    # 设置第三方库的日志级别
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('sklearn').setLevel(logging.WARNING)

def create_project_structure():
    """
    创建项目目录结构
    """
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'configs',
        'reports',
        'logs',
        'src'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    logging.info("项目目录结构创建完成")

def generate_sample_data(output_path: str = 'data/raw/lottery.csv', n_samples: int = 1000):
    """
    生成示例彩票数据（用于测试）
    
    Args:
        output_path: 输出路径
        n_samples: 样本数量
    """
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    logging.info(f"生成 {n_samples} 条示例数据...")
    
    data = []
    start_date = datetime.now() - timedelta(days=n_samples*3)
    
    for i in range(n_samples):
        # 生成日期
        date = start_date + timedelta(days=i*3)
        
        # 生成红球（6个，1-33，不重复）
        red_balls = sorted(np.random.choice(range(1, 34), 6, replace=False))
        red_str = ','.join([f"{ball:02d}" for ball in red_balls])
        
        # 生成蓝球（1个，1-16）
        blue_ball = np.random.randint(1, 17)
        
        # 组合号码
        numbers = f"{red_str}+{blue_ball:02d}"
        
        data.append({
            'issue': f"2024{i+1:03d}",
            'date': date.strftime('%Y-%m-%d'),
            'numbers': numbers,
            'sales': np.random.randint(100000000, 500000000),  # 销售额
            'pool': np.random.randint(50000000, 200000000)     # 奖池
        })
    
    # 创建DataFrame并保存
    df = pd.DataFrame(data)
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_file, index=False, encoding='utf-8')
    logging.info(f"示例数据已保存至: {output_file}")

def validate_lottery_format(numbers_str: str) -> bool:
    """
    验证彩票号码格式
    
    Args:
        numbers_str: 号码字符串
        
    Returns:
        是否有效
    """
    try:
        # 双色球格式: "01,02,03,04,05,06+07"
        if '+' not in numbers_str:
            return False
        
        red_part, blue_part = numbers_str.split('+')
        red_balls = [int(x) for x in red_part.split(',')]
        blue_ball = int(blue_part)
        
        # 验证红球
        if len(red_balls) != 6:
            return False
        if not all(1 <= ball <= 33 for ball in red_balls):
            return False
        if len(set(red_balls)) != 6:  # 有重复
            return False
        
        # 验证蓝球
        if not (1 <= blue_ball <= 16):
            return False
        
        return True
    except:
        return False

def calculate_prize(predicted: str, actual: str) -> Dict[str, Any]:
    """
    计算中奖情况
    
    Args:
        predicted: 预测号码
        actual: 实际开奖号码
        
    Returns:
        中奖信息
    """
    if not validate_lottery_format(predicted) or not validate_lottery_format(actual):
        return {'error': '号码格式错误'}
    
    # 解析号码
    pred_red, pred_blue = predicted.split('+')
    actual_red, actual_blue = actual.split('+')
    
    pred_red_balls = set(int(x) for x in pred_red.split(','))
    actual_red_balls = set(int(x) for x in actual_red.split(','))
    
    pred_blue_ball = int(pred_blue)
    actual_blue_ball = int(actual_blue)
    
    # 计算匹配数
    red_matches = len(pred_red_balls & actual_red_balls)
    blue_match = pred_blue_ball == actual_blue_ball
    
    # 判断奖级（双色球规则）
    prize_level = None
    prize_name = None
    
    if red_matches == 6 and blue_match:
        prize_level = 1
        prize_name = "一等奖"
    elif red_matches == 6:
        prize_level = 2
        prize_name = "二等奖"
    elif red_matches == 5 and blue_match:
        prize_level = 3
        prize_name = "三等奖"
    elif red_matches == 5 or (red_matches == 4 and blue_match):
        prize_level = 4
        prize_name = "四等奖"
    elif red_matches == 4 or (red_matches == 3 and blue_match):
        prize_level = 5
        prize_name = "五等奖"
    elif blue_match:
        prize_level = 6
        prize_name = "六等奖"
    
    return {
        'red_matches': red_matches,
        'blue_match': blue_match,
        'prize_level': prize_level,
        'prize_name': prize_name,
        'is_winner': prize_level is not None
    }

def format_currency(amount: float) -> str:
    """
    格式化货币显示
    
    Args:
        amount: 金额
        
    Returns:
        格式化后的字符串
    """
    if amount >= 100000000:  # 亿
        return f"{amount/100000000:.2f}亿"
    elif amount >= 10000:  # 万
        return f"{amount/10000:.2f}万"
    else:
        return f"{amount:.2f}"

if __name__ == '__main__':
    # 测试功能
    create_project_structure()
    generate_sample_data()