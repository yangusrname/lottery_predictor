#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
中国福利彩票预测工具 - 命令行接口
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_manager import DataManager
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator
from src.predictor import Predictor
from src.utils import load_config, setup_logging

def data_fetch(args):
    """获取并更新数据"""
    config = load_config(args.config)
    data_manager = DataManager(config)
    
    if args.source == 'csv':
        data_manager.load_from_csv(args.file)
    else:
        print("暂时仅支持CSV数据源")
        return
    
    print("数据获取完成！")

def data_clean(args):
    """执行数据清洗"""
    config = load_config(args.config)
    data_manager = DataManager(config)
    
    # 加载原始数据
    raw_data_path = Path(config['data']['source'])
    if raw_data_path.exists():
        data_manager.load_from_csv(str(raw_data_path))
        cleaned_data = data_manager.clean_data()
        
        # 保存清洗后的数据
        output_path = Path('data/processed/cleaned_lottery.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_data.to_csv(output_path, index=False)
        
        print(f"数据清洗完成！已保存至: {output_path}")
    else:
        print(f"错误：找不到数据文件 {raw_data_path}")

def train(args):
    """开始模型训练"""
    config = load_config(args.config)
    
    # 如果通过命令行指定了模型类型，覆盖配置文件中的设置
    if args.model:
        config['model']['type'] = args.model
    
    # 初始化各个模块
    data_manager = DataManager(config)
    feature_engineer = FeatureEngineer(config)
    model_trainer = ModelTrainer(config)
    
    # 加载和准备数据
    print("加载数据...")
    data_path = Path('data/processed/cleaned_lottery.csv')
    if not data_path.exists():
        data_path = Path(config['data']['source'])
    
    data_manager.load_from_csv(str(data_path))
    cleaned_data = data_manager.clean_data()
    
    # 特征工程
    print("进行特征工程...")
    features, labels = feature_engineer.extract_features(cleaned_data)
    
    # 数据切分
    print("切分数据集...")
    train_data, val_data, test_data = data_manager.split_data(features, labels)
    
    # 训练模型
    print(f"开始训练 {config['model']['type']} 模型...")
    model_trainer.train(train_data, val_data)
    
    # 保存模型
    model_extension = '.pth' if config['model']['type'] in ['lstm', 'gru', 'transformer', 'tcn'] else '.pkl'
    model_path = Path(f"models/{config['model']['type']}_model{model_extension}")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_trainer.save_model(str(model_path))
    
    print(f"训练完成！模型已保存至: {model_path}")

def evaluate(args):
    """运行模型评估"""
    config = load_config(args.config)
    
    # 加载模型
    model_path = Path(f"models/{args.model}_model.pth")
    if not model_path.exists():
        # 尝试旧版本的模型文件
        model_path = Path(f"models/{args.model}_model.pkl")
    
    if not model_path.exists():
        print("错误：找不到训练好的模型文件")
        return
    
    # 初始化评估器
    evaluator = ModelEvaluator(config)
    evaluator.load_model(str(model_path))
    
    # 加载测试数据
    data_manager = DataManager(config)
    feature_engineer = FeatureEngineer(config)
    
    data_path = Path('data/processed/cleaned_lottery.csv')
    if not data_path.exists():
        data_path = Path(config['data']['source'])
    
    data_manager.load_from_csv(str(data_path))
    cleaned_data = data_manager.clean_data()
    features, labels = feature_engineer.extract_features(cleaned_data)
    _, _, test_data = data_manager.split_data(features, labels)
    
    # 评估模型
    print("评估模型性能...")
    metrics = evaluator.evaluate(test_data)
    
    # 生成报告
    report_path = Path(f"reports/evaluation_{args.model}.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    evaluator.generate_report(metrics, str(report_path))
    
    print(f"评估完成！报告已保存至: {report_path}")

def predict(args):
    """执行预测并输出结果"""
    config = load_config(args.config)
    
    # 初始化预测器
    predictor = Predictor(config)
    
    # 加载历史数据用于特征提取
    data_manager = DataManager(config)
    feature_engineer = FeatureEngineer(config)
    
    data_path = Path('data/processed/cleaned_lottery.csv')
    if not data_path.exists():
        data_path = Path(config['data']['source'])
    
    data_manager.load_from_csv(str(data_path))
    cleaned_data = data_manager.clean_data()
    
    # 进行预测
    print("正在预测下一期号码...")
    
    if args.ensemble:
        # 集成预测
        model_types = ['lstm', 'gru', 'transformer']
        model_paths = []
        
        for model_type in model_types:
            model_path = Path(f"models/{model_type}_model.pth")
            if model_path.exists():
                model_paths.append(str(model_path))
        
        if len(model_paths) < 2:
            print("错误：集成预测需要至少2个已训练的模型")
            return
        
        predictions = predictor.predict_ensemble(cleaned_data, feature_engineer, model_paths)
    else:
        try:
            # 单模型预测
            model_path = Path(f"models/{args.model}_model.pth")
            if not model_path.exists():
                # 尝试旧版本的模型文件
                model_path = Path(f"models/{args.model}_model.pkl")
            
            if not model_path.exists():
                print("错误：找不到训练好的模型文件")
                return
            
            predictor.load_model(str(model_path))
            predictions = predictor.predict_next(cleaned_data, feature_engineer)
        except Exception as e:
            import traceback
            traceback.print_exc()
    
    # 输出结果
    print("\n预测结果:")
    print("-" * 50)
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictor.save_predictions(predictions, str(output_path))
        print(f"预测结果已保存至: {output_path}")
    else:
        predictor.display_predictions(predictions)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='中国福利彩票预测工具',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # 全局参数
    parser.add_argument('--config', default='configs/config.yml', 
                       help='配置文件路径 (默认: configs/config.yml)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别 (默认: INFO)')
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # data fetch 命令
    parser_fetch = subparsers.add_parser('data-fetch', help='获取并更新数据')
    parser_fetch.add_argument('--source', choices=['csv', 'web'], default='csv',
                            help='数据源类型 (默认: csv)')
    parser_fetch.add_argument('--file', help='CSV文件路径')
    parser_fetch.set_defaults(func=data_fetch)
    
    # data clean 命令
    parser_clean = subparsers.add_parser('data-clean', help='执行数据清洗')
    parser_clean.set_defaults(func=data_clean)
    
    # train 命令
    parser_train = subparsers.add_parser('train', help='开始模型训练')
    parser_train.add_argument('--model', 
                            choices=['lstm', 'gru', 'transformer', 'tcn'],
                            help='模型类型（默认使用配置文件中的设置）')
    parser_train.set_defaults(func=train)
    
    # evaluate 命令
    parser_eval = subparsers.add_parser('evaluate', help='运行模型评估')
    parser_eval.add_argument('--model', required=True,
                           choices=['lstm', 'gru', 'transformer', 'tcn'],
                           help='要评估的模型类型')
    parser_eval.set_defaults(func=evaluate)
    
    # predict 命令
    parser_predict = subparsers.add_parser('predict', help='执行预测并输出结果')
    parser_predict.add_argument('--model', required=True,
                              choices=['lstm', 'gru', 'transformer', 'tcn'],
                              help='用于预测的模型类型')
    parser_predict.add_argument('--output', help='输出文件路径 (可选)')
    parser_predict.add_argument('--ensemble', action='store_true',
                              help='使用集成预测（需要多个已训练的模型）')
    parser_predict.set_defaults(func=predict)
    
    # 解析参数
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level)
    
    # 执行命令
    if hasattr(args, 'func'):
        try:
            args.func(args)
        except Exception as e:
            logging.error(f"执行失败: {str(e)}")
            sys.exit(1)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()