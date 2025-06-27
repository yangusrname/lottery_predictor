# -*- coding: utf-8 -*-
"""
时序预测模块
使用训练好的时序模型进行彩票号码预测
"""

import numpy as np
import pandas as pd
import torch
import joblib
from pathlib import Path
import logging
from typing import Dict, Any, List, Union
from datetime import datetime

logger = logging.getLogger(__name__)

class Predictor:
    """时序预测器类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化预测器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.model = None
        self.model_type = None
        self.sequence_length = config['model'].get('sequence_length', 30)
        self.device = torch.device('cuda' if torch.cuda.is_available() and config['training'].get('use_gpu', True) else 'cpu')
        
    def load_model(self, filepath: str):
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
        """
        model_path = Path(filepath)
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        # 检查文件扩展名
        if model_path.suffix in ['.pth', '.pt']:
            # PyTorch模型
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 获取模型类型和参数
            self.model_type = checkpoint.get('model_type', 'lstm')
            model_params = checkpoint.get('model_params', {})
            self.sequence_length = checkpoint.get('sequence_length', 30)
            
            # 重新创建模型结构
            from src.model_trainer import LSTMModel, GRUModel, TransformerModel, TCNModel
            
            input_size = checkpoint.get('input_size', 82)
            
            if self.model_type == 'lstm':
                self.model = LSTMModel(
                    input_size=input_size,
                    hidden_size=model_params.get('hidden_size', 256),
                    num_layers=model_params.get('num_layers', 3),
                    output_size=49,
                    dropout=model_params.get('dropout', 0.2),
                    bidirectional=model_params.get('bidirectional', True)
                )
            elif self.model_type == 'gru':
                self.model = GRUModel(
                    input_size=input_size,
                    hidden_size=model_params.get('hidden_size', 256),
                    num_layers=model_params.get('num_layers', 3),
                    output_size=49,
                    dropout=model_params.get('dropout', 0.2)
                )
            elif self.model_type == 'transformer':
                self.model = TransformerModel(
                    input_size=input_size,
                    d_model=model_params.get('d_model', 512),
                    nhead=model_params.get('nhead', 8),
                    num_layers=model_params.get('num_layers', 6),
                    output_size=49,
                    dropout=model_params.get('dropout', 0.1)
                )
            elif self.model_type == 'tcn':
                self.model = TCNModel(
                    input_size=input_size,
                    num_channels=model_params.get('num_channels', [64, 128, 256]),
                    kernel_size=model_params.get('kernel_size', 3),
                    dropout=model_params.get('dropout', 0.2),
                    output_size=49
                )
            
            # 加载模型权重
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
        else:
            # 尝试作为sklearn模型加载（向后兼容）
            self.model = joblib.load(model_path)
            self.model_type = 'sklearn'
        
        logger.info(f"模型加载成功: {filepath} (类型: {self.model_type})")
    
    def predict_next(self, historical_data: pd.DataFrame, feature_engineer) -> Dict[str, Any]:
        """
        预测下一期号码
        
        Args:
            historical_data: 历史数据
            feature_engineer: 特征工程器实例
            
        Returns:
            预测结果字典
        """
        if self.model is None:
            raise ValueError("请先加载模型")
        
        # 使用特征工程器准备序列数据
        sequence_features = feature_engineer.transform_for_prediction(
            historical_data, 
            last_n=self.config['prediction'].get('recent_periods', 100)
        )
        
        # 确保序列长度正确
        if len(sequence_features) != self.sequence_length:
            logger.warning(f"序列长度不匹配: {len(sequence_features)} vs {self.sequence_length}")
        
        # 预测
        if self.model_type in ['lstm', 'gru', 'transformer', 'tcn']:
            # 深度学习模型预测
            probabilities = self._predict_deep_learning(sequence_features)
        else:
            # sklearn模型预测（向后兼容）
            probabilities = self._predict_sklearn(sequence_features)
        
        # 解析预测结果
        prediction_result = self._parse_predictions_advanced(probabilities)
        
        # 添加模型信息
        prediction_result['model_type'] = self.model_type
        prediction_result['sequence_length'] = self.sequence_length
        
        return prediction_result
    
    def _predict_deep_learning(self, sequence: np.ndarray) -> np.ndarray:
        """
        使用深度学习模型预测
        
        Args:
            sequence: 输入序列
            
        Returns:
            预测概率
        """
        self.model.eval()
        
        with torch.no_grad():
            # 准备输入
            if len(sequence.shape) == 2:
                sequence = np.expand_dims(sequence, 0)  # 添加batch维度
            
            sequence_tensor = torch.FloatTensor(sequence).to(self.device)
            
            # 预测
            outputs = self.model(sequence_tensor)
            probabilities = outputs.cpu().numpy()[0]
        
        return probabilities
    
    def _predict_sklearn(self, features: np.ndarray) -> np.ndarray:
        """
        使用sklearn模型预测（向后兼容）
        
        Args:
            features: 特征
            
        Returns:
            预测概率
        """
        # 使用最后一个时间步的特征
        if len(features.shape) > 1:
            features = features[-1:] if len(features.shape) == 2 else features
        
        if hasattr(self.model, 'predict_proba'):
            # 获取预测概率
            probas = []
            if hasattr(self.model, 'estimators_'):
                for estimator in self.model.estimators_:
                    proba = estimator.predict_proba(features)
                    if proba.shape[1] == 2:
                        probas.append(proba[0, 1])
                    else:
                        probas.append(proba[0, 0])
            else:
                probas = self.model.predict_proba(features)[0]
            
            return np.array(probas)
        else:
            # 返回二值预测
            predictions = self.model.predict(features)[0]
            return predictions
    
    def _parse_predictions_advanced(self, probabilities: np.ndarray) -> Dict[str, Any]:
        """
        高级预测结果解析
        
        Args:
            probabilities: 预测概率数组
            
        Returns:
            解析后的预测结果
        """
        result = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'red_balls': [],
            'blue_ball': None,
            'confidence_scores': {},
            'probability_distribution': {}
        }
        
        # 分离红球和蓝球概率
        red_probs = probabilities[:33]
        blue_probs = probabilities[33:49]
        
        # 使用高级采样策略选择红球
        threshold = self.config['prediction'].get('probability_threshold', 0.5)
        top_k = self.config['prediction'].get('top_k', 10)
        
        # 1. 首先选择概率超过阈值的号码
        high_prob_indices = np.where(red_probs > threshold)[0]
        
        if len(high_prob_indices) >= 6:
            # 如果高概率号码足够，从中选择概率最高的6个
            selected_indices = high_prob_indices[np.argsort(red_probs[high_prob_indices])[-6:]]
        else:
            # 否则，使用Top-K策略
            top_k_indices = np.argsort(red_probs)[-top_k:]
            
            # 根据概率进行加权随机采样
            probs_normalized = red_probs[top_k_indices] / red_probs[top_k_indices].sum()
            
            # 设置随机种子以保证可重复性
            np.random.seed(int(datetime.now().timestamp()) % 1000000)
            
            # 采样6个不重复的号码
            selected_indices = np.random.choice(
                top_k_indices, 
                size=min(6, len(top_k_indices)), 
                replace=False,
                p=probs_normalized
            )
        
        result['red_balls'] = sorted([int(idx + 1) for idx in selected_indices])
        
        # 记录红球置信度
        for idx in selected_indices:
            result['confidence_scores'][f'red_{idx+1}'] = float(red_probs[idx])
        
        # 2. 选择蓝球（使用概率最高的）
        blue_idx = np.argmax(blue_probs)
        result['blue_ball'] = int(blue_idx + 1)
        result['confidence_scores'][f'blue_{blue_idx+1}'] = float(blue_probs[blue_idx])
        
        # 3. 计算整体置信度
        selected_probs = [red_probs[idx] for idx in selected_indices] + [blue_probs[blue_idx]]
        result['overall_confidence'] = float(np.mean(selected_probs))
        
        # 4. 记录完整的概率分布（用于分析）
        result['probability_distribution'] = {
            'red_balls': {str(i+1): float(p) for i, p in enumerate(red_probs)},
            'blue_balls': {str(i+1): float(p) for i, p in enumerate(blue_probs)}
        }
        
        # 5. 生成替代号码（基于概率排序）
        # 红球替代
        all_red_sorted = np.argsort(red_probs)[::-1]
        alternative_reds = []
        for idx in all_red_sorted:
            ball_num = int(idx + 1)
            if ball_num not in result['red_balls'] and len(alternative_reds) < 6:
                alternative_reds.append({
                    'number': ball_num,
                    'probability': float(red_probs[idx])
                })
        result['alternative_red_balls'] = alternative_reds
        
        # 蓝球替代
        blue_sorted = np.argsort(blue_probs)[::-1]
        alternative_blues = []
        for idx in blue_sorted[:3]:
            ball_num = int(idx + 1)
            if ball_num != result['blue_ball']:
                alternative_blues.append({
                    'number': ball_num,
                    'probability': float(blue_probs[idx])
                })
        result['alternative_blue_balls'] = alternative_blues
        
        # 6. 添加统计信息
        result['statistics'] = {
            'red_mean_prob': float(np.mean(red_probs)),
            'red_max_prob': float(np.max(red_probs)),
            'red_min_prob': float(np.min(red_probs)),
            'blue_mean_prob': float(np.mean(blue_probs)),
            'blue_max_prob': float(np.max(blue_probs)),
            'selected_red_mean_prob': float(np.mean([red_probs[idx] for idx in selected_indices])),
            'entropy': float(-np.sum(probabilities * np.log(probabilities + 1e-10)))  # 预测的不确定性
        }
        
        return result
    
    def predict_ensemble(self, historical_data: pd.DataFrame, feature_engineer, 
                        model_paths: List[str]) -> Dict[str, Any]:
        """
        集成预测（使用多个模型）
        
        Args:
            historical_data: 历史数据
            feature_engineer: 特征工程器
            model_paths: 模型文件路径列表
            
        Returns:
            集成预测结果
        """
        all_probabilities = []
        model_infos = []
        
        for model_path in model_paths:
            try:
                # 加载模型
                self.load_model(model_path)
                
                # 获取预测概率
                sequence_features = feature_engineer.transform_for_prediction(
                    historical_data,
                    last_n=self.config['prediction'].get('recent_periods', 100)
                )
                
                if self.model_type in ['lstm', 'gru', 'transformer', 'tcn']:
                    probs = self._predict_deep_learning(sequence_features)
                else:
                    probs = self._predict_sklearn(sequence_features)
                
                all_probabilities.append(probs)
                model_infos.append({
                    'path': model_path,
                    'type': self.model_type
                })
                
            except Exception as e:
                logger.error(f"加载模型 {model_path} 失败: {str(e)}")
        
        if not all_probabilities:
            raise ValueError("没有成功加载任何模型")
        
        # 集成策略
        ensemble_config = self.config.get('advanced_models', {}).get('ensemble', {})
        voting = ensemble_config.get('voting', 'soft')
        weights = ensemble_config.get('weights', None)
        
        if weights and len(weights) == len(all_probabilities):
            # 加权平均
            weights = np.array(weights) / np.sum(weights)
            ensemble_probs = np.average(all_probabilities, axis=0, weights=weights)
        else:
            # 简单平均
            ensemble_probs = np.mean(all_probabilities, axis=0)
        
        # 解析集成预测结果
        result = self._parse_predictions_advanced(ensemble_probs)
        
        # 添加集成信息
        result['ensemble_info'] = {
            'models': model_infos,
            'voting': voting,
            'num_models': len(all_probabilities)
        }
        
        # 添加各模型的预测对比
        result['model_predictions'] = []
        for i, (probs, info) in enumerate(zip(all_probabilities, model_infos)):
            model_pred = self._parse_predictions_advanced(probs)
            result['model_predictions'].append({
                'model': info['type'],
                'red_balls': model_pred['red_balls'],
                'blue_ball': model_pred['blue_ball'],
                'confidence': model_pred['overall_confidence']
            })
        
        return result
    
    def display_predictions(self, predictions: Dict[str, Any]):
        """
        在控制台显示预测结果（增强版）
        
        Args:
            predictions: 预测结果字典
        """
        print("\n" + "="*70)
        print(f"                    {predictions.get('model_type', 'AI')}模型预测结果")
        print("="*70)
        print(f"预测时间: {predictions['timestamp']}")
        print("-"*70)
        
        # 显示主要预测
        red_balls_str = ', '.join([f"{b:02d}" for b in sorted(predictions['red_balls'])])
        blue_ball_str = f"{predictions['blue_ball']:02d}"
        
        print(f"红球: {red_balls_str}")
        print(f"蓝球: {blue_ball_str}")
        
        # 显示置信度
        if 'overall_confidence' in predictions:
            confidence_pct = predictions['overall_confidence'] * 100
            print(f"\n整体置信度: {confidence_pct:.1f}%")
            
            # 置信度可视化
            bar_length = int(confidence_pct / 2)
            confidence_bar = "█" * bar_length + "░" * (50 - bar_length)
            print(f"[{confidence_bar}]")
        
        # 显示统计信息
        if 'statistics' in predictions:
            stats = predictions['statistics']
            print(f"\n预测统计:")
            print(f"  预测熵值: {stats['entropy']:.3f} (越低越确定)")
            print(f"  红球平均概率: {stats['red_mean_prob']*100:.1f}%")
            print(f"  选中红球平均概率: {stats['selected_red_mean_prob']*100:.1f}%")
        
        # 显示替代号码
        if predictions.get('alternative_red_balls'):
            print("\n备选红球 (按概率排序):")
            for i, alt in enumerate(predictions['alternative_red_balls'][:3]):
                print(f"  {i+1}. {alt['number']:02d} (概率: {alt['probability']*100:.1f}%)")
        
        if predictions.get('alternative_blue_balls'):
            print("\n备选蓝球:")
            for i, alt in enumerate(predictions['alternative_blue_balls'][:2]):
                print(f"  {i+1}. {alt['number']:02d} (概率: {alt['probability']*100:.1f}%)")
        
        # 显示集成信息（如果有）
        if 'ensemble_info' in predictions:
            print(f"\n集成预测信息:")
            print(f"  使用模型数: {predictions['ensemble_info']['num_models']}")
            print(f"  投票方式: {predictions['ensemble_info']['voting']}")
            
            if 'model_predictions' in predictions:
                print("\n各模型预测对比:")
                for i, pred in enumerate(predictions['model_predictions']):
                    red_str = ', '.join([f"{b:02d}" for b in sorted(pred['red_balls'])])
                    print(f"  {pred['model']}: 红[{red_str}] 蓝[{pred['blue_ball']:02d}] "
                          f"置信度:{pred['confidence']*100:.1f}%")
        
        print("\n" + "="*70)
        print("提示: 彩票具有随机性，预测仅供参考，请理性购彩！")
        print("="*70 + "\n")
    
    def save_predictions(self, predictions: Dict[str, Any], filepath: str):
        """
        保存预测结果（增强版）
        
        Args:
            predictions: 预测结果
            filepath: 保存路径
        """
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 准备数据
        data = {
            '预测时间': predictions['timestamp'],
            '模型类型': predictions.get('model_type', 'unknown'),
            '红球1': predictions['red_balls'][0] if len(predictions['red_balls']) > 0 else '',
            '红球2': predictions['red_balls'][1] if len(predictions['red_balls']) > 1 else '',
            '红球3': predictions['red_balls'][2] if len(predictions['red_balls']) > 2 else '',
            '红球4': predictions['red_balls'][3] if len(predictions['red_balls']) > 3 else '',
            '红球5': predictions['red_balls'][4] if len(predictions['red_balls']) > 4 else '',
            '红球6': predictions['red_balls'][5] if len(predictions['red_balls']) > 5 else '',
            '蓝球': predictions['blue_ball'],
            '整体置信度': f"{predictions.get('overall_confidence', 0)*100:.1f}%",
            '预测熵值': predictions.get('statistics', {}).get('entropy', '')
        }
        
        # 保存为CSV
        df = pd.DataFrame([data])
        
        # 如果文件已存在，追加数据
        if output_path.exists():
            existing_df = pd.read_csv(output_path, encoding='utf-8')
            df = pd.concat([existing_df, df], ignore_index=True)
        
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"预测结果已保存至: {output_path}")
        
        # 同时保存详细的JSON格式（包含概率分布）
        import json
        json_path = output_path.with_suffix('.json')
        
        # 读取或创建JSON列表
        predictions_list = []
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                predictions_list = json.load(f)
        
        predictions_list.append(predictions)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(predictions_list, f, indent=2, ensure_ascii=False)
        
        logger.info(f"详细预测结果已保存至: {json_path}")