# -*- coding: utf-8 -*-
"""
预测模块
负责使用训练好的模型进行预测
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import logging
from typing import Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

class Predictor:
    """预测器类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化预测器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.model = None
        
    def load_model(self, filepath: str):
        """
        加载模型
        
        Args:
            filepath: 模型文件路径
        """
        model_path = Path(filepath)
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        self.model = joblib.load(model_path)
        logger.info(f"模型加载成功: {filepath}")
    
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
        
        # 使用特征工程器转换数据
        features = feature_engineer.transform_for_prediction(historical_data)
        
        # 预测
        predictions = self.model.predict(features)
        
        # 如果模型支持概率预测
        probabilities = None
        if hasattr(self.model, 'predict_proba'):
            try:
                # 获取每个号码的预测概率
                proba_results = []
                
                if hasattr(self.model, 'estimators_'):
                    # MultiOutputClassifier
                    for i, estimator in enumerate(self.model.estimators_):
                        proba = estimator.predict_proba(features)
                        # 获取预测为1的概率
                        if proba.shape[1] == 2:
                            proba_results.append(proba[0, 1])
                        else:
                            proba_results.append(proba[0, 0])
                else:
                    # 单输出模型
                    proba = self.model.predict_proba(features)
                    proba_results = proba[0]
                
                probabilities = np.array(proba_results)
            except Exception as e:
                logger.warning(f"无法获取预测概率: {str(e)}")
        
        # 解析预测结果
        prediction_result = self._parse_predictions(predictions[0], probabilities)
        
        return prediction_result
    
    def _parse_predictions(self, predictions: np.ndarray, probabilities: np.ndarray = None) -> Dict[str, Any]:
        """
        解析预测结果
        
        Args:
            predictions: 预测的二值数组
            probabilities: 预测概率数组
            
        Returns:
            解析后的预测结果
        """
        result = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'red_balls': [],
            'blue_ball': None,
            'confidence_scores': {}
        }
        
        # 解析红球（前33个位置）
        red_predictions = predictions[:33]
        red_indices = np.where(red_predictions == 1)[0]
        result['red_balls'] = [int(idx + 1) for idx in red_indices]
        
        # 如果预测的红球数量不对，根据概率选择
        if probabilities is not None and len(result['red_balls']) != 6:
            red_probs = probabilities[:33]
            # 选择概率最高的6个
            top_6_indices = np.argsort(red_probs)[-6:]
            result['red_balls'] = sorted([int(idx + 1) for idx in top_6_indices])
            
            # 记录置信度
            for idx in top_6_indices:
                result['confidence_scores'][f'red_{idx+1}'] = float(red_probs[idx])
        
        # 解析蓝球（后16个位置）
        blue_predictions = predictions[33:49]
        blue_indices = np.where(blue_predictions == 1)[0]
        
        if len(blue_indices) > 0:
            result['blue_ball'] = int(blue_indices[0] + 1)
        elif probabilities is not None:
            # 如果没有预测到蓝球，选择概率最高的
            blue_probs = probabilities[33:49]
            best_blue_idx = np.argmax(blue_probs)
            result['blue_ball'] = int(best_blue_idx + 1)
            result['confidence_scores'][f'blue_{best_blue_idx+1}'] = float(blue_probs[best_blue_idx])
        else:
            # 默认选择一个随机蓝球
            result['blue_ball'] = int(np.random.randint(1, 17))
        
        # 添加整体置信度
        if probabilities is not None:
            selected_indices = result['red_balls'] + [result['blue_ball'] + 32]
            selected_probs = []
            for idx in result['red_balls']:
                if idx-1 < len(probabilities):
                    selected_probs.append(probabilities[idx-1])
            if result['blue_ball'] + 32 < len(probabilities):
                selected_probs.append(probabilities[result['blue_ball'] + 32])
            
            if selected_probs:
                result['overall_confidence'] = float(np.mean(selected_probs))
        
        # 生成推荐的替代号码
        if probabilities is not None:
            # 红球替代（概率次高的）
            red_probs = probabilities[:33]
            sorted_red_indices = np.argsort(red_probs)[::-1]
            alternative_reds = []
            for idx in sorted_red_indices:
                ball_num = int(idx + 1)
                if ball_num not in result['red_balls'] and len(alternative_reds) < 3:
                    alternative_reds.append({
                        'number': ball_num,
                        'probability': float(red_probs[idx])
                    })
            result['alternative_red_balls'] = alternative_reds
            
            # 蓝球替代
            blue_probs = probabilities[33:49]
            sorted_blue_indices = np.argsort(blue_probs)[::-1]
            alternative_blues = []
            for idx in sorted_blue_indices[:3]:
                ball_num = int(idx + 1)
                if ball_num != result['blue_ball']:
                    alternative_blues.append({
                        'number': ball_num,
                        'probability': float(blue_probs[idx])
                    })
            result['alternative_blue_balls'] = alternative_blues
        
        return result
    
    def display_predictions(self, predictions: Dict[str, Any]):
        """
        在控制台显示预测结果
        
        Args:
            predictions: 预测结果字典
        """
        print("\n" + "="*60)
        print("                    预测结果")
        print("="*60)
        print(f"预测时间: {predictions['timestamp']}")
        print("-"*60)
        
        # 显示主要预测
        red_balls_str = ', '.join([f"{b:02d}" for b in sorted(predictions['red_balls'])])
        blue_ball_str = f"{predictions['blue_ball']:02d}"
        
        print(f"红球: {red_balls_str}")
        print(f"蓝球: {blue_ball_str}")
        
        # 显示置信度
        if 'overall_confidence' in predictions:
            confidence_pct = predictions['overall_confidence'] * 100
            print(f"\n整体置信度: {confidence_pct:.1f}%")
        
        # 显示各号码的置信度
        if predictions.get('confidence_scores'):
            print("\n各号码置信度:")
            for ball, conf in sorted(predictions['confidence_scores'].items()):
                print(f"  {ball}: {conf*100:.1f}%")
        
        # 显示替代号码
        if predictions.get('alternative_red_balls'):
            print("\n备选红球 (按概率排序):")
            for alt in predictions['alternative_red_balls']:
                print(f"  {alt['number']:02d} (概率: {alt['probability']*100:.1f}%)")
        
        if predictions.get('alternative_blue_balls'):
            print("\n备选蓝球 (按概率排序):")
            for alt in predictions['alternative_blue_balls']:
                print(f"  {alt['number']:02d} (概率: {alt['probability']*100:.1f}%)")
        
        print("\n" + "="*60)
        print("提示: 彩票具有随机性，预测仅供参考，请理性购彩！")
        print("="*60 + "\n")
    
    def save_predictions(self, predictions: Dict[str, Any], filepath: str):
        """
        保存预测结果到文件
        
        Args:
            predictions: 预测结果
            filepath: 保存路径
        """
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 准备数据
        data = {
            '预测时间': predictions['timestamp'],
            '红球1': predictions['red_balls'][0] if len(predictions['red_balls']) > 0 else '',
            '红球2': predictions['red_balls'][1] if len(predictions['red_balls']) > 1 else '',
            '红球3': predictions['red_balls'][2] if len(predictions['red_balls']) > 2 else '',
            '红球4': predictions['red_balls'][3] if len(predictions['red_balls']) > 3 else '',
            '红球5': predictions['red_balls'][4] if len(predictions['red_balls']) > 4 else '',
            '红球6': predictions['red_balls'][5] if len(predictions['red_balls']) > 5 else '',
            '蓝球': predictions['blue_ball'],
            '整体置信度': predictions.get('overall_confidence', '')
        }
        
        # 保存为CSV
        df = pd.DataFrame([data])
        
        # 如果文件已存在，追加数据
        if output_path.exists():
            existing_df = pd.read_csv(output_path, encoding='utf-8')
            df = pd.concat([existing_df, df], ignore_index=True)
        
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"预测结果已保存至: {output_path}")
        
        # 同时保存详细的JSON格式
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
    
    def batch_predict(self, historical_data: pd.DataFrame, feature_engineer, n_predictions: int = 5) -> List[Dict[str, Any]]:
        """
        批量预测多期号码
        
        Args:
            historical_data: 历史数据
            feature_engineer: 特征工程器
            n_predictions: 预测期数
            
        Returns:
            预测结果列表
        """
        predictions_list = []
        
        for i in range(n_predictions):
            logger.info(f"正在生成第 {i+1}/{n_predictions} 组预测...")
            
            # 每次预测可能会有轻微的随机性
            prediction = self.predict_next(historical_data, feature_engineer)
            prediction['batch_index'] = i + 1
            
            predictions_list.append(prediction)
        
        return predictions_list