# -*- coding: utf-8 -*-
"""
模型评估模块
负责评估模型性能并生成报告
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import joblib
from pathlib import Path
import logging
from typing import Dict, Any, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """模型评估器类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化模型评估器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.model = None
        self.evaluation_results = {}
        
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
    
    def evaluate(self, test_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
        """
        评估模型性能
        
        Args:
            test_data: (X_test, y_test) 测试数据
            
        Returns:
            评估指标字典
        """
        if self.model is None:
            raise ValueError("请先加载模型")
        
        X_test, y_test = test_data
        logger.info(f"开始评估，测试样本数: {len(X_test)}")
        
        # 预测
        y_pred = self.model.predict(X_test)
        
        # 计算各种指标
        metrics = {}
        
        # 1. 整体准确率
        metrics['overall_accuracy'] = accuracy_score(y_test, y_pred)
        
        # 2. 每个输出的指标
        n_outputs = y_test.shape[1] if len(y_test.shape) > 1 else 1
        
        if n_outputs > 1:
            # 多输出情况
            output_metrics = []
            
            for i in range(n_outputs):
                output_name = f'output_{i+1}'
                
                # 对于彩票预测，区分红球和蓝球
                if i < 33:
                    output_name = f'red_ball_{i+1}'
                else:
                    output_name = f'blue_ball_{i-32}'
                
                output_metric = {
                    'name': output_name,
                    'accuracy': accuracy_score(y_test[:, i], y_pred[:, i]),
                    'precision': precision_score(y_test[:, i], y_pred[:, i], zero_division=0),
                    'recall': recall_score(y_test[:, i], y_pred[:, i], zero_division=0),
                    'f1_score': f1_score(y_test[:, i], y_pred[:, i], zero_division=0)
                }
                output_metrics.append(output_metric)
            
            metrics['output_metrics'] = output_metrics
            
            # 计算平均指标
            metrics['avg_accuracy'] = np.mean([m['accuracy'] for m in output_metrics])
            metrics['avg_precision'] = np.mean([m['precision'] for m in output_metrics])
            metrics['avg_recall'] = np.mean([m['recall'] for m in output_metrics])
            metrics['avg_f1_score'] = np.mean([m['f1_score'] for m in output_metrics])
            
            # 3. 组合准确率（所有号码都预测正确的比例）
            exact_match = np.all(y_test == y_pred, axis=1)
            metrics['exact_match_accuracy'] = np.mean(exact_match)
            
            # 4. 部分匹配统计
            matches_per_sample = np.sum(y_test == y_pred, axis=1)
            metrics['avg_matches_per_sample'] = np.mean(matches_per_sample)
            metrics['match_distribution'] = {
                f'{i}_matches': np.sum(matches_per_sample == i) 
                for i in range(min(8, n_outputs+1))  # 0-7个匹配
            }
            
        else:
            # 单输出情况
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # 5. 预测概率分析（如果模型支持）
        if hasattr(self.model, 'predict_proba'):
            try:
                y_proba = self.model.predict_proba(X_test)
                # 分析预测置信度
                if isinstance(y_proba, list):
                    # MultiOutputClassifier的情况
                    avg_confidence = np.mean([np.max(proba, axis=1).mean() 
                                            for proba in y_proba])
                else:
                    avg_confidence = np.max(y_proba, axis=1).mean()
                
                metrics['avg_prediction_confidence'] = avg_confidence
            except:
                logger.warning("无法计算预测置信度")
        
        # 保存评估结果
        self.evaluation_results = metrics
        
        # 打印主要指标
        logger.info("评估完成！主要指标：")
        logger.info(f"  整体准确率: {metrics.get('overall_accuracy', 0):.4f}")
        logger.info(f"  平均准确率: {metrics.get('avg_accuracy', 0):.4f}")
        logger.info(f"  完全匹配率: {metrics.get('exact_match_accuracy', 0):.4f}")
        logger.info(f"  平均匹配数: {metrics.get('avg_matches_per_sample', 0):.2f}")
        
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, Any]:
        """
        交叉验证
        
        Args:
            X: 特征数据
            y: 标签数据
            cv: 折数
            
        Returns:
            交叉验证结果
        """
        if self.model is None:
            raise ValueError("请先加载模型")
        
        logger.info(f"开始 {cv} 折交叉验证...")
        
        # 执行交叉验证
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        cv_results = {
            'cv_scores': scores.tolist(),
            'cv_mean': scores.mean(),
            'cv_std': scores.std(),
            'cv_min': scores.min(),
            'cv_max': scores.max()
        }
        
        logger.info(f"交叉验证完成！平均得分: {cv_results['cv_mean']:.4f} (±{cv_results['cv_std']:.4f})")
        
        return cv_results
    
    def generate_report(self, metrics: Dict[str, Any], filepath: str):
        """
        生成评估报告
        
        Args:
            metrics: 评估指标
            filepath: 报告保存路径
        """
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 生成Markdown格式的报告
        report_lines = [
            "# 模型评估报告",
            f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n## 1. 整体性能指标",
            f"\n- **整体准确率**: {metrics.get('overall_accuracy', 0):.4f}",
        ]
        
        if 'avg_accuracy' in metrics:
            report_lines.extend([
                f"- **平均准确率**: {metrics['avg_accuracy']:.4f}",
                f"- **平均精确率**: {metrics.get('avg_precision', 0):.4f}",
                f"- **平均召回率**: {metrics.get('avg_recall', 0):.4f}",
                f"- **平均F1分数**: {metrics.get('avg_f1_score', 0):.4f}",
                f"- **完全匹配率**: {metrics.get('exact_match_accuracy', 0):.4f}",
                f"- **平均匹配数**: {metrics.get('avg_matches_per_sample', 0):.2f}/7",
            ])
        
        if 'avg_prediction_confidence' in metrics:
            report_lines.append(f"- **平均预测置信度**: {metrics['avg_prediction_confidence']:.4f}")
        
        # 2. 匹配分布
        if 'match_distribution' in metrics:
            report_lines.extend([
                "\n## 2. 匹配分布",
                "\n| 匹配数 | 样本数 | 占比 |",
                "|--------|--------|------|"
            ])
            
            total_samples = sum(metrics['match_distribution'].values())
            for match_count, count in sorted(metrics['match_distribution'].items()):
                match_num = match_count.split('_')[0]
                percentage = (count / total_samples * 100) if total_samples > 0 else 0
                report_lines.append(f"| {match_num} | {count} | {percentage:.1f}% |")
        
        # 3. 各号码性能（前10个）
        if 'output_metrics' in metrics and len(metrics['output_metrics']) > 0:
            report_lines.extend([
                "\n## 3. 各号码预测性能（前10个）",
                "\n| 号码 | 准确率 | 精确率 | 召回率 | F1分数 |",
                "|------|--------|--------|--------|--------|"
            ])
            
            for metric in metrics['output_metrics'][:10]:
                report_lines.append(
                    f"| {metric['name']} | {metric['accuracy']:.4f} | "
                    f"{metric['precision']:.4f} | {metric['recall']:.4f} | "
                    f"{metric['f1_score']:.4f} |"
                )
        
        # 4. 建议
        report_lines.extend([
            "\n## 4. 分析与建议",
            "\n### 性能分析",
        ])
        
        if metrics.get('exact_match_accuracy', 0) < 0.001:
            report_lines.append("- 完全匹配率极低，这在彩票预测中是正常的，因为彩票本质上是随机的")
        
        if metrics.get('avg_matches_per_sample', 0) < 1:
            report_lines.append("- 平均匹配数较低，建议增加训练数据或尝试其他特征工程方法")
        
        report_lines.extend([
            "\n### 改进建议",
            "1. 增加更多的历史数据进行训练",
            "2. 尝试更复杂的特征工程，如号码间的关联性分析",
            "3. 使用集成学习方法组合多个模型",
            "4. 记住：彩票是随机事件，任何预测都仅供参考",
        ])
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        # 同时生成CSV格式的详细数据
        if 'output_metrics' in metrics:
            csv_path = output_path.with_suffix('.csv')
            df = pd.DataFrame(metrics['output_metrics'])
            df.to_csv(csv_path, index=False, encoding='utf-8')
            logger.info(f"详细指标已保存至: {csv_path}")
        
        logger.info(f"评估报告已生成: {output_path}")