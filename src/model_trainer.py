# -*- coding: utf-8 -*-
"""
时序模型训练模块
支持最新的深度学习时序预测模型
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib
import json
from pathlib import Path
import logging
from typing import Dict, Any, Tuple, List
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# 检查是否有GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"使用设备: {device}")

class LotteryDataset(Dataset):
    """彩票数据集类"""
    
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class LSTMModel(nn.Module):
    """改进的LSTM模型"""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2, bidirectional=True):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * (2 if bidirectional else 1),
            num_heads=8,
            dropout=dropout
        )
        
        # 全连接层
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_output_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # 输出激活
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # LSTM前向传播
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # 注意力机制
        lstm_out = lstm_out.transpose(0, 1)  # (seq_len, batch, features)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = attn_out.transpose(0, 1)  # (batch, seq_len, features)
        
        # 使用最后一个时间步的输出
        out = attn_out[:, -1, :]
        
        # 全连接层
        out = self.fc_layers(out)
        out = self.sigmoid(out)
        
        return out

class GRUModel(nn.Module):
    """GRU模型（门控循环单元）"""
    
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(GRUModel, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = self.fc(gru_out[:, -1, :])
        return out

class TransformerModel(nn.Module):
    """Transformer时序预测模型"""
    
    def __init__(self, input_size, d_model=512, nhead=8, num_layers=6, output_size=49, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        # 输入嵌入
        self.input_embedding = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 输出层
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 输入嵌入
        x = self.input_embedding(x)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)
        
        # 使用最后一个时间步
        x = x[:, -1, :]
        
        # 解码输出
        output = self.decoder(x)
        
        return output

class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TCNModel(nn.Module):
    """时间卷积网络（TCN）"""
    
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.2, output_size=49):
        super(TCNModel, self).__init__()
        
        self.tcn = TemporalConvNet(
            num_inputs=input_size,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(num_channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_size),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # TCN expects (batch, channels, length)
        x = x.transpose(1, 2)
        y = self.tcn(x)
        # 取最后一个时间步
        y = y[:, :, -1]
        output = self.decoder(y)
        return output

class TemporalConvNet(nn.Module):
    """TCN实现"""
    
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers += [TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size,
                padding=(kernel_size-1) * dilation_size,
                dropout=dropout
            )]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class TemporalBlock(nn.Module):
    """TCN时间块"""
    
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Chomp1d(nn.Module):
    """裁剪卷积输出"""
    
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class ModelTrainer:
    """时序模型训练器"""
    
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
        self.device = device
        self.sequence_length = config['model'].get('sequence_length', 30)
        
    def _create_model(self, input_size: int, output_size: int = 49):
        """
        创建模型
        
        Args:
            input_size: 输入特征维度
            output_size: 输出维度（49 = 33红球 + 16蓝球）
        """
        logger.info(f"创建 {self.model_type} 模型...")
        
        if self.model_type == 'lstm':
            model = LSTMModel(
                input_size=input_size,
                hidden_size=self.model_params.get('hidden_size', 256),
                num_layers=self.model_params.get('num_layers', 3),
                output_size=output_size,
                dropout=self.model_params.get('dropout', 0.2),
                bidirectional=self.model_params.get('bidirectional', True)
            )
            
        elif self.model_type == 'gru':
            model = GRUModel(
                input_size=input_size,
                hidden_size=self.model_params.get('hidden_size', 256),
                num_layers=self.model_params.get('num_layers', 3),
                output_size=output_size,
                dropout=self.model_params.get('dropout', 0.2)
            )
            
        elif self.model_type == 'transformer':
            model = TransformerModel(
                input_size=input_size,
                d_model=self.model_params.get('d_model', 512),
                nhead=self.model_params.get('nhead', 8),
                num_layers=self.model_params.get('num_layers', 6),
                output_size=output_size,
                dropout=self.model_params.get('dropout', 0.1)
            )
            
        elif self.model_type == 'tcn':
            num_channels = self.model_params.get('num_channels', [64, 128, 256])
            model = TCNModel(
                input_size=input_size,
                num_channels=num_channels,
                kernel_size=self.model_params.get('kernel_size', 3),
                dropout=self.model_params.get('dropout', 0.2),
                output_size=output_size
            )
            
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
        
        return model.to(self.device)
    
    def _prepare_sequences(self, features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备时序数据
        
        Args:
            features: 特征数组
            labels: 标签数组
            
        Returns:
            (sequences, targets) 序列数据和目标
        """
        sequences = []
        targets = []
        
        for i in range(self.sequence_length, len(features)):
            # 获取序列
            seq = features[i-self.sequence_length:i]
            target = labels[i]
            
            sequences.append(seq)
            targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def train(self, train_data: Tuple[np.ndarray, np.ndarray], 
              val_data: Tuple[np.ndarray, np.ndarray] = None):
        """
        训练模型
        
        Args:
            train_data: (X_train, y_train) 训练数据
            val_data: (X_val, y_val) 验证数据
        """
        X_train, y_train = train_data
        
        # 准备序列数据
        X_train_seq, y_train_seq = self._prepare_sequences(X_train, y_train)
        
        if val_data is not None:
            X_val, y_val = val_data
            X_val_seq, y_val_seq = self._prepare_sequences(X_val, y_val)
        
        # 创建数据加载器
        train_dataset = LotteryDataset(X_train_seq, y_train_seq)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.model_params.get('batch_size', 32),
            shuffle=True
        )
        
        if val_data is not None:
            val_dataset = LotteryDataset(X_val_seq, y_val_seq)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.model_params.get('batch_size', 32),
                shuffle=False
            )
        
        # 创建模型
        input_size = X_train_seq.shape[2]
        self.model = self._create_model(input_size)
        
        # 损失函数和优化器
        criterion = nn.BCELoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.model_params.get('learning_rate', 0.001),
            weight_decay=self.model_params.get('weight_decay', 1e-5)
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        ) # , verbose=True
        
        # 训练循环
        epochs = self.model_params.get('epochs', 100)
        best_val_loss = float('inf')
        patience = self.model_params.get('patience', 10)
        patience_counter = 0
        
        logger.info(f"开始训练，训练样本数: {len(X_train_seq)}")
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # 计算准确率
                predicted = (outputs > 0.5).float()
                train_correct += (predicted == batch_y).float().mean().item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_train_acc = train_correct / len(train_loader)
            
            # 验证阶段
            if val_data is not None:
                self.model.eval()
                val_loss = 0.0
                val_correct = 0
                
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        outputs = self.model(batch_x)
                        loss = criterion(outputs, batch_y)
                        
                        val_loss += loss.item()
                        
                        predicted = (outputs > 0.5).float()
                        val_correct += (predicted == batch_y).float().mean().item()
                
                avg_val_loss = val_loss / len(val_loader)
                avg_val_acc = val_correct / len(val_loader)
                
                # 学习率调整
                scheduler.step(avg_val_loss)
                
                # 早停检查
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # 保存最佳模型
                    self._save_best_model()
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    logger.info(f"早停触发，在epoch {epoch+1}")
                    break
                
                logger.info(f"Epoch [{epoch+1}/{epochs}] "
                          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
                          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
            else:
                logger.info(f"Epoch [{epoch+1}/{epochs}] "
                          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}")
            
            # 记录训练历史
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_acc': avg_train_acc,
                'val_loss': avg_val_loss if val_data else None,
                'val_acc': avg_val_acc if val_data else None
            })
        
        logger.info("训练完成!")
        
        # 加载最佳模型
        if val_data is not None and hasattr(self, '_best_model_state'):
            self.model.load_state_dict(self._best_model_state)
            logger.info("已加载最佳模型")
    
    def _save_best_model(self):
        """保存最佳模型状态"""
        self._best_model_state = self.model.state_dict().copy()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: 输入特征序列
            
        Returns:
            预测结果
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        self.model.eval()
        
        # 确保输入是正确的形状
        if len(X.shape) == 2:
            # 如果是单个序列，添加batch维度
            X = X.unsqueeze(0) if isinstance(X, torch.Tensor) else np.expand_dims(X, 0)
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            predictions = (outputs > 0.5).cpu().numpy()
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率
        
        Args:
            X: 输入特征序列
            
        Returns:
            预测概率
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        self.model.eval()
        
        # 确保输入是正确的形状
        if len(X.shape) == 2:
            X = np.expand_dims(X, 0)
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            probabilities = outputs.cpu().numpy()
        
        return probabilities
    
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
        
        # 保存模型状态
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'model_params': self.model_params,
            'training_history': self.training_history,
            'sequence_length': self.sequence_length,
            'input_size': list(self.model.parameters())[0].shape[1] if hasattr(self.model, 'lstm') else None
        }
        
        torch.save(save_dict, output_path)
        
        # 保存元数据
        meta_path = output_path.with_suffix('.meta.json')
        meta_data = {
            'model_type': self.model_type,
            'model_params': self.model_params,
            'training_history': self.training_history,
            'sequence_length': self.sequence_length
        }
        
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"模型已保存至: {output_path}")
    
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
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 恢复模型参数
        self.model_type = checkpoint['model_type']
        self.model_params = checkpoint['model_params']
        self.training_history = checkpoint.get('training_history', [])
        self.sequence_length = checkpoint.get('sequence_length', 30)
        
        # 创建模型结构
        input_size = checkpoint.get('input_size', 82)  # 默认输入大小
        self.model = self._create_model(input_size)
        
        # 加载模型权重
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        logger.info(f"模型加载成功: {filepath}")