# 中国福利彩票预测工具配置文件

# 模型配置
model:
  # 模型类型: randomforest, xgboost, lightgbm, mlp, lstm
  type: randomforest
  
  # 模型参数
  params:
    # RandomForest参数
    n_estimators: 100
    max_depth: 10
    min_samples_split: 2
    min_samples_leaf: 1
    
    # XGBoost额外参数
    learning_rate: 0.1
    subsample: 0.8
    colsample_bytree: 0.8
    
    # LightGBM额外参数
    num_leaves: 31
    
    # MLP参数
    hidden_layers: [100, 50]
    activation: relu
    solver: adam
    max_iter: 500

# 数据配置
data:
  # 数据源文件路径
  source: data/raw/lottery.csv
  
  # 数据集划分比例
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  
  # 特征工程参数
  feature_engineering:
    # 统计窗口大小（用于计算冷热号等）
    window_size: 30
    # 是否使用时间特征
    use_time_features: true
    # 是否使用统计特征
    use_statistical_features: true

# 训练配置
training:
  # 批次大小（仅用于深度学习模型）
  batch_size: 32
  
  # 训练轮数（仅用于深度学习模型）
  epochs: 100
  
  # 早停设置
  early_stopping: true
  patience: 10
  
  # 随机种子
  random_seed: 42

# 预测配置
prediction:
  # 使用最近多少期数据进行预测
  recent_periods: 30
  
  # 批量预测时生成的组数
  batch_size: 5
  
  # 是否显示置信度
  show_confidence: true
  
  # 是否显示备选号码
  show_alternatives: true

# 评估配置
evaluation:
  # 交叉验证折数
  cv_folds: 5
  
  # 评估指标
  metrics:
    - accuracy
    - precision
    - recall
    - f1_score
  
  # 是否生成详细报告
  generate_report: true

# 日志配置
logging:
  # 日志级别: DEBUG, INFO, WARNING, ERROR
  level: INFO
  
  # 日志格式
  format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  
  # 是否保存到文件
  save_to_file: true
  
  # 日志文件路径
  log_file: logs/lottery_predictor.log

# 系统配置
system:
  # CPU核心数（-1表示使用所有可用核心）
  n_jobs: -1
  
  # 内存限制（GB，0表示不限制）
  memory_limit: 0
  
  # 是否显示进度条
  show_progress: true