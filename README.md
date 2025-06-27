# 中国福利彩票预测工具（仅供娱乐！）

基于人工智能的中国福利彩票（双色球）预测工具，使用机器学习算法分析历史开奖数据，提供下期号码预测参考。

> **免责声明**：彩票具有随机性，本工具仅供学习和研究使用，预测结果仅供参考。请理性购彩，切勿沉迷。

## 功能特性

- 🔍 **数据管理**：自动加载、清洗和管理历史开奖数据
- 🧮 **特征工程**：提取号码统计特征、时间特征、冷热号分析等
- 🤖 **多种模型**：支持随机森林、XGBoost、LightGBM、神经网络等算法
- 📊 **模型评估**：全面的性能评估指标和可视化报告
- 🎯 **智能预测**：预测下期号码并提供置信度分析
- 💻 **命令行界面**：简洁易用的CLI命令操作

## 快速开始

### 1. 环境要求

- Python 3.8 或更高版本
- 支持 Windows、Linux、macOS
- 无需 GPU（CPU 训练优化）

### 2. 安装步骤

```bash
# 克隆或下载项目
git clone <项目地址>
cd lottery_predictor

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 3. 准备数据

项目需要历史开奖数据，数据格式示例：

```csv
issue,date,numbers,sales,pool
2024001,2024-01-01,01,05,12,23,28,33+09,350000000,120000000
2024002,2024-01-04,03,08,15,22,27,31+12,380000000,150000000
```

如果没有数据，可以使用工具生成示例数据：

```bash
python -c "from src.utils import generate_sample_data; generate_sample_data()"
```

### 4. 基本使用

#### 查看帮助
```bash
python cli.py --help
```

#### 数据管理
```bash
# 加载CSV数据
python cli.py data-fetch --source csv --file data/raw/lottery.csv

# 清洗数据
python cli.py data-clean
```

#### 训练模型
```bash
# 使用默认配置训练随机森林模型
python cli.py train

# 指定模型类型
python cli.py train --model xgboost

# 使用自定义配置文件
python cli.py train --config configs/my_config.yml
```

#### 评估模型
```bash
python cli.py evaluate --model randomforest
```

#### 预测号码
```bash
# 预测下一期号码
python cli.py predict --model randomforest

# 保存预测结果
python cli.py predict --model randomforest --output predictions/next.csv
```

## 详细使用指南

### 配置文件

配置文件位于 `configs/config.yml`，主要配置项：

```yaml
model:
  type: randomforest  # 模型类型
  params:
    n_estimators: 100  # 树的数量
    max_depth: 10      # 最大深度

data:
  source: data/raw/lottery.csv  # 数据源路径
  train_ratio: 0.7              # 训练集比例
  val_ratio: 0.15               # 验证集比例
  test_ratio: 0.15              # 测试集比例
```

### 支持的模型

1. **RandomForest**（随机森林）
   - 优点：稳定性好，不易过拟合
   - 适用：中小规模数据集

2. **XGBoost**
   - 优点：性能优秀，训练速度快
   - 适用：各种规模数据集

3. **LightGBM**
   - 优点：内存占用少，速度快
   - 适用：大规模数据集

4. **MLP**（多层感知机）
   - 优点：可以学习复杂模式
   - 适用：数据量充足时

5. **LSTM**（长短期记忆网络）
   - 优点：适合时序数据
   - 适用：有明显时间依赖的数据

### 特征说明

系统自动提取以下特征：

1. **号码特征**
   - 号码独热编码
   - 号码统计（平均值、标准差、跨度）
   - 奇偶比例、大小比例

2. **时间特征**
   - 星期几、月份、季度
   - 是否月初/月末、是否周末

3. **统计特征**
   - 冷热号分析（最近30期）
   - 连续未出现期数
   - 号码出现频率

## 项目结构

```
lottery_predictor/
├── cli.py              # 命令行入口
├── requirements.txt    # 依赖包列表
├── README.md          # 项目说明
├── configs/           # 配置文件目录
│   └── config.yml     # 默认配置
├── data/              # 数据目录
│   ├── raw/          # 原始数据
│   └── processed/    # 处理后数据
├── models/           # 模型文件目录
├── reports/          # 评估报告目录
├── logs/             # 日志目录
└── src/              # 源代码目录
    ├── __init__.py
    ├── data_manager.py       # 数据管理模块
    ├── feature_engineering.py # 特征工程模块
    ├── model_trainer.py      # 模型训练模块
    ├── model_evaluator.py    # 模型评估模块
    ├── predictor.py          # 预测模块
    └── utils.py             # 工具函数

```

## 常见问题

### Q1: 预测准确率如何？
A: 由于彩票本质上是随机事件，完全准确预测是不可能的。本工具通过分析历史数据中的统计规律，提供可能性较高的号码组合作为参考。

### Q2: 需要多少历史数据？
A: 建议至少有500期以上的历史数据，数据越多，模型学习的模式越充分。

### Q3: 如何提高预测效果？
A: 可以尝试：
- 增加更多历史数据
- 调整模型超参数
- 尝试不同的模型算法
- 结合多个模型的预测结果

### Q4: 支持其他彩票类型吗？
A: 目前主要针对双色球设计，但代码结构支持扩展。修改数据格式验证和特征提取逻辑即可支持其他类型。

## 开发计划

- [ ] 支持更多彩票类型（大乐透、七乐彩等）
- [ ] 添加数据自动爬取功能
- [ ] 实现模型集成学习
- [ ] 添加Web界面
- [ ] 支持实时预测API

## 贡献指南

欢迎提交Issue和Pull Request！

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 致谢

- 感谢所有开源项目贡献者
- 特别感谢 scikit-learn、XGBoost、LightGBM 等优秀的机器学习库

---

**再次提醒**：彩票有风险，投注需谨慎。本工具仅供学习交流，切勿过度依赖预测结果。
