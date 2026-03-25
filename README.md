# 游戏玩家对话情绪分类模型

## 项目简介

本项目用于游戏场景下玩家对话的情绪分类，支持四分类：
- **0 = 善意**：正常聊天、夸奖、友好、求助、合作
- **1 = 辱骂/恶意**：人身攻击、阴阳怪气骂人、谐音骂人、黑话骂人
- **2 = 中性**：纯信息、无情绪、问规则、问操作
- **3 = 中性玩梗**：无恶意的梗、搞笑、玩梗

## 项目结构

```
game_sentiment_classifier/
├── configs/
│   └── config.py          # 配置文件
├── data/
│   ├── raw/               # 原始数据
│   │   └── meme_supplement.csv  # 补充玩梗语料
│   ├── processed/         # 处理后的数据
│   ├── data_processor.py  # 数据处理脚本
│   └── dataset.py         # 数据集类
├── models/
│   └── model.py           # 模型定义（BERT+MLP）
├── checkpoints/           # 模型检查点
│   └── best_model.pt      # 最佳模型
├── train.py               # 训练脚本
├── inference.py           # 推理脚本
├── evaluate.py            # 评估脚本
├── app_gui.py             # 桌面应用（Tkinter）
└── README.md
```

## 快速开始

### 1. 安装依赖

#### 重要：先安装GPU版PyTorch

**为什么必须单独安装GPU版PyTorch？**

PyPI官方源默认只提供CPU版本的PyTorch。如果直接运行 `pip install torch`，会安装CPU版本，无法利用GPU加速训练。GPU版本需要从PyTorch官方wheel仓库下载。

**安装步骤：**

```bash
# 1. 创建虚拟环境
python -m venv venv

# 2. 激活虚拟环境
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
.\venv\Scripts\activate.bat
# Linux/Mac:
source venv/bin/activate

# 3. 安装GPU版PyTorch（CUDA 12.4）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. 安装其他依赖
pip install -r requirements.txt

# 5. 验证GPU可用
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

**预期输出：**
```
CUDA: True
GPU: NVIDIA GeForce RTX 4060 Ti
```

#### CUDA版本选择

| CUDA版本 | 安装命令 |
|----------|----------|
| CUDA 12.4 | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124` |
| CUDA 12.1 | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` |
| CUDA 11.8 | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` |

查看你的CUDA版本：`nvidia-smi`（查看右上角 CUDA Version）

### 2. 处理数据

```bash
python data/data_processor.py
```

### 3. 训练模型

```bash
python train.py
```

### 4. 推理预测

```bash
python inference.py
```

### 5. 启动桌面应用

```bash
python app_gui.py
```

#### 桌面应用功能

| 功能 | 说明 |
|------|------|
| 情绪分类 | 输入文本，自动识别情绪类别 |
| 智能回复 | 豆包LLM以贴吧老哥风格回复 |
| 联网搜索 | 玩梗时自动搜索了解梗的含义 |

#### 回复策略

| 分类 | 回复策略 |
|------|----------|
| 🟢 善意 | 友善回应 |
| 🔴 辱骂 | 怼回去（文明用语） |
| ⚪ 中性 | 正常回复 |
| 🟡 玩梗 | 接梗或搜索后回复 |

---

## 续训微调指南

### 概述

续训微调允许你在已有模型基础上，使用新的语料进行增量训练。适用于：
- 补充特定领域的语料（如新增玩梗词汇）
- 修复模型在特定场景下的误判
- 持续优化模型效果

### 数据集格式

续训数据集必须是CSV格式，包含两列：

| 列名 | 类型 | 说明 |
|------|------|------|
| text | string | 文本内容 |
| label | int | 标签 (0=善意, 1=辱骂, 2=中性, 3=中性玩梗) |

**示例 (my_finetune_data.csv)：**
```csv
text,label
这波操作太秀了,0
你是个好人,0
你玩得真菜,1
会不会玩啊,1
今天天气不错,2
给我擦皮鞋,3
我要验牌,3
```

### 续训命令

#### 1. 使用新数据集续训（推荐）

```bash
python train.py --resume outputs/checkpoints/best_model.pt --train_data data/my_finetune_data.csv --val_data data/processed/val.csv --test_data data/processed/test.csv --epochs 5
```

#### 2. 仅加载模型权重，使用默认数据

```bash
python train.py --resume outputs/checkpoints/best_model.pt
```

#### 3. 完整参数示例

```bash
python train.py \
    --resume outputs/checkpoints/best_model.pt \
    --train_data data/processed/train.csv \
    --val_data data/processed/val.csv \
    --test_data data/processed/test.csv \
    --epochs 10 \
    --lr 2e-5 \
    --batch_size 16
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--resume` | checkpoint路径，用于续训 | None |
| `--train_data` | 训练数据路径 | data/processed/train.csv |
| `--val_data` | 验证数据路径 | data/processed/val.csv |
| `--test_data` | 测试数据路径 | data/processed/test.csv |
| `--epochs` | 训练轮数 | 10 |
| `--lr` | 学习率 | 3e-5 |
| `--batch_size` | 批次大小 | 32 |

### 续训流程

```
1. 准备数据
   └── 创建CSV文件，格式: text,label

2. 运行续训
   └── python train.py --resume <checkpoint> --train_data <your_data.csv>

3. 验证效果
   └── python evaluate.py
```

### 补充玩梗语料

项目提供了 `data/raw/meme_supplement.csv` 用于手动补充玩梗语料：

**格式：**
```csv
text,label
给我擦皮鞋,3
我要验牌,3
牌没有问题,3
小儿科,3
小瘪三,3
```

**使用方法：**
1. 编辑 `data/raw/meme_supplement.csv`，添加新的玩梗文本
2. 运行数据处理：`python data/data_processor.py`
3. 续训模型：`python train.py --resume outputs/checkpoints/best_model.pt`

---

## 模型架构

```
文本输入 → BERT → CLS池化 → 两层MLP → Softmax → 4分类输出
```

### 关键特性

1. **BERT + 两层MLP分类头**：有效识别阴阳怪气、隐喻、反讽
2. **FGM对抗训练**：提升对谐音、变形词的识别能力
3. **数据增强**：包含游戏黑话、阴阳怪气、谐音骂人等语料

## 数据集

项目使用以下公开数据集：
- **COLDataset**：中文冒犯语言检测数据集（贴吧/知乎）
- **ToxiCN**：中文毒性评论数据集
- **BullyDataset**：网络霸凌数据集（微博）
- **CHIME**：中文网络梗解释数据集

## 训练配置

- 模型：bert-base-chinese
- 学习率：3e-5
- Batch Size：32
- Epochs：10
- 对抗训练：FGM (epsilon=1.0)
- 早停：patience=3

## API接口

```python
from inference import SentimentPredictor

predictor = SentimentPredictor()
result = predictor.predict("这队友太坑了")

# 输出:
# {
#   'text': '这队友太坑了',
#   'label': '辱骂',
#   'label_id': 1,
#   'confidence': 0.92
# }
```

## 评估指标

- Accuracy：整体准确率
- F1 Score：宏平均F1
- 辱骂类召回率：重点指标
- 混淆矩阵：分析误判情况