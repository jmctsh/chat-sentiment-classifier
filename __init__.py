"""
游戏玩家对话情绪分类模型
========================

四分类任务：善意 / 辱骂 / 中性 / 中性玩梗

使用方法：
1. 安装依赖: pip install -r requirements.txt
2. 处理数据: python -m data.data_processor
3. 训练模型: python train.py
4. 推理预测: python inference.py
5. 评估模型: python evaluate.py
"""

__version__ = "1.0.0"
__author__ = "Game Sentiment Classifier"

from models.model import BertMLPClassifier, FGM, PGD
from data.dataset import GameSentimentDataset
from configs.config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, OUTPUT_CONFIG

__all__ = [
    'BertMLPClassifier',
    'FGM',
    'PGD',
    'GameSentimentDataset',
    'DATA_CONFIG',
    'MODEL_CONFIG',
    'TRAIN_CONFIG',
    'OUTPUT_CONFIG'
]
