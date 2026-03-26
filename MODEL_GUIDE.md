# 模型调用指南

本文档说明如何调用训练好的游戏对话情绪分类模型。

## 模型信息

| 项目 | 值 |
|------|-----|
| 基础模型 | bert-base-chinese |
| 分类数量 | 4 |
| 隐藏层维度 | 768 |
| Dropout率 | 0.2 |
| 最大序列长度 | 128 |

## 标签映射

| 标签ID | 标签名称 | 说明 |
|--------|----------|------|
| 0 | 善意 | 正常聊天、夸奖、友好、求助、合作 |
| 1 | 辱骂 | 人身攻击、阴阳怪气骂人、谐音骂人、黑话骂人 |
| 2 | 中性 | 纯信息、无情绪、问规则、问操作 |
| 3 | 中性玩梗 | 无恶意的梗、搞笑、玩梗 |

## 依赖安装

```bash
pip install torch transformers
```

## 快速开始

### 方式一：使用推理脚本（推荐）

```python
from inference import SentimentPredictor

predictor = SentimentPredictor('checkpoints/best_model.pt')
result = predictor.predict("这队友太坑了")
print(result)
# {'text': '这队友太坑了', 'label': '辱骂', 'label_id': 1, 'confidence': 0.9998}
```

### 方式二：手动调用

```python
import torch
from transformers import BertTokenizer
from models.model import BertMLPClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

model = BertMLPClassifier(
    model_name='bert-base-chinese',
    num_labels=4,
    hidden_size=768,
    dropout_rate=0.2
).to(device)

checkpoint = torch.load('checkpoints/best_model.pt', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def predict(text):
    encoding = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    token_type_ids = encoding.get('token_type_ids')
    if token_type_ids is None:
        token_type_ids = torch.zeros((1, 128), dtype=torch.long).to(device)
    else:
        token_type_ids = token_type_ids.to(device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        probs = torch.softmax(outputs['logits'], dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0, pred].item()
    
    labels = {0: '善意', 1: '辱骂', 2: '中性', 3: '中性玩梗'}
    
    return {
        'text': text,
        'label': labels[pred],
        'label_id': pred,
        'confidence': confidence
    }

result = predict("今天天气不错")
print(result)
```

## 批量预测

```python
from inference import SentimentPredictor

predictor = SentimentPredictor()
texts = ["你好", "你是个傻逼", "今天天气不错", "耗子尾汁"]
results = predictor.predict_batch(texts, return_all_scores=True)

for r in results:
    print(f"{r['text']} -> {r['label']} ({r['confidence']:.2%})")
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `checkpoints/best_model.pt` | 训练好的模型权重 |
| `models/model.py` | 模型架构定义（BertMLPClassifier） |
| `configs/config.py` | 配置参数 |
| `inference.py` | 推理脚本 |

## 模型结构

```
文本输入
    ↓
Tokenizer (bert-base-chinese)
    ↓
BERT编码器 (12层Transformer)
    ↓
[CLS]向量 (768维)
    ↓
MLP分类头
    ├── Linear(768 → 768)
    ├── ReLU
    ├── Dropout(0.2)
    └── Linear(768 → 4)
    ↓
Softmax
    ↓
4分类概率输出
```

## 注意事项

1. **Tokenizer会自动下载**：首次运行时会从HuggingFace下载bert-base-chinese的tokenizer，需要网络连接

2. **GPU加速**：如果安装了CUDA版本的PyTorch，模型会自动使用GPU

3. **输入长度**：超过128个字符的文本会被截断

4. **置信度**：confidence字段表示模型对预测结果的置信程度，范围0-1

## 性能指标

在测试集上的表现：

| 指标 | 值 |
|------|-----|
| Accuracy | ~85% |
| Macro F1 | ~0.78 |
| 辱骂召回率 | ~0.82 |

## 联系方式

如有问题，请查看项目README或提交Issue。
