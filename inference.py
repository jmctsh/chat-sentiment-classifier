import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import BertTokenizer
import json
from typing import Dict, List, Optional
import numpy as np

from models.model import BertMLPClassifier
from configs.config import MODEL_CONFIG, OUTPUT_CONFIG

class SentimentPredictor:
    def __init__(self, checkpoint_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_CONFIG['model_name'])
        
        self.model = BertMLPClassifier(
            model_name=MODEL_CONFIG['model_name'],
            num_labels=MODEL_CONFIG['num_labels'],
            hidden_size=MODEL_CONFIG['hidden_size'],
            dropout_rate=MODEL_CONFIG['dropout_rate']
        ).to(self.device)
        
        if checkpoint_path is None:
            checkpoint_path = os.path.join(OUTPUT_CONFIG['checkpoint_dir'], 'best_model.pt')
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from: {checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}")
        
        self.model.eval()
        self.label_map = MODEL_CONFIG['label_map']
    
    def predict(self, text: str, return_all_scores: bool = False) -> Dict:
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        token_type_ids = encoding.get('token_type_ids', torch.zeros((1, 128), dtype=torch.long)).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_label].item()
        
        result = {
            'text': text,
            'label': self.label_map[pred_label],
            'label_id': pred_label,
            'confidence': round(confidence, 4)
        }
        
        if return_all_scores:
            all_scores = {
                self.label_map[i]: round(probs[0, i].item(), 4)
                for i in range(len(self.label_map))
            }
            result['all_scores'] = all_scores
        
        return result
    
    def predict_batch(self, texts: List[str], return_all_scores: bool = False) -> List[Dict]:
        results = []
        for text in texts:
            result = self.predict(text, return_all_scores)
            results.append(result)
        return results
    
    def predict_from_file(self, input_file: str, output_file: str):
        with open(input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        results = self.predict_batch(texts, return_all_scores=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        print(f"Predictions saved to: {output_file}")
        return results


def demo():
    predictor = SentimentPredictor()
    
    test_texts = [
        "这队友太坑了，会不会玩啊",
        "打得不错，下次继续加油",
        "去中路集合",
        "哈哈哈这波操作666",
        "你真是个天才，把塔送了",
        "稳住我们能赢",
        "这波在大气层",
        "别送了行吗",
    ]
    
    print("\n" + "=" * 60)
    print("游戏对话情绪分类演示")
    print("=" * 60)
    
    for text in test_texts:
        result = predictor.predict(text, return_all_scores=True)
        print(f"\n文本: {text}")
        print(f"预测: {result['label']} (置信度: {result['confidence']:.2%})")
        print(f"详细分数: {result['all_scores']}")


if __name__ == "__main__":
    demo()
