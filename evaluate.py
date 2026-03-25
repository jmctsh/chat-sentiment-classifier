import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer

from models.model import BertMLPClassifier
from data.dataset import GameSentimentDataset
from configs.config import DATA_CONFIG, MODEL_CONFIG, OUTPUT_CONFIG

class Evaluator:
    def __init__(self, checkpoint_path: str = None):
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
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.label_map = MODEL_CONFIG['label_map']
    
    def evaluate_dataset(self, data_path: str):
        dataset = GameSentimentDataset(
            data_path,
            self.tokenizer,
            max_length=DATA_CONFIG['max_seq_length']
        )
        
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                
                probs = torch.softmax(outputs['logits'], dim=1)
                preds = torch.argmax(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        return all_preds, all_labels, all_probs
    
    def compute_metrics(self, preds, labels, probs=None):
        metrics = {
            'accuracy': accuracy_score(labels, preds),
            'f1_macro': f1_score(labels, preds, average='macro'),
            'f1_weighted': f1_score(labels, preds, average='weighted'),
            'precision_macro': precision_score(labels, preds, average='macro'),
            'recall_macro': recall_score(labels, preds, average='macro'),
        }
        
        if probs is not None:
            try:
                metrics['auc_macro'] = roc_auc_score(labels, probs, multi_class='ovr', average='macro')
            except:
                pass
        
        return metrics
    
    def print_classification_report(self, preds, labels):
        target_names = [self.label_map[i] for i in range(len(self.label_map))]
        
        print("\n" + "=" * 60)
        print("Classification Report")
        print("=" * 60)
        print(classification_report(labels, preds, target_names=target_names, digits=4))
    
    def plot_confusion_matrix(self, preds, labels, save_path: str = None):
        target_names = [self.label_map[i] for i in range(len(self.label_map))]
        
        cm = confusion_matrix(labels, preds)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=target_names, yticklabels=target_names, ax=axes[0])
        axes[0].set_title('Confusion Matrix (Counts)')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=target_names, yticklabels=target_names, ax=axes[1])
        axes[1].set_title('Confusion Matrix (Normalized)')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.show()
    
    def evaluate(self, data_type: str = 'test'):
        if data_type == 'test':
            data_path = os.path.join(DATA_CONFIG['processed_data_dir'], DATA_CONFIG['test_file'])
        elif data_type == 'val':
            data_path = os.path.join(DATA_CONFIG['processed_data_dir'], DATA_CONFIG['val_file'])
        else:
            data_path = os.path.join(DATA_CONFIG['processed_data_dir'], DATA_CONFIG['train_file'])
        
        print(f"\nEvaluating on {data_type} set...")
        
        preds, labels, probs = self.evaluate_dataset(data_path)
        
        metrics = self.compute_metrics(preds, labels, probs)
        
        print("\n" + "=" * 60)
        print(f"Evaluation Results ({data_type} set)")
        print("=" * 60)
        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")
        
        self.print_classification_report(preds, labels)
        
        cm_path = os.path.join(OUTPUT_CONFIG['log_dir'], f'confusion_matrix_{data_type}.png')
        self.plot_confusion_matrix(preds, labels, save_path=cm_path)
        
        return metrics


def main():
    evaluator = Evaluator()
    evaluator.evaluate('test')


if __name__ == "__main__":
    main()
