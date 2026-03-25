import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import json
from datetime import datetime

from models.model import BertMLPClassifier, FGM, PGD
from data.dataset import GameSentimentDataset, create_dataloader
from configs.config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, OUTPUT_CONFIG

class Trainer:
    def __init__(self, use_fgm=True, use_pgd=False, resume_from=None, train_data_dir=None, val_data_dir=None, test_data_dir=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_CONFIG['model_name'])
        
        self.model = BertMLPClassifier(
            model_name=MODEL_CONFIG['model_name'],
            num_labels=MODEL_CONFIG['num_labels'],
            hidden_size=MODEL_CONFIG['hidden_size'],
            dropout_rate=MODEL_CONFIG['dropout_rate']
        ).to(self.device)
        
        self.use_fgm = use_fgm and TRAIN_CONFIG.get('use_fgm', False)
        self.use_pgd = use_pgd
        
        if self.use_fgm:
            self.fgm = FGM(self.model, epsilon=TRAIN_CONFIG.get('fgm_epsilon', 1.0))
            print("FGM adversarial training enabled")
        
        if self.use_pgd:
            self.pgd = PGD(self.model)
            print("PGD adversarial training enabled")
        
        os.makedirs(OUTPUT_CONFIG['checkpoint_dir'], exist_ok=True)
        os.makedirs(OUTPUT_CONFIG['log_dir'], exist_ok=True)
        
        self.best_f1 = 0
        self.patience_counter = 0
        self.start_epoch = 1
        
        self.train_data_dir = train_data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        
        self.resume_from = resume_from
        if resume_from:
            self._load_checkpoint(resume_from)
    
    def _load_checkpoint(self, checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("Model weights loaded successfully")
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer = None
            self._loaded_optimizer_state = checkpoint['optimizer_state_dict']
            print("Optimizer state loaded (will be applied after optimizer init)")
        
        if 'epoch' in checkpoint:
            self.start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {self.start_epoch}")
        
        if 'metrics' in checkpoint:
            print(f"Previous checkpoint metrics: F1={checkpoint['metrics'].get('f1', 'N/A'):.4f}")
    
    def prepare_data(self):
        print("Loading datasets...")
        
        train_path = self.train_data_dir if self.train_data_dir else os.path.join(DATA_CONFIG['processed_data_dir'], DATA_CONFIG['train_file'])
        val_path = self.val_data_dir if self.val_data_dir else os.path.join(DATA_CONFIG['processed_data_dir'], DATA_CONFIG['val_file'])
        test_path = self.test_data_dir if self.test_data_dir else os.path.join(DATA_CONFIG['processed_data_dir'], DATA_CONFIG['test_file'])
        
        print(f"Train data: {train_path}")
        print(f"Val data: {val_path}")
        print(f"Test data: {test_path}")
        
        train_dataset = GameSentimentDataset(
            train_path,
            self.tokenizer,
            max_length=DATA_CONFIG['max_seq_length']
        )
        
        val_dataset = GameSentimentDataset(
            val_path,
            self.tokenizer,
            max_length=DATA_CONFIG['max_seq_length']
        )
        
        test_dataset = GameSentimentDataset(
            test_path,
            self.tokenizer,
            max_length=DATA_CONFIG['max_seq_length']
        )
        
        self.train_loader = create_dataloader(
            train_dataset, 
            batch_size=TRAIN_CONFIG['batch_size'],
            shuffle=True
        )
        
        self.val_loader = create_dataloader(
            val_dataset,
            batch_size=TRAIN_CONFIG['batch_size'],
            shuffle=False
        )
        
        self.test_loader = create_dataloader(
            test_dataset,
            batch_size=TRAIN_CONFIG['batch_size'],
            shuffle=False
        )
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
    
    def prepare_optimizer(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': TRAIN_CONFIG['weight_decay']
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=TRAIN_CONFIG['learning_rate']
        )
        
        total_steps = len(self.train_loader) * TRAIN_CONFIG['epochs']
        warmup_steps = int(total_steps * TRAIN_CONFIG['warmup_ratio'])
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        if hasattr(self, '_loaded_optimizer_state') and self._loaded_optimizer_state:
            self.optimizer.load_state_dict(self._loaded_optimizer_state)
            print("Optimizer state restored from checkpoint")
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            
            loss = outputs['loss']
            total_loss += loss.item()
            
            loss.backward()
            
            if self.use_fgm:
                self.fgm.attack()
                outputs_adv = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
                loss_adv = outputs_adv['loss']
                loss_adv.backward()
                self.fgm.restore()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), TRAIN_CONFIG['max_grad_norm'])
            
            self.optimizer.step()
            self.scheduler.step()
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def evaluate(self, data_loader, mode='val'):
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"Evaluating ({mode})"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch['token_type_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
                
                loss = outputs['loss']
                total_loss += loss.item()
                
                preds = torch.argmax(outputs['logits'], dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        
        avg_loss = total_loss / len(data_loader)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
        
        if mode == 'test':
            print("\n" + "=" * 50)
            print("Classification Report:")
            print("=" * 50)
            target_names = [MODEL_CONFIG['label_map'][i] for i in range(4)]
            print(classification_report(all_labels, all_preds, target_names=target_names))
            
            print("\nConfusion Matrix:")
            cm = confusion_matrix(all_labels, all_preds)
            print(cm)
        
        return metrics
    
    def save_checkpoint(self, epoch, metrics, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': {
                'model_config': MODEL_CONFIG,
                'train_config': TRAIN_CONFIG
            }
        }
        
        checkpoint_path = os.path.join(
            OUTPUT_CONFIG['checkpoint_dir'],
            f'checkpoint_epoch_{epoch}.pt'
        )
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(OUTPUT_CONFIG['checkpoint_dir'], 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"Best model saved! F1: {metrics['f1']:.4f}")
    
    def train(self):
        self.prepare_data()
        self.prepare_optimizer()
        
        print("\n" + "=" * 50)
        print("Starting training...")
        print("=" * 50)
        
        for epoch in range(self.start_epoch, TRAIN_CONFIG['epochs'] + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{TRAIN_CONFIG['epochs']}")
            print("=" * 50)
            
            train_loss = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss:.4f}")
            
            val_metrics = self.evaluate(self.val_loader, mode='val')
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")
            
            is_best = val_metrics['f1'] > self.best_f1
            if is_best:
                self.best_f1 = val_metrics['f1']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            if self.patience_counter >= TRAIN_CONFIG['early_stop_patience']:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        print("\n" + "=" * 50)
        print("Training completed!")
        print("=" * 50)
        
        print("\nEvaluating on test set...")
        best_checkpoint = torch.load(
            os.path.join(OUTPUT_CONFIG['checkpoint_dir'], 'best_model.pt')
        )
        self.model.load_state_dict(best_checkpoint['model_state_dict'])
        
        test_metrics = self.evaluate(self.test_loader, mode='test')
        
        print(f"\nTest Results:")
        print(f"Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"F1 Score: {test_metrics['f1']:.4f}")
        print(f"Precision: {test_metrics['precision']:.4f}")
        print(f"Recall: {test_metrics['recall']:.4f}")
        
        return test_metrics


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='训练游戏玩家对话情绪分类模型')
    parser.add_argument('--resume', type=str, default=None, help='从指定checkpoint续训，路径如: outputs/checkpoints/best_model.pt')
    parser.add_argument('--train_data', type=str, default=None, help='训练数据路径 (CSV格式)')
    parser.add_argument('--val_data', type=str, default=None, help='验证数据路径 (CSV格式)')
    parser.add_argument('--test_data', type=str, default=None, help='测试数据路径 (CSV格式)')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--lr', type=float, default=None, help='学习率')
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小')
    
    args = parser.parse_args()
    
    if args.epochs:
        TRAIN_CONFIG['epochs'] = args.epochs
    if args.lr:
        TRAIN_CONFIG['learning_rate'] = args.lr
    if args.batch_size:
        TRAIN_CONFIG['batch_size'] = args.batch_size
    
    trainer = Trainer(
        use_fgm=True, 
        use_pgd=False,
        resume_from=args.resume,
        train_data_dir=args.train_data,
        val_data_dir=args.val_data,
        test_data_dir=args.test_data
    )
    trainer.train()


if __name__ == "__main__":
    main()
