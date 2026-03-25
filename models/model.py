import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from typing import Optional

class BertMLPClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        num_labels: int = 4,
        hidden_size: int = 768,
        dropout_rate: float = 0.2,
        pretrained: bool = True
    ):
        super().__init__()
        
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        
        if pretrained:
            self.bert = BertModel.from_pretrained(model_name)
        else:
            self.bert = BertModel.from_pretrained(model_name)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_labels)
        )
        
        self.init_weights()
    
    def init_weights(self):
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        logits = self.mlp(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            "attentions": outputs.attentions if hasattr(outputs, "attentions") else None
        }
    
    def get_embeddings(self, input_ids, attention_mask=None, token_type_ids=None):
        with torch.no_grad():
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        return outputs.pooler_output


class FGM:
    def __init__(self, model: nn.Module, epsilon: float = 1.0):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}
    
    def attack(self, emb_name: str = 'word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)
    
    def restore(self, emb_name: str = 'word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD:
    def __init__(self, model: nn.Module, epsilon: float = 1.0, alpha: float = 0.3):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}
    
    def attack(self, emb_name: str = 'word_embeddings', is_first_attack: bool = False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data)
    
    def restore(self, emb_name: str = 'word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                param.data = self.emb_backup[name]
    
    def project(self, param_name, param_data):
        change = param_data - self.emb_backup[param_name]
        norm = torch.norm(change)
        if norm > self.epsilon:
            change = self.epsilon * change / norm
        return self.emb_backup[param_name] + change
    
    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()
    
    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]
