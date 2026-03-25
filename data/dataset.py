import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
from typing import Optional

class GameSentimentDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: BertTokenizer,
        max_length: int = 128,
        label_map: Optional[dict] = None
    ):
        self.df = pd.read_csv(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = label_map or {0: 0, 1: 1, 2: 2, 3: 3}
        
        self.texts = self.df['text'].tolist()
        self.labels = self.df['label'].tolist()
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        token_type_ids = encoding.get('token_type_ids')
        if token_type_ids is not None:
            token_type_ids = token_type_ids.flatten()
        else:
            token_type_ids = torch.zeros(self.max_length, dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': token_type_ids,
            'labels': torch.tensor(label, dtype=torch.long),
            'text': text
        }


def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=0):
    from torch.utils.data import DataLoader
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
