import os
import re
import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.model_selection import train_test_split

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

LABEL_MAP = {
    0: "善意",
    1: "辱骂", 
    2: "中性",
    3: "中性玩梗"
}

def clean_text(text: str) -> str:
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    text = text.strip()
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http[s]?://\S+', '', text)
    text = re.sub(r'@[\w\u4e00-\u9fff]+', '', text)
    text = re.sub(r'#[\w\u4e00-\u9fff]+#', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def load_coldataset() -> pd.DataFrame:
    cold_dir = os.path.join(RAW_DATA_DIR, "COLDataset", "COLDataset")
    
    train_path = os.path.join(cold_dir, "train.csv")
    dev_path = os.path.join(cold_dir, "dev.csv")
    test_path = os.path.join(cold_dir, "test.csv")
    
    if not all(os.path.exists(p) for p in [train_path, dev_path, test_path]):
        print(f"COLDataset files not found")
        return pd.DataFrame()
    
    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)
    test_df = pd.read_csv(test_path)
    
    df = pd.concat([train_df, dev_df, test_df], ignore_index=True)
    
    if 'fine-grained-label' in df.columns:
        df = df[['TEXT', 'label', 'fine-grained-label']].copy()
        df.columns = ['text', 'label', 'fine_label']
        
        def map_cold_label(row):
            orig_label = row['label']
            fine_label = row.get('fine_label', 0)
            
            if orig_label == 1:
                return 1
            elif fine_label == 3:
                return 0
            else:
                return 2
        
        df['label'] = df.apply(map_cold_label, axis=1)
    else:
        df = df[['TEXT', 'label']].copy()
        df.columns = ['text', 'original_label']
        
        def map_cold_label_simple(row):
            if row['original_label'] == 1:
                return 1
            else:
                return 2
        
        df['label'] = df.apply(map_cold_label_simple, axis=1)
    
    df['text'] = df['text'].apply(clean_text)
    df = df[df['text'].str.len() > 0]
    df = df[['text', 'label']].copy()
    
    print(f"COLDataset loaded: {len(df)} samples")
    print(f"  - 善意(0): {len(df[df['label']==0])}")
    print(f"  - 辱骂(1): {len(df[df['label']==1])}")
    print(f"  - 中性(2): {len(df[df['label']==2])}")
    
    return df

def load_toxicn() -> pd.DataFrame:
    toxicn_path = os.path.join(RAW_DATA_DIR, "ToxiCN", "ToxiCN_1.0.csv")
    
    if not os.path.exists(toxicn_path):
        print(f"ToxiCN_1.0.csv not found")
        return pd.DataFrame()
    
    df = pd.read_csv(toxicn_path)
    
    required_cols = ['content', 'toxic']
    if not all(col in df.columns for col in required_cols):
        print(f"ToxiCN missing required columns")
        return pd.DataFrame()
    
    df = df[['content', 'toxic']].copy()
    df.columns = ['text', 'original_label']
    
    df['text'] = df['text'].apply(clean_text)
    df = df[df['text'].str.len() > 0]
    
    def map_toxicn_label(row):
        orig_label = row['original_label']
        
        if orig_label == 1:
            return 1
        else:
            return 2
    
    df['label'] = df.apply(map_toxicn_label, axis=1)
    df = df[['text', 'label']].copy()
    
    print(f"ToxiCN loaded: {len(df)} samples")
    print(f"  - 辱骂(1): {len(df[df['label']==1])}")
    print(f"  - 中性(2): {len(df[df['label']==2])}")
    
    return df

def load_toxicn_json() -> pd.DataFrame:
    json_path = os.path.join(RAW_DATA_DIR, "ToxiCN", "ToxiCN_ex", "ToxiCN", "data", "train.json")
    
    if not os.path.exists(json_path):
        print(f"ToxiCN train.json not found")
        return pd.DataFrame()
    
    import json
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    records = []
    for item in data:
        records.append({
            'text': item.get('content', ''),
            'toxic': item.get('toxic', 0),
            'toxic_type': item.get('toxic_type', 0),
            'expression': item.get('expression', 0),
            'platform': item.get('platform', '')
        })
    
    df = pd.DataFrame(records)
    df['text'] = df['text'].apply(clean_text)
    df = df[df['text'].str.len() > 0]
    
    def map_toxicn_json_label(row):
        toxic = row['toxic']
        
        if toxic == 1:
            return 1
        else:
            return 2
    
    df['label'] = df.apply(map_toxicn_json_label, axis=1)
    df = df[['text', 'label']].copy()
    
    print(f"ToxiCN JSON loaded: {len(df)} samples")
    
    return df

def load_bully_dataset() -> pd.DataFrame:
    bully_path = os.path.join(RAW_DATA_DIR, "BullyDataset", "weibo_media.txt")
    supply_path = os.path.join(RAW_DATA_DIR, "BullyDataset", "weibo_supplyment.txt")
    
    all_dfs = []
    
    if os.path.exists(bully_path):
        df1 = pd.read_csv(bully_path, sep='\t', encoding='utf-8')
        all_dfs.append(df1)
        print(f"Loaded weibo_media.txt: {len(df1)} rows")
    
    if os.path.exists(supply_path):
        df2 = pd.read_csv(supply_path, sep='\t', encoding='utf-8')
        all_dfs.append(df2)
        print(f"Loaded weibo_supplyment.txt: {len(df2)} rows")
    
    if not all_dfs:
        print(f"BullyDataset files not found")
        return pd.DataFrame()
    
    df = pd.concat(all_dfs, ignore_index=True)
    
    label_cols = [col for col in df.columns if 'annotator' in col.lower() or 'Annotator' in col]
    text_col = 'text' if 'text' in df.columns else None
    
    if not text_col or not label_cols:
        print(f"BullyDataset missing required columns. Available: {df.columns.tolist()}")
        return pd.DataFrame()
    
    df = df[[text_col] + label_cols].copy()
    df.columns = ['text'] + [f'label_{i}' for i in range(len(label_cols))]
    
    df['text'] = df['text'].apply(clean_text)
    df = df[df['text'].str.len() > 0]
    
    label_columns = [col for col in df.columns if col.startswith('label_')]
    
    def map_bully_label(row):
        labels = [row[col] for col in label_columns if pd.notna(row[col])]
        if not labels:
            return 2
        
        majority = sum(labels) / len(labels)
        
        if majority >= 0.5:
            return 1
        else:
            return 2
    
    df['label'] = df.apply(map_bully_label, axis=1)
    df = df[['text', 'label']].copy()
    
    print(f"BullyDataset loaded: {len(df)} samples")
    print(f"  - 辱骂(1): {len(df[df['label']==1])}")
    print(f"  - 中性(2): {len(df[df['label']==2])}")
    
    return df

def load_chime() -> pd.DataFrame:
    import json
    chime_path = os.path.join(RAW_DATA_DIR, "chime", "data", "chime_full.json")
    
    if not os.path.exists(chime_path):
        print(f"CHIME chime_full.json not found")
        return pd.DataFrame()
    
    with open(chime_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    records = []
    for item in data:
        examples = item.get('examples', [])
        for example in examples:
            cleaned = clean_text(example)
            if cleaned:
                records.append({
                    'text': cleaned,
                    'meme': item.get('meme', ''),
                    'type': item.get('type_cn', ''),
                    'offense': item.get('offense', False)
                })
    
    df = pd.DataFrame(records)
    df = df[df['text'].str.len() > 0]
    df = df.drop_duplicates(subset=['text'])
    
    df['label'] = 3
    
    df = df[['text', 'label']].copy()
    
    print(f"CHIME loaded: {len(df)} samples (中性玩梗)")
    print(f"  - 中性玩梗(3): {len(df[df['label']==3])}")
    
    return df

def load_meme_supplement() -> pd.DataFrame:
    supplement_path = os.path.join(RAW_DATA_DIR, "meme_supplement.csv")
    
    if not os.path.exists(supplement_path):
        print(f"meme_supplement.csv not found")
        return pd.DataFrame()
    
    df = pd.read_csv(supplement_path)
    
    if 'text' not in df.columns or 'label' not in df.columns:
        print(f"meme_supplement.csv missing 'text' or 'label' column")
        return pd.DataFrame()
    
    df['text'] = df['text'].apply(clean_text)
    df = df[df['text'].str.len() > 0]
    df = df.drop_duplicates(subset=['text'])
    
    df = df[['text', 'label']].copy()
    
    print(f"Meme Supplement loaded: {len(df)} samples")
    for label in range(4):
        count = len(df[df['label'] == label])
        if count > 0:
            print(f"  - {LABEL_MAP[label]}({label}): {count}")
    
    return df

def load_simplifyweibo_4moods() -> pd.DataFrame:
    weibo_path = os.path.join(RAW_DATA_DIR, "ChineseNlpCorpus", "datasets", "simplifyweibo_4_moods", "simplifyweibo_4_moods.csv")
    
    if not os.path.exists(weibo_path):
        print(f"simplifyweibo_4_moods.csv not found")
        print("Please download from: https://pan.baidu.com/s/16c93E5x373nsGozyWevITg")
        return pd.DataFrame()
    
    df = pd.read_csv(weibo_path)
    
    if 'label' not in df.columns or 'review' not in df.columns:
        print(f"simplifyweibo_4_moods missing required columns")
        return pd.DataFrame()
    
    df = df[['review', 'label']].copy()
    df.columns = ['text', 'original_label']
    
    df['text'] = df['text'].apply(clean_text)
    df = df[df['text'].str.len() > 0]
    
    def map_weibo_4moods_label(row):
        orig_label = row['original_label']
        
        if orig_label == 0:
            return 0
        elif orig_label == 1:
            return 1
        elif orig_label == 2:
            return 1
        else:
            return 2
    
    df['label'] = df.apply(map_weibo_4moods_label, axis=1)
    df = df[['text', 'label']].copy()
    
    print(f"simplifyweibo_4_moods loaded: {len(df)} samples")
    print(f"  - 善意(0): {len(df[df['label']==0])}")
    print(f"  - 辱骂(1): {len(df[df['label']==1])}")
    print(f"  - 中性(2): {len(df[df['label']==2])}")
    
    return df

def process_all_datasets():
    print("=" * 60)
    print("Loading datasets with original golden labels...")
    print("=" * 60)
    
    all_dfs = []
    
    cold_df = load_coldataset()
    if not cold_df.empty:
        all_dfs.append(cold_df)
    
    toxicn_df = load_toxicn()
    if not toxicn_df.empty:
        all_dfs.append(toxicn_df)
    
    toxicn_json_df = load_toxicn_json()
    if not toxicn_json_df.empty:
        all_dfs.append(toxicn_json_df)
    
    bully_df = load_bully_dataset()
    if not bully_df.empty:
        all_dfs.append(bully_df)
    
    chime_df = load_chime()
    if not chime_df.empty:
        all_dfs.append(chime_df)
    
    meme_supplement_df = load_meme_supplement()
    if not meme_supplement_df.empty:
        all_dfs.append(meme_supplement_df)
    
    weibo_df = load_simplifyweibo_4moods()
    if not weibo_df.empty:
        all_dfs.append(weibo_df)
    
    if not all_dfs:
        print("No datasets loaded! Please check data paths.")
        return None, None, None
    
    print("\n" + "=" * 60)
    print("Merging all datasets...")
    print("=" * 60)
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['text'])
    combined_df = combined_df[combined_df['text'].str.len() > 0]
    
    print(f"\nTotal samples: {len(combined_df)}")
    print("\nLabel distribution:")
    for label in range(4):
        count = len(combined_df[combined_df['label'] == label])
        print(f"  {LABEL_MAP[label]}({label}): {count}")
    
    print("\n" + "=" * 60)
    print("Splitting dataset...")
    print("=" * 60)
    
    train_df, temp_df = train_test_split(combined_df, test_size=0.3, random_state=42, stratify=combined_df['label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['label'])
    
    print(f"Train set: {len(train_df)}")
    print(f"Val set: {len(val_df)}")
    print(f"Test set: {len(test_df)}")
    
    print("\nTrain set label distribution:")
    for label in range(4):
        count = len(train_df[train_df['label'] == label])
        print(f"  {LABEL_MAP[label]}({label}): {count}")
    
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    train_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "test.csv"), index=False)
    
    print("\n" + "=" * 60)
    print("Dataset processing complete!")
    print(f"Files saved to: {PROCESSED_DATA_DIR}")
    print("=" * 60)
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    process_all_datasets()
