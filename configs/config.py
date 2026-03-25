import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_CONFIG = {
    "raw_data_dir": os.path.join(BASE_DIR, "data", "raw"),
    "processed_data_dir": os.path.join(BASE_DIR, "data", "processed"),
    "max_seq_length": 128,
    "train_file": "train.csv",
    "val_file": "val.csv",
    "test_file": "test.csv",
}

MODEL_CONFIG = {
    "model_name": "bert-base-chinese",
    "num_labels": 4,
    "hidden_size": 768,
    "dropout_rate": 0.2,
    "label_map": {
        0: "善意",
        1: "辱骂",
        2: "中性",
        3: "中性玩梗"
    }
}

TRAIN_CONFIG = {
    "batch_size": 32,
    "learning_rate": 3e-5,
    "weight_decay": 1e-4,
    "epochs": 10,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "early_stop_patience": 3,
    "seed": 42,
    "use_fgm": True,
    "fgm_epsilon": 1.0,
}

OUTPUT_CONFIG = {
    "checkpoint_dir": os.path.join(BASE_DIR, "checkpoints"),
    "log_dir": os.path.join(BASE_DIR, "logs"),
    "save_steps": 500,
}
