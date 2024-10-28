import torch

CONFIG = {
    'model': 'microsoft/mdeberta-v3-base',
    'dropout': 0.5,
    'max_length': 512,
    'batch_size': 8,
    'epochs': 7,
    'lr': 3e-4,
    'enable_scheduler': True,
    'scheduler': 'CosineAnnealingWarmRestarts',
    'gradient_accumulation_steps': 2,
    'adam_eps': 1e-6, # 1e-8 default
    'freeze_encoder': True,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'tokenizer_use_fast': False,
    'save_path': './models/',
    'load_path': './models/',
    'train_path': './Data/train.csv',
    'dev_path': './Data/dev.csv',
    'test_path': './Data/test.csv',
    'doc_embeds_path': './Data/corpus.json',
    'losses_path': './Data/losses.json',
}