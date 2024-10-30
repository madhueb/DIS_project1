import torch
ROOT = '/content/drive/MyDrive'
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
    'save_path': f'{ROOT}/models',
    'load_path': f'{ROOT}/models',
    'train_path': f'{ROOT}/train.csv',
    'dev_path': f'{ROOT}/dev.csv',
    'train_emb_path': f'{ROOT}/train_emb.csv',
    'dev_emb_path': f'{ROOT}/dev_emb.csv',
    'test_path': f'{ROOT}/test.csv',
    'submit_path': f'{ROOT}/submit.csv',
    'test_emb_path': f'{ROOT}/test_emb.csv',
    'doc_embeds_path': f'{ROOT}/doc_embeds.pkl',
    'doc_encodes_path': f'{ROOT}/doc_encodes.pkl',
    'losses_path': f'{ROOT}/losses.json',
    'k_chunk': 100,
    'k_doc': 10,
    'index_N': 128,
    'index_path': f'{ROOT}/index',
}