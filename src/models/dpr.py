
from transformers import AutoModel
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import gc

from accelerate import Accelerator

# ----------
import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

config = {
    'model': 'microsoft/deberta-v3-base',
    'dropout': 0.5,
    'max_length': 512,
    'batch_size': 8, # anything more results in CUDA OOM [for unfreezed encoder] on Kaggle GPU
    'epochs': 7,
    'lr': 3e-4,
    'enable_scheduler': True,
    'scheduler': 'CosineAnnealingWarmRestarts',
    'gradient_accumulation_steps': 2,
    'adam_eps': 1e-6, # 1e-8 default
    'freeze_encoder': True
}


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class DPRModel(nn.Module):

    def __init__(self, config):
        super(DPRModel, self).__init__()
        self.model_name = config['model']
        self.freeze = config.get('freeze_encoder', True)

        self.query_encoder = AutoModel.from_pretrained(self.model_name)
        if self.freeze:
            for param in self.encoder.base_model.parameters():
                param.requires_grad = False

        self.pooler = MeanPooling()
        # set linear block on top of encoders
        self.query_linear_seq = nn.Sequential(
            nn.Linear(self.query_encoder.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.query_encoder.config.hidden_size)
        )
        self.doc_linear_seq = nn.Sequential(
            nn.Linear(self.query_encoder.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.query_encoder.config.hidden_size)
        )

        # We concat the embeddings of query and doc.
        # Then we concat them with the cosine similarity and pass them through linear layers.
        self.fc1 = nn.Linear(self.query_encoder.config.hidden_size + self.doc_encoder.config.hidden_size + 1, 256)
        self.fc2 = nn.Linear(256, 2)


    def forward(self, query, doc_embeds, query_mask):

        query_outputs = self.query_encoder(query, attention_mask=query_mask, return_dict=True)

        query_outputs = self.pooler(query_outputs['last_hidden_state'], query_mask)

        query_outputs = self.query_linear_seq(query_outputs) + query_outputs
        doc_outputs = self.doc_linear_seq(doc_embeds) + doc_embeds

        # cosine similarity
        cosine_similarity = nn.CosineSimilarity(dim=1)(query_outputs, doc_outputs).unsqueeze(-1)

        # concat and pass through layers
        outputs = torch.cat([query_outputs, doc_outputs, cosine_similarity], dim=1)
        outputs = self.fc1(outputs)
        outputs = nn.ReLU()(outputs)
        outputs = self.fc2(outputs)
        outputs = nn.Softmax(dim=1)(outputs)

        # return the first probability as the relevant score
        return outputs[:, 0]