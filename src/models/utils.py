import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.models.encoder import Encoder


def pooling(hidden_states, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask
    return mean_embeddings


def query_embedding(query, model, tokenizer, config):
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(query,
                           return_tensors='pt',
                           padding=True,
                           truncation=True,
                           add_special_tokens=True,
                           max_length=config['max_length']
                           )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(**inputs, return_dict=True)
        outputs = pooling(outputs['last_hidden_state'], inputs['attention_mask'])
    return outputs