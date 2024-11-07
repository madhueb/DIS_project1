import torch


def pooling(hidden_states, attention_mask):
    """
    Pooling the hidden states into a single mean vector
    Args:
        hidden_states: hidden states from the model
        attention_mask: attention mask from the model
    Returns:
        mean_embeddings: mean embeddings of the hidden states
    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask
    return mean_embeddings


def query_embedding(query, model, tokenizer, config):
    """
    Compute embeddings for the given query
    Args:
        query: input query
        model: pre-trained model
        tokenizer: tokenizer for the model
        config: configuration dictionary
    Returns:
        outputs: embeddings for the query
    """
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(query,
                           return_tensors='pt',
                           padding=True,
                           truncation=True,
                           add_special_tokens=True,
                           max_length=config['max_length']
                           )
        inputs = {k: v.to(config['device']) for k, v in inputs.items()}
        outputs = model(**inputs, return_dict=True)
        if config['use_CLS']:
            outputs = outputs['last_hidden_state'][:, 0, :].cpu().numpy()
        else:
            outputs = pooling(outputs['last_hidden_state'], inputs['attention_mask']).cpu().numpy()
    return outputs