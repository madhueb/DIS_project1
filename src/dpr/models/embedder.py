
from transformers import AutoModel

import torch.nn as nn

from src.dpr.models.utils import pooling

class Embedder(nn.Module):

    def __init__(self, config):
        super(Embedder, self).__init__()
        self.model_name = config['model']
        self.config = config
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()

    def forward(self, inputs):
        outputs = self.model(**inputs, return_dict=True)
        if self.config['use_CLS']:
            outputs = outputs['last_hidden_state'][:, 0, :]
        else:
            outputs = pooling(outputs['last_hidden_state'], inputs['attention_mask'])
        return outputs