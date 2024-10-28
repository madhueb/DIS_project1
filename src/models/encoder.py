
from transformers import AutoModel

import torch.nn as nn

from src.models.utils import pooling

class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.model_name = config['model']
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()

    def forward(self, inputs):
        outputs = self.model(**inputs, return_dict=True)
        outputs = pooling(outputs['last_hidden_state'], inputs['attention_mask'])
        return outputs