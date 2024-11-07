
from transformers import AutoModel

import torch.nn as nn

from src.dpr.models.utils import pooling

class Embedder(nn.Module):
    """
    Embedder class to compute embeddings from a pre-trained model.

    Attributes:
        model_name (str): Pre-trained model name.
        config (dict): Configuration dictionary.
        model (AutoModel): Pre-trained model.
    
    Methods:
        forward(inputs) -> torch.Tensor: Forward pass of the model to compute embeddings from the input data.    

    """


    def __init__(self, config):
        """
        Initializes the Embedder with the given configuration.
        Args:
            config (dict): Configuration dictionary.
        """
        super(Embedder, self).__init__()
        self.model_name = config['model']
        self.config = config
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()

    def forward(self, inputs):
        """
        Forward pass of the model to compute embeddings from the input data.
        Args:
            inputs (dict): Dictionary containing input data.
        Returns:
            torch.Tensor: Embeddings computed by the model.
        """
        outputs = self.model(**inputs, return_dict=True)
        if self.config['use_CLS']:
            outputs = outputs['last_hidden_state'][:, 0, :]
        else:
            outputs = pooling(outputs['last_hidden_state'], inputs['attention_mask'])
        return outputs