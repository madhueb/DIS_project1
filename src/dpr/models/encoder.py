import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder class to encode input data.

    Attributes:
        seq1 (nn.Sequential): Sequential model for the first layer.
        seq2 (nn.Sequential): Sequential model for the second layer.

    Methods:
        forward(inputs) -> torch.Tensor: Forward pass of the model to encode input data.
    """

    def __init__(self, input_size, hidden_size=256):
        """
        Initializes the Encoder with the given input and hidden sizes.
        Args:
            input_size (int): Size of the input data.
            hidden_size (int): Size of the hidden layer.
        """
        super(Encoder, self).__init__()
        self.seq1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )
        self.seq2 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, inputs):
        """
        Forward pass of the model to encode input data.
        Args:
            inputs (torch.Tensor): Input data to encode.
        Returns:
            torch.Tensor: Encoded output data.
        """
        outputs = self.seq1(inputs) + inputs
        outputs = outputs / outputs.norm(dim=-1, keepdim=True)
        # outputs = self.seq2(outputs) + outputs
        outputs = self.seq2(outputs)
        outputs = outputs / outputs.norm(dim=-1, keepdim=True)
        return outputs


