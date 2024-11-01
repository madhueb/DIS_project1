import torch.nn as nn


class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size=256):
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
        outputs = self.seq1(inputs) + inputs
        outputs = outputs / outputs.norm(dim=-1, keepdim=True)
        # outputs = self.seq2(outputs) + outputs
        outputs = self.seq2(outputs)
        outputs = outputs / outputs.norm(dim=-1, keepdim=True)
        return outputs


