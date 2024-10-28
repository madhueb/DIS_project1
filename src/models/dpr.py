

import torch
import torch.nn as nn


class DPRModel(nn.Module):

    def __init__(self):
        super(DPRModel, self).__init__()

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

        # We concat the embeddings of queries and docs.
        # Then we concat them with the cosine similarity and pass them through linear layers.
        self.fc1 = nn.Linear(self.query_encoder.config.hidden_size + self.doc_encoder.config.hidden_size + 1, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, query_embeds, doc_embeds):
        query_outputs = self.query_linear_seq(query_embeds) + query_embeds
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
