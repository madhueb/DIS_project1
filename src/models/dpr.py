
import torch
import torch.nn as nn

from src.models.encoder import Encoder


class DPRModel(nn.Module):

    def __init__(self, embed_size=768):
        super(DPRModel, self).__init__()

        self.query_encoder = Encoder(embed_size)
        self.doc_encoder = Encoder(embed_size)

    def forward(self, query_embeds, doc_embeds):
        query_outputs = self.query_encoder(query_embeds)
        doc_outputs = self.doc_encoder(doc_embeds)

        # cosine similarity
        cosine_similarity = nn.CosineSimilarity(dim=1)(query_outputs, doc_outputs)

        return cosine_similarity

    def save(self, path):
        torch.save(self.query_encoder.state_dict(), f'{path}/query_encoder.pth')
        torch.save(self.doc_encoder.state_dict(), f'{path}/doc_encoder.pth')

    def load(self, query_encoder_path, doc_encoder_path):
        self.query_encoder.load_state_dict(torch.load(query_encoder_path))
        self.doc_encoder.load_state_dict(torch.load(doc_encoder_path))