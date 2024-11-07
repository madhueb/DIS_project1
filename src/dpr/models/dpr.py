
import torch
import torch.nn as nn

from src.dpr.models.encoder import Encoder


class DPRModel(nn.Module):
    """
    DPRModel class for training and evaluating Dense Passage Retrieval models.

    Attributes:
        query_encoder (Encoder): Encoder model for query data.
        doc_encoder (Encoder): Encoder model for document data.

    Methods:
        forward(query_embeds, doc_embeds): Defines the forward pass of the model.
        save(path): Saves the model weights to the specified path.
        load(query_encoder_path, doc_encoder_path): Loads the model weights from the specified paths.
    """

    def __init__(self, embed_size=768):
        """
        Initializes the DPRModel with query and document encoders.
        Args:
            embed_size (int): Size of the input embeddings.
        """
        super(DPRModel, self).__init__()

        self.query_encoder = Encoder(embed_size)
        self.doc_encoder = Encoder(embed_size)

    def forward(self, query_embeds, doc_embeds):
        """
        Defines the forward pass of the model.
        Args:
            query_embeds (torch.Tensor): Query embeddings.
            doc_embeds (torch.Tensor): Document embeddings.
        Returns:
            torch.Tensor: Cosine similarity scores between query and document embeddings.
        """
        query_outputs = self.query_encoder(query_embeds)
        doc_outputs = self.doc_encoder(doc_embeds)

        # cosine similarity
        cosine_similarity = nn.CosineSimilarity(dim=1)(query_outputs, doc_outputs)

        return cosine_similarity

    def save(self, path):
        """
        Saves the model weights to the specified path.
        Args:
            path (str): Path to save the model weights.
        """
        torch.save(self.query_encoder.state_dict(), f'{path}/query_encoder.pth')
        torch.save(self.doc_encoder.state_dict(), f'{path}/doc_encoder.pth')

    def load(self, query_encoder_path, doc_encoder_path):
        """
        Loads the model weights from the specified paths.
        Args:
            query_encoder_path (str): Path to the query encoder weights.
            doc_encoder_path (str): Path to the document encoder weights.
        """
        self.query_encoder.load_state_dict(torch.load(query_encoder_path))
        self.doc_encoder.load_state_dict(torch.load(doc_encoder_path))