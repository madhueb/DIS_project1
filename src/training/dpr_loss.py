import torch
import torch.nn as nn


class DPRLoss(nn.Module):
    # negative log likelihood loss

    def __init__(self, similarity_model):
        super(DPRLoss, self).__init__()
        self.similarity_model = similarity_model


    def forward(self, inputs):

        positive_ = inputs['positive']
        negative_ = inputs['negative']
        query_positive = inputs['query_positive']
        query_negative = inputs['query_negative']
        query_positive_masks = inputs['query_positive_mask']
        query_negative_masks = inputs['query_negative_mask']

        positive_similarity = self.similarity_model(query_positive, positive_)
        positive_similarity = torch.sum(torch.exp(positive_similarity) * query_positive_masks, dim=1)

        negative_similarity = self.similarity_model(query_negative, negative_)
        negative_similarity = torch.sum(torch.exp(negative_similarity) * query_negative_masks, dim=1)

        loss = -torch.log(positive_similarity / (positive_similarity + negative_similarity)).mean()
        return loss
