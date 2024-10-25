import torch
import torch.nn as nn


class DPRLoss(nn.Module):
    # negative log likelihood loss

    def __init__(self, similarity_model):
        super(DPRLoss, self).__init__()
        self.similarity_model = similarity_model


    def get_similarity(self, queries, docs, query_masks):
        query_doc_masks = []
        query_docs = []
        docs_ = []

        for i in range(len(queries)):
            query = queries[i]
            doc = docs[i]
            query_docs.append(torch.cat([query] * len(doc), dim=0))
            docs_.append(doc)

        query_docs = torch.cat(query_docs, dim=0)
        docs_ = torch.cat(docs_, dim=0)

        tmp = 0
        for i in range(query_docs.shape[0]):
            doc_mask = torch.zeros(query_docs.shape[0])
            doc_mask[tmp:tmp + len(docs[i])] = 1
            tmp += len(docs[i])
            query_doc_masks.append(doc_mask)

        query_doc_masks = torch.stack(query_doc_masks)

        # compute similarity
        similarity = self.similarity_model(query_docs, docs_, query_masks)
        similarity_exp = torch.exp(similarity * query_doc_masks)

        return similarity_exp


    def forward(self, inputs):
        # Similarity model just compute similarity between one query and one doc.
        # We need to compute similarity between the query and all positive and negative docs.
        # So we populate the query and the positive docs to have the same number of elements as the negative docs.

        queries = inputs['query']
        queries_mask = inputs['query_mask']
        positives_ = inputs['positive']
        negatives_ = inputs['negative']

        positive_similarity = self.get_similarity(queries, positives_, queries_mask)
        negative_similarity = self.get_similarity(queries, negatives_, queries_mask)

        loss = -torch.log(positive_similarity / (positive_similarity + negative_similarity)).mean()
        return loss
