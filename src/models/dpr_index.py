
import pickle

import numpy as np
import torch

from transformers import AutoTokenizer, AutoModel

import torch.nn as nn
import faiss

from src.models.encoder import Encoder
from src.models.utils import pooling


class DPRIndexModule(nn.Module):

    def __init__(self, config):
        super(DPRIndexModule, self).__init__()
        self.config = config
        self.k_chunk = config.get('k_chunk', 100)
        self.k_doc = config.get('k_doc', 10)

        # Get Query Encoder and Tokenizer
        self.q_tokenizer = AutoTokenizer.from_pretrained(
                config["model"],
                use_fast=config.get("use_fast", False),
            )
        self.q_embedder = AutoModel.from_pretrained(config["model"])
        embed_size = self.q_embedder.config.hidden_size
        self.q_encoder = Encoder(embed_size).to(config['device'])
        self.q_encoder.load_state_dict(torch.load(f"{config['load_path']}/query_encoder.pth"))
        # put models on eval mode
        self.q_embedder.eval()
        self.q_encoder.eval()

        # Get documents encodes
        with open(config['doc_encodes_path'], 'rb') as f:
            self.doc_encodes = pickle.load(f)

        langs = ['en', 'fr', 'de', 'es', 'it', 'ko', 'ar']
        self.index = {}
        N = config.get('index_N', 128)
        self.doc_ids = {lang: [] for lang in langs}
        for lang in langs:
            self.index[lang] = faiss.IndexHNSWFlat(embed_size, N, faiss.METRIC_INNER_PRODUCT)
            if config['device'] != 'cpu':
                res = faiss.StandardGpuResources()
                self.index[lang] = faiss.index_cpu_to_gpu(res, 0, self.index[lang])

        # Add documents to index
        for doc_id, encodes_dict in self.doc_encodes.items():
            self.doc_ids[encodes_dict['lang']].extend([doc_id] * len(encodes_dict['encodes']))
            self.index[encodes_dict['lang']].add(encodes_dict['encodes'])

        for lang in langs:
            print(f"Total vectors in {lang} index:", self.index[lang].ntotal)
            assert len(self.doc_ids[lang]) == self.index[lang].ntotal


    def forward(self, query, langs):
        # Embed query
        top_k_ = []
        with torch.no_grad():
            inputs = self.q_tokenizer(query,
                                       return_tensors='pt',
                                       padding=True,
                                       truncation=True,
                                       add_special_tokens=True,
                                       max_length=self.config['max_length']
                                       )
            inputs = {k: v.to(self.config['device']) for k, v in inputs.items()}
            outputs = self.q_embedder(**inputs, return_dict=True)
            outputs = pooling(outputs['last_hidden_state'], inputs['attention_mask'])
            q_encodes = self.q_encoder(outputs)

            for i, q_encode in enumerate(q_encodes):
                lang = langs[i]
                # Search index
                _, inds = self.index[lang].search(q_encodes[i].detach().cpu().numpy(), self.k_chunk)
                # Get top k chunks
                doc_ids = self.doc_ids[lang][inds]
                # make the list unique
                doc_ids = list(set(doc_ids))

                # Get top k docs
                doc_dict_scores = {}
                chunk_tensor = []
                for doc_id in doc_ids:
                    doc_dict_scores[doc_id] = 0
                    chunk_tensor.extend(self.doc_encodes[doc_id]['encodes'])
                chunk_tensor = torch.tensor(chunk_tensor).to(self.config['device'])
                chunk_scores = torch.exp(torch.nn.CosineSimilarity(dim=1)(q_encode.unsqueeze(0).expand_as(chunk_tensor), chunk_tensor))
                tmp_index = 0
                for doc_id in doc_ids:
                    doc_dict_scores[doc_id] += chunk_scores[tmp_index:tmp_index + len(self.doc_encodes[doc_id]['encodes'])].sum().item()
                    tmp_index += len(self.doc_encodes[doc_id]['encodes'])

                # Sort and get the list of top k docs ids
                top_k_docs = dict(sorted(doc_dict_scores.items(), key=lambda x: x[1], reverse=True)[:self.k_doc]).keys()
                top_k_.append(list(top_k_docs))

        return top_k_