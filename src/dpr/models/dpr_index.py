import pickle

import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModel

import torch.nn as nn

from src.dpr.models.encoder import Encoder
from src.dpr.models.utils import pooling


class DPRIndexModule(nn.Module):
    """
    DPRIndexModule class for creating a dense passage retrieval (DPR) index module.

    Attributes:
        config (dict): Configuration dictionary with keys such as 'model', 'device',
                    'doc_encodes_path', 'load_path', and others specifying 
                    hyperparameters and paths.
        k_chunk (int): Number of chunks per document for retrieval, defaults to 100.
        k_doc (int): Number of documents to retrieve, defaults to 10.
        q_tokenizer (AutoTokenizer): Tokenizer for queries using the specified 
                                    pre-trained model.
        q_embedder (AutoModel): Pre-trained transformer model for embedding queries.
        q_encoder (Encoder): Custom encoder model for processing query embeddings.
        doc_encodes (dict): Dictionary of document encodings loaded from a file.
        doc_ids (dict): Dictionary mapping languages to document IDs.
        all_embeds (dict): Dictionary containing document embeddings for each language.

    Methods:
        __init__(config, load_index=False): Initializes the DPRIndexModule, loads 
                                            query encoder and document encodings.
        forward(query, langs): Computes the top-k documents for the given query
    """




    def __init__(self, config, load_index=False):
        """
        Initializes the DPRIndexModule with the given configuration and loads the query encoder.
        Args:
            config (dict): Configuration dictionary with keys such as 'model', 'device',
                        'doc_encodes_path', 'load_path', and others specifying 
                        hyperparameters and paths.
            load_index (bool): Flag to indicate whether to load the index, defaults to False.
        """
        super(DPRIndexModule, self).__init__()
        self.config = config
        self.k_chunk = config.get('k_chunk', 100)
        self.k_doc = config.get('k_doc', 10)

        # Get Query Encoder and Tokenizer
        self.q_tokenizer = AutoTokenizer.from_pretrained(
                config["model"],
                use_fast=config.get("use_fast", False),
            )
        self.q_embedder = AutoModel.from_pretrained(config["model"]).to(config['device'])
        embed_size = self.q_embedder.config.hidden_size
        self.q_encoder = Encoder(embed_size).to(config['device'])
        self.q_encoder.load_state_dict(torch.load(f"{config['load_path']}/query_encoder.pth"))
        # put models on eval mode
        self.q_embedder.eval()
        self.q_encoder.eval()
        # Get documents encodes
        with open(config['doc_encodes_path'], 'rb') as f:
        # with open(config['doc_embeds_path'], 'rb') as f:
            self.doc_encodes = pickle.load(f)
            print("Loaded doc encodes")

        langs = ['en', 'fr', 'de', 'es', 'it', 'ko', 'ar']
        self.doc_ids = {lang: [] for lang in langs}
        for doc_id, encodes_dict in tqdm(self.doc_encodes.items()):
            # self.doc_ids[encodes_dict['lang']].extend([doc_id] * len(encodes_dict['encodes']))
            self.doc_ids[encodes_dict['lang']].append(doc_id)
        for lang in langs:
            self.doc_ids[lang] = np.array(self.doc_ids[lang])

        self.all_embeds = {}

        encode_lang = {lang: [] for lang in langs}
        for doc_id, encodes_dict in tqdm(self.doc_encodes.items()):
            encode_lang[encodes_dict['lang']].append(
                # np.array(encodes_dict['embeds']) / np.linalg.norm(encodes_dict['embeds']))
                np.array(encodes_dict['encode']) / np.linalg.norm(encodes_dict['encode']))

        print("Adding vectors to index")
        for lang in tqdm(langs):
            self.all_embeds[lang] = torch.tensor(encode_lang[lang]).to(config['device'])

        # if not load_index:
        #     self.index = {}
        #     N = config.get('index_N', 128)
        #     encode_lang = {lang: [] for lang in langs}
        #     print("Creating index")
        #     # res = faiss.StandardGpuResources()
        #     # print("Created resources")
        #     for lang in tqdm(langs):
        #         self.index[lang] = faiss.IndexHNSWFlat(embed_size, N, faiss.METRIC_INNER_PRODUCT)
        #         # if config['device'] != 'cpu':
        #         #     self.index[lang] = faiss.index_cpu_to_gpu(res, 0, self.index[lang])
        #     print("Created index")
        #     # Add documents to index
        #     print("Adding documents to index")
        #     for doc_id, encodes_dict in tqdm(self.doc_encodes.items()):
        #         # self.doc_ids[encodes_dict['lang']].extend([doc_id] * len(encodes_dict['encode']))
        #         # encode_lang[encodes_dict['lang']].extend(encodes_dict['encode'])
        #         encode_lang[encodes_dict['lang']].append(np.array(encodes_dict['embeds']) / np.linalg.norm(encodes_dict['embeds']))
        #         # self.index[encodes_dict['lang']].add(np.array(encodes_dict['encode'], dtype=np.float32))
        #
        #     print("Adding vectors to index")
        #     for lang in tqdm(langs):
        #         self.index[lang].add(np.array(encode_lang[lang], dtype=np.float32))
        #         print(f"Total vectors in {lang} index:", self.index[lang].ntotal)
        #         assert len(self.doc_ids[lang]) == self.index[lang].ntotal
        #
        #     print("Saving index")
        #     # Save index
        #     os.makedirs(config["index_path"], exist_ok=True)
        #
        #     for lang, index in self.index.items():
        #         faiss.write_index(index, f'{config["index_path"]}/{lang}.index')
        # else:
        #     self.index = {}
        #     print("Loading index")
        #     for lang in tqdm(langs):
        #         self.index[lang] = faiss.read_index(f'{config["index_path"]}/{lang}.index')


    def forward(self, query, langs):
        """
        Computes the top-k documents for the given query.
        Args:
            query (list): List of queries.
            langs (list): List of language labels.
        Returns:
            list: List of top-k documents for each query.
        """ 
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
            if self.config['use_CLS']:
                outputs = outputs['last_hidden_state'][:, 0, :]
            else:
                outputs = pooling(outputs['last_hidden_state'], inputs['attention_mask'])
            outputs = self.q_encoder(outputs)
            q_encodes = outputs / torch.norm(outputs, dim=1).unsqueeze(1)

            for i, q_encode in enumerate(q_encodes):
                lang = langs[i]

                # sim = torch.nn.CosineSimilarity(dim=1)(q_encode.unsqueeze(0).expand_as(self.all_embeds[lang]), self.all_embeds[lang])

                # inner product instead of cosine similarity
                sim = (q_encode.unsqueeze(0).expand_as(self.all_embeds[lang]) * self.all_embeds[lang]).sum(dim=1)
                top_k_docs = dict(sorted(enumerate(sim.cpu().numpy()), key=lambda x: x[1], reverse=True)[:self.k_doc]).keys()
                top_k_.append(list(self.doc_ids[lang][list(top_k_docs)]))


                # # Search index
                # # _, inds = self.index[lang].search(q_encodes[i].detach().cpu().numpy().reshape(1, -1), self.k_chunk)
                # _, inds = self.index[lang].search(q_encodes[i].detach().cpu().numpy().reshape(1, -1), self.k_doc)
                #
                # # flatten the list
                # inds = inds.flatten()
                #
                # # Get top k chunks
                # doc_ids = self.doc_ids[lang][inds]
                # top_k_.append(list(doc_ids))

                # # make the list unique
                # doc_ids = list(set(doc_ids))
                #
                # # Get top k docs
                # doc_dict_scores = {}
                # chunk_tensor = []
                # for doc_id in doc_ids:
                #     # chunk_tensor.extend(self.doc_encodes[doc_id]['encode'])
                #     chunk_tensor.extend(self.doc_encodes[doc_id]['embeds'])
                # chunk_tensor = torch.tensor(chunk_tensor).to(self.config['device'])
                # # chunk_scores = torch.exp(torch.nn.CosineSimilarity(dim=1)(q_encode.unsqueeze(0).expand_as(chunk_tensor), chunk_tensor))
                # chunk_scores = torch.nn.CosineSimilarity(dim=1)(q_encode.unsqueeze(0).expand_as(chunk_tensor), chunk_tensor)
                # tmp_index = 0
                # for doc_id in doc_ids:
                #     doc_dict_scores[doc_id] = chunk_scores[tmp_index:tmp_index + len(self.doc_encodes[doc_id]['embeds'])].mean().item()
                #     # doc_dict_scores[doc_id] = chunk_scores[tmp_index:tmp_index + len(self.doc_encodes[doc_id]['encode'])].sum().item()
                #     # doc_dict_scores[doc_id] = chunk_scores[tmp_index:tmp_index + len(self.doc_encodes[doc_id]['encode'])].max().item()
                #     # doc_dict_scores[doc_id] = chunk_scores[tmp_index:tmp_index + len(self.doc_encodes[doc_id]['embeds'])].max().item()
                #     tmp_index += len(self.doc_encodes[doc_id]['embeds'])
                #
                # # Sort and get the list of top k docs ids
                # top_k_docs = dict(sorted(doc_dict_scores.items(), key=lambda x: x[1], reverse=True)[:self.k_doc]).keys()
                # top_k_.append(list(top_k_docs))

        return top_k_
