import torch
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz, load_npz
from scipy.sparse.linalg import norm
from tqdm import tqdm
import pickle
import numpy as np



class Tf_Idf_Vectorizer:
    def __init__(self,min_df =1, max_df =1.0, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.idf = None
        self.vocab = None
        self.device = device
        self.min_df = min_df
        self.max_df = max_df
        self.tfidf_matrix = None

    def fit(self, documents):
        # Construct the vocabulary
        # vocab = set(word for doc in documents for word in doc.split())
        vocab = set(word for doc in documents for word in doc)
        self.vocab = {word: idx for idx, word in enumerate(vocab)}
        num_docs = len(documents)
        vocab_size = len(self.vocab)


        # Compute document frequency (DF) for each word in the vocabulary :
        df = torch.zeros(vocab_size, device=self.device)
        for i, doc in tqdm(enumerate(documents)):
            unique_words = set(doc)
            for word in unique_words:
                if word in self.vocab:
                    idx = self.vocab[word]
                    df[idx] += 1

        # Filter the vocabulary based on max_df and min_df
        # filtered_vocab = {word: idx for word, idx in tqdm(self.vocab.items()) if (df[idx] >= self.min_df and df[idx] <= self.max_df * num_docs)}

        # self.vocab = {word: idx for idx, word in tqdm(enumerate(filtered_vocab))}

        # df = df[list(filtered_vocab.values())]

        # Compute IDF 
        self.idf = torch.log((num_docs +1 ) / df +1)+1
        return self
    

    def transform(self, documents):
        num_docs = len(documents)
        vocab_size = len(self.vocab)
        tf = torch.zeros((num_docs, vocab_size), device=self.device)

        # Compute term frequency (TF) for each document
        for i, doc in tqdm(enumerate(documents)):
            # for word in doc.split():
            for word in doc:
                if word in self.vocab:
                    idx = self.vocab[word]
                    tf[i, idx] += 1

        tf = tf 
        # Compute the TF-IDF matrix
        tf_sparse = csr_matrix(tf)  
        idf_sparse = csr_matrix(self.idf)
        #tf_idf = tf_sparse@idf_sparse
        tf_idf = tf_sparse.multiply(idf_sparse)
        return tf_idf

    def fit_transform(self, documents):
        self.fit(documents)
        self.tfidf_matrix = self.transform(documents)

    

    @torch.no_grad()
    def batch (self ,tfidf, tf_q, batch_size, k):
        # Compute the cosine similarity between the query and the documents
        top_k_sims = {}
        num_docs = tfidf.shape[0]
        tf_q = torch.tensor(tf_q.toarray(), device=self.device)
        for i in tqdm(range(0, num_docs, batch_size)):
            batch = tfidf[i:i + batch_size]
            #Compute cosine similarity between the query and the batch
            batch =torch.tensor(batch.toarray(), device=self.device)
            sims = torch.mm(batch, tf_q.T)/(torch.norm(batch, dim=1)[:, None] * torch.norm(tf_q))

            top_k_index = i+ sims.argsort(axis=0)[-k:]
            top_k_sims.update({idx: sims[idx-i] for idx in top_k_index})
        
        top_k = list(dict(sorted(top_k_sims.items(), key=lambda x: x[1], reverse=True )).keys())
        return top_k
