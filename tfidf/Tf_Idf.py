import torch
from tqdm import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz, load_npz
from scipy.sparse.linalg import norm
from tqdm import tqdm
from sklearn.preprocessing import normalize
import pickle
import numpy as np
from transformers.utils.hub import torch_cache_home


class Tf_Idf_Vectorizer:
    def __init__(self,min_df =1, max_df =1.0, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.idf = None
        self.vocab = None
        self.device = device
        self.min_df = min_df
        self.max_df = max_df
        self.tfidf_matrix = None

    @torch.no_grad()
    def fit(self, documents):
        # Construct the vocabulary
        # vocab = set(word for doc in documents for word in doc.split())
        vocab = set(word for doc in documents for word in doc)
        self.vocab = {word: idx for idx, word in enumerate(vocab)}
        num_docs = len(documents)
        vocab_size = len(self.vocab)


        # Compute document frequency (DF) for each word in the vocabulary :
        df = np.zeros(vocab_size)
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

        # self.idf = np.log((num_docs +1 ) / (df +1))+1
        # self.idf = np.log(num_docs / df) + 1
        self.idf = np.log(num_docs / df)
        return self
    
    @torch.no_grad()
    def transform(self, documents, is_query=False):
        num_docs = len(documents)
        vocab_size = len(self.vocab)
        # tf = torch.zeros((num_docs, vocab_size), device=self.device)
        #
        # # Compute term frequency (TF) for each document
        # for i, doc in tqdm(enumerate(documents)):
        #     # for word in doc.split():
        #     for word in doc:
        #         if word in self.vocab:
        #             idx = self.vocab[word]
        #             tf[i, idx] += 1

        data = []
        row_indices = []
        col_indices = []

        # Compute term frequency (TF) for each document
        for i, doc in tqdm(enumerate(documents)):
            word_count = {}
            for word in doc:
                if word in self.vocab:
                    idx = self.vocab[word]
                    if idx not in word_count:
                        word_count[idx] = 0
                    word_count[idx] += 1

            # Collect data for the CSR representation
            if len(word_count) == 0:
                print(f"Document {i} is empty")
                continue
            max_val = max(word_count.values())
            for idx, count in word_count.items():
                row_indices.append(i)
                col_indices.append(idx)
                data.append(count / max_val)
                # data.append(count)

        tf = csr_matrix((data, (row_indices, col_indices)), shape=(num_docs, vocab_size))
        idf_sparse = csr_matrix(self.idf)
        tf_idf = tf.multiply(idf_sparse)
        tf_idf = normalize(tf_idf, norm='l2', axis=1)
        return tf_idf


        # if is_sparse:
        #     tf = tf.cpu()
        #     # Compute the TF-IDF matrix
        #     tf_sparse = csr_matrix(tf)
        #     idf_sparse = csr_matrix(self.idf)
        #     #tf_idf = tf_sparse@idf_sparse
        #     tf_idf = tf_sparse.multiply(idf_sparse)
        #     # normalize the tfidf matrix
        #     tf_idf = normalize(tf_idf, norm='l2', axis=1)
        #     return tf_idf
        # else:
        #     # Compute the TF-IDF matrix
        #     tf_idf = tf * self.idf
        #     # normalize the tfidf matrix
        #     tf_idf = tf_idf / torch.norm(tf_idf, dim=1).unsqueeze(1)
        #     return tf_idf

    def fit_transform(self, documents):
        self.fit(documents)
        self.tfidf_matrix = self.transform(documents)

    @torch.no_grad()
    def batch (self, tf_q, batch_size, k):
        # # Compute the cosine similarity between the query and the documents
        # top_k_sims = []
        # num_docs = tfidf_matrix.shape[0]
        # # tf_q = torch.tensor(tf_q, device=self.device)
        # for i in tqdm(range(0, num_docs, batch_size)):
        #     batch = tfidf_matrix[i:i + batch_size]
        #     #Compute cosine similarity between the query and the batch
        #     batch = torch.tensor(batch.toarray(), device=self.device)
        #     sims = torch.mm(batch, tf_q.T)
        #
        #     # top_k_index = i + sims.argsort(axis=0)[-k:]
        #     top_k_index = torch.topk(sims.T, k, largest=True).indices.flatten().cpu().numpy()
        #     # top_k_sims.update({idx: sims[idx-i] for idx in top_k_index})
        #     top_k_sims.extend([(i + idx, sims[idx].item()) for idx in top_k_index])
        #
        # top_k = [idx for idx, _ in sorted(top_k_sims, key=lambda x: x[1], reverse=True)][:k]
        # return top_k

        # sims = torch.mm(self.tfidf_matrix, tf_q.T)
        sims = torch.sparse.mm(self.tfidf_matrix, tf_q)
        top_k_index = torch.topk(sims.T, k, largest=True, dim=1).indices.cpu().numpy()
        return top_k_index

