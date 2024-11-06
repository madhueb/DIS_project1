import torch
from tqdm import tqdm
from scipy.sparse import csr_matrix

from tqdm import tqdm
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import norm

import numpy as np


class Tf_Idf_Vectorizer:
    def __init__(self, min_df=1, max_df=1.0, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.idf = None
        self.vocab = None
        self.device = device
        self.min_df = min_df
        self.max_df = max_df
        self.tfidf_matrix = None

    @torch.no_grad()
    def fit(self, documents):
        # Construct the vocabulary

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

        # Compute IDF

        # self.idf = np.log((num_docs +1 ) / (df +1))+1
        # self.idf = np.log(num_docs / df) + 1
        self.idf = np.log(num_docs / df)
        return self

    @torch.no_grad()
    def transform(self, documents, is_query=False):
        num_docs = len(documents)
        vocab_size = len(self.vocab)

        data = []
        row_indices = []
        col_indices = []
        norms = []

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
            # if len(word_count) == 0:
            #     print(f"Document {i} is empty")
            #     continue
            # max_val = max(word_count.values())
            row_norm = 0
            for idx, count in word_count.items():
                row_indices.append(i)
                col_indices.append(idx)
                # data.append(count / max_val)
                data.append(count)
                row_norm += count ** 2
            norms.append(np.sqrt(row_norm))
        if not is_query:
            norm_median = np.median(norms)
            # print(len(data), len(row_indices), len(norms))
            for j in range(len(data)):
                # print(data[j], row_indices[j], col_indices[j])
                data[j] /= 0.5 * norm_median + 0.5 * norms[row_indices[j]]
        else:
            for j in range(len(data)):
                data[j] /= norms[row_indices[j]]

        tf = csr_matrix((data, (row_indices, col_indices)), shape=(num_docs, vocab_size))
        idf_sparse = csr_matrix(self.idf)
        tf_idf = tf.multiply(idf_sparse)
        # # if not is_query:
        # row_norms = np.array([norm(tf_idf.getrow(i)) for i in range(tf_idf.shape[0])])
        # dominator = 0.5 * np.median(row_norms) + 0.5 * row_norms
        # row_factors = np.repeat(dominator, np.diff(tf_idf.indptr))
        # tf_idf.data /= row_factors
        # # else:
        tf_idf = normalize(tf_idf, norm='l2', axis=1)
        return tf_idf

    def fit_transform(self, documents):
        self.fit(documents)
        self.tfidf_matrix = self.transform(documents)

