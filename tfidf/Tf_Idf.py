import torch

class Tf_Idf_Vectorizer:
    def __init__(self,min_df =1, max_df =1.0, device='cuda'):
        self.idf = None
        self.vocab = None
        self.device = device
        self.min_df = min_df
        self.max_df = max_df

    def fit(self, documents):
        # Construct the vocabulary
        # vocab = set(word for doc in documents for word in doc.split())
        vocab = set(word for doc in documents for word in doc)
        self.vocab = {word: idx for idx, word in enumerate(vocab)}
        num_docs = len(documents)
        vocab_size = len(self.vocab)


        # Compute document frequency (DF) for each word in the vocabulary :
        df = torch.zeros(vocab_size, device=self.device)
        for i, doc in enumerate(documents):
            # for word in doc.split():
            for word in doc:
                if word in self.vocab:
                    idx = self.vocab[word]
                    df[idx] += 1

        # Filter the vocabulary based on max_df and min_df
        filtered_vocab = { word: idx for word, idx in self.vocab.items() if (df[idx] >= self.min_df and df[idx] <= self.max_df * num_docs)}

        self.vocab = {word: idx for idx, word in enumerate(filtered_vocab)}

        df = df[list(filtered_vocab.values())]

        # Compute IDF 
        self.idf = torch.log((num_docs + 1) / (df + 1)) + 1
        return self
    

    def transform(self, documents):
        num_docs = len(documents)
        vocab_size = len(self.vocab)
        tf = torch.zeros((num_docs, vocab_size), device=self.device)

        # Compute term frequency (TF) for each document
        for i, doc in enumerate(documents):
            # for word in doc.split():
            for word in doc:
                if word in self.vocab:
                    idx = self.vocab[word]
                    tf[i, idx] += 1

        # Compute the TF-IDF matrix
        tf_idf = tf * self.idf
        return tf_idf

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)
    

