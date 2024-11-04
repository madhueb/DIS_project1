import re
import spacy
import spacy.cli
from tqdm import tqdm
from typing import List


class EnglishTokenizer:
    MODEL_NAME = "en_core_web_sm"

    def __init__(self):
        spacy.cli.download(self.MODEL_NAME)
        self.nlp = spacy.load(self.MODEL_NAME, exclude=["senter", "ner"])
        self.stop_words = set(self.nlp.Defaults.stop_words)

    @staticmethod
    def preprocess_text(text: str) -> str:
        text = re.sub(r"http[s]?://\S+|www\.\S+", " ", text)
        text = re.sub(r"[^\w\s]{4,}", " ", text)
        return re.sub(r"\s+", " ", text.replace("\n", " ")).strip().lower()

    def tokenize_batch(self, texts: List[str], batch_size: int, n_process: int):
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        docs = self.nlp.pipe(
            preprocessed_texts, batch_size=batch_size, n_process=n_process
        )
        tokenized_texts = [
            [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
            for doc in tqdm(docs)
        ]

        return tokenized_texts
