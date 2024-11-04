import pyphen
import re
import spacy
import spacy.cli
from tqdm import tqdm
from typing import List


class GermanTokenizer:
    MODEL_NAME = "de_core_news_md"

    def __init__(self):
        spacy.cli.download(self.MODEL_NAME)
        self.nlp = spacy.load(self.MODEL_NAME, exclude=["senter", "ner"])
        self.stop_words = set(self.nlp.Defaults.stop_words)
        self.pyphen_dic = pyphen.Pyphen(lang="de_DE")

    @staticmethod
    def preprocess_text(text: str) -> str:
        text = re.sub(r"http[s]?://\S+|www\.\S+", " ", text)
        text = re.sub(r"[^\w\s]{4,}", " ", text)
        return re.sub(r"\s+", " ", text.replace("\n", " ")).strip().lower()

    def split_compound_word(self, word: str) -> List[str]:
        split_word = self.pyphen_dic.inserted(word)
        return split_word.split("-")

    def tokenize_batch(self, texts: List[str], batch_size: int, n_process: int):
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        docs = self.nlp.pipe(
            preprocessed_texts, batch_size=batch_size, n_process=n_process
        )
        tokenized_texts = []
        for doc in tqdm(docs):
            tokens = []
            for token in doc:
                if not token.is_stop and not token.is_punct:
                    if len(token) > 5:
                        split_tokens = self.split_compound_word(token.lemma_)
                        tokens.extend(split_tokens)
                    else:
                        tokens.append(token.lemma_)
            tokenized_texts.append(tokens)

        return tokenized_texts
