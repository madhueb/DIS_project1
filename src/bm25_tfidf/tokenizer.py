import string
from typing import List

import nltk
import spacy
import re

import spacy.cli
from tqdm import tqdm

from camel_tools.utils.dediac import dediac_ar
from camel_tools.tokenizers.word import simple_word_tokenize

from camel_tools.disambig.mle import MLEDisambiguator

import importlib.resources


class BaseTokenizer:

    def __init__(self, model_name: str):
        spacy.cli.download(model_name)
        self.nlp = spacy.load(model_name, exclude=["senter", "ner"])
        self.stop_words = set(self.nlp.Defaults.stop_words)

    @staticmethod
    def preprocess_text(text: str) -> str:
        # Step 1: Remove URLs
        text = re.sub(r"http[s]?://\S+|www\.\S+", " ", text)

        # Step 2: Remove long sequences of non-alphanumeric characters (e.g., encoded data or code)
        text = re.sub(r"[^\w\s]{4,}", " ", text)  # Removes any sequence of 4 or more non-alphanumeric characters

        # Step 3: Remove excessive whitespace
        return re.sub(r"\s+", " ", text.replace("\n", " ")).strip().lower()

    def tokenize(
        self, texts: List[str], batch_size: int = 32, n_process: int = 4
    ):
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        docs = self.nlp.pipe(
            preprocessed_texts, batch_size=batch_size, n_process=n_process
        )
        tokenized_texts = [
            [
                token.lemma_
                for token in doc
                if not token.is_stop
                and not token.is_punct
            ]
            for doc in tqdm(docs)
        ]

        return tokenized_texts


class EnglishTokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__("en_core_web_sm")


class FrenchTokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__("fr_core_news_sm")


class ItalianTokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__("it_core_news_sm")


class GermanTokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__("de_core_news_sm")


class SpanishTokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__("es_core_news_sm")


class KoreanTokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__("ko_core_news_sm")


class ArabicTokenizer:

    def __init__(self):
        punctuations = '''`÷×؛<>«»_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ''' + string.punctuation
        self.translator = str.maketrans('', '', punctuations)
        self.mle = MLEDisambiguator.pretrained()
        nltk.download('stopwords')
        with importlib.resources.open_text("src.bm25_tfidf", "ar_stopwords.txt") as file:
            self.stop_words = file.read().split('\n') + list(nltk.corpus.stopwords.words("arabic"))



    def preprocess_text(self, text: str) -> List[str]:
        # Step 1: Remove URLs
        text = re.sub(r"http[s]?://\S+|www\.\S+", " ", text)

        # Step 2: Remove long sequences of non-alphanumeric characters (e.g., encoded data or code)
        text = re.sub(r"[^\w\s]{4,}", " ", text)  # Removes any sequence of 4 or more non-alphanumeric characters

        # Step 3: Remove punctuation
        text = text.translate(self.translator)

        # Step 4: Remove excessive whitespace
        text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip().lower()

        # Step 5: Remove diacritics
        text = dediac_ar(text)

        return text

    def tokenize(
            self, texts: List[str]
    ) -> List[List[str]]:
        queries = [self.preprocess_text(query) for query in texts]
        all_tokens = []
        tokens_len = []
        for query in queries:
            tokens = simple_word_tokenize(query)
            all_tokens.extend(tokens)
            tokens_len.append(len(tokens))

        mle = MLEDisambiguator.pretrained()
        disambig = mle.disambiguate(all_tokens)

        tokens = []
        start = 0
        cnt = 0
        tmp = []
        for d in disambig:
            token = dediac_ar(d.analyses[0].analysis['lex']).translate(self.translator)
            if token not in self.stop_words:
                tmp.append(token)
            cnt += 1
            if cnt == tokens_len[start]:
                tokens.append(tmp)
                tmp = []
                start += 1
                cnt = 0

        return tokens

