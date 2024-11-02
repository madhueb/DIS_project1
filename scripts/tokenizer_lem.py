import argparse
import pickle
from pathlib import Path
import pandas as pd
import spacy
import re
from typing import List
from typing import List
import spacy
import re
import string
import os
import spacy.cli
from tqdm import tqdm

LANGS = ["en", "fr", "de", "it", "es", "ar", "ko"]


class BaseTokenizer:
    # TOKEN_PATTERN = re.compile(r"^[\w.-]+(?:'[\w]+)?$")

    def __init__(self, model_name: str):
        spacy.cli.download(model_name)
        self.nlp = spacy.load(model_name, exclude=["senter"])
        self.stop_words = set(self.nlp.Defaults.stop_words)

    @staticmethod
    def preprocess_text(text: str) -> str:
        # Step 1: Remove URLs
        text = re.sub(r"http[s]?://\S+|www\.\S+", " ", text)

        # Step 2: Remove long sequences of non-alphanumeric characters (e.g., encoded data or code)
        text = re.sub(r"[^\w\s]{4,}", " ", text)  # Removes any sequence of 4 or more non-alphanumeric characters

        # Step 3: Remove excessive whitespace
        return re.sub(r"\s+", " ", text.replace("\n", " ")).strip().lower()

    def tokenize_batch(
        self, texts: List[str], batch_size: int = 64, n_process: int = 8
    ):
        print("Tokenizing...")
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        print("Preprocessed...")
        docs = self.nlp.pipe(
            preprocessed_texts, batch_size=batch_size, n_process=n_process
        )
        print("Docs...")
        tokenized_texts = [
            [
                token.lemma_
                for token in doc
                if not token.is_stop
                and not token.is_punct
                # and self.TOKEN_PATTERN.match(token.text)
            ]
            for doc in tqdm(docs)
        ]
        tokenized_texts_ner = [
            [
                ent.text
                for ent in doc.ents
                if not ent.text.is_stop
                and not ent.text.is_punct
                # and self.TOKEN_PATTERN.match(token.text)
            ]
            for doc in tqdm(docs)
        ]
        return tokenized_texts, tokenized_texts_ner


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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--corpus_df", type=Path, required=True)
    parser.add_argument("-o", "--output_dir", type=Path, required=True)
    parser.add_argument("-l", "--language", type=str, required=True, choices=LANGS)
    parser.add_argument("-n", "--num_splits", type=int, required=True)
    parser.add_argument("-i", "--split_index", type=int, required=True)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-c", "--cores", type=int, default=10)

    args = parser.parse_args()

    print("corpus_df:", args.corpus_df)
    print("output_dir:", args.output_dir)
    print("language:", args.language)
    print("num_splits:", args.num_splits)
    print("split_index:", args.split_index)

    assert args.corpus_df.exists(), "corpus_df does not exist"
    assert args.corpus_df.is_file(), "corpus_df is not a file"

    assert args.output_dir.exists(), "output_dir does not exist"
    assert args.output_dir.is_dir(), "output_dir is not a directory"

    assert args.num_splits > 0, "num_splits must be positive"
    assert 1 <= args.split_index <= args.num_splits, "split_index is invalid"

    corpus_df = pd.read_json(args.corpus_df)
    corpus_df = corpus_df[corpus_df["lang"] == args.language]
    del corpus_df["lang"]
    all_records_count = len(corpus_df)
    records_per_split = all_records_count // args.num_splits
    start_index = (args.split_index - 1) * records_per_split
    if args.split_index == args.num_splits:
        end_index = all_records_count
    else:
        end_index = start_index + records_per_split
    corpus_df = corpus_df.iloc[start_index:end_index]

    if args.language == "en":
        tokenizer = EnglishTokenizer()
    elif args.language == "fr":
        tokenizer = FrenchTokenizer()
    elif args.language == "de":
        tokenizer = GermanTokenizer()
    elif args.language == "it":
        tokenizer = ItalianTokenizer()
    elif args.language == "es":
        tokenizer = SpanishTokenizer()
    elif args.language == "ar":
        pass
    elif args.language == "ko":
        tokenizer = KoreanTokenizer()
    else:
        raise KeyError("language")

    tokenized_texts, texts_ner = tokenizer.tokenize_batch(corpus_df["text"].tolist(), batch_size=args.batch_size, n_process=args.cores)

    output_file = (
        args.output_dir
        / f"tokens_{args.language}_{args.split_index}_{args.num_splits}.pkl"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    with open(output_file, "wb") as f:
        pickle.dump(tokenized_texts, f)

    output_file_ner = (
        args.output_dir
        / f"ner_{args.language}_{args.split_index}_{args.num_splits}.pkl"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    with open(output_file_ner, "wb") as f:
        pickle.dump(texts_ner, f)



if __name__ == "__main__":
    main()