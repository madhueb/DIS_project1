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
from multiprocessing import Pool

import string
from camel_tools.utils.dediac import dediac_ar
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.lemmatizer import Lemmatizer

from nltk.corpus import stopwords
import nltk

import os

LANGS = ["ar"]


class BaseTokenizer:

    def __init__(self):
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('arabic'))
        self.lemmatizer = Lemmatizer()


    def preprocess_text(self, text: str) -> List[str]:
        # Step 1: Remove URLs
        text = re.sub(r"http[s]?://\S+|www\.\S+", " ", text)

        # Step 2: Remove long sequences of non-alphanumeric characters (e.g., encoded data or code)
        text = re.sub(r"[^\w\s]{4,}", " ", text)  # Removes any sequence of 4 or more non-alphanumeric characters

        # Step 3: Remove excessive whitespace
        text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip().lower()

        # Step 4: Remove diacritics
        text = dediac_ar(text)

        # Step 5: Tokenize text
        tokens = simple_word_tokenize(text)

        # Step 6: Lemmatize each token and remove stopwords
        processed_tokens = [
            self.lemmatizer.lemmatize(token)[0] for token in tokens if token not in self.stop_words
        ]
        return processed_tokens


    def tokenize_batch(
            self, texts: List[str], cores: int = 10
    ) -> List[List[str]]:
        with Pool(cores) as pool:
            results = pool.map(self.preprocess_text, texts)
        return results


class ArabicTokenizer(BaseTokenizer):
    def __init__(self):
        # multi linguistic model
        super().__init__()

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

    if args.language == "ar":
        tokenizer = ArabicTokenizer()
    else:
        raise KeyError("language")

    tokenized_texts = tokenizer.tokenize_batch(corpus_df["text"].tolist(), cores=args.cores)

    output_file = (
            args.output_dir
            / f"tokens_{args.language}_{args.split_index}_{args.num_splits}.json"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    with open(output_file, "wb") as f:
        pickle.dump(tokenized_texts, f)


