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
import pyphen

LANGS = ["en", "fr", "de", "it", "es", "ar", "ko"]


class BaseTokenizer:
    """
    Base tokenizer class.
    
    Attributes:
        nlp (spacy.Language): Spacy NLP pipeline.
        stop_words (Set[str]): Set of stop words.
        
    Methods:
        preprocess_text(text: str) -> str: Preprocess text.
        tokenize_batch(texts: List[str], batch_size: int = 64, n_process: int = 8) -> List[List[str]]: Tokenize texts.
        """

    def __init__(self, model_name: str):
        """
        Initialize BaseTokenizer.
        Args:
            model_name (str): Spacy model name.
        """
        spacy.cli.download(model_name)
        self.nlp = spacy.load(model_name, exclude=["senter", "ner"])
        self.stop_words = set(self.nlp.Defaults.stop_words)

    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Preprocess text : Remove URLs, long sequences of non-alphanumeric characters, and excessive whitespace.
        Args:
            text (str): Input text.
        Returns:
            str: Preprocessed text.
        """
        # Step 1: Remove URLs
        text = re.sub(r"http[s]?://\S+|www\.\S+", " ", text)

        # Step 2: Remove long sequences of non-alphanumeric characters (e.g., encoded data or code)
        text = re.sub(r"[^\w\s]{4,}", " ", text)  # Removes any sequence of 4 or more non-alphanumeric characters

        # Step 3: Remove excessive whitespace
        return re.sub(r"\s+", " ", text.replace("\n", " ")).strip().lower()

    def tokenize_batch(self, texts: List[str], batch_size: int = 64, n_process: int = 8):
        """
        Tokenize texts by batches.
        Args:
            texts (List[str]): List of texts.
            batch_size (int): Batch size for processing texts.
            n_process (int): Number of processes to use for tokenization.
        Returns:
            List[List[str]]: List of tokenized texts.
        """
        
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

class GermanTokenizerV2:
    """
    German tokenizer using Pyphen for splitting compound words.
    Attributes:
        nlp (spacy.Language): Spacy NLP pipeline.
        stop_words (Set[str]): Set of stop words.
        pyphen_dic (pyphen.Pyphen): Pyphen dictionary.
    Methods:    
        preprocess_text(text: str) -> str: Preprocess text.
        split_compound_word(word: str) -> List[str]: Split compound word.
        tokenize_batch(texts: List[str], batch_size: int = 64, n_process: int = 8) -> List[List[str]]: Tokenize texts.
    """
    MODEL_NAME = "de_core_news_md"

    def __init__(self):
        """
        Initialize GermanTokenizer.
        """
        spacy.cli.download(self.MODEL_NAME)
        self.nlp = spacy.load(self.MODEL_NAME, exclude=["senter", "ner"])
        self.stop_words = set(self.nlp.Defaults.stop_words)
        self.pyphen_dic = pyphen.Pyphen(lang="de_DE")

    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Preprocess text : Remove URLs, long sequences of non-alphanumeric characters, and excessive whitespace.
        Args:
            text (str): Input text.
        Returns:
            str: Preprocessed text.
        """
        text = re.sub(r"http[s]?://\S+|www\.\S+", " ", text)
        text = re.sub(r"[^\w\s]{4,}", " ", text)
        return re.sub(r"\s+", " ", text.replace("\n", " ")).strip().lower()

    def split_compound_word(self, word: str) -> List[str]:
        """
        Split compound word.
        Args:
            word (str): Compound word.
        Returns:
            List[str]: List of split words.
        """
        split_word = self.pyphen_dic.inserted(word)
        return split_word.split("-")

    def tokenize_batch(self, texts: List[str], batch_size: int, n_process: int):
        """
        Tokenize texts by batches.
        Args:
            texts (List[str]): List of texts.
            batch_size (int): Batch size for processing texts.
            n_process (int): Number of processes to use for tokenization.
        Returns:
            List[List[str]]: List of tokenized texts.
        """
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



class SpanishTokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__("es_core_news_sm")


class KoreanTokenizer(BaseTokenizer):
    def __init__(self):
        super().__init__("ko_core_news_sm")


def main():
    """
    Processes and tokenizes a text corpus in specified languages.

    Command-line Arguments:
        -c, --corpus_df (Path): Path to the input corpus JSON file. Required.
        -o, --output_dir (Path): Path to the directory where output files will be saved. Required.
        -l, --language (str): The language code for filtering and tokenizing (choices defined by LANGS). Required.
        -n, --num_splits (int): The number of parts to split the corpus into. Required.
        -i, --split_index (int): The index of the split to process (1-indexed). Required.
        -b, --batch_size (int): The batch size for tokenization (default is 64).
        --cores (int): The number of processor cores to use for parallel processing (default is 10).


    Saves:
        A pickled file containing tokenized text data.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--corpus_df", type=Path, required=True)
    parser.add_argument("-o", "--output_dir", type=Path, required=True)
    parser.add_argument("-l", "--language", type=str, required=True, choices=LANGS)
    parser.add_argument("-n", "--num_splits", type=int, required=True)
    parser.add_argument("-i", "--split_index", type=int, required=True)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("--cores", type=int, default=10)

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
        # tokenizer = GermanTokenizerV2()
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

    tokenized_texts = tokenizer.tokenize_batch(corpus_df["text"].tolist(), batch_size=args.batch_size, n_process=args.cores)

    output_file = (
        args.output_dir
        / f"tokens_{args.language}_{args.split_index}_{args.num_splits}.pkl"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    with open(output_file, "wb") as f:
        pickle.dump(tokenized_texts, f)


if __name__ == "__main__":
    main()