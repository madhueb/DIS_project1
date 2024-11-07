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

from camel_tools.disambig.mle import MLEDisambiguator

from nltk.corpus import stopwords
import nltk

import os

LANGS = ["ar"]


class BaseTokenizer:
    """
    Arabic tokenizer class.

    Attributes:
        translator (str): Punctuation translator.
        mle (MLEDisambiguator): MLE disambiguator.
        stop_words (List[str]): List of stop words.

    Methods:
        preprocess_text(text: str) -> str: Preprocess text.
        tokenize(texts: List[str]) -> List[List[str]]: Tokenize texts.
    """

    def __init__(self, stop_words_path=None):
        """
        Initialize ArabicTokenizer.
        Args:
            stop_words_path (str): Path to the stop words file.
        """

        punctuations = '''`÷×؛<>«»_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ''' + string.punctuation
        self.translator = str.maketrans('', '', punctuations)
        self.mle = MLEDisambiguator.pretrained()
        nltk.download('stopwords')
        # self.stop_words = set(stopwords.words('arabic'))
        with open(stop_words_path, 'r') as file:
            self.stop_words = file.read().split('\n') + list(nltk.corpus.stopwords.words("arabic"))


    def preprocess_text(self, text: str) -> List[str]:
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

        # Step 3: Remove punctuation
        text = text.translate(self.translator)

        # Step 4: Remove excessive whitespace
        text = re.sub(r"\s+", " ", text.replace("\n", " ")).strip().lower()

        # Step 5: Remove diacritics
        text = dediac_ar(text)

        # Step 6: Tokenize text
        tokens = simple_word_tokenize(text)

        # Step 7: Lemmatize each token and remove stopwords

        disambig = self.mle.disambiguate(tokens)

        lemmas = [dediac_ar(d.analyses[0].analysis['lex']).translate(self.translator) for d in disambig]
        lemmas = [lemma for lemma in lemmas if lemma not in self.stop_words]

        return lemmas

    def tokenize_batch(self, texts: List[str], cores: int = 10) -> List[List[str]]:
        """
        Tokenize texts by batches.
        Args:
            texts (List[str]): List of texts.
            cores (int): Number of cores to use for tokenization.
        Returns:
            List[List[str]]: List of tokenized texts.
        """
        print("Tokenizing...")
        with Pool(cores) as pool:
            results = pool.map(self.preprocess_text, texts)
        return results


class ArabicTokenizer(BaseTokenizer):
    def __init__(self, stop_words_path=None):
        super().__init__(stop_words_path)

def main():
    """
    Processes and tokenizes an arabic text corpus.

    Command-line Arguments:
        -c, --corpus_df (Path): Path to the input corpus JSON file. Required.
        -o, --output_dir (Path): Path to the directory where output files will be saved. Required.
        -l, --language (str): The language code for filtering and tokenizing (choices defined by LANGS). Required.
        -n, --num_splits (int): The number of parts to split the corpus into. Required.
        -i, --split_index (int): The index of the split to process (1-indexed). Required.
        -b, --batch_size (int): The batch size for tokenization (default is 64).
        -p, --cores (int): The number of processor cores to use for parallel processing (default is 10).
        --topwords_path (Path): Optional path to a file containing top words for the tokenizer.


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
    parser.add_argument("-p", "--cores", type=int, default=10)
    parser.add_argument("--topwords_path", type=Path, default=None)

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
        tokenizer = ArabicTokenizer(args.topwords_path)
    else:
        raise KeyError("language")

    tokenized_texts = tokenizer.tokenize_batch(corpus_df["text"].tolist(), cores=args.cores)

    output_file = (
            args.output_dir
            / f"tokens_{args.language}_{args.split_index}_{args.num_splits}.pkl"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    with open(output_file, "wb") as f:
        pickle.dump(tokenized_texts, f)


if __name__ == "__main__":
    main()