import argparse
import gc
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.bm25_tfidf.tokenizer import (
    FrenchTokenizer,
    EnglishTokenizer,
    GermanTokenizer,
    ItalianTokenizer,
    SpanishTokenizer,
    ArabicTokenizer,
    KoreanTokenizer
)

LANGS = ["fr", "de", "it", "es", "ar", "ko", "en"]

tokenizers = {"fr": FrenchTokenizer(), "de": GermanTokenizer(), "it": ItalianTokenizer(), "es": SpanishTokenizer(),
              "ar": ArabicTokenizer(), "ko": KoreanTokenizer(), "en": EnglishTokenizer()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--token_dir", type=Path, default = "./data")
    parser.add_argument("--ids_path", type=Path, default="./data/ids_dict.json")
    parser.add_argument("--bm25_path", type=Path, default=".")

    args = parser.parse_args()

    bm25s = {}

    for lang in LANGS:
        with open(f"{args.bm25_path}/bm25_{lang}.pkl", "rb") as f:
            bm25s[lang] = pickle.load(f)

    # load doc ids dict with json
    with open(args.ids_path, "r") as f:
        ids_dict = json.load(f)

    for lang in LANGS:
        ids_dict[lang] = np.array(ids_dict[lang])

    # queries = pd.read_csv(f'{args.token_dir}/test.csv')
    queries = pd.read_csv(f'{args.token_dir}/train.csv')

    ls = [[] for _ in range(len(queries))]
    queries["docids"] = ls

    for lang in LANGS:
        queries_lang = queries[queries["lang"] == lang][["query", "positive_docs"]].reset_index(drop=True)
        # queries_lang = queries[queries["lang"] == lang][["query"]].reset_index(drop=True)
        tokens = tokenizers[lang].tokenize(queries_lang["query"].tolist(), lang)
        bm25_ind = bm25s[lang]
        doc_ids = []
        for tokenized_query in tokens:
            indices, _ = bm25_ind.match(tokenized_query, k=10)
            doc_ids.append(ids_dict[lang][indices].tolist())

        # queries.loc[queries["lang"] == lang, "docids"] = pd.Series(doc_ids, index=queries.loc[queries["lang"] == lang].index)
        acc = 0
        for i, row in queries_lang.iterrows():
            if row["positive_docs"] in doc_ids[i]:
                acc += 1
        print(f"Accuracy for {lang} : {acc / len(queries_lang)}")
        gc.collect()

    queries = queries[["id", "docids"]]
    queries.to_csv(f"{args.token_dir}/submission.csv", index=False)
