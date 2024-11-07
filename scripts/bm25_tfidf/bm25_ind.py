import argparse
import pickle
from pathlib import Path

from src.bm25_tfidf.bm25 import BM25

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--token_dir", type=Path, default = "./data")
    parser.add_argument("-lang", "--language", type=str, required=True, choices=['ar','de','en','es','fr','it','ko'])
    args = parser.parse_args()

    with open(f'{args.token_dir}/tokens_{args.language}.pkl', "rb") as f:
        docs = pickle.load(f)

    lang_params = {
        "ar": {"k1": 1.6, "b": 1.0},
        "de": {"k1": 2.0, "b": 0.95},
        "en": {"k1": 1.4, "b": 0.5},
        "es": {"k1": 1.7, "b": 0.95},
        "fr": {"k1": 1.8, "b": 0.95},
        "it": {"k1": 2.0, "b": 0.9},
        "ko": {"k1": 1.7, "b": 1.0},
    }

    bm25_ind = BM25(k1=lang_params[args.language]["k1"], b=lang_params[args.language]["b"])
    bm25_ind.fit(docs)

    with open(f"bm25_{args.language}.pkl", "wb") as f:
        pickle.dump(bm25_ind, f)

    print("BM25 model created for " + args.language)