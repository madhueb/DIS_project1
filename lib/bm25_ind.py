import argparse
import pickle

from lib.bm25.bm25 import BM25

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", "--token_dir", type=Path, default = "./data")
    parser.add_argument("-lang", "--language", type=str, required=True, choices=['ar','de','en','es','fr','it','ko'])
    args = parser.parse_args()

    with open(f'{args.token_dir}/tokens_{args.language}.pkl', "rb") as f:
        docs = pickle.load(f)

    bm25_ind = BM25()
    bm25_ind.fit(docs)

    with open(f"bm25_{args.language}.pkl", "wb") as f:
        pickle.dump(bm25_ind, f)

    print("BM25 model created for " + args.language)