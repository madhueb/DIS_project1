import argparse
import gc
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from lib.utils import get_tokens
from lib.bm25_inf import LANGS, ids_dict, bm25s
from tfidf.inferrence import tfidfs, retrieve_top_k


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-dir", "--token_dir", type=Path, default = "./data")
    args = parser.parse_args()

    # queries = pd.read_csv(f'{args.token_dir}/test.csv')
    queries = pd.read_csv(f'{args.token_dir}/train.csv')

    ls = [[] for _ in range(len(queries))]
    queries["docids"] = ls

    k_b = 8
    k = 10

    for lang in LANGS:
        queries_lang = queries[queries["lang"] == lang][["query", "positive_docs"]].reset_index(drop=True)
        # queries_lang = queries[queries["lang"] == lang][["query"]].reset_index(drop=True)
        tokens = get_tokens(queries_lang["query"].tolist(), lang)
        bm25_ind = bm25s[lang]
        bm25_ind_doc_ids = []
        for tokenized_query in tokens:
            indices, _ = bm25_ind.match(tokenized_query, k=k)
            bm25_ind_doc_ids.append(ids_dict[lang][indices].tolist())

        tfidf_doc_ids = [docid.tolist() for docid in retrieve_top_k(tokens, lang, k=k)]

        doc_ids = []
        for i in range(len(queries_lang)):
            docid = list(set(bm25_ind_doc_ids[i][:k_b] + tfidf_doc_ids[i][:k - k_b]))
            if len(docid) < k:
                docid = bm25_ind_doc_ids[i][:k]
            doc_ids.append(docid)

        # queries.loc[queries["lang"] == lang, "docids"] = pd.Series(doc_ids, index=queries.loc[queries["lang"] == lang].index)

        acc = 0
        for i, row in queries_lang.iterrows():
            if row["positive_docs"] in doc_ids[i]:
                acc += 1
        print(f"Accuracy for {lang} : {acc / len(queries_lang)}")

        gc.collect()

    # queries = queries[["id", "docids"]]
    # queries.to_csv(f"{args.token_dir}/submission.csv", index=False)
