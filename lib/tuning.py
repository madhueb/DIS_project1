import argparse
import pandas as pd
from pathlib import Path
import pickle
from typing import List
from lib.utils import get_tokens
from lib.bm25.bm25 import BM25


def recall_at_k(ranks: List[int], k: int) -> float:
    return sum([1 for rank in ranks if rank <= k]) / len(ranks)


def test_bm25(
    k1: float,
    b: float,
    corpus_tokens: List[List[str]],
    queries_tokens: List[List[str]],
    queries_targets: List[int],
    k_for_recall: int,
) -> float:
    bm25 = BM25(k1=k1, b=b)
    bm25.fit(corpus_tokens)

    ranks = []
    for query_tokens, target in zip(queries_tokens, queries_targets):
        _, _, rank_target = bm25.match_and_eval(query_tokens, target)
        ranks.append(rank_target)

    return recall_at_k(ranks, k_for_recall)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--lang", type=str)
    parser.add_argument("--k1s", type=float, nargs="+")
    parser.add_argument("--bs", type=float, nargs="+")
    parser.add_argument("--k_for_recall", type=int, default=10)
    parser.add_argument("--dataset_path", type=Path)
    parser.add_argument("--corpus_path", type=Path)
    parser.add_argument("--output_path", type=Path)

    args = parser.parse_args()

    lang = args.lang
    k1s = args.k1s
    bs = args.bs
    k_for_recall = args.k_for_recall
    dataset_path = args.dataset_path
    corpus_path = args.corpus_path
    output_path = args.output_path

    with open(corpus_path, mode="rb") as f:
        corpus_ids, corpus_tokens = pickle.load(f)

    df = pd.read_csv(dataset_path)
    df = df[df["lang"] == lang]
    df["positive_docs_i"] = df["positive_docs"].apply(lambda x: corpus_ids.index(x))

    queries_tokens = get_tokens(df["query"].tolist(), lang)
    queries_targets = df["positive_docs_i"].tolist()

    results_df = []

    for k1, b in [(k1, b) for k1 in k1s for b in bs]:
        metric = test_bm25(
            k1, b, corpus_tokens, queries_tokens, queries_targets, k_for_recall
        )
        results_df.append({"lang": lang, "k1": k1, "b": b, "recall@10": metric})
        print(f"lang: {lang}, k1: {k1}, b: {b}, recall@10: {metric:.4f}")

    results_df = pd.DataFrame(results_df)
    results_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
