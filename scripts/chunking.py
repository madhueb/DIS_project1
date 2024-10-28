import argparse
from pathlib import Path
import pickle
import re

import pandas as pd
from tqdm import tqdm

from bs4 import BeautifulSoup


def clean_text(text):
    soup = BeautifulSoup(text, "lxml")

    for img in soup.find_all("img"):
        img.decompose()

    for table in soup.find_all("table"):
        table.decompose()

    clean_text = soup.get_text()
    clean_text = re.sub(r"http\S+|www\S+", " ", clean_text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    return clean_text


def chunk_text(text, chunk_size=270, overlap_size=130, trim_threshold=30):
    words = text.split()

    chunks = []
    start = 0
    end = 0

    while start < len(words) and end < len(words):
        end = start + chunk_size

        start_truncation_index = next(
            (
                i
                for i, word in enumerate(words[start:end])
                if word.endswith((".", "!", "?"))
            ),
            None,
        )
        if (
            start > 0
            and start_truncation_index is not None
            and start_truncation_index < trim_threshold
        ):
            start += start_truncation_index + 1

        if start >= len(words):
            break

        end = start + chunk_size

        end_truncation_index = next(
            (
                i
                for i, word in reversed(list(enumerate(words[start:end])))
                if word.endswith((".", "!", "?"))
            ),
            None,
        )
        if (
            end_truncation_index is not None
            and end_truncation_index > 0
            and chunk_size - end_truncation_index <= trim_threshold
        ):
            end = start + end_truncation_index + 1

        chunks.append(" ".join(words[start:end]))

        start = end - overlap_size

    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunking")
    parser.add_argument("corpus", type=Path, help="path to corpus.json")
    parser.add_argument("output", type=Path, help="path to output directory")
    parser.add_argument("num_splits", type=int, default=12, help="number of splits")
    args = parser.parse_args()

    corpus_df = pd.read_json(args.corpus)

    all_chunks = [
        chunk_text(clean_text(row["text"]))
        for _, row in tqdm(corpus_df.iterrows(), total=len(corpus_df))
    ]
    docids = corpus_df["docid"].tolist()
    langs = corpus_df["lang"].tolist()

    del corpus_df

    split_size = len(all_chunks) // args.num_splits + 1
    for i in range(1, args.num_splits + 1):
        start_i = (i - 1) * split_size
        end_i = start_i + split_size
        corpus_i = (
            docids[start_i:end_i],
            langs[start_i:end_i],
            all_chunks[start_i:end_i],
        )
        with open(args.output / f"corpus_{i}.pkl", "wb") as f:
            pickle.dump(corpus_i, f)
        del corpus_i

    corpus = (docids, langs, all_chunks)
    with open(args.output / "corpus.pkl", "wb") as f:
        pickle.dump(corpus, f)
    del corpus


if __name__ == "__main__":
    main()
