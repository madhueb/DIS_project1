import argparse
from pathlib import Path
import pickle
from transformers import AutoTokenizer
from tqdm import tqdm

def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenizing")
    parser.add_argument("--chunks", type=Path, help="path to chunked corpus directory")
    parser.add_argument("--output", type=Path, help="path to output directory")
    parser.add_argument("--split_id", type=int, help="split id")
    args = parser.parse_args()

    with open(args.chunks / f"corpus_{args.split_id}.pkl", "rb") as f:
        corpus_i = pickle.load(f)
    docids_i, langs_i, all_chunks_i = corpus_i

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/mdeberta-v3-base",
        clean_up_tokenization_spaces=True,
        use_fast=False,
    )

    tokenized_all_chunks_i = []
    ids = []
    for i, chunks in enumerate(tqdm(all_chunks_i, desc="Tokenizing")):
        # try:
        tokenized_all_chunks_i.append(
            tokenizer(
                chunks,
                padding='max_length',
                truncation=True,
                add_special_tokens=True,
                max_length=512,
                return_tensors="pt",
            )
        )
        ids.append(i)
        # except Exception as e:
        #     print(e)
        #     print(i, docids_i[i])

    # docids_i = [docids_i[i] for i in ids]
    # langs_i = [langs_i[i] for i in ids]
    tokenized_corpus_i = (docids_i, langs_i, tokenized_all_chunks_i)
    with open(args.output / f"tokenized_corpus_{args.split_id}.pkl", "wb") as f:
        pickle.dump(tokenized_corpus_i, f)


if __name__ == "__main__":
    main()
