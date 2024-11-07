import argparse
from pathlib import Path
import pickle
from transformers import AutoTokenizer
from tqdm import tqdm

def main() -> None:
    """
    Tokenize the corpus.
    
    Command-line Arguments:
        --chunks (Path): Path to chunked corpus directory.
        --output (Path): Path to output directory.
        --split_id (int): Split id.
        
    Output:
        Saves a pickled file containing the tokenized corpus.
    """
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
    for i, text in enumerate(tqdm(all_chunks_i, desc="Tokenizing")):
        tokens = tokenizer(
                text,
                padding=False,
                truncation=False,
                add_special_tokens=False,
                return_tensors="pt",
            )
        tokenized_all_chunks_i.append(tokens["input_ids"].squeeze().tolist())

    assert len(tokenized_all_chunks_i) == len(docids_i)

    tokenized_corpus_i = (docids_i, langs_i, tokenized_all_chunks_i)
    with open(args.output / f"tokenized_corpus_{args.split_id}.pkl", "wb") as f:
        pickle.dump(tokenized_corpus_i, f)


if __name__ == "__main__":
    main()
