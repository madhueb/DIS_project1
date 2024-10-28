import argparse
from pathlib import Path
import pickle
from transformers import AutoTokenizer
import json
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


    json_dict = {}
    for i, docid in enumerate(docids_i):
        json_dict[docid] = {
            "lang": langs_i[i],
            "chunks": all_chunks_i[i]
        }


    for docid in tqdm(json_dict):
        tokenized = tokenizer(
                            json_dict[docid]["chunks"],
                            padding=True,
                            truncation=True,
                            add_special_tokens=True,
                            max_length=512,
                            return_tensors="pt",
                            )
        for key in tokenized:
            tokenized[key] = tokenized[key].tolist()
        json_dict[docid]["chunks"] = tokenized


    with open(args.output / f"tokenized_{args.split_id}.json", "w") as f:
        json.dump(json_dict, f)




if __name__ == "__main__":
    main()
