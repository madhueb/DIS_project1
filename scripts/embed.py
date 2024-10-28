import argparse
from pathlib import Path
import pickle

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from src.models.utils import pooling


def main() -> None:
    parser = argparse.ArgumentParser(description="Embedding")
    parser.add_argument("--tokens", type=Path, help="path to tokenized corpus directory")
    parser.add_argument("--output", type=Path, help="path to output directory")
    parser.add_argument("--split_id", type=int, help="split id")
    args = parser.parse_args()

    with open(args.chunks / f"tokenized_corpus_{args.split_id}.pkl", "rb") as f:
        corpus_i = pickle.load(f)
    docids_i, langs_i, tokenized_all_chunks_i = corpus_i

    model = AutoModel.from_pretrained("microsoft/mdeberta-v3-base").to("cuda" if torch.cuda.is_available() else "cpu")

    embed_all_chunks_i = []
    for i, chunks in enumerate(tqdm(tokenized_all_chunks_i, desc="Embedding")):
        chunks = {k: v.to(model.device) for k, v in chunks.items()}
        with torch.no_grad():
            outputs = model(**chunks, return_dict=True)
            outputs = pooling(outputs['last_hidden_state'], chunks['attention_mask'])
        embed_all_chunks_i.append(outputs)

    embed_all_chunks_i = (docids_i, langs_i, embed_all_chunks_i)
    with open(args.output / f"embed_all_chunks_{args.split_id}.pkl", "wb") as f:
        pickle.dump(embed_all_chunks_i, f)


if __name__ == "__main__":
    main()
