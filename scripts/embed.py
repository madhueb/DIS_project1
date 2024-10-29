import argparse


import pickle

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def pooling(hidden_states, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
    sum_mask = input_mask_expanded.sum(1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    mean_embeddings = sum_embeddings / sum_mask
    return mean_embeddings

def embed(tokens_path, output_path, split_id) -> None:


    with open(f"{tokens_path}/tokenized_corpus_{split_id}.pkl", "rb") as f:
        corpus_i = pickle.load(f)
    docids_i, langs_i, tokenized_all_chunks_i = corpus_i

    model = AutoModel.from_pretrained("microsoft/mdeberta-v3-base").to("cuda" if torch.cuda.is_available() else "cpu")

    embed_all_chunks_i = []
    for i, chunks in enumerate(tqdm(tokenized_all_chunks_i, desc="Embedding")):
        chunks = {k: torch.tensor(v).to(model.device) for k, v in chunks.items()}
        with torch.no_grad():
            outputs = model(**chunks, return_dict=True)
            outputs = pooling(outputs['last_hidden_state'], chunks['attention_mask']).to('cpu')
        embed_all_chunks_i.append(outputs)

    embed_all_chunks_i = (docids_i, langs_i, embed_all_chunks_i)
    with open(f"{output_path}/embed_all_chunks_{split_id}.pkl", "wb") as f:
        pickle.dump(embed_all_chunks_i, f)





def main() -> None:
    parser = argparse.ArgumentParser(description="Embedding")
    parser.add_argument("--tokens", type=str, help="path to tokenized corpus directory")
    parser.add_argument("--output", type=str, help="path to output directory")
    parser.add_argument("--split_id", type=int, help="split id")
    args = parser.parse_args()
    embed(args.tokens, args.output, args.split_id)


if __name__ == "__main__":
    main()
