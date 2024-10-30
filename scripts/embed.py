import argparse


import pickle

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from torch.utils.data import Dataset

class DPRDataset(Dataset):

    def __init__(self, q_id, lang, tokens):
        self.q_id = []
        self.lang = []
        self.tokens = []
        for i, chunks in enumerate(tqdm(tokens, desc="Dataset...")):
            for j in range(len(chunks['input_ids'])):
                self.q_id.append(q_id[i])
                self.lang.append(lang[i])
                chunk_dict = {}
                for k, v in chunks.items():
                    chunk_dict[k] = v[j]
                self.tokens.append(chunk_dict)

    def __getitem__(self, idx):
        return {
            "q_id": self.q_id[idx],
            "lang": self.lang[idx],
            "tokens": self.tokens[idx]
        }

    def __len__(self):
        return len(self.q_id)

    def collate_fn(self, batch):
        return_dict = {}
        q_ids = [sample["q_id"] for sample in batch]
        langs = [sample["lang"] for sample in batch]
        tokens = {k: [sample["tokens"][k] for sample in batch] for k in batch[0]["tokens"].keys()}
        tokens = {k: torch.stack(v) for k, v in tokens.items()}

        return_dict["q_id"] = q_ids
        return_dict["lang"] = langs
        return_dict["tokens"] = tokens

        return return_dict


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
    # Build dataloader
    dataset = DPRDataset(docids_i, langs_i, tokenized_all_chunks_i)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, collate_fn=dataset.collate_fn, shuffle=False, num_workers=4)

    model = AutoModel.from_pretrained("microsoft/mdeberta-v3-base").to("cuda" if torch.cuda.is_available() else "cpu")

    docids_i = []
    langs_i = []
    embed_all_chunks_i = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Embedding..."):
            inputs = {k: v.to(model.device) for k, v in batch["tokens"].items()}
            outputs = model(**inputs, return_dict=True)
            outputs = pooling(outputs['last_hidden_state'], inputs['attention_mask']).cpu().numpy()
            docids_i += batch["q_id"]
            langs_i += batch["lang"]
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
