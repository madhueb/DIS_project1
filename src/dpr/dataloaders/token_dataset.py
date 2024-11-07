import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class TokenDataset(Dataset):

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
