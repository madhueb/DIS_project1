import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class VecDataset(Dataset):

    def __init__(self, q_id, lang, vec):
        self.q_id = q_id
        self.lang = lang
        self.vec = vec


    def __getitem__(self, idx):
        return {
            "q_id": self.q_id[idx],
            "lang": self.lang[idx],
            "vec": self.vec[idx]
        }

    def __len__(self):
        return len(self.q_id)

    def collate_fn(self, batch):
        return_dict = {}
        q_ids = [sample["q_id"] for sample in batch]
        langs = [sample["lang"] for sample in batch]
        vecs = torch.tensor([sample["vec"] for sample in batch], dtype=torch.float32)

        return_dict["q_id"] = q_ids
        return_dict["lang"] = langs
        return_dict["vec"] = vecs

        return return_dict

