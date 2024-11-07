import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class VecDataset(Dataset):
    """
    VecDataset class for processing vectorized query data for language models.

    Attributes:
        q_id (list): List of query IDs.
        lang (list): List of language labels corresponding to each query.
        vec (list): List of vectors corresponding to each query.

    Methods:

        __getitem__(idx): Returns a dictionary with 'q_id', 'lang', and 'vec' for the given index.
        __len__(): Returns the number of entries in the dataset.
        collate_fn(batch): Custom collate function that merges a list of samples into a batch for DataLoader. The function returns a dictionary containing batched 'q_id', 'lang', and 'vec'.

    """

    def __init__(self, q_id, lang, vec):
        """
        Initializes the VecDataset with query IDs, language labels, and vectorized data.
        Args:
            q_id (list): List of query IDs.
            lang (list): List of language labels.
            vec (list): List of vectors.
        """
        self.q_id = q_id
        self.lang = lang
        self.vec = vec


    def __getitem__(self, idx):
        """
        Returns a dictionary with 'q_id', 'lang', and 'vec' for the given index.
        Args:
            idx (int): Index of the dataset element.

        Returns:
            dict: Dictionary containing 'q_id', 'lang', and 'vec'.
        """
        return {
            "q_id": self.q_id[idx],
            "lang": self.lang[idx],
            "vec": self.vec[idx]
        }

    def __len__(self):
        """
        Returns the number of entries in the dataset.
        Returns:
            int: Number of entries in the dataset.
        """
        return len(self.q_id)

    def collate_fn(self, batch):
        """
        Custom collate function that merges a list of samples into a batch for DataLoader.
        Args:
            batch (List[dict]): List of samples containing query and vector data.

        Returns:
            dict: Dictionary containing batched 'q_id', 'lang', and 'vec'.
        """
        return_dict = {}
        q_ids = [sample["q_id"] for sample in batch]
        langs = [sample["lang"] for sample in batch]
        vecs = torch.tensor([sample["vec"] for sample in batch], dtype=torch.float32)

        return_dict["q_id"] = q_ids
        return_dict["lang"] = langs
        return_dict["vec"] = vecs

        return return_dict

