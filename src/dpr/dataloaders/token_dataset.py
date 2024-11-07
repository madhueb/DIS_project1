import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class TokenDataset(Dataset):
    """
    TokenDataset class for processing tokenized query data for language models.

    Attributes:
        q_id (list): List of query IDs.
        lang (list): List of language labels corresponding to each query.
        tokens (list of dict): List of dictionaries containing tokenized data 
                            for each query, including fields like 'input_ids' 
                            and 'attention_mask'.

    Methods:
        __getitem__(idx): Returns a dictionary with 'q_id', 'lang', and 'tokens' 
                        for the given index.
        __len__(): Returns the number of entries in the dataset.
        collate_fn(batch): Custom collate function that merges a list of samples 
                        into a batch for DataLoader. The function returns 
                        a dictionary containing batched 'q_id', 'lang', and 
                        'tokens'.

    """


    def __init__(self, q_id, lang, tokens):
        """
        Initializes the TokenDataset with query IDs, language labels, and tokenized data.
        Args:
            q_id (list): List of query IDs.
            lang (list): List of language labels.
            tokens (list of dict): List of dictionaries containing tokenized data.
        """
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
        """
        Returns a dictionary with 'q_id', 'lang', and 'tokens' for the given index.
        Args:
            idx (int): Index of the dataset element.
        Returns:   
            dict: Dictionary containing 'q_id', 'lang', and 'tokens'.
        """
        return {
            "q_id": self.q_id[idx],
            "lang": self.lang[idx],
            "tokens": self.tokens[idx]
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
        Custom collate function to merge a list of samples into a batch.
        Args:
            batch (List[dict]): List of samples containing 'q_id', 'lang', and 'tokens'.
        Returns:
            dict: Dictionary containing batched 'q_id', 'lang', and 'tokens'.
        """
        return_dict = {}
        q_ids = [sample["q_id"] for sample in batch]
        langs = [sample["lang"] for sample in batch]
        tokens = {k: [sample["tokens"][k] for sample in batch] for k in batch[0]["tokens"].keys()}
        tokens = {k: torch.stack(v) for k, v in tokens.items()}

        return_dict["q_id"] = q_ids
        return_dict["lang"] = langs
        return_dict["tokens"] = tokens

        return return_dict
