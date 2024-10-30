
import torch
from torch.utils.data import Dataset

class DPRDataset(Dataset):

    def __init__(self, query_df, is_test=False, doc_embeds=None):
        self.query_df = query_df.reset_index(drop=True)
        self.doc_embeds = doc_embeds
        self.is_test = is_test

    def __getitem__(self, idx):
        row = self.query_df.iloc[idx]
        inputs = {
            "query": torch.tensor(row['query_embed'][0], dtype=torch.float32),
        }

        if self.is_test:
            inputs['query_id'] = row['query_id']
            inputs['lang'] = row['lang']
            return inputs

        positive_idx = row['positive_docs']
        positive_sample = self.doc_embeds[positive_idx]['embeds']

        negative_indices = row['negative_docs'][1:-1].split(', ')
        negative_sample = [neg for neg_idx in negative_indices for neg in self.doc_embeds[neg_idx[1:-1]]['embeds']]

        inputs['positive'] = positive_sample
        inputs['negative'] = negative_sample

        return inputs

    def __len__(self):
        return len(self.query_df)

    def collate_fn(self, batch):

        return_dict = {}
        queries = [sample["query"] for sample in batch]

        if self.is_test:
            return_dict['query_id'] = [sample["query_id"] for sample in batch]
            return_dict['lang'] = [sample["lang"] for sample in batch]
            queries = torch.stack(queries)
            return_dict['query'] = queries
            return return_dict

        query_positive_masks = []
        query_negative_masks = []
        query_positive = []
        query_negative = []
        positive_ = []
        negative_ = []

        for i in range(len(queries)):
            query = queries[i]
            positive = batch[i]['positive']
            negative = batch[i]['negative']
            query_positive += [query] * len(positive)
            query_negative += [query] * len(negative)
            positive_ += positive
            negative_ += negative

        query_positive = torch.stack(query_positive)
        query_negative = torch.stack(query_negative)
        positive_ = torch.stack(positive_)
        negative_ = torch.stack(negative_)

        tmp = 0
        for i in range(query_positive.shape[0]):
            pos_mask = torch.zeros(query_positive.shape[0])
            pos_mask[tmp:tmp + len(batch[i]['positive'])] = 1
            tmp += len(batch[i]['positive'])
            query_positive_masks.append(pos_mask)

        tmp = 0
        for i in range(query_negative.shape[0]):
            neg_mask = torch.zeros(query_negative.shape[0])
            neg_mask[tmp:tmp + len(batch[i]['negative'])] = 1
            tmp += len(batch[i]['negative'])
            query_negative_masks.append(neg_mask)

        query_positive_masks = torch.stack(query_positive_masks)
        query_negative_masks = torch.stack(query_negative_masks)

        return_dict['positive'] = positive_
        return_dict['negative'] = negative_
        return_dict['query_positive'] = query_positive
        return_dict['query_negative'] = query_negative
        return_dict['query_positive_mask'] = query_positive_masks
        return_dict['query_negative_mask'] = query_negative_masks

        return return_dict



