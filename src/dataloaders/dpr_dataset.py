
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
            "query": torch.tensor(row['query_embed'], dtype=torch.float32),
        }

        if self.is_test:
            inputs['query_id'] = row['query_id']
            inputs['lang'] = row['lang']
            return inputs

        positive_idx = row['positive_docs']
        positive_sample = self.doc_embeds[positive_idx]['embeds']
        positive_sample = torch.tensor(positive_sample, dtype=torch.float32)

        negative_indices = row['negative_docs'][1:-1].split(', ')
        negative_sample_list = [self.doc_embeds[neg_idx]['embeds'] for neg_idx in negative_indices]
        negative_sample = []
        for sample in negative_sample_list:
            negative_sample += sample
        negative_sample = torch.tensor(negative_sample, dtype=torch.float32)

        inputs['positive'] = positive_sample
        inputs['negative'] = negative_sample

        return inputs

    def __len__(self):
        return len(self.query_df)

    def collate_fn(self, batch):

        return_dict = {}
        queries = torch.tensor([sample["query"] for sample in batch])

        if self.is_test:
            return_dict['query_id'] = [sample["query_id"] for sample in batch]
            return_dict['lang'] = [sample["lang"] for sample in batch]
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
            query_positive.append(torch.cat([query] * len(positive), dim=0))
            query_negative.append(torch.cat([query] * len(negative), dim=0))
            positive_.append(positive)
            negative_.append(negative)

        query_positive = torch.cat(query_positive, dim=0)
        query_negative = torch.cat(query_negative, dim=0)
        positive_ = torch.cat(positive_, dim=0)
        negative_ = torch.cat(negative_, dim=0)

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



