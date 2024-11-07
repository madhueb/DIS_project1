
import torch

from src.dpr.dataloaders.dpr_dataset import DPRDataset


def get_train_val_dataloaders(config, train_df, val_df, doc_embeds):
    train_ds = DPRDataset(train_df, doc_embeds=doc_embeds)
    val_ds = DPRDataset(val_df, doc_embeds=doc_embeds)

    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=config['batch_size'],
                                               shuffle=True,
                                               num_workers=2,
                                               pin_memory=True,
                                               collate_fn=train_ds.collate_fn
                                               )
    val_loader = torch.utils.data.DataLoader(val_ds,
                                             batch_size=config['batch_size'],
                                             shuffle=True,
                                             num_workers=2,
                                             pin_memory=True,
                                             collate_fn=val_ds.collate_fn
                                             )

    return train_loader, val_loader

def get_test_dataloader(config, test_df):
    test_ds = DPRDataset(test_df, is_test=True)
    test_loader = torch.utils.data.DataLoader(test_ds,
                                              batch_size=config['batch_size'],
                                              shuffle=False,
                                              num_workers=2,
                                              pin_memory=True,
                                              collate_fn=test_ds.collate_fn
                                              )
    return test_loader

