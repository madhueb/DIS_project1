
import torch

from tqdm.notebook import tqdm
import gc

from src.training.dpr_loss import DPRLoss
import os


class Trainer:
    def __init__(self, model, loaders, config, accelerator):
        self.model = model
        self.train_loader, self.val_loader = loaders
        self.config = config
        self.accelerator = accelerator
        self.model = self.model.to(self.accelerator.device)

        self.optim = self._get_optim()

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optim, T_0=5, eta_min=1e-7)

        self.model, self.optim, self.train_loader, self.val_loader, self.scheduler = self.accelerator.prepare(
            self.model,
            self.optim,
            self.train_loader,
            self.val_loader,
            self.scheduler
        )

        self.loss_fn = DPRLoss(self.model)

        self.train_losses = []
        self.val_losses = []

        # save path
        if not os.path.exists(self.config['save_path']):
            os.makedirs(self.config['save_path'])

    def _get_optim(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config['lr'], eps=self.config['adam_eps'])
        return optimizer

    def train_one_epoch(self, epoch):

        running_loss = 0.
        progress = tqdm(self.train_loader, total=len(self.train_loader))

        for idx, inputs in enumerate(progress):
            with self.accelerator.accumulate(self.model):

                loss = self.loss_fn(inputs)

                running_loss += loss.item()

                self.accelerator.backward(loss)

                self.optim.step()

                if self.config['enable_scheduler']:
                    self.scheduler.step(epoch - 1 + idx / len(self.train_loader))

                self.optim.zero_grad()

                del inputs, loss

        train_loss = running_loss / len(self.train_loader)
        self.train_losses.append(train_loss)

    @torch.no_grad()
    def valid_one_epoch(self, epoch):

        running_loss = 0.
        progress = tqdm(self.val_loader, total=len(self.val_loader))

        for inputs in progress:
            loss = self.loss_fn(inputs)
            running_loss += loss.item()

            del inputs, loss

        val_loss = running_loss / len(self.val_loader)
        self.val_losses.append(val_loss)


    def train(self):

        train_progress = tqdm(
            range(1, self.config['epochs'] + 1),
            leave=True,
            desc="Training..."
        )

        for epoch in train_progress:
            self.model.train()
            train_progress.set_description(f"EPOCH {epoch} / {self.config['epochs']} | training...")
            self.train_one_epoch(epoch)
            self.clear()

            self.model.eval()
            train_progress.set_description(f"EPOCH {epoch} / {self.config['epochs']} | validating...")
            self.valid_one_epoch(epoch)
            self.clear()

            print(f"{'➖️' * 10} EPOCH {epoch} / {self.config['epochs']} {'➖️' * 10}")
            print(f"train loss: {self.train_losses[-1]}")
            print(f"valid loss: {self.val_losses[-1]}\n\n")

            if self.val_losses[-1] == min(self.val_losses):
                # torch.save(self.model.state_dict(), f"{self.config['save_path']}/best_model.pth")
                self.model.save(self.config['save_path'])


    def clear(self):
        gc.collect()
        torch.cuda.empty_cache()