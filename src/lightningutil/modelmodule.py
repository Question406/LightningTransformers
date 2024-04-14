import torch
import pickle
from typing import Any, Dict
import pytorch_lightning as L
from transformers import AutoModelForCausalLM
from transformers.models.llama import LlamaForCausalLM
from transformers.optimization import get_scheduler
import torch.nn as nn
import torch.nn.functional as F

def get_dtype(data_type):
    if data_type == 'bfloat16':
        return torch.bfloat16
    elif data_type == 'float16':
        return torch.float16

class LightningTransformerModule(L.LightningModule):
    def __init__(
        self, 
        data_type = 'bfloat16', # Use bfloat16 as the default training
        learning_rate=1e-5,
        lr_scheduler_type="linear",
        weight_decay=0.0,
        warmup_ratio=0.1,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.data_type = data_type
        self.model = None
        self.warmup_steps = None
        self.num_training_steps = None

    def forward(self, input_ids, attention_mask, labels=None, position_ids=None, **kwargs):
        raise NotImplementedError()
    
    def shared_step(self, input_ids, attention_mask, labels=None, position_ids=None, **kwargs):
        raise NotImplementedError()
    
    def log_losses(self, loss, stage='train'):
        if stage == 'train':
            self.log(f"{stage}_loss", loss.item(), on_step=True, prog_bar=True)
        else:
            self.log(f"{stage}_loss", loss.item(), on_step=False, on_epoch=True, sync_dist=True, prog_bar=True, add_dataloader_idx=False)

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss = self.shared_step(input_ids, attention_mask, labels)
        self.log_losses(loss, 'train')
        self.log(f"lr", 
                 self.lr_schedulers().get_last_lr()[0], 
                 prog_bar=True,
                 sync_dist=False)
        return loss
       
    def validation_step(self, batch, batch_idx, dataloader_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        loss = self.shared_step(input_ids, attention_mask, labels)
        val_name = f'val_forget' if dataloader_idx == 0 else 'val_retain'
        self.log_losses(loss, val_name)

    def test_step(self, *args: Any, **kwargs: Any):
        return self.validation_step(*args, **kwargs)

    def on_before_configure_optimizers(self, num_training_steps):
        self.warmup_steps = int(num_training_steps * self.hparams.warmup_ratio)
        self.num_training_steps = num_training_steps

    def configure_optimizers(self):
        assert self.warmup_steps is not None, "Call on_before_configure_optimizers before configure_optimizers"

        no_decay = ["bias", "LayerNorm.weight", "norm"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.student.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.student.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate)
        scheduler = get_scheduler(
            name=self.hparams.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.num_training_steps,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "name": "learningrate"}]


class LightningLoraTransformerModule(LightningTransformerModule):
    def __init__(
        self, 
        data_type='bfloat16', 
        learning_rate=0.00001, 
        lr_scheduler_type="linear", 
        weight_decay=0, 
        warmup_ratio=0.1
    ):
        super().__init__(data_type, learning_rate, lr_scheduler_type, weight_decay, warmup_ratio)
    
    