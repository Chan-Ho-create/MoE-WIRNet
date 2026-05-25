import subprocess
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from utils.dataset_utils import PromptTrainDataset,PromptValDataset
from net.model_moe import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from torchvision.transforms import ToTensor
from PIL import Image
import os
from math import log10

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


class PromptIRModel(pl.LightningModule):
    def __init__(self, num_tasks=3):
        super().__init__()
        self.num_tasks = num_tasks
        self.net = PromptIR()
        self.task_loss_fns = nn.ModuleDict({str(i): nn.L1Loss() for i in range(num_tasks)})
        self.task_weights = {str(i): 0.25 for i in range(num_tasks)}

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        total_loss, task_losses = self._compute_multi_task_loss(restored, clean_patch, de_id)
        self.log("train_total_loss", total_loss, prog_bar=True)
        return total_loss

    def _compute_multi_task_loss(self, restored, clean_patch, de_id):
        total_loss = 0.0
        batch_size = restored.shape[0]
        for task_id in range(self.num_tasks):
            mask = (de_id == task_id)
            if mask.sum() > 0:
                loss = self.task_loss_fns[str(task_id)](restored[mask], clean_patch[mask])
                total_loss += self.task_weights[str(task_id)] * loss * (mask.sum() / batch_size)
        return total_loss, None

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        with torch.no_grad():
            restored = self.net(degrad_patch)
        mse = nn.functional.mse_loss(restored, clean_patch)
        psnr = 10 * torch.log10(1.0 / mse)
        self.log(f"val_psnr_{dataloader_idx}", psnr, prog_bar=True, sync_dist=True)
        return psnr

    def on_validation_epoch_end(self):
        all_metrics = self.trainer.callback_metrics
        psnr_items = {k: v for k, v in all_metrics.items() if k.startswith("val_psnr_")}
        if psnr_items:
            avg_psnr = torch.stack(list(psnr_items.values())).mean()
            self.log("val_avg_psnr", avg_psnr, prog_bar=True, sync_dist=True)
            print("\n──────── Validation ────────")
            for k, v in sorted(psnr_items.items()):
                print(f"{k:<15}: {v.item():.4f} dB")
            print(f"Average PSNR     : {avg_psnr.item():.4f} dB")
            print("────────────────────────────\n")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=3, max_epochs=opt.epochs)
        return [optimizer], [scheduler]


def main():
    print("Options")
    print(opt)

    if opt.wblogger is not None:
        logger = WandbLogger(project=opt.wblogger, name="PromptIR-Train")
    else:
        logger = TensorBoardLogger(save_dir="logs/")


    trainset = PromptTrainDataset(opt)
    valsets = {name: PromptValDataset(opt, name) for name in ['HAZE', 'RAIN', 'SNOW']}

    trainloader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True,
                             pin_memory=True, drop_last=True, num_workers=opt.num_workers)
    valloaders = [DataLoader(v, batch_size=1, num_workers=2) for v in valsets.values()]

    checkpoint_callback = ModelCheckpoint(
        dirpath=opt.ckpt_dir,
        filename="epoch{epoch:03d}-avgPSNR{val_avg_psnr:.2f}",
        monitor="val_avg_psnr",  
        mode="max",              
        save_top_k=3,
        every_n_epochs=opt.val_interval,
        save_last=True
    )

    model = PromptIRModel(num_tasks=3)

    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        precision="16-mixed",
        accelerator="gpu",
        devices=opt.num_gpus,
        strategy="auto",
        logger=logger,
        callbacks=[checkpoint_callback],
        val_check_interval=None,
        check_val_every_n_epoch=opt.val_interval,
    )

    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=valloaders)


if __name__ == '__main__':
    main()
