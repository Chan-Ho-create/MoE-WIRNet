import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.dataset_utils import PromptTrainDataset, PromptValDataset
from net.model_soft_hard_same import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint


# ===============================================================
# 极简稳定版 CoBa：只依赖每个 epoch 的验证集 loss
# ===============================================================
class CoBaStatus:
    """
    极简稳定版 CoBa：基于验证集 loss 下降斜率的多任务动态加权器
    ---------------------------------------------------------
    输入：每个 epoch 的平均验证 loss（按任务）
    输出：每个任务的权重向量（和为 1）
    ---------------------------------------------------------
    保留指标：
      - RCS: Relative Convergence Speed（相对收敛速度）
      - ACS: Absolute Convergence Speed（绝对收敛速度）
      - DF : Divergence Factor（全局发散系数）
    """

    def __init__(
        self,
        num_tasks=4,
        history_length=10,
        tau=5,
        minimum_weight=0.1,
        device="cpu",
    ):
        self.num_tasks = num_tasks
        self.history_length = history_length   # 斜率窗口长度（最近多少个 epoch）
        self.tau = tau                         # DF 的 softmax 温度
        self.minimum_weight = minimum_weight
        self.device = device

        # 验证集 loss 历史：shape = [num_tasks, T]
        self.history_valid_loss = None

        # 每任务斜率历史：shape = [num_tasks, K]
        self.per_task_slope_history = None

        # 全局最坏任务的斜率历史：shape = [K]
        self.total_slope_history = None

    # -------------------------------------------------------------
    # 记录每个 epoch 的：每任务平均验证 loss
    # -------------------------------------------------------------
    def update_valid_loss(self, valid_loss_per_task: torch.Tensor):
        """
        valid_loss_per_task: shape = [num_tasks]
        """
        valid_loss_per_task = valid_loss_per_task.detach().to(
            self.device, dtype=torch.float64
        )

        if self.history_valid_loss is None:
            self.history_valid_loss = valid_loss_per_task.unsqueeze(1)
        else:
            self.history_valid_loss = torch.cat(
                [self.history_valid_loss, valid_loss_per_task.unsqueeze(1)], dim=1
            )

    # -------------------------------------------------------------
    # 在一个序列上拟合线性斜率： y = a x + b 中的 a
    # -------------------------------------------------------------
    def fit_slope(self, y: torch.Tensor) -> torch.Tensor:
        """
        y: shape = [L]
        """
        L = y.shape[0]
        if L < 2:
            return torch.tensor(0.0, device=self.device, dtype=torch.float64)

        x = torch.arange(L, device=self.device, dtype=torch.float64)

        X = torch.stack([x, torch.ones_like(x)], dim=1)  # [L,2]
        A = X.T @ X
        b = X.T @ y
        try:
            w = torch.linalg.solve(A, b)
            slope = w[0]
        except RuntimeError:
            return torch.tensor(0.0, device=self.device, dtype=torch.float64)

        return slope.clamp(-1e3, 1e3)

    # -------------------------------------------------------------
    # 计算任务权重
    # -------------------------------------------------------------
    def compute_task_weight(self):
        """
        返回：
           per_task_weight: shape = [num_tasks]
           metrics: dict(RCS, ACS, DF)
        """
        assert self.history_valid_loss is not None, "CoBa: 没有验证 loss 历史，无法计算权重"

        # ---------------------------------------------------------
        # （1）准备斜率拟合窗口：取最近 W 个 epoch
        # ---------------------------------------------------------
        loss_hist = self.history_valid_loss           # [num_tasks, T]
        T = loss_hist.shape[1]
        W = min(self.history_length, T)

        loss_window = loss_hist[:, -W:]              # [num_tasks, W]

        # ---------------------------------------------------------
        # （2）每个任务拟合斜率
        # ---------------------------------------------------------
        slopes = torch.zeros(self.num_tasks, dtype=torch.float64, device=self.device)
        for i in range(self.num_tasks):
            slopes[i] = self.fit_slope(loss_window[i])

        # 记录任务斜率历史
        if self.per_task_slope_history is None:
            self.per_task_slope_history = slopes.unsqueeze(1)   # [num_tasks,1]
        else:
            self.per_task_slope_history = torch.cat(
                [self.per_task_slope_history, slopes.unsqueeze(1)], dim=1
            )

        # ---------------------------------------------------------
        # （3）整体最坏任务的 loss 曲线的斜率
        # ---------------------------------------------------------
        max_loss = loss_window.max(dim=0).values      # [W]
        total_slope = self.fit_slope(max_loss)

        if self.total_slope_history is None:
            self.total_slope_history = total_slope.unsqueeze(0)  # [1]
        else:
            self.total_slope_history = torch.cat(
                [self.total_slope_history, total_slope.unsqueeze(0)], dim=0
            )

        # ---------------------------------------------------------
        # （4）RCS：相对收敛速度（基于当前 slopes）
        #     slope 越大（下降慢 / 上升），RCS 越大 → 权重偏大
        # ---------------------------------------------------------
        denom = slopes.abs().sum() + 1e-8
        rcs_logits = self.num_tasks * slopes / denom
        RCS = F.softmax(rcs_logits, dim=-1)

        # ---------------------------------------------------------
        # （5）ACS：绝对收敛速度（基于斜率历史）
        #     用最近 K 个斜率时间点
        # ---------------------------------------------------------
        S = self.per_task_slope_history.shape[1]
        K = min(self.history_length, S)
        slope_window = self.per_task_slope_history[:, -K:]   # [num_tasks, K]

        denom2 = slope_window.abs().sum(dim=1, keepdim=True) + 1e-8
        acs_logits = -K * slope_window / denom2         # 越下降多 → logits 越大
        ACS = F.softmax(acs_logits[:, -1], dim=-1)      # 取最近一次

        # ---------------------------------------------------------
        # （6）DF：发散因子，整体最坏任务是否在发散
        # ---------------------------------------------------------
        total_K = min(self.history_length, self.total_slope_history.shape[0])
        total_window = self.total_slope_history[-total_K:]    # [total_K]

        denom3 = total_window.abs().sum() + 1e-8
        df_logits = -total_K * total_window / denom3          # 越下降多 → logits 越大
        DF = F.softmax(df_logits * self.tau, dim=-1)[-1]      # 当前时刻对应的 DF

        # ---------------------------------------------------------
        # （7）融合 RCS / ACS（用 DF 控制偏向）
        # ---------------------------------------------------------
        weight_logits = DF * RCS + (1.0 - DF) * ACS
        weight = F.softmax(weight_logits * self.num_tasks, dim=-1)

        # 下限保障（避免任务被饿死）
        weight = weight * (1.0 - self.minimum_weight * self.num_tasks)
        weight += self.minimum_weight

        return weight, {"RCS": RCS.detach(), "ACS": ACS.detach(), "DF": DF.detach()}


# ===============================================================
# Lightning 模型定义
# ===============================================================
class PromptIRModel(pl.LightningModule):
    def __init__(
        self,
        num_tasks=4,
        coba_history_length=10,
        coba_tau=5,
        coba_min_weight=0.01,
        coba_warmup_epochs=20,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["net"])  # 可选：保存超参到 ckpt

        self.num_tasks = num_tasks
        self.net = PromptIR()

        # 每个任务一个 L1 loss
        self.task_loss_fns = nn.ModuleDict(
            {str(i): nn.L1Loss() for i in range(num_tasks)}
        )

        # 初始均匀权重
        self.task_weights = {str(i): 1.0 / num_tasks for i in range(num_tasks)}

        # CoBa 控制器（只管“如何根据验证 loss 动态调权重”）
        self.coba = CoBaStatus(
            num_tasks=num_tasks,
            history_length=coba_history_length,
            tau=coba_tau,
            minimum_weight=coba_min_weight,
            device="cpu",  # 训练时会在 on_validation_epoch_end 里同步为 model.device
        )
        self.coba_warmup_epochs = coba_warmup_epochs  # 前多少个 epoch 不改权重

    # ---------------------- 前向 ----------------------
    def forward(self, x):
        return self.net(x)

    # ---------------------- 训练 step ----------------------
    def training_step(self, batch, batch_idx):
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)
        total_loss, _ = self._compute_multi_task_loss(restored, clean_patch, de_id)
        self.log("train_total_loss", total_loss, prog_bar=True, sync_dist=True)
        return total_loss

    def _compute_multi_task_loss(self, restored, clean_patch, de_id):
        """
        restored     : [B,C,H,W]
        clean_patch  : [B,C,H,W]
        de_id        : [B]，每个样本对应任务 id（0~num_tasks-1）
        """
        total_loss = 0.0
        batch_size = restored.shape[0]

        for task_id in range(self.num_tasks):
            mask = (de_id == task_id)
            if mask.sum() > 0:
                loss = self.task_loss_fns[str(task_id)](
                    restored[mask], clean_patch[mask]
                )
                # 按“任务权重 × 该任务样本比例”加权
                total_loss = total_loss + self.task_weights[str(task_id)] * loss * (
                    mask.sum() / batch_size
                )

        return total_loss, None

    # ---------------------- 验证 step ----------------------
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        dataloader_idx: 0: drs, 1: drd, 2: nrs, 3: nrd
        """
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        # 用 MSE 计算 PSNR
        mse = F.mse_loss(restored, clean_patch)
        psnr = 10 * torch.log10(1.0 / (mse + 1e-8))

        # 用 L1 作为验证 loss（给 CoBa 用）
        l1 = F.l1_loss(restored, clean_patch)

        # 记录 per-dataloader 的 PSNR / L1
        self.log(f"val_psnr_{dataloader_idx}", psnr,
                 prog_bar=True, sync_dist=True, on_epoch=True, on_step=False)
        self.log(f"val_l1_{dataloader_idx}", l1,
                 prog_bar=False, sync_dist=True, on_epoch=True, on_step=False)

        return {"psnr": psnr, "l1": l1}

    # ---------------------- 验证结束：汇总 PSNR + 驱动 CoBa ----------------------
    def on_validation_epoch_end(self):
        all_metrics = self.trainer.callback_metrics

        # ---- 汇总 4 个验证集的 PSNR，log val_avg_psnr（给 checkpoint 用）----
        psnr_items = {
            k: v for k, v in all_metrics.items() if k.startswith("val_psnr_")
        }
        if psnr_items:
            avg_psnr = torch.stack(list(psnr_items.values())).mean()
            self.log("val_avg_psnr", avg_psnr,
                     prog_bar=True, sync_dist=True, on_epoch=True)

        # ---- 从 val_l1_0..3 中取出 4 个任务的验证 loss（平均值） ----
        l1_keys = [f"val_l1_{i}" for i in range(self.num_tasks)]
        if not all(k in all_metrics for k in l1_keys):
            return  # 还没完整 4 个验证集，不更新 CoBa

        valid_loss_per_task = torch.stack([all_metrics[k] for k in l1_keys])  # [num_tasks]

        # 更新 CoBa 的 device（和模型保持一致）
        self.coba.device = self.device

        # 把这一 epoch 的验证 loss 喂给 CoBa
        self.coba.update_valid_loss(valid_loss_per_task)

        # ---- warmup：前几个 epoch 不改任务权重 ----
        if self.current_epoch < self.coba_warmup_epochs:
            return

        # 至少要两轮验证才能拟合斜率
        if self.coba.history_valid_loss.shape[1] < 10:
            return

        # ---- 计算新的任务权重 ----
        per_task_weight, metrics = self.coba.compute_task_weight()

        # 更新训练中的权重（转成 float）
        for i in range(self.num_tasks):
            self.task_weights[str(i)] = float(per_task_weight[i].item())

        # 日志：CoBa 权重 + 指标
        weight_dict = {f"coba_weight_{i}": per_task_weight[i].item()
                       for i in range(self.num_tasks)}
        rcs_dict = {f"coba_rcs_{i}": metrics["RCS"][i].item()
                    for i in range(self.num_tasks)}
        acs_dict = {f"coba_acs_{i}": metrics["ACS"][i].item()
                    for i in range(self.num_tasks)}
        df_dict = {"coba_df": metrics["DF"].item()}

        self.log_dict(weight_dict, sync_dist=True)
        self.log_dict(rcs_dict, sync_dist=True)
        self.log_dict(acs_dict, sync_dist=True)
        self.log_dict(df_dict,  sync_dist=True)

        # 控制台打印，方便你观察
        print(f"\n[CoBa] Epoch {self.current_epoch}")
        print(f"  RCS: {[round(x, 4) for x in metrics['RCS'].tolist()]}")
        print(f"  ACS: {[round(x, 4) for x in metrics['ACS'].tolist()]}")
        print(f"  DF : {round(metrics['DF'].item(), 4)}")
        print(f"  Updated Task Weights: {[round(x, 4) for x in per_task_weight.tolist()]}\n")

    # ---------------------- 优化器 & 学习率调度 ----------------------
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=3, max_epochs=opt.epochs
        )
        return [optimizer], [scheduler]


# ===============================================================
# 训练 + 验证主函数
# ===============================================================
def main():
    print("Options")
    print(opt)

    # ---------- 日志 ----------
    if opt.wblogger is not None:
        logger = WandbLogger(project=opt.wblogger, name="Coba")
    else:
        logger = TensorBoardLogger(save_dir="logs/")

    # ---------- 数据 ----------
    trainset = PromptTrainDataset(opt)
    valsets = {name: PromptValDataset(opt, name)
               for name in ['drs', 'drd', 'nrs', 'nrd']}

    trainloader = DataLoader(
        trainset,
        batch_size=opt.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=opt.num_workers,
    )
    # 多个验证 dataloader：顺序对应任务 0,1,2,3
    valloaders = [
        DataLoader(v, batch_size=1, num_workers=1)
        for v in valsets.values()
    ]

    # ---------- 检查点 ----------
    checkpoint_callback = ModelCheckpoint(
        dirpath=opt.ckpt_dir,
        filename="epoch{epoch:03d}-avgPSNR{val_avg_psnr:.2f}",
        monitor="val_avg_psnr",    # 监控平均 PSNR
        mode="max",                # 保存最高 PSNR
        save_top_k=3,
        every_n_epochs=opt.val_interval,
        save_last=True
    )

    # ---------- 模型 ----------
    model = PromptIRModel(
        num_tasks=4,
        coba_history_length=10,
        coba_tau=5,
        coba_min_weight=0.1,
        coba_warmup_epochs=20,
    )

    # ---------- Trainer ----------
    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        precision="16-mixed",
        accelerator="gpu",
        devices=opt.num_gpus,
        strategy="auto",
        logger=logger,
        callbacks=[checkpoint_callback],
        val_check_interval=None,                 # 每个 epoch 后验证
        check_val_every_n_epoch=opt.val_interval # 每隔 N 个 epoch 验证一次
    )

    # ---------- 开始训练 ----------
    trainer.fit(model=model,
                train_dataloaders=trainloader,
                val_dataloaders=valloaders)


if __name__ == '__main__':
    main()
