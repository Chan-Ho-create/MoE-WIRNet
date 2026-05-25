import torch
import torch.nn.functional as F

class CoBaStatus:
    """Convergence Balancer (CoBa-RCS-only): 只使用 RCS 指标的版本"""

    def __init__(
        self,
        coba_warmup_steps=3,
        coba_history_length=3,
        coba_tau=5,
        coba_update_interval=1,
        coba_sample_valid_num=1,
        valid_dataloader=None,
        num_tasks=4,
        device="cuda",
    ):
        self.num_tasks = num_tasks
        self.device = device

        # 参数配置
        self.coba_warmup_steps = coba_warmup_steps
        self.coba_history_length = coba_history_length
        self.coba_tau = coba_tau
        self.coba_update_interval = coba_update_interval
        self.coba_sample_valid_num = coba_sample_valid_num

        # 验证集加载器（dict: {"drs": DataLoader, ...}）
        self.valid_dataloader = valid_dataloader if isinstance(valid_dataloader, dict) else {"default": valid_dataloader}

        # 历史统计信息
        self.valid_task_loss_accumulated = torch.zeros(num_tasks, dtype=torch.float64)
        self.history_task_valid_loss = None
        self.minimum_weight = 1 / (num_tasks * 10)
        self.valid_task_loss_begining = torch.ones(num_tasks, dtype=torch.float64)

    # =========================================================
    # 🔹 验证损失采样（不变）
    # =========================================================
    def sample_valid_batch(self, model, loss_fn, completed_steps):
        if self.valid_dataloader is None:
            return

        device = next(model.parameters()).device
        total_task_loss = torch.zeros(self.num_tasks, dtype=torch.float64, device=device)
        count = 0

        model.eval()
        with torch.no_grad():
            for val_name, vloader in self.valid_dataloader.items():
                if vloader is None:
                    continue
                for v_batch in vloader:
                    ([clean_name, de_id], degrad_patch, clean_patch) = v_batch
                    degrad_patch, clean_patch, de_id = (
                        degrad_patch.to(device),
                        clean_patch.to(device),
                        de_id.to(device),
                    )
                    restored = model(degrad_patch)
                    task_losses = torch.zeros(self.num_tasks, dtype=torch.float64, device=device)

                    for i in range(self.num_tasks):
                        mask = (de_id == i)
                        if mask.sum() > 0:
                            task_loss = loss_fn[str(i)](restored[mask], clean_patch[mask])
                            task_losses[i] = task_loss.detach()
                    total_task_loss += task_losses
                    count += 1

        model.train()
        if count > 0:
            self.valid_task_loss_accumulated = total_task_loss / count

        if self.history_task_valid_loss is None:
            self.history_task_valid_loss = self.valid_task_loss_accumulated.unsqueeze(1)
        else:
            self.history_task_valid_loss = torch.cat(
                (self.history_task_valid_loss, self.valid_task_loss_accumulated.unsqueeze(1)), dim=-1
            )

    # =========================================================
    # 🔹 只使用 RCS 计算任务权重
    # =========================================================
    def compute_per_task_weight(self, completed_steps=None):
        EPS = 1e-8
        task_num = self.num_tasks
        device = self.device

        # ---------- 计算 RCS ----------
        task_slope_fitting = torch.ones(task_num, dtype=torch.float64, device=device)
        start_step = max(0, completed_steps - self.coba_history_length)
        history_steps = torch.arange(start_step, completed_steps, 1, device=device, dtype=torch.float64)

        for i in range(task_num):
            per_task_history_valid_loss = self.history_task_valid_loss[i][-len(history_steps):]
            task_slope_fitting[i] = self.fit_window_slope(history_steps, per_task_history_valid_loss)

        # ✅ RCS = 任务收敛速度的归一化 Softmax
        normalize_task_slope = task_num * task_slope_fitting / (task_slope_fitting.abs().sum() + EPS)
        rcs = F.softmax(normalize_task_slope, dim=-1)

        # ✅ 直接用 RCS 作为任务权重
        per_task_weight = rcs.clone()

        # ---- 防止 NaN 传播 ----
        if torch.isnan(per_task_weight).any():
            per_task_weight = torch.full_like(per_task_weight, 1.0 / task_num)

        if len((per_task_weight < self.minimum_weight).nonzero().squeeze(0)) > 0:
            per_task_weight = per_task_weight * (1 - self.minimum_weight * task_num)
            per_task_weight += self.minimum_weight

        metrics = {
            "RCS": rcs.detach(),
            "ACS": torch.zeros_like(rcs),  # 为兼容日志
            "DF": torch.tensor(0.0, device=device),
        }
        return per_task_weight, metrics

    # =========================================================
    # 🔹 拟合斜率（不变）
    # =========================================================
    def fit_window_slope(self, x, y):
        EPS = 1e-8
        device = self.device

        y = y[y != 0]
        x = x[:len(y)]
        if len(y) < 2:
            return torch.tensor(0.0, device=device, dtype=torch.float64)

        x = x.to(device=device, dtype=torch.float64)
        y = y.to(device=device, dtype=torch.float64)
        X = torch.stack((x, torch.ones_like(x, device=device, dtype=torch.float64))).T
        ws = torch.flip(torch.arange(1, len(y) + 1, device=device, dtype=torch.float64), dims=[0])

        A = X.T @ (ws[:, None] * X)
        b = X.T @ (ws * y)
        try:
            w = torch.linalg.solve(A + EPS * torch.eye(2, device=device, dtype=torch.float64), b)
        except RuntimeError:
            return torch.tensor(0.0, device=device, dtype=torch.float64)
        slope = w[0].clamp(min=-1e3, max=1e3)
        return slope
