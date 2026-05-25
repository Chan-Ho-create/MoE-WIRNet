import torch
import torch.nn.functional as F

class CoBaStatus:
    """Convergence Balancer (CoBa): 多任务动态加权控制器（支持多验证集版本）"""

    def __init__(
        self,
        coba_warmup_steps=10,
        coba_history_length=10,
        coba_tau=5,
        coba_update_interval=1,
        coba_sample_valid_num=1,
        valid_dataloader=None,  # ✅ 接受 dict 格式的多验证集
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

        # 验证集加载器（dict: {"drs": DataLoader, "drd": ...}）
        self.valid_dataloader = valid_dataloader if isinstance(valid_dataloader, dict) else {"default": valid_dataloader}

        # 历史统计信息
        self.valid_task_loss_accumulated = torch.zeros(num_tasks, dtype=torch.float64)
        self.history_task_valid_loss = None
        self.per_task_slope_list = None
        self.total_slope_list = None

        # 防止任务被饿死
        self.minimum_weight = 1 / (num_tasks * 10)
        self.valid_task_loss_begining = torch.ones(num_tasks, dtype=torch.float64)

    # =========================================================
    # 🔹 计算每个任务在整个验证集上的平均损失
    # =========================================================
    def sample_valid_batch(self, model, loss_fn, completed_steps):
        """使用整个验证集计算每个任务的平均验证损失（支持多验证集）"""
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

                    # 各任务分别计算
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

        # ---------------- 历史记录 ----------------
        if self.history_task_valid_loss is None or not torch.is_tensor(self.history_task_valid_loss):
            # ✅ 初始化：第 0 轮或 warmup 阶段的第一次
            self.history_task_valid_loss = self.valid_task_loss_accumulated.unsqueeze(1)
        else:
            # ✅ 后续 epoch 直接拼接
            self.history_task_valid_loss = torch.cat(
                (self.history_task_valid_loss, self.valid_task_loss_accumulated.unsqueeze(1)), dim=-1
            )


    # =========================================================
    # 🔹 根据历史趋势计算任务权重
    # =========================================================
    def compute_per_task_weight(self, completed_steps=None):
        """根据历史验证损失趋势计算每任务权重"""
        EPS = 1e-8
        task_num = self.num_tasks
        device = self.device

        task_slope_fitting = torch.ones(task_num, dtype=torch.float64, device=device)

        # 构造历史窗口
        start_step = max(0, completed_steps // self.coba_update_interval - self.coba_history_length)
        history_steps = torch.arange(start_step, completed_steps, 1, device=device, dtype=torch.float64)

        # 拟合每个任务的斜率
        for i in range(task_num):
            per_task_history_valid_loss = self.history_task_valid_loss[i][-len(history_steps):]
            task_slope_fitting[i] = self.fit_window_slope(history_steps, per_task_history_valid_loss)

        # 拟合整体趋势
        history_total_valid_loss, _ = torch.max(
            self.history_task_valid_loss[:, -len(history_steps):], dim=0
        )
        total_slope_fitting = self.fit_window_slope(history_steps, history_total_valid_loss)

        # 更新历史记录
        if completed_steps == self.coba_warmup_steps:
            self.per_task_slope_list = task_slope_fitting.unsqueeze(1)
            self.total_slope_list = total_slope_fitting.unsqueeze(0)
        else:
            self.per_task_slope_list = torch.cat(
                (self.per_task_slope_list, task_slope_fitting.unsqueeze(1)), dim=-1
            )
            self.total_slope_list = torch.cat(
                (self.total_slope_list, total_slope_fitting.unsqueeze(0)), dim=0
            )

        # ---------------------- RCS ----------------------
        normalize_task_slope = task_num * task_slope_fitting / (task_slope_fitting.abs().sum() + EPS)
        rcs = F.softmax(normalize_task_slope, dim=-1)

        # ---------------------- ACS ----------------------
        history_per_task_slope_list = self.per_task_slope_list[:, start_step:]
        reverse_norm_iter_slope = -len(history_per_task_slope_list[0]) * history_per_task_slope_list \
                                  / (history_per_task_slope_list.abs().sum(dim=-1, keepdim=True) + EPS)
        current_step_rn_slope = reverse_norm_iter_slope[:, -1]
        acs = F.softmax(current_step_rn_slope, dim=-1)

        # ---------------------- DF ----------------------
        normalize_total_iter_slope = -len(self.total_slope_list) * self.total_slope_list \
                                     / (self.total_slope_list.abs().sum() + EPS)
        divergence_factor = F.softmax(normalize_total_iter_slope * self.coba_tau, dim=-1)[-1] \
                            * len(self.total_slope_list)

        # ---- 防止 NaN 传播 ----
        if torch.isnan(rcs).any():
            rcs = torch.full_like(rcs, 1.0 / task_num)
        if torch.isnan(acs).any():
            acs = torch.full_like(acs, 1.0 / task_num)
        if torch.isnan(divergence_factor):
            divergence_factor = torch.tensor(0.5, device=device)

        # ---------------------- 权重合成 ----------------------
        weight_logits = divergence_factor * rcs + (1 - divergence_factor) * acs
        per_task_weight = F.softmax(weight_logits * task_num, dim=-1)

        # ✅ 限制最小值
        if torch.isnan(per_task_weight).any():
            per_task_weight = torch.full_like(per_task_weight, 1.0 / task_num)
        if len((per_task_weight < self.minimum_weight).nonzero().squeeze(0)) > 0:
            per_task_weight = per_task_weight * (1 - self.minimum_weight * task_num)
            per_task_weight += self.minimum_weight

        # 记录中间指标
        metrics = {
            "RCS": rcs.detach(),
            "ACS": acs.detach(),
            "DF": divergence_factor.detach(),
        }
        return per_task_weight, metrics

    # =========================================================
    # 🔹 斜率拟合（带权线性回归）
    # =========================================================
    def fit_window_slope(self, x, y):
        """在时间窗口中拟合线性斜率（GPU安全 + 稳定）"""
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
