import copy
import logging
from typing import Any, Callable, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MultimodalDistillationLoss(nn.Module):
    """
    Distillation objective combining ground-truth token cross-entropy with
    temperature-scaled Kullback-Leibler divergence over teacher output distributions:

        $\mathcal{L} = (1 - \alpha) \mathcal{L}_{\text{CE}}(\hat{Y}_{\text{student}}, Y)
                       + \alpha T^2 \mathcal{D}_{\text{KL}}\left(\text{Softmax}\left(\frac{Z_{\text{s}}}{T}\right)
                       \,\middle\Vert{}\, \text{Softmax}\left(\frac{Z_{\text{t}}}{T}\right)\right)$
    """
    def __init__(self, alpha: float = 0.5, temperature: float = 2.0, ignore_index: int = -100):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ignore_index = ignore_index

    def forward(
        self,
        student_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        loss = torch.tensor(0.0, device=student_logits.device, dtype=student_logits.dtype)

        # 1. Task Ground-Truth Cross Entropy
        if labels is not None:
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.ignore_index,
            )
        else:
            ce_loss = torch.tensor(0.0, device=student_logits.device, dtype=student_logits.dtype)

        if teacher_logits is None or self.alpha <= 0.0:
            return ce_loss

        # 2. Distillation KL Divergence
        # Apply mask to calculate KL only on non-ignored response tokens to save memory
        t = self.temperature
        if labels is not None:
            mask = (labels[..., 1:] != self.ignore_index).view(-1)
            s_flat = student_logits[..., :-1, :].contiguous().view(-1, student_logits.size(-1))
            t_flat = teacher_logits[..., :-1, :].contiguous().view(-1, teacher_logits.size(-1))

            if mask.any():
                s_selected = s_flat[mask]
                t_selected = t_flat[mask]
            else:
                s_selected = s_flat
                t_selected = t_flat
        else:
            s_selected = student_logits.view(-1, student_logits.size(-1))
            t_selected = teacher_logits.view(-1, teacher_logits.size(-1))

        # Log-Softmax for student, Softmax for teacher
        log_prob_student = F.log_softmax(s_selected / t, dim=-1)
        prob_teacher = F.softmax(t_selected / t, dim=-1)

        kl_loss = F.kl_div(log_prob_student, prob_teacher, reduction="batchmean") * (t * t)

        return (1.0 - self.alpha) * ce_loss + self.alpha * kl_loss


def run_recovery_epochs(
    student_model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader],
    device: torch.device,
    teacher_model: Optional[nn.Module] = None,
    epochs: int = 3,
    lr: float = 2e-5,
    weight_decay: float = 1e-2,
    alpha: float = 0.5,
    temperature: float = 2.0,
    eval_fn: Optional[Callable[[nn.Module], float]] = None,
    target_metric: Optional[float] = None,
    safety_floor: Optional[float] = None,
) -> Tuple[float, Dict[str, torch.Tensor]]:
    """
    Executes micro-recovery distillation on the pruned student model using AdamW
    and Cosine Annealing learning rate schedule.
    """
    trainable_params = [p for p in student_model.parameters() if p.requires_grad]
    if not trainable_params:
        logger.warning("No trainable parameters found in student model for recovery.")
        return 0.0, student_model.state_dict()

    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = MultimodalDistillationLoss(alpha=alpha, temperature=temperature)

    if teacher_model is not None:
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False

    best_metric = -float("inf")
    best_state = copy.deepcopy(student_model.state_dict())

    for epoch in range(epochs):
        student_model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Recovery Epoch {epoch + 1}/{epochs}")

        for batch in pbar:
            if isinstance(batch, (list, tuple)):
                batch_dict = {"input_ids": batch[0], "labels": batch[1]}
            else:
                batch_dict = batch

            batch_device = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch_dict.items()
            }

            optimizer.zero_grad()

            student_outputs = student_model(**batch_device)
            student_logits = (
                student_outputs.logits if hasattr(student_outputs, "logits") else student_outputs
            )

            teacher_logits = None
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_outputs = teacher_model(**batch_device)
                    teacher_logits = (
                        teacher_outputs.logits
                        if hasattr(teacher_outputs, "logits")
                        else teacher_outputs
                    )

            labels = batch_device.get("labels")
            loss = criterion(student_logits, labels=labels, teacher_logits=teacher_logits)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{running_loss / (pbar.n + 1):.4f}"})

        scheduler.step()

        # Validation evaluation pass
        if eval_fn is not None:
            metric_val = eval_fn(student_model)
            logger.info(f"Epoch {epoch + 1} validation metric: {metric_val:.4f}")

            if metric_val > best_metric:
                best_metric = metric_val
                best_state = copy.deepcopy(student_model.state_dict())

            if target_metric is not None and metric_val >= target_metric:
                logger.info(f"Target metric {target_metric} reached. Early stopping.")
                break

    return best_metric, best_state
