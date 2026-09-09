import os
import copy
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
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
        # 1. Ground-Truth Cross-Entropy Loss
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

        log_prob_student = F.log_softmax(s_selected / t, dim=-1)
        prob_teacher = F.softmax(t_selected.to(s_selected.device) / t, dim=-1)

        kl_loss = F.kl_div(log_prob_student, prob_teacher, reduction="batchmean") * (t * t)
        return (1.0 - self.alpha) * ce_loss + self.alpha * kl_loss


def adapt_teacher_to_task(
    teacher_model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int = 3,
    lr: float = 5e-5,
    weight_decay: float = 0.01,
) -> nn.Module:
    """
    Performs rapid preliminary task adaptation (warmup) on the teacher model
    using O(100) calibration samples. Aligns teacher representations with downstream
    task syntax before computing AGOP covariance or generating distillation targets.
    """
    logger.info(f"Adapting teacher model to downstream task ({epochs} warmup epochs, lr={lr})...")
    teacher_model.train()
    for param in teacher_model.parameters():
        param.requires_grad = True

    optimizer = AdamW(teacher_model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Teacher Warmup Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            batch_device = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            optimizer.zero_grad()
            outputs = teacher_model(**batch_device)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            labels = batch_device.get("labels")

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(teacher_model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{running_loss / (pbar.n + 1):.4f}"})

    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    logger.info("Teacher task adaptation complete.")
    return teacher_model


def cache_teacher_logits(
    teacher_model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    cache_path: Optional[str] = None,
) -> List[torch.Tensor]:
    """
    Computes teacher output distributions across the training batches under torch.no_grad()
    and offloads them to CPU RAM (or disk). Enables eviction of the full teacher model
    from GPU VRAM prior to student distillation recovery.
    """
    logger.info("Generating offline teacher logit cache on CPU...")
    teacher_model.eval()
    cached_logits: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Caching Teacher Logits"):
            batch_device = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            outputs = teacher_model(**batch_device)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            # Offload to CPU in half precision to conserve memory
            cached_logits.append(logits.detach().to(torch.bfloat16).cpu())

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        torch.save(cached_logits, cache_path)
        logger.info(f"Serialized {len(cached_logits)} cached teacher batches to: {cache_path}")

    return cached_logits


def run_recovery_epochs(
    student_model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader],
    device: torch.device,
    teacher_model: Optional[nn.Module] = None,
    cached_teacher_logits: Optional[List[torch.Tensor]] = None,
    epochs: int = 3,
    lr: float = 2e-5,
    weight_decay: float = 1e-2,
    alpha: float = 0.5,
    temperature: float = 2.0,
    eval_fn: Optional[Callable[[nn.Module], float]] = None,
    target_metric: Optional[float] = None,
) -> Tuple[float, Dict[str, torch.Tensor]]:
    """
    Executes micro-recovery distillation on the pruned student model using AdamW
    and Cosine Annealing. Supports distillation against an offline CPU logit cache
    (0 GB teacher GPU overhead) or a live teacher module.
    """
    trainable_params = [p for p in student_model.parameters() if p.requires_grad]
    if not trainable_params:
        logger.warning("No trainable parameters identified in student model for recovery.")
        return 0.0, student_model.state_dict()

    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95))
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = MultimodalDistillationLoss(alpha=alpha, temperature=temperature)

    best_metric = -float("inf")
    best_state = copy.deepcopy(student_model.state_dict())

    for epoch in range(epochs):
        student_model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Recovery Epoch {epoch + 1}/{epochs}")

        for batch_idx, batch in enumerate(pbar):
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

            # Retrieve teacher logits (from CPU offline cache or live forward pass)
            teacher_logits = None
            if cached_teacher_logits is not None and batch_idx < len(cached_teacher_logits):
                teacher_logits = cached_teacher_logits[batch_idx].to(device, non_blocking=True)
            elif teacher_model is not None:
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

        if eval_fn is not None:
            metric_val = eval_fn(student_model)
            logger.info(f"Epoch {epoch + 1} validation metric: {metric_val:.4f}")
            if metric_val > best_metric:
                best_metric = metric_val
                best_state = copy.deepcopy(student_model.state_dict())
    
            if target_metric is not None and metric_val >= target_metric:
                logger.info(f"Target metric threshold {target_metric} reached. Early stopping.")
                break
    
    # Ensure that without an explicit eval_fn, the trained student weights are preserved
    if eval_fn is None:
        best_state = copy.deepcopy(student_model.state_dict())
    
    return best_metric, best_state
