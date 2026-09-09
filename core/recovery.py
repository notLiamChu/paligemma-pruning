import copy
import os
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MultimodalDistillationLoss(nn.Module):
    """
    Combined task cross-entropy and soft-target KL-divergence distillation loss.
    Masks both objectives to focus strictly on target response tokens (where labels != ignore_index).
    """
    def __init__(
        self,
        alpha: float = 0.5,
        temperature: float = 2.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(
        self,
        student_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        teacher_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = student_logits.device

        # Autoregressive next-token prediction alignment
        shift_s_logits = student_logits[..., :-1, :].contiguous()
        num_classes = shift_s_logits.size(-1)

        loss_ce = torch.tensor(0.0, device=device)
        mask = None

        if labels is not None:
            shift_labels = labels[..., 1:].contiguous().to(device)
            loss_ce = self.ce_loss(
                shift_s_logits.view(-1, num_classes),
                shift_labels.view(-1),
            )
            mask = (shift_labels != self.ignore_index)

        if teacher_logits is None or self.alpha == 0.0:
            return loss_ce

        shift_t_logits = teacher_logits[..., :-1, :].contiguous().to(device)

        # Filter by active label mask if available to avoid computing KL over padding/prompts
        if mask is not None and mask.any():
            s_active = shift_s_logits[mask]
            t_active = shift_t_logits[mask]
        else:
            s_active = shift_s_logits.view(-1, num_classes)
            t_active = shift_t_logits.view(-1, num_classes)

        # Temperature-scaled probability distributions
        s_log_probs = F.log_softmax(s_active / self.temperature, dim=-1)
        t_probs = F.softmax(t_active / self.temperature, dim=-1)

        loss_kl = self.kl_loss(s_log_probs, t_probs) * (self.temperature ** 2)

        if labels is None:
            return loss_kl

        return (1.0 - self.alpha) * loss_ce + self.alpha * loss_kl


def adapt_teacher_to_task(
    teacher_model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 3,
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    freeze_language_decoder: bool = True,
) -> nn.Module:
    """
    Warms up the teacher model on downstream RefCOCO samples before AGOP calculation.
    Optionally freezes language decoder parameters to prevent GPU VRAM exhaustion on 16/24 GB cards.
    """
    teacher_model.train()

    if freeze_language_decoder and hasattr(teacher_model, "language_model"):
        logger.info("Freezing language decoder during teacher adaptation to preserve VRAM.")
        for p in teacher_model.language_model.parameters():
            p.requires_grad = False

    trainable_params = [p for p in teacher_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    logger.info(f"Starting teacher task adaptation ({epochs} epochs, {len(trainable_params)} active param tensors)...")
    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Teacher Warmup Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            batch_device = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }
            optimizer.zero_grad()
            outputs = teacher_model(**batch_device)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{running_loss / (pbar.n + 1):.4f}"})

    teacher_model.eval()
    return teacher_model


def cache_teacher_logits(
    teacher_model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    cache_path: Optional[str] = None,
) -> List[torch.Tensor]:
    """
    Precomputes teacher logits across calibration batches and stores them in CPU memory or on disk.
    If the specified cache_path already exists on disk, it is loaded directly to save computation time.
    """
    if cache_path is not None and os.path.exists(cache_path):
        logger.info(f"Loading pre-existing teacher logit cache from: {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    teacher_model.eval()
    cached: List[torch.Tensor] = []
    logger.info("Caching offline teacher logits across training batches...")

    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Teacher Logit Caching"):
            batch_device = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }
            out = teacher_model(**batch_device)
            cached.append(out.logits.detach().cpu())

    if cache_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        torch.save(cached, cache_path)
        logger.info(f"Teacher logits cached to disk: {cache_path}")

    return cached


def run_recovery_epochs(
    student_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    teacher_model: Optional[nn.Module] = None,
    cached_teacher_logits: Optional[List[torch.Tensor]] = None,
    epochs: int = 3,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    alpha: float = 0.5,
    temperature: float = 2.0,
    eval_fn: Optional[Callable[[nn.Module], float]] = None,
    target_metric: Optional[float] = None,
) -> Tuple[float, Dict[str, Any]]:
    """
    Runs AdamW micro-recovery distillation on student model parameters.
    Can distill online from an in-memory teacher or offline from cached logits.
    """
    student_model.train()
    trainable_params = [p for p in student_model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = MultimodalDistillationLoss(alpha=alpha, temperature=temperature)

    best_metric = float("-inf")
    best_state = copy.deepcopy(student_model.state_dict())

    logger.info(f"Starting recovery distillation ({epochs} epochs, lr={lr})...")

    for epoch in range(epochs):
        student_model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Recovery Epoch {epoch + 1}/{epochs}")

        for batch_idx, batch in enumerate(pbar):
            batch_device = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
                if isinstance(v, torch.Tensor)
            }

            optimizer.zero_grad()
            outputs = student_model(**batch_device)
            student_logits = outputs.logits if hasattr(outputs, "logits") else outputs

            teacher_logits = None
            if cached_teacher_logits is not None:
                teacher_logits = cached_teacher_logits[batch_idx].to(device, non_blocking=True)
            elif teacher_model is not None:
                with torch.no_grad():
                    t_out = teacher_model(**batch_device)
                    teacher_logits = t_out.logits

            loss = criterion(
                student_logits=student_logits,
                labels=batch_device.get("labels"),
                teacher_logits=teacher_logits,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{running_loss / (pbar.n + 1):.4f}"})

        scheduler.step()

        if eval_fn is not None:
            student_model.eval()
            metric_val = eval_fn(student_model)
            logger.info(f"Epoch {epoch + 1} validation metric: {metric_val:.4f}")
            if metric_val > best_metric:
                best_metric = metric_val
                best_state = copy.deepcopy(student_model.state_dict())

            if target_metric is not None and metric_val >= target_metric:
                logger.info(f"Target metric threshold {target_metric} reached. Early stopping.")
                break

    if eval_fn is None:
        best_state = copy.deepcopy(student_model.state_dict())

    return best_metric, best_state
