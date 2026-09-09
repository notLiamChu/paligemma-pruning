import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TargetLayerSpec:
    """
    Metadata specification for an AGOP target module within PaliGemma.
    Identifies whether the module belongs to SigLIP or Gemma 2 SwiGLU.
    """
    def __init__(self, name: str, module_type: str, layer_idx: int, hook_module: nn.Module, parent_block: nn.Module):
        self.name = name
        self.module_type = module_type  # 'siglip' or 'gemma'
        self.layer_idx = layer_idx
        self.hook_module = hook_module
        self.parent_block = parent_block

    def __repr__(self) -> str:
        return f"TargetLayerSpec(name='{self.name}', type='{self.module_type}', layer={self.layer_idx})"


def get_agop_targets(
    model: nn.Module,
    target_components: str = "both",
    target_layers: Optional[List[int]] = None,
) -> List[TargetLayerSpec]:
    """
    Discovers pruneable MLP layers in PaliGemma 2 and identifies the precise
    submodule to hook for channel sensitivity extraction.

    For SigLIP (standard MLP):
      - Hooks the input to `fc2` (equivalent to post-activation of `fc1`).
      - Channel dimension: `fc2.in_features` == `fc1.out_features`.

    For Gemma 2 (SwiGLU):
      - Hooks the input to `down_proj` (the Hadamard product: act(gate(x)) * up(x)).
      - Channel dimension: `down_proj.in_features` == `gate_proj.out_features`.

    Args:
        model: Loaded PaliGemma model.
        target_components: 'vision', 'language', or 'both'.
        target_layers: Optional explicit list of integer layer indices.
    """
    specs: List[TargetLayerSpec] = []

    # 1. Inspect SigLIP Vision Tower
    if target_components in ("vision", "both"):
        if hasattr(model, "vision_tower") and hasattr(model.vision_tower, "vision_model"):
            vis_layers = model.vision_tower.vision_model.encoder.layers
            for i, layer in enumerate(vis_layers):
                if target_layers is not None and i not in target_layers:
                    continue
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "fc2"):
                    specs.append(
                        TargetLayerSpec(
                            name=f"vision.layers.{i}.mlp",
                            module_type="siglip",
                            layer_idx=i,
                            hook_module=layer.mlp.fc2,
                            parent_block=layer,
                        )
                    )

    # 2. Inspect Gemma 2 Language Decoder
    if target_components in ("language", "both"):
        if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
            lang_layers = model.language_model.model.layers
            for i, layer in enumerate(lang_layers):
                if target_layers is not None and i not in target_layers:
                    continue
                if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
                    specs.append(
                        TargetLayerSpec(
                            name=f"language.layers.{i}.mlp",
                            module_type="gemma",
                            layer_idx=i,
                            hook_module=layer.mlp.down_proj,
                            parent_block=layer,
                        )
                    )

    return specs


def compute_sample_score(
    outputs: Any,
    batch: Dict[str, torch.Tensor],
    score_mode: str = "nll_loss",
) -> torch.Tensor:
    """
    Computes a scalar sensitivity score $s_i$ for gradient extraction.

    Modes:
      - 'nll_loss': Negative log-likelihood (negative task cross-entropy).
                    Maximizing sensitivity aligns with preserving ground truth tokens.
      - 'target_mean_logit': Average unnormalized logit over target response tokens.
      - 'last_token_logit': Maximum logit of the final generated token.
    """
    if score_mode == "nll_loss":
        if hasattr(outputs, "loss") and outputs.loss is not None:
            return -outputs.loss
        
        # Calculate cross entropy manually if loss not precomputed
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        labels = batch.get("labels")
        if labels is None:
            raise ValueError("score_mode='nll_loss' requires 'labels' in batch dictionary.")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        return -loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    elif score_mode == "target_mean_logit":
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        labels = batch.get("labels")
        if labels is not None:
            mask = (labels != -100)
            if mask.any():
                return logits[mask].mean()
        return logits.mean()

    elif score_mode == "last_token_logit":
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        return logits[:, -1, :].max(dim=-1).values.sum()

    else:
        raise ValueError(f"Unsupported score_mode: '{score_mode}'")


def compute_agop_for_layer(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    target_spec: TargetLayerSpec,
    device: torch.device,
    num_samples: int = 64,
    score_mode: str = "nll_loss",
    custom_score_fn: Optional[Callable] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Computes the channel-space AGOP covariance matrix for a single PaliGemma layer:
        $G = \mathbb{E}[g g^\top]$
    where $g \in \mathbb{R}^{d_{ff}}$ is the sequence-averaged gradient of the task score
    with respect to intermediate channel activations.

    Outer product accumulation is handled in float64 on CPU to conserve GPU memory.
    """
    model.eval()

    # Ensure parameters in target block require gradients for backpropagation
    for param in target_spec.parent_block.parameters():
        param.requires_grad = True

    activations: Dict[str, torch.Tensor] = {}
    hook_handle = None

    # Forward hook captures input tensor to contraction projection (fc2 or down_proj)
    def hook_fn(module, args, output):
        # args[0] is the intermediate activation [Batch, Sequence, d_ff]
        activations["H"] = args[0]

    hook_handle = target_spec.hook_module.register_forward_hook(hook_fn)

    agop: Optional[torch.Tensor] = None
    processed_samples = 0

    try:
        pbar = tqdm(total=num_samples, desc=f"AGOP [{target_spec.name}]")

        for batch in data_loader:
            if processed_samples >= num_samples:
                break

            # Handle both dict batches and tuple batches
            if isinstance(batch, (list, tuple)):
                inputs, labels = batch[0], batch[1]
                batch_dict = {"input_ids": inputs, "labels": labels}
            else:
                batch_dict = batch

            # Move tensors to device
            batch_device = {
                k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch_dict.items()
            }

            activations.clear()
            model.zero_grad(set_to_none=True)

            # Forward pass
            outputs = model(**batch_device)
            H = activations.get("H")

            if H is None:
                raise RuntimeError(
                    f"Hook did not capture activations for {target_spec.name}. "
                    "Ensure target module is executed during forward pass."
                )

            # Initialize double-precision accumulation matrix on CPU
            if agop is None:
                d_ff = H.shape[-1]
                agop = torch.zeros((d_ff, d_ff), dtype=torch.float64, device="cpu")

            # Determine scalar score
            if custom_score_fn is not None:
                score = custom_score_fn(outputs, batch_device)
            else:
                score = compute_sample_score(outputs, batch_device, score_mode=score_mode)

            # Compute sensitivity gradient with respect to intermediate activations H
            grad_H = torch.autograd.grad(
                outputs=score,
                inputs=H,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]  # Shape: [Batch, Sequence, d_ff]

            batch_size = H.size(0)

            # Process each sample's gradient trajectory
            for b in range(batch_size):
                if processed_samples >= num_samples:
                    break

                # Mean-pool across sequence length to yield channel sensitivity vector g
                # Shape: [Sequence, d_ff] -> [d_ff]
                sample_grad = grad_H[b]
                g = sample_grad.mean(dim=0).detach().to(torch.float64).cpu()

                # Accumulate outer product rank-1 update
                agop += torch.outer(g, g)
                processed_samples += 1
                pbar.update(1)

            del outputs, H, grad_H, score
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        pbar.close()

    finally:
        if hook_handle is not None:
            hook_handle.remove()

    if agop is None or processed_samples == 0:
        raise RuntimeError(f"No samples were processed for AGOP extraction on {target_spec.name}.")

    # Normalize by sample count
    agop /= max(processed_samples, 1)

    # Enforce numerical symmetry: G = 0.5 * (G + G^T)
    agop = 0.5 * (agop + agop.T)

    # Compute sorted eigenvalues and clamp negative precision artifacts
    eigvals = torch.linalg.eigvalsh(agop)
    eigvals = torch.flip(eigvals, dims=[0]).clamp_min(0.0)

    return agop, eigvals, processed_samples


def estimate_agop_layers(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    target_specs: List[TargetLayerSpec],
    device: torch.device,
    num_samples: int = 64,
    score_mode: str = "nll_loss",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, int]]:
    """
    Iterates sequentially through target specs to extract AGOP matrices and eigenspectra.
    """
    agop_matrices: Dict[str, torch.Tensor] = {}
    eigenvalues: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}

    for spec in target_specs:
        logger.info(f"Extracting AGOP covariance for {spec.name}...")
        g_mat, vals, n = compute_agop_for_layer(
            model=model,
            data_loader=data_loader,
            target_spec=spec,
            device=device,
            num_samples=num_samples,
            score_mode=score_mode,
        )
        agop_matrices[spec.name] = g_mat
        eigenvalues[spec.name] = vals
        counts[spec.name] = n

    return agop_matrices, eigenvalues, counts
