import logging
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def find_k_from_energy_threshold(eigvals: torch.Tensor, energy_threshold: float = 0.90) -> int:
    """
    Finds the smallest subspace dimension $k$ that captures at least the specified
    cumulative spectral energy ratio $\tau$:
        $k = \min \left\{ m : \frac{\sum_{i=1}^m \lambda_i}{\sum_{j=1}^d \lambda_j} \ge \tau \right\}$
    """
    total_energy = eigvals.sum()
    if total_energy <= 0.0:
        return max(1, int(len(eigvals) * (1.0 - energy_threshold)))

    cum_energy = torch.cumsum(eigvals / total_energy, dim=0)
    cutoff = torch.nonzero(cum_energy >= energy_threshold)
    if cutoff.numel() > 0:
        return cutoff[0].item() + 1
    return len(eigvals)


def extract_eigen_coordinates(
    agop_mat: torch.Tensor,
    k: int,
    sigma: float = 0.95,
) -> torch.Tensor:
    """
    Extracts channel indices via AGOP Principal Subspace Coordinate Energy.
    Identifies a minimal coordinate support set preserving a cumulative energy ratio $\sigma$
    across each of the top-$k$ dominant eigenvectors.
    """
    orig_dim = agop_mat.shape[0]
    vals, vecs = torch.linalg.eigh(agop_mat)

    # Top-k eigenvectors (columns corresponding to largest eigenvalues)
    top_k_vecs = vecs[:, -k:]

    union_mask = torch.zeros(orig_dim, dtype=torch.bool, device=agop_mat.device)

    for i in range(k):
        v_i = top_k_vecs[:, i]
        energy_i = v_i ** 2

        sorted_energies, sorted_indices = torch.sort(energy_i, descending=True)
        cum_energy = torch.cumsum(sorted_energies, dim=0)

        cutoff = torch.nonzero(cum_energy >= sigma)
        num_to_keep = cutoff[0].item() + 1 if cutoff.numel() > 0 else orig_dim

        keep_indices = sorted_indices[:num_to_keep]
        union_mask[keep_indices] = True

    selected_indices = torch.nonzero(union_mask).squeeze().view(-1)
    selected_indices, _ = torch.sort(selected_indices)

    if selected_indices.numel() == 0:
        logger.warning("Eigen coordinate threshold selected 0 channels. Retaining dominant coordinate.")
        dominant_idx = torch.argmax(torch.abs(top_k_vecs[:, -1]))
        selected_indices = dominant_idx.unsqueeze(0)

    return selected_indices


def slice_siglip_mlp(
    mlp_module: nn.Module,
    indices: torch.Tensor,
) -> nn.Module:
    """
    Physically slices a SigLIP 2-layer MLP along intermediate channel dimension $d_{ff} \to k$:
      - fc1: [d_model, d_ff] -> [d_model, k]  (weight matrix: [k, d_model])
      - fc2: [d_ff, d_model] -> [k, d_model]  (weight matrix: [d_model, k])
    """
    fc1: nn.Linear = mlp_module.fc1
    fc2: nn.Linear = mlp_module.fc2

    device = fc1.weight.device
    dtype = fc1.weight.dtype
    indices = indices.to(device)
    k = indices.numel()
    d_model = fc1.in_features

    # 1. Slice fc1: Linear(d_model, k)
    has_bias1 = fc1.bias is not None
    new_fc1 = nn.Linear(d_model, k, bias=has_bias1, dtype=dtype, device=device)
    with torch.no_grad():
        new_fc1.weight.copy_(fc1.weight.data[indices, :])
        if has_bias1:
            new_fc1.bias.copy_(fc1.bias.data[indices])

    # 2. Slice fc2: Linear(k, d_model)
    has_bias2 = fc2.bias is not None
    new_fc2 = nn.Linear(k, d_model, bias=has_bias2, dtype=dtype, device=device)
    with torch.no_grad():
        new_fc2.weight.copy_(fc2.weight.data[:, indices])
        if has_bias2:
            new_fc2.bias.copy_(fc2.bias.data)  # fc2 bias outputs to d_model; shape unchanged

    mlp_module.fc1 = new_fc1
    mlp_module.fc2 = new_fc2
    return mlp_module


def slice_gemma_swiglu(
    mlp_module: nn.Module,
    indices: torch.Tensor,
) -> nn.Module:
    """
    Physically slices a Gemma 2 SwiGLU block across its three synchronized projections:
      - gate_proj: [d_model, d_ff] -> [d_model, k]  (weight: [k, d_model])
      - up_proj:   [d_model, d_ff] -> [d_model, k]  (weight: [k, d_model])
      - down_proj: [d_ff, d_model] -> [k, d_model]  (weight: [d_model, k])
    """
    gate: nn.Linear = mlp_module.gate_proj
    up: nn.Linear = mlp_module.up_proj
    down: nn.Linear = mlp_module.down_proj

    device = gate.weight.device
    dtype = gate.weight.dtype
    indices = indices.to(device)
    k = indices.numel()
    d_model = gate.in_features

    has_bias_gate = gate.bias is not None
    has_bias_up = up.bias is not None
    has_bias_down = down.bias is not None

    new_gate = nn.Linear(d_model, k, bias=has_bias_gate, dtype=dtype, device=device)
    new_up = nn.Linear(d_model, k, bias=has_bias_up, dtype=dtype, device=device)
    new_down = nn.Linear(k, d_model, bias=has_bias_down, dtype=dtype, device=device)

    with torch.no_grad():
        new_gate.weight.copy_(gate.weight.data[indices, :])
        if has_bias_gate:
            new_gate.bias.copy_(gate.bias.data[indices])

        new_up.weight.copy_(up.weight.data[indices, :])
        if has_bias_up:
            new_up.bias.copy_(up.bias.data[indices])

        new_down.weight.copy_(down.weight.data[:, indices])
        if has_bias_down:
            new_down.bias.copy_(down.bias.data)

    mlp_module.gate_proj = new_gate
    mlp_module.up_proj = new_up
    mlp_module.down_proj = new_down
    return mlp_module


def slice_generic_linear_pair(
    block: nn.Module,
    indices: torch.Tensor,
) -> nn.Module:
    """
    Fallback slicer for synthetic test blocks or generic architectures
    utilizing 'linear1' and 'linear2'.
    """
    linear1: nn.Linear = block.linear1
    linear2: nn.Linear = block.linear2

    device = linear1.weight.device
    dtype = linear1.weight.dtype
    indices = indices.to(device)
    k = indices.numel()
    d_model = linear1.in_features

    has_b1 = linear1.bias is not None
    has_b2 = linear2.bias is not None

    new_l1 = nn.Linear(d_model, k, bias=has_b1, dtype=dtype, device=device)
    new_l2 = nn.Linear(k, d_model, bias=has_b2, dtype=dtype, device=device)

    with torch.no_grad():
        new_l1.weight.copy_(linear1.weight.data[indices, :])
        if has_b1:
            new_l1.bias.copy_(linear1.bias.data[indices])

        new_l2.weight.copy_(linear2.weight.data[:, indices])
        if has_b2:
            new_l2.bias.copy_(linear2.bias.data)

    block.linear1 = new_l1
    block.linear2 = new_l2
    return block


def apply_physical_slicing(
    layer_block: nn.Module,
    indices: torch.Tensor,
) -> nn.Module:
    """
    Dispatches physical slicing to the appropriate module architecture:
    SigLIP (mlp.fc1/fc2), Gemma 2 (mlp.gate_proj/up_proj/down_proj), or generic.
    """
    # Direct inspection if the block itself is an MLP module
    if hasattr(layer_block, "gate_proj") and hasattr(layer_block, "down_proj"):
        slice_gemma_swiglu(layer_block, indices)
        return layer_block
    if hasattr(layer_block, "fc1") and hasattr(layer_block, "fc2"):
        slice_siglip_mlp(layer_block, indices)
        return layer_block

    # Inspection if the block is a parent layer containing an .mlp submodule
    if hasattr(layer_block, "mlp"):
        mlp = layer_block.mlp
        if hasattr(mlp, "gate_proj") and hasattr(mlp, "down_proj"):
            slice_gemma_swiglu(mlp, indices)
            return layer_block
        elif hasattr(mlp, "fc1") and hasattr(mlp, "fc2"):
            slice_siglip_mlp(mlp, indices)
            return layer_block

    if hasattr(layer_block, "linear1") and hasattr(layer_block, "linear2"):
        slice_generic_linear_pair(layer_block, indices)
        return layer_block

    raise AttributeError(
        f"Unable to identify pruneable MLP structure in block of type {type(layer_block)}."
    )


def perform_agop_eigen_surgery(
    layer_block: nn.Module,
    agop_mat: torch.Tensor,
    k: int,
    sigma: float = 0.95,
) -> Tuple[nn.Module, torch.Tensor]:
    """
    Extracts coordinate support from top-$k$ eigenvectors matching cumulative energy $\sigma$,
    then slices the layer block in-place.
    """
    indices = extract_eigen_coordinates(agop_mat, k=k, sigma=sigma)
    orig_dim = agop_mat.shape[0]

    if indices.numel() == orig_dim:
        logger.info(f"Selected channels match original dimension ({orig_dim}). Skipping physical slicing.")
        return layer_block, indices

    apply_physical_slicing(layer_block, indices)
    logger.info(f"Eigen surgery complete: {orig_dim} -> {indices.numel()} channels (sigma={sigma}).")
    return layer_block, indices
