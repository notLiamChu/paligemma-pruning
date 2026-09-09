import os
import json
import logging
from typing import Dict, Any, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, PaliGemmaProcessor

from models.loader import resolve_torch_dtype

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "architecture_manifest.json"
WEIGHTS_FILENAME = "sliced_model_state.pt"


def extract_layer_dimensions(model: nn.Module) -> Dict[str, Any]:
    """
    Extracts the current physical channel dimensions of all MLP blocks
    in both the SigLIP vision tower and Gemma 2 language decoder.
    """
    vision_layers = model.vision_tower.vision_model.encoder.layers
    vision_dims = [layer.mlp.fc1.out_features for layer in vision_layers]

    lang_layers = model.language_model.model.layers
    lang_dims = [layer.mlp.gate_proj.out_features for layer in lang_layers]

    return {
        "vision_intermediate_sizes": vision_dims,
        "language_intermediate_sizes": lang_dims,
        "vision_hidden_size": vision_layers[0].mlp.fc1.in_features,
        "language_hidden_size": lang_layers[0].mlp.gate_proj.in_features,
        "vision_num_layers": len(vision_layers),
        "language_num_layers": len(lang_layers),
    }


def save_sliced_paligemma(
    model: nn.Module,
    save_dir: str,
    processor: Optional[PaliGemmaProcessor] = None,
    base_model_id: str = "google/paligemma2-3b-pt-224",
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Serializes a physically pruned PaliGemma model with heterogeneous layer sizes.

    Saves:
      1. architecture_manifest.json: records per-layer sliced dimensions.
      2. sliced_model_state.pt: state dict containing trimmed weight tensors.
      3. Processor/tokenizer assets (if processor is provided).
    """
    os.makedirs(save_dir, exist_ok=True)
    metadata = metadata or {}

    # 1. Build and save architectural manifest
    manifest = {
        "base_model_id": base_model_id,
        "metadata": metadata,
        **extract_layer_dimensions(model),
    }

    manifest_path = os.path.join(save_dir, MANIFEST_FILENAME)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved architecture manifest to: {manifest_path}")

    # 2. Save model state dictionary
    weights_path = os.path.join(save_dir, WEIGHTS_FILENAME)
    logger.info(f"Saving sliced model weights to: {weights_path}...")
    torch.save(model.state_dict(), weights_path)

    # 3. Save processor configuration if provided
    if processor is not None:
        processor.save_pretrained(save_dir)
        logger.info(f"Saved processor artifacts to: {save_dir}")

    return save_dir


def load_sliced_paligemma(
    checkpoint_dir: str,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Union[str, torch.dtype] = torch.bfloat16,
    hf_token: Optional[str] = None,
    eval_mode: bool = True,
) -> Tuple[PaliGemmaForConditionalGeneration, PaliGemmaProcessor, Dict[str, Any]]:
    """
    Reconstructs and loads a physically pruned PaliGemma model from disk.

    Steps:
      1. Reads architecture_manifest.json.
      2. Instantiates base PaliGemma architecture.
      3. Reconstructs non-uniform MLP layers to match sliced channel dimensions.
      4. Injects trimmed state dict weights into the reconstructed modules.
    """
    manifest_path = os.path.join(checkpoint_dir, MANIFEST_FILENAME)
    weights_path = os.path.join(checkpoint_dir, WEIGHTS_FILENAME)

    if not os.path.exists(manifest_path) or not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Checkpoint directory {checkpoint_dir} must contain both '{MANIFEST_FILENAME}' "
            f"and '{WEIGHTS_FILENAME}'."
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    base_model_id = manifest["base_model_id"]
    target_dtype = resolve_torch_dtype(dtype)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    logger.info(f"Loading processor for base model: {base_model_id}")
    try:
        processor = AutoProcessor.from_pretrained(checkpoint_dir)
    except Exception:
        processor = AutoProcessor.from_pretrained(base_model_id, token=token)

    logger.info(f"Instantiating skeleton model from: {base_model_id}...")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=target_dtype,
        token=token,
        low_cpu_mem_usage=True,
    )

    # Reconstruct non-uniform layers to match the manifest before loading state dict
    _reconstruct_sliced_layers(model, manifest, target_dtype)

    logger.info(f"Loading sliced weights from: {weights_path}...")
    state_dict = torch.load(weights_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        logger.warning(f"State dict discrepancies: missing={missing}, unexpected={unexpected}")

    model.to(target_device)
    if eval_mode:
        model.eval()

    return model, processor, manifest


def _reconstruct_sliced_layers(
    model: nn.Module,
    manifest: Dict[str, Any],
    dtype: torch.dtype,
) -> None:
    """
    Reconstructs the linear projection modules of SigLIP and Gemma 2 to match
    heterogeneous channel dimensions recorded in the manifest.
    """
    # 1. Reconstruct SigLIP Vision Tower MLPs
    vis_layers = model.vision_tower.vision_model.encoder.layers
    for i, target_k in enumerate(manifest["vision_intermediate_sizes"]):
        current_k = vis_layers[i].mlp.fc1.out_features
        if target_k != current_k:
            d_model = vis_layers[i].mlp.fc1.in_features
            has_bias1 = vis_layers[i].mlp.fc1.bias is not None
            has_bias2 = vis_layers[i].mlp.fc2.bias is not None

            vis_layers[i].mlp.fc1 = nn.Linear(d_model, target_k, bias=has_bias1, dtype=dtype)
            vis_layers[i].mlp.fc2 = nn.Linear(target_k, d_model, bias=has_bias2, dtype=dtype)
            logger.debug(f"Reconstructed SigLIP Layer {i} MLP: {current_k} -> {target_k}")

    # 2. Reconstruct Gemma 2 Language Decoder SwiGLU MLPs
    lang_layers = model.language_model.model.layers
    for i, target_k in enumerate(manifest["language_intermediate_sizes"]):
        current_k = lang_layers[i].mlp.gate_proj.out_features
        if target_k != current_k:
            d_model = lang_layers[i].mlp.gate_proj.in_features
            has_bias_gate = lang_layers[i].mlp.gate_proj.bias is not None
            has_bias_up = lang_layers[i].mlp.up_proj.bias is not None
            has_bias_down = lang_layers[i].mlp.down_proj.bias is not None

            lang_layers[i].mlp.gate_proj = nn.Linear(
                d_model, target_k, bias=has_bias_gate, dtype=dtype
            )
            lang_layers[i].mlp.up_proj = nn.Linear(
                d_model, target_k, bias=has_bias_up, dtype=dtype
            )
            lang_layers[i].mlp.down_proj = nn.Linear(
                target_k, d_model, bias=has_bias_down, dtype=dtype
            )
            logger.debug(f"Reconstructed Gemma 2 Layer {i} SwiGLU: {current_k} -> {target_k}")
