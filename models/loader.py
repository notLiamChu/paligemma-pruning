import os
import logging
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import (
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
)

logger = logging.getLogger(__name__)


def resolve_torch_dtype(dtype_str: Union[str, torch.dtype]) -> torch.dtype:
    """Converts a string representation into a valid torch.dtype."""
    if isinstance(dtype_str, torch.dtype):
        return dtype_str

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    normalized = dtype_str.lower().strip()
    if normalized not in dtype_map:
        raise ValueError(
            f"Unsupported dtype '{dtype_str}'. Supported options: {list(dtype_map.keys())}"
        )
    return dtype_map[normalized]


def load_paligemma_model(
    model_id: str = "google/paligemma2-3b-pt-224",
    dtype: Union[str, torch.dtype] = torch.bfloat16,
    device: Optional[Union[str, torch.device]] = None,
    hf_token: Optional[str] = None,
    eval_mode: bool = True,
) -> Tuple[PaliGemmaForConditionalGeneration, PaliGemmaProcessor]:
    """
    Loads PaliGemma / PaliGemma 2 from Hugging Face with specified precision and device allocation.

    Args:
        model_id: Hugging Face hub repository ID (e.g., 'google/paligemma2-3b-pt-224').
        dtype: Computation precision ('bfloat16', 'float16', 'float32').
        device: Target device ('cuda', 'cuda:0', 'cpu', or None for auto-detection).
        hf_token: Hugging Face access token (PaliGemma models require accepting Google's license).
        eval_mode: If True, sets model to eval mode and freezes parameters by default.

    Returns:
        A tuple containing (model, processor).
    """
    target_dtype = resolve_torch_dtype(dtype)
    token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    logger.info(f"Loading processor for: {model_id}")
    processor = AutoProcessor.from_pretrained(model_id, token=token)

    logger.info(f"Loading PaliGemma model ({target_dtype}) onto {target_device}...")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=target_dtype,
        token=token,
        low_cpu_mem_usage=True,
    )

    model.to(target_device)

    if eval_mode:
        model.eval()

    # Verify structural anchors for AGOP surgery
    _verify_architecture_integrity(model)

    return model, processor


def _verify_architecture_integrity(model: nn.Module) -> None:
    """Verifies that expected vision encoder and decoder submodules exist."""
    has_vision = hasattr(model, "vision_tower") and hasattr(
        model.vision_tower, "vision_model"
    )
    has_lang = hasattr(model, "language_model") and hasattr(
        model.language_model, "model"
    )

    if not has_vision or not has_lang:
        raise AttributeError(
            "Model architecture does not match expected PaliGemma structure with "
            "`vision_tower.vision_model` and `language_model.model`."
        )
