import re
import logging
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from transformers import PaliGemmaProcessor

logger = logging.getLogger(__name__)


class SegmentationEvaluator:
    """
    Evaluates spatial segmentation and grounding performance in PaliGemma 2.
    Parses specialized segmentation tokens (<seg000>-<seg063>) and location
    tokens (<loc0000>-<loc1023>), computing mean Intersection-over-Union (mIoU).
    """
    def __init__(self, processor: PaliGemmaProcessor, mask_resolution: int = 64):
        self.processor = processor
        self.mask_resolution = mask_resolution
        self.seg_token_pattern = re.compile(r"<seg(\d{3})>")
        self.loc_token_pattern = re.compile(r"<loc(\d{4})>")

    def extract_seg_indices(self, generated_text: str) -> List[int]:
        """Extracts integer indices from <segXXX> tokens in generated text."""
        matches = self.seg_token_pattern.findall(generated_text)
        return [int(m) for m in matches]

    def extract_loc_coordinates(self, generated_text: str) -> List[float]:
        """Extracts normalized coordinates [0.0, 1.0] from <locXXXX> tokens."""
        matches = self.loc_token_pattern.findall(generated_text)
        return [int(m) / 1024.0 for m in matches]

    def compute_iou(
        self,
        pred_mask: Union[torch.Tensor, np.ndarray],
        gt_mask: Union[torch.Tensor, np.ndarray],
        threshold: float = 0.5,
    ) -> float:
        """
        Calculates Intersection over Union between binary masks:
            IoU = |pred ∩ gt| / |pred ∪ gt|
        """
        if isinstance(pred_mask, torch.Tensor):
            pred_mask = pred_mask.detach().cpu().numpy()
        if isinstance(gt_mask, torch.Tensor):
            gt_mask = gt_mask.detach().cpu().numpy()

        p_binary = (pred_mask > threshold).astype(bool)
        g_binary = (gt_mask > threshold).astype(bool)

        intersection = np.logical_and(p_binary, g_binary).sum()
        union = np.logical_or(p_binary, g_binary).sum()

        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        return float(intersection / union)

    def evaluate_batch_predictions(
        self,
        pred_texts: List[str],
        gt_texts: List[str],
        gt_masks: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """
        Computes exact sequence accuracy, token extraction adherence,
        and mIoU across a validation batch.
        """
        total = len(pred_texts)
        exact_matches = 0
        valid_seg_count = 0
        total_iou = 0.0

        for i in range(total):
            p_text = pred_texts[i].strip()
            g_text = gt_texts[i].strip()

            if p_text == g_text:
                exact_matches += 1

            p_indices = self.extract_seg_indices(p_text)
            g_indices = self.extract_seg_indices(g_text)

            # Check whether target segmentation tokens are present
            if len(p_indices) > 0 and len(g_indices) > 0:
                valid_seg_count += 1
                # Token-level intersection over union across codebook indices
                set_p = set(p_indices)
                set_g = set(g_indices)
                inter = len(set_p.intersection(set_g))
                uni = len(set_p.union(set_g))
                token_iou = inter / uni if uni > 0 else 0.0
                total_iou += token_iou

        seq_acc = exact_matches / max(total, 1)
        mean_iou = total_iou / max(valid_seg_count, 1)
        valid_rate = valid_seg_count / max(total, 1)

        return {
            "sequence_accuracy": seq_acc,
            "mean_iou": mean_iou,
            "valid_token_rate": valid_rate,
            "samples_evaluated": total,
        }
