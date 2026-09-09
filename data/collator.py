import logging
from typing import Any, Dict, List, Optional, Union
import torch
from PIL import Image
from transformers import PaliGemmaProcessor

logger = logging.getLogger(__name__)


class PaliGemmaDataCollator:
    """
    Multimodal batch collator for PaliGemma 2.
    Preprocesses paired image-text instances, formats autoregressive targets,
    and applies label masking (-100) over prompt tokens so loss is computed
    strictly on the response sequence.
    """
    def __init__(
        self,
        processor: PaliGemmaProcessor,
        max_length: int = 512,
        image_size: int = 224,
    ):
        self.processor = processor
        self.max_length = max_length
        self.image_size = image_size

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collates a list of dictionaries with keys:
          - 'image': PIL.Image or torch.Tensor
          - 'prompt': str (input instruction/query)
          - 'suffix': str (target ground truth response)
        """
        images: List[Image.Image] = []
        prompts: List[str] = []
        suffixes: List[str] = []

        for item in batch:
            img = item["image"]
            if not isinstance(img, Image.Image):
                # Convert tensor or numpy array to PIL RGB Image
                img = Image.fromarray(img) if hasattr(img, "__array__") else img
                if hasattr(img, "convert"):
                    img = img.convert("RGB")
            images.append(img)
            prompts.append(item.get("prompt", ""))
            suffixes.append(item.get("suffix", ""))

        # Process multimodal inputs with suffix targets for training & distillation loss
        model_inputs = self.processor(
            images=images,
            text=prompts,
            suffix=suffixes,
            return_tensors="pt",
            padding="longest",
            max_length=self.max_length,
            truncation=True,
        )

        # Process prompt-only inputs for zero-leakage evaluation generation
        prompt_inputs = self.processor(
            images=images,
            text=prompts,
            return_tensors="pt",
            padding="longest",
            max_length=self.max_length,
            truncation=True,
        )
        model_inputs["prompt_input_ids"] = prompt_inputs["input_ids"]
        if "attention_mask" in prompt_inputs:
            model_inputs["prompt_attention_mask"] = prompt_inputs["attention_mask"]
        model_inputs["suffix_text"] = suffixes

        # If PaliGemmaProcessor did not build masked labels natively, construct them
        if "labels" not in model_inputs or model_inputs["labels"] is None:
            input_ids = model_inputs["input_ids"]
            labels = input_ids.clone()

            prompt_encodings = self.processor.tokenizer(
                prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )

            for i, prompt_ids in enumerate(prompt_encodings["input_ids"]):
                num_image_tokens = getattr(self.processor, "image_seq_length", 256)
                prompt_len = len(prompt_ids) + num_image_tokens
                labels[i, :prompt_len] = -100

            if "attention_mask" in model_inputs:
                labels[model_inputs["attention_mask"] == 0] = -100

            model_inputs["labels"] = labels

        return model_inputs
