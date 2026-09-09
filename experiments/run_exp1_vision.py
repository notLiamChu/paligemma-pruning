import os
import copy
import argparse
import logging
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from models.loader import load_paligemma_model
from models.inspector import inspect_paligemma_architecture, print_architecture_report
from models.checkpointing import save_sliced_paligemma
from core.agop import get_agop_targets, compute_agop_for_layer
from core.surgery import find_k_from_energy_threshold, perform_agop_eigen_surgery
from core.recovery import run_recovery_epochs
from data.collator import PaliGemmaDataCollator
from metrics.segmentation_eval import SegmentationEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class SyntheticMultimodalDataset(Dataset):
    """Synthetic dataset providing synthetic image-query pairs for calibration and verification."""
    def __init__(self, num_samples: int = 32, image_size: int = 224):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict:
        img = Image.new("RGB", (self.image_size, self.image_size), color=(idx * 7 % 255, 120, 200))
        return {
            "image": img,
            "prompt": "segment foreground object",
            "suffix": "<loc0120><loc0150><loc0800><loc0850><seg012><seg045>",
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 1: SigLIP Vision Tower Physical Pruning")
    parser.add_argument("--config", type=str, default="configs/exp1_vision_only.yaml", help="Path to config YAML")
    parser.add_argument("--device", type=str, default=None, help="Target device override (e.g. 'cuda:0', 'cpu')")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face authentication token")
    parser.add_argument("--dry-run", action="store_true", help="Run with synthetic dataset for quick verification")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(
        args.device or config["model"].get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Initializing Experiment 1 on device: {device}")

    # Load teacher model (unpruned reference)
    teacher_model, processor = load_paligemma_model(
        model_id=config["model"]["model_id"],
        dtype=config["model"]["dtype"],
        device=device,
        hf_token=args.hf_token,
        eval_mode=True,
    )
    teacher_model.eval()

    # Create mutable student copy for pruning
    student_model = copy.deepcopy(teacher_model)
    student_model.train()

    logger.info("Baseline architecture footprint:")
    profile_before = inspect_paligemma_architecture(student_model)
    print_architecture_report(profile_before)

    collator = PaliGemmaDataCollator(
        processor=processor,
        max_length=config["data"]["max_length"],
        image_size=config["data"]["image_size"],
    )

    dataset = SyntheticMultimodalDataset(num_samples=config["pruning"]["agop_samples"])
    calibration_loader = DataLoader(
        dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        collate_fn=collator,
    )

    specs = get_agop_targets(
        student_model,
        target_components=config["pruning"]["target_components"],
        target_layers=config["pruning"]["target_layers"],
    )
    logger.info(f"Identified {len(specs)} target MLP layers in SigLIP vision encoder.")

    tau = config["pruning"]["energy_threshold"]
    sigma = config["pruning"]["eigen_coordinate_sigma"]

    for spec in specs:
        logger.info(f"\n--- Processing {spec.name} ---")
        agop_mat, eigvals, n = compute_agop_for_layer(
            model=student_model,
            data_loader=calibration_loader,
            target_spec=spec,
            device=device,
            num_samples=config["pruning"]["agop_samples"],
            score_mode=config["pruning"]["score_mode"],
        )

        k = find_k_from_energy_threshold(eigvals, energy_threshold=tau)
        orig_d_ff = agop_mat.shape[0]
        logger.info(f"Spectral cutoff: {orig_d_ff} -> k={k} channels (tau={tau})")

        # Perform physical channel slicing
        perform_agop_eigen_surgery(spec.parent_block, agop_mat, k=k, sigma=sigma)

    logger.info("\nPost-surgery architecture footprint:")
    profile_after = inspect_paligemma_architecture(student_model)
    print_architecture_report(profile_after)

    # Freeze Gemma 2 language decoder; optimize only pruned vision encoder and projector
    for p in student_model.language_model.parameters():
        p.requires_grad = False
    for p in student_model.vision_tower.parameters():
        p.requires_grad = True
    if hasattr(student_model, "multi_modal_projector"):
        for p in student_model.multi_modal_projector.parameters():
            p.requires_grad = True

    logger.info("Initiating micro-recovery distillation on pruned vision parameters...")
    run_recovery_epochs(
        student_model=student_model,
        train_loader=calibration_loader,
        val_loader=None,
        device=device,
        teacher_model=teacher_model,
        epochs=config["distillation"]["epochs"],
        lr=config["distillation"]["learning_rate"],
        weight_decay=config["distillation"]["weight_decay"],
        alpha=config["distillation"]["alpha"],
        temperature=config["distillation"]["temperature"],
    )

    out_dir = config["output"]["checkpoint_dir"]
    save_sliced_paligemma(
        model=student_model,
        save_dir=out_dir,
        processor=processor,
        base_model_id=config["model"]["model_id"],
        metadata={
            "experiment": config["experiment"]["name"],
            "energy_threshold_tau": tau,
            "eigen_coordinate_sigma": sigma,
            "vision_pruned_params": profile_before.vision_mlp_params - profile_after.vision_mlp_params,
        },
    )
    logger.info(f"Experiment 1 complete. Sliced checkpoint serialized to: {out_dir}")


if __name__ == "__main__":
    main()
