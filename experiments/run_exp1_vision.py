import gc
import copy
import argparse
import logging
import yaml
import torch
from torch.utils.data import DataLoader

from models.loader import load_paligemma_model
from models.inspector import inspect_paligemma_architecture, print_architecture_report
from models.checkpointing import save_sliced_paligemma
from core.agop import get_agop_targets, compute_agop_for_layer
from core.surgery import find_k_from_energy_threshold, perform_agop_eigen_surgery
from core.recovery import adapt_teacher_to_task, cache_teacher_logits, run_recovery_epochs
from data.segmentation import build_refcoco_dataloaders
from metrics.segmentation_eval import SegmentationEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment 1: SigLIP Vision Tower Physical Pruning")
    parser.add_argument("--config", type=str, default="configs/exp1_vision_only.yaml", help="Path to config YAML")
    parser.add_argument("--device", type=str, default=None, help="Target device override (e.g. 'cuda:0', 'cpu')")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face authentication token")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device(
        args.device or config["model"].get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info(f"Initializing Experiment 1 on device: {device}")

    # 1. Load Base PaliGemma Model
    teacher_model, processor = load_paligemma_model(
        model_id=config["model"]["model_id"],
        dtype=config["model"]["dtype"],
        device=device,
        hf_token=args.hf_token,
        eval_mode=False,
    )

    # 2. Build RefCOCO DataLoaders (Train and Val)
    train_loader, val_loader = build_refcoco_dataloaders(
        processor=processor,
        batch_size=config["data"]["batch_size"],
        num_train_samples=config["data"]["num_train_samples"],
        num_val_samples=config["data"]["num_val_samples"],
        max_length=config["data"]["max_length"],
        image_size=config["data"]["image_size"],
        hf_dataset_id=config["data"].get("hf_dataset_id", "lmms-lab/RefCOCO"),
    )

    evaluator = SegmentationEvaluator(processor=processor)

    # 3. Optional Teacher Task Adaptation (Warmup)
    if config.get("teacher_adaptation", {}).get("enabled", True):
        teacher_model = adapt_teacher_to_task(
            teacher_model=teacher_model,
            train_loader=train_loader,
            device=device,
            epochs=config["teacher_adaptation"]["epochs"],
            lr=config["teacher_adaptation"]["learning_rate"],
            weight_decay=config["teacher_adaptation"]["weight_decay"],
        )

    # 4. Clone Mutable Student from Adapted Teacher
    student_model = copy.deepcopy(teacher_model)
    student_model.train()

    logger.info("\nBaseline architecture profile:")
    profile_before = inspect_paligemma_architecture(student_model)
    print_architecture_report(profile_before)

    # 5. Offline Teacher Logit Caching
    cached_teacher_logits = None
    if config.get("offline_distillation", {}).get("cache_logits", True):
        cache_file = config["offline_distillation"].get("cache_file")
        cached_teacher_logits = cache_teacher_logits(
            teacher_model=teacher_model,
            train_loader=train_loader,
            device=device,
            cache_path=cache_file,
        )
        # Evict teacher from GPU memory
        logger.info("Evicting teacher model from GPU VRAM to reclaim memory...")
        del teacher_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 6. Execute AGOP Extraction and Channel Slicing on SigLIP Vision Layers
    specs = get_agop_targets(
        student_model,
        target_components=config["pruning"]["target_components"],
        target_layers=config["pruning"]["target_layers"],
    )
    logger.info(f"Targeting {len(specs)} SigLIP vision MLP blocks for AGOP surgery.")

    tau = config["pruning"]["energy_threshold"]
    sigma = config["pruning"]["eigen_coordinate_sigma"]

    for spec in specs:
        logger.info(f"\n--- Extracting AGOP for {spec.name} ---")
        agop_mat, eigvals, n = compute_agop_for_layer(
            model=student_model,
            data_loader=train_loader,
            target_spec=spec,
            device=device,
            num_samples=config["pruning"]["agop_samples"],
            score_mode=config["pruning"]["score_mode"],
        )

        k = find_k_from_energy_threshold(eigvals, energy_threshold=tau)
        orig_d_ff = agop_mat.shape[0]
        logger.info(f"Spectral energy cutoff: {orig_d_ff} -> k={k} channels (tau={tau})")

        # In-place physical eigen surgery
        perform_agop_eigen_surgery(spec.parent_block, agop_mat, k=k, sigma=sigma)

    logger.info("\nPost-surgery architecture profile:")
    profile_after = inspect_paligemma_architecture(student_model)
    print_architecture_report(profile_after)

    # 7. Freeze Language Model; Train Pruned Vision Parameters and Projector
    for p in student_model.language_model.parameters():
        p.requires_grad = False
    for p in student_model.vision_tower.parameters():
        p.requires_grad = True
    if hasattr(student_model, "multi_modal_projector"):
        for p in student_model.multi_modal_projector.parameters():
            p.requires_grad = True

    # 8. Micro-Recovery Distillation using Offline Teacher Logits (0 GB Teacher GPU Overhead)
    logger.info("Starting micro-recovery distillation on pruned vision parameters...")
    run_recovery_epochs(
        student_model=student_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        teacher_model=None,
        cached_teacher_logits=cached_teacher_logits,
        epochs=config["distillation"]["epochs"],
        lr=config["distillation"]["learning_rate"],
        weight_decay=config["distillation"]["weight_decay"],
        alpha=config["distillation"]["alpha"],
        temperature=config["distillation"]["temperature"],
    )

    # 9. Serialize Pruned Checkpoint & Manifest
    out_dir = config["output"]["checkpoint_dir"]
    save_sliced_paligemma(
        model=student_model,
        save_dir=out_dir,
        processor=processor,
        base_model_id=config["model"]["model_id"],
        metadata={
            "experiment": config["experiment"]["name"],
            "dataset": config["data"]["dataset_name"],
            "energy_threshold_tau": tau,
            "eigen_coordinate_sigma": sigma,
            "vision_pruned_params": profile_before.vision_mlp_params - profile_after.vision_mlp_params,
        },
    )
    logger.info(f"Experiment 1 complete. Physically sliced checkpoint saved to: {out_dir}")


if __name__ == "__main__":
    main()
