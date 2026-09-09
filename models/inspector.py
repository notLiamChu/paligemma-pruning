from dataclasses import dataclass
from typing import Dict, Any
import torch
import torch.nn as nn


@dataclass
class ModelProfile:
    total_params: int
    trainable_params: int
    
    # Vision Tower (SigLIP)
    vision_total_params: int
    vision_mlp_params: int
    vision_layers_count: int
    siglip_d_model: int
    siglip_d_ff: int
    
    # Language Decoder (Gemma 2)
    language_total_params: int
    language_mlp_params: int
    language_layers_count: int
    gemma_d_model: int
    gemma_d_ff: int
    
    # Projector & Other
    projector_params: int
    other_params: int
    
    # Pruning Targets
    total_pruneable_mlp_params: int
    pruneable_ratio_of_total: float
    
    # Memory Footprint
    memory_footprint_bf16_mb: float
    memory_footprint_fp32_mb: float


def inspect_paligemma_architecture(model: nn.Module) -> ModelProfile:
    """
    Profiles PaliGemma parameter mass, isolating MLP channels in both
    SigLIP (standard MLP) and Gemma 2 (SwiGLU).
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 1. SigLIP Vision Tower
    vision_tower = model.vision_tower.vision_model
    vision_layers = vision_tower.encoder.layers
    vision_layers_count = len(vision_layers)
    vision_total_params = sum(p.numel() for p in model.vision_tower.parameters())

    vision_mlp_params = 0
    sample_vis_layer = vision_layers[0].mlp
    siglip_d_model = sample_vis_layer.fc1.in_features
    siglip_d_ff = sample_vis_layer.fc1.out_features

    for layer in vision_layers:
        mlp = layer.mlp
        vision_mlp_params += sum(p.numel() for p in mlp.parameters())

    # 2. Gemma 2 Language Decoder
    language_model = model.language_model
    lang_layers = language_model.model.layers
    language_layers_count = len(lang_layers)
    language_total_params = sum(p.numel() for p in language_model.parameters())

    language_mlp_params = 0
    sample_lang_layer = lang_layers[0].mlp
    gemma_d_model = sample_lang_layer.gate_proj.in_features
    gemma_d_ff = sample_lang_layer.gate_proj.out_features

    for layer in lang_layers:
        mlp = layer.mlp
        language_mlp_params += sum(p.numel() for p in mlp.parameters())

    # 3. Multimodal Projector
    projector_params = (
        sum(p.numel() for p in model.multi_modal_projector.parameters())
        if hasattr(model, "multi_modal_projector")
        else 0
    )

    # 4. Aggregated Pruning Metrics
    pruneable_mlp_params = vision_mlp_params + language_mlp_params
    pruneable_ratio = (pruneable_mlp_params / total_params) if total_params > 0 else 0.0
    other_params = total_params - (vision_total_params + language_total_params + projector_params)

    # 2 bytes per parameter for BF16, 4 bytes for FP32
    memory_bf16_mb = (total_params * 2) / (1024 * 1024)
    memory_fp32_mb = (total_params * 4) / (1024 * 1024)

    return ModelProfile(
        total_params=total_params,
        trainable_params=trainable_params,
        vision_total_params=vision_total_params,
        vision_mlp_params=vision_mlp_params,
        vision_layers_count=vision_layers_count,
        siglip_d_model=siglip_d_model,
        siglip_d_ff=siglip_d_ff,
        language_total_params=language_total_params,
        language_mlp_params=language_mlp_params,
        language_layers_count=language_layers_count,
        gemma_d_model=gemma_d_model,
        gemma_d_ff=gemma_d_ff,
        projector_params=projector_params,
        other_params=other_params,
        total_pruneable_mlp_params=pruneable_mlp_params,
        pruneable_ratio_of_total=pruneable_ratio,
        memory_footprint_bf16_mb=memory_bf16_mb,
        memory_footprint_fp32_mb=memory_fp32_mb,
    )


def print_architecture_report(profile: ModelProfile) -> None:
    """Renders a clean tabular summary of the parameter distribution."""
    sep = "─" * 72
    print(f"\n{sep}")
    print("              PALIGEMMA ARCHITECTURAL FOOTPRINT REPORT")
    print(sep)
    print(f"Total Parameters:             {profile.total_params:>15,}  (100.0%)")
    print(f"Active Trainable Parameters:  {profile.trainable_params:>15,}  ({profile.trainable_params / profile.total_params * 100:>5.1f}%)")
    print(sep)
    print(f"Vision Tower (SigLIP):        {profile.vision_total_params:>15,}  ({profile.vision_total_params / profile.total_params * 100:>5.1f}%)")
    print(f"  • Layers:                   {profile.vision_layers_count:>15}")
    print(f"  • Dimensions:               {f'd={profile.siglip_d_model}, d_ff={profile.siglip_d_ff}':>15}")
    print(f"  • MLP Parameter Pool:       {profile.vision_mlp_params:>15,}  ({profile.vision_mlp_params / profile.total_params * 100:>5.1f}%)")
    print(sep)
    print(f"Language Decoder (Gemma 2):   {profile.language_total_params:>15,}  ({profile.language_total_params / profile.total_params * 100:>5.1f}%)")
    print(f"  • Layers:                   {profile.language_layers_count:>15}")
    print(f"  • Dimensions:               {f'd={profile.gemma_d_model}, d_ff={profile.gemma_d_ff}':>15}")
    print(f"  • SwiGLU MLP Parameter Pool:{profile.language_mlp_params:>15,}  ({profile.language_mlp_params / profile.total_params * 100:>5.1f}%)")
    print(sep)
    print(f"Multimodal Projector:         {profile.projector_params:>15,}  ({profile.projector_params / profile.total_params * 100:>5.1f}%)")
    print(sep)
    print("PRUNING TARGET SPECIFICATION:")
    print(f"Total Pruneable MLP Mass:     {profile.total_pruneable_mlp_params:>15,}  ({profile.pruneable_ratio_of_total * 100:>5.1f}%)")
    print(sep)
    print("MEMORY FOOTPRINT (WEIGHTS ONLY):")
    print(f"  • In BF16 / FP16 Precision: {profile.memory_footprint_bf16_mb:>12.2f} MB  ({profile.memory_footprint_bf16_mb / 1024:.2f} GB)")
    print(f"  • In FP32 Precision:        {profile.memory_footprint_fp32_mb:>12.2f} MB  ({profile.memory_footprint_fp32_mb / 1024:.2f} GB)")
    print(f"{sep}\n")
