from core.agop import (
    TargetLayerSpec,
    get_agop_targets,
    compute_agop_for_layer,
    estimate_agop_layers,
    compute_sample_score,
)
from core.surgery import (
    find_k_from_energy_threshold,
    extract_eigen_coordinates,
    slice_siglip_mlp,
    slice_gemma_swiglu,
    apply_physical_slicing,
    perform_agop_eigen_surgery,
)
from core.recovery import (
    MultimodalDistillationLoss,
    adapt_teacher_to_task,
    cache_teacher_logits,
    run_recovery_epochs,
)

__all__ = [
    "TargetLayerSpec",
    "get_agop_targets",
    "compute_agop_for_layer",
    "estimate_agop_layers",
    "compute_sample_score",
    "find_k_from_energy_threshold",
    "extract_eigen_coordinates",
    "slice_siglip_mlp",
    "slice_gemma_swiglu",
    "apply_physical_slicing",
    "perform_agop_eigen_surgery",
    "MultimodalDistillationLoss",
    "adapt_teacher_to_task",
    "cache_teacher_logits",
    "run_recovery_epochs",
]
