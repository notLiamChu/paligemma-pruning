import argparse
import logging
from models.loader import load_paligemma_model
from models.inspector import inspect_paligemma_architecture, print_architecture_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect PaliGemma parameter mass and physical MLP channel layout."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="google/paligemma2-3b-pt-224",
        help="Hugging Face model checkpoint ID.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to load model on ('cpu', 'cuda', 'cuda:0'). Defaults to 'cpu' for safety.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Precision format for weight loading.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional Hugging Face authentication token for gated model access.",
    )

    args = parser.parse_args()

    print(f"\nInitializing model inspect run for: {args.model_id}")
    model, _ = load_paligemma_model(
        model_id=args.model_id,
        dtype=args.dtype,
        device=args.device,
        hf_token=args.hf_token,
        eval_mode=True,
    )

    profile = inspect_paligemma_architecture(model)
    print_architecture_report(profile)


if __name__ == "__main__":
    main()
