"""Evaluate a trained checkpoint on a dataset with FF++ class structure."""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from train import FFPPDataset, build_model, evaluate, val_transform


def parse_args():
    parser = argparse.ArgumentParser(description="Validate DINOv2-SRCA on FF++-style data")
    parser.add_argument("--data-root", required=True, help="Directory containing FF++ class folders")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint produced by train.py")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-json", help="Optional path for metrics JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = FFPPDataset(args.data_root, val_transform, return_two_views=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        drop_last=False,
    )

    model = build_model(pretrained=False)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6 does not expose weights_only.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    metrics = evaluate(
        model=model,
        loader=loader,
        criterion=nn.BCEWithLogitsLoss(),
        device=device,
        threshold=args.threshold,
        description="FF++-style validation",
    )
    result = {key: float(value) for key, value in metrics.items()}
    print("Metrics | " + " | ".join(f"{key}: {value:.4f}" for key, value in result.items()))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved metrics to: {output_path}")


if __name__ == "__main__":
    main()
