"""Cross-dataset evaluation on a Celeb-DF-style real/fake dataset."""

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from train import CelebDFDataset, build_model, evaluate, val_transform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test DINOv2-SRCA on cross-dataset data arranged as "
            "<data-root>/real and <data-root>/fake"
        )
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Dataset directory containing real/ and fake/ folders",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints",
        help=(
            "A checkpoint file or a directory. For a directory, the .pth/.pt "
            "checkpoint with the highest saved val_auc is selected"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-json", help="Optional path for the result JSON")
    return parser.parse_args()


def load_checkpoint(path: Path) -> Any:
    """Load a checkpoint on CPU across supported PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions without the weights_only argument.
        return torch.load(path, map_location="cpu")


def checkpoint_val_auc(checkpoint: Any) -> float:
    if not isinstance(checkpoint, dict):
        return float("-inf")
    value = checkpoint.get("val_auc", float("-inf"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def find_best_checkpoint(path: Path) -> tuple[Path, Any]:
    """Resolve a file or select the checkpoint with the largest val_auc."""
    if path.is_file():
        return path, load_checkpoint(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint path not found: {path}")

    candidates = sorted(
        candidate
        for pattern in ("*.pth", "*.pt")
        for candidate in path.rglob(pattern)
        if candidate.is_file()
    )
    if not candidates:
        raise FileNotFoundError(f"No .pth or .pt checkpoint found in: {path}")

    best_path: Path | None = None
    best_checkpoint: Any = None
    best_auc = float("-inf")
    errors: list[str] = []
    for candidate in candidates:
        try:
            checkpoint = load_checkpoint(candidate)
        except Exception as error:  # Keep looking if an unrelated file is invalid.
            errors.append(f"{candidate}: {error}")
            continue
        auc = checkpoint_val_auc(checkpoint)
        if best_path is None or auc > best_auc:
            best_path = candidate
            best_checkpoint = checkpoint
            best_auc = auc

    if best_path is None:
        details = "\n".join(errors)
        raise RuntimeError(f"Could not load any checkpoint in {path}:\n{details}")
    return best_path, best_checkpoint


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")

    checkpoint_path, checkpoint = find_best_checkpoint(Path(args.checkpoint))
    saved_auc = checkpoint_val_auc(checkpoint)
    auc_text = f"{saved_auc:.4f}" if saved_auc != float("-inf") else "not saved"
    print(f"Selected checkpoint: {checkpoint_path} | validation AUC: {auc_text}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    dataset = CelebDFDataset(args.data_root, val_transform)
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
    state_dict = (
        checkpoint.get("model_state_dict", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    metrics = evaluate(
        model=model,
        loader=loader,
        criterion=nn.BCEWithLogitsLoss(),
        device=device,
        threshold=args.threshold,
        description="Cross-dataset test",
    )
    result = {key: float(value) for key, value in metrics.items()}
    print("Test | " + " | ".join(f"{key}: {value:.4f}" for key, value in result.items()))

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(checkpoint_path),
            "checkpoint_val_auc": None if saved_auc == float("-inf") else saved_auc,
            "data_root": str(Path(args.data_root)),
            "threshold": args.threshold,
            "metrics": result,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved metrics to: {output_path}")


if __name__ == "__main__":
    main()
