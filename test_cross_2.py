"""Evaluate a checkpoint on real/fake data with one intermediate folder."""

import argparse
import json
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from test_cross_1 import checkpoint_val_auc, find_best_checkpoint
from train import build_model, evaluate, val_transform


class NestedCelebDFDataset(Dataset):
    """Load images from root/{real,fake}/<subfolder>/<image>."""

    CLASS_TO_LABEL = {"real": 0, "fake": 1}
    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: list[Tuple[Path, int]] = []

        if not self.root_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        for class_name, label in self.CLASS_TO_LABEL.items():
            class_dir = self.root_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Class directory not found: {class_dir}")

            subfolders = sorted(path for path in class_dir.iterdir() if path.is_dir())
            if not subfolders:
                raise RuntimeError(f"No subfolder found in: {class_dir}")

            class_samples: list[Tuple[Path, int]] = []
            for subfolder in subfolders:
                image_paths = sorted(
                    path
                    for path in subfolder.iterdir()
                    if path.is_file() and path.suffix.lower() in self.VALID_EXTENSIONS
                )
                class_samples.extend((path, label) for path in image_paths)

            if not class_samples:
                raise RuntimeError(
                    f"No supported images found one level below: {class_dir}"
                )
            self.samples.extend(class_samples)

        real_count = sum(label == 0 for _, label in self.samples)
        fake_count = len(self.samples) - real_count
        print(
            f"Nested cross dataset | Total: {len(self.samples):,} | "
            f"Real: {real_count:,} | Fake: {fake_count:,}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test DINOv2-SRCA on data arranged as "
            "<data-root>/{real,fake}/<subfolder>/<image>"
        )
    )
    parser.add_argument("--data-root", required=True, help="Root of the test dataset")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints",
        help=(
            "A checkpoint file or directory. If it is a directory, the checkpoint "
            "with the highest saved val_auc is selected"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-json", help="Optional path for the result JSON")
    return parser.parse_args()


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
    dataset = NestedCelebDFDataset(args.data_root, val_transform)
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
        description="Nested cross-dataset test",
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
