"""Train a frozen DINOv2 + frozen ArcFace Buffalo_L fusion detector.

The data pipeline, augmentations, metrics, and validation layouts are reused
from ``train_dino.py``.  Only the fusion head is optimized; both feature
extractors always run in inference mode.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from tqdm.auto import tqdm

from train_dino import (
    CelebDFDataset,
    FFPPDataset,
    calculate_metrics,
    train_transform,
    val_transform,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ArcFaceBuffaloLEncoder:
    """Inference-only InsightFace Buffalo_L wrapper.

    FaceAnalysis performs detection/alignment before recognition. If no face is
    detected, the default fallback sends the resized full image to Buffalo_L's
    recognition model so every input still has a deterministic embedding.
    """

    embedding_dim = 512

    def __init__(
        self,
        root: str = "~/.insightface",
        providers: Sequence[str] = ("CPUExecutionProvider",),
        det_size: int = 224,
        no_face_policy: str = "full-image",
    ) -> None:
        try:
            import cv2
            from insightface.app import FaceAnalysis
        except ImportError as error:
            raise ImportError(
                "ArcFace Buffalo_L requires insightface, opencv-python and "
                "onnxruntime (or onnxruntime-gpu)."
            ) from error

        if no_face_policy not in {"full-image", "zero"}:
            raise ValueError("no_face_policy must be 'full-image' or 'zero'")

        self.cv2 = cv2
        self.no_face_policy = no_face_policy
        self.app = FaceAnalysis(
            name="buffalo_l",
            root=str(Path(root).expanduser()),
            providers=list(providers),
            allowed_modules=["detection", "recognition"],
        )
        ctx_id = 0 if "CUDAExecutionProvider" in providers else -1
        self.app.prepare(ctx_id=ctx_id, det_size=(det_size, det_size))
        self.recognition_model = self.app.models.get("recognition")
        if self.recognition_model is None:
            raise RuntimeError("Buffalo_L recognition model was not loaded")

    @staticmethod
    def _largest_face(faces):
        return max(
            faces,
            key=lambda face: float(
                (face.bbox[2] - face.bbox[0])
                * (face.bbox[3] - face.bbox[1])
            ),
        )

    def _fallback_embedding(self, bgr_image: np.ndarray) -> np.ndarray:
        if self.no_face_policy == "zero":
            return np.zeros(self.embedding_dim, dtype=np.float32)
        face_crop = self.cv2.resize(bgr_image, (112, 112))
        embedding = self.recognition_model.get_feat(face_crop).reshape(-1)
        return embedding.astype(np.float32, copy=False)

    def __call__(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected Bx3xHxW images, got {tuple(images.shape)}")

        mean = images.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = images.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
        rgb_batch = (
            (images.detach().float() * std + mean)
            .clamp(0, 1)
            .mul(255)
            .byte()
            .permute(0, 2, 3, 1)
            .cpu()
            .numpy()
        )

        embeddings = []
        for rgb_image in rgb_batch:
            bgr_image = np.ascontiguousarray(rgb_image[:, :, ::-1])
            faces = self.app.get(bgr_image)
            if faces:
                embedding = self._largest_face(faces).embedding
                embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
            else:
                embedding = self._fallback_embedding(bgr_image)
            if embedding.size != self.embedding_dim:
                raise RuntimeError(
                    f"Expected a 512-D ArcFace embedding, got {embedding.size}"
                )
            embeddings.append(embedding)

        return torch.from_numpy(np.stack(embeddings)).to(
            device=images.device, dtype=torch.float32, non_blocking=True
        )


class DINOArcFaceFusionHead(nn.Module):
    def __init__(
        self,
        dino_dim: int = 768,
        arcface_dim: int = 512,
        fusion_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.dino_projection = nn.Sequential(
            nn.Linear(dino_dim * 2, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, fusion_dim),
            nn.LayerNorm(fusion_dim),
        )
        self.identity_projection = nn.Sequential(
            nn.Linear(arcface_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.Sigmoid(),
        )

        combined_dim = fusion_dim * 5
        self.fusion_mlp = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion_skip = nn.Linear(combined_dim, fusion_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
        self._initialize()

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        dino_cls: Tensor,
        dino_patch_tokens: Tensor,
        arcface_embedding: Tensor,
        return_features: bool = False,
    ):
        dino_mean = dino_patch_tokens.mean(dim=1)
        z_dino = self.dino_projection(torch.cat([dino_cls, dino_mean], dim=1))
        z_identity = self.identity_projection(
            F.normalize(arcface_embedding, p=2, dim=1)
        )
        gate = self.gate(torch.cat([z_dino, z_identity], dim=1))
        z_gated = gate * z_dino + (1.0 - gate) * z_identity
        fusion_input = torch.cat(
            [
                z_dino,
                z_identity,
                z_gated,
                torch.abs(z_dino - z_identity),
                z_dino * z_identity,
            ],
            dim=1,
        )
        fused = self.fusion_mlp(fusion_input) + self.fusion_skip(fusion_input)
        logits = self.classifier(fused).squeeze(1)
        if return_features:
            return {
                "logits": logits,
                "fused_feature": fused,
                "normalized_fused": F.normalize(fused, p=2, dim=1),
                "z_dino": z_dino,
                "z_identity": z_identity,
                "gate": gate,
            }
        return logits


class FrozenDINOArcFaceDetector(nn.Module):
    def __init__(
        self,
        dino_checkpoint: str,
        arcface_root: str = "~/.insightface",
        arcface_providers: Sequence[str] = ("CPUExecutionProvider",),
        arcface_det_size: int = 224,
        no_face_policy: str = "full-image",
        fusion_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.dino = torch.hub.load(
            repo_or_dir="facebookresearch/dinov2",
            model="dinov2_vitb14",
            pretrained=True,
        )
        self._load_dino_checkpoint(dino_checkpoint)
        for parameter in self.dino.parameters():
            parameter.requires_grad = False
        self.dino.eval()

        self.arcface = ArcFaceBuffaloLEncoder(
            root=arcface_root,
            providers=arcface_providers,
            det_size=arcface_det_size,
            no_face_policy=no_face_policy,
        )
        self.head = DINOArcFaceFusionHead(
            dino_dim=int(self.dino.embed_dim),
            arcface_dim=self.arcface.embedding_dim,
            fusion_dim=fusion_dim,
            dropout=dropout,
        )

    def _load_dino_checkpoint(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"DINO checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = checkpoint.get("model_state_dict", checkpoint)
        if not isinstance(state, dict):
            raise TypeError("DINO checkpoint must contain a state_dict")

        dino_state = {}
        for key, value in state.items():
            clean_key = key.removeprefix("module.")
            if clean_key.startswith("dino."):
                dino_state[clean_key.removeprefix("dino.")] = value
            elif clean_key in self.dino.state_dict():
                dino_state[clean_key] = value
        if not dino_state:
            raise RuntimeError(
                "No DINO weights found. Expected keys beginning with 'dino.' "
                "or a raw DINOv2 state_dict."
            )
        incompatible = self.dino.load_state_dict(dino_state, strict=False)
        if incompatible.missing_keys:
            raise RuntimeError(
                "DINO checkpoint is incomplete; missing keys include: "
                + ", ".join(incompatible.missing_keys[:5])
            )
        if incompatible.unexpected_keys:
            raise RuntimeError(
                "Unexpected DINO keys: " + ", ".join(incompatible.unexpected_keys[:5])
            )
        print(f"Loaded frozen DINOv2 weights from: {path}")

    def train(self, mode: bool = True):
        super().train(mode)
        self.dino.eval()
        return self

    def forward(self, images: Tensor, return_features: bool = False):
        # no_grad avoids storing an activation graph for either frozen encoder.
        with torch.no_grad():
            dino_features = self.dino.forward_features(images)
            dino_cls = dino_features["x_norm_clstoken"]
            dino_patches = dino_features["x_norm_patchtokens"]
            arcface_embedding = self.arcface(images)
        return self.head(
            dino_cls,
            dino_patches,
            arcface_embedding,
            return_features=return_features,
        )


def consistency_loss(output_1: Dict[str, Tensor], output_2: Dict[str, Tensor]) -> Tensor:
    feature_loss = 1.0 - F.cosine_similarity(
        output_1["normalized_fused"], output_2["normalized_fused"], dim=1
    ).mean()
    probability_loss = F.mse_loss(
        torch.sigmoid(output_1["logits"].float()),
        torch.sigmoid(output_2["logits"].float()),
    )
    return feature_loss + 0.5 * probability_loss


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    scaler,
    lambda_cons: float,
    threshold: float,
    max_grad_norm: float = 1.0,
):
    model.train()
    totals = {"loss": 0.0, "bce": 0.0, "cons": 0.0}
    sample_count = 0
    all_labels, all_probabilities = [], []

    for view_1, view_2, labels in tqdm(loader, desc="Training", leave=False):
        view_1 = view_1.to(device, non_blocking=True)
        view_2 = view_2.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True).view(-1)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            output_1 = model(view_1, return_features=True)
            output_2 = model(view_2, return_features=True)
            bce = 0.5 * (
                criterion(output_1["logits"], labels)
                + criterion(output_2["logits"], labels)
            )
            cons = consistency_loss(output_1, output_2)
            loss = bce + lambda_cons * cons

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.head.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        sample_count += batch_size
        totals["loss"] += loss.item() * batch_size
        totals["bce"] += bce.item() * batch_size
        totals["cons"] += cons.item() * batch_size
        logits = 0.5 * (output_1["logits"] + output_2["logits"])
        all_labels.append(labels.detach().cpu())
        all_probabilities.append(torch.sigmoid(logits.float()).detach().cpu())

    metrics = calculate_metrics(
        torch.cat(all_labels).numpy(),
        torch.cat(all_probabilities).numpy(),
        threshold,
    )
    metrics.update({key: value / sample_count for key, value in totals.items()})
    return metrics


@torch.inference_mode()
def evaluate(model, loader, criterion, device, threshold: float):
    model.eval()
    running_loss, sample_count = 0.0, 0
    all_labels, all_probabilities = [], []
    for images, labels in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True).view(-1)
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
        ):
            logits = model(images).view(-1)
            loss = criterion(logits, labels)
        batch_size = labels.size(0)
        sample_count += batch_size
        running_loss += loss.item() * batch_size
        all_labels.append(labels.cpu())
        all_probabilities.append(torch.sigmoid(logits.float()).cpu())

    metrics = calculate_metrics(
        torch.cat(all_labels).numpy(),
        torch.cat(all_probabilities).numpy(),
        threshold,
    )
    metrics["loss"] = running_loss / sample_count
    return metrics


def train_loop(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    scaler,
    epochs: int,
    patience: int,
    checkpoint_path: str,
    threshold: float,
):
    output_path = Path(checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_auc, stale_epochs, history = float("-inf"), 0, []

    for epoch in range(1, epochs + 1):
        assert not any(parameter.requires_grad for parameter in model.dino.parameters())
        lambda_cons = min(0.20, 0.05 + 0.03 * (epoch - 1))
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            lambda_cons, threshold,
        )
        val_metrics = evaluate(model, val_loader, criterion, device, threshold)
        scheduler.step(val_metrics["auc"])
        lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch [{epoch:02d}/{epochs:02d}] | encoders: frozen | LR: {lr:.2e}")
        print(
            f"Train | Loss: {train_metrics['loss']:.4f} | "
            f"BCE: {train_metrics['bce']:.4f} | Cons: {train_metrics['cons']:.4f} | "
            f"Acc: {train_metrics['acc']:.4f} | AUC: {train_metrics['auc']:.4f} | "
            f"F1: {train_metrics['f1']:.4f}"
        )
        print(
            f"Valid | Loss: {val_metrics['loss']:.4f} | "
            f"Acc: {val_metrics['acc']:.4f} | AUC: {val_metrics['auc']:.4f} | "
            f"F1: {val_metrics['f1']:.4f} | "
            f"Precision: {val_metrics['precision']:.4f} | "
            f"Recall: {val_metrics['recall']:.4f}"
        )
        history.append({
            "epoch": epoch,
            "encoders_frozen": True,
            "lambda_cons": lambda_cons,
            "train": train_metrics,
            "val": val_metrics,
        })

        if val_metrics["auc"] > best_auc + 1e-4:
            best_auc, stale_epochs = val_metrics["auc"], 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": best_auc,
                    "history": history,
                    "architecture": "dinov2_vitb14+arcface_buffalo_l_fusion",
                },
                output_path,
            )
            print(f"Saved best model | Epoch: {epoch} | Val AUC: {best_auc:.4f}")
        else:
            stale_epochs += 1
            print(f"Early stopping: {stale_epochs}/{patience}")
            if stale_epochs >= patience:
                print("Early stopping triggered.")
                break
    return history


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train frozen DINOv2 + ArcFace Buffalo_L fusion detector"
    )
    parser.add_argument("--train-root", required=True, help="FF++ train directory")
    parser.add_argument("--val-root", required=True, help="Validation directory")
    parser.add_argument(
        "--val-format", choices=("celebdf", "ffpp"), default="celebdf"
    )
    parser.add_argument(
        "--dino-checkpoint",
        required=True,
        help="Completed train_dino.py checkpoint (or raw DINOv2 state_dict)",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_dino_arcface_fusion.pth",
        help="Output checkpoint",
    )
    parser.add_argument("--arcface-root", default="~/.insightface")
    parser.add_argument(
        "--arcface-provider",
        choices=("cpu", "cuda"),
        default="cpu",
        help="ONNX Runtime provider used by Buffalo_L",
    )
    parser.add_argument("--arcface-det-size", type=int, default=224)
    parser.add_argument(
        "--no-face-policy", choices=("full-image", "zero"), default="full-image"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if args.arcface_det_size <= 0:
        raise ValueError("--arcface-det-size must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_dataset = FFPPDataset(args.train_root, train_transform, return_two_views=True)
    if args.val_format == "ffpp":
        val_dataset = FFPPDataset(args.val_root, val_transform, return_two_views=False)
    else:
        val_dataset = CelebDFDataset(args.val_root, val_transform)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, drop_last=True, **loader_options
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, shuffle=False, drop_last=False, **loader_options
    )

    providers = (
        ("CUDAExecutionProvider", "CPUExecutionProvider")
        if args.arcface_provider == "cuda"
        else ("CPUExecutionProvider",)
    )
    model = FrozenDINOArcFaceDetector(
        dino_checkpoint=args.dino_checkpoint,
        arcface_root=args.arcface_root,
        arcface_providers=providers,
        arcface_det_size=args.arcface_det_size,
        no_face_policy=args.no_face_policy,
    ).to(device)
    trainable_parameters = list(model.head.parameters())
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-7
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Total PyTorch: {total / 1e6:.2f}M | Trainable: {trainable / 1e6:.2f}M")
    train_loop(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, scaler, args.epochs, args.patience, args.checkpoint, args.threshold,
    )


if __name__ == "__main__":
    main()
