"""Train DINOv2-SRCA fused with a frozen ArcFace Buffalo-L identity branch.

Architecture:
    image -> DINOv2 + RGB/SRM + SRCA
                |- CLS feature (768)
                |- mean patch feature (768)
                `- CNN forensic feature (256)
          -> frozen Buffalo-L identity embedding (512) -> projection (256)
          -> concatenate (2048) -> fusion (256) -> binary classifier

Buffalo-L runs through InsightFace/ONNX Runtime and is deliberately kept out of
the PyTorch state dict and optimizer. Only the 512 -> 256 identity projection,
fusion layers, SRCA/CNN branches, and classifier are trained.
"""

from __future__ import annotations

import argparse
import random
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from train_dino import (
    CelebDFDataset,
    DINOv2SRCADetector,
    FFPPDataset,
    training_loop,
    train_transform,
    val_transform,
)


class FrozenBuffaloL:
    """Non-differentiable, frozen InsightFace Buffalo-L feature extractor."""

    embedding_dim = 512

    def __init__(
        self,
        provider: str = "cuda",
        model_root: str = "~/.insightface",
        det_size: int = 224,
    ) -> None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as error:
            raise ImportError(
                "ArcFace requires insightface. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from error

        provider = provider.lower()
        if provider not in {"cuda", "cpu"}:
            raise ValueError("provider must be either 'cuda' or 'cpu'")

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if provider == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.app = FaceAnalysis(
            name="buffalo_l",
            root=model_root,
            allowed_modules=["detection", "recognition"],
            providers=providers,
        )
        self.app.prepare(
            ctx_id=0 if provider == "cuda" else -1,
            det_size=(det_size, det_size),
        )

    @staticmethod
    def _to_bgr_uint8(images: Tensor) -> list[np.ndarray]:
        """Undo ImageNet normalization and convert a batch to OpenCV images."""
        images = images.detach().float().cpu()
        mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        images = (images * std + mean).clamp(0.0, 1.0)
        images = images.permute(0, 2, 3, 1).numpy()
        rgb_batch = np.rint(images * 255.0).astype(np.uint8)
        return [image[..., ::-1].copy() for image in rgb_batch]

    def __call__(self, images: Tensor) -> tuple[Tensor, Tensor]:
        embeddings: list[np.ndarray] = []
        detected: list[bool] = []

        for image in self._to_bgr_uint8(images):
            faces = self.app.get(image)
            if not faces:
                embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
                detected.append(False)
                continue

            # Use the largest face when an image contains multiple detections.
            face = max(
                faces,
                key=lambda item: float(
                    (item.bbox[2] - item.bbox[0])
                    * (item.bbox[3] - item.bbox[1])
                ),
            )
            embedding = getattr(face, "normed_embedding", None)
            if embedding is None:
                embedding = face.embedding
                norm = max(float(np.linalg.norm(embedding)), 1e-12)
                embedding = embedding / norm
            embeddings.append(np.asarray(embedding, dtype=np.float32))
            detected.append(True)

        device = images.device
        embedding_tensor = torch.from_numpy(np.stack(embeddings)).to(device)
        detection_mask = torch.tensor(detected, dtype=torch.bool, device=device)
        return embedding_tensor, detection_mask


class DINOv2SRCAArcFaceDetector(DINOv2SRCADetector):
    """DINOv2-SRCA detector with frozen Buffalo-L identity fusion."""

    def __init__(
        self,
        *,
        arcface_provider: str = "cuda",
        arcface_model_root: str = "~/.insightface",
        arcface_det_size: int = 224,
        identity_dim: int = 256,
        fusion_dim: int = 256,
        classifier_hidden_dim: int = 384,
        dropout: float = 0.2,
        **kwargs,
    ) -> None:
        # The parent creates all DINO/SRM/SRCA feature extractors.
        super().__init__(
            projection_dim=256,
            classifier_hidden_dim=classifier_hidden_dim,
            dropout=dropout,
            **kwargs,
        )
        if self.dino_dim != 768:
            raise ValueError(
                f"This fusion expects DINOv2 ViT-B features of size 768, got {self.dino_dim}."
            )

        self.arcface = FrozenBuffaloL(
            provider=arcface_provider,
            model_root=arcface_model_root,
            det_size=arcface_det_size,
        )
        self.identity_projection = nn.Sequential(
            nn.Linear(512, identity_dim),
            nn.LayerNorm(identity_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        fusion_input_dim = self.dino_dim * 2 + 256 + identity_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, classifier_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim // 2, 1),
        )
        self._initialize_arcface_fusion_layers()

    def _initialize_arcface_fusion_layers(self) -> None:
        for root in (self.identity_projection, self.fusion, self.classifier):
            for module in root.modules():
                if isinstance(module, nn.Linear):
                    nn.init.trunc_normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward_features(self, images: Tensor) -> dict[str, Tensor]:
        # Compute ArcFace first because it consumes de-normalized CPU images.
        with torch.inference_mode():
            identity_embedding, identity_detected = self.arcface(images)

        features = super().forward_features(images)
        identity_feature = self.identity_projection(identity_embedding)

        # The three requested DINO/SRCA outputs plus projected identity.
        fusion_input = torch.cat(
            [
                features["cls_feature"],
                features["mean_patch_feature"],
                features["forensic_feature"],
                identity_feature,
            ],
            dim=1,
        )
        fused_feature = self.fusion(fusion_input)
        features.update(
            {
                "identity_embedding": identity_embedding,
                "identity_feature": identity_feature,
                "identity_detected": identity_detected,
                "fusion_input": fusion_input,
                "invariant_feature": fused_feature,
                "normalized_invariant": F.normalize(fused_feature, p=2, dim=1),
            }
        )
        return features

    def forward(self, images: Tensor, return_features: bool = False):
        features = self.forward_features(images)
        logits = self.classifier(features["invariant_feature"])
        if return_features:
            return {"logits": logits, **features}
        return logits


def build_model(
    pretrained: bool = True,
    arcface_provider: str = "cuda",
    arcface_model_root: str = "~/.insightface",
    arcface_det_size: int = 224,
) -> DINOv2SRCAArcFaceDetector:
    return DINOv2SRCAArcFaceDetector(
        pretrained=pretrained,
        freeze_dino=True,
        train_dino_norms=False,
        insert_positions=(2, 5, 8),
        cnn_channels=(48, 96, 192),
        local_dim=256,
        arcface_provider=arcface_provider,
        arcface_model_root=arcface_model_root,
        arcface_det_size=arcface_det_size,
        identity_dim=256,
        fusion_dim=256,
        classifier_hidden_dim=384,
        dropout=0.2,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DINOv2-SRCA + frozen ArcFace Buffalo-L detector"
    )
    parser.add_argument("--train-root", required=True, help="FF++ train directory")
    parser.add_argument("--val-root", required=True, help="Validation directory")
    parser.add_argument("--val-format", choices=("celebdf", "ffpp"), default="celebdf")
    parser.add_argument(
        "--checkpoint", default="checkpoints/best_dinov2_srca_arcface.pth"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--arcface-provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--arcface-model-root", default="~/.insightface")
    parser.add_argument("--arcface-det-size", type=int, default=224)
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.arcface_provider == "cuda" and device.type != "cuda":
        raise RuntimeError("--arcface-provider cuda requires a CUDA-capable environment")
    print(f"PyTorch device: {device} | ArcFace provider: {args.arcface_provider}")

    train_dataset = FFPPDataset(args.train_root, train_transform, return_two_views=True)
    val_dataset = (
        FFPPDataset(args.val_root, val_transform, return_two_views=False)
        if args.val_format == "ffpp"
        else CelebDFDataset(args.val_root, val_transform)
    )
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

    model = build_model(
        pretrained=not args.no_pretrained,
        arcface_provider=args.arcface_provider,
        arcface_model_root=args.arcface_model_root,
        arcface_det_size=args.arcface_det_size,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    assert not any(p.requires_grad for p in model.dino.parameters())
    optimizer = torch.optim.AdamW(
        [{"params": trainable_parameters, "lr": 1e-4, "weight_decay": 1e-4,
          "name": "srca_arcface_fusion"}]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-7
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total PyTorch: {total / 1e6:.2f}M | Trainable: {trainable / 1e6:.2f}M")
    training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        scaler=scaler,
        num_epochs=args.epochs,
        early_stopping_patience=args.patience,
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()


