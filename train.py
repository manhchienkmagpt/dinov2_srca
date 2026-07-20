# Generated from DINOv2_SRCA_generalized.ipynb; CLI entry point is below.


import io
import random

from PIL import Image
from torchvision import transforms


IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class RandomJPEGCompression:
    """Simulate codec and quality differences across datasets."""

    def __init__(
        self,
        quality_range: tuple[int, int] = (35, 100),
        probability: float = 0.5,
    ) -> None:
        self.quality_range = quality_range
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image

        quality = random.randint(*self.quality_range)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer).convert("RGB")
        compressed.load()
        return compressed


class RandomDownUpScale:
    """Downsample and then upscale to reduce resolution-based shortcuts."""

    def __init__(
        self,
        min_scale: float = 0.45,
        probability: float = 0.35,
    ) -> None:
        self.min_scale = min_scale
        self.probability = probability

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image

        width, height = image.size
        scale = random.uniform(self.min_scale, 0.9)
        small_size = (
            max(16, int(width * scale)),
            max(16, int(height * scale)),
        )
        interpolation = random.choice([
            Image.Resampling.BILINEAR,
            Image.Resampling.BICUBIC,
            Image.Resampling.LANCZOS,
        ])
        image = image.resize(small_size, interpolation)
        return image.resize((width, height), interpolation)


# The two views share the same label but use different degradations.
# The model is encouraged to keep features stable under compression, blur, and color shifts.
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([
        transforms.RandomRotation(degrees=8),
    ], p=0.35),
    transforms.RandomApply([
        transforms.RandomAffine(
            degrees=0,
            translate=(0.04, 0.04),
            scale=(0.92, 1.08),
        ),
    ], p=0.35),
    RandomDownUpScale(min_scale=0.45, probability=0.35),
    RandomJPEGCompression((35, 100), probability=0.60),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
    ], p=0.30),
    transforms.RandomApply([
        transforms.ColorJitter(
            brightness=0.20,
            contrast=0.20,
            saturation=0.15,
            hue=0.04,
        ),
    ], p=0.50),
    transforms.ToTensor(),
    transforms.RandomErasing(
        p=0.15,
        scale=(0.01, 0.06),
        ratio=(0.5, 2.0),
        value="random",
    ),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])



from pathlib import Path
from typing import Callable, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset


class FFPPDataset(Dataset):
    FAKE_FOLDERS = [
        "Deepfakes",
        "Face2Face",
        "FaceShifter",
        "FaceSwap",
        "NeuralTextures",
    ]
    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        return_two_views: bool = True,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.return_two_views = return_two_views
        self.samples: list[Tuple[Path, int]] = []

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Not found: {self.root_dir}")

        self._add_images("Real", 0)
        for folder_name in self.FAKE_FOLDERS:
            self._add_images(folder_name, 1)

        if not self.samples:
            raise RuntimeError(f"No images found in {self.root_dir}")

        real_count = sum(label == 0 for _, label in self.samples)
        fake_count = len(self.samples) - real_count
        print(f"FF++ | Total: {len(self.samples):,} | Real: {real_count:,} | Fake: {fake_count:,}")

    def _add_images(self, folder_name: str, label: int) -> None:
        folder = self.root_dir / folder_name
        if not folder.exists():
            raise FileNotFoundError(f"Not found: {folder}")
        paths = sorted(
            path for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in self.VALID_EXTENSIONS
        )
        self.samples.extend((path, label) for path in paths)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        if self.transform is None:
            return image, label

        view_1 = self.transform(image)
        if self.return_two_views:
            view_2 = self.transform(image)
            return view_1, view_2, label

        return view_1, label


class CelebDFDataset(Dataset):
    CLASS_TO_LABEL = {"real": 0, "fake": 1}
    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root_dir: str, transform: Optional[Callable] = None) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: list[Tuple[Path, int]] = []

        for class_name, label in self.CLASS_TO_LABEL.items():
            folder = self.root_dir / class_name
            if not folder.exists():
                raise FileNotFoundError(f"Not found: {folder}")
            paths = sorted(
                path for path in folder.rglob("*")
                if path.is_file() and path.suffix.lower() in self.VALID_EXTENSIONS
            )
            self.samples.extend((path, label) for path in paths)

        if not self.samples:
            raise RuntimeError(f"No images found in {self.root_dir}")

        real_count = sum(label == 0 for _, label in self.samples)
        fake_count = len(self.samples) - real_count
        print(f"Celeb-DF | Total: {len(self.samples):,} | Real: {real_count:,} | Fake: {fake_count:,}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label



from typing import List, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


# ============================================================
# 1. SRM FILTER
# ============================================================

class SRMFilter(nn.Module):
    """
    Three fixed SRM-style high-pass filters.

    Input:
        x: B x 3 x H x W

    Output:
        residual: B x 3 x H x W

    Each kernel is applied to a grayscale image to produce
    three distinct residual maps.
    """

    def __init__(
        self,
        learnable: bool = False,
        clamp_value: float | None = 3.0,
    ) -> None:
        super().__init__()

        # Kernel 1: second-order residual.
        kernel_1 = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 2.0, -1.0, 0.0],
                [0.0, 2.0, -4.0, 2.0, 0.0],
                [0.0, -1.0, 2.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ) / 4.0

        # Kernel 2: extended Laplacian high-pass filter.
        kernel_2 = torch.tensor(
            [
                [-1.0, 2.0, -2.0, 2.0, -1.0],
                [2.0, -6.0, 8.0, -6.0, 2.0],
                [-2.0, 8.0, -12.0, 8.0, -2.0],
                [2.0, -6.0, 8.0, -6.0, 2.0],
                [-1.0, 2.0, -2.0, 2.0, -1.0],
            ],
            dtype=torch.float32,
        ) / 12.0

        # Kernel 3: local residual.
        kernel_3 = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0, 0.0],
                [0.0, -1.0, 4.0, -1.0, 0.0],
                [0.0, 0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ) / 4.0

        kernels = torch.stack(
            [kernel_1, kernel_2, kernel_3],
            dim=0,
        ).unsqueeze(1)

        # Shape: 3 x 1 x 5 x 5
        self.weight = nn.Parameter(
            kernels,
            requires_grad=learnable,
        )

        self.clamp_value = clamp_value

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError(
                "SRMFilter expects input shape B x 3 x H x W, "
                f"received {tuple(x.shape)}."
            )

        # RGB -> grayscale.
        gray = (
            0.299 * x[:, 0:1]
            + 0.587 * x[:, 1:2]
            + 0.114 * x[:, 2:3]
        )

        residual = F.conv2d(
            gray,
            self.weight,
            bias=None,
            stride=1,
            padding=2,
        )

        if self.clamp_value is not None:
            residual = residual.clamp(
                min=-self.clamp_value,
                max=self.clamp_value,
            )

        return residual


# ============================================================
# 2. CNN COMPONENTS
# ============================================================

class ConvNormAct(nn.Module):
    """Conv2d -> BatchNorm -> GELU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__()

        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class DepthwiseSeparableBlock(nn.Module):
    """
    Depthwise convolution -> Pointwise convolution
    with a residual connection when the shapes match.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.depthwise = ConvNormAct(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=stride,
            groups=in_channels,
        )

        self.pointwise = ConvNormAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
        )

        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.output_activation = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)

        x = self.depthwise(x)
        x = self.pointwise(x)

        return self.output_activation(x + identity)


class MultiScaleCNNExtractor(nn.Module):
    """
    Multi-scale CNN used for both the RGB and SRM branches.

    For a 224 x 224 input:
        Stage 1: 56 x 56
        Stage 2: 28 x 28
        Stage 3: 14 x 14

    The features are then adaptively pooled to match the DINOv2
    patch-grid size, for example 16 x 16 for a 224 x 224 image.
    """

    def __init__(
        self,
        in_channels: int = 3,
        channels: Sequence[int] = (64, 128, 256),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        c1, c2, c3 = channels

        self.stem = nn.Sequential(
            ConvNormAct(
                in_channels,
                c1,
                kernel_size=3,
                stride=2,
            ),
            DepthwiseSeparableBlock(
                c1,
                c1,
                stride=1,
            ),
        )

        self.stage_1 = nn.Sequential(
            DepthwiseSeparableBlock(
                c1,
                c1,
                stride=2,
            ),
            DepthwiseSeparableBlock(
                c1,
                c1,
                stride=1,
            ),
        )

        self.stage_2 = nn.Sequential(
            DepthwiseSeparableBlock(
                c1,
                c2,
                stride=2,
            ),
            DepthwiseSeparableBlock(
                c2,
                c2,
                stride=1,
            ),
        )

        self.stage_3 = nn.Sequential(
            DepthwiseSeparableBlock(
                c2,
                c3,
                stride=2,
            ),
            DepthwiseSeparableBlock(
                c3,
                c3,
                stride=1,
            ),
        )

        self.dropout = nn.Dropout2d(dropout)

        self.out_channels = tuple(channels)

    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.stem(x)

        feature_1 = self.stage_1(x)
        feature_2 = self.stage_2(feature_1)
        feature_3 = self.stage_3(feature_2)

        feature_1 = self.dropout(feature_1)
        feature_2 = self.dropout(feature_2)
        feature_3 = self.dropout(feature_3)

        return [feature_1, feature_2, feature_3]


# ============================================================
# 3. FEATURE MAP -> PATCH TOKENS
# ============================================================

class FeatureMapToTokens(nn.Module):
    """
    Convert a CNN feature map into local tokens.

    B x C x H x W
        -> AdaptivePool(grid_h, grid_w)
        -> Conv 1x1
        -> B x N x local_dim
    """

    def __init__(
        self,
        in_channels: int,
        token_dim: int = 256,
    ) -> None:
        super().__init__()

        self.projection = nn.Sequential(
            nn.Conv2d(
                in_channels,
                token_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(token_dim),
            nn.GELU(),
        )

        self.norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        feature_map: Tensor,
        grid_size: Tuple[int, int],
    ) -> Tensor:
        feature_map = F.adaptive_avg_pool2d(
            feature_map,
            output_size=grid_size,
        )

        feature_map = self.projection(feature_map)

        # B x C x H x W -> B x HW x C
        tokens = feature_map.flatten(2).transpose(1, 2)
        tokens = self.norm(tokens)

        return tokens


# ============================================================
# 4. RGB-SRM GATED FUSION
# ============================================================

class GatedLocalFusion(nn.Module):
    """
    Learn a gate for each token and channel:

        fused = gate * RGB + (1 - gate) * SRM
    """

    def __init__(
        self,
        token_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.rgb_norm = nn.LayerNorm(token_dim)
        self.srm_norm = nn.LayerNorm(token_dim)

        self.gate = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim, token_dim),
            nn.Sigmoid(),
        )

        self.output = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(token_dim),
        )

    def forward(
        self,
        rgb_tokens: Tensor,
        srm_tokens: Tensor,
    ) -> Tensor:
        if rgb_tokens.shape != srm_tokens.shape:
            raise ValueError(
                "RGB and SRM tokens must have the same shape, "
                f"received {tuple(rgb_tokens.shape)} and "
                f"{tuple(srm_tokens.shape)}."
            )

        rgb_tokens = self.rgb_norm(rgb_tokens)
        srm_tokens = self.srm_norm(srm_tokens)

        gate = self.gate(
            torch.cat(
                [rgb_tokens, srm_tokens],
                dim=-1,
            )
        )

        fused_tokens = (
            gate * rgb_tokens
            + (1.0 - gate) * srm_tokens
        )

        return self.output(fused_tokens)


# ============================================================
# 5. SPATIAL RESIDUAL CROSS-ATTENTION
# ============================================================

class SpatialResidualCrossAttention(nn.Module):
    """
    The module consists of two steps:

    1. CNN refinement over DINOv2 patch tokens.
    2. Cross-attention:
           Query = DINO patch tokens
           Key, Value = fused RGB-SRM local tokens

    CLS and register tokens remain unchanged.
    """

    def __init__(
        self,
        dino_dim: int = 768,
        local_dim: int = 256,
        attention_dim: int = 256,
        num_heads: int = 8,
        conv_bottleneck: int = 192,
        dropout: float = 0.1,
        initial_gate: float = 1e-3,
    ) -> None:
        super().__init__()

        if attention_dim % num_heads != 0:
            raise ValueError(
                "attention_dim must be divisible by num_heads."
            )

        self.dino_dim = dino_dim

        # --------------------------------------------------
        # CNN refinement over patch tokens.
        # --------------------------------------------------
        self.patch_norm_1 = nn.LayerNorm(dino_dim)

        self.token_conv = nn.Sequential(
            nn.Conv2d(
                dino_dim,
                conv_bottleneck,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(conv_bottleneck),
            nn.GELU(),

            nn.Conv2d(
                conv_bottleneck,
                conv_bottleneck,
                kernel_size=3,
                padding=1,
                groups=conv_bottleneck,
                bias=False,
            ),
            nn.BatchNorm2d(conv_bottleneck),
            nn.GELU(),

            nn.Conv2d(
                conv_bottleneck,
                dino_dim,
                kernel_size=1,
                bias=True,
            ),
        )

        # --------------------------------------------------
        # Cross-attention
        # --------------------------------------------------
        self.patch_norm_2 = nn.LayerNorm(dino_dim)
        self.local_norm = nn.LayerNorm(local_dim)

        self.query_projection = nn.Linear(
            dino_dim,
            attention_dim,
        )

        self.key_projection = nn.Linear(
            local_dim,
            attention_dim,
        )

        self.value_projection = nn.Linear(
            local_dim,
            attention_dim,
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.attention_output = nn.Sequential(
            nn.Linear(attention_dim, dino_dim),
            nn.Dropout(dropout),
        )

        # Two learned gates.
        #
        # Use a small initialization instead of setting both the gate and output projection
        # to zero, which would prevent initial gradient flow.
        self.conv_gate = nn.Parameter(
            torch.tensor(float(initial_gate))
        )

        self.attention_gate = nn.Parameter(
            torch.tensor(float(initial_gate))
        )

    def forward(
        self,
        tokens: Tensor,
        local_tokens: Tensor,
        grid_size: Tuple[int, int],
        num_prefix_tokens: int = 1,
    ) -> Tensor:
        if tokens.ndim != 3:
            raise ValueError(
                "tokens must have shape B x N x C."
            )

        prefix_tokens = tokens[:, :num_prefix_tokens]
        patch_tokens = tokens[:, num_prefix_tokens:]

        batch_size, num_patches, channels = patch_tokens.shape
        grid_height, grid_width = grid_size

        if channels != self.dino_dim:
            raise ValueError(
                f"Expected DINO dimension {self.dino_dim}, "
                f"received {channels}."
            )

        if num_patches != grid_height * grid_width:
            raise ValueError(
                f"Patch count {num_patches} does not match "
                f"grid {grid_height} x {grid_width}."
            )

        if local_tokens.shape[1] != num_patches:
            raise ValueError(
                "Local token count must equal DINO patch count. "
                f"Received {local_tokens.shape[1]} and {num_patches}."
            )

        # --------------------------------------------------
        # A. CNN refinement
        # --------------------------------------------------
        normalized_patch_tokens = self.patch_norm_1(
            patch_tokens
        )

        patch_map = (
            normalized_patch_tokens
            .transpose(1, 2)
            .reshape(
                batch_size,
                channels,
                grid_height,
                grid_width,
            )
        )

        conv_map = self.token_conv(patch_map)

        conv_tokens = (
            conv_map
            .flatten(2)
            .transpose(1, 2)
        )

        patch_tokens = (
            patch_tokens
            + self.conv_gate * conv_tokens
        )

        # --------------------------------------------------
        # B. Cross-attention
        # --------------------------------------------------
        query = self.query_projection(
            self.patch_norm_2(patch_tokens)
        )

        normalized_local_tokens = self.local_norm(
            local_tokens
        )

        key = self.key_projection(
            normalized_local_tokens
        )

        value = self.value_projection(
            normalized_local_tokens
        )

        attended_tokens, _ = self.cross_attention(
            query=query,
            key=key,
            value=value,
            need_weights=False,
        )

        attended_tokens = self.attention_output(
            attended_tokens
        )

        patch_tokens = (
            patch_tokens
            + self.attention_gate * attended_tokens
        )

        return torch.cat(
            [prefix_tokens, patch_tokens],
            dim=1,
        )


# ============================================================
# 6. COMPLETE MODEL
# ============================================================


class DINOv2SRCADetector(nn.Module):
    """
    Detector designed for cross-dataset generalization:

    DINOv2 ViT-B/14
      + RGB/SRM local multi-scale features
      + gated SRCA
      + semantic/forensic disentanglement
      + invariant embedding for consistency learning.

    forward(..., return_features=True) returns logits and the features required
    by the auxiliary losses.
    """

    def __init__(
        self,
        num_classes: int = 1,
        pretrained: bool = True,
        freeze_dino: bool = True,
        train_dino_norms: bool = False,
        insert_positions: Sequence[int] = (2, 5, 8),
        local_dim: int = 256,
        attention_dim: int = 256,
        attention_heads: int = 8,
        conv_bottleneck: int = 192,
        cnn_channels: Sequence[int] = (48, 96, 192),
        projection_dim: int = 256,
        classifier_hidden_dim: int = 384,
        dropout: float = 0.2,
        learnable_srm: bool = False,
        force_dino_eval: bool = True,
    ) -> None:
        super().__init__()

        if len(insert_positions) != 3:
            raise ValueError("Exactly three SRCA insertion positions are required.")
        if tuple(sorted(insert_positions)) != tuple(insert_positions):
            raise ValueError("insert_positions must be strictly increasing.")

        self.insert_positions = tuple(insert_positions)
        self.force_dino_eval = force_dino_eval
        self.freeze_dino_backbone = freeze_dino

        self.dino = torch.hub.load(
            repo_or_dir="facebookresearch/dinov2",
            model="dinov2_vitb14",
            pretrained=pretrained,
        )
        self.dino_dim = int(self.dino.embed_dim)
        self.patch_size = self._get_patch_size()
        self.num_register_tokens = int(
            getattr(self.dino, "num_register_tokens", 0)
        )
        self.num_prefix_tokens = 1 + self.num_register_tokens

        self.srm_filter = SRMFilter(learnable=learnable_srm)

        self.rgb_extractor = MultiScaleCNNExtractor(
            in_channels=3,
            channels=cnn_channels,
            dropout=0.05,
        )
        self.srm_extractor = MultiScaleCNNExtractor(
            in_channels=3,
            channels=cnn_channels,
            dropout=0.05,
        )

        self.rgb_tokenizers = nn.ModuleList([
            FeatureMapToTokens(channel, local_dim)
            for channel in cnn_channels
        ])
        self.srm_tokenizers = nn.ModuleList([
            FeatureMapToTokens(channel, local_dim)
            for channel in cnn_channels
        ])
        self.local_fusions = nn.ModuleList([
            GatedLocalFusion(local_dim, dropout)
            for _ in range(3)
        ])

        self.srca_modules = nn.ModuleList([
            SpatialResidualCrossAttention(
                dino_dim=self.dino_dim,
                local_dim=local_dim,
                attention_dim=attention_dim,
                num_heads=attention_heads,
                conv_bottleneck=conv_bottleneck,
                dropout=dropout,
                initial_gate=1e-3,
            )
            for _ in range(3)
        ])

        final_cnn_channels = int(cnn_channels[-1])
        self.rgb_global_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(final_cnn_channels, local_dim),
            nn.LayerNorm(local_dim),
            nn.GELU(),
        )
        self.srm_global_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(final_cnn_channels, local_dim),
            nn.LayerNorm(local_dim),
            nn.GELU(),
        )
        self.global_cnn_fusion = nn.Sequential(
            nn.Linear(local_dim * 2, local_dim),
            nn.LayerNorm(local_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Separate projections for measuring semantic-forensic orthogonality.
        self.semantic_projection = nn.Sequential(
            nn.Linear(self.dino_dim * 2, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )
        self.forensic_projection = nn.Sequential(
            nn.Linear(local_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
        )

        # Invariant embedding used for the consistency loss.
        self.invariant_projection = nn.Sequential(
            nn.Linear(projection_dim * 2, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, classifier_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim // 2, num_classes),
        )

        self._initialize_new_layers()
        if freeze_dino:
            self.freeze_dino(train_norms=train_dino_norms)

    def _get_patch_size(self) -> int:
        patch_size = self.dino.patch_embed.patch_size
        if isinstance(patch_size, tuple):
            if patch_size[0] != patch_size[1]:
                raise ValueError("Only square patches are supported.")
            return int(patch_size[0])
        return int(patch_size)

    def _initialize_new_layers(self) -> None:
        excluded_prefix = "dino."
        for name, module in self.named_modules():
            if name.startswith(excluded_prefix):
                continue
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def freeze_dino(self, train_norms: bool = False) -> None:
        for parameter in self.dino.parameters():
            parameter.requires_grad = False
        if train_norms:
            for module in self.dino.modules():
                if isinstance(module, nn.LayerNorm):
                    for parameter in module.parameters():
                        parameter.requires_grad = True
        self.freeze_dino_backbone = True

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_dino_backbone and self.force_dino_eval:
            self.dino.eval()
        return self

    def _get_patch_grid(self, images: Tensor) -> Tuple[int, int]:
        height, width = images.shape[-2:]
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                f"Image dimensions {(height, width)} must be divisible by "
                f"the patch size {self.patch_size}."
            )
        return height // self.patch_size, width // self.patch_size

    @staticmethod
    def _imagenet_denormalize(images: Tensor) -> Tensor:
        mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (images * std + mean).clamp(0.0, 1.0)

    def _build_local_tokens(
        self,
        images: Tensor,
        grid_size: Tuple[int, int],
    ) -> Tuple[List[Tensor], Tensor]:
        rgb_features = self.rgb_extractor(images)

        # SRM should operate in the [0, 1] domain, not on normalized images.
        raw_images = self._imagenet_denormalize(images)
        srm_images = self.srm_filter(raw_images)
        srm_features = self.srm_extractor(srm_images)

        local_tokens: List[Tensor] = []
        for index in range(3):
            rgb_tokens = self.rgb_tokenizers[index](
                rgb_features[index], grid_size
            )
            srm_tokens = self.srm_tokenizers[index](
                srm_features[index], grid_size
            )
            local_tokens.append(
                self.local_fusions[index](rgb_tokens, srm_tokens)
            )

        rgb_global = self.rgb_global_projection(rgb_features[-1])
        srm_global = self.srm_global_projection(srm_features[-1])
        forensic_feature = self.global_cnn_fusion(
            torch.cat([rgb_global, srm_global], dim=1)
        )
        return local_tokens, forensic_feature

    def forward_features(self, images: Tensor) -> dict[str, Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                f"Input must have shape Bx3xHxW; received {tuple(images.shape)}."
            )

        grid_size = self._get_patch_grid(images)
        local_tokens, cnn_feature = self._build_local_tokens(
            images, grid_size
        )

        tokens = self.dino.prepare_tokens_with_masks(images, masks=None)
        srca_index = 0
        for block_index, block in enumerate(self.dino.blocks):
            tokens = block(tokens)
            if block_index in self.insert_positions:
                tokens = self.srca_modules[srca_index](
                    tokens=tokens,
                    local_tokens=local_tokens[srca_index],
                    grid_size=grid_size,
                    num_prefix_tokens=self.num_prefix_tokens,
                )
                srca_index += 1

        tokens = self.dino.norm(tokens)
        cls_feature = tokens[:, 0]
        patch_tokens = tokens[:, self.num_prefix_tokens:]
        mean_patch_feature = patch_tokens.mean(dim=1)

        semantic_feature = self.semantic_projection(
            torch.cat([cls_feature, mean_patch_feature], dim=1)
        )
        forensic_feature = self.forensic_projection(cnn_feature)

        invariant_feature = self.invariant_projection(
            torch.cat([semantic_feature, forensic_feature], dim=1)
        )
        normalized_invariant = F.normalize(
            invariant_feature, p=2, dim=1
        )

        return {
            "tokens": tokens,
            "cls_feature": cls_feature,
            "patch_tokens": patch_tokens,
            "mean_patch_feature": mean_patch_feature,
            "semantic_feature": semantic_feature,
            "forensic_feature": forensic_feature,
            "invariant_feature": invariant_feature,
            "normalized_invariant": normalized_invariant,
        }

    def forward(
        self,
        images: Tensor,
        return_features: bool = False,
    ):
        features = self.forward_features(images)
        logits = self.classifier(features["invariant_feature"])
        if return_features:
            return {"logits": logits, **features}
        return logits



import os
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.nn import functional as F
from tqdm.auto import tqdm


def calculate_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    predictions = (probabilities >= threshold).astype(np.int64)
    return {
        "acc": float(accuracy_score(labels, predictions)),
        "auc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) >= 2 else float("nan"),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
    }


def orthogonality_loss(semantic_feature: torch.Tensor, forensic_feature: torch.Tensor) -> torch.Tensor:
    semantic = F.normalize(semantic_feature, dim=1)
    forensic = F.normalize(forensic_feature, dim=1)
    return (semantic * forensic).sum(dim=1).abs().mean()


def consistency_loss(output_1: dict, output_2: dict) -> torch.Tensor:
    feature_loss = 1.0 - F.cosine_similarity(
        output_1["normalized_invariant"],
        output_2["normalized_invariant"],
        dim=1,
    ).mean()

    probability_1 = torch.sigmoid(output_1["logits"].float())
    probability_2 = torch.sigmoid(output_2["logits"].float())
    prediction_loss = F.mse_loss(probability_1, probability_2)
    return feature_loss + 0.5 * prediction_loss


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    scaler,
    lambda_cons: float = 0.20,
    lambda_orth: float = 0.05,
    threshold: float = 0.5,
    max_grad_norm: float = 1.0,
):
    model.train()
    running = {"loss": 0.0, "bce": 0.0, "cons": 0.0, "orth": 0.0}
    total_samples = 0
    labels_all, probabilities_all = [], []

    for view_1, view_2, labels in tqdm(train_loader, desc="Training", leave=False):
        view_1 = view_1.to(device, non_blocking=True)
        view_2 = view_2.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True).view(-1)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output_1 = model(view_1, return_features=True)
            output_2 = model(view_2, return_features=True)
            logits_1 = output_1["logits"].view(-1)
            logits_2 = output_2["logits"].view(-1)

            bce = 0.5 * (
                criterion(logits_1, labels)
                + criterion(logits_2, labels)
            )
            cons = consistency_loss(output_1, output_2)
            orth = 0.5 * (
                orthogonality_loss(output_1["semantic_feature"], output_1["forensic_feature"])
                + orthogonality_loss(output_2["semantic_feature"], output_2["forensic_feature"])
            )
            loss = bce + lambda_cons * cons + lambda_orth * orth

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad and p.grad is not None],
            max_grad_norm,
        )
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        total_samples += batch_size
        running["loss"] += loss.item() * batch_size
        running["bce"] += bce.item() * batch_size
        running["cons"] += cons.item() * batch_size
        running["orth"] += orth.item() * batch_size

        mean_logits = 0.5 * (logits_1 + logits_2)
        labels_all.append(labels.detach().cpu())
        probabilities_all.append(torch.sigmoid(mean_logits.float()).detach().cpu())

    metrics = calculate_metrics(
        torch.cat(labels_all).numpy(),
        torch.cat(probabilities_all).numpy(),
        threshold,
    )
    metrics.update({key: value / total_samples for key, value in running.items()})
    return metrics


@torch.inference_mode()
def evaluate(model, loader, criterion, device, threshold: float = 0.5, description: str = "Validation"):
    model.eval()
    running_loss, total_samples = 0.0, 0
    labels_all, probabilities_all = [], []

    for images, labels in tqdm(loader, desc=description, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True).view(-1)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(images).view(-1)
            loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_samples += batch_size
        running_loss += loss.item() * batch_size
        labels_all.append(labels.cpu())
        probabilities_all.append(torch.sigmoid(logits.float()).cpu())

    metrics = calculate_metrics(
        torch.cat(labels_all).numpy(),
        torch.cat(probabilities_all).numpy(),
        threshold,
    )
    metrics["loss"] = running_loss / total_samples
    return metrics



def training_loop(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    scaler,
    num_epochs: int = 30,
    early_stopping_patience: int = 6,
    checkpoint_path: str = "checkpoints/best_dinov2_srca_generalized.pth",
    threshold: float = 0.5,
):
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    best_auc = float("-inf")
    patience_count = 0
    history = []

    for epoch in range(1, num_epochs + 1):
        # Ensure DINO gradients are never accidentally enabled.
        assert not any(p.requires_grad for p in model.dino.parameters())

        # Ramp up auxiliary losses so early training prioritizes a stable classifier.
        lambda_cons = min(0.20, 0.05 + 0.03 * (epoch - 1))
        lambda_orth = min(0.05, 0.01 + 0.01 * (epoch - 1))

        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            lambda_cons=lambda_cons,
            lambda_orth=lambda_orth,
            threshold=threshold,
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=threshold,
            description="Celeb-DF validation",
        )

        scheduler.step(val_metrics["auc"])

        learning_rates = {
            group.get("name", f"group_{index}"): group["lr"]
            for index, group in enumerate(optimizer.param_groups)
        }
        print(f"\nEpoch [{epoch:02d}/{num_epochs:02d}] | DINO: fully frozen")
        print("LR | " + " | ".join(f"{name}: {lr:.2e}" for name, lr in learning_rates.items()))
        print(
            "Train | "
            f"Loss: {train_metrics['loss']:.4f} | BCE: {train_metrics['bce']:.4f} | "
            f"Cons: {train_metrics['cons']:.4f} | Orth: {train_metrics['orth']:.4f} | "
            f"Acc: {train_metrics['acc']:.4f} | AUC: {train_metrics['auc']:.4f} | "
            f"F1: {train_metrics['f1']:.4f}"
        )
        print(
            "Valid | "
            f"Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['acc']:.4f} | "
            f"AUC: {val_metrics['auc']:.4f} | F1: {val_metrics['f1']:.4f} | "
            f"Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f}"
        )

        history.append({
            "epoch": epoch,
            "dino_frozen": True,
            "lambda_cons": lambda_cons,
            "lambda_orth": lambda_orth,
            "train": train_metrics,
            "val": val_metrics,
        })

        if val_metrics["auc"] > best_auc + 1e-4:
            best_auc = val_metrics["auc"]
            patience_count = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": best_auc,
                    "history": history,
                },
                checkpoint_path,
            )
            print(f"Saved best model | Epoch: {epoch} | Val AUC: {best_auc:.4f}")
        else:
            patience_count += 1
            print(f"Early stopping: {patience_count}/{early_stopping_patience}")

        if patience_count >= early_stopping_patience:
            print("Early stopping triggered.")
            break

    return history


def build_model(pretrained: bool = True) -> DINOv2SRCADetector:
    """Build the detector with the exact architecture used by the notebook."""
    return DINOv2SRCADetector(
        pretrained=pretrained,
        freeze_dino=True,
        train_dino_norms=False,
        insert_positions=(2, 5, 8),
        cnn_channels=(48, 96, 192),
        local_dim=256,
        projection_dim=256,
        classifier_hidden_dim=384,
        dropout=0.2,
    )


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Train DINOv2-SRCA deepfake detector")
    parser.add_argument("--train-root", required=True, help="FF++ train directory")
    parser.add_argument("--val-root", required=True, help="Validation directory")
    parser.add_argument(
        "--val-format",
        choices=("celebdf", "ffpp"),
        default="celebdf",
        help="Validation class layout (notebook default: celebdf)",
    )
    parser.add_argument("--checkpoint", default="checkpoints/best_dinov2_srca_generalized.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Do not load pretrained DINOv2 weights (mainly for debugging)",
    )
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
    pin_memory = device.type == "cuda"
    print(f"Device: {device}")

    train_dataset = FFPPDataset(args.train_root, train_transform, return_two_views=True)
    if args.val_format == "ffpp":
        val_dataset = FFPPDataset(args.val_root, val_transform, return_two_views=False)
    else:
        val_dataset = CelebDFDataset(args.val_root, val_transform)

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, drop_last=True, **loader_options
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, shuffle=False, drop_last=False, **loader_options
    )

    model = build_model(pretrained=not args.no_pretrained).to(device)
    criterion = nn.BCEWithLogitsLoss()
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert not any(parameter.requires_grad for parameter in model.dino.parameters())
    optimizer = torch.optim.AdamW([
        {
            "params": trainable_parameters,
            "lr": 1e-4,
            "weight_decay": 1e-4,
            "name": "trainable_modules",
        }
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-7
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"Total: {total / 1e6:.2f}M | Trainable: {trainable / 1e6:.2f}M")
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

