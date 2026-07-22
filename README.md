# DINOv2-SRCA deepfake detector

Project duoc tach tu `DINOv2_SRCA_generalized.ipynb`.

## Dataset

Thu muc train theo cau truc FF++:

```text
dataset_root/
  Real/
  Deepfakes/
  Face2Face/
  FaceShifter/
  FaceSwap/
  NeuralTextures/
```

Anh co the nam truc tiep hoac trong cac thu muc con cua tung class.
Mac dinh validation cua `train.py` giong notebook, theo cau truc
`real/` va `fake/` cua Celeb-DF. De validation bang cau truc FF++ o tren,
them `--val-format ffpp`.

## Train

```bash
python train_dino.py `
  --train-root /path/to/train `
  --val-root /path/to/valid `
  --checkpoint checkpoints/best_dinov2_srca_generalized.pth
```

## Train DINOv2 + ArcFace Buffalo_L (GPU)

ArcFace su dung CUDAExecutionProvider cua ONNX Runtime. Can cai
onnxruntime-gpu va CUDA/cuDNN tuong thich truoc khi train.

```bash
python train_dino_arcface.py `
  --train-root /path/to/train `
  --val-root /path/to/valid `
  --arcface-provider cuda `
  --checkpoint checkpoints/best_dinov2_srca_arcface.pth
```

## Validate checkpoint tren bo du lieu FF++ khac

```bash
python test_origin.py `
  --data-root /path/to/dataset `
  --checkpoint checkpoints/best_dinov2_srca_generalized.pth
```

Them `--help` de xem tat ca tuy chon. Lan chay dau tien can mang de tai
ma nguon va pretrained weights cua DINOv2 tu PyTorch Hub. Cai dependencies:

```bash
pip install -r requirements.txt
```


## Test WDF
```bash
pip install -r requirements.txt

python test_cross_1.py `
  --data-root /path/to/cross_dataset `
  --checkpoint checkpoints `
  --batch-size 32 `
  --output-json results/cross_1.json
```

## Test UADFV

```bash
python test_cross_2.py `
  --data-root /path/to/dataset_root `
  --checkpoint checkpoints `
  --batch-size 32 `
  --output-json results/cross_2.json
```
