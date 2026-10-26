"""End-to-end plumbing tests (no trained weights or real data required).

Run from the project root:
    python tests/test_pipeline.py

Creates a few synthetic fixture images with PIL, then checks that:
  1. every supported CNN backbone builds and produces a scalar logit,
  2. the dataset + transforms yield correctly shaped tensors,
  3. the train/val split keeps per-split transforms separate (regression
     test for the old shared-dataset transform bug),
  4. FocalLoss returns a finite value,
  5. single-image inference returns a probability in [0, 1].
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import DeepfakeDataset, get_transforms  # noqa: E402
from inference import predict_image  # noqa: E402
from models import SUPPORTED_MODELS, get_model  # noqa: E402
from train import FocalLoss, build_splits  # noqa: E402


def make_fixtures(root: Path, per_class=4):
    rng = np.random.default_rng(0)
    for label in ("real", "fake"):
        (root / label).mkdir(parents=True)
        for i in range(per_class):
            arr = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr).save(root / label / f"{label}_{i}.jpg")


def main():
    device = torch.device("cpu")
    x = torch.randn(2, 3, 224, 224)

    print("1) Model construction + forward pass")
    for name in SUPPORTED_MODELS:
        model = get_model(name, pretrained=False)
        out = model(x)
        assert out.shape == (2,), f"{name}: expected (2,), got {tuple(out.shape)}"
        print(f"   {name}: OK {tuple(out.shape)}")

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_fixtures(root)
        train_tf, val_tf = get_transforms()

        print("2) Dataset + transforms")
        ds = DeepfakeDataset(str(root), transform=train_tf)
        assert len(ds) == 8, len(ds)
        img, label = ds[0]
        assert img.shape == (3, 224, 224) and label.item() in (0.0, 1.0)
        print(f"   samples={len(ds)}, tensor={tuple(img.shape)}")

        print("3) Split transform isolation (regression test)")
        train_ds, val_ds = build_splits(str(root), train_tf, val_tf, val_split=0.25, seed=42)
        assert train_ds.dataset.transform is train_tf
        assert val_ds.dataset.transform is val_tf
        assert len(train_ds) + len(val_ds) == 8
        print(f"   train={len(train_ds)}, val={len(val_ds)}, transforms independent")

        print("4) FocalLoss")
        loss = FocalLoss()(torch.randn(4), torch.tensor([0.0, 1.0, 1.0, 0.0]))
        assert torch.isfinite(loss)
        print(f"   loss={loss.item():.4f}")

        print("5) Single-image inference")
        model = get_model("efficientnet_b0", pretrained=False).to(device).eval()
        prob = predict_image(str(root / "real" / "real_0.jpg"), model, val_tf, device)
        assert 0.0 <= prob <= 1.0
        print(f"   probability={prob:.4f} (random weights — plumbing check only)")

    print("\nAll pipeline tests passed.")


if __name__ == "__main__":
    main()
