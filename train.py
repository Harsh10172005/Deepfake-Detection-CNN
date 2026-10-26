"""Train the CNN deepfake detector.

Example:
    python train.py --data_dir data/faces/train \
        --model_name efficientnet_b0 --epochs 3 --batch_size 32

Checkpoints are written to --save_dir as best_<model_name>.pth and contain
the model weights, optimizer state, best validation AUC, the epoch, and the
full training arguments — everything needed to reproduce or resume a run.
The best checkpoint is selected by validation AUC on a held-out split that
is carved out of the training data with a fixed seed (see build_splits);
the test set is never touched here — that is evaluate.py's job.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import DeepfakeDataset, get_transforms
from models import get_model


class FocalLoss(nn.Module):
    """Focal loss with label smoothing (Lin et al., 2017).

    Down-weights easy examples so training focuses on hard, ambiguous
    images; label smoothing reduces overconfident logits.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', label_smoothing=0.05):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        return focal_loss


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return _summarize(running_loss, all_labels, all_preds, dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return _summarize(running_loss, all_labels, all_preds, dataloader)


def _summarize(running_loss, all_labels, all_preds, dataloader):
    epoch_loss = running_loss / len(dataloader.dataset)
    try:
        epoch_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        epoch_auc = 0.0
    epoch_acc = accuracy_score(all_labels, np.array(all_preds) > 0.5)
    return epoch_loss, epoch_acc, epoch_auc


def build_splits(data_dir, train_transform, val_transform, val_split, seed, max_samples=None, val_dir=None):
    """Build train/val datasets.

    If ``val_dir`` is given, it is used as the validation set as-is and
    ``data_dir`` is used entirely for training. This is the preferred mode
    when the data was already split at the VIDEO level by the preparation
    script: no frame of a validation video ever appears in training, so
    checkpoint selection cannot leak either. Without ``val_dir``, a seeded
    image-level split is carved from ``data_dir`` (fine for smoke tests;
    the reported metrics always come from the separate held-out test set).
    """
    if val_dir:
        train_ds = DeepfakeDataset(data_dir, transform=train_transform, max_samples=max_samples)
        if len(train_ds) == 0:
            return None, None
        val_ds = DeepfakeDataset(val_dir, transform=val_transform, max_samples=max_samples)
        return train_ds, val_ds
    # Deterministic image-level split with NO transform sharing: two
    # separate DeepfakeDataset instances are created (identical, sorted
    # sample order) so the training subset keeps the training augmentations
    # and the validation subset the clean val transform. (Mutating
    # ``subset.dataset.transform`` on a shared dataset — the previous
    # implementation — silently changed the training transform too.)
    train_ds = DeepfakeDataset(data_dir, transform=train_transform, max_samples=max_samples)
    if len(train_ds) == 0:
        return None, None
    val_ds = DeepfakeDataset(data_dir, transform=val_transform, max_samples=max_samples)

    n = len(train_ds)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()
    val_n = int(val_split * n)
    val_idx, train_idx = indices[:val_n], indices[val_n:]
    return Subset(train_ds, train_idx), Subset(val_ds, val_idx)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else ('mps' if torch.backends.mps.is_available() else 'cpu')
    )
    print(f"Using device: {device}")

    train_transform, val_transform = get_transforms(img_size=args.img_size)

    max_samples = 100 if args.demo_mode else args.max_samples
    train_dataset, val_dataset = build_splits(
        args.data_dir, train_transform, val_transform,
        val_split=args.val_split, seed=args.seed, max_samples=max_samples,
        val_dir=args.val_dir,
    )
    if train_dataset is None:
        print(f"Error: No data found in {args.data_dir}")
        print("Ensure you have 'real' and 'fake' subdirectories inside the data directory.")
        return

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = get_model(args.model_name).to(device)

    if args.init_weights:
        ckpt = torch.load(args.init_weights, map_location=device)
        state = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state)
        print(f"Initialized weights from {args.init_weights} "
              f"(checkpoint epoch: {ckpt.get('epoch') if isinstance(ckpt, dict) else 'n/a'})")

    criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

    best_val_auc = 0.0
    os.makedirs(args.save_dir, exist_ok=True)

    print("Starting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss, train_acc, train_auc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_auc)

        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            save_path = os.path.join(args.save_dir, f'best_{args.model_name}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_auc': best_val_auc,
                'args': vars(args),
            }, save_path)
            print(f"--> Saved new best model to {save_path} (AUC: {best_val_auc:.4f})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Deepfake Detector (CNN)")
    parser.add_argument('--data_dir', type=str, required=True, help="Data directory containing 'real' and 'fake' subfolders")
    parser.add_argument('--model_name', type=str, default='efficientnet_b0', help="CNN backbone (efficientnet_b0, efficientnet_v2_s, efficientnet_v2_m, resnet50)")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay")
    parser.add_argument('--img_size', type=int, default=224, help="Image size for model input")
    parser.add_argument('--val_split', type=float, default=0.2, help="Validation split ratio (carved from the training data; ignored when --val_dir is given)")
    parser.add_argument('--val_dir', type=str, default=None, help="Separate validation directory with 'real' and 'fake' subfolders (e.g. the video-level val split from the prep script). Preferred over --val_split: no shared videos between train and val.")
    parser.add_argument('--init_weights', type=str, default=None, help="Optional checkpoint (.pth) to initialize weights from before training — e.g. to continue fine-tuning a previously saved best checkpoint. Accepts a train.py checkpoint dict or a raw state_dict.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of dataloader workers")
    parser.add_argument('--save_dir', type=str, default='checkpoints', help="Directory to save model checkpoints")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for the split and training")
    parser.add_argument('--max_samples', type=int, default=None, help="Cap on images loaded (balanced per class); useful for CPU training")
    parser.add_argument('--demo_mode', action='store_true', help="Smoke-test mode: only 100 images, 1 epoch")

    args = parser.parse_args()

    if args.demo_mode:
        print("\n!!! RUNNING IN DEMO MODE !!!")
        print("Dataset is limited to 100 images and epochs forced to 1 for rapid execution.\n")
        args.epochs = 1
    main(args)
