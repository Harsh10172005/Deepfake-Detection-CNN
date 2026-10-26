"""Honest, reproducible evaluation of a trained CNN checkpoint.

Runs the model over a held-out dataset directory (real/ + fake/ subfolders)
that training never touched, and writes the measured results to results/:

    results/<out_name>.json                  all metrics + run metadata
    results/confusion_matrix_<out_name>.png  confusion matrix plot
    results/roc_curve_<out_name>.png         ROC curve plot

Nothing here estimates or rounds in the model's favour: the numbers are
exactly what sklearn computes from the model's raw probabilities at the
decision threshold (0.5 unless --threshold says otherwise).

Alongside the per-frame metrics, the JSON also carries a `video_level`
block: each source video's score is the mean of its face-crop
probabilities, thresholded the same way. Per-video scoring is the
standard reporting protocol for video benchmarks such as Celeb-DF,
because a video — not a single frame — is what a detector judges in
practice. Both blocks are computed from the same raw probabilities.

Example:
    python evaluate.py --weights checkpoints/best_cnn_efficientnet_b0.pth \
        --data_dir data/celebdf/test --dataset_name "Celeb-DF official test set (128-video subset)" \
        --out_name metrics
"""
import argparse
import json
import os
import re
from datetime import datetime, timezone

import matplotlib
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from dataset import DeepfakeDataset, get_transforms  # noqa: E402
from inference import load_model  # noqa: E402


@torch.no_grad()
def collect_probs(model, dataloader, device):
    model.eval()
    all_probs, all_labels = [], []
    for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
        outputs = model(inputs.to(device))
        all_probs.extend(torch.sigmoid(outputs).cpu().numpy())
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_probs)


_FRAME_SUFFIX = re.compile(r'^(.+)_f\d+\.jpg$')


def video_id_of(path):
    """Source-video id of a face crop named '<video_id>_f<frame>.jpg'.

    Crops produced by scripts/prepare_faceforensics.py follow that
    naming; any other file is treated as its own one-crop 'video'.
    """
    name = os.path.basename(path)
    m = _FRAME_SUFFIX.match(name)
    return m.group(1) if m else os.path.splitext(name)[0]


def video_level_metrics(labels, probs, paths, threshold):
    """Aggregate per-crop probabilities to one score per source video."""
    groups = {}
    for label, prob, path in zip(labels, probs, paths):
        vid = video_id_of(path)
        entry = groups.setdefault(vid, {'label': int(label), 'probs': []})
        entry['probs'].append(float(prob))
    v_labels = np.array([g['label'] for g in groups.values()])
    v_probs = np.array([float(np.mean(g['probs'])) for g in groups.values()])
    v_preds = (v_probs > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(v_labels, v_preds, labels=[0, 1]).ravel()
    try:
        auc = float(roc_auc_score(v_labels, v_probs))
    except ValueError:
        auc = None
    return {
        'n_videos': int(len(groups)),
        'n_real_videos': int((v_labels == 0).sum()),
        'n_fake_videos': int((v_labels == 1).sum()),
        'threshold': threshold,
        'accuracy': float(accuracy_score(v_labels, v_preds)),
        'precision': float(precision_score(v_labels, v_preds, zero_division=0)),
        'recall': float(recall_score(v_labels, v_preds, zero_division=0)),
        'f1_score': float(f1_score(v_labels, v_preds, zero_division=0)),
        'roc_auc': auc,
        'confusion_matrix': {
            'true_negative': int(tn), 'false_positive': int(fp),
            'false_negative': int(fn), 'true_positive': int(tp),
        },
    }


def plot_confusion_matrix(cm, path):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1], labels=['Pred Real', 'Pred Fake'])
    ax.set_yticks([0, 1], labels=['True Real', 'True Fake'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=13)
    ax.set_title('Confusion Matrix')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc(labels, probs, auc, path):
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.plot(fpr, tpr, color='#2563eb', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], color='#9ca3af', lw=1, linestyle='--', label='Chance')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(args):
    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else ('mps' if torch.backends.mps.is_available() else 'cpu')
    )
    print(f"Using device: {device}")

    _, val_transform = get_transforms(img_size=args.img_size)
    dataset = DeepfakeDataset(args.data_dir, transform=val_transform, max_samples=args.max_samples)
    if len(dataset) == 0:
        raise SystemExit(f"No images found under {args.data_dir} (expected real/ and fake/ subfolders)")

    n_real = sum(1 for _, y in dataset.samples if y == 0)
    n_fake = sum(1 for _, y in dataset.samples if y == 1)
    print(f"Evaluating on {len(dataset)} images ({n_real} real, {n_fake} fake)")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = load_model(args.model_name, args.weights, device)

    labels, probs = collect_probs(model, dataloader, device)
    paths = [p for p, _ in dataset.samples]
    assert len(paths) == len(probs), "sample/probability order mismatch"
    preds = (probs > args.threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    results = {
        'model_name': args.model_name,
        'weights': args.weights,
        'dataset_name': args.dataset_name,
        'dataset_dir': args.data_dir,
        'n_samples': int(len(dataset)),
        'n_real': int(n_real),
        'n_fake': int(n_fake),
        'threshold': args.threshold,
        'accuracy': float(accuracy_score(labels, preds)),
        'precision': float(precision_score(labels, preds, zero_division=0)),
        'recall': float(recall_score(labels, preds, zero_division=0)),
        'f1_score': float(f1_score(labels, preds, zero_division=0)),
        'roc_auc': float(roc_auc_score(labels, probs)),
        'confusion_matrix': {
            'true_negative': int(tn), 'false_positive': int(fp),
            'false_negative': int(fn), 'true_positive': int(tp),
        },
        'video_level': video_level_metrics(labels, probs, paths, args.threshold),
        'evaluated_at': datetime.now(timezone.utc).isoformat(),
        'torch_version': torch.__version__,
        'device': str(device),
    }

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, f'{args.out_name}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    cm = np.array([[tn, fp], [fn, tp]])
    plot_confusion_matrix(cm, os.path.join(args.out_dir, f'confusion_matrix_{args.out_name}.png'))
    plot_roc(labels, probs, results['roc_auc'], os.path.join(args.out_dir, f'roc_curve_{args.out_name}.png'))

    print(f"\n=== Results ({args.dataset_name}) ===")
    print(f"Accuracy : {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall   : {results['recall']:.4f}")
    print(f"F1-score : {results['f1_score']:.4f}")
    print(f"ROC-AUC  : {results['roc_auc']:.4f}")
    print(f"Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
    vl = results['video_level']
    vl_cm = vl['confusion_matrix']
    print(f"Video-level ({vl['n_videos']} videos): "
          f"Accuracy {vl['accuracy']:.4f}  F1 {vl['f1_score']:.4f}  "
          f"ROC-AUC {vl['roc_auc']:.4f}  "
          f"Confusion TN={vl_cm['true_negative']} FP={vl_cm['false_positive']} "
          f"FN={vl_cm['false_negative']} TP={vl_cm['true_positive']}")
    print(f"Saved -> {json_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained CNN checkpoint on a held-out dataset")
    parser.add_argument('--weights', type=str, required=True, help="Path to the .pth checkpoint")
    parser.add_argument('--model_name', type=str, default='efficientnet_b0', help="CNN backbone used for the checkpoint")
    parser.add_argument('--data_dir', type=str, required=True, help="Dataset folder with real/ and fake/ subfolders (never used in training)")
    parser.add_argument('--dataset_name', type=str, default='held-out test set', help="Human-readable dataset label stored in the results")
    parser.add_argument('--max_samples', type=int, default=None, help="Optional cap on evaluated images (balanced per class)")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--threshold', type=float, default=0.5, help="Decision threshold on the fake probability (default 0.5)")
    parser.add_argument('--out_dir', type=str, default='results')
    parser.add_argument('--out_name', type=str, default='metrics', help="Stem for the output files (metrics -> results/metrics.json)")
    main(parser.parse_args())
