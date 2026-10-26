"""Dataset and augmentation pipeline.

Expected layout (ImageFolder-style, two classes):

    data_dir/
        real/   *.jpg / *.png   -> label 0
        fake/   *.jpg / *.png   -> label 1

File lists are sorted so dataset order (and therefore any index-based
train/val split) is deterministic across runs and machines.
"""
import glob
import os

import albumentations as A
import numpy as np
import torch
from PIL import Image
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def _list_images(folder):
    files = (
        glob.glob(os.path.join(folder, '*.[jp][pn]*[g]'))
        + glob.glob(os.path.join(folder, '*.jpeg'))
    )
    return sorted(files)


class DeepfakeDataset(Dataset):
    def __init__(self, data_dir, transform=None, max_samples=None):
        """
        data_dir: folder containing 'real' and 'fake' subdirectories.
        transform: an albumentations Compose (see get_transforms).
        max_samples: optional cap; applied per class (half real, half fake)
            so the subset stays balanced.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        real_dir = os.path.join(data_dir, 'real')
        fake_dir = os.path.join(data_dir, 'fake')

        if os.path.exists(real_dir):
            real_files = _list_images(real_dir)
            if max_samples:
                real_files = real_files[:max_samples // 2]
            self.samples.extend((f, 0) for f in real_files)

        if os.path.exists(fake_dir):
            fake_files = _list_images(fake_dir)
            if max_samples:
                fake_files = fake_files[:max_samples // 2]
            self.samples.extend((f, 1) for f in fake_files)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        image = Image.open(path).convert('RGB')
        image = np.array(image)

        if self.transform:
            image = self.transform(image=image)['image']

        return image, torch.tensor(label, dtype=torch.float32)


def get_transforms(img_size=224):
    """Training augmentations and the deterministic validation transform.

    Augmentations simulate real-world degradations that fake/real images
    undergo when shared (recompression, noise, blur) plus geometric variety.
    """
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.ImageCompression(quality_range=(60, 100), p=0.5),
        A.GaussNoise(std_range=(0.04, 0.2), p=0.3),
        A.ISONoise(p=0.3),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(1, int(img_size * 0.1)), hole_width_range=(1, int(img_size * 0.1)), p=0.2),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return train_transform, val_transform
