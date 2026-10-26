"""Prepare an image dataset from deepfake *videos* (FaceForensics++ or DFDC).

Both source datasets ship as videos; this script turns them into the
real/fake face-crop folders the rest of this project trains on:

    1. discover videos (+ labels) in the source layout,
    2. sample frames from each video,
    3. detect the face in each frame (MTCNN; OpenCV Haar fallback),
    4. save square face crops to  out_dir/{train,val,test}/{real,fake}/,
    5. write manifest.csv recording every crop's source video.

Leakage control (important): the split is assigned per VIDEO GROUP, never
per frame. For FaceForensics++ a manipulated video is grouped with its
target video (and the official FF++ split lists are used when present).
For DFDC a fake is grouped with its `original` video from metadata.json,
so a clip and all manipulations of it stay in the same split.

FaceForensics++ example:
    python scripts/prepare_faceforensics.py --source ffpp \
        --ffpp_root /path/to/FaceForensics++ \
        --methods Deepfakes --compression c23 --out_dir data/ffpp

DFDC example (one extracted competition train part):
    python scripts/prepare_faceforensics.py --source dfdc \
        --dfdc_dir /path/to/dfdc_train_part_00 --out_dir data/dfdc
"""
import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------------------------------
# Video discovery — each item is (video_id, label, path, group_id)
# --------------------------------------------------------------------------

def find_videos_ffpp(root: Path, compression: str, methods):
    originals = root / "original_sequences" / "youtube" / compression / "videos"
    if originals.is_dir():
        for p in sorted(originals.glob("*.mp4")):
            yield p.stem, "real", p, p.stem

    man_root = root / "manipulated_sequences"
    if man_root.is_dir():
        for method_dir in sorted(man_root.iterdir()):
            if methods and method_dir.name not in methods:
                continue
            videos = method_dir / compression / "videos"
            if videos.is_dir():
                for p in sorted(videos.glob("*.mp4")):
                    # '<source>_<target>' — group with the target footage
                    group = p.stem.split("_")[-1]
                    yield p.stem, "fake", p, group


def find_videos_dfdc(folder: Path):
    """DFDC train parts are a flat folder of .mp4 files + metadata.json:
    {filename: {"label": "FAKE"|"REAL", "original": <file>, "split": ...}}"""
    meta_path = folder / "metadata.json"
    if not meta_path.exists():
        raise SystemExit(f"metadata.json not found in {folder} — point --dfdc_dir at an extracted DFDC train part.")
    metadata = json.loads(meta_path.read_text())
    for filename, info in sorted(metadata.items()):
        path = folder / filename
        if not path.exists():
            continue
        label = "fake" if str(info.get("label", "")).upper() == "FAKE" else "real"
        original = info.get("original")
        group = Path(original).stem if (label == "fake" and original) else path.stem
        yield path.stem, label, path, group


def find_videos_celebdf(fake_dir: Path, real_dir: Path):
    """Celeb-DF (Kaggle layout): separate folders of .mp4 files per class.
    File names carry no pairing information (they were renumbered), so each
    video is its own split group."""
    for p in sorted(real_dir.glob("*.mp4")):
        yield p.stem, "real", p, p.stem
    for p in sorted(fake_dir.glob("*.mp4")):
        yield p.stem, "fake", p, p.stem


def find_test_videos_celebdf(test_dir: Path, labels_csv: Path | None):
    """The official Celeb-DF test set: labels come from a CSV
    (filename,label) when provided, else from the official naming rule —
    '<idA>_<idB>_<n>.mp4' (two identity tokens) is a face-swap (fake);
    anything else is real. The rule reproduces the official test split of
    178 real / 340 fake on the full 518-video set."""
    import re
    labels = {}
    if labels_csv and labels_csv.exists():
        with open(labels_csv) as f:
            for row in csv.DictReader(f):
                labels[row["filename"]] = row["label"]
    for p in sorted(test_dir.glob("*.mp4")):
        label = labels.get(p.name)
        if label is None:
            label = "fake" if len(re.findall(r"id\d+", p.name)) >= 2 else "real"
        # Sentinel group guarantees these always land in the test split.
        yield p.stem, label, p, f"TEST::{p.stem}"


# --------------------------------------------------------------------------
# Splitting (by video group, never by frame)
# --------------------------------------------------------------------------

def official_splits(root: Path):
    """Load official FF++ train/val/test video-id lists if present."""
    splits = {}
    for name in ("train", "val", "test"):
        for candidate in (root / f"{name}.json", root / "splits" / f"{name}.json"):
            if candidate.exists():
                data = json.loads(candidate.read_text())
                ids = set()
                for item in data:
                    if isinstance(item, str):
                        ids.add(item)
                    elif isinstance(item, (list, tuple)):
                        ids.update(str(x) for x in item)
                splits[name] = ids
                break
    return splits if len(splits) == 3 else None


def assign_splits(group_ids, seed: int, ratios=(0.70, 0.15, 0.15), official=None):
    if official:
        print("Using official FaceForensics++ split lists.")
        return {g: name for name, ids in official.items() for g in ids if g in group_ids}

    print("Using a seeded video-level shuffle split.")
    ids = sorted(group_ids)
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    if ratios[2] <= 0:
        n_val = n - n_train  # no test split wanted: remainder goes to val
    assignment = {}
    for g in ids[:n_train]:
        assignment[g] = "train"
    for g in ids[n_train:n_train + n_val]:
        assignment[g] = "val"
    for g in ids[n_train + n_val:]:
        assignment[g] = "test"
    return assignment


# --------------------------------------------------------------------------
# Face detection / cropping
# --------------------------------------------------------------------------

class FaceCropper:
    def __init__(self, detector: str, margin: float = 0.35):
        self.margin = margin
        self._mtcnn = None
        self._haar = None
        if detector == "mtcnn":
            from facenet_pytorch import MTCNN
            import torch
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
            self._mtcnn = MTCNN(keep_all=False, device=device)
        else:
            cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(cascade)

    def largest_face_box(self, frame_rgb):
        """Return (x1, y1, x2, y2) of the largest detected face, or None."""
        if self._mtcnn is not None:
            boxes, _ = self._mtcnn.detect(frame_rgb)
            if boxes is None:
                return None
            areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in boxes]
            return boxes[int(np.argmax(areas))]
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        faces = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return (x, y, x + w, y + h)

    def crop(self, frame_rgb, box, out_size=256):
        h, w = frame_rgb.shape[:2]
        x1, y1, x2, y2 = box
        side = max(x2 - x1, y2 - y1) * (1 + self.margin)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        nx1 = int(max(0, cx - side / 2))
        ny1 = int(max(0, cy - side / 2))
        nx2 = int(min(w, cx + side / 2))
        ny2 = int(min(h, cy + side / 2))
        face = frame_rgb[ny1:ny2, nx1:nx2]
        if face.size == 0:
            return None
        return cv2.resize(face, (out_size, out_size), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------------------------
# Frame sampling + main
# --------------------------------------------------------------------------

def sample_frames(video_path: Path, cap: int):
    """Read up to `cap` evenly spaced frames (RGB) from a video."""
    cap_obj = cv2.VideoCapture(str(video_path))
    total = int(cap_obj.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap_obj.release()
        return
    step = max(1, total // max(1, cap))
    idx = 0
    taken = 0
    while taken < cap:
        cap_obj.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap_obj.read()
        if not ok:
            break
        yield idx, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        taken += 1
        idx += step
    cap_obj.release()


def main(args):
    out_dir = Path(args.out_dir)

    fixed_test_groups = set()
    if args.source == "ffpp":
        root = Path(args.ffpp_root)
        methods = set(m.strip() for m in args.methods.split(",")) if args.methods else set()
        videos = list(find_videos_ffpp(root, args.compression, methods))
        official = official_splits(root)
    elif args.source == "dfdc":
        videos = list(find_videos_dfdc(Path(args.dfdc_dir)))
        official = None
    else:  # celebdf
        videos = list(find_videos_celebdf(Path(args.celeb_fake_dir), Path(args.celeb_real_dir)))
        if args.celeb_test_dir:
            test_videos = list(find_test_videos_celebdf(
                Path(args.celeb_test_dir),
                Path(args.test_labels) if args.test_labels else None))
            fixed_test_groups = {g for _, _, _, g in test_videos}
            videos += test_videos
        official = None

    if not videos:
        raise SystemExit("No videos discovered — check the source path/layout.")
    print(f"Discovered {len(videos)} videos "
          f"({sum(1 for _, l, _, _ in videos if l == 'real')} real, "
          f"{sum(1 for _, l, _, _ in videos if l == 'fake')} fake)")

    groups = {g for _, _, _, g in videos}
    if args.source == "celebdf":
        # Pool splits into train/val only (85/15); the official test set is fixed.
        pool_groups = groups - fixed_test_groups
        assignment = assign_splits(pool_groups, args.seed, ratios=(0.85, 0.15, 0.0))
        assignment.update({g: "test" for g in fixed_test_groups})
    else:
        assignment = assign_splits(groups, args.seed, official=official)
    print("Video groups per split:",
          {s: sum(1 for g in groups if assignment.get(g) == s) for s in ("train", "val", "test")})

    cropper = FaceCropper(args.detector, margin=args.margin)
    rows = []
    counts = {}
    for vid, label, path, group in videos:
        split = assignment.get(group)
        if split is None:
            continue
        dest_dir = out_dir / split / label
        dest_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        for frame_idx, frame in sample_frames(path, args.frames_per_video):
            box = cropper.largest_face_box(frame)
            if box is None:
                continue
            face = cropper.crop(frame, box)
            if face is None:
                continue
            out_path = dest_dir / f"{vid}_f{frame_idx:04d}.jpg"
            cv2.imwrite(str(out_path), cv2.cvtColor(face, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            rows.append({"path": str(out_path.relative_to(out_dir)), "label": label,
                         "split": split, "video_id": vid, "frame": frame_idx})
            saved += 1
        counts[(split, label)] = counts.get((split, label), 0) + saved
        print(f"  {vid} [{label} -> {split}]: {saved} face crops")

    manifest = out_dir / "manifest.csv"
    with open(manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "split", "video_id", "frame"])
        writer.writeheader()
        writer.writerows(rows)

    print("\nCrops per split/class:")
    for split in ("train", "val", "test"):
        print(f"  {split}: real={counts.get((split, 'real'), 0)}, fake={counts.get((split, 'fake'), 0)}")
    print(f"Total crops: {len(rows)} -> {out_dir} (manifest: {manifest})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Turn deepfake videos (FF++, DFDC or Celeb-DF) into face-crop image folders")
    parser.add_argument("--source", default="ffpp", choices=["ffpp", "dfdc", "celebdf"], help="Source dataset layout")
    parser.add_argument("--ffpp_root", default="", help="[ffpp] Folder containing original_sequences/ and manipulated_sequences/")
    parser.add_argument("--dfdc_dir", default="", help="[dfdc] Extracted DFDC train-part folder (mp4 files + metadata.json)")
    parser.add_argument("--celeb_fake_dir", default="", help="[celebdf] Folder of fake .mp4 videos (training pool)")
    parser.add_argument("--celeb_real_dir", default="", help="[celebdf] Folder of real .mp4 videos (training pool)")
    parser.add_argument("--celeb_test_dir", default="", help="[celebdf] Folder with the official test-set .mp4 videos")
    parser.add_argument("--test_labels", default="", help="[celebdf] Optional CSV (filename,label) for the test folder")
    parser.add_argument("--out_dir", default="data/dataset", help="Output root for {train,val,test}/{real,fake}")
    parser.add_argument("--methods", default="Deepfakes", help="[ffpp] Comma-separated manipulation methods to include")
    parser.add_argument("--compression", default="c23", choices=["c0", "c23", "c40"], help="[ffpp] Compression level")
    parser.add_argument("--frames_per_video", type=int, default=40, help="Max frames sampled per video")
    parser.add_argument("--margin", type=float, default=0.35, help="Extra margin around the detected face box")
    parser.add_argument("--detector", default="mtcnn", choices=["mtcnn", "haar"], help="Face detector")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the video-level split")
    main(parser.parse_args())
