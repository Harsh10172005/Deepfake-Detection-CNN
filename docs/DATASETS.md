# Datasets

This project trains and evaluates on **Celeb-DF v2** — and on nothing else.
This document records what the dataset is, where it comes from, and exactly
how it is used here, so every number in the README can be traced back to
real data.

## Celeb-DF v2

| | |
|---|---|
| **Name** | Celeb-DF (v2) |
| **Paper** | Li, Yang, Sun, Qi, Lyu — *Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics*, CVPR 2020 ([arXiv:1909.12962](https://arxiv.org/abs/1909.12962)) |
| **Content** | Real face videos of celebrities (collected from YouTube and celebrity footage) and high-quality face-swap deepfakes generated from them. The fakes were specifically produced with improved synthesis (better masks, colour consistency, temporal flicker reduction), which makes Celeb-DF noticeably harder than first-generation deepfake datasets. |
| **How it was obtained here** | Downloaded from the Kaggle competition page [kaggle.com/c/deepfake-detection](https://www.kaggle.com/c/deepfake-detection) ("Download All", a single 10.16 GB archive containing 6,530 files in one `deep-fake-dataset` folder — a community repack of Celeb-DF v2 distributed under that competition's name). The download requires a free Kaggle account and accepting the competition rules. |
| **License note** | The dataset is **not** redistributed in this repository — no videos or face crops are committed. Anyone who wants to reproduce the training must obtain the data themselves, as above, and respect the original Celeb-DF terms (research use). |

### Folder layout of the download, as used here

- `Celeb-Youtube-fake/` — face-swap deepfake videos → training pool, label **fake**.
- `YouTube-real/` — real face videos → training pool, label **real**.
- `Celeb-real/` — real celebrity footage → added to the training pool in
  round 2 (all 482 videos), so the pool's real side also covers the
  source the official test set's real videos come from.
- `test/` — the **official Celeb-DF test set**: 518 videos. Labels follow the
  official Celeb-DF filename protocol (a name containing two `id` tokens,
  e.g. `id2_id9_0008.mp4`, is a face-swap between two identities → **fake**;
  a single-identity name → **real**). Counting by that rule gives
  178 real / 340 fake, matching the published test protocol.
### Subsets used in this project

Compute-bounded subsets (training runs on CPU), selected deterministically:

- **Round 1 (shipped model) training pool:** 200 fake videos (from
  `Celeb-Youtube-fake/`) + 150 real videos (from `YouTube-real/`).
- **Round 2 training pool:** 600 fake videos + 632 real videos
  (150 `YouTube-real` + all 482 `Celeb-real`) — 36,434 face crops from
  1,231 videos (train 30,974 / validation 5,460, split by video).
- **Test set:** 128 videos from the official test folder — 64 real + 64 fake,
  evenly spaced through the sorted official list, so the evaluated subset
  is balanced and covers the whole list rather than one block of it.
- Test labels come from `test_labels.csv`, generated from the official
  filename rule for all 518 test videos and checked against the rule counts
  above.

*Exact face-crop counts per split/class for the reported run are stated in
the README's results section, in `results/metrics.json`, and in the prep
pipeline's `manifest.csv`.*

### From videos to images (`scripts/prepare_faceforensics.py --source celebdf`)

1. From each video, up to 30 evenly spaced frames are sampled.
2. The largest face in each frame is detected (MTCNN via facenet-pytorch
   is supported; the reported runs used the OpenCV Haar cascade),
   cropped with a 35% margin, and saved as a 256×256 JPEG (quality 92).
3. Every crop is logged in the run's `manifest.csv` (path, label, split,
   source video, frame index) so the dataset is fully auditable.

### Split protocol — no frame leakage

The split is assigned **per video, never per frame**: every frame of a
video lands in exactly one split.

- Each training pool (350 videos in round 1; 1,232 in round 2) is split
  **train / validation = 85% / 15%** by video, with a fixed seed (42).
- The test set is **only** the official Celeb-DF test subset above — pool
  videos never enter it, and test videos are never trained on.

Splitting by frames instead would put near-identical frames of the same
person in both train and test and inflate the measured accuracy — a common
way deepfake projects accidentally report meaningless numbers.

## Datasets considered and deliberately NOT used

- **FaceForensics++** (Rössler et al., ICCV 2019): the other major
  benchmark, and a planned future upgrade. It is gated (personal access
  form + research license via TUM, approval takes days), so it could not
  be the primary dataset on this project's timeline. The prep script keeps
  an `ffpp` mode so it can be added later with the same pipeline.
- **DFDC** (Facebook/Meta Deepfake Detection Challenge): large and strong,
  but far bigger than this CPU-bounded rebuild needs; Celeb-DF's official
  test protocol gives a cleaner, citable evaluation.
- **CIFAKE** (real photos vs Stable-Diffusion images of objects, animals,
  vehicles, …): a good general AI-image dataset, but it is not face data,
  so it cannot support a *face deepfake* claim.
- **140k Real and Fake Faces** (FFHQ photographs vs StyleGAN faces):
  face-only and easy to obtain, but entirely GAN-generated stills — a
  narrower and easier task than video face manipulation.
- The small synthetic "dataset" bundled with the original version of this
  project (200 procedurally generated noise images) was discarded: it was
  placeholder data, not real imagery, and no results were ever measured on
  it.
