"""Training data: OCT5k B-scans with all three graders' masks, split into cross-validation folds
by patient.

In plain words: This file feeds the model during training. It reads each B-scan with one grader's
mask, applies small random changes (flip, shift, brightness) so the model learns the anatomy rather
than memorising pictures, and splits the patients into five groups ("folds"): each fold's model is
tested only on patients it never saw.

Choices and their sources:
- Folds by patient, never by scan: B-scans of one eye look alike, so a scan-level split would
  leak the patient into the test set. WRC splits on subject ID the same way (Domalpally et al.,
  Ophthalmol Sci 2024, "Strong versus weak data labeling").
- Balanced sampling of scans with EZ loss (ez_balance): the team's HRF segmentation README
  (Faisal, hrf-oct-segmentation) pairs each rare positive B-scan with negatives. EZ loss is the
  endpoint we care about and appears in only ~5% of scans.
- Horizontal flip + vertical border shift: WRC's oct-evaluation augmentation (MONAI RandFlipd;
  RandAffined padding_mode="border").
- One grader's mask drawn at random per scan per epoch, intensity jitter: ours. Drawing a random
  grader shows the model the real spread between graders instead of one grader's habits.
"""
import csv                               # the manifest
import functools                         # lru_cache: load the data once
from collections import defaultdict      # cohort -> patients
from pathlib import Path                 # file paths

import lightning as L                    # PyTorch Lightning: the training loop
import numpy as np                       # arrays
import torch                             # tensors, random draws
from PIL import Image                    # reading images and masks
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler  # batching and sampling

from gpu_service.unet.unet import preprocess, shift_rows  # shared with the service

EZ = 3  # mask label of the EZ band (IS/OS to IBRPE); see data_prep/build_manifest.py


@functools.lru_cache(maxsize=1)
def load_scans(manifest: str) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Read every graded scan into RAM once: rows, images (N, H, W) and masks (N, 3, 512, 512), uint8.
    Cached, so all folds in one process share a single copy. 1,672 scans fit in ~1.7 GB, so
    training never waits on the disk."""
    root = Path(manifest).parent  # manifest paths are relative to the manifest itself
    rows = list(csv.DictReader(open(manifest)))                                    # one dict per scan
    images = np.stack([np.array(Image.open(root / r["image"])) for r in rows])      # all images, stacked
    masks = np.stack([[np.array(Image.open(root / r[f"mask_g{g}"])) for g in (1, 2, 3)] for r in rows])  # 3 per scan
    return rows, images, masks


def assign_folds(rows: list[dict], n_folds: int, seed: int) -> dict[str, int]:
    """Patient -> fold. Patients are shuffled within each cohort and dealt round-robin, carrying
    on where the previous cohort stopped, so every fold gets a similar mix of AMD, DME and Normal
    (60 patients -> 12 per fold)."""
    by_cohort = defaultdict(set)
    for r in rows:
        by_cohort[r["cohort"]].add(r["patient"])       # AMD -> {AMD01, ...}
    rng, folds, dealt = np.random.default_rng(seed), {}, 0
    for cohort in sorted(by_cohort):  # sorted everywhere: same seed -> same folds, on any machine
        patients = sorted(by_cohort[cohort])
        rng.shuffle(patients)                          # random order within the cohort
        for p in patients:
            folds[p] = dealt % n_folds                 # deal like cards: 0, 1, 2, 3, 4, 0, ...
            dealt += 1
    return folds


def shows_ez_loss(scans) -> np.ndarray:
    """Per scan: do the graders, on average, see EZ missing in more than 5% of image columns?
    scans yields (graders, H, W) masks. About 5% of OCT5k scans qualify, mostly DME.
    The 5% cut-off is ours (a scan-level yes/no for the sampler, not a clinical threshold)."""
    # (m == EZ).any(0) = columns with any EZ pixel; its mean = share of columns with EZ present
    return np.array([np.mean([(m == EZ).any(0).mean() for m in scan]) < 0.95 for scan in scans])


class Scans(Dataset):
    """Training: each scan once per epoch, with one grader's mask drawn at random, plus flip,
    vertical shift and mild intensity jitter. Evaluation: every (scan, grader) pair, unaugmented.
    shift_mode: "border" (repeats the edge row; the served model) or "roll" (wraps; the old bug),
    see gpu_service/unet/unet.py shift_rows."""

    def __init__(self, images, masks, index, train: bool, max_shift: int = 0, flip: bool = False,
                 shift_mode: str = "roll"):
        self.images, self.masks, self.index = images, masks, index            # index = which scans
        self.train, self.max_shift, self.flip, self.shift_mode = train, max_shift, flip, shift_mode

    def __len__(self):
        return len(self.index) * (1 if self.train else 3)  # evaluation sees each grader's mask

    def __getitem__(self, i):
        # s = which scan, g = which grader's mask
        s, g = (self.index[i], int(torch.randint(3, ()))) if self.train else (self.index[i // 3], i % 3)
        x = preprocess(self.images[s])[0]                         # (1, 512, 512) in [0, 1]
        y = torch.from_numpy(self.masks[s, g].astype(np.int64))  # int64: what cross_entropy expects
        if self.train:
            # Every geometric transform is applied to image AND mask, so they stay aligned.
            if self.flip and torch.rand(()) < 0.5:                # half the time: mirror left-right
                x, y = x.flip(-1), y.flip(-1)
            if self.max_shift:
                shift = int(torch.randint(-self.max_shift, self.max_shift + 1, ()))  # rows, -16..16
                x, y = shift_rows(x, shift, self.shift_mode), shift_rows(y, shift, self.shift_mode)
            # Intensity jitter, image only: gain in [0.9, 1.1], offset in [-0.05, 0.05].
            # (analysis/confidence_scores.py reproduces this exact order of random draws.)
            x = (x * (0.9 + 0.2 * torch.rand(())) + 0.1 * (torch.rand(()) - 0.5)).clamp(0, 1)
        return x, y


class OCT5kDataModule(L.LightningDataModule):
    """Lightning wrapper: which scans train, which validate, and how they are batched."""

    def __init__(self, manifest, fold, n_folds=5, seed=42, batch_size=4, num_workers=4, max_shift=16, flip=True,
                 ez_balance=False, shift_mode="roll"):
        super().__init__()
        self.manifest, self.fold, self.n_folds, self.seed = manifest, fold, n_folds, seed
        self.batch_size, self.num_workers, self.max_shift, self.flip = batch_size, num_workers, max_shift, flip
        self.ez_balance, self.shift_mode = ez_balance, shift_mode

    def setup(self, stage=None):
        rows, images, masks = load_scans(self.manifest)          # cached after the first fold
        folds = assign_folds(rows, self.n_folds, self.seed)      # patient -> fold
        # Fold k is the test set (untouched here); fold k+1 is validation for early stopping.
        # So every scan is tested exactly once, by a model that never saw its patient.
        val = (self.fold + 1) % self.n_folds
        which = np.array([folds[r["patient"]] for r in rows])    # each scan's fold
        train_idx = np.flatnonzero((which != self.fold) & (which != val))  # the other 3 folds
        val_idx = np.flatnonzero(which == val)
        self.train_ds = Scans(images, masks, train_idx, True, self.max_shift, self.flip, self.shift_mode)
        self.val_ds = Scans(images, masks, val_idx, False)
        self.sampler = None                                      # default: plain shuffling
        if self.ez_balance:
            # Half of each epoch's draws come from scans with EZ loss, as the team balances rare
            # positives against negatives for HRF. Validation keeps the natural mix.
            loss = shows_ez_loss(masks[i] for i in train_idx)  # no fancy indexing: that copies ~0.8 GB
            # Each group's weights sum to 0.5, so a draw is equally likely to come from either group.
            weights = np.where(loss, 0.5 / max(loss.sum(), 1), 0.5 / max((~loss).sum(), 1))
            self.sampler = WeightedRandomSampler(weights.tolist(), num_samples=len(train_idx), replacement=True)

    def train_dataloader(self):
        # A sampler and shuffle=True cannot be combined; the sampler already randomises the order.
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=self.sampler is None, sampler=self.sampler,
                          num_workers=self.num_workers, persistent_workers=self.num_workers > 0)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size * 2,  # no gradients, so larger batches fit
                          num_workers=self.num_workers, persistent_workers=self.num_workers > 0)
