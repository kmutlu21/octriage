"""Why this confidence score? Step 1: compute two PUBLISHED confidence scores that need no extra
training, for every out-of-fold scan, so analysis/confidence_eval.py can compare them with ours.
The rule for that comparison was written and frozen first: analysis/prereg/a0_confidence.md.

In plain words: Our confidence score is our own design, so it was compared with two published ones.
This file computes those two for every scan; analysis/confidence_eval.py compares all three under a
rule written before any score existed.

- Wang et al., Neurocomputing 2019;335:34-45: test-time augmentation drawn from the same prior as
  the training augmentation, N = 20 draws (their 2D setting), and the volume variation coefficient
  VVC = std/mean of each structure's size across the draws. Here the prior is gpu_service/unet/data.py's
  training augmentation and a structure is a retinal band; the band average is ours (Zenk's
  mean-over-classes convention).
- Zenk et al., Medical Image Analysis 2024: the single-network baseline, pixel-wise predictive
  entropy of the un-augmented softmax, averaged over all pixels.

Scores are signed so that higher = more confident, like the served TTA confidence. Each scan is
scored by the fold model that never saw its patient, exactly as in gpu_service/unet/predict.py.
Takes ~30 min on an RTX 2060 (20 extra forward passes per scan).

    python -m analysis.confidence_scores --config gpu_service/unet/config.yaml
"""
import argparse                          # command-line options
import csv                               # the scores table
import json                              # the run record
import time                              # timing
import zlib                              # crc32: a stable per-scan seed
from pathlib import Path                 # file paths

import torch                             # the model

from gpu_service.unet.data import assign_folds, load_scans     # same data and folds as training
from gpu_service.unet.train import load_config                 # same YAML
from gpu_service.unet.unet import load_checkpoint, preprocess  # same checkpoints and preprocessing
from gpu_service.tta import BANDS, apply, invert        # the service's shift/flip and its undo

N_DRAWS = 20  # Wang 2019, 2D experiments
DRAW_SETS = {"A": 0, "B": 1}  # A decides; B only checks that the score does not depend on the draw


def draw_augment(max_shift: int, g: torch.Generator | None = None) -> tuple[bool, int, float, float]:
    """One draw from the training augmentation prior, in gpu_service/unet/data.py's order of random calls:
    flip ~ Bernoulli(0.5), vertical shift ~ U{-max_shift..max_shift}, gain ~ U(0.9, 1.1),
    offset ~ U(-0.05, 0.05). tests/test_confidence_scores.py proves the draws match training's."""
    flip = bool(torch.rand((), generator=g) < 0.5)                     # mirror or not
    shift = int(torch.randint(-max_shift, max_shift + 1, (), generator=g))  # rows up/down
    gain = float(0.9 + 0.2 * torch.rand((), generator=g))             # brightness x 0.9..1.1
    offset = float(0.1 * (torch.rand((), generator=g) - 0.5))         # brightness + -0.05..0.05
    return flip, shift, gain, offset


def augment(x: torch.Tensor, flip: bool, shift: int, gain: float, offset: float,
            mode: str = "roll") -> torch.Tensor:
    """gpu_service/unet/data.py's training transform: flip, vertical shift (roll or border, as trained), then
    intensity jitter clamped to [0, 1]."""
    return (apply(x, flip, shift, mode) * gain + offset).clamp(0, 1)


def vvc(masks: torch.Tensor, bands=BANDS) -> list[float]:
    """Per band: std/mean of its pixel area across the masks (N, H, W). A band absent from every
    mask has no variation (0/0 is taken as 0: every draw agrees)."""
    out = []
    for b in bands:
        area = (masks == b).sum((-2, -1)).double()  # pixel count of band b in each of the N masks
        mean = float(area.mean())
        out.append(0.0 if mean == 0 else float(area.std(unbiased=False)) / mean)  # coefficient of variation
    return out


def mean_predictive_entropy(probs: torch.Tensor) -> float:
    """probs (C, H, W) softmax -> mean over pixels of -sum_c p_c log p_c (0 log 0 = 0).
    Entropy is 0 where the model is certain of one class and highest where it is torn."""
    return float(-torch.special.xlogy(probs, probs).sum(0).mean())  # xlogy(0, 0) = 0, no NaN


def scan_generator(seed: int, scan_id: str) -> torch.Generator:
    """Per-scan draws that do not depend on the order in which scans are processed."""
    return torch.Generator().manual_seed(zlib.crc32(f"{seed}:{scan_id}".encode()))  # seed from the scan's name


@torch.no_grad()
def wang_vvc(model, x: torch.Tensor, draws, chunk: int = 10, mode: str = "roll") -> list[float]:
    """x (1, 1, H, W) on the model's device; each draw is augmented, predicted, and its mask
    mapped back to the original geometry before the band areas are measured."""
    masks = []
    for i in range(0, len(draws), chunk):  # 10 at a time, to fit in GPU memory
        part = draws[i:i + chunk]
        pred = model(torch.cat([augment(x, *d, mode=mode) for d in part])).argmax(1)  # 10 masks at once
        masks += [invert(p, d[0], d[1], mode) for p, d in zip(pred, part)]  # undo shift + flip
    return vvc(torch.stack(masks))


@torch.no_grad()
def score(model, x: torch.Tensor, scan_id: str, max_shift: int, shift_mode: str = "roll") -> dict:
    row = {"mean_pe": mean_predictive_entropy(model(x).softmax(1)[0])}  # Zenk's baseline, one pass
    row["s_pe"] = -row["mean_pe"]  # negate: low entropy = high confidence
    for name, seed in DRAW_SETS.items():                 # draw set A, then B
        g = scan_generator(seed, scan_id)
        v = wang_vvc(model, x, [draw_augment(max_shift, g) for _ in range(N_DRAWS)], mode=shift_mode)
        row.update({f"vvc_{name}_b{b}": vi for b, vi in zip(BANDS, v)})  # per band, for inspection
        row[f"s_wang_{name}"] = -sum(v) / len(v)  # negate: low variation = high confidence
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="gpu_service/unet/config.yaml")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N scans of each fold (smoke test)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    exp = Path("outputs") / cfg["experiment_name"]
    out = exp / "analysis" / "a0_confidence"
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = False  # deterministic kernels, so a rerun gives the same scores

    rows, images, _ = load_scans(cfg["data"]["manifest"])
    folds = assign_folds(rows, cfg["data"]["n_folds"], cfg["seed"])
    shift_mode = cfg["data"].get("shift_mode", "roll")  # Wang: the test-time prior is the training prior
    results, t0 = [], time.time()
    for fold in range(cfg["data"]["n_folds"]):
        model, name = load_checkpoint(exp / f"fold{fold}.pt", device)
        if model.shift_mode != shift_mode:  # the checkpoint and the config must describe the same training
            raise SystemExit(f"{name} was trained with shift {model.shift_mode!r}, the config says {shift_mode!r}")
        todo = [i for i, r in enumerate(rows) if folds[r["patient"]] == fold]  # this fold's scans
        for i in todo[:args.limit or None]:              # all of them unless --limit
            r = rows[i]
            results.append({"scan_id": r["scan_id"], "fold": fold, "model": name,
                            **score(model, preprocess(images[i]).to(device), r["scan_id"], cfg["data"]["max_shift"],
                                    shift_mode)})
        print(f"fold {fold}: {sum(x['fold'] == fold for x in results)} scans, {time.time() - t0:.0f} s", flush=True)

    results.sort(key=lambda x: x["scan_id"])             # stable order for diffs
    with open(out / "scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(results[0]))
        w.writeheader()
        w.writerows(results)
    (out / "scores_run.json").write_text(json.dumps({  # provenance: how, where and how long
        "config": args.config, "n_scans": len(results), "n_draws": N_DRAWS, "draw_sets": DRAW_SETS,
        "max_shift": cfg["data"]["max_shift"], "shift_mode": shift_mode, "device": device,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "torch": torch.__version__, "seconds": round(time.time() - t0, 1)}, indent=2))
    print(f"wrote {len(results)} rows -> {out / 'scores.csv'}")


if __name__ == "__main__":
    main()
