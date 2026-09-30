"""Test-time augmentation (TTA) confidence: how stable is the model's mask for this scan?

In plain words: Show the model six slightly shifted or mirrored copies of the same B-scan. If it
draws the same layers every time, it is confident; where its drawings disagree, it is unsure, and
that scan should be read first.

Run the model on 6 copies of the scan (identity, horizontal flip, shifted 8 px up and down, and
flip + each shift), undo each transform on the output, and measure how much the 6 predicted
masks agree with their consensus.

    confidence = mean Dice, over the retinal bands (labels 1-4) and over the 6 passes,
                 between each pass's mask and the consensus mask

1.0 means every pass drew every band identically; it drops where the model's boundaries wobble.
Dice is also how grader agreement is measured, so confidence and inter-grader agreement land on
the same scale (analysis/gate_analysis.py relates the two).

Sources and what is ours:
- TTA as an uncertainty signal: Wang et al., Neurocomputing 2019;335:34-45 (they used random
  transforms and a volume variation score).
- The 6 fixed transforms and the Dice-vs-consensus score are OUR design; no single paper uses
  exactly this. analysis/confidence_eval.py compares it, under a rule fixed in advance, with two
  published scores that need no extra training (Wang 2019; Zenk et al., MedIA 2024).

The shift is done the way the model was trained (the checkpoint's `shift_mode`, gpu_service/unet/unet.py):
"border" for the served model. A border shift loses the rows pushed off the edge, so undoing it
repeats the edge row of the prediction there (vitreous at the top, choroid at the bottom).
"""
import torch  # tensors and the model

from gpu_service.unet.unet import shift_rows  # the same vertical shift the training used

SHIFT = 8  # px, vertical: the layers run horizontally, so a vertical shift moves every boundary
# (flip?, vertical shift) for each of the 6 passes; the first one is the unmodified scan.
AUGMENTS = [(False, 0), (True, 0), (False, SHIFT), (False, -SHIFT), (True, SHIFT), (True, -SHIFT)]
BANDS = (1, 2, 3, 4)  # ILM-OPL, OPL-IS/OS, EZ band, RPE; background above and below is excluded


def apply(x: torch.Tensor, flip: bool, shift: int, mode: str = "roll") -> torch.Tensor:
    """Transform the input image: flip left-right, then shift rows."""
    x = x.flip(-1) if flip else x        # -1 = the last axis = columns: a mirror image
    return shift_rows(x, shift, mode)    # move every row up or down


def invert(y: torch.Tensor, flip: bool, shift: int, mode: str = "roll") -> torch.Tensor:
    """Map a prediction back onto the original scan: undo the shift, then the flip (reverse order)."""
    y = shift_rows(y, -shift, mode)      # shift back by the same amount
    return y.flip(-1) if flip else y     # mirror back


def dice(a: torch.Tensor, b: torch.Tensor, label: int) -> float:
    """Dice = 2|A and B| / (|A| + |B|) for one label: 1 = identical, 0 = no overlap."""
    a, b = a == label, b == label        # the pixels of this label in each mask
    total = int(a.sum() + b.sum())       # |A| + |B|
    return 1.0 if total == 0 else 2 * int((a & b).sum()) / total  # absent in both = agreement


@torch.no_grad()  # inference only: no gradients, less memory
def predict_tta(model: torch.nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, float]:
    """x: (1, 1, H, W) on the model's device. Returns (uint8 mask (H, W), confidence in [0, 1])."""
    mode = getattr(model, "shift_mode", "roll")  # set by load_checkpoint from the checkpoint
    # One forward pass over a batch of 6 transformed copies; softmax turns logits into class probabilities.
    probs = model(torch.cat([apply(x, f, s, mode) for f, s in AUGMENTS])).softmax(1)  # (6, C, H, W)
    probs = torch.stack([invert(p, f, s, mode) for p, (f, s) in zip(probs, AUGMENTS)])  # all back on the original
    consensus = probs.mean(0).argmax(0)  # average the 6 probability maps, then take the top class: the served mask
    passes = probs.argmax(1)             # each pass's own mask, (6, H, W)
    scores = [dice(p, consensus, b) for p in passes for b in BANDS]  # 6 passes x 4 bands = 24 Dice values
    return consensus.to(torch.uint8), sum(scores) / len(scores)      # the mask, and the mean of the 24
