"""A plain U-Net (Ronneberger et al., MICCAI 2015) for 6-class retinal-layer segmentation, plus the
preprocessing and vertical shift that training and the service must share.

In plain words: A U-Net is a neural network that labels every pixel of an image. It first shrinks
the image step by step to see the big picture (which layer is where), then grows it back to full
size to draw sharp edges, copying fine detail across from the shrinking side ("skip connections").
Here each pixel of a B-scan gets one of six labels: vitreous, four retinal bands, or tissue below
the RPE.

The network is a STAND-IN. The gate is model-agnostic: any model that answers the service's
contract (gpu_service/app.py) can replace it, including WRC's own nnSAM. A U-Net was kept because
WRC's 12-model comparison found U-Net/FPN the best family for GA (Safai et al., IOVS 2024).
Width 32, depth 4 (7.8M parameters), BatchNorm and no pretrained encoder are our choices, sized
for a 6 GB GPU.
"""
import numpy as np                   # arrays
import torch                         # tensors and the network
import torch.nn.functional as F      # resize, pooling
from torch import nn                 # network layers

SIZE = 512  # OCT5k masks are 512x512; source B-scans (496 wide) are resized to match


def preprocess(image: np.ndarray) -> torch.Tensor:
    """uint8 grayscale B-scan (H, W) -> float tensor (1, 1, 512, 512) in [0, 1].
    Used by training, prediction and the service alike, so they can never disagree on it."""
    x = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32) / 255.0)[None, None]  # add batch + channel dims
    return F.interpolate(x, size=(SIZE, SIZE), mode="bilinear", align_corners=False)      # resize to 512 x 512


SHIFT_MODES = ("roll", "border")  # the two ways to shift an image up/down (see below)


def shift_rows(x: torch.Tensor, k: int, mode: str) -> torch.Tensor:
    """Move the image (or mask, or class probabilities) k rows down (k < 0: up).

    "roll" wraps the rows pushed off one edge round to the other, so choroid from the bottom
    lands above the retina. Our first two models were trained that way and learned to draw
    choroid in the top rows (in the second, 1,501 of 1,672 out-of-fold masks had choroid in the
    top row or vitreous in the bottom row; the served model has 0). Kept only so the bug stays
    reproducible in a test (tests/test_training.py) and old checkpoints still load.
    "border" repeats the edge row instead, as the team's OCT augmentation pads
    (WRC oct-evaluation code: MONAI RandAffined padding_mode="border"), so vitreous stays on top
    and choroid below. The served model uses "border". A zero fill would be wrong too: nnU-Net
    relabels its padding as class 0, which here is vitreous, so vitreous would appear under the
    choroid.
    """
    if not k:                                      # no shift: return as is
        return x
    if mode == "roll":
        return x.roll(k, dims=-2)  # dims=-2 = the row axis, whatever the leading dimensions are
    if mode != "border":
        raise ValueError(f"shift mode {mode!r} is not one of {SHIFT_MODES}")
    # Output row r takes input row r - k, clamped into the image: rows shifted in from outside
    # repeat the nearest edge row.
    rows = (torch.arange(x.shape[-2], device=x.device) - k).clamp(0, x.shape[-2] - 1)
    return x.index_select(-2, rows)                # pick those rows, in that order


def block(c_in: int, c_out: int) -> nn.Sequential:
    """Two 3x3 convolutions, each followed by BatchNorm and ReLU: the U-Net's basic unit.
    bias=False because BatchNorm adds its own shift right after."""
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Encoder halves the resolution `depth` times while doubling the channels; the decoder
    upsamples back and concatenates the matching encoder output (the "skip"), which keeps the
    fine boundary detail that thin layers like EZ and RPE need."""

    def __init__(self, in_channels: int = 1, n_classes: int = 6, width: int = 32, depth: int = 4):
        super().__init__()
        # saved in the checkpoint, so the service can rebuild the same network
        self.hparams = {"in_channels": in_channels, "n_classes": n_classes, "width": width, "depth": depth}
        ch = [width * 2**i for i in range(depth + 1)]  # 32, 64, 128, 256, 512 channels
        # encoder: one block at full resolution, then one per halving
        self.down = nn.ModuleList([block(in_channels, ch[0])] + [block(ch[i], ch[i + 1]) for i in range(depth)])
        # decoder: a 2x2 transposed convolution doubles the resolution and halves the channels
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([block(2 * ch[i], ch[i]) for i in reversed(range(depth))])  # 2x: skip + upsampled
        self.head = nn.Conv2d(ch[0], n_classes, 1)  # 1x1 conv: one score (logit) per class per pixel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []                                      # encoder outputs, kept for the decoder
        for i, down in enumerate(self.down):
            x = down(x if i == 0 else F.max_pool2d(x, 2))  # halve resolution before every block but the first
            skips.append(x)
        x = skips.pop()  # the bottleneck (lowest resolution) starts the decoder
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], dim=1))  # upsample, glue on the matching skip, convolve
        return self.head(x)  # (B, n_classes, H, W) logits; softmax/argmax happen outside


# The checkpoint format is the contract between training and the service. It records the
# vertical shift the model was trained with, so test-time augmentation shifts the same way;
# checkpoints from before the field existed were trained with "roll".
def save_checkpoint(path, model: UNet, name: str, shift_mode: str = "roll") -> None:
    torch.save({"name": name, "hparams": model.hparams, "state_dict": model.state_dict(),
                "shift_mode": shift_mode}, path)


def load_checkpoint(path, device: str) -> tuple[UNet, str]:
    """The model comes back in eval mode with a `shift_mode` attribute, which gpu_service/tta.py reads."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)  # weights_only: never unpickle code
    model = UNet(**ckpt["hparams"])                                 # same architecture as trained
    model.load_state_dict(ckpt["state_dict"])                       # the trained weights
    model.shift_mode = ckpt.get("shift_mode", "roll")               # old checkpoints: "roll"
    return model.to(device).eval(), ckpt["name"]  # eval(): BatchNorm uses its stored statistics
