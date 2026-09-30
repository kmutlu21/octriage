"""Test-time augmentation: every transform is undone exactly, a stable model scores 1.0, an
unstable one scores lower, and checkpoints round-trip with their shift mode."""
import numpy as np       # arrays
import pytest            # the test framework
import torch             # tensors
from torch import nn     # tiny fake models

from gpu_service.unet.unet import UNet, load_checkpoint, preprocess, save_checkpoint, shift_rows
from gpu_service import tta  # the code under test

torch.manual_seed(0)     # the same random inputs every run


@pytest.mark.parametrize("flip, shift", tta.AUGMENTS)
def test_invert_undoes_apply(flip, shift):
    x = torch.rand(1, 1, 16, 12)
    assert torch.equal(tta.invert(tta.apply(x, flip, shift), flip, shift), x)   # apply then undo = original


@pytest.mark.parametrize("k", [1, 8, 16, -1, -8, -16])
def test_border_shift_never_wraps_and_roll_does(k):
    x = torch.arange(32.0).view(1, 1, 32, 1).expand(1, 1, 32, 5)  # every row holds its own index
    border = shift_rows(x, k, "border")[0, 0, :, 0]
    rolled = shift_rows(x, k, "roll")[0, 0, :, 0]
    if k > 0:  # moved down: the top k rows repeat row 0; roll brings the bottom rows round instead
        assert border[:k].eq(0).all() and torch.equal(border[k:], torch.arange(32.0 - k))
        assert torch.equal(rolled[:k], torch.arange(32.0 - k, 32))
    else:  # moved up: the bottom rows repeat row 31; roll brings the top rows round
        assert border[k:].eq(31).all() and torch.equal(border[:k], torch.arange(-k, 32.0))
        assert torch.equal(rolled[k:], torch.arange(0, -k, dtype=torch.float))


@pytest.mark.parametrize("flip, shift", tta.AUGMENTS)
def test_border_invert_restores_every_row_but_the_ones_pushed_off(flip, shift):
    x = torch.rand(1, 1, 40, 12)
    back = tta.invert(tta.apply(x, flip, shift, "border"), flip, shift, "border")
    keep = slice(tta.SHIFT, 40 - tta.SHIFT)            # rows that never left the image
    assert torch.equal(back[..., keep, :], x[..., keep, :])


def test_unknown_shift_mode_is_refused():
    with pytest.raises(ValueError):
        shift_rows(torch.rand(1, 1, 8, 8), 2, "reflect")


def test_dice_counts_absent_in_both_as_agreement():
    a = torch.zeros(4, 4, dtype=torch.long)
    assert tta.dice(a, a, 3) == 1.0                    # label 3 absent from both: agreement
    b = a.clone(); b[0, :2] = 3                        # label 3 at columns 0-1
    c = a.clone(); c[0, 1:3] = 3                       # label 3 at columns 1-2
    assert tta.dice(b, c, 3) == 0.5                    # overlap 1: 2 x 1 / (2 + 2)


class Pointwise(nn.Module):
    """Label depends only on each pixel's own intensity: exactly flip- and shift-equivariant."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 6, 1)
        with torch.no_grad():  # logits k*x - k^2/10 make label k win on an intensity band
            k = torch.arange(6.0)
            self.conv.weight.copy_(k.view(6, 1, 1, 1))
            self.conv.bias.copy_(-(k**2) / 10)

    def forward(self, x):
        return self.conv(x)


class PositionDependent(Pointwise):
    """Adds a fixed top-to-bottom ramp, so shifting the input changes the answer."""
    def forward(self, x):
        ramp = torch.linspace(0, 3, x.shape[-2]).view(1, 1, -1, 1)
        return super().forward(x) + ramp * torch.tensor([0, 1, 0, 0, 0, 0.0]).view(1, 6, 1, 1)


def test_equivariant_model_is_fully_confident():
    x = torch.rand(1, 1, 64, 48)
    mask, conf = tta.predict_tta(Pointwise().eval(), x)
    assert conf == 1.0                                 # all 6 passes agree exactly
    assert torch.equal(mask.long(), Pointwise()(x).argmax(1)[0])


def test_border_tta_is_fully_confident_when_the_edges_are_background():
    # Like an OCT B-scan: uniform vitreous on top, uniform choroid at the bottom, deeper than the
    # shift. The rows a border shift loses are refilled with the same background, so nothing changes.
    x = torch.rand(1, 1, 64, 48)
    x[..., :2 * tta.SHIFT, :], x[..., -2 * tta.SHIFT:, :] = 0.05, 0.95
    model = Pointwise().eval()
    model.shift_mode = "border"
    mask, conf = tta.predict_tta(model, x)
    assert conf == 1.0 and torch.equal(mask.long(), model(x).argmax(1)[0])


def test_unstable_model_loses_confidence():
    _, conf = tta.predict_tta(PositionDependent().eval(), torch.rand(1, 1, 64, 48))
    assert 0.0 < conf < 1.0                            # shifting changes its answer: lower score


def test_preprocess_matches_mask_geometry():
    x = preprocess(np.full((512, 496), 255, np.uint8))  # source B-scans are 496 wide x 512 tall
    assert x.shape == (1, 1, 512, 512) and float(x.max()) == 1.0


def test_unet_shape_and_checkpoint_round_trip(tmp_path):
    model = UNet(width=4, depth=2).eval()
    x = torch.rand(1, 1, 64, 64)
    assert model(x).shape == (1, 6, 64, 64)
    save_checkpoint(tmp_path / "m.pt", model, "tiny")
    loaded, name = load_checkpoint(tmp_path / "m.pt", "cpu")
    assert name == "tiny" and torch.equal(loaded(x), model(x)) and loaded.shift_mode == "roll"
    save_checkpoint(tmp_path / "b.pt", model, "tiny-border", "border")
    assert load_checkpoint(tmp_path / "b.pt", "cpu")[0].shift_mode == "border"


def test_checkpoints_from_before_the_shift_field_load_as_roll(tmp_path):
    # Our first two models' checkpoints were saved without the field and were trained with roll.
    model = UNet(width=4, depth=2)
    torch.save({"name": "old", "hparams": model.hparams, "state_dict": model.state_dict()}, tmp_path / "old.pt")
    assert load_checkpoint(tmp_path / "old.pt", "cpu")[0].shift_mode == "roll"
