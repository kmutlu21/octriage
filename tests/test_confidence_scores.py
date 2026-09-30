"""The published confidence scores (Wang 2019, Zenk 2024): test-time draws reproduce the
training augmentation exactly, and each score reacts the right way to certainty."""
import math              # log(6)

import numpy as np       # arrays
import pytest            # the test framework
import torch             # tensors
from torch import nn     # tiny fake models

from analysis import confidence_scores as cs  # the code under test
from gpu_service.unet.data import Scans                # the training augmentation it must reproduce
from gpu_service.unet.unet import preprocess


@pytest.mark.parametrize("shift_mode", ["roll", "border"])
@pytest.mark.parametrize("seed", [0, 1, 7, 123])
def test_test_time_draws_reproduce_the_training_augmentation_exactly(seed, shift_mode):
    # Wang 2019's rule: test-time transforms follow the training prior. Same RNG stream in,
    # the same augmented image out, bit for bit.
    rng = np.random.default_rng(seed)
    images = rng.integers(0, 256, (1, 40, 31), dtype=np.uint8)
    ds = Scans(images, np.zeros((1, 3, 512, 512), np.uint8), [0], train=True, max_shift=16, flip=True,
               shift_mode=shift_mode)
    torch.manual_seed(seed)
    x_train, _ = ds[0]                             # one training draw
    torch.manual_seed(seed)                        # rewind the random stream
    torch.randint(3, ())  # Scans draws the grader first
    x_test = cs.augment(preprocess(images[0])[0], *cs.draw_augment(16), mode=shift_mode)
    assert torch.equal(x_train, x_test)            # identical, bit for bit


def test_draws_cover_the_prior():
    g = torch.Generator().manual_seed(0)
    d = np.array([cs.draw_augment(16, g) for _ in range(4000)], dtype=float)   # columns: flip, shift, gain, offset
    assert set(d[:, 1]) == set(range(-16, 17))     # every shift from -16 to 16 occurs
    assert 0.45 < d[:, 0].mean() < 0.55
    assert d[:, 2].min() >= 0.9 and d[:, 2].max() <= 1.1
    assert d[:, 3].min() >= -0.05 and d[:, 3].max() <= 0.05


def test_draws_do_not_depend_on_scan_order():
    g = cs.scan_generator(0, "AMD01_b010")
    a = [cs.draw_augment(16, g) for _ in range(3)]
    cs.draw_augment(16, cs.scan_generator(0, "DME11_b001"))
    g = cs.scan_generator(0, "AMD01_b010")
    b = [cs.draw_augment(16, g) for _ in range(3)]
    assert a == b and len(set(a)) == 3
    assert cs.draw_augment(16, cs.scan_generator(1, "AMD01_b010")) != a[0]


def test_vvc_is_std_over_mean_of_band_area():
    masks = torch.zeros(2, 4, 4, dtype=torch.long)
    masks[0, 0, :2] = 1  # band 1: area 2
    masks[1, 0, :]  = 1  # band 1: area 4
    masks[:, 3, :] = 2   # band 2: area 4 in both
    v = cs.vvc(masks)
    assert v[0] == pytest.approx(1 / 3)  # mean 3, population std 1
    assert v[1] == 0.0                    # identical areas
    assert v[2] == v[3] == 0.0            # absent in every draw: all agree


def test_mean_predictive_entropy():
    assert cs.mean_predictive_entropy(torch.full((6, 3, 3), 1 / 6)) == pytest.approx(math.log(6))  # torn: maximum
    one_hot = torch.zeros(6, 3, 3); one_hot[2] = 1
    assert cs.mean_predictive_entropy(one_hot) == 0.0   # certain: zero


class Constant(nn.Module):
    """Predicts band 2 everywhere, whatever the input: perfectly stable."""
    def forward(self, x):
        logits = torch.zeros(x.shape[0], 6, *x.shape[-2:])
        logits[:, 2] = 5.0
        return logits


class PositionDependent(nn.Module):
    """Label depends on the pixel's intensity and on its row, so flips, shifts and jitter move it."""
    def forward(self, x):
        k = torch.arange(6.0).view(1, 6, 1, 1)
        ramp = torch.linspace(0, 3, x.shape[-2]).view(1, 1, -1, 1)
        return k * x - k**2 / 10 + ramp * (k == 1)


def test_stable_model_scores_full_confidence_and_unstable_model_less():
    x = torch.rand(1, 1, 64, 48)
    stable = cs.score(Constant().eval(), x, "s", max_shift=16)
    assert stable["s_wang_A"] == stable["s_wang_B"] == 0.0
    unstable = cs.score(PositionDependent().eval(), x, "s", max_shift=16)
    assert unstable["s_wang_A"] < 0 and unstable["s_wang_B"] < 0
    assert unstable["s_pe"] < stable["s_pe"] <= 0


def test_chunking_does_not_change_the_score():
    x = torch.rand(1, 1, 64, 48)
    draws = [cs.draw_augment(16, cs.scan_generator(0, "s")) for _ in range(cs.N_DRAWS)]
    model = PositionDependent().eval()
    assert cs.wang_vvc(model, x, draws, chunk=3) == cs.wang_vvc(model, x, draws, chunk=20)
