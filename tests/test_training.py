"""Training pieces: patient-level, cohort-balanced folds; augmentation moves image and mask
together; the border shift never wraps (and roll does: the old bug, reproduced); losses,
EZ-balanced sampling, optimiser and the data-wait timer."""
from collections import Counter  # counting patients per fold

import numpy as np       # arrays
import pytest            # the test framework
import torch             # tensors
import torch.nn.functional as F  # one_hot, losses

from gpu_service.unet.data import Scans, assign_folds                   # data code under test
from gpu_service.unet.train import LayerSegmenter, StepTimer, soft_dice  # training code under test

# The real OCT5k mix: 21 AMD, 19 DME, 20 Normal patients, one visit each.
ROWS = [{"cohort": c, "patient": f"{c}{i:02d}"} for c, n in (("AMD", 21), ("DME", 19), ("Normal", 20))
        for i in range(1, n + 1)]


def test_folds_are_patient_level_balanced_and_stratified():
    folds = assign_folds(ROWS, 5, seed=42)
    assert len(folds) == 60 and set(folds.values()) == set(range(5))   # every patient in one of 5 folds
    assert Counter(folds.values()) == {k: 12 for k in range(5)}        # 12 patients each
    for cohort in ("AMD", "DME", "Normal"):
        per_fold = Counter(f for p, f in folds.items() if p.startswith(cohort))
        assert max(per_fold.values()) - min(per_fold.values()) <= 1    # cohorts spread evenly


def test_folds_are_reproducible_from_the_seed():
    assert assign_folds(ROWS, 5, 42) == assign_folds(ROWS, 5, 42) != assign_folds(ROWS, 5, 7)


def scan_arrays(n=2):
    """Same region for every grader but a different label value, so the grader drawn is visible.
    The region is off-centre, so a flip or shift that hit only the image would break alignment."""
    masks = np.zeros((n, 3, 512, 512), np.uint8)
    for g in range(3):
        masks[:, g, 100:300, 50:400] = g + 1
    return (masks[:, 0] * 40).astype(np.uint8), masks


@pytest.mark.parametrize("shift_mode", ["roll", "border"])
def test_training_augmentation_moves_image_and_mask_together(shift_mode):
    images, masks = scan_arrays()
    ds = Scans(images, masks, np.array([0, 1]), train=True, max_shift=16, flip=True, shift_mode=shift_mode)
    torch.manual_seed(0)
    graders, flipped = set(), set()
    for _ in range(40):
        x, y = ds[0]
        # Jitter keeps background <= 0.05 and foreground >= 0.09, so 0.07 separates them.
        assert torch.equal(x[0] > 0.07, y > 0)
        graders.add(int(y.max()))
        flipped.add(bool(y[200, 450]))  # only a flipped mask has foreground at column 450
    assert graders == {1, 2, 3} and flipped == {True, False}   # all graders drawn; flips both ways


def test_border_shift_keeps_vitreous_on_top_and_choroid_below():
    # Our earlier "roll" shift put choroid (5) in the top rows of 1,232 of 1,672 out-of-fold masks;
    # the "border" shift the served model uses must never produce a training mask like that.
    masks = np.zeros((1, 3, 512, 512), np.uint8)
    masks[..., 200:260, :], masks[..., 260:280, :], masks[..., 280:, :] = 1, 4, 5
    images = masks[:, 0] * 40
    seen = {}
    for mode in ("roll", "border"):
        ds = Scans(images, masks, np.array([0]), train=True, max_shift=16, flip=True, shift_mode=mode)
        torch.manual_seed(0)
        ys = [ds[0][1] for _ in range(60)]
        seen[mode] = {int(y[0].max()) for y in ys}, {int(y[-1].min()) for y in ys}
    assert seen["border"] == ({0}, {5})    # top row always vitreous, bottom row always choroid
    assert 5 in seen["roll"][0]  # the wrap-around, reproduced


def test_evaluation_yields_every_grader_unaugmented():
    images, masks = scan_arrays()
    ds = Scans(images, masks, np.array([1]), train=False)
    assert len(ds) == 3                                # one item per grader
    for g in range(3):
        x, y = ds[g]
        assert torch.equal(y, torch.from_numpy(masks[1, g].astype(np.int64)))


def test_loss_rewards_the_right_answer():
    model = LayerSegmenter(width=4, depth=2)
    y = torch.randint(0, 6, (2, 32, 32))
    perfect = F.one_hot(y, 6).permute(0, 3, 1, 2).float() * 20   # logits that pick the right label
    assert soft_dice(perfect, y) > 0.99
    assert model.loss(perfect, y) < 0.05 < model.loss(torch.zeros_like(perfect), y)


def test_tiny_model_trains_one_step():
    model = LayerSegmenter(width=4, depth=2)
    x, y = torch.rand(2, 1, 64, 64), torch.randint(0, 6, (2, 64, 64))
    loss = model.loss(model(x), y)
    loss.backward()                                    # gradients flow
    assert torch.isfinite(loss) and all(p.grad is not None for p in model.parameters())   # to every weight


def test_step_timer_logs_wait_and_compute_each_epoch():
    import lightning as L
    from torch.utils.data import DataLoader, TensorDataset

    data = TensorDataset(torch.rand(4, 1, 32, 32), torch.randint(0, 6, (4, 32, 32)))
    trainer = L.Trainer(max_epochs=1, accelerator="cpu", logger=False, enable_checkpointing=False,
                        enable_progress_bar=False, enable_model_summary=False, callbacks=[StepTimer()])
    loader = DataLoader(data, batch_size=2)
    trainer.fit(LayerSegmenter(width=4, depth=2), loader, loader)
    wait, compute = (float(trainer.callback_metrics[k]) for k in ("time_data_wait_s", "time_compute_s"))
    assert wait >= 0 and compute > 0


def band_mask(gap=None):
    """A 64x64 mask with every band present; `gap` (a column slice) removes EZ there, as EZ loss
    does: the band above (2) meets RPE (4) directly."""
    y = torch.zeros(64, 64, dtype=torch.long)
    y[10:30], y[30:40], y[40:44], y[44:48], y[48:] = 1, 2, 3, 4, 5
    if gap is not None:
        y[40:44, gap] = 2
    return y


def test_ez_column_loss_penalises_painting_over_ez_loss():
    from gpu_service.unet.train import ez_column_loss

    truth = band_mask(gap=slice(20, 40))[None]
    painted = F.one_hot(band_mask()[None], 6).permute(0, 3, 1, 2).float() * 20  # EZ drawn everywhere
    faithful = F.one_hot(truth, 6).permute(0, 3, 1, 2).float() * 20
    assert ez_column_loss(faithful, truth) < 0.01 < ez_column_loss(painted, truth)   # painting over the gap costs


def test_ez_weight_zero_keeps_the_original_loss():
    y = band_mask(gap=slice(20, 40))[None]
    logits = torch.randn(1, 6, 64, 64)
    assert torch.equal(LayerSegmenter(width=4, depth=2).loss(logits, y),
                       LayerSegmenter(width=4, depth=2, ez_weight=0.0).loss(logits, y))
    assert LayerSegmenter(width=4, depth=2, ez_weight=1.0).loss(logits, y) > LayerSegmenter(width=4, depth=2).loss(logits, y)


def test_scans_with_ez_loss_are_found_by_grader_average():
    from gpu_service.unet.data import shows_ez_loss

    whole, lost = band_mask().numpy(), band_mask(gap=slice(0, 20)).numpy()  # 20/64 columns lost
    scans = [np.stack([whole] * 3), np.stack([lost, lost, whole]), np.stack([lost, whole, whole])]
    # mean presence: 1.0; (0.69+0.69+1)/3 = 0.79; (0.69+1+1)/3 = 0.90 -> all but the first show loss
    assert shows_ez_loss(scans).tolist() == [False, True, True]


def test_ez_balance_gives_half_the_draws_to_scans_with_ez_loss(monkeypatch):
    import gpu_service.unet.data as data

    rows = [{"cohort": "DME", "patient": f"P{i:02d}"} for i in range(60)]
    masks = np.stack([np.stack([band_mask(gap=slice(0, 20) if i % 10 == 0 else None).numpy()] * 3)
                      for i in range(60)])
    monkeypatch.setattr(data, "load_scans", lambda manifest: (rows, np.zeros((60, 64, 64), np.uint8), masks))
    dm = data.OCT5kDataModule("unused.csv", fold=0, ez_balance=True, num_workers=0)
    dm.setup()
    loss = data.shows_ez_loss(masks[i] for i in dm.train_ds.index)
    w = dm.sampler.weights.numpy()
    assert 0 < loss.sum() < len(loss) and np.isclose(w[loss].sum(), 0.5) and np.isclose(w[~loss].sum(), 0.5)


def test_optimizer_is_adam_by_default_and_plateau_scheduled_when_asked():
    plain = LayerSegmenter(width=4, depth=2).configure_optimizers()
    assert isinstance(plain, torch.optim.AdamW) and plain.defaults["weight_decay"] == 0.0
    tuned = LayerSegmenter(width=4, depth=2, weight_decay=0.01, lr_patience=5).configure_optimizers()
    sched = tuned["lr_scheduler"]["scheduler"]
    assert tuned["optimizer"].defaults["weight_decay"] == 0.01
    assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau) and sched.patience == 5
    assert tuned["lr_scheduler"]["monitor"] == "val_loss"


@pytest.mark.parametrize("lr_patience", [0, 1])
def test_training_callbacks_work_with_and_without_a_scheduler(tmp_path, lr_patience):
    import lightning as L
    from lightning.pytorch.callbacks import LearningRateMonitor
    from lightning.pytorch.loggers import CSVLogger
    from torch.utils.data import DataLoader, TensorDataset

    data = TensorDataset(torch.rand(4, 1, 32, 32), torch.randint(0, 6, (4, 32, 32)))
    loader = DataLoader(data, batch_size=2)
    trainer = L.Trainer(max_epochs=2, accelerator="cpu", logger=CSVLogger(tmp_path), enable_checkpointing=False,
                        enable_progress_bar=False, enable_model_summary=False,
                        callbacks=[StepTimer(), LearningRateMonitor(logging_interval="epoch")])
    trainer.fit(LayerSegmenter(width=4, depth=2, ez_weight=1.0, lr_patience=lr_patience), loader, loader)
    logged = (tmp_path / "lightning_logs" / "version_0" / "metrics.csv").read_text().splitlines()[0]
    assert all(k in logged for k in ("lr-AdamW", "time_data_wait_s", "val_ez_presence_mae"))


def test_ez_column_loss_matches_torch_binary_cross_entropy():
    # Written out by hand because F.binary_cross_entropy is banned under CUDA autocast (the
    # 16-mixed training run failed on it); this pins the hand-written version to torch's.
    from gpu_service.unet.train import ez_column_loss

    torch.manual_seed(0)
    logits, y = torch.randn(2, 6, 64, 64), torch.stack([band_mask(gap=slice(10, 30)), band_mask()])
    p = logits.softmax(1)[:, 3].amax(1).clamp(1e-6, 1 - 1e-6)
    expected = F.binary_cross_entropy(p.flatten(), (y == 3).any(1).float().flatten())
    assert torch.isclose(ez_column_loss(logits, y), expected, atol=1e-6)
