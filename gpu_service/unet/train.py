"""Train the U-Net with 5-fold cross-validation by patient. Structure follows
lighting_templates (MIT; see LICENSE): one YAML config drives data, model, trainer and callbacks, and the
config is copied next to the checkpoints it produced.

In plain words: Training shows the model the B-scans with the graders' masks and nudges its weights
to reduce its mistakes (the "loss"). It stops when the score on held-out patients stops improving
and keeps the best checkpoint. This runs once per fold, so every patient has a model that never saw
them.

    python -m gpu_service.unet.train --config gpu_service/unet/config.yaml             # all folds
    python -m gpu_service.unet.train --config gpu_service/unet/config.yaml --folds 0   # one fold
    python -m gpu_service.unet.train --config gpu_service/unet/config.yaml --smoke     # a few batches, prints speed and memory

Writes, per fold k: outputs/<experiment>/fold<k>.pt (the best checkpoint, in the format the
service loads) and outputs/<experiment>/fold<k>/metrics.csv (the training curves).
On a Slurm cluster, gpu_service/slurm/train.sbatch runs one fold per array task.
"""
import argparse                          # command-line options
import shutil                            # copying the config next to the checkpoints
import time                              # timing
from pathlib import Path                 # file paths

import lightning as L                    # PyTorch Lightning: the training loop
import torch                             # tensors, optimiser
import torch.nn.functional as F          # losses
import yaml                              # the config file
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger  # training curves to metrics.csv

from gpu_service.unet.data import EZ, OCT5kDataModule  # the data and the EZ label
from gpu_service.unet.unet import UNet, save_checkpoint  # the network and the service's checkpoint format

BANDS = {1: "inner", 2: "onl", 3: "ez", 4: "rpe"}  # mask labels 1-4; see data_prep/build_manifest.py


def soft_dice(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mean soft Dice over bands 1-4, so the thin EZ and RPE bands count as much as the big ones.
    "Soft": uses probabilities instead of hard labels, so it has a gradient. The +1 terms keep
    it defined when a band is absent from the batch."""
    probs = logits.float().softmax(1)[:, 1:5]  # .float(): compute in 32-bit even under mixed precision
    target = F.one_hot(y, logits.shape[1]).permute(0, 3, 1, 2)[:, 1:5].float()  # (B, H, W) -> (B, C, H, W)
    inter = (probs * target).sum((0, 2, 3))  # per band, summed over the batch and all pixels
    # Dice = 2 x overlap / (predicted + true), per band, then averaged over the 4 bands
    return ((2 * inter + 1) / (probs.sum((0, 2, 3)) + target.sum((0, 2, 3)) + 1)).mean()


def ez_column_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per image column, is there any EZ? The model's answer is its highest EZ probability in
    the column; the grader's is whether they drew EZ there. Scored only where the grader drew
    retina. Pixel losses barely notice a band painted across a few columns of EZ loss, yet those
    columns are the endpoint (EZ integrity is the share of intact EZ).

    OURS. The idea of targeting EZ loss explicitly comes from the team's ez-rpe-reportgear, which
    reports photoreceptor loss area. This loss came in with our second model's recipe, which a
    comparison pre-registered against the first model accepted: the invented EZ gaps in AMD and
    Normal scans disappeared, but real EZ loss in DME is still under-read (see the README)."""
    p = logits.float().softmax(1)[:, EZ].amax(1).clamp(1e-6, 1 - 1e-6)  # (B, W); clamp avoids log(0)
    present = (y == EZ).any(1).float()           # (B, W): grader drew EZ in the column?
    retina = ((y >= 1) & (y <= 4)).any(1)        # (B, W): any retina in the column at all?
    if not retina.any():
        return logits.sum() * 0.0  # keeps the graph; nothing to score
    p, t = p[retina], present[retina]            # keep only the columns with retina
    # Binary cross-entropy written out: F.binary_cross_entropy refuses to run under CUDA autocast.
    return -(t * p.log() + (1 - t) * (1 - p).log()).mean()


class LayerSegmenter(L.LightningModule):
    """The training wrapper: loss, logging and optimiser around the plain UNet in gpu_service/unet/unet.py.
    Only the inner UNet is saved for the service (save_checkpoint), not this wrapper."""

    def __init__(self, width=32, depth=4, lr=1e-3, dice_weight=1.0, ez_weight=0.0, weight_decay=0.0,
                 lr_patience=0):
        super().__init__()
        self.save_hyperparameters()  # stores the arguments in the checkpoint and as self.hparams
        self.net = UNet(width=width, depth=depth)

    def forward(self, x):
        return self.net(x)                       # logits (B, 6, H, W)

    def loss(self, logits, y):
        # Cross-entropy (every pixel right) + soft Dice (every band, thin ones included, well
        # overlapped): the CE + Dice pairing nnU-Net uses by default (Isensee et al., Nat Methods 2021).
        loss = F.cross_entropy(logits.float(), y) + self.hparams.dice_weight * (1 - soft_dice(logits, y))
        if self.hparams.ez_weight:               # the served model uses the EZ column loss too
            loss = loss + self.hparams.ez_weight * ez_column_loss(logits, y)
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch                             # images, masks
        loss = self.loss(self(x), y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)  # one value per epoch
        return loss                              # Lightning backpropagates it

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.log("val_loss", self.loss(logits, y))  # early stopping and checkpointing watch this
        pred = logits.argmax(1)                  # the predicted label per pixel
        for label, name in BANDS.items():  # per-band Dice, for the training curves only
            p, t = pred == label, y == label
            dice = (2 * (p & t).sum() + 1) / (p.sum() + t.sum() + 1)
            self.log(f"val_dice_{name}", dice.float())
        # How far the predicted share of columns with EZ is from the grader's, per scan.
        presence = [(m == EZ).any(1).float().mean(1) for m in (pred, y)]
        self.log("val_ez_presence_mae", (presence[0] - presence[1]).abs().mean())

    def configure_optimizers(self):
        # AdamW with weight decay 0.01 as in the team's oct-analysis model; with
        # weight_decay=0 AdamW is plain Adam.
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        if not self.hparams.lr_patience:         # no schedule: constant learning rate
            return opt
        # Halve the learning rate when validation loss stalls (as in the team's oct-evaluation model).
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                             patience=self.hparams.lr_patience)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": plateau, "monitor": "val_loss"}}


class StepTimer(L.Callback):
    """Per epoch: seconds the training loop waited for the next batch (loading, augmentation,
    transfer) vs seconds it spent computing. A large wait share means the data pipeline, not the
    GPU, limits training -- the thing to check before buying a faster GPU."""

    def on_train_epoch_start(self, trainer, pl_module):
        self.wait = self.compute = 0.0           # reset the two totals
        self.mark = time.perf_counter()          # a stopwatch reading

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        now = time.perf_counter()
        self.wait += now - self.mark  # time since the last batch finished = waiting for this one
        self.mark = now

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # otherwise queued GPU work would be counted as the next wait
        now = time.perf_counter()
        self.compute += now - self.mark          # time spent on this batch = compute
        self.mark = now

    def on_train_epoch_end(self, trainer, pl_module):
        pl_module.log_dict({"time_data_wait_s": self.wait, "time_compute_s": self.compute})  # into metrics.csv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="gpu_service/unet/config.yaml")  # the YAML recipe
    parser.add_argument("--folds", type=int, nargs="*", help="default: all")
    parser.add_argument("--smoke", action="store_true")                     # a quick speed/memory check
    return parser.parse_args()


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)                 # YAML -> dict (safe_load: no code execution)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.smoke:
        cfg["experiment_name"] += "_smoke"  # never mixes with real checkpoints
    out = Path("outputs") / cfg["experiment_name"]
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, out / "config.yaml")  # the YAML travels with its checkpoints
    trainer_cfg = dict(cfg["trainer"])           # a copy, so the smoke settings don't leak into cfg
    if args.smoke:
        trainer_cfg.update(max_epochs=1, limit_train_batches=30, limit_val_batches=10)
    shift_mode = cfg["data"].get("shift_mode", "roll")  # recorded in the checkpoint for the service

    for fold in args.folds if args.folds is not None else range(cfg["data"]["n_folds"]):
        # Seed = config seed + fold: each fold's initial weights and draws are fixed but different.
        # The fold SPLIT uses the config seed alone (inside the data module), so it is the same for all.
        L.seed_everything(cfg["seed"] + fold, workers=True)
        data = OCT5kDataModule(fold=fold, seed=cfg["seed"], **cfg["data"])
        model = LayerSegmenter(**cfg["model"])
        # keep the best epoch's weights (by val_loss, set in the config)
        checkpoint = ModelCheckpoint(dirpath=out / f"fold{fold}" / "checkpoints", **cfg["checkpoint"])
        callbacks = [EarlyStopping(**cfg["early_stopping"]), checkpoint, StepTimer(),
                     LearningRateMonitor(logging_interval="epoch")]
        trainer = L.Trainer(**trainer_cfg, logger=CSVLogger(out, name=f"fold{fold}", version=""),
                            callbacks=callbacks)
        start = time.perf_counter()
        trainer.fit(model, datamodule=data)      # train until early stopping or max_epochs
        minutes = (time.perf_counter() - start) / 60
        peak = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0  # GB
        print(f"fold {fold}: {trainer.current_epoch} epochs in {minutes:.1f} min, peak GPU {peak:.2f} GB, "
              f"best val_loss {checkpoint.best_model_score}", flush=True)
        if args.smoke:
            return
        # Reload the best epoch (lowest val_loss), not the last, and save just the network.
        best = LayerSegmenter.load_from_checkpoint(checkpoint.best_model_path)
        save_checkpoint(out / f"fold{fold}.pt", best.net, f"{cfg['experiment_name']}-fold{fold}", shift_mode)


if __name__ == "__main__":
    main()
