# The U-Net: model card

A stand-in: the gears work with any model that answers the service's contract ([`gpu_service/app.py`](../app.py)).
The layout follows Mitchell et al. 2019, "Model Cards for Model Reporting" (doi:10.1145/3287560.3287596).

**What it does.** Labels every pixel of a Spectralis OCT B-scan as vitreous, one of four retinal bands
(ILM–OPL, OPL–IS/OS, EZ, RPE) or tissue below the RPE. Trained and tested only on OCT5k (AMD, DME,
healthy eyes; one device); not tested on trial data or other devices. A research model, not a medical device.

**The network.** A plain 2D U-Net ([Ronneberger 2015](https://doi.org/10.1007/978-3-319-24574-4_28)), 7.76 million parameters:

```
input     one B-scan, resized to 512 × 512
encoder   512² × 32 → 256² × 64 → 128² × 128 → 64² × 256 → 32² × 512 channels
          each level: two 3×3 convolutions, each + BatchNorm + ReLU; 2×2 max-pooling between levels
decoder   2×2 transposed convolution up, join the encoder level of the same size (skip), two 3×3 convolutions
output    1×1 convolution → a score for each of the 6 labels, per pixel
```
Width 32 and depth 4 are ours, sized for a 6 GB GPU.

**Training** (each value with its source in [`config.yaml`](config.yaml)). Loss = cross-entropy + soft Dice over the four
bands + an EZ column loss (ours: is there EZ in each image column?). AdamW (learning rate 0.001, weight
decay 0.01); learning rate halved after 5 epochs without improvement; early stopping on validation loss
(patience 10, at most 100 epochs); batch 4. Augmentation: horizontal flip, and a vertical shift of up to
±16 px that repeats the edge row. Half of each epoch's draws are B-scans with EZ loss (they are rare).

**How it was validated.** The 60 patients were split into 5 folds of 12, by patient, so no patient is in
two folds. For fold k: train on three folds, stop early on fold k+1, test on fold k. Every B-scan is
therefore scored by the one model that never saw its patient. Agreement = Dice per band, averaged over the
four bands: the model against each of the three graders, and each grader against the other two.

| Test fold | Patients / B-scans | Epochs run (best) | Model vs graders | Graders vs graders | Poor masks |
|---|---|---|---|---|---|
| 0 | 12 / 329 | 56 (46) | 0.919 | 0.916 | 11 |
| 1 | 12 / 342 | 30 (20) | 0.908 | 0.907 | 16 |
| 2 | 12 / 294 | 47 (37) | 0.913 | 0.910 | 11 |
| 3 | 12 / 377 | 37 (27) | 0.928 | 0.923 | 2 |
| 4 | 12 / 330 | 56 (46) | 0.915 | 0.911 | 14 |
| **All** | **60 / 1,672** | | **0.917** | **0.914** | **54** |

| Cohort | Model vs graders | Graders vs graders | EZ band only: model / graders |
|---|---|---|---|
| AMD | 0.919 | 0.916 | 0.894 / 0.884 |
| DME | 0.892 | 0.892 | 0.802 / 0.795 |
| Normal | 0.932 | 0.926 | 0.912 / 0.904 |

**Known limits.** One training run per fold; cuDNN autotuning makes a retrain differ slightly (predictions
from the saved weights reproduce exactly). DME is the weak spot: the model draws the EZ band as well as the
graders agree with each other, yet it under-reads EZ loss, the trial endpoint (EZ-loss area ICC 0.72 with
the model added, 0.97 among graders). The service runs fold 0's model; the analysis uses all five.
