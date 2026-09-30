# A0 step 1 — which confidence score, and is an ensemble needed? (pre-registered)

Written 2026-09-26, before any of the scores below had been computed. `analysis/confidence_eval.py`
implements exactly this and records this file's sha256 in its output, so the rule cannot drift
after the results are seen. Nothing here needs model training.

Labels: **[ref]** = taken exactly from the cited source; **[ours]** = our choice, with the reason.

## Why
The gate's current confidence score (6 fixed flip/shift passes, each pass's mask compared by Dice
with the consensus) follows no published method exactly: Wang 2019 used random transforms and the
volume variation coefficient; Roy 2019 used pairwise Dice between MC-dropout samples; Zenk 2024
used pairwise Dice between ensemble members. The rule (Kaan, 2026-09-26): follow references,
leave no room for our own interpretation; try the methods that need no training first.

## The scores (higher = more confident; all from the same fold model that never saw the patient)
| Score | Definition | Source |
|---|---|---|
| `s_current` | mean Dice (bands 1-4) of each of 6 fixed passes vs their consensus, as served today | **[ours]**, `service/tta.py` |
| `s_wang` | −mean over bands 1-4 of VVC_b = σ_b/μ_b of band b's pixel area across N = 20 test-time augmentations, each inverted before measuring | **[ref]** Wang et al., Neurocomputing 2019;335:34-45, §3.5 (VVC) and N = 20 for 2D |
| `s_pe` | −mean over all pixels of the predictive entropy −Σ_c p_c log p_c of the un-augmented softmax | **[ref]** Zenk et al., MedIA 2024 (doi:10.1016/j.media.2024.103392), single-network baseline with mean aggregation |

Details of `s_wang`:
- **[ref] rule, [ours] values:** Wang's rule is that the test-time transforms follow the same prior as the training augmentation ("similarly to data augmentation at training time"). Wang's own priors (rotation 0–2π, scale 0.8–1.2) were for fetal MRI. Ours are the model's training augmentations, `model/data.py`:
  - horizontal flip ~ Bernoulli(0.5);
  - vertical roll ~ U{−16, …, 16} px;
  - intensity x·U(0.9, 1.1) + U(−0.05, 0.05), clamped to [0, 1].
- **Draws:** 20 draws per scan with a fixed seed (draw set A, seed 0). A second, independent draw set B (seed 1) exists only for the stability check.
- **[ours] Band average:** averaging over the 4 bands follows Zenk's convention for multi-class data (mean over classes).
- **[ours] Absent bands:** if a band is absent in all 20 draws (μ_b = 0), VVC_b = 0: every draw agrees.
- **[ours] σ:** the population standard deviation of the 20 areas.

**[ours] What is evaluated:** every score is judged on the same mask, the one the gate returns today (the 6-pass consensus, `oof_masks/`). The comparison then isolates the ranking. The mask the gate returns does not change.

## Risk and metric
- **Risk per scan: [ref] form, [ours] reference.** risk = 1 − Dice (Zenk 2024: R = 1 − DSC, mean over classes). Dice here is the pre-fixed "model level": mean Dice (bands 1-4) of the mask against each of the three graders (fixed in session 2, before any out-of-fold result). Zenk had one ground truth; OCT5k has three.
- **Metric: [ref]** AURC (area under the risk–coverage curve), computed by Zenk's own code (`analysis/third_party/zenk_metrics.py`, MIC-DKFZ/segmentation_failures_benchmark @ a1af98be, Apache-2.0). Lower is better.
- **Inference:** a paired patient bootstrap, 2,000 resamples, seed 0 (the same as `gate_analysis.py`). It gives a 95% percentile CI for every pairwise AURC difference.

## Decision rule
**Q2, which no-training score should the gate use?**
1. Take the score with the lowest AURC on all 1,672 scans.
2. The tied set = that score, plus every score whose AURC-difference CI against it includes 0.
3. Choose from the tied set by these tie-breakers, in order:
   - (a) **[ref] before [ours]** (Kaan's rule);
   - (b) fewer forward passes (`s_pe` 1 < `s_current` 6 < `s_wang` 20; "simpler wins ties", agreed in session 4).

**Q1, is the chosen score good enough, or is an ensemble needed?**
- **Anchor: [ref]** WRC's own tiered workflow (Domalpally, Slater et al., Ophthalmol Sci 2022;2:100198) reviewed 27.6% of images and left a 1.5% error rate.
- **Test:** send the least-confident 27.6% of scans to review (round(0.276 × 1672) = 461; ties broken by scan_id). The chosen score passes Q1 if the error rate among the passed scans is ≤ 1.5% (point estimate, as the 2022 paper reported).
- **[ours] Error:** the pre-fixed definition, model level < human level − 0.05. The 2022 paper's errors were image labels, not segmentations, so the anchor is an analogy.

**Outcomes:**
- **A:** the chosen score is [ref] and passes Q1 → no ensemble training now. The ensemble stays in the design register (ideal vs built).
- **B:** the chosen score is `s_current` ([ours]), because no [ref] score is tied with it → the "no interpretation" rule and performance conflict → recommend the Zenk-exact seed ensemble to Kaan (his decision; about 21 GPU-hours).
- **C:** the chosen score fails Q1 → recommend the Zenk-exact seed ensemble (Zenk 2024: ensembles beat single networks in most cases).

## Reported, but not part of the decision
- AURC per grader (risk against grader 1, 2, 3 alone).
- AURC excluding the 4 frames with no visible retina (flagged post hoc in session 2).
- Zenk's normalised AURC and e-AURC.
- Errors among passed at a matched review rate of 11.8%, today's CV operating point (matched workload: Fleming 2024; Rudnicka 2025).
- **Stability of `s_wang`** (no reference sets a threshold, so this is descriptive only):
  - Spearman correlation between draw sets A and B;
  - AURC of each draw set;
  - the share of scans whose pass/review decision differs between A and B at 27.6% and at 11.8% review.
- **Not in scope:** changing the service, re-deriving the gear cutoff (the rounding bug is bundled with any gear change), and the retina check (unchanged; not under test).
