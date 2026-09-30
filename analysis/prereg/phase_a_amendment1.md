# Phase A — Amendment 1 (declared POST HOC, 2026-09-26, before the amended results were computed)

The registered analysis (`phase_a.md`, sha256 dfd24884…) was run first. Its results are kept
unchanged in `outputs/unet_w32_cv5_ez/analysis/phase_a/registered/` and are the primary record:
- **A1 (layer order):** flags 1,587 of 1,672 B-scans, so it is **not adopted**.
- **A2b (continuity):** 3 errors pass vs 2 with confidence alone at the same workload, so it is **not adopted**.
- **A2 (endpoints):** CST limits of agreement of −109 to +141 µm, and 59 of 60 visits "decentred" for the model and 47–49 of 60 for each grader.

Diagnosis (debugging only; no error or outcome was used):
1. **Stray labels in model masks.** The model's pixel-wise masks contain out-of-order labels.
   - The largest share is a wrap-around artifact of our own training augmentation. `model/data.py` shifts images with `roll`, so choroid (label 5) appears in the top rows of 1,232 B-scans, up to 15 rows.
   - Stray labels remain elsewhere too: 1,025 B-scans are still out of order after the edge runs are removed.
   - The registered boundary rule ("first row with label ≥ k") jumps to any such stray pixel. The RPE then lands at row 0, and the thickness map, fovea and CST are corrupted.
   - The human masks are always ordered, so the flaw is invisible on them.
2. **Sampling coarser than the threshold.** B-scans are 308–411 µm apart in 50 of 60 volumes. A fovea on any B-scan other than the middle one is therefore > 200 µm "off", and the graders' own maps trip the decentration rule.

## The only changes
- **(a) Boundaries by count:** boundary k in a column = the number of pixels whose label is < k (undefined if no pixel has a label ≥ k).
  - This is the cumulative-thickness form, in which boundaries are ordered by construction. He et al. (MICCAI 2019; MedIA 2021) use it to guarantee layer order.
  - On an ordered column it equals the registered rule exactly, so every human mask gives identical results; a test checks this on real OCT5k masks.
  - On a model mask, a stray pixel moves a boundary by one pixel instead of to its own row.
  - Thickness (ILM → top of RPE) becomes the count of pixels labelled 1–3 in the column, which is how the service already measures band thickness.
- **(b) Decentration only where it can be measured:** the 200 µm rule (Pak 2013) applies only when the B-scan spacing is ≤ 200 µm, i.e. N = 61 here. Elsewhere it is reported as "not assessable at this B-scan spacing" and does not set Tier 1. **[ours]** Reason: a displacement smaller than the sampling step cannot be measured.

Everything else in `phase_a.md` is unchanged:
- the A1 rule (still zero tolerance; its column count is reported, not gated);
- the continuity learning rule and CV;
- the A4 decision rule and the secondary analyses.

Decisions taken from this amended run are post hoc and are reported as such next to the registered ones.
