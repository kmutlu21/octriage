# Phase C: the real reading schedule, three definitions of "poor", honest intervals

Written and frozen 2026-09-28, before any number below was computed. Code: `analysis/phase_c.py`,
which refuses to run if this file's sha256 changes. Nothing here re-decides the gate: the cutoffs,
flags and tiers are the ones already produced (`scans.csv`, `phase_a/amended/`). This phase only
describes them better. Every result below is reported whatever it shows.

## Why
Review round 11 found three weaknesses in how the results were shown:
1. The priority curve reads B-scans in one global queue (all flagged B-scans of all visits first).
   At WRC a grader gets a task per visit and reads that visit, so the schedule is visit first.
2. The 0.05 Dice margin that defines a "poor mask" is ours. Zenk et al. 2024 (Requirement R3):
   a failure threshold is hard to set "when inter-annotator variability is unknown"; OCT5k has
   three graders per B-scan, so the margin can be derived from them.
3. The bootstrap held the cutoffs fixed and did not stratify by cohort, so the CIs are too narrow.

## C1. The schedule a grader would follow (primary figure)
- Visits in the worklist order already written (`phase_a/amended/visits.csv`, column `rank`:
  tier 1 first; within a tier, most flagged B-scans first).
- Inside a visit, the gear's reading order (`triage/triage.py::reading_order`): flagged B-scans
  first (no retina first, then least confident), then the rest, least confident first.
- Whole visits are read before the next one starts.
- Report, against B-scans read in that schedule: poor masks found (of 54); and, against visits read,
  visits with a CST error found (of 4) and visits holding a poor mask found (of 20).
- Report the shares at the end of Tier 1 and at the end of Tier 2.
- Baseline: a random order (expected value: the diagonal). Perfect order also drawn.

## C2. A margin derived from the graders (robustness)
- For each B-scan and each grader g, with a and b the other two graders:
  human gap_g = Dice(a, b) − mean(Dice(g, a), Dice(g, b)), each Dice averaged over the four bands
  exactly as `gate_analysis.score_scans` does.
- δ = the 95th percentile of human gap_g over all B-scans and graders.
- A mask is poor(C2) if human_level − model_level > δ: the model falls further below the graders
  than 95% of human gradings do.
- Report δ; the number of poor(C2) masks; how many sit in the flagged set; how many in the
  routine queue; with the gate's flags unchanged.

## C3. WRC's thickness threshold (robustness)
- Per B-scan and band (1–4), mean band thickness over all 512 columns = band pixels per column
  × 3.5 µm, averaged (a column without the band counts 0).
- A mask is poor(C3) if, for any band, |model − mean of the three graders| > 20 µm: the lower end
  of the "reproducibility thresholds for clinical trial human grading (±20 µm to ±30 µm)" in
  Balasubramanian et al., SIIM 2026.
- Report the same four numbers as C2.

## C4. Intervals
- Patient bootstrap stratified by cohort (resample AMD, DME and Normal patients separately, each to
  its own count), 2,000 resamples, seed 0.
- Nested: in every resample the per-fold cutoffs are chosen again (`gate_analysis.cv_gate`), so
  the interval includes the uncertainty of choosing the cutoff.
- Statistics: share flagged; share of poor masks inside the flagged set (the headline 49 of 54);
  poor masks in the routine queue per routine B-scan.
- These replace the earlier CIs in the report and README, whatever they show. The earlier ones stay
  in `results.json`.

## Threshold-free evidence (already frozen, cited not recomputed)
- AURC from `analysis/prereg/a0_confidence.md` (ours 0.062; single-pass entropy 0.064; Wang 0.064).
