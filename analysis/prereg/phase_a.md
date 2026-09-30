# Phase A — volume checks, trial endpoints, worklist, and whether each check earns its place (pre-registered)

Written 2026-09-26, after A0 (analysis/prereg/a0_confidence.md) and before any Phase A quantity had
been computed from a model mask. `analysis/phase_a.py` implements exactly this, and it refuses to run
if this file's sha256 differs from the one frozen in the code. The confidence score stays as served
(A0 decision, Kaan 2026-09-26).

Labels: **[ref]** = taken from the cited source; **[ours]** = our choice, with the reason. WRC/UW
sources are listed first.

## Geometry (all checks)
- **Axial scale: 3.5 µm per pixel** (the OCT5k paper, Sci Data doi:10.1038/s41597-024-04259-z: "axial resolution is 3.5 µm"). **[ours] caveat:** that is a resolution figure, not necessarily the sampling; Spectralis often samples at ~3.9 µm. That puts ±10% on every µm and mm² value.
- **Lateral scale: 8.9 mm / 512 = 17.38 µm per mask pixel.** The paper's "8.9×7.4 mm²". **[ours] inference:** 8.9 mm runs along the B-scan.
- **B-scan spacing: 7.4 mm / (N − 1).** N = the volume's B-scan count, i.e. the highest b-index. **[ours] inference:** 7.4 mm runs across the B-scans. This gives 411 / 308 / 247 / 206 / 123 µm for N = 19 / 25 / 31 / 37 / 61.
- **Boundaries from a label map [ours, exact on the human masks]:** boundary k (1 ILM, 2 OPL, 3 IS-OS, 4 IBRPE, 5 OBRPE) in a column = the first row whose label is ≥ k. It is undefined if no such row exists. On human masks this returns the traced boundary exactly, because of the painting rule (`prep/build_manifest.py`). Model and grader masks go through the same function.

## A1 — layer order (per B-scan)
- **Rule:** a column is out of order if, reading top to bottom, a label is followed by a smaller label. The B-scan is **flagged if any column is out of order**.
- **Why zero tolerance:**
  - **[ref]** He et al., MICCAI 2019 and MedIA 2021: retinal layers must keep their topological order.
  - **[ref]** Garvin et al., IEEE TMI 2009 (doi:10.1109/TMI.2009.2016958): surface ordering is a hard feasibility constraint.
  - **[ref data]** 0 of 5,016 human masks break the order in any column (verified 2026-09-26).
  - **[ref, WRC]** Boundary line errors are the most common OCT artifact at the reading centre (Domalpally et al., Retina 2009;29:775-81).

## A2b — continuity between neighbouring B-scans (per B-scan)
- **The raw quantity:** for each adjacent pair of B-scans (i, i+1), each boundary k, and each column x, the depth change Δ = (z_k(i+1, x) − z_k(i, x)) × 3.5 µm.
- **Column constraint [ref]:** Garvin 2009 §III-C learns feasibility constraints from reference tracings as mean ± 2.6 SD, "to allow for 99% of the expected" values. Here:
  - the learned values are μ_k and σ_k of Δ over the graders' tracings (all three graders, mask-derived boundaries);
  - they are learned separately for each B-scan spacing;
  - a column violates if any boundary's Δ falls outside μ_k ± 2.6 σ_k.
  - **[ours]** The constraints are constant over columns, which is Garvin's original constant formulation (§II-A1). Garvin's per-column extension needs a common coordinate grid, and volumes of 19 to 61 B-scans don't share one.
- **Pair constraint [ours; each threshold is learned by Garvin's same rule]:** f = the fraction of columns that violate. The pair is flagged if f > mean + 2.6 SD of f over the graders' own pairs at that spacing. Why a second step: by construction about 1% of the graders' own columns fall outside the column limits, so flagging a single violating column would flag the reference standard itself.
- **B-scan flag [ours]:** a B-scan is flagged if every neighbouring pair it has is flagged. That means both pairs for an interior B-scan, and one pair at the volume's edge or next to a missing B-scan. The reason: one bad B-scan breaks continuity with both of its neighbours.
- **No leakage:** limits are learned on the other four folds' patients (the model's folds) and applied to the held-out fold. A spacing that has no training patients (possible only for N = 37, a single patient) uses limits pooled over all training spacings **[ours]**. Deployment limits are learned on all 60 patients.

## A2 — en face maps and trial endpoints (per visit, i.e. per volume)
Computed the same way for the model and for each grader; a rater's B-scan joins the map only if it passes the retina check (≥ 90% of columns show retina, the gear's existing rule).
- **Thickness map [ref, WRC]:** T = ILM → top of RPE (IBRPE), the centre-subfield definition of Pak, …, Danis, IOVS 2013;54:4512-8 (doi:10.1167/iovs.13-12265), UW Fundus Photograph Reading Center.
- **Fovea [ours]:** the minimum of T within 1.5 mm of the scan centre (the ETDRS inner-ring radius). No reading-centre method exists to copy: Pak 2013 placed the fovea by hand. We report it against each grader, using the same algorithm on the grader's map.
- **Decentration flag [ref, WRC]:** the fovea is more than 200 µm from the scan centre. Pak 2013: "statistically significant error when the displacement distance … exceeded 200 μm". Decentration was 15.4% of manual remeasurements in Domalpally 2009.
- **CST [ref]:** the mean T within 500 µm of the fovea, i.e. the central 1-mm subfield (ETDRS grid). The grid is re-centred on the fovea, following Pak 2013's recommendation of "post hoc repositioning of the grid". CST is a secondary endpoint in SCORE2 (NCT01969708) and Protocol T (NCT01627249). **[ours]** Only sampled points are used; there is no interpolation between B-scans.
- **EZ-loss area [ref]:** the number of en face points where the column shows retina but no EZ band, × 17.38 µm × the B-scan spacing, in mm². The area of EZ loss is the FDA-approved Encelto endpoint (2025); pixel area from spacing follows the team's `ez-rpe-reportgear`.

## A3 — per-visit findings and the worklist
- **Per visit:**
  - B-scans flagged by each adopted check;
  - which flagged B-scans lie within the central subfield (their y within 500 µm of the fovea);
  - the decentration flag, CST and EZ-loss area;
  - a provenance line (model, confidence threshold, the checks applied).
- **Priority tier [ref for the ranking principle; the tiers themselves are ours]:**
  - **Tier 1:** decentred, OR a flagged B-scan inside the central subfield. Either can change the endpoint (FDA 2018 imaging guidance p.14: the endpoint variables are named in the charter; Holmen et al., JAMA Ophthalmol 2020: artifact severity graded by the grid area affected).
  - **Tier 2:** flagged B-scans outside the central subfield only.
  - **Tier 3:** nothing flagged.
  - **Within a tier:** more flagged B-scans first, then visit id.
- **Every visit is still read.** The worklist orders the reading; it does not skip any (FDA 2018 p.25).

## A4 — does each check earn its place? (decision rule)
**Primary, per B-scan** (1,672 B-scans; error = model level < human level − 0.05, fixed in session 2):
- **Base gate G0:** the current gate (confidence at each fold's cross-validated cutoff, AND the retina check), i.e. `scans.csv` `passed`.
- **Step 1, A1:**
  - G1 = G0, plus every A1-flagged B-scan sent to review.
  - Comparator M1: the same number of reviews spent on confidence alone. Order the B-scans with retina-check failures first, then by ascending confidence, and review the first |review(G1)|.
  - **Adopt A1 if G1 lets strictly fewer errors pass than M1, AND G1 reviews ≤ 27.6% of B-scans** (WRC 2022, Ophthalmol Sci 2022;2:100198). Matched workload follows Fleming 2024 and Rudnicka 2025.
- **Step 2, A2b:** the same test, starting from the gate adopted after step 1.
- **Ties** (equal errors passed) → not adopted: the simpler gate wins.
- **Reported alongside:** the patient-bootstrap 95% CI of the difference in errors passed (2,000 resamples, seed 0).

**Secondary, per visit (descriptive; 60 visits):**
- **Endpoint error [ref, WRC]:** |CST_model − mean of the graders' CSTs| > 20.1 µm, the SD-OCT CST coefficient of repeatability (Domalpally et al., OSLI 2010, doi:10.3928/15428877-20100325-01). Report how many endpoint-error visits land in Tier 1 and in Tiers 1+2, and the same for visits containing at least one B-scan error.
- **Endpoint agreement [ref: the WRC agreement statistics used in Bogost 2025 and Etheridge 2020]:**
  - CST and EZ-loss area: ICC(2,1) for the graders alone and for the graders + model; Bland–Altman against each grader (patient = visit, so there is no clustering).
  - Fovea distance, model vs each grader, in µm.
  - Share of visits decentred, for each rater.

## Not in scope
- Changing the confidence score or retraining.
- Any cutoff other than the confidence cutoff. The gear's cutoff becomes the exact deployment value 0.988926516…; the old rounding bug is fixed in the same rebuild.
