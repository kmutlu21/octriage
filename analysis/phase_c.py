"""Phase C: the real reading schedule, two more definitions of "poor", and honest intervals,
exactly as pre-registered in analysis/prereg/phase_c.md (the script refuses to run if that file
changed since it was frozen). Nothing here re-decides the gate; it describes it better.

In plain words: Simulates a grader reading the visits in priority order and counts how many poor
masks and CST errors are found as the reading goes on, compared with a random order. It also checks
that the result holds under two other definitions of a poor mask, and how much the numbers would
move with another set of patients.

  C1  the schedule a grader would follow: whole visits in worklist order, the gear's reading
      order inside each visit
  C2  "poor" by a margin derived from the graders (95th percentile of leave-one-grader-out gaps)
  C3  "poor" by WRC's thickness threshold (any band's mean thickness > 20 µm off the graders')
  C4  patient bootstrap stratified by cohort, with the per-fold cutoffs chosen again in every resample

    python -m analysis.phase_c --experiment outputs/unet_w32_cv5_ez_border
"""
import argparse   # command-line options
import hashlib    # sha256 of the frozen pre-registration
import json       # results file
from pathlib import Path  # file paths

import numpy as np   # arrays
import pandas as pd  # tables
from PIL import Image  # reading masks

from analysis import gate_analysis as ga   # dice, BANDS, cv_gate, youden_threshold, MIN_RETINA
from oct_checks.geometry import AXIAL_UM            # 3.5 µm per pixel, top to bottom

ROOT = Path(__file__).resolve().parents[1]
PREREG = Path(__file__).parent / "prereg" / "phase_c.md"
PREREG_SHA256 = "9298d030a10421023e4d20a4e8e44e0cf350b1bc19d7160de1ee236f60a2b0ec"  # frozen 2026-09-28
WRC_UM = 20.0            # lower end of WRC's "±20 µm to ±30 µm" reproducibility thresholds (SIIM 2026)
HUMAN_PCT = 95           # δ = this percentile of the graders' own gaps


# ---------- C1: the schedule ----------
def schedule(bscans: pd.DataFrame, visits: pd.DataFrame) -> pd.DataFrame:
    """Every B-scan once, in the order a grader would read them: visits by worklist rank, and inside
    a visit the gear's reading order (triage.reading_order): flagged first (no retina first, then
    least confident), then the rest, least confident first."""
    parts = []
    for v in visits.sort_values("rank").itertuples():                       # visits in worklist order
        d = bscans[bscans.patient == v.visit].copy()
        d["flagged"] = d.review_final.astype(bool)                          # the gate's read-first set
        d["hard"] = d.flagged & ~d.retina_ok.astype(bool)                   # hard failure: no retina
        first = d[d.flagged].sort_values(["hard", "confidence", "bscan"], ascending=[False, True, True])
        rest = d[~d.flagged].sort_values(["confidence", "bscan"])
        parts.append(pd.concat([first, rest]).assign(tier=v.tier, visit_rank=v.rank))
    out = pd.concat(parts, ignore_index=True)
    out["read"] = np.arange(1, len(out) + 1)                                 # B-scans read so far
    out["poor_found"] = out.error.astype(int).cumsum()                      # poor masks found so far
    return out


# ---------- C2 and C3: other definitions of "poor" ----------
def band_mean(pair_dice: list) -> float:
    return float(np.mean(pair_dice))


def per_scan_measures(df: pd.DataFrame, data: Path, exp: Path) -> pd.DataFrame:
    """For every B-scan: the three leave-one-grader-out gaps (C2) and the worst band's thickness
    difference, model vs the graders' mean (C3)."""
    rows = []
    for r in df.itertuples():
        g = [np.array(Image.open(data / getattr(r, f"mask_g{i}"))) for i in (1, 2, 3)]  # grader masks
        m = np.array(Image.open(exp / r.pred_mask))                                   # the model's mask
        d = {(i, j): band_mean([ga.dice(g[i], g[j], b) for b in ga.BANDS]) for i, j in ga.PAIRS}
        pair = lambda i, j: d[(min(i, j), max(i, j))]
        gaps = []
        for k in range(3):                                   # grader k against the other two, a and b
            a, b = [x for x in range(3) if x != k]
            gaps.append(pair(a, b) - (pair(k, a) + pair(k, b)) / 2)
        thick = lambda mask: np.array([(mask == b).sum(0).mean() * AXIAL_UM for b in ga.BANDS])  # µm per band
        diff = np.abs(thick(m) - np.mean([thick(x) for x in g], axis=0))              # model vs graders' mean
        rows.append({"scan_id": r.scan_id, "gap_g1": gaps[0], "gap_g2": gaps[1], "gap_g3": gaps[2],
                     "worst_band_um": float(diff.max()), "worst_band": int(ga.BANDS[int(diff.argmax())])})
    return df.merge(pd.DataFrame(rows), on="scan_id")


def where(poor: pd.Series, flagged: pd.Series) -> dict:
    """How many poor masks there are, and where the gate put them."""
    poor, flagged = poor.astype(bool), flagged.astype(bool)
    return {"n_poor": int(poor.sum()), "in_flagged": int((poor & flagged).sum()),
            "in_routine": int((poor & ~flagged).sum()), "n_routine": int((~flagged).sum()),
            "share_in_flagged": float((poor & flagged).sum() / poor.sum()) if poor.any() else None}


# ---------- C4: stratified, nested bootstrap ----------
def fast_youden(conf: np.ndarray, ok: np.ndarray) -> float:
    """gate_analysis.youden_threshold, vectorised: same candidates (observed confidences), same
    tie rule (the smallest t with the largest J). Checked against the original in main()."""
    if ok.all() or not ok.any():
        return float(conf.min())
    t = np.unique(conf)                                                  # ascending candidates
    s_ok, s_err = np.sort(conf[ok]), np.sort(conf[~ok])
    ge = lambda s: len(s) - np.searchsorted(s, t, side="left")          # how many >= each t
    j = ge(s_ok) / ok.sum() - ge(s_err) / (~ok).sum()
    return float(t[int(np.argmax(j))])                                   # argmax keeps the first maximum


def fast_cv_gate(df: pd.DataFrame) -> np.ndarray:
    """gate_analysis.cv_gate with fast_youden: each fold's cutoff chosen on the other folds."""
    conf, err = df.confidence.to_numpy(), df.error.to_numpy(bool)
    retina_ok = df.retina_present_fraction.to_numpy() >= ga.MIN_RETINA
    fold = df.fold.to_numpy()
    passed = np.zeros(len(df), bool)
    for k in np.unique(fold):
        train, test = fold != k, fold == k
        t = fast_youden(conf[train], ~err[train])
        passed[test] = (conf[test] >= t) & retina_ok[test]
    return passed


def stats(df: pd.DataFrame, passed: np.ndarray) -> dict:
    err = df.error.to_numpy(bool)
    return {"flagged_share": float((~passed).mean()),
            "poor_in_flagged_share": float((err & ~passed).sum() / err.sum()),
            "poor_rate_routine": float((err & passed).sum() / passed.sum())}


def nested_bootstrap(df: pd.DataFrame, n: int = ga.N_BOOT, seed: int = 0) -> dict:
    """Resample patients within each cohort (to its own count), re-choose the cutoffs, recompute."""
    rng = np.random.default_rng(seed)
    by_cohort = {c: d.groupby("patient").indices for c, d in df.groupby("cohort")}
    frames = {c: d.reset_index(drop=True) for c, d in df.groupby("cohort")}
    vals = []
    for _ in range(n):
        parts = []
        for c, groups in by_cohort.items():
            pts = list(groups)
            pick = rng.choice(pts, len(pts))                              # patients, with repeats
            parts.append(frames[c].iloc[np.concatenate([groups[p] for p in pick])])
        d = pd.concat(parts, ignore_index=True)
        vals.append(stats(d, fast_cv_gate(d)))
    point = stats(df, fast_cv_gate(df))
    return {k: [point[k], *map(float, np.nanpercentile([v[k] for v in vals], [2.5, 97.5]))] for k in point}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", type=Path, default=Path("outputs/unet_w32_cv5_ez_border"))
    ap.add_argument("--data", type=Path, default=Path("data"))
    args = ap.parse_args()
    sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    if sha != PREREG_SHA256:
        raise SystemExit(f"{PREREG} changed after it was frozen (sha256 {sha}); the rule no longer holds")
    an = args.experiment / "analysis"
    out = an / "phase_c"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(an / "scans.csv")                                     # the gate's flags and outcomes
    bscans = pd.read_csv(an / "phase_a" / "amended" / "bscans.csv")        # + B-scan numbers, final flags
    visits = pd.read_csv(an / "phase_a" / "amended" / "visits.csv")        # tiers, ranks, CST errors

    # C1
    sch = schedule(bscans, visits)
    n, total = len(sch), int(sch.error.sum())
    vis = visits.sort_values("rank").reset_index(drop=True)
    vis["cst_found"] = vis.cst_error.fillna(False).astype(bool).cumsum()
    vis["poor_visits_found"] = vis.any_bscan_error.astype(bool).cumsum()
    def at_end_of(tier: int) -> dict:
        s, v = sch[sch.tier <= tier], vis[vis.tier <= tier]
        return {"bscans_read": len(s), "bscans_read_share": len(s) / n, "poor_found": int(s.error.sum()),
                "visits_read": len(v), "cst_error_visits_found": int(v.cst_found.iloc[-1]),
                "poor_visits_found": int(v.poor_visits_found.iloc[-1])}
    c1 = {"n_bscans": n, "n_poor": total, "n_visits": len(vis),
          "n_cst_error_visits": int(vis.cst_found.iloc[-1]), "n_poor_visits": int(vis.poor_visits_found.iloc[-1]),
          "end_of_tier1": at_end_of(1), "end_of_tier2": at_end_of(2),
          "all_poor_found_after": int(sch.read[sch.poor_found >= total].iloc[0])}
    sch[["read", "scan_id", "patient", "tier", "visit_rank", "flagged", "hard", "confidence", "error",
         "poor_found"]].to_csv(out / "schedule.csv", index=False, lineterminator="\n")
    vis[["rank", "visit", "tier", "n_bscans", "cst_error", "any_bscan_error", "cst_found",
         "poor_visits_found"]].to_csv(out / "visits_schedule.csv", index=False, lineterminator="\n")

    # C2, C3
    m = per_scan_measures(df, args.data, args.experiment)
    human_gaps = m[["gap_g1", "gap_g2", "gap_g3"]].to_numpy().ravel()
    delta = float(np.percentile(human_gaps, HUMAN_PCT))
    flagged = ~m.passed.astype(bool)
    c2 = {"delta": delta, "human_gap_median": float(np.median(human_gaps)),
          **where((m.human_level - m.model_level) > delta, flagged)}
    c3 = {"threshold_um": WRC_UM, **where(m.worst_band_um > WRC_UM, flagged),
          "worst_band_counts": m[m.worst_band_um > WRC_UM].worst_band.value_counts().sort_index().to_dict()}
    primary = where(m.error, flagged)                                      # the 0.05 margin, for comparison
    m[["scan_id", "gap_g1", "gap_g2", "gap_g3", "worst_band_um", "worst_band"]].to_csv(
        out / "per_scan.csv", index=False, lineterminator="\n")

    # C4: check the fast cutoff equals the original on every fold, then bootstrap
    for k in sorted(df.fold.unique()):
        tr = df.fold != k
        assert fast_youden(df.confidence[tr].to_numpy(), ~df.error[tr].to_numpy(bool)) == \
            ga.youden_threshold(df.confidence[tr], ~df.error[tr]), f"fast Youden differs on fold {k}"
    assert (fast_cv_gate(df) == df.passed.to_numpy(bool)).all(), "fast cv_gate differs from scans.csv"
    c4 = nested_bootstrap(df)

    res = {"prereg": {"file": PREREG.relative_to(ROOT).as_posix(), "sha256": PREREG_SHA256},
           "c1_schedule": c1, "primary_margin_0.05": primary, "c2_grader_margin": c2, "c3_wrc_um": c3,
           "c4_stratified_nested_ci": c4,
           "c4_note": "[estimate, 2.5%, 97.5%]; cohorts resampled separately; per-fold cutoffs re-chosen per resample"}
    (out / "results.json").write_text(json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                                      newline="\n")
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
