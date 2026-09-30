"""Why this confidence score? Step 2: compare ours with two published scores, exactly as
pre-registered in analysis/prereg/a0_confidence.md (the rule was written and frozen before any
score was computed). The script refuses to run if that file no longer matches the hash it had
when it was frozen, so the rule cannot be edited after seeing results.

In plain words: Is our confidence score good, or would a published one do better? This compares all
three on the same scans by how many poor masks each leaves unflagged at the same workload, under a
rule written down before any score was computed.

Which confidence score should the gate use (Q2), and is the chosen one good enough that an
ensemble is not needed (Q1)? Three scores, all judged on the mask the gate returns today:
  s_current  our 6-pass TTA agreement (gpu_service/tta.py)                          [ours]
  s_pe       single-network mean predictive entropy (Zenk et al., MedIA 2024)  [ref]
  s_wang     test-time-augmentation volume variation, N = 20 (Wang et al. 2019) [ref]
Risk = 1 - mean Dice with the graders (Zenk's R = 1 - DSC); metric = AURC via Zenk's own code
(analysis/third_party/zenk_metrics.py, Apache-2.0).

AURC in one sentence: sort scans from most to least confident, pass them one by one, and
average the risk of the passed set along the way; lower = the score keeps bad masks out better.

Result on the served model: ours has the lowest AURC and both differences exclude 0, so the rule
keeps ours ("outcome B"). The rule's B clause recommends a seed ensemble (Zenk's best method,
~21 GPU-hours); it was not trained (a scope decision). At WRC, nnSAM's five fold models would
give the ensemble score without new training.

    python -m analysis.confidence_scores --config gpu_service/unet/config.yaml     # step 1, GPU, ~30 min
    python -m analysis.confidence_eval --experiment outputs/unet_w32_cv5_ez_border
"""
import argparse                          # command-line options
import hashlib                           # sha256 of the frozen rule file
import json                              # results.json
from itertools import combinations       # every pair of scores
from pathlib import Path                 # file paths

import matplotlib

matplotlib.use("Agg")                    # draw to files, no screen needed
import matplotlib.pyplot as plt          # the risk-coverage figure
import numpy as np                       # arrays
import pandas as pd                      # tables
from PIL import Image                    # reading masks

from analysis import gate_analysis as ga                 # shared statistics (bootstrap, Dice, Spearman)
from analysis.third_party import zenk_metrics as zm      # Zenk et al.'s own AURC code (Apache-2.0)

PREREG = Path(__file__).parent / "prereg" / "a0_confidence.md"
PREREG_SHA256 = "0aedd00a45f0918bc92e6b80ecc91965d61c1f5ee2e7f3e4a7d6422f29a4e359"  # frozen 2026-09-26 16:08
SCORES = {"s_current": {"ref": False, "passes": 6},   # [ours]
          "s_pe": {"ref": True, "passes": 1},         # Zenk 2024 single network
          "s_wang": {"ref": True, "passes": 20}}      # Wang 2019
WRC_REVIEW, WRC_ERROR = 0.276, 0.015  # Domalpally et al., Ophthalmol Sci 2022;2:100198


# ---------- pure pieces (tested in tests/test_confidence_eval.py) ----------
def aurc(confids, risks, metric: str = "aurc") -> float:
    """Zenk's implementation, unchanged; higher confidence = more trusted."""
    stats = zm.StatsCache(confids=np.asarray(confids, float), risks=np.asarray(risks, float))  # Zenk's input object
    return float(zm.get_metric_function(metric)(stats))         # "aurc", "norm-aurc" or "e-aurc"


def passed_at_review(conf, scan_ids, review_rate: float) -> np.ndarray:
    """The least-confident round(review_rate * n) scans go to review; ties broken by scan_id."""
    conf, n = np.asarray(conf, float), len(conf)
    order = np.lexsort((np.asarray(scan_ids), -conf))  # most confident first; lexsort's LAST key sorts first
    passed = np.zeros(n, bool)
    passed[order[: n - round(review_rate * n)]] = True  # the most confident (1 - rate) x n pass
    return passed


def error_among_passed(d: pd.DataFrame, score: str, review_rate: float) -> float:
    """Share of poor masks left in the routine queue when `review_rate` of scans are flagged by `score`."""
    p = passed_at_review(d[score], d.scan_id, review_rate)
    return float(d.error.to_numpy()[p].mean())


def decide(aurc_est: dict, diff_ci: dict, q1_error: dict, scores: dict = SCORES) -> dict:
    """The pre-registered rule. diff_ci[(a, b)] = 95% CI of AURC(a) - AURC(b), for every ordered pair.
    Q2: lowest AURC; tied set = it plus every score whose difference CI to it includes 0; choose
    [ref] before [ours], then fewer forward passes. Q1: error among passed at 27.6% review <= 1.5%.
    Outcome A: [ref] and passes Q1 (no ensemble now); B: [ours] kept because no [ref] score ties it
    (recommend the ensemble to Kaan); C: fails Q1 (recommend the ensemble)."""
    best = min(aurc_est, key=aurc_est.get)                        # the score with the lowest AURC
    # plus every score whose difference to the best is not shown (its CI includes 0)
    tied = [best] + [s for s in aurc_est if s != best and diff_ci[(s, best)][0] <= 0 <= diff_ci[(s, best)][1]]
    chosen = min(tied, key=lambda s: (not scores[s]["ref"], scores[s]["passes"]))  # published first, then cheaper
    passes_q1 = q1_error[chosen] <= WRC_ERROR                     # WRC 2022's 1.5% bar
    outcome = "C" if not passes_q1 else ("A" if scores[chosen]["ref"] else "B")
    return {"lowest_aurc": best, "tied_set": tied, "chosen": chosen,
            "q1_error_among_passed_at_27.6pct": q1_error[chosen], "q1_passes": passes_q1, "outcome": outcome,
            "meaning": {"A": "reference score is good enough: no ensemble training now",
                        "B": "only our own score performs: recommend the Zenk-exact seed ensemble to Kaan",
                        "C": "chosen score misses the WRC 2022 bar: recommend the Zenk-exact seed ensemble"}[outcome]}


# ---------- data ----------
def load(exp: Path, data_root: Path) -> pd.DataFrame:
    """Join the gate analysis's per-scan table with step 1's scores; check they describe the same scans."""
    scans = pd.read_csv(exp / "analysis" / "scans.csv")                       # from gate_analysis.py
    a0 = pd.read_csv(exp / "analysis" / "a0_confidence" / "scores.csv")       # from analysis/confidence_scores.py
    df = scans.merge(a0[["scan_id", "fold", "s_pe", "s_wang_A", "s_wang_B"]], on="scan_id", suffixes=("", "_a0"))
    assert len(df) == len(scans) == len(a0), "every scan must have all three scores"
    assert (df.fold == df.fold_a0).all(), "scores must come from the same fold models"
    df = df.assign(s_current=df.confidence, s_wang=df.s_wang_A)  # ours = the gate's confidence; Wang = draw set A
    for g in (1, 2, 3):  # per-grader Dice of the served mask, for the per-grader sensitivity
        df[f"dice_g{g}"] = [np.mean([ga.dice(np.array(Image.open(exp / m)), np.array(Image.open(data_root / gm)), b)
                                     for b in ga.BANDS])
                            for m, gm in zip(df.pred_mask, df[f"mask_g{g}"])]
    assert np.allclose(df[["dice_g1", "dice_g2", "dice_g3"]].mean(axis=1), df.model_level), "must match the pre-fixed model level"
    return df.assign(risk=1 - df.model_level)                    # Zenk's risk = 1 - Dice


# ---------- figure ----------
def figure(df: pd.DataFrame, path: Path, current_review: float) -> None:
    """Risk-coverage curves: x = share of scans passed, y = mean risk of the passed scans."""
    # dataviz reference palette, categorical slots 1-3 in fixed order; line style is the second channel
    style = {"s_current": ("#2a78d6", "-", "current TTA agreement (ours)"),
             "s_pe": ("#eb6834", "--", "single-pass entropy (Zenk 2024)"),
             "s_wang": ("#1baf7a", "-.", "TTA volume variation (Wang 2019)")}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s, (color, ls, label) in style.items():                  # one curve per score
        cov, risk, _ = zm.StatsCache(confids=df[s].to_numpy(float), risks=df.risk.to_numpy(float)).rc_curve_stats
        ax.plot(cov, risk, color=color, ls=ls, lw=2, label=f"{label}: AURC {aurc(df[s], df.risk):.4f}")
    # The best any score could do: rank by the true risk itself.
    cov, risk, _ = zm.StatsCache(confids=-df.risk.to_numpy(float), risks=df.risk.to_numpy(float)).rc_curve_stats
    ax.plot(cov, risk, color="#8a8983", ls=":", lw=1.5, label=f"optimal ranking: AURC {aurc(-df.risk, df.risk):.4f}")
    ymin, ymax = 0.03, 0.10  # fixed across models so their figures compare
    # vertical lines at WRC 2022's review rate and at the gate's own
    for rate, text in ((WRC_REVIEW, "27.6% reviewed (WRC 2022)"), (current_review, f"{current_review:.1%} reviewed (gate today)")):
        ax.axvline(1 - rate, color="#52514e", lw=0.8)
        ax.text(1 - rate, ymin + 0.002, f" {text}", color="#52514e", fontsize=8, va="bottom", ha="right", rotation=90)
    top = df.sort_values("s_current", ascending=False).iloc[0]   # the scan our score trusts most
    if top.grader_drew_no_retina:  # an earlier model ranked a noise frame most confident, off the chart's top
        ax.text(0.2, ymax - 0.002, f"Off-scale at the far left: {top.scan_id} (no visible retina, risk {top.risk:.2f}) is "
                "rated\nfully confident by the current and Wang scores; the gate's retina check reviews it.",
                color="#52514e", fontsize=7.5, va="top")
    ax.set(xlabel="coverage (share of scans passed without review)", ylabel="mean risk of passed scans (1 - Dice)",
           title="Risk-coverage: which score best keeps bad masks out of the passed set", xlim=(0, 1), ylim=(ymin, ymax))
    ax.grid(color="#e6e5e0", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, loc="lower left", frameon=False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", type=Path, default=Path("outputs/unet_w32_cv5_ez_border"))  # the trained CV run
    ap.add_argument("--data", type=Path, default=Path("data"))                 # OCT5k masks
    args = ap.parse_args()
    sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()        # fingerprint of the rule file today
    if sha != PREREG_SHA256:                                     # edited after freezing: refuse
        raise SystemExit(f"{PREREG} changed after it was frozen (sha256 {sha}); the rule no longer holds")
    out = args.experiment / "analysis" / "a0_confidence"
    df = load(args.experiment, args.data)                        # one row per scan, all three scores
    # The gate's real review rate, from gate_analysis.py (8.8% for the served model).
    current_review = json.loads((args.experiment / "analysis" / "results.json").read_text())["gate_cross_validated"]["reviewed"][0]

    aurc_ci = {s: ga.bootstrap(df, lambda d, s=s: aurc(d[s], d.risk)) for s in SCORES}  # AURC + 95% CI each
    diff_ci = {}                                                 # CI of each pairwise AURC difference
    for a, b in combinations(SCORES, 2):  # paired: every call resamples the same patients (seed 0)
        est, lo, hi = ga.bootstrap(df, lambda d, a=a, b=b: aurc(d[a], d.risk) - aurc(d[b], d.risk))
        diff_ci[(a, b)], diff_ci[(b, a)] = (lo, hi), (-hi, -lo)  # a - b and b - a
    q1 = {s: error_among_passed(df, s, WRC_REVIEW) for s in SCORES}  # error left at 27.6% flagged
    decision = decide({s: v[0] for s, v in aurc_ci.items()}, diff_ci, q1)  # apply the frozen rule

    valid = df[~df.grader_drew_no_retina]                        # without the 4 noise frames
    res = {
        # Repo-relative path, so the result does not depend on where the repo was cloned.
        "prereg": {"file": PREREG.relative_to(Path(__file__).parents[1]).as_posix(), "sha256": sha},
        "n_scans": len(df), "n_patients": int(df.patient.nunique()),
        "decision": decision,
        "aurc": aurc_ci,
        "aurc_difference_ci": {f"{a} - {b}": [aurc(df[a], df.risk) - aurc(df[b], df.risk), *diff_ci[(a, b)]]
                               for a, b in combinations(SCORES, 2)},
        "reference_points": {"optimal_aurc": aurc(-df.risk, df.risk), "random_aurc": float(df.risk.mean())},
        "q1_error_among_passed_at_27.6pct": {s: ga.bootstrap(df, lambda d, s=s: error_among_passed(d, s, WRC_REVIEW))
                                            for s in SCORES},
        "reported_not_decisive": {  # pre-registered as "report, but do not decide on"
            "per_grader_aurc": {s: {f"g{g}": aurc(df[s], 1 - df[f"dice_g{g}"]) for g in (1, 2, 3)} for s in SCORES},
            "excluding_no_retina_frames": {"n": len(valid), **{s: ga.bootstrap(valid, lambda d, s=s: aurc(d[s], d.risk))
                                                              for s in SCORES}},
            "norm_aurc": {s: aurc(df[s], df.risk, "norm-aurc") for s in SCORES},
            "e_aurc": {s: aurc(df[s], df.risk, "e-aurc") for s in SCORES},
            f"matched_review_{current_review:.4f}": {
                s: {"error_among_passed": error_among_passed(df, s, current_review),
                    "errors_caught": float((df.error & ~passed_at_review(df[s], df.scan_id, current_review)).sum()
                                           / df.error.sum())} for s in SCORES},
            "wang_stability_A_vs_B": {  # does Wang's random score change with a different random draw?
                "spearman": ga.spearman(df.s_wang_A, df.s_wang_B),
                "aurc_A": aurc(df.s_wang_A, df.risk), "aurc_B": aurc(df.s_wang_B, df.risk),
                **{f"decisions_differ_at_{r:.4f}_review": float(np.mean(passed_at_review(df.s_wang_A, df.scan_id, r)
                                                                       != passed_at_review(df.s_wang_B, df.scan_id, r)))
                   for r in (WRC_REVIEW, current_review)}},
            "spearman_between_scores": {f"{a} vs {b}": ga.spearman(df[a], df[b]) for a, b in combinations(SCORES, 2)},
            "by_cohort_aurc": {c: {s: aurc(sub[s], sub.risk) for s in SCORES} for c, sub in df.groupby("cohort")},
        },
    }
    # POST HOC, added after an earlier model's results were seen. The pre-registered tie rule treats
    # "no difference shown" (CI includes 0) as "equal", and the chosen score then passed ~3x more
    # errors than ours at the gate's real workload. These check that without changing the verdict
    # above: paired CIs at the current review rate, and without the 4 no-retina frames that the
    # retina check already reviews.
    pairs = [("s_pe", "s_current"), ("s_wang", "s_current")]     # each published score vs ours
    res["posthoc_added_after_results"] = {
        "why": "tie rule = CI includes 0, which is not equivalence; check the practical cost of the chosen score",
        f"errors_passed_at_{current_review:.4f}_review": {
            "n_errors": int(df.error.sum()),
            **{s: int((df.error & passed_at_review(df[s], df.scan_id, current_review)).sum()) for s in SCORES},
            **{f"{a} - {b} (error among passed)": ga.bootstrap(df, lambda d, a=a, b=b: error_among_passed(d, a, current_review)
                                                            - error_among_passed(d, b, current_review)) for a, b in pairs}},
        "excluding_no_retina_frames_aurc_difference": {
            f"{a} - {b}": ga.bootstrap(valid, lambda d, a=a, b=b: aurc(d[a], d.risk) - aurc(d[b], d.risk)) for a, b in pairs},
    }
    # "\n" line endings on every OS, so a rerun on Windows is byte-identical to one in the container.
    (out / "results.json").write_text(json.dumps(res, indent=2), newline="\n")
    df[["scan_id", "patient", "cohort", "fold", "risk", "error", "dice_g1", "dice_g2", "dice_g3", "grader_drew_no_retina",
        *SCORES, "s_wang_B"]].to_csv(out / "per_scan.csv", index=False, lineterminator="\n")
    figure(df, out / "risk_coverage.png", current_review)
    # the decision and the AURCs, on screen
    print(json.dumps({k: res[k] for k in ("decision", "aurc", "aurc_difference_ci", "reference_points")}, indent=2))


if __name__ == "__main__":
    main()
