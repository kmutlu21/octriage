"""Does the confidence gate work? Out-of-fold predictions vs the three OCT5k graders.

In plain words: The main test. For every B-scan: did the gate flag it, and was the model's mask
actually poor (it agrees with the graders clearly less than they agree with each other)? From that
come the headline numbers: how many scans are flagged, and how many poor masks the flags catch.

Mirrors Domalpally et al., Ophthalmology Science 2022;2:100198, where an AI probability
score and an ROC-chosen cutoff routed likely mislabels to human review (27.6% reviewed, error
rate 11.7% -> 1.5%). The same questions, asked of a segmentation model: how many scans are
flagged, how many poor masks are among the flagged, how many are left in the rest.

Naming: the code keeps the 2022 paper's words ("passed", "reviewed"), because results.json uses
them. Here "reviewed" = flagged to read first and "passed" = the routine queue, which is still
read: the gate orders the reading, it does not skip anything (docs/figures.py priority_curve
shows how fast the poor masks are found in that order).

Definitions, fixed before any out-of-fold result was seen:
- Agreement of two masks: Dice per retinal band (labels 1-4), averaged over the bands.
- Human level of a scan: mean agreement over the three grader pairs.
- Model level: mean agreement of the model with each grader. (Agreement with a consensus mask
  would flatter the model, because the consensus contains each grader.)
- Error ("poor mask"): model level < human level - MARGIN (0.05; 0.02 and 0.10 as sensitivity).
  The model may disagree with the graders as much as they disagree with each other, but not
  much more. The 0.05 is OURS; no published margin exists for layer masks. For scale only (a
  different task): WRC graders rated GA masks with mean Dice 0.91 "minor edits" and 0.79
  "moderate edits" (Mankad et al., ARVO 2025).
- Gate: pass if confidence >= threshold and retina_present_fraction >= 0.9 (the gear's rule).
- Threshold: Youden's J on the ROC curve, as WRC 2022 chose its cutoff from ROC curves.
  Performance is reported cross-validated: each fold's scans are gated with a threshold chosen
  on the other four folds, so no scan helps choose its own cutoff.
- Clustering: 60 patients x ~28 B-scans, and scans of one eye are not independent, so every
  confidence interval is a patient bootstrap (resample whole patients).

    python -m analysis.gate_analysis --experiment outputs/unet_w32_cv5_ez_border
Writes <experiment>/analysis/results.json (every number in the README), scans.csv (one row per
scan) and figures/.
"""
import argparse                          # command-line options
import json                              # results.json
from pathlib import Path                 # file paths

import matplotlib

matplotlib.use("Agg")  # draw to files, no screen needed (servers, containers)
import matplotlib.pyplot as plt          # figures
import numpy as np                       # arrays
import pandas as pd                      # tables
from PIL import Image                    # reading masks

from gpu_service.app import measurements  # the service's own measurement code, applied to graders too

BANDS = (1, 2, 3, 4)                 # retinal bands; vitreous (0) and below-RPE (5) are not scored
PAIRS = ((0, 1), (0, 2), (1, 2))     # the three grader pairs
MARGIN, SENSITIVITY_MARGINS = 0.05, (0.02, 0.10)  # "poor mask" margin; stricter/looser re-runs
MIN_RETINA = 0.9                     # same as the gear's min_retina_fraction
N_BOOT = 2000                        # bootstrap resamples


# ---------- statistics ----------
def dice(a: np.ndarray, b: np.ndarray, label: int) -> float:
    """Dice of one label between two masks: 2 x overlap / (size in a + size in b)."""
    a, b = a == label, b == label                  # True where each mask has this label
    total = int(a.sum() + b.sum())
    # both empty = perfect agreement (1.0)
    return 1.0 if total == 0 else 2 * int((a & b).sum()) / total  # same convention as gpu_service/tta.py


def rank(x) -> np.ndarray:
    return pd.Series(x).rank().to_numpy()  # average ranks for ties


def spearman(x, y) -> float:
    """Spearman correlation = Pearson correlation of the ranks."""
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def auc(score, positive) -> float:
    """P(a random positive scores above a random negative); ties count half (Mann-Whitney U)."""
    pos = np.asarray(positive, bool)
    n1, n0 = pos.sum(), (~pos).sum()               # number of positives, negatives
    if n1 == 0 or n0 == 0:                         # undefined with only one class
        return float("nan")
    # Mann-Whitney: sum of the positives' ranks, minus the smallest possible sum, over all pairs
    return float((rank(score)[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def youden_threshold(conf, ok) -> float:
    """The cutoff maximising (share of acceptable scans passed) - (share of errors passed),
    i.e. Youden's J = sensitivity + specificity - 1. Candidates are the observed confidences."""
    conf, ok = np.asarray(conf), np.asarray(ok, bool)
    if ok.all() or not ok.any():                   # one class only: pass everything
        return float(conf.min())
    # try every observed confidence as the cutoff t; keep the t with the largest J
    return float(max(np.unique(conf), key=lambda t: ((conf >= t) & ok).sum() / ok.sum()
                     - ((conf >= t) & ~ok).sum() / (~ok).sum()))


def icc_a1(x: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, absolute agreement, single rater (Shrout & Fleiss 1979).
    x is (subjects, raters). 'Absolute agreement': a rater who is always 2 um thicker lowers it.
    Tested against Shrout & Fleiss's worked example in tests/test_analysis.py."""
    n, k = x.shape                                         # n subjects, k raters
    grand = x.mean()                                       # the grand mean
    msr = k * ((x.mean(1) - grand) ** 2).sum() / (n - 1)   # between-subject mean square
    msc = n * ((x.mean(0) - grand) ** 2).sum() / (k - 1)   # between-rater mean square
    resid = x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + grand  # what subject + rater don't explain
    mse = (resid**2).sum() / ((n - 1) * (k - 1))           # residual mean square
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def bootstrap(df: pd.DataFrame, stat, n=N_BOOT, seed=0) -> list:
    """[estimate, 2.5%, 97.5%], resampling whole patients (with replacement).
    Fixed seed: every call resamples the same patients, so results are reproducible and paired."""
    rng = np.random.default_rng(seed)
    groups = df.groupby("patient").indices  # patient -> row positions
    patients = list(groups)
    # each resample: draw 60 patients with replacement, take ALL their B-scans, recompute the statistic
    vals = [stat(df.iloc[np.concatenate([groups[p] for p in rng.choice(patients, len(patients))])])
            for _ in range(n)]
    return [float(stat(df)), *map(float, np.nanpercentile(vals, [2.5, 97.5]))]  # estimate, 95% CI


# ---------- per-scan table ----------
def score_scans(oof: pd.DataFrame, data_root: Path, exp_dir: Path) -> pd.DataFrame:
    """For every scan: the graders' agreement with each other, the model's agreement with them,
    and each grader's EZ measurements (for the Bland-Altman comparison)."""
    rows = []
    for r in oof.itertuples():                                                   # every B-scan
        g = [np.array(Image.open(data_root / getattr(r, f"mask_g{i}"))) for i in (1, 2, 3)]  # 3 grader masks
        m = np.array(Image.open(exp_dir / r.pred_mask))                          # the model's mask
        human = np.mean([[dice(g[i], g[j], b) for b in BANDS] for i, j in PAIRS], axis=0)  # per band
        model = np.mean([[dice(m, gi, b) for b in BANDS] for gi in g], axis=0)   # model vs each grader, per band
        row = {"scan_id": r.scan_id, "human_level": human.mean(), "model_level": model.mean(),
               "human_ez": human[2], "model_ez": model[2],  # index 2 = band 3 = EZ
               "grader_drew_no_retina": any(not (gi == 1).any() for gi in g)}
        for i, gi in enumerate(g, 1):                     # each grader's EZ, measured the service's way
            meas = measurements(gi)
            row[f"ez_thick_g{i}"], row[f"ez_frac_g{i}"] = meas["ez_mean_thickness_px"], meas["ez_present_fraction"]
        rows.append(row)
    return oof.merge(pd.DataFrame(rows), on="scan_id")   # add the new columns to the predictions table


def cv_gate(df: pd.DataFrame, error: np.ndarray) -> tuple[np.ndarray, dict]:
    """Gate each fold with a threshold chosen on the other folds only."""
    passed, thresholds = np.zeros(len(df), bool), {}
    for k in sorted(df.fold.unique()):
        train, test = (df.fold != k).to_numpy(), (df.fold == k).to_numpy()   # other folds / this fold
        t = youden_threshold(df.confidence[train], ~error[train])           # cutoff from the other folds
        thresholds[int(k)] = t
        # the gear's rule on this fold: confident enough AND retina in >= 90% of columns
        passed[test] = (df.confidence[test] >= t).to_numpy() & (df.retina_present_fraction[test] >= MIN_RETINA).to_numpy()
    return passed, thresholds


def operating(df: pd.DataFrame) -> dict:
    """The 2022 paper's headline numbers, for the rows in df (needs `passed` and `error`)."""
    p, e = df.passed.to_numpy(), df.error.to_numpy()
    return {"reviewed": float(1 - p.mean()),                      # share sent to a person
            "error_rate_without_gate": float(e.mean()),           # poor masks if nobody reviewed
            "error_rate_among_passed": float(e[p].mean()) if p.any() else float("nan"),  # what slips through
            "errors_caught": float((e & ~p).sum() / e.sum()) if e.any() else float("nan")}  # share of poor masks reviewed


# ---------- measurement agreement (EZ) ----------
def ez_agreement(df: pd.DataFrame, col: str) -> dict:
    """ICC and Bland-Altman (bias, 95% limits of agreement), graders vs graders and model vs
    graders: the agreement statistics WRC's own papers use (e.g. Bogost et al., Ophthalmol Sci 2025)."""
    graders = df[[f"{col}_g{i}" for i in (1, 2, 3)]].to_numpy()   # (scans, 3)
    model = df[f"{col}_model"].to_numpy()[:, None]                 # (scans, 1)
    grader_diffs = np.concatenate([graders[:, i] - graders[:, j] for i, j in PAIRS])  # grader minus grader
    model_diffs = (model - graders).ravel()                        # model minus each grader
    # Bland-Altman: mean difference (bias) and bias +- 1.96 SD (95% limits of agreement)
    ba = lambda d: {"bias": float(d.mean()), "loa": [float(d.mean() - 1.96 * d.std(ddof=1)),
                                                     float(d.mean() + 1.96 * d.std(ddof=1))]}
    return {"icc_graders": icc_a1(graders), "icc_graders_plus_model": icc_a1(np.hstack([graders, model])),
            "grader_vs_grader": ba(grader_diffs), "model_vs_grader": ba(model_diffs)}


# ---------- figures ----------
def figures(df: pd.DataFrame, fig_dir: Path, threshold: float) -> None:
    """Three diagnostic figures: ROC, confidence vs grader agreement, EZ Bland-Altman."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = {"AMD": "#1f77b4", "DME": "#d62728", "Normal": "#2ca02c"}
    ok = ~df.error.to_numpy()                                      # acceptable masks

    # 1. ROC curve of confidence as a pass signal, with the chosen cutoff marked.
    fig, ax = plt.subplots(figsize=(5, 5))
    ts = np.unique(df.confidence)                                  # every possible cutoff
    # x: share of poor masks that would pass; y: share of acceptable masks that would pass
    ax.plot([((df.confidence >= t) & ~ok).sum() / max((~ok).sum(), 1) for t in ts],
            [((df.confidence >= t) & ok).sum() / ok.sum() for t in ts], color="k")
    ax.scatter(((df.confidence >= threshold) & ~ok).sum() / max((~ok).sum(), 1),
               ((df.confidence >= threshold) & ok).sum() / ok.sum(), color="r", zorder=3,
               label=f"Youden cutoff {threshold:.3f}")
    ax.plot([0, 1], [0, 1], ls=":", color="grey")                  # the diagonal = a random score
    ax.set(xlabel="errors passed (FPR)", ylabel="acceptable scans passed (TPR)",
           title=f"Confidence as a pass signal, AUC {auc(df.confidence, ok):.3f}")
    ax.legend(loc="lower right")
    fig.savefig(fig_dir / "roc.png", dpi=150, bbox_inches="tight")

    # 2. Does the model's confidence drop where the graders themselves disagree?
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for c, sub in df.groupby("cohort"):
        ax.scatter(sub.human_level, sub.confidence, s=8, alpha=0.6, color=colors.get(c), label=c)
    ax.set(xlabel="inter-grader agreement (mean pairwise Dice)", ylabel="model confidence (TTA)",
           title=f"Confidence vs human agreement, Spearman {spearman(df.human_level, df.confidence):.2f}")
    ax.legend()
    fig.savefig(fig_dir / "confidence_vs_graders.png", dpi=150, bbox_inches="tight")

    # 3. Bland-Altman: EZ thickness, model vs each grader (red) over grader vs grader (grey).
    fig, ax = plt.subplots(figsize=(6, 4.5))
    g = df[[f"ez_thick_g{i}" for i in (1, 2, 3)]].to_numpy()
    for (i, j), label in zip(PAIRS, ("grader vs grader", None, None)):
        ax.scatter((g[:, i] + g[:, j]) / 2, g[:, i] - g[:, j], s=4, alpha=0.3, color="grey", label=label)
    for i in range(3):
        mdl = df.ez_thick_model.to_numpy()
        ax.scatter((mdl + g[:, i]) / 2, mdl - g[:, i], s=4, alpha=0.3, color="r",
                   label="model vs grader" if i == 0 else None)
    ax.axhline(0, color="k", lw=0.8)                               # zero difference
    ax.set(xlabel="mean EZ thickness (px)", ylabel="difference (px)", title="Bland-Altman: EZ band thickness")
    ax.legend()
    fig.savefig(fig_dir / "bland_altman_ez.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", type=Path, default=Path("outputs/unet_w32_cv5_ez_border"))  # the trained CV run
    ap.add_argument("--data", type=Path, default=Path("data"))                 # OCT5k masks
    args = ap.parse_args()
    out = args.experiment / "analysis"

    oof = pd.read_csv(args.experiment / "oof_predictions.csv")  # written by gpu_service/unet/predict.py
    df = score_scans(oof, args.data, args.experiment)          # add human and model agreement per scan
    df = df.rename(columns={"ez_mean_thickness_px": "ez_thick_model", "ez_present_fraction": "ez_frac_model"})
    df["error"] = (df.model_level < df.human_level - MARGIN).to_numpy()   # "poor mask": the definition above
    df["passed"], thresholds = cv_gate(df, df.error.to_numpy())           # routine queue (True) or read first
    # The cutoff to deploy: chosen on all scans (the per-fold cutoffs above are for honest evaluation).
    # This exact value is the gear's default confidence_threshold (tests/test_gear_config.py checks it).
    deploy_t = youden_threshold(df.confidence, ~df.error)

    res = {
        "n_scans": len(df), "n_patients": int(df.patient.nunique()), "margin": MARGIN,
        "human_level_mean": float(df.human_level.mean()), "model_level_mean": float(df.model_level.mean()),
        "auc_confidence_for_acceptable": bootstrap(df, lambda d: auc(d.confidence, ~d.error)),
        "spearman_confidence_vs_inter_grader": bootstrap(df, lambda d: spearman(d.confidence, d.human_level)),
        "spearman_confidence_vs_model_level": bootstrap(df, lambda d: spearman(d.confidence, d.model_level)),
        "cv_thresholds_per_fold": thresholds, "deployment_threshold_all_folds": deploy_t,
        "gate_cross_validated": {k: bootstrap(df, lambda d, k=k: operating(d)[k]) for k in operating(df)},
        "by_cohort": {c: {"n": len(s), "auc": auc(s.confidence, ~s.error), **operating(s),
                          "spearman_conf_vs_inter_grader": spearman(s.confidence, s.human_level)}
                      for c, s in df.groupby("cohort")},
        "sensitivity_margins": {},
        "ez_thickness": {"all": ez_agreement(df, "ez_thick"), "passed": ez_agreement(df[df.passed], "ez_thick"),
                         "reviewed": ez_agreement(df[~df.passed], "ez_thick")},
        "ez_present_fraction": {"all": ez_agreement(df, "ez_frac")},
    }
    # Sensitivity: does the conclusion survive a stricter (0.02) or looser (0.10) error margin,
    # or judging the EZ band alone?
    for m in SENSITIVITY_MARGINS:
        e = (df.model_level < df.human_level - m).to_numpy()   # poor masks under the other margin
        d = df.assign(error=e, passed=cv_gate(df, e)[0])       # re-gate with cutoffs chosen for that margin
        res["sensitivity_margins"][str(m)] = {"auc": auc(d.confidence, ~d.error), **operating(d)}
    e = (df.model_ez < df.human_ez - MARGIN).to_numpy()        # poor mask judged on the EZ band only
    d = df.assign(error=e, passed=cv_gate(df, e)[0])
    res["sensitivity_margins"]["ez_band_only"] = {"auc": auc(d.confidence, ~d.error), **operating(d)}

    # POST HOC, added after the results were seen: frames where a grader drew no inner retina at
    # all. On inspection these are noise frames with no visible retina (two graders drew one anyway,
    # the third painted the frame as RPE), so their "errors" are errors in the reference, not the model.
    # Reported separately; the headline numbers above keep them.
    valid = df[~df.grader_drew_no_retina].reset_index(drop=True)   # drop the 4 noise frames
    valid = valid.assign(passed=cv_gate(valid, valid.error.to_numpy())[0])
    res["posthoc_excluding_invalid_frames"] = {
        "excluded": df.scan_id[df.grader_drew_no_retina].tolist(),
        "auc": bootstrap(valid, lambda d: auc(d.confidence, ~d.error)),
        **{k: bootstrap(valid, lambda d, k=k: operating(d)[k]) for k in operating(valid)}}

    out.mkdir(parents=True, exist_ok=True)
    # "\n" line endings on every OS, so a rerun on Windows is byte-identical to one in the container.
    df.to_csv(out / "scans.csv", index=False, lineterminator="\n")
    (out / "results.json").write_text(json.dumps(res, indent=2), newline="\n")
    figures(df, out / "figures", deploy_t)
    # the headline numbers, on screen
    print(json.dumps({k: res[k] for k in ("n_scans", "human_level_mean", "model_level_mean",
                                          "auc_confidence_for_acceptable", "spearman_confidence_vs_inter_grader",
                                          "gate_cross_validated")}, indent=2))


if __name__ == "__main__":
    main()
