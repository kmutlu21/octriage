"""Phase A: the whole-visit analysis, exactly as pre-registered in analysis/prereg/phase_a.md
(the script refuses to run if that file changed since it was frozen).

In plain words: The gate works on single B-scans, but graders read whole visits. This script looks
at each visit: are the layers in order, do neighbouring B-scans agree, how far is the model's CST
from the graders', and which priority does the visit get? It also tests whether an extra check would
catch more poor masks than simply reading more low-confidence scans.

It asks, for the 1,672 out-of-fold masks and the 60 visits:
  A1  layer order: does a B-scan break the anatomical order of the layers?
  A2b continuity: is a B-scan consistent with its neighbours (limits learned from graders)?
  A2  endpoints per visit: CST and EZ-loss area, model vs the three graders
  A3  the reading worklist: tiers 1 / 2 / 3 per visit
  A4  does each check EARN a place in the gate? It must let fewer poor masks through than
      spending the same number of reviews on confidence alone (a matched-workload test).
All checks come from oct_checks/, the same code the gears run.

Amendment 1 (analysis/prereg/phase_a_amendment1.md), declared after the registered run showed
a flaw and before its results were read: boundaries by counting pixels (robust to stray pixels),
and decentration judged only where the B-scan spacing can resolve 200 µm. Both runs are kept.

    python -m analysis.phase_a --experiment outputs/unet_w32_cv5_ez_border --amendment 1
"""
import argparse   # command-line options
import hashlib    # sha256 of the frozen pre-registration files
import json       # results files
from pathlib import Path  # file paths

import matplotlib

matplotlib.use("Agg")  # draw to files, no screen needed
import matplotlib.pyplot as plt  # the CST figure
import numpy as np               # arrays
import pandas as pd              # tables
from PIL import Image            # reading mask PNGs

from analysis import gate_analysis as ga           # shared statistics (ICC, bootstrap, pairs)
from oct_checks import boundaries as qb                     # boundaries and the layer-order check
from oct_checks import continuity, enface, worklist         # neighbour check, endpoints, tiers
from oct_checks.geometry import AXIAL_UM, LATERAL_UM, bscan_spacing_um  # pixel sizes, B-scan spacing

ROOT = Path(__file__).resolve().parents[1]          # the repo folder (for repo-relative paths in results)
PREREG = Path(__file__).parent / "prereg" / "phase_a.md"
PREREG_SHA256 = "dfd248842e0a17413b5ff776f9c96b80b04839691c0d6b0463f3d6caa1e25f47"  # frozen 2026-09-26 17:20
# A correction the frozen file cannot take (its hash is checked): it cites FDA 2018 p.25 for "every visit
# is still read". That is reading-centre practice, not an FDA rule; p.25 suggests verifying a subset.
AMENDMENT = Path(__file__).parent / "prereg" / "phase_a_amendment1.md"
AMENDMENT_SHA256 = "c4c11e36f6ca2d8184466f909b0554b2518b9b9facd2a87818aff3c6991a38ba"  # frozen 2026-09-26 17:29
# --amendment 0 reproduces the registered run; 1 (default) applies Amendment 1, declared post hoc:
# boundaries by count, and decentration only where the B-scan spacing can resolve 200 µm.
METHOD = {0: "first_row", 1: "count"}
MIN_RETINA = ga.MIN_RETINA                 # the gear's retina check, 0.9
MAX_REVIEW = 0.276                         # WRC 2022 reviewed 27.6% (Ophthalmol Sci 2022;2:100198): the ceiling
CST_REPEATABILITY_UM = 20.1                # CST coefficient of repeatability, SD-OCT (Domalpally et al. OSLI 2010)
RATERS = ("model", "g1", "g2", "g3")       # the model and the three graders


# ---------- A4: does a check earn its place? ----------
def matched_step(review_base, flag, error, retina_ok, confidence) -> dict:
    """Add a check's flags to the review set, then spend the same number of reviews on confidence
    alone (retina failures first, then least confident) and compare the errors each lets pass."""
    review_base, flag, error = (np.asarray(a, bool) for a in (review_base, flag, error))
    review = review_base | flag                                   # reviewed with the check added
    k = int(review.sum())                                         # how many reviews that costs
    order = np.lexsort((np.asarray(confidence, float), np.asarray(retina_ok, bool)))  # no retina first, then least confident
    comparator = np.zeros(len(review), bool)
    comparator[order[:k]] = True                                  # the same k reviews, spent on confidence alone
    passed_with, passed_cmp = int((error & ~review).sum()), int((error & ~comparator).sum())  # errors left unread
    return {"reviewed": k / len(review), "n_reviewed": k, "newly_reviewed": int((review & ~review_base).sum()),
            "errors_passed_with_check": passed_with, "errors_passed_confidence_only": passed_cmp,
            # adopt only if it lets FEWER errors through at the same workload, within WRC's 27.6% ceiling
            "adopt": passed_with < passed_cmp and k / len(review) <= MAX_REVIEW, "review": review}


def step_bootstrap(df: pd.DataFrame, base_col: str, flag_col: str) -> list:
    """Patient-bootstrap CI of (errors passed with the check) - (errors passed with confidence alone)."""
    def stat(d):
        s = matched_step(d[base_col], d[flag_col], d.error, d.retina_ok, d.confidence)
        return s["errors_passed_with_check"] - s["errors_passed_confidence_only"]
    return ga.bootstrap(df, stat)


# ---------- loading ----------
def load_scans(exp: Path, data: Path, method: str) -> tuple[pd.DataFrame, dict]:
    """The per-scan table from gate_analysis.py, plus per-column summaries of every mask
    (the model's and the three graders'), and the layer-order count of the model's mask."""
    df = pd.read_csv(exp / "analysis" / "scans.csv")                      # one row per B-scan
    man = pd.read_csv(data / "manifest.csv")[["scan_id", "bscan"]]         # B-scan numbers
    df = df.merge(man, on="scan_id")
    summaries = {}                                                         # (scan, rater) -> summary
    for r in df.itertuples():
        m = np.array(Image.open(exp / r.pred_mask))                        # the model's mask
        summaries[(r.scan_id, "model")] = qb.column_summary(m, method)
        df.loc[r.Index, "a1_violation_columns"] = qb.order_violation_columns(m)  # A1 on the model
        for g in (1, 2, 3):                                                # each grader's mask
            gm = np.array(Image.open(data / getattr(r, f"mask_g{g}")))
            assert qb.order_violation_columns(gm) == 0, f"human mask out of order: {r.scan_id} g{g}"  # sanity
            summaries[(r.scan_id, f"g{g}")] = qb.column_summary(gm, method)
    df["retina_ok"] = df.retina_present_fraction >= MIN_RETINA             # the retina check per B-scan
    return df, summaries


def volumes(df: pd.DataFrame, summaries: dict) -> dict:
    """patient -> {n, spacing, cohort, fold, ids: per position scan_id or None, per rater summaries list}."""
    out = {}
    for p, sub in df.groupby("patient"):                                   # one visit per patient
        n = int(sub.bscan.max())                                           # B-scans in the volume
        ids = [None] * n                                                   # position -> scan id
        for r in sub.itertuples():
            ids[r.bscan - 1] = r.scan_id
        out[p] = {"n": n, "spacing": bscan_spacing_um(n), "cohort": sub.cohort.iloc[0], "fold": int(sub.fold.iloc[0]),
                  "ids": ids, **{rt: [summaries[(i, rt)] if i else None for i in ids] for rt in RATERS}}
    return out


def z_volume(summ: list) -> np.ndarray:
    """Stack one rater's boundary rows into (n_bscans, 5, W), NaN for missing B-scans."""
    width = next(s["z"].shape[1] for s in summ if s is not None)
    return np.stack([s["z"] if s is not None else np.full((5, width), np.nan) for s in summ])


def learn_limits(vols: dict, patients) -> dict:
    """Continuity limits per volume size (B-scan spacing), plus pooled, from the graders' tracings."""
    by_n = {}                                                              # volume size -> grader volumes
    for p in patients:
        by_n.setdefault(vols[p]["n"], []).extend(z_volume(vols[p][g]) for g in ("g1", "g2", "g3"))
    table = {n: continuity.learn(zs, AXIAL_UM) for n, zs in by_n.items()}  # Garvin's rule per size
    table["pooled"] = continuity.learn([z for zs in by_n.values() for z in zs], AXIAL_UM)  # all sizes together
    return table


# ---------- A2 ----------
def visit_endpoints(v: dict, rater: str, amendment: int) -> dict:
    """CST, EZ-loss area and centring for one visit, from one rater's masks."""
    summ = v[rater]
    usable = np.array([s is not None and s["retina"].mean() >= MIN_RETINA for s in summ])  # retina check
    thick, loss = enface.maps(summ, usable, AXIAL_UM)
    return enface.endpoints(thick, loss, v["spacing"], LATERAL_UM, decentration_needs_sampling=amendment >= 1)


def agreement(values: pd.DataFrame) -> dict:
    """values: visits x (g1, g2, g3, model), complete rows only. ICC with and without the model,
    and Bland-Altman of model minus each grader."""
    v = values.dropna()                                                    # visits every rater could measure
    g, m = v[["g1", "g2", "g3"]].to_numpy(), v[["model"]].to_numpy()       # graders (n x 3), model (n x 1)
    diffs = (m - g).ravel()                                                # model minus each grader
    # ICC(A,1) of the graders alone, then with the model as a 4th rater: if the model reads like a
    # grader, adding it barely changes the ICC. Limits of agreement = bias +- 1.96 SD (Bland-Altman).
    return {"n_visits": len(v), "icc_graders": ga.icc_a1(g), "icc_graders_plus_model": ga.icc_a1(np.hstack([g, m])),
            "model_minus_grader": {"bias": float(diffs.mean()), "loa": [float(diffs.mean() - 1.96 * diffs.std(ddof=1)),
                                                                       float(diffs.mean() + 1.96 * diffs.std(ddof=1))]},
            # the spread between human graders: the yardstick for the model's spread
            "grader_minus_grader_sd": float(np.concatenate([g[:, i] - g[:, j] for i, j in ga.PAIRS]).std(ddof=1))}


def figure_cst(visits: pd.DataFrame, path: Path) -> None:
    """Bland-Altman of CST: model minus the graders' mean, one point per visit, repeatability band shaded."""
    v = visits.dropna(subset=["cst_model", "cst_g1", "cst_g2", "cst_g3"])   # visits with a CST from everyone
    graders = v[["cst_g1", "cst_g2", "cst_g3"]].mean(axis=1)                  # the graders' average CST
    mean, diff = (v.cst_model + graders) / 2, v.cst_model - graders           # x: average, y: difference
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.axhspan(-CST_REPEATABILITY_UM, CST_REPEATABILITY_UM, color="#e6e5e0", zorder=0)  # ±20.1 µm
    ax.axhline(0, color="#52514e", lw=0.8)                                    # zero = perfect agreement
    for c, marker in (("AMD", "o"), ("DME", "s"), ("Normal", "^")):           # one marker shape per cohort
        s = v.cohort == c
        ax.scatter(mean[s], diff[s], s=28, marker=marker, color="#2a78d6", edgecolor="#fcfcfb", lw=0.8, label=c)
    ax.text(ax.get_xlim()[1], -CST_REPEATABILITY_UM - 2, "shaded: ±20.1 µm CST repeatability (Domalpally 2010) ",
            va="top", ha="right", fontsize=8, color="#52514e")
    ax.set(xlabel="CST, mean of model and graders (µm)", ylabel="model − graders' mean (µm)",
           title="Central subfield thickness: model vs graders, one point per visit")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, title="cohort (marker)", title_fontsize=8)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", type=Path, default=Path("outputs/unet_w32_cv5_ez_border"))  # the trained CV run
    ap.add_argument("--data", type=Path, default=Path("data"))                 # OCT5k images, masks, manifest
    ap.add_argument("--amendment", type=int, choices=(0, 1), default=1)       # 0 = registered run, 1 = amended
    args = ap.parse_args()
    # Refuse to run if a frozen rule file was edited after it was frozen.
    for path, frozen in ((PREREG, PREREG_SHA256), (AMENDMENT, AMENDMENT_SHA256))[: args.amendment + 1]:
        sha = hashlib.sha256(path.read_bytes()).hexdigest()                  # fingerprint of the file today
        if sha != frozen:                                                     # differs from the frozen one
            raise SystemExit(f"{path} changed after it was frozen (sha256 {sha}); the rule no longer holds")
    out = args.experiment / "analysis" / "phase_a" / ("amended" if args.amendment else "registered")  # both runs kept
    out.mkdir(parents=True, exist_ok=True)
    gate = json.loads((args.experiment / "analysis" / "results.json").read_text())   # the gate's results
    fold_t = {int(k): v for k, v in gate["cv_thresholds_per_fold"].items()}          # per-fold cutoffs

    df, summaries = load_scans(args.experiment, args.data, METHOD[args.amendment])  # per-B-scan table + boundaries
    vols = volumes(df, summaries)                                                    # regrouped into 60 visits
    df = df.set_index("scan_id")                                                     # look rows up by scan id

    # A2b, cross-validated: limits from the other four folds' graders, applied to this fold's model masks
    a2b = {}
    for k in sorted(df.fold.unique()):                                             # each of the 5 folds
        table = learn_limits(vols, [p for p in vols if vols[p]["fold"] != k])       # never this fold's graders
        for p in (p for p in vols if vols[p]["fold"] == k):                        # this fold's visits
            v = vols[p]
            # limits for this volume size (B-scan spacing), or the pooled limits if the size is not in the
            # table. frac[i] = share of columns that jump too far between B-scans i and i+1; flags = B-scans
            # that jump away from BOTH neighbours (the rule in oct_checks/continuity.py)
            frac, flags = continuity.flag_bscans(z_volume(v["model"]), table.get(v["n"], table["pooled"]), AXIAL_UM)
            for i, sid in enumerate(v["ids"]):
                if sid is not None:                                                # skip positions with no B-scan
                    a2b[sid] = (bool(flags[i]), frac[i - 1] if i > 0 else np.nan, frac[i] if i < len(frac) else np.nan)
    df["a2b_flag"] = [a2b[s][0] for s in df.index]                                  # continuity flag
    df["a2b_frac_prev"] = [a2b[s][1] for s in df.index]                             # jump share to the previous B-scan
    df["a2b_frac_next"] = [a2b[s][2] for s in df.index]                             # ... and to the next one
    deploy = learn_limits(vols, list(vols))                        # limits from all graders: the gear's input
    (out / "continuity_limits.json").write_text(json.dumps(
        {"source": "Garvin 2009 mean +- 2.6 SD, learned on all 60 OCT5k patients' grader tracings",
         "axial_um": AXIAL_UM, "z_sd": continuity.Z,
         "by_n_bscans": {str(n): l.to_dict() for n, l in deploy.items() if n != "pooled"},
         "pooled": deploy["pooled"].to_dict()}, indent=2), newline="\n")

    # A4 primary: sequential, each check against the same reviews spent on confidence alone
    df = df.reset_index()                                          # scan_id back to a column
    df["a1_flag"] = df.a1_violation_columns > 0                    # zero tolerance: one column flags it
    df["review_g0"] = ~df.passed                                   # the gate's own flags
    s1 = matched_step(df.review_g0, df.a1_flag, df.error, df.retina_ok, df.confidence)   # step 1: layer order
    df["review_s1"] = s1.pop("review") if s1["adopt"] else df.review_g0            # keep the check only if it earned it
    s1.pop("review", None)                                                         # drop the array before saving
    s1["diff_ci_errors_passed"] = step_bootstrap(df, "review_g0", "a1_flag")       # 95% CI by patient bootstrap
    s2 = matched_step(df.review_s1, df.a2b_flag, df.error, df.retina_ok, df.confidence)  # step 2: continuity
    df["review_final"] = s2.pop("review") if s2["adopt"] else df.review_s1          # same rule for step 2
    s2.pop("review", None)
    s2["diff_ci_errors_passed"] = step_bootstrap(df, "review_s1", "a2b_flag")
    adopted = [name for name, s in (("layer order (A1)", s1), ("continuity (A2b)", s2)) if s["adopt"]]

    def reasons(r) -> list:
        """Why a B-scan is flagged: every check that fired (only adopted checks count)."""
        out = []
        if r.confidence < fold_t[r.fold]:                  # below this fold's confidence cutoff
            out.append("confidence")
        if not r.retina_ok:                                # retina in < 90% of columns
            out.append("no retina")
        if s1["adopt"] and r.a1_flag:                      # only if layer order was adopted (it was not)
            out.append("layer order")
        if s2["adopt"] and r.a2b_flag:                     # only if continuity was adopted (it was not)
            out.append("continuity")
        return out
    df["reasons"] = [reasons(r) for r in df.itertuples()]
    # every flagged B-scan must have a reason and every reason must mean a flag: no silent reviews
    assert (df.reasons.str.len() > 0).equals(df.review_final), "reasons must explain every review"

    # A2 + A3 per visit
    rows, wl = [], []                                        # per-visit results; worklist entries
    by_id = df.set_index("scan_id")
    for p, v in vols.items():                                # each of the 60 visits
        ends = {rt: visit_endpoints(v, rt, args.amendment) for rt in RATERS}      # model and each grader
        m = ends["model"]                                                         # the model's endpoints
        # B-scan number -> its reasons, for this visit's flagged B-scans only
        flagged = {i + 1: by_id.reasons[sid] for i, sid in enumerate(v["ids"]) if sid and by_id.reasons[sid]}
        # what the tier rule (oct_checks/worklist.py) needs about this visit
        wl.append({"visit": p, "cohort": v["cohort"], "n_bscans": v["n"], "flagged": flagged,
                   "central_bscans": m["central_bscans"], "decentred": m["decentred"],
                   "fovea_offset_um": m["fovea_offset_um"], "cst_um": m["cst_um"], "ez_loss_mm2": m["ez_loss_mm2"],
                   "missing": [i + 1 for i, sid in enumerate(v["ids"]) if sid is None]})
        row = {"visit": p, "cohort": v["cohort"], "fold": v["fold"], "n_bscans": v["n"],   # one row of visits.csv
               "decentration_assessable": m["decentration_assessable"],
               "any_bscan_error": bool(by_id.loc[[s for s in v["ids"] if s], "error"].any())}
        for rt, e in ends.items():                                                # model, g1, g2, g3 side by side
            row.update({f"cst_{rt}": e["cst_um"], f"ez_loss_{rt}": e["ez_loss_mm2"], f"decentred_{rt}": e["decentred"],
                        f"fovea_x_{rt}": e["fovea_x_um"], f"fovea_y_{rt}": e["fovea_y_um"]})
        rows.append(row)
    ranked = worklist.rank(wl)                                                     # tiers + reading order
    # add each visit's tier and rank in the reading order
    visits = pd.DataFrame(rows).merge(pd.DataFrame([{"visit": v["visit"], "tier": v["tier"], "rank": i}
                                                    for i, v in enumerate(ranked, 1)]), on="visit")
    # a visit's CST is wrong if the model is > 20.1 µm (repeatability) from the graders' mean; None if unmeasurable
    gap = (visits.cst_model.astype(float) - visits[["cst_g1", "cst_g2", "cst_g3"]].astype(float).mean(axis=1)).abs()
    visits["cst_error"] = pd.Series(np.where(gap.isna(), None, gap > CST_REPEATABILITY_UM), dtype=object)  # > 20.1 µm

    # distance between the model's fovea and each grader's fovea, in µm
    fov = {g: np.hypot(visits.fovea_x_model.astype(float) - visits[f"fovea_x_{g}"].astype(float),
                       visits.fovea_y_model.astype(float) - visits[f"fovea_y_{g}"].astype(float)) for g in ("g1", "g2", "g3")}
    # per tier: how many visits, and how many of them have the given problem (e.g. a CST error)
    tier_tab = lambda col: {str(t): {"visits": int((visits.tier == t).sum()),
                                     col: int(visits.loc[visits.tier == t, col].fillna(False).astype(bool).sum())}
                            for t in (1, 2, 3)}
    rel = lambda p: p.relative_to(ROOT).as_posix()                 # repo-relative, so results don't depend on the clone
    res = {
        "prereg": {"file": rel(PREREG), "sha256": PREREG_SHA256,
                   "amendment": {"file": rel(AMENDMENT), "sha256": AMENDMENT_SHA256, "post_hoc": True}
                   if args.amendment else None},
        "n_scans": len(df), "n_visits": len(visits),
        "a4_primary": {"g0": {"reviewed": float(df.review_g0.mean()),
                              "errors_passed": int((df.error & ~df.review_g0).sum()), "n_errors": int(df.error.sum())},
                       "step1_layer_order": s1, "step2_continuity": s2, "adopted": adopted,
                       "final": {"reviewed": float(df.review_final.mean()),
                                 "errors_passed": int((df.error & ~df.review_final).sum())}},
        "a1": {"bscans_flagged": int(df.a1_flag.sum()), "by_cohort": df.groupby("cohort").a1_flag.sum().astype(int).to_dict(),
               "flagged_that_are_errors": int((df.a1_flag & df.error).sum()),
               "flagged_already_reviewed_by_g0": int((df.a1_flag & df.review_g0).sum())},
        "a2b": {"bscans_flagged": int(df.a2b_flag.sum()), "flagged_that_are_errors": int((df.a2b_flag & df.error).sum()),
                "flagged_already_reviewed_by_g0": int((df.a2b_flag & df.review_g0).sum()),
                "deploy_limits_by_n_bscans": {str(n): {"pair_limit": l.pair_limit, "n_pairs": l.n_pairs}
                                              for n, l in deploy.items() if n != "pooled"}},
        "a2_endpoints": {
            "cst_um": agreement(visits[["cst_g1", "cst_g2", "cst_g3", "cst_model"]].astype(float).set_axis(
                ["g1", "g2", "g3", "model"], axis=1)),
            "ez_loss_mm2": agreement(visits[["ez_loss_g1", "ez_loss_g2", "ez_loss_g3", "ez_loss_model"]].astype(float)
                                     .set_axis(["g1", "g2", "g3", "model"], axis=1)),
            "fovea_distance_model_vs_grader_um": {g: {"median": float(d.median()), "max": float(d.max()),
                                                      "over_200": int((d > 200).sum())} for g, d in fov.items()},
            "decentred_visits": {rt: int(visits[f"decentred_{rt}"].sum()) for rt in RATERS},
            "decentration_assessable_visits": int(visits.decentration_assessable.sum()),
            "cst_missing_visits": {rt: visits.visit[visits[f"cst_{rt}"].isna()].tolist() for rt in RATERS},
        },
        "a4_secondary_visits": {
            "cst_error_visits": visits.visit[visits.cst_error.fillna(False).astype(bool)].tolist(),
            "by_tier_cst_error": tier_tab("cst_error"), "by_tier_any_bscan_error": tier_tab("any_bscan_error"),
        },
    }
    # "\n" line endings on every OS, so a rerun on Windows is byte-identical to one in the container.
    (out / "results.json").write_text(json.dumps(res, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                                      newline="\n")
    df.drop(columns=["reasons"]).assign(reasons=df.reasons.str.join("+")).to_csv(out / "bscans.csv", index=False,
                                                                              lineterminator="\n")
    visits.to_csv(out / "visits.csv", index=False, lineterminator="\n")
    # provenance printed on the worklist page: which model, cutoff, checks and frozen rules produced it
    prov = {"model": df.model.iloc[0], "confidence cutoff": gate["deployment_threshold_all_folds"],
            "checks": ", ".join(["confidence", "retina"] + [a.split(" (")[0] for a in adopted]),
            "rules": f"analysis/prereg/phase_a.md sha256 {PREREG_SHA256[:12]}…"
                     + (f" + Amendment 1 (post hoc) sha256 {AMENDMENT_SHA256[:12]}…" if args.amendment else ""),
            "geometry": "OCT5k paper; ±10% µm uncertainty",
            "decentration": "assessed only where B-scans are ≤ 200 µm apart" if args.amendment else "always assessed"}
    worklist.write_csv(ranked, out / "worklist.csv")          # the reading order as a table
    worklist.write_html(ranked, out / "worklist.html", prov)  # ... and as a page
    figure_cst(visits, out / "cst_bland_altman.png")          # the CST agreement figure
    print(json.dumps({"a4_primary": res["a4_primary"], "a1": res["a1"], "a2b": {k: v for k, v in res["a2b"].items()
                                                                                 if k != "deploy_limits_by_n_bscans"},
                      "cst": res["a2_endpoints"]["cst_um"], "ez": res["a2_endpoints"]["ez_loss_mm2"],
                      "decentred": res["a2_endpoints"]["decentred_visits"],
                      "secondary": res["a4_secondary_visits"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
