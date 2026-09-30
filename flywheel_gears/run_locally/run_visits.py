"""End to end on real OCT5k visits, through both gears exactly as Flywheel runs them (flyw gear run,
no account needed):

In plain words: The full demo. For each visit it builds the volume, runs the relay gear (images to
the GPU service and back), runs the triage gear (reading order and checklist), then checks that the
gears reproduce the analysis exactly. One visit: python -m flywheel_gears.run_locally.run_visits DME11; the grader's
checklist is then outputs/pipeline_demo/DME11/DME11_triage.txt.

  1. every visit: its B-scan images packed into one NRRD volume (data_prep/pack_volume.py)
  2. relay gear (one job per visit): every image -> GPU model service -> masks + confidences,
     returned as one label volume with the answers in its header
  3. triage gear (one job per visit): reading order, verdicts and reasons, CST, EZ-loss area,
     en face map, tier, signals, and the checklist a grader reads
  4. all visits -> worklist.html (the reading order a grader would open)

It then checks the run against the analysis: the served model is fold 0, so for fold-0 patients
the gears must reproduce the out-of-fold confidence and the analysis's visit endpoints and tiers.
Needs Docker, flyw, and the GPU service on the host (port 8765).

    python -m flywheel_gears.run_locally.run_visits DME11 DME30 Normal40 Normal49 --port 8765
"""
import argparse          # command-line options
import json              # results and findings files
import shutil            # copying files
import time              # timing each visit
from pathlib import Path  # file paths

import numpy as np       # arrays
import pandas as pd      # the manifest and the analysis tables
from PIL import Image    # reading the B-scan images

from flywheel_gears.run_locally.flyw import gear_run, qc_records   # run a gear job locally; read its QC records
from data_prep.pack_volume import pack                 # B-scan images -> one volume
from oct_checks import worklist                           # ranking visits into a reading order
from oct_checks.geometry import bscan_spacing_um          # distance between B-scans
from flywheel_gears.relay.relay import read_answers, read_volume, write_volume  # the relay's NRRD helpers

ROOT = Path(__file__).resolve().parents[2]        # the repo folder (this file is two folders down)
GEARS = ("relay", "triage")                       # the two gear folders, under flywheel_gears/


def run_visit(patient: str, man: pd.DataFrame, out: Path, port: int, limits: Path | None) -> dict:
    """Both gears on one visit; returns the triage gear's findings."""
    rows = man[man.patient == patient].sort_values("bscan")          # this visit's B-scans, in order
    vdir, stage = out / patient, out / "staging" / patient            # results folder; inputs folder
    for d in (vdir, stage):
        d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    images = {int(r.bscan): np.array(Image.open(ROOT / "data" / r.image).convert("L")) for r in rows.itertuples()}
    n = int(rows.bscan.max())                                         # B-scans in the acquired volume
    vol, header = pack(images, n, bscan_spacing_um(n), fill=0)        # step 1: the image volume
    vol_path = stage / f"{patient}.nrrd"
    write_volume(vol_path, vol, header)
    o = gear_run(ROOT / "flywheel_gears" / "relay", {"volume": vol_path},                # step 2: the relay, one job
                 {"service_url": f"http://host.docker.internal:{port}"})
    relayed = vdir / f"{patient}_relay.nrrd"
    shutil.copyfile(o / relayed.name, relayed)
    shutil.copyfile(o / ".metadata.json", vdir / "relay_metadata.json")
    inputs = {"relay_output": relayed}
    if limits:
        inputs["continuity_limits"] = limits                          # optional: enables the continuity signal
    o = gear_run(ROOT / "flywheel_gears" / "triage", inputs, {})                         # step 3: the triage, one job
    for f in o.iterdir():                                             # keep its outputs
        shutil.copyfile(f, vdir / (f.name if f.name != ".metadata.json" else "triage_metadata.json"))
    print(f"{patient}: {len(rows)} B-scans relayed + triaged in {time.time() - t0:.0f} s", flush=True)
    return worklist.load_findings([vdir / f"{patient}_triage.json"])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("patients", nargs="+")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "pipeline_demo")
    ap.add_argument("--experiment", type=Path, default=ROOT / "outputs" / "unet_w32_cv5_ez_border",
                    help="the experiment whose fold-0 model the service serves; the run is checked against it")
    args = ap.parse_args()
    args.out, args.experiment = args.out.resolve(), args.experiment.resolve()  # flyw runs in the gear's folder
    version = {g: json.loads((ROOT / "flywheel_gears" / g / "manifest.json").read_text())["version"] for g in GEARS}
    man = pd.read_csv(ROOT / "data" / "manifest.csv")                # every B-scan and its files
    limits = args.experiment / "analysis/phase_a/amended/continuity_limits.json"  # learned from graders
    limits = limits if limits.exists() else None
    saved = {g: (ROOT / "flywheel_gears" / g / "config.json").read_text() if (ROOT / "flywheel_gears" / g / "config.json").exists() else None
             for g in GEARS}                                          # local run state, restored afterwards
    try:
        found = [run_visit(p, man, args.out, args.port, limits) for p in args.patients]
    finally:  # the gears' local config.json files are run artifacts: put them back as they were
        for g, text in saved.items():
            if text is not None:
                (ROOT / "flywheel_gears" / g / "config.json").write_text(text)
            else:
                (ROOT / "flywheel_gears" / g / "config.json").unlink(missing_ok=True)
    ranked = worklist.rank(found)                                     # step 4: the reading order
    worklist.write_csv(ranked, args.out / "worklist.csv")
    worklist.write_html(ranked, args.out / "worklist.html",
                        {"pipeline": f"relay {version['relay']} -> triage {version['triage']}, one job each per "
                                     "visit (flyw gear run)",
                         "model": f"{args.experiment.name}-fold0 (these patients were held out of its training)"})

    # Consistency with the analysis (fold-0 model = the model that scored these patients out of fold).
    scans = pd.read_csv(args.experiment / "analysis/scans.csv").set_index("scan_id")
    visits = pd.read_csv(args.experiment / "analysis/phase_a/amended/visits.csv").set_index("visit")
    check = {}
    for v in found:
        _, header = read_volume(args.out / v["visit"] / f"{v['visit']}_relay.nrrd")
        answers = read_answers(header)                                         # the relay's per-B-scan answers
        ids = man[man.patient == v["visit"]].set_index("bscan").scan_id        # B-scan number -> scan id
        dconf = max(abs(a["confidence"] - scans.confidence[ids[b]]) for b, a in answers.items())  # worst gap
        qc, tags = qc_records(args.out / v["visit"], "triage_metadata.json")   # what Flywheel would record
        a = visits.loc[v["visit"]]                                              # the analysis's row
        check[v["visit"]] = {"fold": int(a.fold), "max_confidence_diff_vs_analysis": dconf,
                             "cst_gear": v["cst_um"], "cst_analysis": float(a.cst_model),
                             "ez_loss_gear": v["ez_loss_mm2"], "ez_loss_analysis": float(a.ez_loss_model),
                             "tier_gear": v["tier"], "tier_analysis": int(a.tier), "tags": tags,
                             "qc_states": {k: r["state"] for k, r in qc.items()},
                             "continuity_flagged": v["continuity"]["flagged_bscans"],
                             "layer_order_bscans": v["layer_order"]["bscans_with_violations"]}
    (args.out / "consistency.json").write_text(json.dumps(check, indent=2), newline="\n")
    print(json.dumps(check, indent=1))


if __name__ == "__main__":
    main()
