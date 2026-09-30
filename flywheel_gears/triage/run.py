"""Flywheel gear `triage`: decide which images of one visit a grader should read first, and why.

Input: the relay's output (<name>_relay.nrrd: the masks, with every image's confidence and
measurements in its header), and optionally the continuity limits learned from graders.
Written back to Flywheel:
  - a tag on the input: triage-priority1 / 2 / 3 (the visit's reading priority)
  - one native QC record per check, "na" where a check cannot run here:
      triage          pass (priority 3) / fail (priority 1 or 2), with the reading order and the endpoints
      confidence      fail if any image is below the cutoff                  (gate)
      retina          fail if any image shows retina in < min_retina_fraction (gate)
      layer_order     fail if any image breaks the layer order by >= 20.1 µm  (signal)
      continuity      pass / fail / na                                       (signal)
      image_quality   na: no validated gradability model                     (design only)
      longitudinal    na: no previous visit                                  (design only)
  - <name>_triage.txt: the checklist a grader reads, image by image, in reading order
  - <name>_triage.json: everything, machine-readable; <name>_enface_thickness.png: the map

It holds no model. If an input is unreadable the job fails and nothing is written or tagged.
"""
import json             # the findings file and the optional limits
import logging          # the job log
import sys              # exit code on failure
from pathlib import Path  # file paths

import nrrd                                            # reads the label volume
from flywheel_gear_toolkit import GearToolkitContext   # Flywheel's gear plumbing (MIT)

import triage  # the pure logic: triage.py (+ its modality module)

log = logging.getLogger("triage")
HEADER_KEY = "relay"  # where the relay stored the per-image answers (flywheel_gears/relay/relay.py)
AXIS_FIELDS = ("spacings", "units", "labels")  # per-axis NRRD fields (see relay.read_volume)


def main(ctx: GearToolkitContext) -> None:
    cfg = ctx.config                                                          # the job's settings
    src = ctx.get_input("relay_output")                                       # the input's Flywheel record
    labels, header = nrrd.read(ctx.get_input_path("relay_output"), index_order="C")  # (n, H, W) labels
    # pynrrd reverses `sizes` for C order but not the per-axis fields: reverse them back (as relay does)
    header = {k: (list(v)[::-1] if k in AXIS_FIELDS else v) for k, v in header.items()}
    answers = {int(k): v for k, v in json.loads(header[HEADER_KEY])["slices"].items()}  # per-image answers
    lim_path = ctx.get_input_path("continuity_limits")                        # optional input: None if absent
    limits = json.loads(Path(lim_path).read_text()) if lim_path else None
    name = Path(src["location"]["name"]).name.split(".")[0].removesuffix("_relay")  # e.g. DME11
    f, images = triage.run(labels, header, answers, cfg, limits, name)
    first = [b for b in f["reading_order"] if f["slices"][str(b)]["verdict"] == "read first"]
    log.info("%s: priority %d, %d of %d images read first, CST %s um", name, f["tier"], len(first),
             len(f["slices"]), f["cst_um"] and round(f["cst_um"]))

    (ctx.output_dir / f"{name}_triage.json").write_text(json.dumps(f, indent=2, ensure_ascii=False), encoding="utf-8")
    (ctx.output_dir / f"{name}_triage.txt").write_text(triage.report_text(f), encoding="utf-8")
    for fname, data in images.items():                                        # the en face map
        (ctx.output_dir / fname).write_bytes(data)

    def failing(check: str) -> list[int]:                                     # images where a check failed
        return [b for b in f["reading_order"]
                if any(x["check"] == check and x["state"] == "fail" for x in f["slices"][str(b)]["checks"])]

    ends = {k: f[k] for k in ("cst_um", "ez_loss_mm2", "decentred", "decentration_assessable", "fovea_offset_um")}
    ctx.metadata.add_qc_result(src, "triage", "pass" if f["tier"] == 3 else "fail", priority=f["tier"],
                               why=f["visit_checks"][0], read_first=first, reading_order=f["reading_order"], **ends)
    ctx.metadata.add_qc_result(src, "confidence", "fail" if failing("confidence") else "pass",
                               below_cutoff=failing("confidence"), cutoff=f["confidence_threshold"])
    ctx.metadata.add_qc_result(src, "retina", "fail" if failing("retina") else "pass", no_retina=failing("retina"))
    ctx.metadata.add_qc_result(src, "layer_order", "fail" if failing("layer_order") else "pass",
                               images=failing("layer_order"), note="signal only (breaks >= 20.1 µm): changes no verdict or priority")
    cont = f["continuity"]
    ctx.metadata.add_qc_result(src, "continuity", ("fail" if cont["flagged_bscans"] else "pass") if cont["assessed"]
                               else "na", **cont, note="signal only: changes no verdict or priority")
    ctx.metadata.add_qc_result(src, "image_quality", "na",
                               reason="no validated gradability model (needs WRC's quality grades to build and test)")
    ctx.metadata.add_qc_result(src, "longitudinal", "na", **f["longitudinal"])
    ctx.metadata.add_file_tags(src, f"{cfg['tag_prefix']}{f['tier']}")        # e.g. triage-priority1


if __name__ == "__main__":
    # clean_on_error: a failed job leaves no outputs behind
    with GearToolkitContext(clean_on_error=True) as context:
        context.init_logging()
        try:
            main(context)
        except Exception as e:  # any failure: one clear line in the job log, then a failed job
            log.error("triage not applied, no outputs or tags written: %s: %s", type(e).__name__, e)
            sys.exit(1)                                                  # mark the job failed
