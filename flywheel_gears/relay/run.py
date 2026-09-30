"""Flywheel gear `relay`: send every image of one volume to the model service, write the masks back.

What happens in one job
  1. A gear rule fires (flywheel_gears/relay/rule.json: a volume tagged `ai-ready`). Flywheel starts this
     container and mounts the file under /flywheel/v0/input/volume/.
  2. Check the input, then ask the service's /health who is answering (relay.py).
  3. For each image: POST it to <service_url>/predict, where the model runs on a GPU.
     relay.py checks every answer and tries a dropped connection again.
  4. Write back ONE file, <name>_relay.nrrd: the masks as a label volume, with every image's
     confidence and measurements in its header. Its name starts the triage gear's rule.
  5. Tag the input volume `relay-done`, so the rule never sends it twice.

If anything goes wrong, the job fails with one clear log line, and nothing is written or tagged
(the toolkit empties the output folder of a failed job). A failed job can simply be run again.

This file is only the Flywheel plumbing; the logic is in relay.py. Try it without a Flywheel site:
flywheel_gears/run_locally/run_visits.py runs this gear with Flywheel's own `flyw gear run`.
"""
import logging            # the job log
import sys                # exit code on failure
from pathlib import Path  # file paths

from flywheel_gear_toolkit import GearToolkitContext   # Flywheel's gear plumbing (MIT)

import relay  # the logic in relay.py

log = logging.getLogger("relay")


def main(ctx: GearToolkitContext) -> None:
    cfg = ctx.config                                                   # the job's settings
    vol_file = ctx.get_input("volume")                                 # the input's Flywheel record
    volume, header = relay.read_volume(ctx.get_input_path("volume"))  # (n, H, W) images + header
    relay.check_volume(volume, header)                                 # before any image leaves
    _, ask = relay.service(cfg["service_url"])                         # /health first, then one ask per image
    labels, out_header, answers = relay.relay_volume(volume, header, ask)
    name = Path(vol_file["location"]["name"]).name.split(".")[0]       # e.g. DME11 from DME11.nrrd
    out_name = f"{name}_relay.nrrd"
    relay.write_volume(ctx.output_dir / out_name, labels, out_header)
    confidences = [a["confidence"] for a in answers.values()]
    model = next(iter(answers.values()))["model"]                      # one model per volume (relay.py)
    log.info("%s: %d images relayed, model %s, confidence %.4f-%.4f", name, len(answers), model,
             min(confidences), max(confidences))
    # File info on the output, so Flywheel can search it; the full answers stay in the NRRD header.
    ctx.metadata.update_file_metadata(
        out_name, container_type=ctx.destination["type"],
        info={"relay": {"source": vol_file["location"]["name"], "images": len(answers), "model": model,
                        "missing_slices": sorted(relay.missing_slices(header))}})
    ctx.metadata.add_file_tags(vol_file, cfg["done_tag"])             # never relayed twice


if __name__ == "__main__":
    # clean_on_error: if main() raises, the toolkit empties output/, so a failed job leaves nothing
    with GearToolkitContext(clean_on_error=True) as context:
        context.init_logging()
        try:
            main(context)
        except Exception as e:  # any failure: one clear line in the job log, then a failed job
            log.error("relay failed, no outputs or tags written: %s: %s", type(e).__name__, e)
            sys.exit(1)
