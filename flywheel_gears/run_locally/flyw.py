"""Run one gear job exactly as Flywheel would, on this machine, without a Flywheel site or account.

In plain words: Flywheel's own command-line tool can run a gear's container on a laptop the way a
Flywheel site would. This file wraps it, so the demo runs both gears with no Flywheel account.

Uses Flywheel's own CLI (`flyw gear run`), which starts the gear's container with the same mounts
and command the Flywheel engine uses. Needs Docker and the flyw CLI.

Two local-run workarounds (found by running it; a Flywheel site fills these in itself):
- flyw writes no destination/hierarchy into config.json, and the toolkit needs them.
- flyw leaves every finished container behind, so they are removed here.
"""
import json         # the gear's config.json and manifest.json
import os           # the TEMP folder where flyw prepares jobs
import shutil       # finding the flyw executable; removing old job folders
import subprocess   # running flyw and docker
from pathlib import Path  # file paths

PREPARED = Path(os.environ.get("TEMP", "/tmp")) / "gear"   # where `flyw gear run --prepare` puts the job
LOCAL = {"type": "acquisition", "id": "local-acq"}          # a stand-in parent container for local runs


def flyw_exe() -> str:
    """The flyw CLI: on PATH, or where its Windows installer puts it (~/.fw/flyw.bat)."""
    exe = shutil.which("flyw") or shutil.which("flyw.bat")      # on PATH?
    fallback = Path.home() / ".fw" / "flyw.bat"                   # the installer's default location
    if exe:
        return exe
    if fallback.exists():
        return str(fallback)
    raise SystemExit("flyw not found: install the Flywheel CLI (docs.flywheel.io)")


def gear_run(gear_dir: Path, inputs: dict, config: dict) -> Path:
    """One gear job; returns its output folder (with .metadata.json = what Flywheel would apply).
    Config = the manifest defaults plus `config`."""
    flyw = flyw_exe()
    manifest = json.loads((gear_dir / "manifest.json").read_text())             # name, version, image, defaults
    args = [a for k, v in inputs.items() for a in ("--input", f"{k}={Path(v).resolve()}")]  # absolute paths
    if not (gear_dir / "config.json").exists():                                  # once per gear folder,
        subprocess.run([flyw, "gear", "config", "--new"], cwd=gear_dir, check=True, capture_output=True)
    subprocess.run([flyw, "gear", "config", *args], cwd=gear_dir, check=True, capture_output=True)  # set inputs
    cfg = json.loads((gear_dir / "config.json").read_text())                     # what flyw wrote
    cfg["config"] = {k: s["default"] for k, s in manifest["config"].items() if "default" in s} | config
    cfg["destination"] = LOCAL                                                   # workaround 1
    for spec in cfg["inputs"].values():
        spec["hierarchy"] = LOCAL                                                # workaround 1, per input
    (gear_dir / "config.json").write_text(json.dumps(cfg, indent=4))
    job = PREPARED / f"{manifest['name']}_{manifest['version']}"                 # e.g. %TEMP%/gear/relay_0.2.0
    shutil.rmtree(job, ignore_errors=True)                                       # start clean
    subprocess.run([flyw, "gear", "run", "--prepare"], cwd=gear_dir, check=True, capture_output=True)
    r = subprocess.run([flyw, "gear", "run", str(job)], cwd=gear_dir, capture_output=True, text=True)  # run it
    image = manifest["custom"]["gear-builder"]["image"]                          # the gear's Docker image
    left = subprocess.run(["docker", "ps", "-aq", "--filter", f"ancestor={image}", "--filter", "status=exited"],
                          capture_output=True, text=True).stdout.split()         # finished containers
    if left:
        subprocess.run(["docker", "rm", *left], capture_output=True)             # workaround 2
    if r.returncode != 0 or not (job / "output" / ".metadata.json").exists():    # the job failed
        raise RuntimeError(f"{manifest['name']} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return job / "output"


def qc_records(output: Path, name: str = ".metadata.json") -> tuple[dict, list]:
    """From a job's metadata file: the QC records ({check: {...}}) and tags of the file that carries
    them (the input). The toolkit also stores a "job_info" entry (gear version, inputs, config)."""
    meta = json.loads((output / name).read_text(encoding="utf-8"))              # what Flywheel would apply
    first = next(f for f in meta["acquisition"]["files"] if "qc" in f.get("info", {}))  # the file with QC
    gear_qc = next(iter(first["info"]["qc"].values()))                           # info.qc.<gear name>
    records = {k: v for k, v in gear_qc.items() if k != "job_info"}             # the checks only
    return records, first["tags"]                                                # {check: record}, [tags]
