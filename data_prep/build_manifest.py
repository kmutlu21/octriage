"""Build the OCT5k manifest: one row per manually graded B-scan, joined to its
source image and to all three graders' masks and layer-boundary files.
Everything downstream (training, prediction, analysis) reads only this one table.

In plain words: The dataset arrives as thousands of separate files: images, three graders' masks,
boundary tracings. This script matches them into one table, one row per B-scan, and writes nothing
if a single file is missing or does not match.

The data (details in the README):
- OCT5k (Sci Data 2025, doi:10.1038/s41597-024-04259-z; CC0, doi:10.5522/04/22128671): 1,672
  B-scans from 60 patients (AMD, DME, Normal), each segmented by 3 graders into 5 boundaries.
- OCT5k ships masks but not images. The images come from a separate archive (Rasti et al.,
  IEEE TMI 2018, "Macular-Dataset-R.Rasti_old"); a near-identical archive with a different
  layout (Dataset_3x50_Final) does not join.

So every join and every file is checked, and on any structural failure the script exits 1
without writing a manifest: a half-joined table would silently train on the wrong images.
Crossing boundaries are a data finding, not a failure: those rows are kept and the crossing
layers are named in the `qc` column (180 gradings in 132 scans).

Mask labels (used everywhere): 0 vitreous (above ILM), 1 ILM-OPL, 2 OPL-IS/OS, 3 EZ band
(IS/OS-IBRPE), 4 RPE (IBRPE-OBRPE), 5 below OBRPE (choroid).

    python data_prep/build_manifest.py
    python data_prep/build_manifest.py --images data/images/Dataset_3x50_Final   # fails the join, on purpose
"""
import argparse                          # command-line options
import csv                               # reading/writing the tables
import os                                # os.path.relpath for relative paths
import re                                # parsing archive paths
import sys                               # the exit code
from collections import Counter, defaultdict  # counting; lists that create themselves
from datetime import datetime            # parsing the visit folder names
from pathlib import Path, PurePosixPath  # file paths (PurePosix: always "/" separators)

import numpy as np                       # arrays
from PIL import Image                    # reading images and masks

ROOT = Path(__file__).resolve().parents[1]  # the repo folder
GRADERS = (1, 2, 3)                          # OCT5k's three graders
# Mask label k (1-5) starts at boundary LAYERS[k-1]; label 0 is the vitreous above the ILM.
LAYERS = ("ILM", "OPL", "IS-OS", "IBRPE", "OBRPE")
MASK_SHAPE = (512, 512)                      # every OCT5k mask is 512 x 512
# A scan's path inside the archives, e.g. "<root>/AMD (1).E2E/<visit folder>/Image 12".
SCAN_RE = re.compile(
    r"[^/]+/(?P<cohort>AMD|DME|Normal) \((?P<num>\d+)\)\.E2E/(?P<visit>[^/]+)/Image (?P<bscan>\d+)"
)
VISIT_FMT = "%m- %d- %Y %I- %M- %S %p"  # folder names like "2- 25- 2017 9- 10- 42 PM"
# The manifest's columns, in order.
FIELDS = ["scan_id", "cohort", "patient", "visit_time", "bscan", "image", "image_w", "image_h",
          *(f"mask_g{g}" for g in GRADERS), *(f"boundary_g{g}" for g in GRADERS), "qc"]


def index(root: Path, suffixes: set[str]) -> dict[str, list[Path]]:
    """Map extension-less relative path -> matching files (more than one = ambiguous).
    Joining on the path without its extension matches a mask 'x.png' to its image 'x.tif'."""
    out = defaultdict(list)                        # key -> list of files (a new key starts an empty list)
    for p in root.rglob("*"):                      # every file under root, at any depth
        if p.suffix.lower() in suffixes:           # only the wanted file types
            out[p.relative_to(root).with_suffix("").as_posix()].append(p)  # key = path without extension
    return out


def parse_key(key: str) -> dict:
    """Path key -> cohort, patient id (e.g. AMD01), visit time, B-scan number."""
    m = SCAN_RE.fullmatch(key)                     # the whole key must match the pattern
    if not m:
        raise ValueError(f"unexpected scan path: {key}")
    return {
        "cohort": m["cohort"],                                          # AMD / DME / Normal
        "patient": f"{m['cohort']}{int(m['num']):02d}",                  # "AMD (1)" -> AMD01
        "visit_time": datetime.strptime(m["visit"], VISIT_FMT).isoformat(),  # folder name -> ISO time
        "bscan": int(m["bscan"]),                                       # "Image 12" -> 12
    }


def official_pairs(paths_csv: Path) -> dict[str, str]:
    """The dataset authors' own list: mask key -> source image key in their archive.
    Our join must agree with theirs for every scan (checked in main)."""
    pairs = {}
    with open(paths_csv, newline="") as f:
        for dst, src in csv.reader(f):             # each row: (mask path, source image path)
            key = PurePosixPath(dst.split("Images_Manual/", 1)[1]).with_suffix("").as_posix()  # mask key
            pairs[key] = PurePosixPath(re.sub(r"^\./[^/]+/+", "", src)).with_suffix("").as_posix()  # image key
    return pairs


def boundaries_to_mask(rows: np.ndarray, height: int) -> np.ndarray:
    """Render per-column boundary rows (width x 5) as a label mask (height x width).

    Layers are painted top to bottom, each overwriting the last, so a pixel takes the
    deepest layer whose boundary is at or above it. That is how OCT5k drew its masks
    where a grader's boundaries cross; counting boundaries instead disagrees there.
    """
    above = rows[None, :, :] <= np.arange(height)[:, None, None]      # (H, W, 5): boundary at/above pixel?
    deepest = rows.shape[1] - np.argmax(above[:, :, ::-1], axis=2)    # index of the deepest such boundary, 1-5
    return np.where(above.any(axis=2), deepest, 0).astype(np.uint8)   # no boundary above = vitreous (0)


def check_grading(mask_path: Path, boundary_path: Path) -> tuple[list[str], list[str]]:
    """Return (structural errors, data findings) for one grader's mask + boundaries."""
    with Image.open(mask_path) as im:
        mask = np.array(im)                        # the grader's label mask
    if mask.shape != MASK_SHAPE or mask.dtype != np.uint8 or mask.max() > len(LAYERS):  # wrong size or labels
        return [f"bad mask {mask.shape} {mask.dtype} max={mask.max()}: {mask_path}"], []
    with open(boundary_path) as f:
        header = f.readline().strip()              # first line: "x,ILM,OPL,IS-OS,IBRPE,OBRPE"
        rows = np.loadtxt(f, delimiter=",", dtype=int, ndmin=2)  # then one line per image column
    if header != ",".join(("x", *LAYERS)) or rows.shape != (MASK_SHAPE[1], len(LAYERS) + 1):  # wrong format
        return [f"bad boundary file {rows.shape} '{header}': {boundary_path}"], []
    rows = rows[:, 1:]  # drop the x column; what remains is (width, 5) boundary rows
    # A deeper boundary drawn above a shallower one, e.g. IBRPE above IS-OS where the EZ collapses.
    crossed = (np.diff(rows, axis=1) < 0).any(axis=0)      # per boundary pair: crosses in any column?
    findings = [f"{LAYERS[j + 1]}-above-{LAYERS[j]}" for j in np.flatnonzero(crossed)]  # name the pairs
    # Guards the label map: every mask must be exactly its own boundaries, rendered.
    agree = (boundaries_to_mask(rows, MASK_SHAPE[0]) == mask).mean()  # share of pixels that match
    if agree < 1:                                          # any pixel differs: the files disagree
        return [f"mask differs from its boundaries on {1 - agree:.2%} of pixels: {mask_path}"], findings
    return [], findings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--oct5k", type=Path, default=ROOT / "data/OCT5k")               # the OCT5k download
    ap.add_argument("--images", type=Path, default=ROOT / "data/images/Rasti_old")   # the source images
    ap.add_argument("--out", type=Path, default=ROOT / "data/manifest.csv")          # where to write
    args = ap.parse_args(argv)

    # Index every file by its extension-less path, per grader.
    masks = {g: index(args.oct5k / f"Masks/Masks_Manual/Grading_{g}", {".png"}) for g in GRADERS}
    bounds = {g: index(args.oct5k / f"Boundaries/Boundaries_Manual/Grading_{g}", {".csv"}) for g in GRADERS}
    images = index(args.images, {".tif", ".tiff", ".png"})
    official = official_pairs(args.oct5k / "Scripts/paths/manual_paths.csv")
    keys = sorted(set().union(*masks.values()))  # every scan any grader segmented
    errors, rows = [], []

    # Structural checks: every grader has every file, every scan has exactly one image,
    # and our join agrees with the dataset authors' own path list.
    for g in GRADERS:
        for name, found in (("mask", masks[g]), ("boundary", bounds[g])):
            errors += [f"grader {g} has no {name} for {k}" for k in keys if k not in found]
    unmatched = [k for k in keys if not images.get(k)]                     # no image at all
    ambiguous = [k for k in keys if len(images.get(k, [])) > 1]            # more than one candidate image
    errors += [f"no source image for {k}" for k in unmatched]
    errors += [f"{len(images[k])} source images for {k}" for k in ambiguous]
    errors += [f"dataset's own path list disagrees for {k}" for k in keys if official.get(k) != k]

    for k in keys:                                                         # one manifest row per scan
        try:
            row = {"scan_id": None, **parse_key(k)}                        # scan_id first, filled below
        except ValueError as e:
            errors.append(str(e))
            continue
        row["scan_id"] = f"{row['patient']}_b{row['bscan']:03d}"  # e.g. DME11_b005
        if len(images.get(k, [])) == 1:                                    # exactly one image: record it
            row["image"] = images[k][0]
            with Image.open(images[k][0]) as im:
                row["image_w"], row["image_h"] = im.size                  # e.g. 496 x 512
        qc = []                                                            # data findings for this scan
        for g in GRADERS:
            if k in masks[g] and k in bounds[g]:
                row[f"mask_g{g}"], row[f"boundary_g{g}"] = masks[g][k][0], bounds[g][k][0]
                errs, findings = check_grading(masks[g][k][0], bounds[g][k][0])
                errors += errs
                qc += [f"g{g}:{x}" for x in findings]                      # e.g. "g2:IBRPE-above-IS-OS"
        row["qc"] = ";".join(qc)
        rows.append(row)

    # A summary a person can check at a glance (the "look at your data" step).
    n = len(keys)
    patients = {r["patient"]: r["cohort"] for r in rows}                   # patient -> cohort
    print(f"scans: {n} | patients: {len(patients)} {dict(sorted(Counter(patients.values()).items()))}")
    print(f"image join vs {args.images.name}: {n - len(unmatched)}/{n} ({(n - len(unmatched)) / max(n, 1):.1%}), "
          f"{len(ambiguous)} ambiguous")
    print(f"dataset's own path list agrees: {sum(official.get(k) == k for k in keys)}/{n}")
    # (scan, grader, crossing) for every crossing boundary found
    findings = [(r["scan_id"], *x.split(":")) for r in rows for x in filter(None, r["qc"].split(";"))]
    print(f"crossing boundaries: {len({f[:2] for f in findings})}/{n * len(GRADERS)} gradings, "
          f"{len({f[0] for f in findings})} scans: {dict(Counter(f[2] for f in findings)) or 'none'}")

    if errors:                                                             # refuse to write a broken table
        print(f"\nFAILED with {len(errors)} errors, manifest NOT written. First 10:", *errors[:10], sep="\n  ")
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["patient"], r["bscan"])):  # by patient, then B-scan
            # Paths are stored relative to the manifest, so the data folder can move as a unit.
            w.writerow({k: Path(os.path.relpath(v, args.out.parent)).as_posix() if isinstance(v, Path) else v
                        for k, v in r.items()})
    print(f"wrote {len(rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
