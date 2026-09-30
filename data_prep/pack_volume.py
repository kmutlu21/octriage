"""Pack one visit's per-B-scan files into a single NRRD volume: the input the relay gear reads.

In plain words: OCT5k stores each B-scan as its own image, but a Flywheel visit is one volume. This
stacks a visit's B-scans in order into one NRRD file (a simple medical-image format: the pixels plus
a text header with the spacing in mm), which is what the relay gear reads.

Stand-in for WRC's pipeline, which delivers whole volumes (DICOM zips/E2E; the team's
ez-rpe-reportgear reads NRRD). Positions with no B-scan are listed in the header field
"missing slices" (1-based) and filled with `fill`: 0 in an image volume, 255 in a label volume
(the "not acquired" label the triage gear skips). The spacing goes in the header in mm (across
B-scans / axial / lateral); by default OCT5k's geometry (oct_checks/geometry.py).

    python -m data_prep.pack_volume --n-bscans 19 --out DME11.nrrd images/*Image*.TIFF      # images
    python -m data_prep.pack_volume --labels --n-bscans 19 --out DME11_masks.nrrd masks/*.png
"""
import argparse          # command-line options
import re                # finding the B-scan number in a file name
from pathlib import Path  # file paths

import numpy as np       # arrays
from PIL import Image    # reading the images and mask PNGs

from oct_checks.geometry import AXIAL_UM, SCAN_WIDTH_UM, bscan_spacing_um  # OCT5k geometry in µm
from flywheel_gears.relay.relay import MISSING, MISSING_KEY, write_volume         # the "not acquired" label, header field, writer

B_INDEX = re.compile(r"_b(\d+)")  # "..._b017_..." -> 17


def bscan_index(path: Path) -> int:
    """The 1-based B-scan number in a file name like DME11_b017_layers.png."""
    m = B_INDEX.search(Path(path).stem)       # look for _b<digits> in the name
    if not m:
        raise ValueError(f"no _b<index> in file name: {path}")
    return int(m.group(1))


def pack(slices: dict, n_bscans: int, spacing_um: float, axial_um: float = AXIAL_UM,
         lateral_um: float | None = None, fill: int = MISSING) -> tuple[np.ndarray, dict]:
    """slices: {1-based B-scan number: (H, W) uint8 image or labels} -> (volume (n, H, W), NRRD header).
    lateral_um defaults to the 8.9 mm scan width spread over the slices' own width."""
    shape = next(iter(slices.values())).shape                       # (H, W) of one slice
    lateral_um = lateral_um if lateral_um is not None else SCAN_WIDTH_UM / shape[1]
    vol = np.full((n_bscans, *shape), fill, np.uint8)               # every position "not acquired" to start
    for b, s in slices.items():
        if not 1 <= b <= n_bscans:                                  # a B-scan number outside the volume
            raise ValueError(f"B-scan {b} outside 1..{n_bscans}")
        vol[b - 1] = s                                              # put the slice at its position
    missing = [b for b in range(1, n_bscans + 1) if b not in slices]  # positions with no B-scan
    header = {"spacings": [spacing_um / 1000, axial_um / 1000, lateral_um / 1000],  # µm -> mm
              "units": ["mm", "mm", "mm"],
              "labels": ["bscan", "axial", "lateral"],             # what each axis is
              MISSING_KEY: " ".join(str(b) for b in missing)}      # 1-based, for the relay
    return vol, header


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--n-bscans", type=int, required=True, help="B-scans in the acquired volume (the highest index)")
    ap.add_argument("--labels", action="store_true", help="the files are label masks (missing = 255), not images")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    slices = {bscan_index(p): np.array(Image.open(p).convert("L")) for p in args.files}  # read each by number
    vol, header = pack(slices, args.n_bscans, bscan_spacing_um(args.n_bscans), fill=MISSING if args.labels else 0)
    write_volume(args.out, vol, header)                                       # save it
    print(f"{args.out}: {len(slices)} of {args.n_bscans} B-scans, spacing {header['spacings']} mm")


if __name__ == "__main__":
    main()
