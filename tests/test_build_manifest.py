"""Data prep on a tiny fake OCT5k tree: the join, the boundary-to-mask rendering, and
that any structural problem fails WITHOUT writing a manifest."""
import csv               # reading the manifest back

import numpy as np       # arrays
import pytest            # the test framework
from PIL import Image    # writing fake masks and images

from data_prep import build_manifest as bm  # the code under test

# Two scan paths in the archive's own layout: "<root>/<cohort> (<n>).E2E/<visit time>/Image <b-scan>"
KEYS = [
    "AMD Part1/AMD (1).E2E/2- 25- 2017 9- 10- 42 PM/Image 2",
    "DME/DME (7).E2E/3- 1- 2017 10- 5- 3 AM/Image 11",
]


def test_parse_key():
    assert bm.parse_key(KEYS[0]) == {
        "cohort": "AMD", "patient": "AMD01", "visit_time": "2017-02-25T21:10:42", "bscan": 2}
    assert bm.parse_key(KEYS[1])["visit_time"] == "2017-03-01T10:05:03"   # AM times too
    with pytest.raises(ValueError):
        bm.parse_key("AMD/AMD (1)/Image (2)")  # the Dataset_3x50_Final layout


def test_render_monotonic_column():
    col = np.array([[1, 3, 5, 6, 8]])       # one column: the 5 boundaries at rows 1, 3, 5, 6, 8
    assert bm.boundaries_to_mask(col, 10)[:, 0].tolist() == [0, 1, 1, 2, 2, 3, 4, 4, 5, 5]


def test_render_crossing_paints_deepest_layer():
    # IBRPE (row 4) drawn above IS-OS (row 6): the EZ band (label 3) vanishes and the
    # RPE band (label 4) starts at IBRPE. Counting boundaries would give 3 at rows 4-5.
    col = np.array([[1, 2, 6, 4, 8]])
    assert bm.boundaries_to_mask(col, 10)[:, 0].tolist() == [0, 1, 2, 2, 4, 4, 4, 4, 5, 5]


def boundaries(cross=False):
    """Flat boundaries across all 512 columns; optionally IBRPE crossing IS-OS in 10 columns."""
    b = np.tile([100, 150, 200, 210, 220], (512, 1))
    if cross:
        b[10:20, 3] = 195  # IBRPE above IS-OS in 10 columns
    return b


@pytest.fixture
def tree(tmp_path):
    """A fake OCT5k download + image archive: 2 scans x 3 graders, consistent everywhere."""
    oct5k = tmp_path / "OCT5k"
    for g in bm.GRADERS:
        for i, key in enumerate(KEYS):
            b = boundaries(cross=(g == 2 and i == 1))                       # grader 2 crosses on scan 2
            mask = oct5k / f"Masks/Masks_Manual/Grading_{g}/{key}.png"
            mask.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(bm.boundaries_to_mask(b, 512)).save(mask)       # the mask = its boundaries
            bnd = oct5k / f"Boundaries/Boundaries_Manual/Grading_{g}/{key}.csv"
            bnd.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(bnd, np.column_stack([np.arange(512), b]), fmt="%d", delimiter=",",
                       header="x," + ",".join(bm.LAYERS), comments="")
    for key in KEYS:                                                        # one source image per scan
        img = tmp_path / f"images/Rasti_old/{key}.TIFF"
        img.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.zeros((512, 496), np.uint8)).save(img)
    paths = oct5k / "Scripts/paths/manual_paths.csv"                        # the authors' own path list
    paths.parent.mkdir(parents=True)
    paths.write_text("\n".join(
        f"../Images/Images_Manual/{k}.png,./Macular-Dataset-R.Rasti_old/{k}.TIFF" for k in KEYS))
    return tmp_path


def run(tree):
    """Run the script on the fake tree; returns its exit code."""
    return bm.main(["--oct5k", str(tree / "OCT5k"), "--images", str(tree / "images/Rasti_old"),
                    "--out", str(tree / "manifest.csv")])


def test_builds_manifest(tree):
    assert run(tree) == 0                                           # success
    rows = list(csv.DictReader(open(tree / "manifest.csv")))
    assert [r["scan_id"] for r in rows] == ["AMD01_b002", "DME07_b011"]
    assert (rows[0]["image_w"], rows[0]["image_h"]) == ("496", "512")
    assert rows[0]["image"] == f"images/Rasti_old/{KEYS[0]}.TIFF"  # relative to the manifest
    assert rows[0]["qc"] == ""                                      # no crossing
    assert rows[1]["qc"] == "g2:IBRPE-above-IS-OS"                  # the crossing is named, row kept


def test_missing_image_fails_without_writing(tree):
    (tree / f"images/Rasti_old/{KEYS[1]}.TIFF").unlink()            # delete one source image
    assert run(tree) == 1                                           # failure
    assert not (tree / "manifest.csv").exists()                     # and no half-joined table


def test_mask_that_disagrees_with_its_boundaries_fails(tree):
    mask = tree / f"OCT5k/Masks/Masks_Manual/Grading_3/{KEYS[0]}.png"
    a = np.array(Image.open(mask))
    a[300, 300] = 1                                                 # change a single pixel
    Image.fromarray(a).save(mask)
    assert run(tree) == 1                                           # caught
