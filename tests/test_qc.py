"""The QC building blocks the triage gear and Phase A share: boundaries from a mask, the layer-order
count, the neighbour (continuity) check, the en face endpoints (fovea, CST, EZ loss, decentration)
and the three reading tiers."""
from pathlib import Path  # file paths

import numpy as np       # arrays
import pandas as pd      # tables
import pytest            # the test framework

from oct_checks import boundaries as qb                    # boundaries and layer order
from oct_checks import continuity, enface, worklist        # neighbour check; endpoints; tiers
from oct_checks.geometry import LATERAL_UM, bscan_spacing_um, en_face_coords  # scan geometry

DATA = Path(__file__).resolve().parents[1] / "data"  # OCT5k, for the one test that uses real masks


def paint(rows: np.ndarray, height: int = 64) -> np.ndarray:
    """OCT5k's painting rule: each pixel takes the deepest layer whose boundary is at or above it."""
    r = np.arange(height)[:, None, None]                          # row numbers
    # for each pixel: the largest label k whose boundary row is at or above it (0 if none)
    return (np.arange(1, 6)[None, :, None] * (rows[None] <= r)).max(1).astype(np.uint8)


def flat_rows(width=16, top=10, step=5):
    """5 flat boundaries, 5 rows apart, across `width` columns: an idealised B-scan."""
    return np.array([[top + k * step] * width for k in range(5)], float)


@pytest.mark.parametrize("method", ["count", "first_row"])
def test_boundaries_recover_painted_rows_and_crossings(method):
    rows = flat_rows()
    assert np.array_equal(qb.boundaries(paint(rows), method), rows)   # paint, read back: same rows
    crossed = rows.copy()
    crossed[3, :4] = rows[2, :4] - 2  # IBRPE drawn above IS-OS: the EZ band vanishes there
    z = qb.boundaries(paint(crossed), method)
    assert np.array_equal(z[2, :4], crossed[3, :4]) and np.array_equal(z[2, 4:], rows[2, 4:])


def test_a_stray_pixel_moves_a_count_boundary_by_one_pixel_not_to_its_row():
    rows = flat_rows()
    mask = paint(rows)
    mask[0, 5] = 5  # a wrapped-around choroid pixel in the top row (the roll artifact)
    assert qb.boundaries(mask, "first_row")[3, 5] == 0  # RPE "at row 0": thickness corrupted
    z = qb.boundaries(mask, "count")
    assert np.array_equal(z[:, 5], rows[:, 5] - 1)  # every boundary one pixel up
    assert z[3, 5] - z[0, 5] == rows[3, 5] - rows[0, 5]  # thickness unchanged


def test_boundaries_are_nan_where_a_layer_is_absent():
    z = qb.boundaries(np.zeros((8, 3), np.uint8))      # all vitreous: no layer anywhere
    assert np.isnan(z).all()                           # unknown, not zero


@pytest.mark.skipif(not (DATA / "manifest.csv").exists(), reason="OCT5k not present")
def test_boundaries_match_the_graders_boundary_files_exactly():
    m = pd.read_csv(DATA / "manifest.csv")
    from PIL import Image
    for r in m.sample(5, random_state=0).itertuples():   # 5 random real B-scans
        for g in (1, 2, 3):
            mask = np.array(Image.open(DATA / getattr(r, f"mask_g{g}")))
            traced = pd.read_csv(DATA / getattr(r, f"boundary_g{g}"))[list(qb.NAMES)].to_numpy().T
            # where a grader drew IBRPE above IS-OS, the mask keeps the deeper layer: compare the rest
            ordered = np.all(np.diff(traced, axis=0) >= 0, axis=0)
            for method in ("count", "first_row"):  # Amendment 1 changes nothing on human masks
                assert np.array_equal(qb.boundaries(mask, method)[:, ordered], traced[:, ordered])
            assert qb.order_violation_columns(mask) == 0      # human masks never break the order


def test_order_violation_counts_columns():
    mask = paint(flat_rows())
    assert qb.order_violation_columns(mask) == 0       # clean: no violations
    mask[40, 3] = 1  # a speck of inner retina below the RPE
    mask[45:47, 7] = 0  # vitreous inside the retina
    assert qb.order_violation_columns(mask) == 2


def volume(n=9, width=32, shift_at=None, shift=12, noise=0.5, seed=0):
    """Boundary rows (n B-scans, 5, width) with a little noise; one B-scan optionally moved down."""
    rng = np.random.default_rng(seed)
    z = np.repeat(flat_rows(width)[None], n, axis=0) + rng.normal(0, noise, (n, 5, width))
    if shift_at is not None:
        z[shift_at] += shift                           # this B-scan jumps 12 rows away from its neighbours
    return z


def test_continuity_flags_the_one_displaced_bscan_not_its_neighbours():
    limits = continuity.learn([volume(seed=s) for s in range(5)], axial_um=3.5)  # "graders": 5 smooth volumes
    _, flags = continuity.flag_bscans(volume(shift_at=4, seed=9), limits, axial_um=3.5)
    assert flags.tolist() == [i == 4 for i in range(9)]   # only the jumping B-scan, not its neighbours
    _, flags = continuity.flag_bscans(volume(shift_at=8, seed=9), limits, axial_um=3.5)
    assert flags.tolist() == [i == 8 for i in range(9)]  # the edge B-scan has one pair
    _, flags = continuity.flag_bscans(volume(seed=9), limits, axial_um=3.5)
    assert not flags.any()                                 # a smooth volume: nothing flagged


def test_continuity_skips_missing_bscans():
    limits = continuity.learn([volume(seed=s) for s in range(5)], axial_um=3.5)
    z = volume(shift_at=3, seed=9)
    z[4] = np.nan  # b005 missing: b004 is judged on its one remaining pair
    frac, flags = continuity.flag_bscans(z, limits, axial_um=3.5)
    assert np.isnan(frac[3]) and np.isnan(frac[4])
    assert flags[3] and not flags[2] and not flags[4]


def test_enface_finds_the_thinnest_point_and_reads_cst():
    n, width, spacing = 19, 64, bscan_spacing_um(19)
    y, x = en_face_coords(n, width, spacing)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    fy, fx = y[10], x[40]  # fovea one B-scan and 8 columns off centre
    thick = 250 + 1e-4 * ((yy - fy) ** 2 + (xx - fx) ** 2)   # a bowl: thinnest at the fovea
    out = enface.endpoints(thick, np.zeros_like(thick, bool), spacing, LATERAL_UM)
    assert out["fovea_bscan"] == 11 and out["fovea_x_um"] == pytest.approx(fx)
    assert out["fovea_offset_um"] > 200 and not out["decentration_assessable"] and not out["decentred"]
    registered = enface.endpoints(thick, np.zeros_like(thick, bool), spacing, LATERAL_UM, decentration_needs_sampling=False)
    assert registered["decentred"]  # the registered rule assessed it even at 411 µm spacing
    central = np.hypot(yy - fy, xx - fx) <= 500
    assert out["cst_um"] == pytest.approx(thick[central].mean())  # CST = mean inside the 1-mm circle
    assert out["central_bscans"] == [10, 11, 12]                  # B-scans crossing that circle


def test_decentration_is_assessed_where_bscans_resolve_200_um():
    n, width, spacing = 61, 64, bscan_spacing_um(61)  # 123 µm apart
    y, x = en_face_coords(n, width, spacing)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    thick = 250 + 1e-4 * ((yy - y[32]) ** 2 + xx ** 2)  # fovea 2 B-scans = 247 µm off centre
    out = enface.endpoints(thick, np.zeros_like(thick, bool), spacing, LATERAL_UM)
    assert out["decentration_assessable"] and out["decentred"]


def test_enface_ez_loss_area_and_missing_retina():
    n, width, spacing = 19, 64, bscan_spacing_um(19)
    thick = np.full((n, width), np.nan)                # no usable retina at all
    loss = np.zeros((n, width), bool)
    loss[3, :10] = True                                # EZ lost at 10 points
    out = enface.endpoints(thick, loss, spacing, LATERAL_UM)
    assert out["ez_loss_mm2"] == pytest.approx(10 * LATERAL_UM * spacing / 1e6)
    assert not out["fovea_found"] and out["decentred"] and out["cst_um"] is None


def test_maps_use_ilm_to_top_of_rpe_and_skip_unusable_bscans():
    s = qb.column_summary(paint(flat_rows(width=4)))
    thick, loss = enface.maps([s, None, s], np.array([True, True, False]), axial_um=3.5)
    assert np.allclose(thick[0], 15 * 3.5) and np.isnan(thick[1:]).all() and not loss.any()


def test_worklist_tiers_and_order(tmp_path):
    base = {"n_bscans": 5, "decentred": False, "central_bscans": [3], "cst_um": 280.0, "ez_loss_mm2": 0.0,
            "fovea_offset_um": 50.0, "cohort": "AMD"}
    visits = [{**base, "visit": "A", "flagged": {}},
              {**base, "visit": "B", "flagged": {1: ["confidence"]}},
              {**base, "visit": "C", "flagged": {3: ["layer order"]}},
              {**base, "visit": "D", "flagged": {}, "decentred": True},
              {**base, "visit": "E", "flagged": {1: ["confidence"], 5: ["continuity"]}}]
    ranked = worklist.rank(visits)
    # C: a flag on the central B-scan -> 1; D: decentred -> 1; E, B: flags elsewhere -> 2 (more flags first); A -> 3
    assert [(v["visit"], v["tier"]) for v in ranked] == [("C", 1), ("D", 1), ("E", 2), ("B", 2), ("A", 3)]
    worklist.write_html(ranked, tmp_path / "w.html", {"model": "x"})
    worklist.write_csv(ranked, tmp_path / "w.csv")
    assert "Priority 1" in (tmp_path / "w.html").read_text(encoding="utf-8")
    assert pd.read_csv(tmp_path / "w.csv").visit.tolist() == ["C", "D", "E", "B", "A"]


def test_flag_summary_groups_reasons_into_ranges():
    assert worklist._ranges([1, 2, 3, 7, 9, 10]) == "b001–b003, b007, b009–b010"
    s = worklist._flag_summary({1: ["confidence"], 2: ["confidence"], 5: ["no retina", "confidence"]})
    assert s == "confidence: b001–b002, b005 (3)<br>no retina: b005 (1)"
