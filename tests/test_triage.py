"""The triage gear's logic (flywheel_gears/triage/triage.py + flywheel_gears/triage/oct_layers.py): the confidence cutoff is
inclusive, the retina check catches stable-but-empty masks, tiers follow where the flags are,
signals and findings never change a verdict or the tier, every checklist line says where to look,
and the reading order puts hard failures, then the least confident, first. Answers from a model
other than the one the cutoff was chosen for are refused."""
import io      # reading the PNG map back

import numpy as np       # arrays
import pytest            # the test framework
from PIL import Image    # reading the PNG map back

from oct_checks.geometry import bscan_spacing_um  # µm between B-scans
from flywheel_gears.triage import oct_layers, triage     # the code under test

CFG ={"confidence_threshold": 0.9, "min_retina_fraction": 0.9}
MM = {"units": ["mm"] * 3}


def bscan(width=64, height=64, top=10, thick=20):
    """An ordered label map: vitreous, four bands, then below the RPE."""
    m = np.zeros((height, width), np.uint8)
    for k, start in enumerate((top, top + thick // 2, top + thick - 4, top + thick), 1):
        m[start:] = k                              # each band starts below the previous one
    m[top + thick + 3:] = 5                        # choroid below a 3-row RPE
    return m


def volume(n=19, width=64):
    """A 19-B-scan volume of identical B-scans, thinner in the middle one (the fovea)."""
    vol = np.stack([bscan(width) for _ in range(n)])
    vol[9, :, 30:34] = bscan(4, top=16, thick=12)  # a thin spot on the middle B-scan: the fovea
    return vol


def header(n=19):
    """The label volume's header as the relay writes it: spacings in mm."""
    return {"spacings": [bscan_spacing_um(n) / 1000, 0.0035, 0.01738], **MM}


def answers(n=19, low=(), no_retina=()):
    """The relay's per-B-scan answers: confidence 0.99 except `low` (0.5); retina 1.0 except `no_retina`."""
    return {b: {"model": "m", "confidence": 0.5 if b in low else 0.99,
                "measurements": {"retina_present_fraction": 0.1 if b in no_retina else 1.0}}
            for b in range(1, n + 1)}


def run(vol=None, ans=None, cfg=CFG, limits=None):
    vol = volume() if vol is None else vol
    return triage.run(vol, header(len(vol)), answers(len(vol)) if ans is None else ans, cfg, limits, "V")[0]


@pytest.mark.parametrize("raw, expected", [(0.9, 0.9), ("0.9", 0.9), ("1", 1.0), (0, 0.0)])
def test_fraction_config_accepts_numbers_and_numeric_strings(raw, expected):
    assert triage.fraction_from({"k": raw}, "k") == expected  # strings from flyw are converted


@pytest.mark.parametrize("raw", ["1.5", "-0.1", "nan", "high"])
def test_fraction_config_rejects_out_of_range_or_junk(raw):
    with pytest.raises(ValueError):                          # never triage with a nonsense setting
        triage.fraction_from({"k": raw}, "k")


def test_cutoff_is_inclusive_and_never_reads_x_below_x():
    assert triage.confidence_check(0.9, 0.9)["state"] == "pass"            # exactly at the cutoff passes
    c = triage.confidence_check(0.98874, 0.988926516228434)
    # the unrounded cutoff printed with 3 decimals once made "0.989 < threshold 0.989"
    assert c["state"] == "fail" and c["text"] == "✗ confidence 0.988740 < cutoff 0.988927"


def test_tiers_follow_where_the_flags_are():
    f = run()
    assert f["tier"] == 3 and f["fovea_bscan"] == 10 and f["cst_um"] is not None
    assert not f["decentration_assessable"]  # 411 µm between B-scans cannot resolve 200 µm
    f = run(ans=answers(low=(1,)))
    assert f["tier"] == 2 and f["flagged"] == {1: ["confidence"]}
    assert f["visit_checks"][0] == "Priority 2: flagged B-scan 1, none inside the central 1-mm circle"
    f = run(ans=answers(low=(9, 10)))
    assert f["tier"] == 1                     # the flagged B-scans cross the central subfield
    assert f["visit_checks"][0] == "Priority 1: flagged B-scans 9-10 cross the central 1-mm circle, where CST is measured"


def test_stable_but_no_retina_is_a_hard_failure_read_before_the_least_confident():
    # The failure found on the GPU: a random-weight U-Net finds no retina, stably, with confidence 1.0.
    ans = answers(low=(3,), no_retina=(7,))
    ans[7]["confidence"] = 1.0
    f = run(ans=ans)
    assert f["slices"]["7"]["verdict"] == "read first" and f["flagged"][7] == ["no retina"]
    assert f["reading_order"][:2] == [7, 3]            # hard failure first, then the least confident
    assert len(f["reading_order"]) == 19               # nothing is left out


def test_a_bscan_without_an_answer_is_flagged_and_missing_ones_are_listed():
    vol = volume()
    vol[0] = oct_layers.MISSING                        # B-scan 1 not acquired
    ans = answers()
    del ans[10]                                        # B-scan 10 never got a model result
    f = run(vol, ans)
    assert f["missing"] == [1] and f["flagged"] == {10: ["no result"]} and f["tier"] == 1
    assert "1" not in f["slices"] and f["reading_order"][0] == 10
    assert "not acquired: B-scan 1" in f["visit_checks"]


def test_signals_and_findings_never_change_a_verdict_or_the_tier():
    base = run()
    vol = volume()
    vol[4, 40:, 5] = 0                                 # vitreous under the RPE in one column of B-scan 5 (84 µm)
    vol[6, 45, 7] = 0                                  # one stray pixel in B-scan 7 (3.5 µm): does not count
    vol[2, :, 10:40] = np.where(vol[2, :, 10:40] == 3, 2, vol[2, :, 10:40])  # EZ missing in 30 columns of B-scan 3
    f = run(vol)
    assert f["tier"] == base["tier"] and not f["flagged"]                # still nothing to read first
    lo = [x for x in f["slices"]["5"]["checks"] if x["check"] == "layer_order"][0]
    assert lo["state"] == "fail" and lo["columns"] == 1                  # located, and a signal only
    assert lo["x_mm"][0] == lo["x_mm"][1] and "from the fovea along the B-scan" in lo["text"]
    assert lo["text"].startswith("✗ layer order broken by ≥ 20.1 µm in 1 columns")
    assert [x for x in f["slices"]["7"]["checks"] if x["check"] == "layer_order"][0]["state"] == "pass"
    ez = [x for x in f["slices"]["3"]["checks"] if x["check"] == "ez"][0]
    assert ez["role"] == "finding" and ez["text"].startswith("• EZ not seen in 47%")   # 30 of 64 columns


def test_continuity_flags_a_bscan_that_jumps_away_from_both_neighbours():
    vol = volume()
    vol[5] = bscan(64, top=40)                         # B-scan 6 sits 30 rows lower
    tight = {"mu": [0.0] * 5, "sd": [1.0] * 5, "pair_limit": 0.5, "n_pairs": 1}  # limits: changes near 0 µm
    f = run(vol, limits={"by_n_bscans": {}, "pooled": tight})
    assert f["continuity"]["assessed"] and f["continuity"]["flagged_bscans"] == [6]  # only the jumping one
    assert any(x["check"] == "continuity" and x["state"] == "fail" for x in f["slices"]["6"]["checks"])
    assert not f["flagged"]                            # a signal: nothing read first because of it
    assert run()["continuity"]["assessed"] is False    # no limits given -> not assessed, and says so


def test_what_cannot_be_assessed_says_so():
    lines = run()["visit_checks"]
    assert any(x.startswith("– image quality not assessed") for x in lines)
    assert any(x.startswith("– change since the last visit not assessed") for x in lines)
    assert any(x.startswith("– decentration not assessable: B-scans 411 µm apart") for x in lines)


def test_report_text_lists_every_bscan_in_reading_order():
    text = triage.report_text(run(ans=answers(low=(9, 10))))
    assert text.startswith("V · priority 1 · 2 of 19 images to read first")
    assert text.index("Image 9 (read first)") < text.index("Image 1 (routine)")
    assert "✗ confidence 0.500000 < cutoff 0.900000" in text


def test_module_rules():
    with pytest.raises(KeyError):
        run(cfg={**CFG, "modality": "fundus"})                          # no such module yet
    with pytest.raises(ValueError):                                     # OCT needs the retina number
        oct_layers.slice_gates({"measurements": {}}, CFG)
    with pytest.raises(ValueError):                                     # spacings must be in mm
        oct_layers.spacing_um({"spacings": [1, 1, 1], "units": ["um", "um", "um"]})


def test_render_enface_is_a_png():
    vol = volume()
    f, thick = run(), oct_layers.volume_maps(vol, 3.5, 0.9)[3]
    img = Image.open(io.BytesIO(oct_layers.render_enface(thick, f, px_per_mm=60)))
    # drawn to scale: 64 columns x 17.38 µm wide, 18 gaps x 411 µm tall
    assert img.format == "PNG" and img.size == (round(64 * 17.38 / 1000 * 60), round(18 * 7400 / 18 / 1000 * 60))


def test_answers_from_another_model_are_refused_not_judged():
    assert run(cfg={**CFG, "cutoff_model": "m"})["tier"] in (1, 2, 3)   # the model the cutoff was made for
    with pytest.raises(ValueError, match="derive a cutoff"):            # another model: its own cutoff first
        run(cfg={**CFG, "cutoff_model": "nnSAM"})
