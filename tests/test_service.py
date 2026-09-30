"""The service's answer passes the relay's own validation (the contract between the two), the
triage gear can judge it, non-images are refused, EZ loss counts as zero thickness, and the
layer-order count is reported."""
import io      # in-memory files

import numpy as np                          # arrays
from fastapi.testclient import TestClient   # calls the web app in-process, no server needed
from PIL import Image                       # image encoding/decoding

from oct_checks import boundaries as qb             # the layer-order check
from flywheel_gears.relay import relay                     # the relay's parser: the other side of the contract
from gpu_service import app as svc              # the service (the stub, since MODEL_PATH is not set in tests)
from flywheel_gears.triage import oct_layers, triage       # the triage gear's checks

client = TestClient(svc.app)   # a fake HTTP client wired straight into the app


def png_bytes(a: np.ndarray) -> bytes:
    """An array as PNG file bytes, like the relay sends."""
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return buf.getvalue()


def test_predict_response_passes_the_relays_validation_and_the_triage_checks():
    r = client.post("/predict", content=png_bytes(np.zeros((512, 496), np.uint8)))  # a blank B-scan
    assert r.status_code == 200                              # the service answered
    p = relay.parse_response(r.json())  # the contract, checked by the relay's own parser
    assert p.model == "stub-0" and p.confidence == 0.9       # the stub's fixed answer
    assert p.mask.shape == (512, 512) and np.unique(p.mask).tolist() == [0, 1, 2, 3, 4, 5]  # all six labels
    assert triage.confidence_check(p.confidence, 0.9)["state"] == "pass"  # the stub stays usable for gear tests
    assert oct_layers.slice_gates({"measurements": p.measurements}, {"min_retina_fraction": 0.9})[0]["state"] == "pass"
    assert qb.order_violation_columns(p.mask) == 0           # the stub's bands are in anatomical order


def test_predict_rejects_non_images_damaged_images_and_huge_uploads():
    assert client.post("/predict", content=b"not an image").status_code == 415  # 415 = unsupported media
    damaged = png_bytes(np.zeros((64, 64), np.uint8))[:60]                       # a PNG cut short
    assert client.post("/predict", content=damaged).status_code == 415
    assert client.post("/predict", content=b"0" * (svc.MAX_BYTES + 1)).status_code == 413  # too large


def test_health_names_the_model_the_relay_will_log():
    assert client.get("/health").json() == {"status": "ok", "model": "stub-0", "device": "none"}


def test_measurements_count_absent_ez_as_zero():
    mask = np.zeros((10, 4), np.uint8)   # 10 rows x 4 columns, all vitreous (0)
    mask[2:5, 0] = svc.EZ   # 3 px of EZ in column 0
    mask[2:4, 1] = svc.EZ   # 2 px in column 1; columns 2-3 have lost it
    mask[5:7, :3] = svc.RPE  # 2 px of RPE in columns 0-2; column 3 shows no retina at all
    assert svc.measurements(mask) == {
        "ez_mean_thickness_px": 1.25,    # (3 + 2 + 0 + 0) / 4 columns: lost EZ counts as 0
        "ez_present_fraction": 0.5,      # EZ in 2 of 4 columns
        "rpe_mean_thickness_px": 1.5,    # (2 + 2 + 2 + 0) / 4
        "retina_present_fraction": 0.75,  # retina in 3 of 4 columns
        # This toy mask leaves label 0 (vitreous) below the RPE in columns 0-2, which anatomy never
        # allows, so 3 columns break the layer order; column 3 (all vitreous) does not.
        "order_violation_columns": 3.0}


def test_layer_order_count_finds_columns_that_step_back_up():
    mask = np.zeros((6, 3), np.uint8)    # 6 rows x 3 columns
    mask[:, 0] = [0, 1, 2, 3, 4, 5]      # column 0: perfect order
    mask[:, 1] = [0, 1, 3, 2, 4, 5]      # column 1: EZ above OPL-IS/OS -> out of order
    mask[:, 2] = [0, 1, 2, 3, 4, 0]      # column 2: vitreous below the RPE -> out of order
    assert svc.measurements(mask)["order_violation_columns"] == 2.0  # columns 1 and 2
    assert qb.order_violation_by_column(mask).tolist() == [False, True, True]  # and which ones


def test_out_of_order_thickness_counts_every_misplaced_pixel_not_just_the_first_step():
    mask = np.zeros((8, 3), np.uint8)
    mask[:, 0] = [0, 1, 2, 3, 4, 5, 5, 5]    # column 0: in order -> 0
    mask[:, 1] = [0, 1, 2, 3, 4, 5, 0, 5]    # column 1: one stray vitreous pixel under the RPE -> 1
    mask[:, 2] = [0, 1, 4, 2, 2, 2, 3, 4]    # column 2: RPE above the bands -> the 4 pixels under it
    assert qb.out_of_order_pixels(mask).tolist() == [0, 1, 4]
