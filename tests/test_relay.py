"""The relay gear's logic (flywheel_gears/relay/relay.py): the input and every answer from the model service are
checked, images not acquired are never sent, the masks come back as one label volume with the
answers in its header, the NRRD files carry their per-axis fields in the order other readers
expect, and a dropped connection is tried again while a refused request is not."""
import base64        # building fake mask_png answers
import io            # in-memory PNGs
import urllib.error  # the HTTP errors the retry rule tells apart

import numpy as np       # arrays
import pytest            # the test framework
from PIL import Image    # PNG encoding

from flywheel_gears.relay import relay  # the code under test


def png_b64(a: np.ndarray) -> str:
    """An array as base64 PNG text, as the service sends masks (8-bit unless the array says otherwise)."""
    buf = io.BytesIO()
    Image.fromarray(a if a.dtype == np.uint16 else a.astype(np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def body(**over):
    """A valid service answer; keyword arguments replace fields to make it invalid."""
    return {"model": "m", "confidence": 0.8, "mask_png": png_b64(np.zeros((8, 6))),
            "measurements": {"retina_present_fraction": 1.0, "ez": 3.5}, **over}


def test_parse_valid_response():
    p = relay.parse_response(body())
    assert (p.model, p.confidence) == ("m", 0.8)
    assert p.mask.shape == (8, 6) and p.mask.dtype == np.uint8          # decoded to a label array
    assert p.measurements == {"retina_present_fraction": 1.0, "ez": 3.5}


@pytest.mark.parametrize("bad", [
    {"confidence": 1.2}, {"confidence": float("nan")}, {"confidence": "0.8"},     # impossible scores
    {"confidence": True},                                                         # a bool is not a score
    {"mask_png": "not base64!"}, {"mask_png": base64.b64encode(b"GIF89a").decode()},  # not a PNG
    {"mask_png": base64.b64encode(b"\x89PNG\r\n\x1a\nbroken").decode()},          # PNG signature, no image
    {"mask_png": png_b64(np.zeros((4, 4, 3)))},                                   # not 2-D labels
    {"mask_png": png_b64(np.full((4, 4), 300, np.uint16))},                       # 16-bit labels
    {"mask_png": png_b64(np.full((4, 4), 255))},                                  # the reserved label
    {"measurements": {"ez": float("inf")}},                                       # not finite
    {"measurements": {"ez": "3"}}, {"measurements": {"ez": True}},                # not a number
    {"measurements": [1.0, 2.0]},                                                 # not name -> number
])
def test_parse_rejects_malformed_response(bad):
    with pytest.raises(ValueError):                                     # a bad answer never becomes a verdict
        relay.parse_response(body(**bad))


def test_parse_rejects_missing_fields():
    with pytest.raises(KeyError):                                       # no confidence at all
        relay.parse_response({"model": "m", "mask_png": png_b64(np.zeros((2, 2)))})


def fake_service(calls: list, models=("m",)):
    """A stand-in model service: answers every image with a 4 x 10 mask whose value is the call number."""
    def ask(png: bytes) -> dict:
        calls.append(np.array(Image.open(io.BytesIO(png))))              # what the service received
        mask = np.full((4, 10), len(calls), np.uint8)
        return {"model": models[(len(calls) - 1) % len(models)], "confidence": 0.5 + len(calls) / 10,
                "mask_png": png_b64(mask), "measurements": {"retina_present_fraction": 1.0}}
    return ask


def header(missing: str = "") -> dict:
    return {"spacings": [0.4, 0.0035, 0.018], "units": ["mm"] * 3, "labels": ["bscan", "axial", "lateral"],
            relay.MISSING_KEY: missing}


def test_relay_skips_missing_slices_and_keeps_the_field_of_view():
    vol = np.stack([np.full((4, 5), i, np.uint8) for i in (10, 20, 30)])  # 3 images, 4 rows x 5 columns
    calls = []
    labels, out, answers = relay.relay_volume(vol, header("2"), fake_service(calls))
    assert len(calls) == 2 and calls[1][0, 0] == 30                      # image 2 was never sent
    assert (labels[1] == relay.MISSING).all()                            # and is marked not acquired
    assert labels.shape == (3, 4, 10) and set(answers) == {"1", "3"}
    assert out["spacings"] == pytest.approx([0.4, 0.0035, 0.018 * 5 / 10])  # 5 columns became 10: same width
    assert relay.read_answers(out)[3]["confidence"] == pytest.approx(0.7)
    assert out[relay.MISSING_KEY] == "2"


def test_relay_refuses_masks_of_different_shapes_and_empty_volumes():
    shapes = iter([(4, 10), (5, 10)])
    def ask(png):
        return {"model": "m", "confidence": 0.9, "mask_png": png_b64(np.zeros(next(shapes))), "measurements": {}}
    with pytest.raises(ValueError):
        relay.relay_volume(np.zeros((2, 4, 5), np.uint8), header(), ask)
    with pytest.raises(ValueError):                                     # nothing acquired: nothing to send
        relay.relay_volume(np.zeros((1, 4, 5), np.uint8), header("1"), ask)


def test_relay_refuses_a_model_change_inside_a_volume():
    with pytest.raises(ValueError, match="changed model"):              # one cutoff belongs to one model
        relay.relay_volume(np.zeros((2, 4, 5), np.uint8), header(), fake_service([], models=("a", "b")))


@pytest.mark.parametrize("volume, head", [
    (np.zeros((4, 5), np.uint8), header()),                            # one image, not a stack
    (np.zeros((2, 4, 5), np.uint16), header()),                        # 16-bit pixels
    (np.zeros((2, 4, 5), np.uint8), {}),                                # no spacings
    (np.zeros((2, 4, 5), np.uint8), {**header(), "spacings": [0.4, 0, 0.018]}),  # a zero spacing
])
def test_relay_refuses_an_input_it_cannot_describe(volume, head):
    calls = []
    with pytest.raises(ValueError):
        relay.relay_volume(volume, head, fake_service(calls))
    assert calls == []                                                  # refused before any image left


def test_volumes_round_trip_with_axis_fields_in_file_order(tmp_path):
    vol = np.zeros((3, 4, 5), np.uint8)                                  # images x rows x columns
    relay.write_volume(tmp_path / "v.nrrd", vol, header("2"))
    text = (tmp_path / "v.nrrd").read_bytes().split(b"\n\n")[0].decode("latin1")
    # NRRD lists axes fastest first: columns (5) first, images (3) last, and spacings must match.
    assert "sizes: 5 4 3" in text and "0.017999" in text.split("spacings:")[1].split()[0]
    data, h = relay.read_volume(tmp_path / "v.nrrd")                     # back in our order
    assert data.shape == (3, 4, 5) and list(h["spacings"]) == pytest.approx([0.4, 0.0035, 0.018])
    assert relay.missing_slices(h) == {2}


# ---------- the network rules ----------

def flaky(failures: list):
    """A call that raises each error in `failures` in turn, then answers."""
    def call():
        if failures:
            raise failures.pop(0)
        return {"ok": True}
    return call


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("http://gpu-host/predict", code, "error", {}, None)


def test_a_dropped_connection_or_server_error_is_tried_again():
    pauses = []
    errors = [urllib.error.URLError("connection refused"), http_error(503)]
    assert relay.with_retries(flaky(errors), sleep=pauses.append) == {"ok": True}
    assert pauses == [2, 4]                                             # a growing pause between tries


def test_a_refused_request_fails_at_once_and_retries_stop():
    with pytest.raises(urllib.error.HTTPError):                         # 4xx: our request is wrong
        relay.with_retries(flaky([http_error(415)]), sleep=lambda s: None)
    errors = [TimeoutError()] * relay.ATTEMPTS                          # down for good: give up
    with pytest.raises(TimeoutError):
        relay.with_retries(flaky(list(errors)), sleep=lambda s: None)


@pytest.mark.parametrize("url", ["gpu-host:8000", "", "ftp://gpu-host"])
def test_the_service_url_must_be_a_web_address(url):
    with pytest.raises(ValueError):
        relay.check_service_url(url)


def test_the_service_url_loses_a_trailing_slash():
    assert relay.check_service_url("http://gpu-host:8000/") == "http://gpu-host:8000"
