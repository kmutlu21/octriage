"""The relay's logic: carry every image of one volume to a model service and bring the answers back.

What this file does
    Takes a volume (a stack of 2-D images, e.g. the B-scans of one OCT visit), sends each image to a
    model service over HTTP, checks every answer, and returns the masks as ONE label volume with
    every image's answer stored in that volume's header.

Why a relay at all
    WRC's Flywheel runs its gears on machines without GPUs, so the model lives on a GPU machine as a
    service that loads it once. The gear only carries data there and back, one whole volume per
    job (the idea comes from WRC's fw_model_serving proof of concept; see the README).

What it knows
    Nothing about OCT or any disease. A model service is anything that answers this contract
    (gpu_service/app.py is one implementation):
        GET  <url>/health   -> {"status": "ok", "model": str, ...}
        POST <url>/predict  (body: one PNG image)
                            -> {"model": str, "confidence": number in [0, 1],
                                "mask_png": base64 PNG of 8-bit labels,
                                "measurements": {name: finite number}}
    Label 255 is reserved: it marks images that were never acquired.

Safety rule
    Anything malformed (the input volume, or any answer) raises ValueError or KeyError and the job
    fails. A bad answer must never turn into a verdict downstream.

Network
    Each request has a time limit. A dropped connection, a time-out or a server error (5xx) is
    tried again, up to 3 times with a growing pause; a refusal (4xx) fails at once, because
    sending the same request again cannot help.

This file has no Flywheel imports, so it can be tested on its own (tests/test_relay.py);
flywheel_gears/relay/run.py does the Flywheel plumbing.
"""
import base64          # the mask arrives as base64 text
import binascii        # the error base64 raises on bad input
import http.client     # the error a connection dropped mid-answer raises
import io              # in-memory files for PNG encoding and decoding
import json            # the service's answers, and the per-image answers stored in the NRRD header
import logging         # progress and retries go to the job log
import math            # isfinite, to reject NaN and infinity
import time            # pauses between retries
import urllib.error    # HTTP errors
import urllib.request  # plain HTTP, no extra dependency
from dataclasses import dataclass  # a small read-only record type
from typing import Callable        # the type of the "ask the service" function

import nrrd              # NRRD volumes (pynrrd, MIT)
import numpy as np       # arrays
from PIL import Image    # PNG encoding and decoding

MISSING = 255             # label written for an image that was not acquired (the triage gear skips it)
HEADER_KEY = "relay"      # the NRRD header field carrying the per-image answers (JSON)
MISSING_KEY = "missing slices"  # NRRD header field listing images not acquired (1-based, space-separated)
AXIS_FIELDS = ("spacings", "units", "labels")  # NRRD fields with one entry per axis
TIMEOUT_S = 120   # per request; generous, because the first request after the service starts is slow
ATTEMPTS = 3      # tries per request: a dropped connection should not fail a whole visit
PAUSE_S = 2       # seconds before the 2nd try; doubled before the 3rd
log = logging.getLogger("relay")


def write_volume(path, volume: np.ndarray, header: dict) -> None:
    """Write a (images, rows, columns) array. NRRD lists per-axis fields fastest axis first; pynrrd
    reverses `sizes` for index_order="C" but not the other per-axis fields, so reverse them here,
    or another reader (e.g. 3D Slicer) would put the image spacing on the column axis."""
    h = {k: (list(v)[::-1] if k in AXIS_FIELDS else v) for k, v in header.items()}
    nrrd.write(str(path), volume, h, index_order="C")


def read_volume(path) -> tuple[np.ndarray, dict]:
    """Read a volume written by write_volume (or any correct NRRD) as (images, rows, columns)."""
    data, h = nrrd.read(str(path), index_order="C")
    return data, {k: (list(v)[::-1] if k in AXIS_FIELDS else v) for k, v in h.items()}


def check_volume(volume: np.ndarray, header: dict) -> None:
    """Refuse an input the service could misread, with a message a person can act on."""
    if volume.ndim != 3:
        raise ValueError(f"the input must be a stack of 2-D images (3 axes), got {volume.ndim} axes")
    if volume.dtype != np.uint8:
        raise ValueError(f"the input must be 8-bit greyscale, got {volume.dtype}")
    spacings = header.get("spacings")
    if spacings is None or len(spacings) != 3:
        raise ValueError("the input's header needs 'spacings' in mm for all 3 axes")
    if not all(math.isfinite(float(s)) and float(s) > 0 for s in spacings):
        raise ValueError(f"'spacings' must be positive numbers, got {list(spacings)}")


def is_number(v) -> bool:
    """A real, finite number. bool is excluded on purpose: in Python, True counts as the number 1."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


@dataclass(frozen=True)  # frozen: a checked answer cannot be altered afterwards
class Prediction:
    model: str            # which model answered, e.g. "unet_w32_cv5_ez_border-fold0"
    confidence: float     # 0-1, the model's own stability score (for our model: gpu_service/tta.py)
    mask: np.ndarray      # the label mask, decoded from the PNG
    measurements: dict    # whatever numbers the model reports (band thicknesses, retina share, ...)


def parse_response(body: dict) -> Prediction:
    """Check one answer from the service. Anything malformed raises ValueError or KeyError."""
    confidence = body["confidence"]                        # KeyError if missing
    if not is_number(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be a number in [0, 1], got {confidence!r}")
    try:
        png = base64.b64decode(body["mask_png"], validate=True)  # validate: reject stray characters
    except binascii.Error as e:
        raise ValueError(f"mask_png is not valid base64: {e}") from None
    if not png.startswith(b"\x89PNG\r\n\x1a\n"):  # the 8-byte signature every PNG file starts with
        raise ValueError("mask_png is not a PNG")
    try:
        mask = np.array(Image.open(io.BytesIO(png)))       # decode to a label array
    except Exception as e:  # a PNG signature on broken data: PIL raises several error types
        raise ValueError(f"mask_png cannot be decoded: {e}") from None
    if mask.ndim != 2:                                     # a label mask is one channel, 2-D
        raise ValueError(f"mask must be 2-D labels, got shape {mask.shape}")
    if mask.dtype != np.uint8:                             # 16-bit labels would be cut to 8 bits later
        raise ValueError(f"mask must hold 8-bit labels, got {mask.dtype}")
    if (mask == MISSING).any():                            # 255 would be read as "not acquired"
        raise ValueError(f"mask uses label {MISSING}, which is reserved for images not acquired")
    measurements = body.get("measurements", {})
    if not isinstance(measurements, dict) or not all(is_number(v) for v in measurements.values()):
        raise ValueError(f"measurements must map names to finite numbers, got {measurements!r}")
    return Prediction(str(body["model"]), float(confidence), mask, dict(measurements))


def missing_slices(header: dict) -> set[int]:
    """The 1-based image numbers the header marks as not acquired (none if the field is absent)."""
    return {int(s) for s in str(header.get(MISSING_KEY, "")).split()}


def to_png(image: np.ndarray) -> bytes:
    """One 2-D image as PNG bytes: lossless, so the service sees exactly these pixels. PIL writes no
    text or dates into the file, so the request carries pixels and nothing else."""
    buf = io.BytesIO()
    Image.fromarray(np.asarray(image)).save(buf, format="PNG")
    return buf.getvalue()


def relay_volume(volume: np.ndarray, header: dict,
                 ask: Callable[[bytes], dict]) -> tuple[np.ndarray, dict, dict]:
    """volume (n, H, W) images + their NRRD header; ask(png bytes) -> the service's JSON answer.
    Returns (labels (n, H', W'), the header for the label volume, the per-image answers).

    - Images listed as missing are not sent; their labels are MISSING (255).
    - The spacing is rescaled when the model answers on a different grid (our U-Net answers on
      512 x 512 while OCT5k images are 512 x 496): same field of view, more or fewer pixels."""
    check_volume(volume, header)
    skip = missing_slices(header)
    answers, masks = {}, {}                               # image number -> answer; image number -> mask
    for i, image in enumerate(volume, start=1):           # image numbers are 1-based, like B-scan numbers
        if i in skip:
            continue                                      # never send an image that was not acquired
        p = parse_response(ask(to_png(image)))            # one request, checked
        masks[i] = p.mask
        answers[str(i)] = {"model": p.model, "confidence": p.confidence, "measurements": p.measurements}
    if not masks:
        raise ValueError("the volume has no acquired images to send")
    if len({a["model"] for a in answers.values()}) != 1:  # one volume, one model: never mix two
        raise ValueError("the service changed model in the middle of a volume")
    shapes = {m.shape for m in masks.values()}
    if len(shapes) != 1:                                  # every answer must be on the same grid
        raise ValueError(f"the service returned masks of different shapes: {shapes}")
    (h, w), (_, in_h, in_w) = shapes.pop(), volume.shape
    labels = np.full((len(volume), h, w), MISSING, np.uint8)  # every position "missing" to start
    for i, m in masks.items():
        labels[i - 1] = m                                 # put each mask at its image position
    spacings = [float(s) for s in header["spacings"]]     # mm per step: across images, rows, columns
    out = {"spacings": [spacings[0], spacings[1] * in_h / h, spacings[2] * in_w / w],  # same field of view
           "units": list(header.get("units", ["mm"] * 3)),
           "labels": list(header.get("labels", ["slice", "axial", "lateral"])),
           MISSING_KEY: " ".join(str(s) for s in sorted(skip)),
           HEADER_KEY: json.dumps({"slices": answers}, separators=(",", ":"))}  # one line, no newlines
    return labels, out, answers


def read_answers(header: dict) -> dict:
    """The per-image answers a relay stored in a label volume's header: {image number: answer}."""
    return {int(k): v for k, v in json.loads(header[HEADER_KEY])["slices"].items()}


# ---------- talking to the service ----------

def check_service_url(url: str) -> str:
    """The setting must be a web address; strip a trailing slash so '<url>/predict' is well formed."""
    if not str(url).startswith(("http://", "https://")):
        raise ValueError(f"service_url must start with http:// or https://, got {url!r}")
    return str(url).rstrip("/")


def with_retries(call, sleep=time.sleep):
    """Run call(); on a network failure or a server error (5xx), wait and try again, up to ATTEMPTS.
    A 4xx answer means our request itself was refused, so trying again cannot help: fail at once.
    (sleep is a parameter so the tests can skip the real pauses.)"""
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return call()
        except urllib.error.HTTPError as e:                    # the service answered with an error code
            if e.code < 500 or attempt == ATTEMPTS:
                raise
            problem = f"HTTP {e.code}"
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException) as e:
            if attempt == ATTEMPTS:                             # unreachable, timed out, or cut off
                raise
            problem = repr(e)
        pause = PAUSE_S * 2 ** (attempt - 1)
        log.warning("service did not answer (%s), attempt %d of %d; trying again in %d s",
                    problem, attempt, ATTEMPTS, pause)
        sleep(pause)


def get_json(url: str) -> dict:
    """GET a URL and read its JSON answer (used for /health)."""
    with urllib.request.urlopen(url, timeout=TIMEOUT_S) as resp:
        return json.load(resp)


def post_png(url: str, png: bytes) -> dict:
    """POST one PNG image and read the JSON answer. Only the pixels are sent: no file name, no IDs."""
    req = urllib.request.Request(url, data=png, headers={"Content-Type": "image/png"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        return json.load(resp)


def service(url: str) -> tuple[dict, Callable[[bytes], dict]]:
    """Check the address, ask /health who is answering, and return (health, ask): ask(png) POSTs one
    image to /predict with retries and logs progress every 10 images."""
    url = check_service_url(url)
    health = with_retries(lambda: get_json(url + "/health"))
    log.info("service %s is up: model %s on %s", url, health.get("model"), health.get("device"))
    answered = []                                          # one entry per answered request

    def ask(png: bytes) -> dict:
        answer = with_retries(lambda: post_png(url + "/predict", png))
        answered.append(1)
        if len(answered) % 10 == 0:
            log.info("%d images answered", len(answered))
        return answer
    return health, ask
