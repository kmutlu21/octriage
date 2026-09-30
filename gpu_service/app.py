"""The model service: the GPU side of the pipeline. The relay gear POSTs each image's bytes to /predict.

What it does
    Loads the model once when it starts, then answers one image per request: the mask, the model's
    confidence in it, and a few measurements. GET /health says whether it is up and which model
    it serves.

Why a separate service rather than the model inside the gear
    WRC's Flywheel runs gears on machines without GPUs, and a gear that loads a model pays container
    start-up plus weight loading on every job. The service loads the model once and answers each
    scan in ~0.2 s on a GPU. At WRC it would run on their own hardware (UW Platform R);
    gpu_service/slurm/serve.sbatch starts it as a Slurm job.

Response contract (what flywheel_gears/relay/relay.py checks; any model that returns this can be swapped in):
    {"model": str, "confidence": float in [0, 1], "mask_png": base64 PNG of 8-bit labels,
     "measurements": {name: float}}

With MODEL_PATH set, the service loads that U-Net checkpoint and answers with test-time-augmentation
confidence (gpu_service/tta.py). Without it, a stub answers (flat, ordered bands and a fixed confidence
from STUB_CONFIDENCE), so the gears can be tested without a GPU.

    uvicorn gpu_service.app:app --port 8000                                   # stub
    MODEL_PATH=weights/unet_w32_cv5_ez_border-fold0.pt uvicorn gpu_service.app:app --port 8000

Not built yet: encryption and a login. On a real network the service would sit behind HTTPS with
a key, inside UW; the demo runs on one machine.
"""
import base64  # the mask PNG travels as base64 text inside JSON
import io      # in-memory files for reading and writing images
import os      # environment variables (MODEL_PATH, STUB_CONFIDENCE)

import numpy as np                                   # arrays
from fastapi import FastAPI, HTTPException, Request  # the web framework
from PIL import Image, UnidentifiedImageError        # image decoding and encoding
from starlette.concurrency import run_in_threadpool  # run the model without blocking the web loop

from oct_checks.boundaries import order_violation_columns    # the anatomy check, shared with the triage gear

# Mask labels (data_prep/build_manifest.py): 0 vitreous, 1 ILM-OPL, 2 OPL-IS/OS, 3 EZ band, 4 RPE, 5 below.
EZ, RPE = 3, 4


class StubPredictor:
    """Flat, anatomically ordered bands at fixed rows: plausible enough to pass the gate.
    Lets the gears, the rules and the tags be tested end to end on a laptop with no torch."""
    name, device = "stub-0", "none"                                  # reported by /health
    BOUNDARY_ROWS = np.array([150, 200, 230, 240, 250])              # ILM, OPL, IS-OS, IBRPE, OBRPE rows

    def __init__(self, confidence: float):
        self.confidence = confidence                                 # the fixed score it always returns

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        # Label of each row = how many boundaries lie at or above it (0 above the ILM ... 5 below OBRPE).
        column = (np.arange(512)[:, None] >= self.BOUNDARY_ROWS).sum(axis=1).astype(np.uint8)
        return np.repeat(column[:, None], 512, axis=1), self.confidence  # same column 512 times


class UNetPredictor:
    """The real model: a trained U-Net checkpoint (gpu_service/unet/unet.py), scored with TTA (gpu_service/tta.py)."""

    def __init__(self, checkpoint: str):
        import torch  # imported here so the stub runs without torch installed

        from gpu_service.unet.unet import load_checkpoint, preprocess  # the shared model code
        from gpu_service.tta import predict_tta                   # the 6-copy confidence

        self.device = "cuda" if torch.cuda.is_available() else "cpu"            # GPU if there is one
        self.model, self.name = load_checkpoint(checkpoint, self.device)        # weights, loaded once
        self._preprocess, self._predict = preprocess, predict_tta              # keep references for __call__

    def __call__(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        mask, confidence = self._predict(self.model, self._preprocess(image).to(self.device))  # segment + score
        return mask.cpu().numpy(), confidence                                    # back to a numpy mask


def measurements(mask: np.ndarray) -> dict[str, float]:
    """Band thickness in mask pixels (vertical pixels = source pixels), EZ coverage, how much of the
    scan shows any retina at all (the triage gear's plausibility check), and how many columns break the
    anatomical layer order (a signal the triage gear reports; it does not decide the verdict).

    Columns where a band is absent count as zero thickness, so EZ loss lowers the mean
    instead of vanishing from it; `ez_present_fraction` reports how much EZ is left (the
    per-B-scan analogue of WRC's EZ Integrity Index, Erb et al., Ophthalmol Sci 2025).
    The same function measures the graders' masks in analysis/gate_analysis.py, so model and
    graders are measured identically.
    """
    ez_cols = (mask == EZ).sum(axis=0)                               # EZ pixels in each image column
    return {
        "ez_mean_thickness_px": float(ez_cols.mean()),               # average EZ thickness, in pixels
        "ez_present_fraction": float((ez_cols > 0).mean()),          # share of columns that show EZ
        "rpe_mean_thickness_px": float((mask == RPE).sum(axis=0).mean()),  # average RPE thickness
        # Share of columns with any retinal band (labels 1-4) anywhere in them.
        "retina_present_fraction": float(np.isin(mask, (1, 2, EZ, RPE)).any(axis=0).mean()),
        "order_violation_columns": float(order_violation_columns(mask)),  # anatomy check (oct_checks/boundaries.py)
    }


app = FastAPI(title="OCTriage model service")                         # the web application
MAX_BYTES = 20 * 1024 * 1024  # one B-scan PNG is ~100 KB; refuse anything absurd before decoding it
# Built once, at import (i.e. when uvicorn starts), so the model is loaded once and reused.
predictor = (UNetPredictor(os.environ["MODEL_PATH"]) if os.environ.get("MODEL_PATH")
             else StubPredictor(float(os.environ.get("STUB_CONFIDENCE", "0.9"))))


@app.get("/health")
def health() -> dict:
    """For a person or a monitor: is the service up, which model, on which device."""
    return {"status": "ok", "model": predictor.name, "device": predictor.device}


@app.post("/predict")
async def predict(request: Request) -> dict:
    body = await request.body()                                     # the image's bytes
    if len(body) > MAX_BYTES:
        raise HTTPException(413, f"image larger than {MAX_BYTES} bytes")  # 413 = payload too large
    try:
        # Raw bytes in (TIFF or PNG); "L" = 8-bit grayscale, which is what the model was trained on.
        image = np.array(Image.open(io.BytesIO(body)).convert("L"))
    except (UnidentifiedImageError, OSError):                       # not an image, or a damaged one
        raise HTTPException(415, "request body is not a readable image") from None  # 415 = unsupported media
    # The model runs in a worker thread, so the service can still answer /health meanwhile.
    mask, confidence = await run_in_threadpool(predictor, image)    # segment and score
    buf = io.BytesIO()                                              # an in-memory PNG file
    Image.fromarray(mask).save(buf, format="PNG")                   # PNG is lossless, so labels survive exactly
    return {"model": predictor.name, "confidence": confidence,
            "mask_png": base64.b64encode(buf.getvalue()).decode(), "measurements": measurements(mask)}
