# gpu_service

Runs on the GPU machine. It loads the model once and answers the relay gear, one B-scan per request.

| Path | What |
|---|---|
| `app.py` | The FastAPI service: `GET /health` (up? which model?) and `POST /predict` (an image in; the mask, the confidence and measurements out) |
| `tta.py` | The confidence score: six flipped or shifted copies of a B-scan, segmented and compared |
| `unet/` | The U-Net, its training and out-of-fold prediction; one YAML (`config.yaml`) drives it all. Its [model card](unet/MODEL_CARD.md): network, training, validation per fold and cohort |
| `slurm/` | The same Docker image as Slurm jobs under Apptainer: `train.sbatch`, `score.sbatch`, `serve.sbatch` |
| `Dockerfile` | One image for the service, training and scoring (build it from the repo root) |

Any model can replace the U-Net if `/predict` returns the same JSON:
`{"model", "confidence", "mask_png", "measurements"}` (details in `app.py`). A new model needs its
own cutoff (`analysis/gate_analysis.py`), and the triage gear's `cutoff_model` set to its name.

```bash
docker build -f gpu_service/Dockerfile -t local/octriage-service:0.6.1 .
docker run --rm --gpus all -p 8765:8000 -v "$PWD/weights:/weights:ro" \
  -e MODEL_PATH=/weights/unet_w32_cv5_ez_border-fold0.pt local/octriage-service:0.6.1
uvicorn gpu_service.app:app --port 8765          # no GPU or weights: a stub that answers the same way
```
