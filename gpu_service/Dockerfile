# One image for inference, training and analysis. Build from the repo root:
#   docker build -f gpu_service/Dockerfile -t local/octriage-service:0.6.1 .
# Serve the model (GPU), with the weights mounted read-only:
#   docker run --rm --gpus all -p 8765:8000 -v <repo>/weights:/weights:ro \
#     -e MODEL_PATH=/weights/unet_w32_cv5_ez_border-fold0.pt local/octriage-service:0.6.1
# Train (data read-only, outputs written back to the repo's outputs/):
#   docker run --rm --gpus all --shm-size=2g -v <repo>/data:/data:ro -v <repo>/outputs:/app/outputs \
#     local/octriage-service:0.6.1 python -m gpu_service.unet.train --config gpu_service/unet/config.yaml
# On a Slurm cluster the same image runs under Apptainer/Singularity: see gpu_service/slurm/*.sbatch.
FROM python:3.12-slim

# torch 2.7.1+cu118 is the newest build that runs on a CUDA 11.6 driver (via CUDA minor-version
# compatibility); that is the driver on the RTX 2060 this was developed on. On a current driver
# or cluster (e.g. H200s), rebuild with e.g.
#   --build-arg TORCH_VERSION=2.14.0 --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu126
ARG TORCH_VERSION=2.7.1
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir torch==${TORCH_VERSION} --index-url ${TORCH_INDEX}

WORKDIR /app
# Each requirements file gets its own layer, before any code, so code edits rebuild in seconds.
COPY gpu_service/requirements.txt gpu_service/requirements.txt
RUN pip install --no-cache-dir -r gpu_service/requirements.txt
# Training runs in this same image, so the served model and the trained model share one environment.
COPY gpu_service/unet/requirements.txt gpu_service/unet/requirements.txt
RUN pip install --no-cache-dir -r gpu_service/unet/requirements.txt
# The analyses run in it too, so one image regenerates every number in the README.
COPY analysis/requirements.txt analysis/requirements.txt
RUN pip install --no-cache-dir -r analysis/requirements.txt
COPY oct_checks/ oct_checks/
COPY gpu_service/ gpu_service/
COPY analysis/ analysis/

EXPOSE 8000
CMD ["uvicorn", "gpu_service.app:app", "--host", "0.0.0.0", "--port", "8000"]
