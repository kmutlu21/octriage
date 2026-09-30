#!/bin/bash
# The GPU inference service as a Slurm job; point the gear's service_url at the address it prints.
# It holds one GPU until --time runs out or you scancel it. The model loads once, so each scan
# then costs ~0.2 s instead of a container start and a weight load per gear job.
#   sbatch gpu_service/slurm/serve.sbatch                     # MODEL=weights/unet_w32_cv5_ez_border-fold0.pt, PORT=8766
#   MODEL=weights/other.pt PORT=9000 sbatch gpu_service/slurm/serve.sbatch
# Apptainer shares the node's network: the port opens on the node itself, so choose a free one.
# There is no authentication, so keep it inside the cluster network.
# Open question for WRC: whether Flywheel can reach a Platform R node at all (access there is
# through Azure Virtual Desktop); IT decides that, and it decides between this service and batch jobs.
#SBATCH --job-name=oct-serve
#SBATCH --partition=gpu          # GPUs live in the gpu partition
#SBATCH --gres=gpu:1             # one GPU
#SBATCH --cpus-per-task=2        # decoding PNGs; the GPU does the rest
#SBATCH --mem=4G
#SBATCH --time=01:00:00          # the service stops when this runs out
#SBATCH --output=logs/%x_%j.out  # job name, job id
set -euo pipefail                # stop at the first error, unset variable or failed pipe
SIF=${SIF:-containers/octriage-service_0.6.1.sif}
MODEL=${MODEL:-weights/unet_w32_cv5_ez_border-fold0.pt}   # the served model (fold 0)
PORT=${PORT:-8766}
RUN=$(command -v apptainer || command -v singularity)

echo "service: http://$(hostname):$PORT  model: $MODEL  GPU: $CUDA_VISIBLE_DEVICES"
# The weights folder is mounted read-only at /weights; the service loads MODEL_PATH at start-up.
"$RUN" exec --nv --cleanenv --pwd /app \
  --env PYTHONNOUSERSITE=1 --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?no GPU allocated}" \
  --env MODEL_PATH="/weights/$(basename "$MODEL")" --bind "$PWD/$(dirname "$MODEL"):/weights:ro" \
  "$SIF" uvicorn gpu_service.app:app --host 0.0.0.0 --port "$PORT"
