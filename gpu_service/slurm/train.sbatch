#!/bin/bash
# Train one cross-validation fold per array task. Folds are independent (fold k tests on its own
# 12 patients), so Slurm runs them side by side on as many GPUs as it can give us.
# The code runs inside the service's Docker image, converted to an Apptainer/Singularity .sif,
# so one image serves both Docker and the cluster.
# Submit from the project directory, which holds data/ outputs/ containers/ logs/:
#   sbatch gpu_service/slurm/train.sbatch                          # folds 0-4, at most 2 at a time
#   sbatch --array=2 gpu_service/slurm/train.sbatch                # rerun fold 2 only
#   TRAIN_ARGS=--smoke sbatch --array=0-1 gpu_service/slurm/train.sbatch  # a few batches, to test the plumbing
# Tested on a single-node Slurm 23.11 + Apptainer 1.5.4 bench (RTX 2060): 5 folds in ~5 h.
# UW Platform R (untested): add --gres=gpu:nvidia_h200:1 --mem=64G and set
# SIF=docker://<registry>/octriage-service:<tag>. Its docs run containers through a Slurm plugin
# (--singularity-container=...); if apptainer is not on the compute nodes, use that instead.
#SBATCH --job-name=oct-train
#SBATCH --partition=gpu          # GPUs live in the gpu partition
#SBATCH --gres=gpu:1             # one GPU per fold
#SBATCH --cpus-per-task=6        # data-loader workers (config: num_workers 4) + the main process
#SBATCH --mem=6G                 # all 1,672 scans are held in RAM (~1.7 GB), see gpu_service/unet/data.py
#SBATCH --time=06:00:00          # wall-clock limit per fold
#SBATCH --array=0-4%2            # tasks 0-4 = folds 0-4; %2 = at most 2 at once
#SBATCH --output=logs/%x_%A_%a.out   # job name, array job id, task id
set -euo pipefail   # stop at the first error, unset variable or failed pipe
SIF=${SIF:-containers/octriage-service_0.6.1.sif}   # the image; override with SIF=... sbatch ...
CONFIG=${CONFIG:-gpu_service/unet/config.yaml}                  # the training recipe
RUN=$(command -v apptainer || command -v singularity)  # whichever container runtime the node has

# Copy the data to the node's own disk first: many small reads over a shared filesystem stall training.
SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/oct-train.XXXXXX")   # a private temporary folder on the node
trap 'rm -rf "$SCRATCH"' EXIT   # clean the node's disk however the job ends
start=$SECONDS                  # bash's built-in seconds counter
cp -r data "$SCRATCH/data"
echo "fold $SLURM_ARRAY_TASK_ID on $(hostname), GPU $CUDA_VISIBLE_DEVICES: data staged in $((SECONDS - start)) s"

# --nv: expose the GPU. --cleanenv: do not leak the host's Python/env into the container.
# --cleanenv drops CUDA_VISIBLE_DEVICES, so pass it on, or a shared node would train on GPU 0.
# Data is mounted read-only; outputs/ is the only place the job writes.
"$RUN" exec --nv --cleanenv --pwd /app \
  --env PYTHONNOUSERSITE=1 --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?no GPU allocated}" \
  --bind "$SCRATCH/data:/data:ro" --bind "$PWD/outputs:/app/outputs" \
  "$SIF" python -m gpu_service.unet.train --config "$CONFIG" --folds "$SLURM_ARRAY_TASK_ID" ${TRAIN_ARGS:-}
