#!/bin/bash
# Out-of-fold predictions for every scan (the same TTA code the service runs), then the gate
# analysis: regenerates outputs/<experiment>/analysis/results.json from the trained folds.
# Chain it after training so it starts only if every fold succeeded:
#   TRAIN=$(sbatch --parsable gpu_service/slurm/train.sbatch)
#   sbatch --dependency=afterok:$TRAIN gpu_service/slurm/score.sbatch
#SBATCH --job-name=oct-score
#SBATCH --partition=gpu          # GPUs live in the gpu partition
#SBATCH --gres=gpu:1             # one GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G                 # the scans are held in RAM, as in training
#SBATCH --time=02:00:00          # ~7 min on the RTX 2060 bench
#SBATCH --output=logs/%x_%j.out  # job name, job id
set -euo pipefail                # stop at the first error, unset variable or failed pipe
SIF=${SIF:-containers/octriage-service_0.6.1.sif}
CONFIG=${CONFIG:-gpu_service/unet/config.yaml}
RUN=$(command -v apptainer || command -v singularity)
in_image() {  # run a command inside the image; --cleanenv drops CUDA_VISIBLE_DEVICES, so pass it on explicitly
  "$RUN" exec --nv --cleanenv --pwd /app \
    --env PYTHONNOUSERSITE=1 --env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:?no GPU allocated}" \
    --bind "$PWD/data:/data:ro" --bind "$PWD/outputs:/app/outputs" "$SIF" "$@"
}

# Read the experiment name from the config, with the image's own Python and YAML parser.
EXPERIMENT=$(in_image python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['experiment_name'])" "$CONFIG")
echo "scoring $EXPERIMENT on $(hostname), GPU $CUDA_VISIBLE_DEVICES"
in_image python -m gpu_service.unet.predict --config "$CONFIG"                                    # 1,672 out-of-fold masks
in_image python -m analysis.gate_analysis --experiment "outputs/$EXPERIMENT" --data /data  # results.json
