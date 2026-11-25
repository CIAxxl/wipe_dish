#!/usr/bin/env bash
#SBATCH --job-name=wipe-v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=wipe-v2-%j.out

set -euo pipefail
cd "${WIPE_V2_REPO:-$SLURM_SUBMIT_DIR}"
nvidia-smi
exec srun bash scripts/real_world/wipe_v2/run.sh "$@"
