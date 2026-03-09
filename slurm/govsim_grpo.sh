#!/bin/bash
#SBATCH --job-name=govsim-grpo
#SBATCH --account=def-zhijing
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/govsim_grpo_%j.out
#SBATCH --error=slurm/logs/govsim_grpo_%j.err
#SBATCH --exclude=ng31004

# ============================================================
# GovSim GRPO Fine-tuning: Universalization Reasoning
#
# Trains gemma3-4b-it to exhibit cooperative behavior with
# universalization reasoning in commons dilemma simulations.
#
# Usage:
#   sbatch slurm/govsim_grpo.sh
#   MAX_STEPS=500 sbatch slurm/govsim_grpo.sh  # shorter run
# ============================================================

set -eo pipefail

MAX_STEPS="${MAX_STEPS:-1000}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-models/gemma3-4b-cooperative-v2}"
DATASET_PATH="${DATASET_PATH:-training/traces/generated_traces_1.5k.json}"

echo "============================================================"
echo "Job:       GovSim GRPO Fine-tuning"
echo "Date:      $(date)"
echo "Node:      $(hostname)"
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Max Steps: ${MAX_STEPS}"
echo "Seed:      ${SEED}"
echo "Output:    ${OUTPUT_DIR}"
echo "Dataset:   ${DATASET_PATH}"
echo "============================================================"

# --- Environment ---
cd /lustre07/scratch/marimeir/govsim

# Use micromamba govsim env (conda pytorch + CUDA libs, avoids CC pip conflicts)
eval "$(micromamba shell hook -s bash)"
micromamba activate govsim

# HuggingFace cache and offline mode (model is pre-cached)
export HF_HOME="/lustre07/scratch/marimeir/huggingface_cache"
export HUGGINGFACE_HUB_CACHE="/lustre07/scratch/marimeir/huggingface_cache/hub"
export HF_TOKEN=$(cat /home/marimeir/.cache/huggingface/token 2>/dev/null)
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Load .env if present
if [ -f .env ]; then
    source .env
fi

# Create output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p slurm/logs

# --- Verify prerequisites ---
if [ ! -f "${DATASET_PATH}" ]; then
    echo "ERROR: Dataset not found: ${DATASET_PATH}"
    exit 1
fi
echo "Dataset found: ${DATASET_PATH} ($(wc -c < "${DATASET_PATH}") bytes)"

# --- Run Training ---
echo ""
echo "============================================================"
echo "Starting GRPO Training"
echo "Started: $(date)"
echo "============================================================"

python -m training.train_grpo \
    --dataset_path "${DATASET_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --model_name "google/gemma-3-4b-it" \
    --max_steps "${MAX_STEPS}" \
    --num_generations 8 \
    --learning_rate 1e-6 \
    --batch_size 1 \
    --grad_accum 4 \
    --max_completion_length 2048 \
    --seed "${SEED}"

echo ""
echo "============================================================"
echo "Training complete!"
echo "End: $(date)"
echo "Output: ${OUTPUT_DIR}/final_adapter/"
echo "============================================================"
