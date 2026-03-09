#!/bin/bash
#SBATCH --job-name=govsim-sft-3gpu
#SBATCH --account=def-zhijing
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/sft_3gpu_%j.out
#SBATCH --error=slurm/logs/sft_3gpu_%j.err
#SBATCH --exclude=ng31004

# ============================================================
# GovSim SFT Fine-tuning: Mixed Dataset, 3-GPU DeepSpeed
#
# Trains gemma3-4b-it on GovSim + general cooperative traces
# using LoRA SFT with DeepSpeed ZeRO Stage 2 across 3 A100s.
#
# Usage:
#   sbatch slurm/govsim_sft_3gpu.sh
#   OUTPUT_DIR=models/gemma3-4b-sft-v4 sbatch slurm/govsim_sft_3gpu.sh
# ============================================================

set -eo pipefail

SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-models/gemma3-4b-sft-v4}"
DATASET_PATH="${DATASET_PATH:-training/traces/generated_traces_1k.json}"
GENERAL_TRACES_PATH="${GENERAL_TRACES_PATH:-training/traces/general_traces.json}"
MIX_RATIO="${MIX_RATIO:-0.2}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"

echo "============================================================"
echo "Job:       GovSim SFT Fine-tuning (3-GPU)"
echo "Date:      $(date)"
echo "Node:      $(hostname)"
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "GPUs:      $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Epochs:    ${NUM_EPOCHS}"
echo "LR:        ${LEARNING_RATE}"
echo "Mix Ratio: ${MIX_RATIO}"
echo "Seed:      ${SEED}"
echo "Output:    ${OUTPUT_DIR}"
echo "Dataset:   ${DATASET_PATH}"
echo "General:   ${GENERAL_TRACES_PATH}"
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
export PYTHONUNBUFFERED=1
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
    echo "ERROR: GovSim dataset not found: ${DATASET_PATH}"
    exit 1
fi
echo "GovSim dataset found: ${DATASET_PATH} ($(wc -c < "${DATASET_PATH}") bytes)"

if [ ! -f "${GENERAL_TRACES_PATH}" ]; then
    echo "WARNING: General traces not found: ${GENERAL_TRACES_PATH}"
    echo "Proceeding with GovSim-only training."
    GENERAL_ARG=""
else
    echo "General traces found: ${GENERAL_TRACES_PATH} ($(wc -c < "${GENERAL_TRACES_PATH}") bytes)"
    GENERAL_ARG="--general_traces_path ${GENERAL_TRACES_PATH} --mix_ratio ${MIX_RATIO}"
fi

# --- Run Training ---
echo ""
echo "============================================================"
echo "Starting SFT Training (3-GPU DeepSpeed ZeRO-2)"
echo "Started: $(date)"
echo "============================================================"

python -m training.train_sft \
    --dataset_path "${DATASET_PATH}" \
    ${GENERAL_ARG} \
    --output_dir "${OUTPUT_DIR}" \
    --model_name "google/gemma-3-4b-it" \
    --num_train_epochs "${NUM_EPOCHS}" \
    --learning_rate "${LEARNING_RATE}" \
    --batch_size "${BATCH_SIZE}" \
    --grad_accum "${GRAD_ACCUM}" \
    --seed "${SEED}"

echo ""
echo "============================================================"
echo "Training complete!"
echo "End: $(date)"
echo "Output: ${OUTPUT_DIR}/final_adapter/"
echo "============================================================"
