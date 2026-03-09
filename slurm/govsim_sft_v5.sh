#!/bin/bash
#SBATCH --job-name=govsim-sft-v5
#SBATCH --account=def-zhijing
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/sft_v5_%j.out
#SBATCH --error=slurm/logs/sft_v5_%j.err
#SBATCH --exclude=ng31004

# ============================================================
# GovSim SFT v5: 50% GovSim + 50% General-Sum Game Traces
#
# Uses combined general traces (cooperative + GSG) for broader
# coverage of negotiation, coordination, and social reasoning.
# ============================================================

set -eo pipefail

SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-models/gemma3-4b-sft-v5}"
DATASET_PATH="${DATASET_PATH:-training/traces/generated_traces_1k.json}"
GENERAL_TRACES_PATH="${GENERAL_TRACES_PATH:-training/traces/general_combined.json}"
MIX_RATIO="${MIX_RATIO:-0.5}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"

echo "============================================================"
echo "Job:       GovSim SFT v5 (50% general-sum)"
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

eval "$(micromamba shell hook -s bash)"
micromamba activate govsim

export HF_HOME="/lustre07/scratch/marimeir/huggingface_cache"
export HUGGINGFACE_HUB_CACHE="/lustre07/scratch/marimeir/huggingface_cache/hub"
export HF_TOKEN=$(cat /home/marimeir/.cache/huggingface/token 2>/dev/null)
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export HF_DATASETS_OFFLINE=1

if [ -f .env ]; then
    source .env
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p slurm/logs

# --- Verify prerequisites ---
if [ ! -f "${DATASET_PATH}" ]; then
    echo "ERROR: GovSim dataset not found: ${DATASET_PATH}"
    exit 1
fi
echo "GovSim dataset: ${DATASET_PATH} ($(wc -c < "${DATASET_PATH}") bytes)"

if [ ! -f "${GENERAL_TRACES_PATH}" ]; then
    echo "ERROR: General traces not found: ${GENERAL_TRACES_PATH}"
    exit 1
fi
echo "General traces: ${GENERAL_TRACES_PATH} ($(wc -c < "${GENERAL_TRACES_PATH}") bytes)"

# --- Run Training ---
echo ""
echo "============================================================"
echo "Starting SFT v5 Training"
echo "Started: $(date)"
echo "============================================================"

python -m training.train_sft \
    --dataset_path "${DATASET_PATH}" \
    --general_traces_path "${GENERAL_TRACES_PATH}" \
    --mix_ratio "${MIX_RATIO}" \
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
