#!/bin/bash
#SBATCH --job-name=govsim-sft-grpo-v2
#SBATCH --account=def-zhijing
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/govsim_sft_grpo_v2_%j.out
#SBATCH --error=slurm/logs/govsim_sft_grpo_v2_%j.err
#SBATCH --exclude=ng31004

# ============================================================
# Two-Stage Training v2: SFT → GRPO
#
# Fixes from v1:
# - Prompt format matches evaluation exactly (user-only, no system msg)
# - ~1000 traces (333 per scenario)
# ============================================================

set -eo pipefail

SEED="${SEED:-42}"
DATASET_PATH="${DATASET_PATH:-training/traces/generated_traces_1k.json}"
SFT_OUTPUT="${SFT_OUTPUT:-models/gemma3-4b-sft-v2}"
GRPO_OUTPUT="${GRPO_OUTPUT:-models/gemma3-4b-sft-grpo-v2}"
SFT_EPOCHS="${SFT_EPOCHS:-3}"
SFT_LR="${SFT_LR:-2e-5}"
GRPO_STEPS="${GRPO_STEPS:-500}"
GRPO_LR="${GRPO_LR:-5e-7}"

echo "============================================================"
echo "Job:       GovSim SFT → GRPO Pipeline v2"
echo "Date:      $(date)"
echo "Node:      $(hostname)"
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Dataset:   ${DATASET_PATH}"
echo "SFT out:   ${SFT_OUTPUT} (${SFT_EPOCHS} epochs, lr=${SFT_LR})"
echo "GRPO out:  ${GRPO_OUTPUT} (${GRPO_STEPS} steps, lr=${GRPO_LR})"
echo "Seed:      ${SEED}"
echo "============================================================"

# --- Environment ---
cd /lustre07/scratch/marimeir/govsim

eval "$(micromamba shell hook -s bash)"
micromamba activate govsim

export PYTHONPATH="./"
export HF_HOME="/lustre07/scratch/marimeir/huggingface_cache"
export HUGGINGFACE_HUB_CACHE="/lustre07/scratch/marimeir/huggingface_cache/hub"
export HF_TOKEN=$(cat /home/marimeir/.cache/huggingface/token 2>/dev/null)
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

if [ -f .env ]; then
    source .env
fi

mkdir -p "${SFT_OUTPUT}" "${GRPO_OUTPUT}" slurm/logs

# --- Verify prerequisites ---
if [ ! -f "${DATASET_PATH}" ]; then
    echo "ERROR: Dataset not found: ${DATASET_PATH}"
    exit 1
fi
echo "Dataset found: ${DATASET_PATH} ($(wc -c < "${DATASET_PATH}") bytes)"

# ============================================================
# Stage 1: SFT
# ============================================================
echo ""
echo "============================================================"
echo "Stage 1: Supervised Fine-Tuning"
echo "Started: $(date)"
echo "============================================================"

python -m training.train_sft \
    --dataset_path "${DATASET_PATH}" \
    --output_dir "${SFT_OUTPUT}" \
    --model_name "google/gemma-3-4b-it" \
    --num_train_epochs "${SFT_EPOCHS}" \
    --learning_rate "${SFT_LR}" \
    --batch_size 1 \
    --grad_accum 8 \
    --max_length 4096 \
    --seed "${SEED}"

echo "SFT complete: $(date)"

# ============================================================
# Stage 2: Merge SFT adapter, then GRPO
# ============================================================
echo ""
echo "============================================================"
echo "Stage 2a: Merging SFT adapter into base model"
echo "Started: $(date)"
echo "============================================================"

SFT_MERGED="${SFT_OUTPUT}-merged"
if [ ! -d "${SFT_MERGED}" ] || [ ! -f "${SFT_MERGED}/model.safetensors.index.json" ]; then
    python -m training.evaluate merge_and_save \
        --adapter_path "${SFT_OUTPUT}/final_adapter" \
        --output_path "${SFT_MERGED}" \
        --base_model "google/gemma-3-4b-it"
    echo "Merge complete: $(date)"
else
    echo "Merged SFT model already exists at ${SFT_MERGED}, skipping merge."
fi

echo ""
echo "============================================================"
echo "Stage 2b: GRPO on SFT-merged model"
echo "Started: $(date)"
echo "============================================================"

python -m training.train_grpo \
    --dataset_path "${DATASET_PATH}" \
    --output_dir "${GRPO_OUTPUT}" \
    --model_name "${SFT_MERGED}" \
    --max_steps "${GRPO_STEPS}" \
    --num_generations 8 \
    --learning_rate "${GRPO_LR}" \
    --batch_size 1 \
    --grad_accum 4 \
    --max_completion_length 2048 \
    --seed "${SEED}"

echo ""
echo "============================================================"
echo "All training complete!"
echo "End: $(date)"
echo "SFT adapter:    ${SFT_OUTPUT}/final_adapter/"
echo "SFT merged:     ${SFT_MERGED}/"
echo "GRPO adapter:   ${GRPO_OUTPUT}/final_adapter/"
echo "============================================================"
