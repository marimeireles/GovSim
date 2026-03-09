#!/bin/bash
#SBATCH --job-name=govsim-eval
#SBATCH --account=def-zhijing
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/benchmark_finetuned_%j.out
#SBATCH --error=slurm/logs/benchmark_finetuned_%j.err

# ============================================================
# GovSim Benchmark: Fine-tuned gemma3-4b-it (GRPO checkpoint-750)
#
# Step 1: Merge LoRA adapter into base model
# Step 2: Run all 3 scenarios (fishing, sheep, pollution)
#
# Usage:
#   sbatch slurm/benchmark_finetuned.sh
# ============================================================

set -eo pipefail

ADAPTER_PATH="${ADAPTER_PATH:-models/gemma3-4b-cooperative-v2/checkpoint-750}"
MERGED_PATH="${MERGED_PATH:-models/gemma3-4b-cooperative-v2-merged}"
SEED="${SEED:-42}"

echo "============================================================"
echo "Job:       GovSim Fine-tuned Benchmark"
echo "Adapter:   ${ADAPTER_PATH}"
echo "Merged:    ${MERGED_PATH}"
echo "Seed:      ${SEED}"
echo "Date:      $(date)"
echo "Node:      $(hostname)"
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
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
export WANDB_MODE="${WANDB_MODE:-disabled}"

mkdir -p slurm/logs

# --- Step 1: Merge LoRA adapter ---
if [ ! -d "${MERGED_PATH}" ] || [ ! -f "${MERGED_PATH}/model.safetensors.index.json" ]; then
    echo ""
    echo "============================================================"
    echo "Step 1: Merging LoRA adapter into base model"
    echo "Started: $(date)"
    echo "============================================================"

    python -m training.evaluate merge_and_save \
        --adapter_path "${ADAPTER_PATH}" \
        --output_path "${MERGED_PATH}" \
        --base_model "google/gemma-3-4b-it"

    echo "Merge complete: $(date)"
else
    echo "Merged model already exists at ${MERGED_PATH}, skipping merge."
fi

# --- Step 2: Run all 3 benchmark scenarios ---
EXPERIMENTS=(
    fish_baseline_concurrent
    sheep_baseline_concurrent
    pollution_baseline_concurrent
)

for EXP in "${EXPERIMENTS[@]}"; do
    echo ""
    echo "============================================================"
    echo "Running: ${EXP}"
    echo "Started: $(date)"
    echo "============================================================"

    python -m simulation.main \
        experiment="${EXP}" \
        llm.path="${MERGED_PATH}" \
        llm.backend=transformers \
        llm.is_api=false \
        llm.temperature=0.0 \
        seed="${SEED}" \
        group_name="gemma3-4b-finetuned-v2-ckpt750" \
        debug=true

    echo "Finished: ${EXP} at $(date)"
done

echo ""
echo "============================================================"
echo "All benchmarks complete!"
echo "End: $(date)"
echo "Results in: simulation/results/"
echo "============================================================"
