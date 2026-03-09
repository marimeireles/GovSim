#!/bin/bash
#SBATCH --job-name=govsim-eval-v3
#SBATCH --account=def-zhijing
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/benchmark_sft_grpo_%j.out
#SBATCH --error=slurm/logs/benchmark_sft_grpo_%j.err

# ============================================================
# GovSim Benchmark: SFT+GRPO fine-tuned gemma3-4b-it
#
# Merges the GRPO adapter (trained on SFT-merged base) and
# runs all 3 GovSim scenarios.
#
# Usage:
#   sbatch slurm/benchmark_sft_grpo.sh
# ============================================================

set -eo pipefail

# The GRPO adapter is on top of the SFT-merged model
SFT_MERGED="${SFT_MERGED:-models/gemma3-4b-sft-v1-merged}"
GRPO_ADAPTER="${GRPO_ADAPTER:-models/gemma3-4b-sft-grpo-v1/final_adapter}"
FINAL_MERGED="${FINAL_MERGED:-models/gemma3-4b-sft-grpo-v1-merged}"
GROUP_NAME="${GROUP_NAME:-gemma3-4b-sft-grpo-v1}"
SEED="${SEED:-42}"

echo "============================================================"
echo "Job:         GovSim SFT+GRPO Benchmark"
echo "SFT merged:  ${SFT_MERGED}"
echo "GRPO adapter: ${GRPO_ADAPTER}"
echo "Final merged: ${FINAL_MERGED}"
echo "Group:       ${GROUP_NAME}"
echo "Date:        $(date)"
echo "Node:        $(hostname)"
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "GPU:         $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
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

# --- Step 1: Merge GRPO adapter into SFT-merged base ---
if [ ! -d "${FINAL_MERGED}" ] || [ ! -f "${FINAL_MERGED}/model.safetensors.index.json" ]; then
    echo ""
    echo "============================================================"
    echo "Step 1: Merging GRPO adapter into SFT-merged base"
    echo "Started: $(date)"
    echo "============================================================"

    python -m training.evaluate merge_and_save \
        --adapter_path "${GRPO_ADAPTER}" \
        --output_path "${FINAL_MERGED}" \
        --base_model "${SFT_MERGED}"

    echo "Merge complete: $(date)"
else
    echo "Final merged model already exists at ${FINAL_MERGED}, skipping merge."
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
        llm.path="${FINAL_MERGED}" \
        llm.backend=transformers \
        llm.is_api=false \
        llm.temperature=0.0 \
        seed="${SEED}" \
        group_name="${GROUP_NAME}" \
        debug=true

    echo "Finished: ${EXP} at $(date)"
done

echo ""
echo "============================================================"
echo "All GovSim benchmarks complete!"
echo "End: $(date)"
echo "Results in: simulation/results/"
echo "============================================================"
