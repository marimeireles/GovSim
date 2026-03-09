#!/bin/bash
#SBATCH --job-name=bench-sft-v4
#SBATCH --account=def-zhijing
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/benchmark_sft_v4_%j.out
#SBATCH --error=slurm/logs/benchmark_sft_v4_%j.err

# ============================================================
# GovSim Benchmark: sft-v4 (mixed dataset, 3 epochs, 2e-5 LR)
# ============================================================

set -eo pipefail

SEED="${SEED:-42}"
MODEL="${MODEL:-models/gemma3-4b-sft-v4-merged}"

echo "============================================================"
echo "Job:       GovSim Benchmark (sft-v4)"
echo "Model:     ${MODEL}"
echo "Seed:      ${SEED}"
echo "Date:      $(date)"
echo "Node:      $(hostname)"
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================================"

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
export PYTHONUNBUFFERED=1

mkdir -p slurm/logs

# --- Step 0: Merge LoRA if needed ---
ADAPTER_PATH="${ADAPTER_PATH:-models/gemma3-4b-sft-v4/final_adapter}"
if [ ! -d "${MODEL}" ] || [ ! -f "${MODEL}/model.safetensors.index.json" ]; then
    echo ""
    echo "============================================================"
    echo "Merging LoRA adapter into base model"
    echo "============================================================"

    python -m training.evaluate merge_and_save \
        --adapter_path "${ADAPTER_PATH}" \
        --output_path "${MODEL}" \
        --base_model "google/gemma-3-4b-it"

    echo "Merge complete: $(date)"
else
    echo "Merged model exists at ${MODEL}, skipping merge."
fi

# --- Run all 3 GovSim scenarios ---
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
        llm.path="${MODEL}" \
        llm.backend=transformers \
        llm.is_api=false \
        llm.temperature=0.0 \
        seed="${SEED}" \
        group_name="sft-v4-bench" \
        debug=true

    echo "Finished: ${EXP} at $(date)"
done

echo ""
echo "============================================================"
echo "All GovSim benchmarks complete!"
echo "End: $(date)"
echo "Results in: simulation/results/"
echo "============================================================"
