#!/bin/bash
#SBATCH --job-name=bench-sft-v5
#SBATCH --account=def-zhijing
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/benchmark_sft_v5_%j.out
#SBATCH --error=slurm/logs/benchmark_sft_v5_%j.err

# ============================================================
# GovSim + Concordia Benchmark: sft-v5 (50% general-sum)
#
# Step 1: Merge LoRA adapter
# Step 2: Run GovSim benchmarks (fishing, sheep, pollution)
# Step 3: Run Concordia benchmarks (haggling, pub_coord, multi_item)
# ============================================================

set -eo pipefail

SEED="${SEED:-42}"
ADAPTER_PATH="${ADAPTER_PATH:-models/gemma3-4b-sft-v5/final_adapter}"
MERGED_PATH="${MERGED_PATH:-models/gemma3-4b-sft-v5-merged}"
BASE_MODEL="${BASE_MODEL:-google/gemma-3-4b-it}"
GROUP_NAME="${GROUP_NAME:-sft-v5-bench}"

echo "============================================================"
echo "Job:       GovSim + Concordia Benchmark (sft-v5)"
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
export PYTHONUNBUFFERED=1

mkdir -p slurm/logs

# --- Step 1: Merge LoRA ---
if [ ! -d "${MERGED_PATH}" ] || [ ! -f "${MERGED_PATH}/model.safetensors.index.json" ]; then
    echo ""
    echo "============================================================"
    echo "Step 1: Merging LoRA adapter"
    echo "============================================================"

    python -m training.evaluate merge_and_save \
        --adapter_path "${ADAPTER_PATH}" \
        --output_path "${MERGED_PATH}" \
        --base_model "${BASE_MODEL}"

    echo "Merge complete: $(date)"
else
    echo "Merged model exists at ${MERGED_PATH}, skipping merge."
fi

# --- Step 2: GovSim Benchmarks ---
GOVSIM_EXPERIMENTS=(
    fish_baseline_concurrent
    sheep_baseline_concurrent
    pollution_baseline_concurrent
)

for EXP in "${GOVSIM_EXPERIMENTS[@]}"; do
    echo ""
    echo "============================================================"
    echo "GovSim: ${EXP}"
    echo "Started: $(date)"
    echo "============================================================"

    python -m simulation.main \
        experiment="${EXP}" \
        llm.path="${MERGED_PATH}" \
        llm.backend=transformers \
        llm.is_api=false \
        llm.temperature=0.0 \
        seed="${SEED}" \
        group_name="${GROUP_NAME}" \
        debug=true

    echo "Finished: ${EXP} at $(date)"
done

# --- Step 3: Concordia Benchmarks ---
echo ""
echo "============================================================"
echo "Starting Concordia Benchmarks"
echo "============================================================"

cd /lustre07/scratch/marimeir/concordia

CONCORDIA_SCENARIOS=(
    "haggling/fruitville"
    "pub_coordination/london_mini"
    "haggling_multi_item/fruitville_multi"
)

for SCENARIO in "${CONCORDIA_SCENARIOS[@]}"; do
    echo ""
    echo "============================================================"
    echo "Concordia: ${SCENARIO}"
    echo "Started: $(date)"
    echo "============================================================"

    python bench/run_benchmark.py \
        --model_path "${MERGED_PATH}" \
        --scenario "${SCENARIO}" \
        --model_name "gemma3-4b-sft-v5" \
        2>&1 || echo "WARNING: Concordia ${SCENARIO} failed"

    echo "Finished: ${SCENARIO} at $(date)"
done

echo ""
echo "============================================================"
echo "All benchmarks complete!"
echo "End: $(date)"
echo "============================================================"
