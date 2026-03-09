#!/bin/bash
#SBATCH --job-name=govsim-bench
#SBATCH --account=def-zhijing
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm/logs/benchmark_gemma3_%j.out
#SBATCH --error=slurm/logs/benchmark_gemma3_%j.err

# ============================================================
# GovSim Baseline Benchmark: gemma3-4b-it
#
# Runs all 3 scenarios (fishing, sheep, pollution) with
# gemma3-4b-it as all 5 agents. Baseline comparison.
#
# Usage:
#   sbatch slurm/benchmark_gemma3.sh
#   SEED=123 sbatch slurm/benchmark_gemma3.sh
# ============================================================

set -eo pipefail

SEED="${SEED:-42}"
MODEL="${MODEL:-google/gemma-3-4b-it}"

echo "============================================================"
echo "Job:       GovSim Baseline Benchmark"
echo "Model:     ${MODEL}"
echo "Seed:      ${SEED}"
echo "Date:      $(date)"
echo "Node:      $(hostname)"
echo "Job ID:    ${SLURM_JOB_ID:-local}"
echo "GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================================"

# --- Environment ---
cd /lustre07/scratch/marimeir/govsim

# Use micromamba govsim env (conda pytorch + CUDA libs, avoids CC pip conflicts)
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

# --- Run all 3 baseline scenarios ---
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
        group_name="gemma3-4b-it-baseline" \
        debug=true

    echo "Finished: ${EXP} at $(date)"
done

echo ""
echo "============================================================"
echo "All benchmarks complete!"
echo "End: $(date)"
echo "Results in: simulation/results/"
echo "============================================================"
