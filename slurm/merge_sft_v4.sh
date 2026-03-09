#!/bin/bash
#SBATCH --job-name=merge-sft-v4
#SBATCH --account=def-zhijing
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/merge_sft_v4_%j.out
#SBATCH --error=slurm/logs/merge_sft_v4_%j.err

set -eo pipefail

cd /lustre07/scratch/marimeir/govsim

eval "$(micromamba shell hook -s bash)"
micromamba activate govsim

export PYTHONPATH="./"
export HF_HOME="/lustre07/scratch/marimeir/huggingface_cache"
export HUGGINGFACE_HUB_CACHE="/lustre07/scratch/marimeir/huggingface_cache/hub"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1

ADAPTER_PATH="${ADAPTER_PATH:-models/gemma3-4b-sft-v4/final_adapter}"
OUTPUT_PATH="${OUTPUT_PATH:-models/gemma3-4b-sft-v4-merged}"
BASE_MODEL="${BASE_MODEL:-google/gemma-3-4b-it}"

echo "Merging LoRA adapter into base model..."
echo "Adapter: ${ADAPTER_PATH}"
echo "Base:    ${BASE_MODEL}"
echo "Output:  ${OUTPUT_PATH}"

python -m training.evaluate merge_and_save \
    --adapter_path "${ADAPTER_PATH}" \
    --output_path "${OUTPUT_PATH}" \
    --base_model "${BASE_MODEL}"

echo "Merge complete: $(date)"
ls -la "${OUTPUT_PATH}/"
