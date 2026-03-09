#!/bin/bash
#SBATCH --job-name=merge-sft-grpo
#SBATCH --account=def-zhijing
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm/logs/merge_sft_grpo_%j.out
#SBATCH --error=slurm/logs/merge_sft_grpo_%j.err

set -eo pipefail

cd /lustre07/scratch/marimeir/govsim

eval "$(micromamba shell hook -s bash)"
micromamba activate govsim

export PYTHONPATH="./"
export HF_HOME="/lustre07/scratch/marimeir/huggingface_cache"
export HUGGINGFACE_HUB_CACHE="/lustre07/scratch/marimeir/huggingface_cache/hub"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

echo "Merging GRPO adapter into SFT-merged base..."
python -m training.evaluate merge_and_save \
    --adapter_path "models/gemma3-4b-sft-grpo-v1/final_adapter" \
    --output_path "models/gemma3-4b-sft-grpo-v1-merged" \
    --base_model "models/gemma3-4b-sft-v1-merged"

echo "Merge complete: $(date)"
ls -la models/gemma3-4b-sft-grpo-v1-merged/
