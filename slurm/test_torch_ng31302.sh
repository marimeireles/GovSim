#!/bin/bash
#SBATCH --job-name=test-ng31302
#SBATCH --account=def-zhijing
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=slurm/logs/test_ng31302_%j.out
#SBATCH --error=slurm/logs/test_ng31302_%j.err
#SBATCH --nodelist=ng31302

echo "Node: $(hostname) | Date: $(date)"
source /lustre07/scratch/marimeir/govsim/.venv/bin/activate

echo "[1] import torch..."
timeout 60 python -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" && echo "PASS" || echo "FAIL (exit $?)"

echo "[2] import trl..."
timeout 60 python -c "from trl import GRPOConfig; print('TRL OK')" && echo "PASS" || echo "FAIL (exit $?)"

echo "Done: $(date)"
