#!/bin/bash
#SBATCH --job-name=test-torch
#SBATCH --account=def-zhijing
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=slurm/logs/test_torch_%j.out
#SBATCH --error=slurm/logs/test_torch_%j.err

set -eo pipefail

echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Modules loaded:"
module list 2>&1
echo ""

cd /lustre07/scratch/marimeir/govsim
source .venv/bin/activate

echo "Python: $(which python)"
echo "Testing step by step..."

echo "[1] Basic python..."
timeout 30 python -c "print('Python OK')" && echo "PASS" || echo "FAIL"

echo "[2] import torch..."
timeout 60 python -c "import torch; print(f'torch {torch.__version__}')" && echo "PASS" || echo "FAIL"

echo "[3] torch.cuda..."
timeout 60 python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')" && echo "PASS" || echo "FAIL"

echo "[4] import trl..."
timeout 60 python -c "from trl import GRPOConfig; print('TRL OK')" && echo "PASS" || echo "FAIL"

echo "[5] import transformers..."
timeout 60 python -c "from transformers import AutoModelForCausalLM; print('transformers OK')" && echo "PASS" || echo "FAIL"

echo "[6] Full training import..."
timeout 120 python -c "from training.train_grpo import main; print('training module OK')" && echo "PASS" || echo "FAIL"

echo "All tests done: $(date)"
