#!/bin/bash
#SBATCH --job-name=test-venvs
#SBATCH --account=def-zhijing
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=slurm/logs/test_venvs_%j.out
#SBATCH --error=slurm/logs/test_venvs_%j.err

echo "Node: $(hostname) | Date: $(date)"

echo ""
echo "=== Test 1: increase-syn venv ==="
source /home/marimeir/scratch/increase-syn/.venv/bin/activate
echo "Python: $(which python)"
timeout 30 python -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" && echo "PASS" || echo "FAIL (exit $?)"
deactivate 2>/dev/null

echo ""
echo "=== Test 2: govsim venv ==="
source /lustre07/scratch/marimeir/govsim/.venv/bin/activate
echo "Python: $(which python)"
timeout 30 python -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" && echo "PASS" || echo "FAIL (exit $?)"
deactivate 2>/dev/null

echo ""
echo "=== Test 3: govsim venv with module purge ==="
module purge --force 2>/dev/null
module load StdEnv/2023 2>/dev/null
source /lustre07/scratch/marimeir/govsim/.venv/bin/activate
echo "Python: $(which python)"
timeout 30 python -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" && echo "PASS" || echo "FAIL (exit $?)"

echo ""
echo "Done: $(date)"
