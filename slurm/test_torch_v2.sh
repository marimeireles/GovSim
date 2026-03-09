#!/bin/bash
#SBATCH --job-name=test-torch2
#SBATCH --account=def-zhijing
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=slurm/logs/test_torch2_%j.out
#SBATCH --error=slurm/logs/test_torch2_%j.err

echo "Node: $(hostname) | Date: $(date)"

source /lustre07/scratch/marimeir/govsim/.venv/bin/activate

# Disable torch compile cache and triton cache (could hang on Lustre)
export TORCH_COMPILE_DISABLE=1
export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_JOB_ID}
export TORCH_EXTENSIONS_DIR=/tmp/torch_ext_${SLURM_JOB_ID}
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_${SLURM_JOB_ID}

# Disable MPI-based distributed init that openmpi module might trigger
unset PMIX_RANK
unset OMPI_COMM_WORLD_SIZE
unset OMPI_COMM_WORLD_RANK
unset PMI_RANK
unset PMI_SIZE

echo "[1] import torch with caches on local disk..."
timeout 60 python -c "import torch; print(f'torch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" && echo "PASS" || echo "FAIL (exit $?)"

echo "[2] import trl..."
timeout 60 python -c "from trl import GRPOConfig; print('TRL OK')" && echo "PASS" || echo "FAIL (exit $?)"

echo "Done: $(date)"
