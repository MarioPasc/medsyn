#!/bin/bash
#SBATCH -J distdiff_generate
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/distdiff_generate_%j.out
#SBATCH --error=logs/distdiff_generate_%j.err

# DistDiff Synthetic Image Generation Script for SLURM Cluster
#
# Usage:
#   # Generate for PathMNIST
#   sbatch scripts/slurm/distdiff_generate.sh pathmnist
#
#   # Generate for BloodMNIST
#   sbatch scripts/slurm/distdiff_generate.sh bloodmnist
#
#   # Generate for DermaMNIST
#   sbatch scripts/slurm/distdiff_generate.sh dermamnist
#
#   # Generate with specific split (for multi-GPU parallel execution)
#   SPLIT=0 TOTAL_SPLIT=4 sbatch scripts/slurm/distdiff_generate.sh pathmnist
#
# Environment setup:
#   - Set CONDA_ENV to your conda environment name
#   - Set DATASET_PATH to the directory containing NPZ files
#   - Set OUTPUT_DIR to the directory for outputs
#
# Prerequisites:
#   - Train the guide model first using distdiff_train.sh

# Configuration
DATASET=${1:-pathmnist}  # Default to pathmnist if no argument
CONDA_ENV=${CONDA_ENV:-medsyn}
DATASET_PATH=${DATASET_PATH:-/media/mpascual/Sandisk2TB/research/medsyn}
OUTPUT_DIR=${OUTPUT_DIR:-/media/mpascual/Sandisk2TB/research/medsyn/outputs}
SPLIT=${SPLIT:-0}
TOTAL_SPLIT=${TOTAL_SPLIT:-1}

# Create log directory
mkdir -p logs

# Print job info
echo "======================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Dataset: $DATASET"
echo "Split: $SPLIT / $TOTAL_SPLIT"
echo "Start time: $(date)"
echo "======================================"

# Load modules (adjust for your cluster)
module load anaconda cuda 2>/dev/null || true

# Activate conda environment
source activate $CONDA_ENV 2>/dev/null || conda activate $CONDA_ENV

# Export paths
export DATASET_PATH
export OUTPUT_DIR

# Navigate to project root
cd $SLURM_SUBMIT_DIR/../..

# Run generation
echo "Starting DistDiff synthetic image generation for $DATASET..."
if [ "$TOTAL_SPLIT" -gt 1 ]; then
    distdiff-generate config/adapters/distdiff_${DATASET}_generate.yaml --split $SPLIT --total-split $TOTAL_SPLIT
else
    distdiff-generate config/adapters/distdiff_${DATASET}_generate.yaml
fi

echo "======================================"
echo "End time: $(date)"
echo "Job completed!"
echo "======================================"
