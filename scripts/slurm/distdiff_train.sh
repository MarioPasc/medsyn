#!/bin/bash
#SBATCH -J distdiff_train
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/distdiff_train_%j.out
#SBATCH --error=logs/distdiff_train_%j.err

# DistDiff Guide Model Training Script for SLURM Cluster
#
# Usage:
#   sbatch scripts/slurm/distdiff_train.sh pathmnist
#   sbatch scripts/slurm/distdiff_train.sh bloodmnist
#   sbatch scripts/slurm/distdiff_train.sh dermamnist
#
# Environment setup:
#   - Set CONDA_ENV to your conda environment name
#   - Set DATASET_PATH to the directory containing NPZ files
#   - Set OUTPUT_DIR to the directory for model outputs

# Configuration
DATASET=${1:-pathmnist}  # Default to pathmnist if no argument
CONDA_ENV=${CONDA_ENV:-medsyn}
DATASET_PATH=${DATASET_PATH:-/media/mpascual/Sandisk2TB/research/medsyn}
OUTPUT_DIR=${OUTPUT_DIR:-/media/mpascual/Sandisk2TB/research/medsyn/outputs}

# Create log directory
mkdir -p logs

# Print job info
echo "======================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Dataset: $DATASET"
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

# Run training
echo "Starting DistDiff guide model training for $DATASET..."
distdiff-train config/adapters/distdiff_${DATASET}_train.yaml

echo "======================================"
echo "End time: $(date)"
echo "Job completed!"
echo "======================================"
