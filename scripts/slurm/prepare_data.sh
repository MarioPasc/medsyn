#!/bin/bash
#SBATCH -J medsyn_prepare_data
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/prepare_%j.out
#SBATCH --error=logs/prepare_%j.err

# Data Preparation Script for SLURM Cluster
#
# Usage:
#   sbatch scripts/slurm/prepare_data.sh pathmnist
#   sbatch scripts/slurm/prepare_data.sh bloodmnist
#   sbatch scripts/slurm/prepare_data.sh dermamnist
#
# This script downloads and preprocesses MedMNIST datasets,
# creating unified NPZ files for training.

# Configuration
DATASET=${1:-pathmnist}  # Default to pathmnist if no argument
CONDA_ENV=${CONDA_ENV:-medsyn}

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
module load anaconda 2>/dev/null || true

# Activate conda environment
source activate $CONDA_ENV 2>/dev/null || conda activate $CONDA_ENV

# Navigate to project root
cd $SLURM_SUBMIT_DIR/..

# Run data preparation
echo "Preparing $DATASET dataset..."
medsyn-prepare-data --config config/${DATASET}.yaml

echo "======================================"
echo "End time: $(date)"
echo "Job completed!"
echo "======================================"
