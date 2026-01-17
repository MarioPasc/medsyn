#!/bin/bash
#SBATCH -J medsyn_generate
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/generate_%j.out
#SBATCH --error=logs/generate_%j.err

# Sample Generation Script for SLURM Cluster
#
# Usage:
#   # Generate with CBDM
#   sbatch scripts/slurm/generate_samples.sh cbdm pathmnist 1000
#
#   # Generate with MedFusion
#   sbatch scripts/slurm/generate_samples.sh medfusion pathmnist 1000
#
# Arguments:
#   $1: Model type (cbdm or medfusion)
#   $2: Dataset name (pathmnist, bloodmnist, dermamnist)
#   $3: Number of samples to generate

# Configuration
MODEL=${1:-cbdm}
DATASET=${2:-pathmnist}
NUM_SAMPLES=${3:-1000}
CONDA_ENV=${CONDA_ENV:-medsyn}
DATASET_PATH=${DATASET_PATH:-/media/mpascual/Sandisk2TB/research/medsyn}
OUTPUT_DIR=${OUTPUT_DIR:-/media/mpascual/Sandisk2TB/research/medsyn/outputs}

# Create log directory
mkdir -p logs

# Print job info
echo "======================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Num Samples: $NUM_SAMPLES"
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
cd $SLURM_SUBMIT_DIR/..

# Run generation
echo "Generating $NUM_SAMPLES samples with $MODEL for $DATASET..."
if [ "$MODEL" == "cbdm" ]; then
    cbdm-generate config/adapters/cbdm_${DATASET}.yaml --num-samples $NUM_SAMPLES
elif [ "$MODEL" == "medfusion" ]; then
    medfusion-generate config/adapters/medfusion_${DATASET}.yaml --num-samples $NUM_SAMPLES
else
    echo "Unknown model: $MODEL"
    exit 1
fi

echo "======================================"
echo "End time: $(date)"
echo "Job completed!"
echo "======================================"
