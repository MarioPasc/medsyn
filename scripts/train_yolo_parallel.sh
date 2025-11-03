#!/bin/bash
#SBATCH --job-name=yolo_parallel_training
#SBATCH --output=logs/yolo_parallel_%j.out
#SBATCH --error=logs/yolo_parallel_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Professional SLURM script for parallel YOLO classifier training
# This script trains two YOLO classifiers in parallel on 2 GPUs:
#   1. Real dataset only (is_synth=0)
#   2. Real + Synthetic dataset (balanced, is_synth=0 OR is_synth=1)

echo "================================================================================"
echo "YOLO Parallel Training - Job ID: $SLURM_JOB_ID"
echo "================================================================================"
echo "Start time: $(date)"
echo "Hostname: $(hostname)"
echo "Working directory: $(pwd)"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "Number of tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "================================================================================"

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
echo "Activating conda environment..."
source ~/.bashrc
conda activate medsyn

# Configuration files
CONFIG_FILE="config/medsyn_cfg.yaml"
HPARAMS_FILE="config/yolo_hyperparameters.yaml"

# Check if configuration files exist
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$HPARAMS_FILE" ]; then
    echo "ERROR: Hyperparameters file not found: $HPARAMS_FILE"
    exit 1
fi

echo "Configuration files validated."
echo "Config: $CONFIG_FILE"
echo "Hyperparameters: $HPARAMS_FILE"
echo ""

# Function to run training for a specific mode
run_training() {
    local GPU_ID=$1
    local TRAINING_MODE=$2
    local EXP_NAME=$3
    local LOG_FILE=$4

    echo "================================================================================"
    echo "Starting training on GPU $GPU_ID"
    echo "Training mode: $TRAINING_MODE"
    echo "Experiment name: $EXP_NAME"
    echo "Log file: $LOG_FILE"
    echo "================================================================================"

    CUDA_VISIBLE_DEVICES=$GPU_ID python -m medsyn.cli.classify \
        --config "$CONFIG_FILE" \
        --hparams "$HPARAMS_FILE" \
        --training_mode "$TRAINING_MODE" \
        --name "$EXP_NAME" \
        --device "0" \
        --log_level "INFO" \
        > "$LOG_FILE" 2>&1

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "Training on GPU $GPU_ID completed successfully (mode: $TRAINING_MODE)"
        echo "Results saved to experiment: $EXP_NAME"
    else
        echo "ERROR: Training on GPU $GPU_ID failed with exit code $EXIT_CODE (mode: $TRAINING_MODE)"
    fi

    return $EXIT_CODE
}

# Get GPU IDs from CUDA_VISIBLE_DEVICES or default to 0,1
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    GPU0=0
    GPU1=1
else
    IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"
    GPU0=${GPU_ARRAY[0]}
    GPU1=${GPU_ARRAY[1]}
fi

echo "Using GPUs: $GPU0 and $GPU1"
echo ""

# Create timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Training 1: Real dataset only (is_synth=0)
EXP_NAME_REAL="pathmnist_real_only_${TIMESTAMP}"
LOG_FILE_REAL="logs/yolo_real_only_${TIMESTAMP}.log"

# Training 2: Real + Synthetic dataset (balanced)
EXP_NAME_SYNTH="pathmnist_real_synth_${TIMESTAMP}"
LOG_FILE_SYNTH="logs/yolo_real_synth_${TIMESTAMP}.log"

echo "================================================================================"
echo "Launching parallel training jobs..."
echo "================================================================================"

# Run both trainings in parallel using background processes
run_training "$GPU0" "real" "$EXP_NAME_REAL" "$LOG_FILE_REAL" &
PID_REAL=$!

run_training "$GPU1" "real_synth" "$EXP_NAME_SYNTH" "$LOG_FILE_SYNTH" &
PID_SYNTH=$!

echo "Process IDs:"
echo "  - Real-only training (GPU $GPU0): PID $PID_REAL"
echo "  - Real+Synth training (GPU $GPU1): PID $PID_SYNTH"
echo ""

# Wait for both processes to complete
echo "Waiting for training processes to complete..."
echo ""

wait $PID_REAL
EXIT_REAL=$?

wait $PID_SYNTH
EXIT_SYNTH=$?

echo ""
echo "================================================================================"
echo "Training Summary"
echo "================================================================================"
echo "Real-only training (GPU $GPU0):"
if [ $EXIT_REAL -eq 0 ]; then
    echo "  ✓ SUCCESS"
    echo "  Experiment: $EXP_NAME_REAL"
    echo "  Log: $LOG_FILE_REAL"
else
    echo "  ✗ FAILED (exit code: $EXIT_REAL)"
    echo "  Log: $LOG_FILE_REAL"
fi

echo ""
echo "Real+Synth training (GPU $GPU1):"
if [ $EXIT_SYNTH -eq 0 ]; then
    echo "  ✓ SUCCESS"
    echo "  Experiment: $EXP_NAME_SYNTH"
    echo "  Log: $LOG_FILE_SYNTH"
else
    echo "  ✗ FAILED (exit code: $EXIT_SYNTH)"
    echo "  Log: $LOG_FILE_SYNTH"
fi

echo ""
echo "End time: $(date)"
echo "================================================================================"

# Exit with error if any training failed
if [ $EXIT_REAL -ne 0 ] || [ $EXIT_SYNTH -ne 0 ]; then
    echo "ERROR: One or more training runs failed"
    exit 1
fi

echo "All training runs completed successfully!"
exit 0
