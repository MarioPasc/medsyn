#!/usr/bin/env bash
#SBATCH -J yolo_classifier_parallel
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:2
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail

# ========================================================================
# YOLO Classifier Parallel Training (Real vs Real+Synth)
# Fixed version addressing all pitfalls identified in code review
# ========================================================================

# Save the launch directory (where .out and .err files are written)
LAUNCH_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
echo "Launch directory: ${LAUNCH_DIR}"

echo "================================================================================"
echo "YOLO Classifier Parallel Training - Job ID: $SLURM_JOB_ID"
echo "================================================================================"
echo "Start time: $(date)"
echo "Hostname: $(hostname)"
echo "GPUs allocated: $CUDA_VISIBLE_DEVICES"
echo "================================================================================"

# ---------- Inputs (UPDATE THESE PATHS FOR YOUR SETUP) ----------
# Path to your NPZ file with merged synthetic data
DATA_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/pathmnist/merged.npz"
REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/medsyn"
RESULTS_DST="/mnt/home/users/tic_163_uma/mpascual/fscratch/results/yolo_classifier"

# ---------- LocalScratch layout ----------
MYLOCALSCRATCH="${LOCALSCRATCH%/}/${USER}/${SLURM_JOB_ID}"
WORKDIR="${MYLOCALSCRATCH}/work"
REPO_DIR="${WORKDIR}/medsyn"
DATA_DIR="${WORKDIR}/datasets"
OUT_DIR="${WORKDIR}/results"

mkdir -p "${WORKDIR}" "${DATA_DIR}" "${OUT_DIR}"
echo "Localscratch workdir: ${WORKDIR}"

# ---------- Error trap: copy logs to launch directory on failure ----------
cleanup_on_error() {
    local exit_code=$?

    # Only run cleanup if there was an error (non-zero exit code)
    if [ $exit_code -ne 0 ]; then
        echo "" >&2
        echo "================================================================================" >&2
        echo "ERROR: Job failed with exit code $exit_code! Copying logs to launch directory..." >&2
        echo "================================================================================" >&2

        # Create emergency logs directory in launch location
        EMERGENCY_LOG_DIR="${LAUNCH_DIR}/emergency_logs_${SLURM_JOB_ID}"
        mkdir -p "${EMERGENCY_LOG_DIR}"

        # Copy all logs and results from localscratch
        if [ -d "${OUT_DIR}" ]; then
            echo "Copying logs from: ${OUT_DIR}" >&2
            rsync -av "${OUT_DIR}/" "${EMERGENCY_LOG_DIR}/" 2>&1 || true
            echo "Emergency logs saved to: ${EMERGENCY_LOG_DIR}" >&2
        fi

        # Also sync to permanent storage if possible
        if [ -d "${RESULTS_DST}" ]; then
            mkdir -p "${RESULTS_DST}"
            rsync -av "${OUT_DIR}/" "${RESULTS_DST}/" 2>&1 || true
            echo "Also synced to: ${RESULTS_DST}" >&2
        fi

        echo "================================================================================" >&2
    fi
}

trap cleanup_on_error EXIT

# ---------- 1) Repo to localscratch ----------
if [ -d "${REPO_DIR}" ] && [ -f "${REPO_DIR}/pyproject.toml" ]; then
  echo "[repo] found at ${REPO_DIR}. skip copy."
else
  echo "[repo] copying from ${REPO_SRC} ..."
  mkdir -p "${REPO_DIR}"
  rsync -a "${REPO_SRC}/" "${REPO_DIR}/"
fi

# ---------- 2) Dataset to localscratch ----------
DATA_DST="${DATA_DIR}/$(basename ${DATA_SRC})"
if [ -f "${DATA_DST}" ]; then
  echo "[data] found at ${DATA_DST}. skip copy."
else
  echo "[data] copying ${DATA_SRC} -> ${DATA_DST}"
  rsync -a "${DATA_SRC}" "${DATA_DST}"
fi

# ---------- 3) Load conda module and activate env ----------
module_loaded=0
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
  if module avail 2>/dev/null | grep -qi "^${m}[[:space:]]"; then
    module load "$m" && module_loaded=1 && break
  fi
done

if [ "$module_loaded" -eq 0 ]; then
  echo "[env] no conda module loaded; assuming conda already in PATH."
fi

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate medsyn 2>/dev/null || source activate medsyn
else
  source activate medsyn
fi

echo "[python] $(which python)"
python -c "import sys; print('Python', sys.version.split()[0])"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('GPU count:', torch.cuda.device_count())"

# ---------- 4) Change to repo directory ----------
cd "${REPO_DIR}"
echo "[working directory] $(pwd)"

# ---------- 5) Set environment variables (FIX FOR P1.1) ----------
export DATASET_PATH="${DATA_DST}"
export OUTPUT_DIR="${OUT_DIR}"
echo "[env] DATASET_PATH=${DATASET_PATH}"
echo "[env] OUTPUT_DIR=${OUTPUT_DIR}"

# Verify NPZ structure
echo ""
echo "Verifying NPZ structure..."
python -c "
import numpy as np
z = np.load('${DATA_DST}')
print('NPZ keys:', list(z.keys()))
for k in sorted(z.keys()):
    if k.endswith('_images'):
        split = k.split('_')[0]
        print(f'{split}:')
        print(f'  images: {z[k].shape} {z[k].dtype}')
        print(f'  labels: {z[split+\"_labels\"].shape} {z[split+\"_labels\"].dtype}')
        is_synth = z.get(split+'_is_synth', None)
        if is_synth is not None:
            print(f'  is_synth: {is_synth.shape} {is_synth.dtype}')
            print(f'    Real: {(~is_synth).sum()}, Synth: {is_synth.sum()}')
"
echo ""

# ---------- 6) Configuration files ----------
CONFIG_FILE="config/picasso_cfg.yaml"
HPARAMS_FILE="config/yolo_hyperparameters.yaml"

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

# ---------- 7) Update config to fix project path (FIX FOR P1.2) ----------
echo ""
echo "Updating YAML config with runtime paths..."
CONFIG_BACKUP="${CONFIG_FILE}.backup_${SLURM_JOB_ID}"
cp "${CONFIG_FILE}" "${CONFIG_BACKUP}"

# Use sed to update the yolo project path
sed -i "s|project:.*|project: ${OUT_DIR}|" "${CONFIG_FILE}"
echo "Updated yolo.project to: ${OUT_DIR}"

# ---------- 8) Display GPU information ----------
echo ""
echo "================================================================================"
echo "GPU Information"
echo "================================================================================"
nvidia-smi || true
echo ""

# ---------- 9) Function to run training ----------
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

    # Run training, saving logs to file AND stderr to .err file
    # Use 'tee' to duplicate stderr so it appears in both the log and SLURM .err
    CUDA_VISIBLE_DEVICES=$GPU_ID python -m medsyn.cli.classify \
        --config "$CONFIG_FILE" \
        --hparams "$HPARAMS_FILE" \
        --training_mode "$TRAINING_MODE" \
        --name "$EXP_NAME" \
        --device "$GPU_ID" \
        --log_level "INFO" \
        > "$LOG_FILE" 2> >(tee -a "$LOG_FILE" >&2)

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Training on GPU $GPU_ID completed successfully (mode: $TRAINING_MODE)"
    else
        echo "✗ Training on GPU $GPU_ID failed with exit code $EXIT_CODE (mode: $TRAINING_MODE)" >&2
        echo "✗ Training on GPU $GPU_ID failed with exit code $EXIT_CODE (mode: $TRAINING_MODE)"
        echo "Log file: $LOG_FILE" >&2
    fi

    return $EXIT_CODE
}

# ---------- 10) Launch parallel training jobs ----------
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Real-only training
EXP_NAME_REAL="pathmnist_real_only_${TIMESTAMP}"
LOG_FILE_REAL="${OUT_DIR}/yolo_real_only_${TIMESTAMP}.log"

# Real+Synth training
EXP_NAME_SYNTH="pathmnist_real_synth_${TIMESTAMP}"
LOG_FILE_SYNTH="${OUT_DIR}/yolo_real_synth_${TIMESTAMP}.log"

echo ""
echo "================================================================================"
echo "Launching Parallel Training Jobs"
echo "================================================================================"
echo "Experiment 1 (Real-only): $EXP_NAME_REAL"
echo "Experiment 2 (Real+Synth): $EXP_NAME_SYNTH"
echo "================================================================================"
echo ""

# Launch both trainings in parallel
run_training "0" "real" "$EXP_NAME_REAL" "$LOG_FILE_REAL" &
PID_REAL=$!

run_training "1" "real_synth" "$EXP_NAME_SYNTH" "$LOG_FILE_SYNTH" &
PID_SYNTH=$!

echo "Process IDs:"
echo "  - Real-only training (GPU 0): PID $PID_REAL"
echo "  - Real+Synth training (GPU 1): PID $PID_SYNTH"
echo ""

# Wait for both processes
echo "Waiting for training processes to complete..."
wait $PID_REAL
EXIT_REAL=$?

wait $PID_SYNTH
EXIT_SYNTH=$?

# ---------- 11) Restore config backup ----------
echo ""
echo "Restoring original config..."
mv "${CONFIG_BACKUP}" "${CONFIG_FILE}"

# ---------- 12) Training summary ----------
echo ""
echo "================================================================================"
echo "Training Summary"
echo "================================================================================"
echo "Real-only training:"
if [ $EXIT_REAL -eq 0 ]; then
    echo "  ✓ SUCCESS"
    echo "  Experiment: $EXP_NAME_REAL"
    echo "  Log: $LOG_FILE_REAL"
else
    echo "  ✗ FAILED (exit code: $EXIT_REAL)"
    echo "  Check log: $LOG_FILE_REAL"
fi

echo ""
echo "Real+Synth training:"
if [ $EXIT_SYNTH -eq 0 ]; then
    echo "  ✓ SUCCESS"
    echo "  Experiment: $EXP_NAME_SYNTH"
    echo "  Log: $LOG_FILE_SYNTH"
else
    echo "  ✗ FAILED (exit code: $EXIT_SYNTH)"
    echo "  Check log: $LOG_FILE_SYNTH"
fi

# ---------- 13) Sync results back to permanent storage (FIX FOR P3.5) ----------
echo ""
echo "================================================================================"
echo "Syncing Results to Permanent Storage"
echo "================================================================================"

mkdir -p "${RESULTS_DST}"
echo "Target: ${RESULTS_DST}"

# Find YOLO results in output directory
if [ -d "${OUT_DIR}" ]; then
    echo "Syncing from: ${OUT_DIR}"
    rsync -av "${OUT_DIR}/" "${RESULTS_DST}/"
    echo "✓ Results synced successfully"

    # List what was saved
    echo ""
    echo "Saved experiments:"
    find "${RESULTS_DST}" -type d -name "*${TIMESTAMP}*" -maxdepth 2
else
    echo "⚠️  Warning: Output directory not found: ${OUT_DIR}"
fi

echo ""
echo "================================================================================"
echo "End time: $(date)"
echo "================================================================================"

# ---------- 14) Cleanup ----------
echo ""
echo "Cleaning up localscratch..."
if cd "${LOCALSCRATCH%/}/${USER}"; then
  [ -n "${MYLOCALSCRATCH:-}" ] && rm -rf --one-file-system "${MYLOCALSCRATCH}"
  echo "✓ Cleanup complete"
fi

# Exit with error if any training failed
if [ $EXIT_REAL -ne 0 ] || [ $EXIT_SYNTH -ne 0 ]; then
    echo ""
    echo "ERROR: One or more training runs failed"
    # Trap will handle copying logs on exit
    exit 1
fi

echo ""
echo "================================================================================"
echo "All training runs completed successfully!"
echo "Results saved to: ${RESULTS_DST}"
echo "================================================================================"

exit 0
