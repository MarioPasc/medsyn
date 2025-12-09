#!/usr/bin/env bash
#SBATCH -J log_gamma1_temp10
#SBATCH --time=1-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail

# ========================================================================
# DISTRIBUTED TRAINING CONFIGURATION
# ========================================================================

# ---------- Inputs ----------
EXPERIMENT_NAME="gamma1_temp10_training"

DATA_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/pathmnist/PathMNIST.npz"
REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/medsyn"
RESULTS_DST="/mnt/home/users/tic_163_uma/mpascual/fscratch/results/${EXPERIMENT_NAME}"

# Relative to REPO_SRC
CONFIG_FILE="experiments/gamma1_temp10/gamma1_temp10.yaml"
CONFIG_BACKUP="experiments/gamma1_temp10/gamma1_temp10.yaml.backup"

# Supercomputer mode: Disable tqdm, use structured logging for .out/.err files
export IS_SUPERCOMPUTER=1  # Enables clean logging optimized for batch jobs

# Important: Set CUDA environment variables for multi-GPU
# Hardcoded number of GPUs for this job (must match --gres=gpu:N above)
NUM_GPUS=1
MEDSYN_DEBUG_DATALOADER=1
export CUDA_VISIBLE_DEVICES=0  # Assuming 1 GPU allocated by SLURM
export NCCL_DEBUG=INFO  # For debugging distributed training (optional)

echo "================================================================================"
echo "🚀 Multi-GPU Distributed Training with $NUM_GPUS GPUs"
echo "================================================================================"


# ---------- LocalScratch layout ----------
MYLOCALSCRATCH="${LOCALSCRATCH%/}/${USER}/${SLURM_JOB_ID}"
WORKDIR="${MYLOCALSCRATCH}/work"
REPO_DIR="${WORKDIR}/medsyn"
DATA_DIR="${WORKDIR}/datasets"
OUT_DIR="${WORKDIR}/results"

mkdir -p "${WORKDIR}" "${DATA_DIR}" "${OUT_DIR}"
echo "Localscratch workdir: ${WORKDIR}"

# ---------- 1) Repo to localscratch ----------
echo "[repo] copying from ${REPO_SRC} ..."
mkdir -p "${REPO_DIR}"
rsync -a "${REPO_SRC}/" "${REPO_DIR}/"

# ---------- 2) Dataset to localscratch ----------
DATA_DST="${DATA_DIR}/PathMNIST.npz"
echo "[data] copying ${DATA_SRC} -> ${DATA_DST}"
rsync -a "${DATA_SRC}" "${DATA_DST}"

# ---------- 3) Load conda module and activate prebuilt env ----------
# Try common module names seen on clusters
module_loaded=0
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
  if module avail 2>/dev/null | grep -qi "^${m}[[:space:]]"; then
    module load "$m" && module_loaded=1 && break
  fi
done
# If environment module is not needed because conda is already in PATH, continue
if [ "$module_loaded" -eq 0 ]; then
  echo "[env] no conda module loaded; assuming conda already in PATH."
fi

# Activate your existing env named 'medsyn' (precreated by you, offline)
# Support both old and new activation methods
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate medsyn 2>/dev/null || source activate medsyn
else
  # Fallback if only 'source activate' exists in module
  source activate medsyn
fi

# Verify
echo "[python] $(which python || true)"
python -c "import sys; print('Python', sys.version.split()[0])"
python -c "import torch, os; print('CUDA', torch.cuda.is_available())"
echo "[torch] $(python -c 'import torch; print(torch.__version__)')"
echo "[cuda devices] $(python -c 'import torch; print(torch.cuda.device_count())')"

# ---------- 4) Modify config to enable distributed training ----------
cd "${REPO_DIR}"

echo ""
echo "================================================================================"
echo "⚙️  Configuring Distributed Training in YAML"
echo "================================================================================"

# Backup original config
cp "${CONFIG_FILE}" "${CONFIG_BACKUP}"
echo "[config] Backed up original config to ${CONFIG_BACKUP}"

# Modify the config file to enable distributed training
# We use sed to replace the dist.enabled and dist.num_gpus values
echo "[config] Enabling distributed training with ${NUM_GPUS} GPUs..."

# Enable distributed training: change 'enabled: false' to 'enabled: true' in the dist section
# This assumes the dist section exists (which it does after our previous changes)
sed -i '/ccddpm:/,/^[^ ]/ {
  /dist:/,/^[[:space:]]*[a-z]/ {
    s/enabled: false/enabled: true/
    s/num_gpus: .*/num_gpus: '"${NUM_GPUS}"'/
  }
}' "${CONFIG_FILE}"

echo "[config] Modified ${CONFIG_FILE}:"
echo "  • dist.enabled: true"
echo "  • dist.num_gpus: ${NUM_GPUS}"

# Display the modified dist section for verification
echo ""
echo "[config] Verifying dist configuration in YAML:"
grep -A 8 "^  dist:" "${CONFIG_FILE}" || echo "  (Could not extract dist section for display)"
echo ""

# ---------- 5) Set environment variables ----------
export DATASET_PATH="${DATA_DST}"
export OUTPUT_DIR="${OUT_DIR}"

# ---------- 6) Display GPU information ----------
echo ""
echo "================================================================================"
echo "🖥️  GPU Information"
echo "================================================================================"
nvidia-smi || true
echo ""

# ---------- 7) Run distributed training with torchrun ----------
echo ""
echo "================================================================================"
echo "🏋️  Starting Distributed Training with torchrun"
echo "================================================================================"
echo "[torchrun] Command: torchrun --standalone --nnodes=1 --nproc_per_node=${NUM_GPUS}"
echo "[training] Config: ${CONFIG_FILE}"
echo "[training] Dataset: ${DATA_DST}"
echo "[training] Output: ${OUT_DIR}"
echo ""

# Use torchrun for distributed training
# torchrun automatically sets up the distributed environment (RANK, WORLD_SIZE, etc.)
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NUM_GPUS}" \
  -m medsyn.cli.train_ccDDPM \
  --config "${CONFIG_FILE}" \
  --dataset "${DATA_DST}" \
  --outdir "${OUT_DIR}"

TRAIN_EXIT_CODE=$?

echo ""
echo "================================================================================"
echo "Training completed with exit code: ${TRAIN_EXIT_CODE}"
echo "================================================================================"
echo ""

# ---------- 8) Restore original config ----------
echo "[config] Restoring original config from backup..."
mv "${CONFIG_BACKUP}" "${CONFIG_FILE}"
echo "[config] Original config restored"

# ---------- 9) Sync results back ----------
if [ ${TRAIN_EXIT_CODE} -eq 0 ]; then
  echo ""
  echo "================================================================================"
  echo "📦 Syncing Results Back to Permanent Storage"
  echo "================================================================================"
  echo "[sync] copying results back to ${RESULTS_DST}"
  mkdir -p "${RESULTS_DST}"

  # Debug: Show current working directory and environment
  echo "[debug] Current directory: $(pwd)"
  echo "[debug] WORKDIR=${WORKDIR}"
  echo "[debug] REPO_DIR=${REPO_DIR}"
  echo "[debug] OUT_DIR=${OUT_DIR}"
  echo ""

  # Debug: List what's actually in key directories
  echo "[debug] Checking WORKDIR structure:"
  ls -la "${WORKDIR}" 2>/dev/null || echo "  (not accessible)"
  echo ""

  echo "[debug] Checking REPO_DIR structure:"
  ls -la "${REPO_DIR}" 2>/dev/null || echo "  (not accessible)"
  echo ""

  echo "[debug] Looking for outputs/ in REPO_DIR:"
  if [ -d "${REPO_DIR}/outputs" ]; then
    echo "  Found! Contents:"
    find "${REPO_DIR}/outputs" -type d -maxdepth 3 2>/dev/null | head -20
  else
    echo "  ${REPO_DIR}/outputs does not exist"
  fi
  echo ""

  # Known candidates where medsyn ccDDPM writes outputs
  # Priority order: explicit CLI outdir > default locations
  CANDIDATES=(
    "${OUT_DIR}"                                      # CLI --outdir (highest priority)
    "${REPO_DIR}/outputs/ccddpm"                      # Default from config: ./outputs/ccddpm
    "${REPO_DIR}/outputs"                             # Parent outputs directory
    "${REPO_DIR}/samples/ccddpm"                      # Inference/generation outputs
    "${REPO_DIR}/samples"                             # Inference parent directory
    "${WORKDIR}/outputs/ccddpm"                       # Workdir-relative path
    "${WORKDIR}/outputs"                              # Workdir outputs parent
  )

  echo "[sync] Searching for outputs in candidate directories..."
  found_any=0
  for src in "${CANDIDATES[@]}"; do
    echo "[sync] Checking: ${src}"
    if [ -d "${src}" ]; then
      echo "  → Directory exists!"

      # List contents for debugging
      echo "  → Contents:"
      ls -la "${src}" 2>/dev/null | head -10
      echo ""

      # Check for typical ccDDPM output structure
      if [ -d "${src}/ckpts" ] || [ -d "${src}/samples" ] || [ -f "${src}/training_metrics.csv" ]; then
        echo "  ✓ Contains ccDDPM outputs (ckpts/, samples/, or CSV logs)"
        echo "  → Syncing to ${RESULTS_DST}"
        rsync -av "${src}/" "${RESULTS_DST}/"
        found_any=1
        break  # Stop after first successful sync
      elif [ -d "${src}/class_0" ] || [ -d "${src}/class_1" ]; then
        echo "  ✓ Contains generation outputs (class_X/ directories)"
        echo "  → Syncing to ${RESULTS_DST}"
        rsync -av "${src}/" "${RESULTS_DST}/"
        found_any=1
        break  # Stop after first successful sync
      else
        echo "  ℹ Directory exists but no recognizable ccDDPM structure"
        # Don't sync unknown directories automatically
      fi
    else
      echo "  ✗ Directory does not exist"
    fi
  done

  if [ $found_any -eq 0 ]; then
    echo ""
    echo "[ERROR] ❌ No output directories found in expected locations!"
    echo "        Searched: ${CANDIDATES[*]}"
    echo ""
    echo "Attempting broader search in WORKDIR and REPO_DIR..."
    find "${WORKDIR}" -name "*.pt" -o -name "training_metrics.csv" 2>/dev/null | head -20
    echo ""
    echo "⚠️  Results may have been lost. Check training output paths."
  else
    echo ""
    echo "================================================================================"
    echo "✅ Results successfully copied to ${RESULTS_DST}"
    echo "================================================================================"
    echo ""
  fi
else
  echo ""
  echo "================================================================================"
  echo "❌ Training failed with exit code ${TRAIN_EXIT_CODE}"
  echo "================================================================================"
  echo "Skipping result sync due to training failure."
  echo "Check the error log for details: ${SLURM_JOB_NAME}.${SLURM_JOB_ID}.err"
  echo ""
fi

# ---------- 10) Cleanup ----------
echo "🧹 Cleaning up localscratch..."
if cd "${LOCALSCRATCH%/}/${USER}"; then
  [ -n "${MYLOCALSCRATCH:-}" ] && rm -rf --one-file-system "${MYLOCALSCRATCH}"
  echo "✓ Cleanup complete"
fi

echo ""
echo "================================================================================"
echo "🎉 Job Complete!"
echo "================================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "GPUs used: ${NUM_GPUS}"
echo "Results location: ${RESULTS_DST}"
echo "================================================================================"

exit ${TRAIN_EXIT_CODE}
