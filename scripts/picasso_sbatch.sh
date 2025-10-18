#!/usr/bin/env bash
#SBATCH -J ccddpm_pathmnist
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail

# ---------- Inputs ----------
DATA_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/pathmnist/PathMNIST.npz"
REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/medsyn"
RESULTS_DST="/mnt/home/users/tic_163_uma/mpascual/fscratch/results/PathMNIST_ccDDPM"

# ---------- LocalScratch layout ----------
MYLOCALSCRATCH="${LOCALSCRATCH%/}/${USER}/${SLURM_JOB_ID}"
WORKDIR="${MYLOCALSCRATCH}/work"
REPO_DIR="${WORKDIR}/medsyn"
DATA_DIR="${WORKDIR}/datasets"
OUT_DIR="${WORKDIR}/results"

mkdir -p "${WORKDIR}" "${DATA_DIR}" "${OUT_DIR}"
echo "Localscratch workdir: ${WORKDIR}"

# ---------- 1) Repo to localscratch (idempotent) ----------
if [ -d "${REPO_DIR}" ] && [ -f "${REPO_DIR}/pyproject.toml" ]; then
  echo "[repo] found at ${REPO_DIR}. skip copy."
else
  echo "[repo] copying from ${REPO_SRC} ..."
  mkdir -p "${REPO_DIR}"
  rsync -a "${REPO_SRC}/" "${REPO_DIR}/"
fi

# ---------- 2) Dataset to localscratch (idempotent) ----------
DATA_DST="${DATA_DIR}/PathMNIST.npz"
if [ -f "${DATA_DST}" ]; then
  echo "[data] found at ${DATA_DST}. skip copy."
else
  echo "[data] copying ${DATA_SRC} -> ${DATA_DST}"
  rsync -a "${DATA_SRC}" "${DATA_DST}"
fi

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

# ---------- 4) Run training ----------
cd "${REPO_DIR}"
export DATASET_PATH="${DATA_DST}"
export OUTPUT_DIR="${OUT_DIR}"

nvidia-smi || true

# Prefer console entry point, fallback to module form if not present
if command -v ccddpm-train >/dev/null 2>&1; then
  srun ccddpm-train \
    --config config/picasso_cfg.yaml \
    --dataset "${DATA_DST}" \
    --outdir  "${OUT_DIR}"
else
  echo "[warn] ccddpm-train not on PATH; using python -m fallback."
  srun python -m medsyn.cli.train_ccDDPM \
    --config config/picasso_cfg.yaml \
    --dataset "${DATA_DST}" \
    --outdir  "${OUT_DIR}"
fi

# ---------- 5) Sync results back ----------
copy_back() {
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
    find "${REPO_DIR}/outputs" -type d -maxdepth 3 2>/dev/null | head -20
  else
    echo "  ${REPO_DIR}/outputs does not exist"
  fi
  echo ""

  # Known candidates where medsyn ccDDPM writes outputs
  # Priority order: explicit CLI outdir > default locations
  CANDIDATES=(
    "${OUT_DIR}"                                      # CLI --outdir (highest priority)
    "$(pwd)/outputs/ccddpm"                           # Relative to current directory
    "$(pwd)/outputs"                                  # Current dir outputs parent
    "${REPO_DIR}/outputs/ccddpm"                      # Default from config: ./outputs/ccddpm
    "${REPO_DIR}/outputs"                             # Parent outputs directory
    "${REPO_DIR}/samples/ccddpm"                      # Inference/generation outputs
    "${REPO_DIR}/samples"                             # Inference parent directory
    "${REPO_DIR}/medsyn/outputs/ccddpm"               # Fallback pkg-relative path
    "${WORKDIR}/outputs/ccddpm"                       # Workdir-relative path
    "${WORKDIR}/outputs"                              # Workdir outputs parent
    "${WORKDIR}/medsyn/outputs/ccddpm"                # Full workdir path to repo outputs
    "${WORKDIR}/medsyn/outputs"                       # Full workdir path to repo outputs parent
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
      
      # Check for typical ccDDPM output structure
      if [ -d "${src}/ckpts" ] || [ -d "${src}/samples" ] || [ -f "${src}/training_metrics.csv" ]; then
        echo "  ✓ Contains ccDDPM outputs (ckpts/, samples/, or CSV logs)"
        echo "  → syncing to ${RESULTS_DST}"
        rsync -av "${src}/" "${RESULTS_DST}/"
        found_any=1
      elif [ -d "${src}/class_0" ] || [ -d "${src}/class_1" ]; then
        echo "  ✓ Contains generation outputs (class_X/ directories)"
        echo "  → syncing to ${RESULTS_DST}"
        rsync -av "${src}/" "${RESULTS_DST}/"
        found_any=1
      else
        echo "  ℹ Directory exists but no recognizable ccDDPM structure, syncing anyway"
        rsync -av "${src}/" "${RESULTS_DST}/"
        found_any=1
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
  else
    echo ""
    echo "[sync] ✓ Results successfully copied to ${RESULTS_DST}"
  fi
}

trap copy_back EXIT

# ---------- 6) Cleanup ----------
if cd "${LOCALSCRATCH%/}/${USER}"; then
  [ -n "${MYLOCALSCRATCH:-}" ] && rm -rf --one-file-system "${MYLOCALSCRATCH}"
fi