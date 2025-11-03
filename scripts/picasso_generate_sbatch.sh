#!/usr/bin/env bash
#SBATCH -J ccddpm_generate
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail

# ---------- Inputs ----------
# Path to the repository
REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/medsyn"

# Path to trained checkpoint (modify to point to your best.pt)
CHECKPOINT_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/results/PathMNIST_ccDDPM_parallel/ckpts/best.pt"

# Where to store generated images
RESULTS_DST="/mnt/home/users/tic_163_uma/mpascual/fscratch/results/PathMNIST_ccDDPM_parallel_generated"

# ---------- LocalScratch layout ----------
MYLOCALSCRATCH="${LOCALSCRATCH%/}/${USER}/${SLURM_JOB_ID}"
WORKDIR="${MYLOCALSCRATCH}/work"
REPO_DIR="${WORKDIR}/medsyn"
CKPT_DIR="${WORKDIR}/checkpoints"
OUT_DIR="${WORKDIR}/generated"

mkdir -p "${WORKDIR}" "${CKPT_DIR}" "${OUT_DIR}"
echo "Localscratch workdir: ${WORKDIR}"

# ---------- 1) Repo to localscratch (idempotent) ----------
if [ -d "${REPO_DIR}" ] && [ -f "${REPO_DIR}/pyproject.toml" ]; then
  echo "[repo] found at ${REPO_DIR}. skip copy."
else
  echo "[repo] copying from ${REPO_SRC} ..."
  mkdir -p "${REPO_DIR}"
  rsync -a "${REPO_SRC}/" "${REPO_DIR}/"
fi

# ---------- 2) Checkpoint to localscratch (idempotent) ----------
CKPT_DST="${CKPT_DIR}/best.pt"
if [ -f "${CKPT_DST}" ]; then
  echo "[checkpoint] found at ${CKPT_DST}. skip copy."
else
  echo "[checkpoint] copying ${CHECKPOINT_SRC} -> ${CKPT_DST}"
  rsync -a "${CHECKPOINT_SRC}" "${CKPT_DST}"
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

# ---------- 4) Update config to use localscratch paths ----------
cd "${REPO_DIR}"

# Create a temporary config file with updated paths
CONFIG_TEMP="${WORKDIR}/picasso_cfg_generate.yaml"
cp config/picasso_cfg.yaml "${CONFIG_TEMP}"

# Update the checkpoint path in the config using sed
# This assumes the config uses the ${OUTPUT_DIR} variable pattern
# We'll override it via environment variable instead
export OUTPUT_DIR="${OUT_DIR}"

echo "[config] Using checkpoint: ${CKPT_DST}"
echo "[config] Output directory: ${OUT_DIR}"

# ---------- 5) Run generation ----------
nvidia-smi || true

echo ""
echo "================================================================================"
echo "🎨 Starting Image Generation"
echo "================================================================================"
echo "[generate] Checkpoint: ${CKPT_DST}"
echo "[generate] Output dir: ${OUT_DIR}"
echo "[generate] Config: ${CONFIG_TEMP}"
echo ""

# Update the checkpoint path in the temp config
sed -i "s|checkpoint:.*|checkpoint: ${CKPT_DST}|g" "${CONFIG_TEMP}"

# Prefer console entry point, fallback to module form if not present
if command -v ccddpm-generate >/dev/null 2>&1; then
  srun ccddpm-generate "${CONFIG_TEMP}"
else
  echo "[warn] ccddpm-generate not on PATH; using python -m fallback."
  srun python -m medsyn.cli.generate_ccDDPM "${CONFIG_TEMP}"
fi

# ---------- 6) Sync results back ----------
echo ""
echo "================================================================================"
echo "📦 Syncing Generated Images to Permanent Storage"
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
echo "[debug] Checking OUT_DIR structure:"
ls -la "${OUT_DIR}" 2>/dev/null || echo "  (not accessible)"
echo ""

echo "[debug] Checking for generated images:"
if [ -d "${OUT_DIR}" ]; then
  echo "  Found! Contents:"
  find "${OUT_DIR}" -type d -maxdepth 3 2>/dev/null | head -20
  echo ""
  echo "  NPZ files:"
  find "${OUT_DIR}" -name "*.npz" 2>/dev/null | head -10
  echo ""
  echo "  PNG files (sample):"
  find "${OUT_DIR}" -name "*.png" 2>/dev/null | head -10
else
  echo "  ${OUT_DIR} does not exist"
fi
echo ""

# Known candidates where generation outputs are written
# Priority order: explicit output dir > config defaults
CANDIDATES=(
  "${OUT_DIR}"                                      # Explicit output directory (highest priority)
  "${REPO_DIR}/outputs/synth"                       # Default from config
  "${REPO_DIR}/outputs/samples"                     # Alternative location
  "${WORKDIR}/synth"                                # Workdir-relative path
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

    # Check for typical generation output structure
    if [ -d "${src}/train" ] || [ -d "${src}/val" ] || [ -d "${src}/test" ]; then
      echo "  ✓ Contains generation outputs (train/, val/, test/ split directories)"
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
    elif find "${src}" -name "*.npz" 2>/dev/null | grep -q .; then
      echo "  ✓ Contains NPZ files"
      echo "  → Syncing to ${RESULTS_DST}"
      rsync -av "${src}/" "${RESULTS_DST}/"
      found_any=1
      break  # Stop after first successful sync
    else
      echo "  ℹ Directory exists but no recognizable generation structure"
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
  find "${WORKDIR}" -name "*.npz" -o -name "*_synth_*.png" 2>/dev/null | head -20
  echo ""
  echo "⚠️  Results may have been lost. Check generation output paths."
else
  echo ""
  echo "================================================================================"
  echo "✅ Generated images successfully copied to ${RESULTS_DST}"
  echo "================================================================================"

  # Display summary statistics
  if [ -d "${RESULTS_DST}" ]; then
    echo ""
    echo "📊 Generation Summary:"
    echo "  Total PNG files: $(find "${RESULTS_DST}" -name "*.png" 2>/dev/null | wc -l)"
    echo "  Total NPZ files: $(find "${RESULTS_DST}" -name "*.npz" 2>/dev/null | wc -l)"
    echo "  Total JSON files: $(find "${RESULTS_DST}" -name "*.json" 2>/dev/null | wc -l)"
    echo ""
    echo "  Splits generated:"
    for split_dir in "${RESULTS_DST}"/train "${RESULTS_DST}"/val "${RESULTS_DST}"/test; do
      if [ -d "${split_dir}" ]; then
        split_name=$(basename "${split_dir}")
        num_images=$(find "${split_dir}" -name "*.png" -not -name "denoising_*" 2>/dev/null | wc -l)
        echo "    - ${split_name}: ${num_images} images"
      fi
    done
    echo ""
  fi
  echo "================================================================================"
  echo ""
fi

# ---------- 7) Cleanup ----------
echo "🧹 Cleaning up localscratch..."
if cd "${LOCALSCRATCH%/}/${USER}"; then
  [ -n "${MYLOCALSCRATCH:-}" ] && rm -rf --one-file-system "${MYLOCALSCRATCH}"
  echo "✓ Cleanup complete"
fi

echo ""
echo "================================================================================"
echo "🎉 Generation job completed!"
echo "================================================================================"
