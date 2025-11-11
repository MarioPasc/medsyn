#!/usr/bin/env bash
#SBATCH -J ccddpm_audit_privacy
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
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

# Path to dataset NPZ (needed by the project YAML)
DATA_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/pathmnist/PathMNIST.npz"

# Where to store audit results
RESULTS_DST="/mnt/home/users/tic_163_uma/mpascual/fscratch/results/PathMNIST_ccDDPM_audit"

# ---------- LocalScratch layout ----------
MYLOCALSCRATCH="${LOCALSCRATCH%/}/${USER}/${SLURM_JOB_ID}"
WORKDIR="${MYLOCALSCRATCH}/work"
REPO_DIR="${WORKDIR}/medsyn"
CKPT_DIR="${WORKDIR}/checkpoints"
DATA_DIR="${WORKDIR}/datasets"
OUT_DIR="${WORKDIR}/audit_results"

mkdir -p "${WORKDIR}" "${CKPT_DIR}" "${DATA_DIR}" "${OUT_DIR}"
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

# ---------- 3) Checkpoint to localscratch (idempotent) ----------
CKPT_DST="${CKPT_DIR}/best.pt"
if [ -f "${CKPT_DST}" ]; then
  echo "[checkpoint] found at ${CKPT_DST}. skip copy."
else
  echo "[checkpoint] copying ${CHECKPOINT_SRC} -> ${CKPT_DST}"
  rsync -a "${CHECKPOINT_SRC}" "${CKPT_DST}"
fi

# ---------- 4) Load conda module and activate prebuilt env ----------
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

# ---------- 5) Prepare configuration files ----------
cd "${REPO_DIR}"

# Create a temporary project config file with updated paths for localscratch
CONFIG_TEMP="${WORKDIR}/picasso_cfg_audit.yaml"
cp config/picasso_cfg.yaml "${CONFIG_TEMP}"

# Create a temporary audit config with updated paths
AUDIT_CONFIG_TEMP="${WORKDIR}/audit_ccddpm_privacy.yaml"
cp config/audit_ccddpm_privacy.yaml "${AUDIT_CONFIG_TEMP}"

# Update the audit config with localscratch paths
cat > "${AUDIT_CONFIG_TEMP}" << EOF
# Configuration file for ccDDPM Privacy Audit
# This file contains all parameters for running memorization and membership inference attacks

# Input/Output Paths
io:
  yaml: ${CONFIG_TEMP}  # Path to project YAML configuration
  checkpoint: ${CKPT_DST}  # Path to model checkpoint .pt file
  outdir: ${OUT_DIR}  # Output directory for results

# Generation Configuration
generation:
  use_existing_synth: true  # Use existing synthetic images from dataset (is_synth=True) instead of generating new ones
  existing_synth_split: "train"  # Which split to load existing synthetic images from (train/val/test)
  per_class: 64  # Number of samples to generate per class (only used if use_existing_synth=false)
  guidance_scale: 2.0  # Classifier-free guidance scale for inference (only used if use_existing_synth=false)
  num_inference_steps: 250  # Number of DDPM sampling steps (only used if use_existing_synth=false)
  seed: 1337  # Random seed for reproducibility
  save_grid: true  # Whether to save a grid of generated samples

# Nearest Neighbor Analysis Configuration
nearest_neighbor:
  k: 5  # Number of nearest neighbors to search (currently unused but reserved)
  lpips_threshold: 0.12  # LPIPS threshold for identifying "near-copy" (lower = more similar)
  ssim_threshold: 0.90  # SSIM threshold for identifying "near-copy" (higher = more similar)
  use_clip: false  # Use CLIP embeddings instead of ResNet50 (requires open_clip)
  batch_size: 128  # Batch size for embedding extraction (reduced for memory efficiency)

# Membership Inference Attack Configuration
membership_inference:
  n_timesteps_per_sample: 32  # Number of timestep samples to average loss over per sample
  auc_points: 200  # Number of points for ROC curve discretization

# Logging Configuration
logging:
  level: INFO  # Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
  save_intermediate: true  # Save intermediate results for debugging
  log_file: ${OUT_DIR}/audit.log  # Path to save logs to file
  verbose_progress: true  # Show detailed progress bars
EOF

# Set environment variables for the project config
export DATASET_PATH="${DATA_DST}"
export OUTPUT_DIR="${OUT_DIR}"

echo "[config] Project config: ${CONFIG_TEMP}"
echo "[config] Audit config: ${AUDIT_CONFIG_TEMP}"
echo "[config] Checkpoint: ${CKPT_DST}"
echo "[config] Dataset: ${DATA_DST}"
echo "[config] Output directory: ${OUT_DIR}"

# ---------- 6) Run privacy audit ----------
nvidia-smi || true

echo ""
echo "================================================================================"
echo "🔒 Starting Privacy Audit (Memorization + Membership Inference)"
echo "================================================================================"
echo "[audit] Checkpoint: ${CKPT_DST}"
echo "[audit] Dataset: ${DATA_DST}"
echo "[audit] Output dir: ${OUT_DIR}"
echo "[audit] Config: ${AUDIT_CONFIG_TEMP}"
echo ""

# Prefer console entry point, fallback to module form if not present
if command -v ccddpm-audit >/dev/null 2>&1; then
  srun ccddpm-audit --config "${AUDIT_CONFIG_TEMP}"
else
  echo "[warn] ccddpm-audit not on PATH; using python -m fallback."
  srun python -m medsyn.utils.audit_ccddpm_privacy --config "${AUDIT_CONFIG_TEMP}"
fi

# ---------- 7) Sync results back ----------
echo ""
echo "================================================================================"
echo "📦 Syncing Audit Results to Permanent Storage"
echo "================================================================================"
echo "[sync] copying results back to ${RESULTS_DST}"
mkdir -p "${RESULTS_DST}"

# Debug: Show current working directory and environment
echo "[debug] Current directory: $(pwd)"
echo "[debug] WORKDIR=${WORKDIR}"
echo "[debug] REPO_DIR=${REPO_DIR}"
echo "[debug] OUT_DIR=${OUT_DIR}"
echo ""

# Debug: List what's actually in the output directory
echo "[debug] Checking OUT_DIR structure:"
ls -la "${OUT_DIR}" 2>/dev/null || echo "  (not accessible)"
echo ""

echo "[debug] Checking for audit outputs:"
if [ -d "${OUT_DIR}" ]; then
  echo "  Found! Contents:"
  find "${OUT_DIR}" -type f -maxdepth 2 2>/dev/null | head -20
  echo ""
else
  echo "  ${OUT_DIR} does not exist"
fi
echo ""

# Known candidates where audit outputs are written
# Priority order: explicit output dir > config defaults
CANDIDATES=(
  "${OUT_DIR}"                                      # Explicit output directory (highest priority)
  "${REPO_DIR}/outputs/audit"                       # Default from config
  "${WORKDIR}/audit_results"                        # Workdir-relative path
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

    # Check for typical audit output files
    if [ -f "${src}/audit_ccddpm_privacy.npz" ] || [ -f "${src}/audit_summary.json" ] || [ -f "${src}/memorization_suspects.csv" ]; then
      echo "  ✓ Contains audit outputs (NPZ, JSON, or CSV files)"
      echo "  → Syncing to ${RESULTS_DST}"
      rsync -av "${src}/" "${RESULTS_DST}/"
      found_any=1
      break  # Stop after first successful sync
    else
      echo "  ℹ Directory exists but no recognizable audit structure"
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
  find "${WORKDIR}" -name "*.npz" -o -name "audit_summary.json" -o -name "memorization_suspects.csv" 2>/dev/null | head -20
  echo ""
  echo "⚠️  Results may have been lost. Check audit output paths."
else
  echo ""
  echo "================================================================================"
  echo "✅ Audit results successfully copied to ${RESULTS_DST}"
  echo "================================================================================"

  # Display summary statistics
  if [ -d "${RESULTS_DST}" ]; then
    echo ""
    echo "📊 Audit Results Summary:"

    # Check for main output files
    if [ -f "${RESULTS_DST}/audit_summary.json" ]; then
      echo ""
      echo "  📄 Audit Summary (audit_summary.json):"
      cat "${RESULTS_DST}/audit_summary.json" 2>/dev/null | head -50
      echo ""
    fi

    if [ -f "${RESULTS_DST}/memorization_suspects.csv" ]; then
      num_suspects=$(tail -n +2 "${RESULTS_DST}/memorization_suspects.csv" 2>/dev/null | grep -c "True" || echo "0")
      echo "  🔍 Memorization suspects identified: ${num_suspects}"
    fi

    if [ -f "${RESULTS_DST}/mia_roc.csv" ]; then
      num_roc_points=$(tail -n +2 "${RESULTS_DST}/mia_roc.csv" 2>/dev/null | wc -l || echo "0")
      echo "  📈 MIA ROC curve points: ${num_roc_points}"
    fi

    if [ -f "${RESULTS_DST}/samples_grid.png" ]; then
      echo "  🖼️  Sample grid saved: samples_grid.png"
    fi

    echo ""
    echo "  📁 Total files generated:"
    echo "    - NPZ files: $(find "${RESULTS_DST}" -name "*.npz" 2>/dev/null | wc -l)"
    echo "    - CSV files: $(find "${RESULTS_DST}" -name "*.csv" 2>/dev/null | wc -l)"
    echo "    - JSON files: $(find "${RESULTS_DST}" -name "*.json" 2>/dev/null | wc -l)"
    echo "    - PNG files: $(find "${RESULTS_DST}" -name "*.png" 2>/dev/null | wc -l)"
    echo "    - Log files: $(find "${RESULTS_DST}" -name "*.log" 2>/dev/null | wc -l)"
    echo ""
  fi
  echo "================================================================================"
  echo ""
fi

# ---------- 8) Cleanup ----------
echo "🧹 Cleaning up localscratch..."
if cd "${LOCALSCRATCH%/}/${USER}"; then
  [ -n "${MYLOCALSCRATCH:-}" ] && rm -rf --one-file-system "${MYLOCALSCRATCH}"
  echo "✓ Cleanup complete"
fi

echo ""
echo "================================================================================"
echo "🎉 Privacy audit job completed!"
echo "================================================================================"
