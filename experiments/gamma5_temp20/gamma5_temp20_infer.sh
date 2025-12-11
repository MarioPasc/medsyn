#!/usr/bin/env bash
#SBATCH -J log_gamma5_temp20
#SBATCH --time=6:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:4
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

# ==============================================================================
# Parallel Multi-GPU Image Generation for ccDDPM with Batch Processing
# ==============================================================================
# This script parallelizes image generation across N GPUs for maximum efficiency.
# Each GPU independently generates images for different classes/splits using
# batch generation (4 images per batch by default) to maximize GPU utilization.
#
# Configuration:
#   --gres=gpu:N  - Request N GPUs (default: 4)
#   --cpus-per-task - Should be >= 4*N for optimal data loading (default: 16 for 4 GPUs)
#   --mem - ~16GB per GPU with batch_size=4 (default: 64G for 4 GPUs)
#
# Performance with batch_size=4:
#   - Expected speedup: ~3-4x vs single-image generation
#   - GPU memory usage: ~1GB per GPU (was 0.3GB without batching)
#   - Total throughput: ~4 GPUs * 4 images/batch = 16x single-GPU baseline
#
# Usage:
#   sbatch scripts/picasso_generate_parallel_sbatch.sh
# ==============================================================================

set -euo pipefail

# ---------- Configuration ----------
# Path to the repository
REPO_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/medsyn"

# Path to trained checkpoint
CHECKPOINT_SRC="/mnt/home/users/tic_163_uma/mpascual/fscratch/results/gamma5_temp20_training/ckpts/best.pt"

# Where to store generated images
RESULTS_DST="/mnt/home/users/tic_163_uma/mpascual/fscratch/results/gamma5_temp20_inference"

# Number of GPUs to use (auto-detected from SLURM, or set manually)
NUM_GPUS=${SLURM_GPUS_ON_NODE:-4}

# ---------- LocalScratch Setup ----------
MYLOCALSCRATCH="${LOCALSCRATCH%/}/${USER}/${SLURM_JOB_ID}"
WORKDIR="${MYLOCALSCRATCH}/work"
REPO_DIR="${WORKDIR}/medsyn"
CKPT_DIR="${WORKDIR}/checkpoints"
OUT_DIR="${WORKDIR}/generated"
CONFIG_TEMP="${REPO_DIR}/experiments/gamma5_temp20/gamma5_temp20.yaml"

mkdir -p "${WORKDIR}" "${CKPT_DIR}" "${OUT_DIR}"

echo "================================================================================"
echo "🚀 Parallel Multi-GPU Image Generation for ccDDPM"
echo "================================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Number of GPUs: ${NUM_GPUS}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Localscratch: ${MYLOCALSCRATCH}"
echo "================================================================================"
echo ""

# ---------- Copy Repository ----------
if [ -d "${REPO_DIR}" ] && [ -f "${REPO_DIR}/pyproject.toml" ]; then
  echo "[✓] Repository found at ${REPO_DIR}, skipping copy"
else
  echo "[→] Copying repository from ${REPO_SRC}..."
  mkdir -p "${REPO_DIR}"
  rsync -a "${REPO_SRC}/" "${REPO_DIR}/"
  echo "[✓] Repository copied"
fi

# ---------- Copy Checkpoint ----------
CKPT_DST="${CKPT_DIR}/best.pt"
if [ -f "${CKPT_DST}" ]; then
  echo "[✓] Checkpoint found at ${CKPT_DST}, skipping copy"
else
  echo "[→] Copying checkpoint from ${CHECKPOINT_SRC}..."
  rsync -a "${CHECKPOINT_SRC}" "${CKPT_DST}"
  echo "[✓] Checkpoint copied ($(du -h ${CKPT_DST} | cut -f1))"
fi

# ---------- Environment Setup ----------
echo ""
echo "[→] Setting up conda environment..."

# Try to load conda module
module_loaded=0
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
  if module avail 2>/dev/null | grep -qi "^${m}[[:space:]]"; then
    module load "$m" && module_loaded=1 && break
  fi
done

if [ "$module_loaded" -eq 0 ]; then
  echo "[ℹ] No conda module loaded; assuming conda in PATH"
fi

# Activate environment
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate medsyn 2>/dev/null || source activate medsyn
else
  source activate medsyn
fi

# Verify environment
echo "[✓] Python: $(which python)"
python -c "import sys; print('[✓] Python version:', sys.version.split()[0])"
python -c "import torch; print('[✓] PyTorch:', torch.__version__)"
python -c "import torch; print('[✓] CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('[✓] Number of GPUs:', torch.cuda.device_count())"

# ---------- GPU Information ----------
echo ""
echo "================================================================================"
echo "📊 GPU Information"
echo "================================================================================"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits | \
  awk -F', ' 'NR<='"${NUM_GPUS}"' {printf "GPU %s: %s (%s MB total, %s MB free)\n", $1, $2, $3, $4}'
echo "================================================================================"

# ---------- Verify GPU Availability ----------
echo ""
echo "[→] Verifying allocated GPUs are available..."
MIN_FREE_MEMORY_MB=8000  # Minimum 8GB free memory required per GPU
GPU_CHECK_FAILED=0

while IFS=', ' read -r gpu_idx gpu_name mem_total mem_free; do
  if [ "$gpu_idx" -lt "$NUM_GPUS" ]; then
    if [ "$mem_free" -lt "$MIN_FREE_MEMORY_MB" ]; then
      echo "[⚠] GPU $gpu_idx: Only ${mem_free}MB free (need ${MIN_FREE_MEMORY_MB}MB minimum)"
      GPU_CHECK_FAILED=1
    else
      echo "[✓] GPU $gpu_idx: ${mem_free}MB free - OK"
    fi
  fi
done < <(nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits)

if [ "$GPU_CHECK_FAILED" -eq 1 ]; then
  echo ""
  echo "[ERROR] One or more GPUs do not have sufficient free memory!"
  echo "        This may indicate GPUs are being used by other processes."
  echo "        Consider:"
  echo "          - Waiting for current jobs to complete"
  echo "          - Requesting different GPU resources"
  echo "          - Checking with 'nvidia-smi' for running processes"
  echo ""
  echo "Proceeding anyway (SLURM should have allocated dedicated GPUs)..."
fi
echo ""

# ---------- Prepare Configuration ----------
cd "${REPO_DIR}"

# cp experiments/AdamW_Batch64_lowt_enhancement_gamma5/config_AdamW_Batch64_lowt_enhancement_gamma5.yaml "${CONFIG_TEMP}"

# Update checkpoint path in config
sed -i "s|checkpoint:.*|checkpoint: ${CKPT_DST}|g" "${CONFIG_TEMP}"

# Update output path in config
# Assuming the config uses generate.npz_with_synth_images.save_to
sed -i "s|save_to:.*|save_to: ${OUT_DIR}|g" "${CONFIG_TEMP}"

echo ""
echo "[✓] Configuration prepared"
echo "    Config: ${CONFIG_TEMP}"
echo "    Checkpoint: ${CKPT_DST}"
echo "    Output: ${OUT_DIR}"

# ---------- Run Parallel Generation ----------
echo ""
echo "================================================================================"
echo "🎨 Starting Parallel Image Generation on ${NUM_GPUS} GPUs"
echo "================================================================================"
echo "Start time: $(date)"
echo ""

START_TIME=$(date +%s)

# Run parallel generation
srun python -m medsyn.cli.generate_ccDDPM_parallel \
  "${CONFIG_TEMP}" \
  --num-gpus "${NUM_GPUS}" \
  --batch-size 8 \
  --no-pngs

EXIT_CODE=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

echo ""
echo "================================================================================"
if [ $EXIT_CODE -eq 0 ]; then
  echo "✅ Generation completed successfully"
else
  echo "❌ Generation failed with exit code ${EXIT_CODE}"
fi
echo "================================================================================"
echo "End time: $(date)"
echo "Total time: ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "================================================================================"

if [ $EXIT_CODE -ne 0 ]; then
  echo ""
  echo "[ERROR] Generation failed. Check logs above for details."
  exit $EXIT_CODE
fi

# ---------- Sync Results ----------
echo ""
echo "================================================================================"
echo "📦 Syncing Generated Images to Permanent Storage"
echo "================================================================================"
echo "[→] Copying results to ${RESULTS_DST}..."

mkdir -p "${RESULTS_DST}"

# Debug: Check output directory
echo ""
echo "[ℹ] Output directory contents:"
ls -lh "${OUT_DIR}" 2>/dev/null || echo "  (empty or not accessible)"
echo ""

# Sync all generated data
if [ -d "${OUT_DIR}" ]; then
  # Count files before sync
  NUM_PNG=$(find "${OUT_DIR}" -name "*.png" 2>/dev/null | wc -l)
  NUM_NPZ=$(find "${OUT_DIR}" -name "*.npz" 2>/dev/null | wc -l)
  NUM_JSON=$(find "${OUT_DIR}" -name "*.json" 2>/dev/null | wc -l)

  echo "[→] Syncing ${NUM_PNG} PNG files, ${NUM_NPZ} NPZ files, ${NUM_JSON} JSON files..."

  # Sync with progress
  rsync -av --info=progress2 "${OUT_DIR}/" "${RESULTS_DST}/"

  SYNC_EXIT=$?

  if [ $SYNC_EXIT -eq 0 ]; then
    echo "[✓] Sync completed successfully"
  else
    echo "[ERROR] Sync failed with exit code ${SYNC_EXIT}"
    exit $SYNC_EXIT
  fi
else
  echo "[ERROR] Output directory ${OUT_DIR} does not exist!"
  exit 1
fi

# ---------- Summary Statistics ----------
echo ""
echo "================================================================================"
echo "📊 Generation Summary"
echo "================================================================================"

if [ -d "${RESULTS_DST}" ]; then
  echo "Output location: ${RESULTS_DST}"
  echo ""

  # Count files
  TOTAL_PNG=$(find "${RESULTS_DST}" -name "*.png" -not -name "denoising_*" 2>/dev/null | wc -l)
  TOTAL_NPZ=$(find "${RESULTS_DST}" -name "*.npz" 2>/dev/null | wc -l)
  TOTAL_JSON=$(find "${RESULTS_DST}" -name "*.json" 2>/dev/null | wc -l)
  TOTAL_VIS=$(find "${RESULTS_DST}" -name "denoising_*.png" 2>/dev/null | wc -l)

  echo "Generated files:"
  echo "  - Images: ${TOTAL_PNG}"
  echo "  - NPZ files: ${TOTAL_NPZ}"
  echo "  - JSON indexes: ${TOTAL_JSON}"
  echo "  - Visualizations: ${TOTAL_VIS}"
  echo ""

  # Per-split statistics
  echo "Per-split breakdown:"
  for split_dir in "${RESULTS_DST}"/train "${RESULTS_DST}"/val "${RESULTS_DST}"/test; do
    if [ -d "${split_dir}" ]; then
      split_name=$(basename "${split_dir}")
      num_images=$(find "${split_dir}" -name "*.png" -not -name "denoising_*" 2>/dev/null | wc -l)
      num_classes=$(find "${split_dir}" -type d -name "class_*" 2>/dev/null | wc -l)
      echo "  - ${split_name}: ${num_images} images across ${num_classes} classes"

      # Show NPZ file size
      npz_file=$(find "${split_dir}" -name "*_${split_name}_synth.npz" 2>/dev/null | head -1)
      if [ -f "${npz_file}" ]; then
        npz_size=$(du -h "${npz_file}" | cut -f1)
        echo "    NPZ file: ${npz_size}"
      fi
    fi
  done

  echo ""

  # Performance metrics
  if [ $ELAPSED -gt 0 ]; then
    IMAGES_PER_SEC=$(echo "scale=2; ${TOTAL_PNG} / ${ELAPSED}" | bc)
    echo "Performance:"
    echo "  - Total time: ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
    echo "  - Throughput: ${IMAGES_PER_SEC} images/second"
    echo "  - GPUs used: ${NUM_GPUS}"
    SPEEDUP=$(echo "scale=2; ${NUM_GPUS} * 1.0" | bc)
    echo "  - Estimated speedup: ~${SPEEDUP}x vs single GPU"
  fi
fi

echo "================================================================================"

# ---------- Cleanup ----------
echo ""
echo "[→] Cleaning up localscratch..."
if cd "${LOCALSCRATCH%/}/${USER}"; then
  [ -n "${MYLOCALSCRATCH:-}" ] && rm -rf --one-file-system "${MYLOCALSCRATCH}"
  echo "[✓] Cleanup complete"
fi

# ---------- Final Status ----------
echo ""
echo "================================================================================"
echo "🎉 Parallel generation job completed successfully!"
echo "================================================================================"
echo "Results location: ${RESULTS_DST}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Total time: ${ELAPSED_MIN}m ${ELAPSED_SEC}s"
echo "GPUs used: ${NUM_GPUS}"
echo "================================================================================"
echo ""

exit 0
