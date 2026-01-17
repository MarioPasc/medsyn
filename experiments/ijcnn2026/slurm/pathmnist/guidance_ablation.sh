#!/usr/bin/env bash
#SBATCH -J guidance_ablation
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail

# ============================================================================
# IJCNN 2026 - PathMNIST Guidance Scale Ablation
# Uses EXISTING best PathMNIST model - NO retraining needed
# Tests guidance scales: 1.0, 1.5, 2.0, 2.5, 3.0
# ============================================================================

# ---------- PATHS (MODIFY THESE FOR YOUR CLUSTER) ----------
CHECKPOINT="/path/to/best_pathmnist_model/ckpts/best.pt"  # TODO: Update to best PathMNIST model
REPO_SRC="/path/to/medsyn"  # TODO: Update
OUTPUT_BASE="/path/to/results/guidance_ablation"  # TODO: Update
CONFIG_FILE="experiments/not_considering_minSNR/temp10/temp10.yaml"  # Use existing PathMNIST config

export IS_SUPERCOMPUTER=1
export CUDA_VISIBLE_DEVICES=0

for i in {0..7}; do
  if [[ -z $(nvidia-smi -i $i --query-compute-apps=pid --format=csv,noheader 2>/dev/null) ]] && nvidia-smi -i $i &>/dev/null; then
    export CUDA_VISIBLE_DEVICES=$i
    break
  fi
done

echo "================================================================================"
echo "IJCNN 2026 - Guidance Scale Ablation for PathMNIST"
echo "Testing scales: 1.0, 1.5, 2.0, 2.5, 3.0"
echo "================================================================================"

# ---------- LocalScratch ----------
MYLOCALSCRATCH="${LOCALSCRATCH%/}/${USER}/${SLURM_JOB_ID}"
WORKDIR="${MYLOCALSCRATCH}/work"
REPO_DIR="${WORKDIR}/medsyn"

mkdir -p "${WORKDIR}"

# Copy repo
rsync -a "${REPO_SRC}/" "${REPO_DIR}/"

# Load conda
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
  if module avail 2>/dev/null | grep -qi "^${m}[[:space:]]"; then
    module load "$m" && break
  fi
done

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate medsyn 2>/dev/null || source activate medsyn
else
  source activate medsyn
fi

cd "${REPO_DIR}"

# Run generation for each guidance scale
GUIDANCE_SCALES="1.0 1.5 2.0 2.5 3.0"

for scale in ${GUIDANCE_SCALES}; do
  echo ""
  echo "========================================"
  echo "Generating with guidance_scale=${scale}"
  echo "========================================"

  OUT_DIR="${WORKDIR}/guidance_${scale}"
  mkdir -p "${OUT_DIR}"

  python -m medsyn.cli.generate_ccDDPM \
    --config "${CONFIG_FILE}" \
    --checkpoint "${CHECKPOINT}" \
    --output "${OUT_DIR}" \
    --guidance_scale "${scale}" \
    --num_samples 10000 \
    --batch_size 64

  # Sync this scale's results
  SCALE_DST="${OUTPUT_BASE}/guidance_${scale}"
  mkdir -p "${SCALE_DST}"
  rsync -av "${OUT_DIR}/" "${SCALE_DST}/"
  echo "Saved guidance_scale=${scale} results to ${SCALE_DST}"
done

echo ""
echo "================================================================================"
echo "Guidance Scale Ablation Complete!"
echo "Results saved to: ${OUTPUT_BASE}"
echo "================================================================================"

# Cleanup
if cd "${LOCALSCRATCH%/}/${USER}"; then
  [ -n "${MYLOCALSCRATCH:-}" ] && rm -rf --one-file-system "${MYLOCALSCRATCH}"
fi

exit 0
