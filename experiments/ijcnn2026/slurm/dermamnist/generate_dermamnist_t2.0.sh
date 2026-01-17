#!/usr/bin/env bash
#SBATCH -J gen_derm_t2.0
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail

# ============================================================================
# IJCNN 2026 - DermaMNIST Generation - Temperature 1.0
# ============================================================================

EXPERIMENT_NAME="dermamnist_t2.0"
TEMPERATURE="2.0"

# ---------- PATHS (MODIFY THESE FOR YOUR CLUSTER) ----------
CHECKPOINT="/path/to/results/dermamnist_t${TEMPERATURE}/ckpts/best.pt"  # TODO: Update
REPO_SRC="/path/to/medsyn"  # TODO: Update
OUTPUT_DST="/path/to/results/${EXPERIMENT_NAME}/synth"  # TODO: Update

CONFIG_FILE="experiments/ijcnn2026/configs/dermamnist_t${TEMPERATURE}.yaml"

export IS_SUPERCOMPUTER=1
export CUDA_VISIBLE_DEVICES=0

for i in {0..7}; do
  if [[ -z $(nvidia-smi -i $i --query-compute-apps=pid --format=csv,noheader 2>/dev/null) ]] && nvidia-smi -i $i &>/dev/null; then
    export CUDA_VISIBLE_DEVICES=$i
    break
  fi
done

echo "================================================================================"
echo "IJCNN 2026 - DermaMNIST Generation - Temperature ${TEMPERATURE}"
echo "================================================================================"

# ---------- LocalScratch ----------
MYLOCALSCRATCH="${LOCALSCRATCH%/}/${USER}/${SLURM_JOB_ID}"
WORKDIR="${MYLOCALSCRATCH}/work"
REPO_DIR="${WORKDIR}/medsyn"
OUT_DIR="${WORKDIR}/synth"

mkdir -p "${WORKDIR}" "${OUT_DIR}"

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

# Generate synthetic samples
python -m medsyn.cli.generate_ccDDPM \
  --config "${CONFIG_FILE}" \
  --checkpoint "${CHECKPOINT}" \
  --output "${OUT_DIR}" \
  --num_samples 10000 \
  --batch_size 64

GEN_EXIT_CODE=$?

# Sync results
if [ ${GEN_EXIT_CODE} -eq 0 ]; then
  mkdir -p "${OUTPUT_DST}"
  rsync -av "${OUT_DIR}/" "${OUTPUT_DST}/"
  echo "Generated samples saved to ${OUTPUT_DST}"
fi

# Cleanup
if cd "${LOCALSCRATCH%/}/${USER}"; then
  [ -n "${MYLOCALSCRATCH:-}" ] && rm -rf --one-file-system "${MYLOCALSCRATCH}"
fi

exit ${GEN_EXIT_CODE}
