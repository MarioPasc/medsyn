#!/usr/bin/env bash
#SBATCH -J dermamnist_t2.5
#SBATCH --time=1-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:1
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail

# ============================================================================
# IJCNN 2026 - DermaMNIST Training - Temperature 1.0
# ============================================================================

EXPERIMENT_NAME="dermamnist_t2.5"
TEMPERATURE="2.5"

# ---------- PATHS (MODIFY THESE FOR YOUR CLUSTER) ----------
DATA_SRC="/path/to/dermamnist/DermaMNIST.npz"  # TODO: Update path
REPO_SRC="/path/to/medsyn"                      # TODO: Update path
RESULTS_DST="/path/to/results/${EXPERIMENT_NAME}"  # TODO: Update path
FID_WEIGHTS="/path/to/fid_weights/weights-inception-2015-12-05-6726825d.pth"  # TODO: Update path

CONFIG_FILE="experiments/ijcnn2026/configs/dermamnist_t${TEMPERATURE}.yaml"
CONFIG_BACKUP="${CONFIG_FILE}.backup"

export IS_SUPERCOMPUTER=1
NUM_GPUS=1

# Dynamic GPU assignment
export CUDA_VISIBLE_DEVICES=0
for i in {0..7}; do
  if [[ -z $(nvidia-smi -i $i --query-compute-apps=pid --format=csv,noheader 2>/dev/null) ]] && nvidia-smi -i $i &>/dev/null; then
    export CUDA_VISIBLE_DEVICES=$i
    echo "Auto-assigned to GPU: $i"
    break
  fi
done

echo "================================================================================"
echo "IJCNN 2026 - DermaMNIST Training - Temperature ${TEMPERATURE}"
echo "================================================================================"

# ---------- LocalScratch layout ----------
MYLOCALSCRATCH="${LOCALSCRATCH%/}/${USER}/${SLURM_JOB_ID}"
WORKDIR="${MYLOCALSCRATCH}/work"
REPO_DIR="${WORKDIR}/medsyn"
DATA_DIR="${WORKDIR}/datasets"
OUT_DIR="${WORKDIR}/results"

mkdir -p "${WORKDIR}" "${DATA_DIR}" "${OUT_DIR}"
echo "Localscratch workdir: ${WORKDIR}"

# ---------- Copy repo and data ----------
echo "[repo] copying from ${REPO_SRC} ..."
mkdir -p "${REPO_DIR}"
rsync -a "${REPO_SRC}/" "${REPO_DIR}/"

DATA_DST="${DATA_DIR}/DermaMNIST.npz"
echo "[data] copying ${DATA_SRC} -> ${DATA_DST}"
rsync -a "${DATA_SRC}" "${DATA_DST}"

# ---------- Load conda ----------
module_loaded=0
for m in miniconda3 Miniconda3 anaconda3 Anaconda3 miniforge mambaforge; do
  if module avail 2>/dev/null | grep -qi "^${m}[[:space:]]"; then
    module load "$m" && module_loaded=1 && break
  fi
done

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh" || true
  conda activate medsyn 2>/dev/null || source activate medsyn
else
  source activate medsyn
fi

echo "[python] $(which python || true)"
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"

# ---------- Run training ----------
cd "${REPO_DIR}"
cp "${CONFIG_FILE}" "${CONFIG_BACKUP}"

# Enable DDP in config
sed -i '/ccddpm:/,/^[^ ]/ {
  /dist:/,/^[[:space:]]*[a-z]/ {
    s/enabled: false/enabled: true/
    s/num_gpus: .*/num_gpus: '"${NUM_GPUS}"'/
  }
}' "${CONFIG_FILE}"

export DATASET_PATH="${DATA_DST}"
export OUTPUT_DIR="${OUT_DIR}"
export DATA_RAW_DIR="${DATA_DIR}"
export FID_WEIGHTS_PATH="${FID_WEIGHTS}"

echo "[training] Starting training with temperature ${TEMPERATURE}"
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NUM_GPUS}" \
  -m medsyn.cli.train_ccDDPM \
  --config "${CONFIG_FILE}" \
  --dataset "${DATA_DST}" \
  --outdir "${OUT_DIR}"

TRAIN_EXIT_CODE=$?

# ---------- Restore config and sync results ----------
mv "${CONFIG_BACKUP}" "${CONFIG_FILE}"

if [ ${TRAIN_EXIT_CODE} -eq 0 ]; then
  echo "[sync] Copying results to ${RESULTS_DST}"
  mkdir -p "${RESULTS_DST}"
  rsync -av "${OUT_DIR}/" "${RESULTS_DST}/"
  echo "Results saved to ${RESULTS_DST}"
else
  echo "Training failed with exit code ${TRAIN_EXIT_CODE}"
fi

# ---------- Cleanup ----------
if cd "${LOCALSCRATCH%/}/${USER}"; then
  [ -n "${MYLOCALSCRATCH:-}" ] && rm -rf --one-file-system "${MYLOCALSCRATCH}"
fi

echo "Job complete! Exit code: ${TRAIN_EXIT_CODE}"
exit ${TRAIN_EXIT_CODE}
