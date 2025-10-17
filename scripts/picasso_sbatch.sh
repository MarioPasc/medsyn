#!/usr/bin/env bash
#SBATCH -J ccddpm_pathmnist
#SBATCH --time=20:00:00
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
MYLOCALSCRATCH="${LOCALSCRATCH}/${USER}/${SLURM_JOB_ID}"
WORKDIR="${MYLOCALSCRATCH}/work"
REPO_DIR="${WORKDIR}/medsyn"
DATA_DIR="${WORKDIR}/datasets"
OUT_DIR="${WORKDIR}/results"
ENV_DIR="${WORKDIR}/conda_env"

mkdir -p "${WORKDIR}" "${DATA_DIR}" "${OUT_DIR}"
echo "Localscratch workdir: ${WORKDIR}"

# ---------- 1) Repo to localscratch (skip if present) ----------
if [ -d "${REPO_DIR}" ] && [ -f "${REPO_DIR}/pyproject.toml" ]; then
  echo "[repo] found at ${REPO_DIR}. skip copy."
else
  echo "[repo] copying from ${REPO_SRC} ..."
  mkdir -p "${REPO_DIR}"
  rsync -a "${REPO_SRC}/" "${REPO_DIR}/"
fi

# ---------- 2) Dataset to localscratch (skip if present) ----------
DATA_DST="${DATA_DIR}/PathMNIST.npz"
if [ -f "${DATA_DST}" ]; then
  echo "[data] found at ${DATA_DST}. skip copy."
else
  echo "[data] copying ${DATA_SRC} -> ${DATA_DST}"
  rsync -a "${DATA_SRC}" "${DATA_DST}"
fi

# ---------- 3) Conda env in localscratch (reuse if present) ----------
if module avail 2>/dev/null | grep -qiE 'conda|anaconda|miniconda|miniforge'; then
  module load Miniconda3 || module load Anaconda || module load miniforge || true
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if [ -d "${ENV_DIR}" ] && [ -f "${ENV_DIR}/conda-meta/history" ]; then
  echo "[env] found at ${ENV_DIR}. reusing."
else
  echo "[env] creating at ${ENV_DIR}"
  conda create -y -p "${ENV_DIR}" python=3.10 || conda create -y -p "${ENV_DIR}" python=3.11
fi

conda activate "${ENV_DIR}"

# Install project only if not installed yet
if python -c "import importlib; importlib.import_module('medsyn')" 2>/dev/null; then
  echo "[pip] medsyn already importable. skip install."
else
  echo "[pip] editable install of medsyn"
  python -m pip install --upgrade pip wheel
  python -m pip install -e "${REPO_DIR}"
  # Optional: add framework deps here if your wheels are not preinstalled
  # python -m pip install 'torch==<ver>+cu118' torchvision --index-url https://download.pytorch.org/whl/cu118
  # python -m pip install diffusers accelerate
fi

# ---------- 4) Run training ----------
cd "${REPO_DIR}"
export DATASET_PATH="${DATA_DST}"
export OUTPUT_DIR="${OUT_DIR}"

nvidia-smi || true
python - <<'PY'
import torch
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
PY

ccddpm-train --config $REPO_SRC/config/picasso_cfg.yaml

# ---------- 5) Sync results back ----------
mkdir -p "${RESULTS_DST}"
rsync -a "${OUT_DIR}/" "${RESULTS_DST}/"

# ---------- 6) Cleanup ----------
if cd "${LOCALSCRATCH}/${USER}"; then
  [ -n "${MYLOCALSCRATCH:-}" ] && rm -rf --one-file-system "${MYLOCALSCRATCH}"
fi
