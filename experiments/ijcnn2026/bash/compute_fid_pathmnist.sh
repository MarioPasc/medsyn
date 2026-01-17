#!/usr/bin/env bash
# ============================================================================
# IJCNN 2026 - FID Computation for PathMNIST
# ============================================================================

set -euo pipefail

# ---------- PATHS (MODIFY THESE) ----------
GROUND_TRUTH="/path/to/pathmnist/merged.npz"  # TODO: Update
CFG_SYNTH_BASE="/path/to/results"  # TODO: Base path for cfg-medsyn results
DISTDIFF_SYNTH="/path/to/distdiff/synth.npz"  # TODO: Path to DistDiff synthetic samples
OUTPUT_DIR="/path/to/results/fid"  # TODO: Output directory for FID results
FID_WEIGHTS="/path/to/fid_weights"  # TODO: Path to InceptionV3 weights
DEVICE="cuda:0"

mkdir -p "${OUTPUT_DIR}"

echo "================================================================================"
echo "IJCNN 2026 - FID Computation for PathMNIST"
echo "================================================================================"

# Compute FID for each temperature
for temp in 1.0 1.5 2.0 2.5 3.0; do
  CFG_SYNTH="${CFG_SYNTH_BASE}/pathmnist_t${temp}/synth/train/pathmnist_train_synth.npz"

  if [ -f "${CFG_SYNTH}" ]; then
    echo "Computing FID for temperature ${temp}..."
    python -m medsyn.analysis.ddpm_performance.metrics.fid \
      --ground-truth "${GROUND_TRUTH}" \
      --model "CFG_t${temp}" "${CFG_SYNTH}" \
      --output "${OUTPUT_DIR}/fid_pathmnist_t${temp}.csv" \
      --counts 500 \
      --weights-dir "${FID_WEIGHTS}" \
      --device "${DEVICE}" \
      --verbose
  else
    echo "Warning: ${CFG_SYNTH} not found, skipping temperature ${temp}"
  fi
done

# Compute FID for DistDiff baseline
if [ -f "${DISTDIFF_SYNTH}" ]; then
  echo "Computing FID for DistDiff baseline..."
  python -m medsyn.analysis.ddpm_performance.metrics.fid \
    --ground-truth "${GROUND_TRUTH}" \
    --model "DistDiff" "${DISTDIFF_SYNTH}" \
    --output "${OUTPUT_DIR}/fid_pathmnist_distdiff.csv" \
    --counts 500 \
    --weights-dir "${FID_WEIGHTS}" \
    --device "${DEVICE}" \
    --verbose
fi

echo "FID computation complete. Results in ${OUTPUT_DIR}"
