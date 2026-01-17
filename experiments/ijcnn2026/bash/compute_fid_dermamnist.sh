#!/usr/bin/env bash
# ============================================================================
# IJCNN 2026 - FID Computation for DermaMNIST
# ============================================================================

set -euo pipefail

# ---------- PATHS (MODIFY THESE) ----------
GROUND_TRUTH="/path/to/dermamnist/merged.npz"  # TODO: Update
CFG_SYNTH_BASE="/path/to/results"  # TODO: Base path for cfg-medsyn results
OUTPUT_DIR="/path/to/results/fid"  # TODO: Output directory for FID results
FID_WEIGHTS="/path/to/fid_weights"  # TODO: Path to InceptionV3 weights
DEVICE="cuda:0"

mkdir -p "${OUTPUT_DIR}"

echo "================================================================================"
echo "IJCNN 2026 - FID Computation for DermaMNIST"
echo "================================================================================"

# Compute FID for each temperature
for temp in 1.0 1.5 2.0 2.5 3.0; do
  CFG_SYNTH="${CFG_SYNTH_BASE}/dermamnist_t${temp}/synth/train/dermamnist_train_synth.npz"

  if [ -f "${CFG_SYNTH}" ]; then
    echo "Computing FID for temperature ${temp}..."
    python -m medsyn.analysis.ddpm_performance.metrics.fid \
      --ground-truth "${GROUND_TRUTH}" \
      --model "CFG_t${temp}" "${CFG_SYNTH}" \
      --output "${OUTPUT_DIR}/fid_dermamnist_t${temp}.csv" \
      --counts 500 \
      --num-classes 7 \
      --weights-dir "${FID_WEIGHTS}" \
      --device "${DEVICE}" \
      --verbose
  else
    echo "Warning: ${CFG_SYNTH} not found, skipping temperature ${temp}"
  fi
done

echo "FID computation complete. Results in ${OUTPUT_DIR}"
