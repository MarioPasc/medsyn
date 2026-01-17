#!/usr/bin/env bash
# ============================================================================
# IJCNN 2026 - Classifier Training Script
# Trains ResNet-50, CLIP-ViT-B16, DINOv3 on different augmentation regimes
# ============================================================================

set -euo pipefail

# ---------- PATHS (MODIFY THESE) ----------
REPO_DIR="/path/to/medsyn"  # TODO: Update
DATA_BASE="/path/to/data"  # TODO: Base path for datasets
RESULTS_BASE="/path/to/results/classifiers"  # TODO: Output directory
SEEDS="42 123 456 789 1337"  # 5 seeds for statistical robustness

# Dataset to process (pathmnist or dermamnist)
DATASET="${1:-pathmnist}"
NUM_CLASSES=9
if [ "${DATASET}" == "dermamnist" ]; then
  NUM_CLASSES=7
fi

echo "================================================================================"
echo "IJCNN 2026 - Classifier Training for ${DATASET}"
echo "================================================================================"

cd "${REPO_DIR}"

# Classifier architectures
ARCHITECTURES="resnet50 clip_vit_b16 dinov3_vitb16"

# Training regimes
# M0: Real only
# M1: Real + traditional augmentation
# M2: Real + DistDiff (PathMNIST only)
# M3: Real + Ours (CFG-MedSyn)

for arch in ${ARCHITECTURES}; do
  echo ""
  echo "========================================"
  echo "Training ${arch} classifiers"
  echo "========================================"

  for seed in ${SEEDS}; do
    echo "Seed: ${seed}"

    # M0: Real only
    echo "  M0: Real only..."
    python -m medsyn.cli.train_classifier \
      --config "config/classification/${DATASET}_real_only.yaml" \
      --model "config/classification/${arch}.yaml" \
      --output "${RESULTS_BASE}/${DATASET}/M0/${arch}/seed_${seed}" \
      --seed "${seed}"

    # M1: Real + traditional augmentation
    echo "  M1: Real + traditional aug..."
    python -m medsyn.cli.train_classifier \
      --config "config/classification/${DATASET}_real_plus_trad_aug.yaml" \
      --model "config/classification/${arch}.yaml" \
      --output "${RESULTS_BASE}/${DATASET}/M1/${arch}/seed_${seed}" \
      --seed "${seed}"

    # M3: Real + CFG-MedSyn (for each temperature)
    for temp in 1.0 1.5 2.0 2.5 3.0; do
      echo "  M3: Real + CFG-MedSyn (t=${temp})..."
      python -m medsyn.cli.train_classifier \
        --config "config/classification/${DATASET}_real_plus_synth.yaml" \
        --model "config/classification/${arch}.yaml" \
        --synth_data "${DATA_BASE}/${DATASET}_t${temp}/synth/train/${DATASET}_train_synth.npz" \
        --output "${RESULTS_BASE}/${DATASET}/M3_t${temp}/${arch}/seed_${seed}" \
        --seed "${seed}"
    done

    # M2: Real + DistDiff (PathMNIST only)
    if [ "${DATASET}" == "pathmnist" ]; then
      echo "  M2: Real + DistDiff..."
      python -m medsyn.cli.train_classifier \
        --config "config/classification/${DATASET}_real_plus_synth.yaml" \
        --model "config/classification/${arch}.yaml" \
        --synth_data "${DATA_BASE}/distdiff/synth.npz" \
        --output "${RESULTS_BASE}/${DATASET}/M2/${arch}/seed_${seed}" \
        --seed "${seed}"
    fi
  done
done

echo ""
echo "================================================================================"
echo "Classifier training complete!"
echo "Results saved to: ${RESULTS_BASE}"
echo "================================================================================"
