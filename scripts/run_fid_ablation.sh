#!/bin/bash

# Exit on error
set -e

# Ablation study FID computation

# Common variables
GT_PATH="/media/hddb/mario/data/medsyn/merged.npz"
WEIGHTS_DIR="/media/hddb/mario/data/medsyn/pretrained/fid/"
SAMPLES_DIR="/media/hddb/mario/data/medsyn/ablation_study"
OUTPUT_DIR="/media/hddb/mario/results/medsyn"
DEVICE="cuda:0"
COUNTS=1000

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

echo "==================================================="
echo "Starting FID jobs for ablation study on $DEVICE"
echo "==================================================="

# Job 1: exp1_no_snr_classweight_temp2
echo "[1/3] Processing exp1_no_snr_classweight_temp2 on $DEVICE..."
python medsyn/analysis/ddpm_performance/metrics/fid.py \
    --ground-truth "$GT_PATH" \
    --model exp1_no_snr_classweight_temp2 "$SAMPLES_DIR/exp1_no_snr_classweight_temp2.npz" \
    --output "$OUTPUT_DIR/fid_exp1_no_snr_classweight_temp2.csv" \
    --counts $COUNTS \
    --weights-dir "$WEIGHTS_DIR" \
    --verbose \
    --device "$DEVICE"

# Job 2: exp2_snr_no_classweight
echo "[2/3] Processing exp2_snr_no_classweight on $DEVICE..."
python medsyn/analysis/ddpm_performance/metrics/fid.py \
    --ground-truth "$GT_PATH" \
    --model exp2_snr_no_classweight "$SAMPLES_DIR/exp2_snr_no_classweight.npz" \
    --output "$OUTPUT_DIR/fid_exp2_snr_no_classweight.csv" \
    --counts $COUNTS \
    --weights-dir "$WEIGHTS_DIR" \
    --verbose \
    --device "$DEVICE"

# Job 3: exp3_baseline_no_weighting
echo "[3/3] Processing exp3_baseline_no_weighting on $DEVICE..."
python medsyn/analysis/ddpm_performance/metrics/fid.py \
    --ground-truth "$GT_PATH" \
    --model exp3_baseline_no_weighting "$SAMPLES_DIR/exp3_baseline_no_weighting.npz" \
    --output "$OUTPUT_DIR/fid_exp3_baseline_no_weighting.csv" \
    --counts $COUNTS \
    --weights-dir "$WEIGHTS_DIR" \
    --verbose \
    --device "$DEVICE"

echo "==================================================="
echo "All ablation study FID jobs completed successfully."
echo "==================================================="
