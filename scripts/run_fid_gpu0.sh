#!/bin/bash

# Exit on error
set -e

# GPU 0 - 4 jobs

# Common variables
GT_PATH="/media/hddb/mario/data/medsyn/merged.npz"
WEIGHTS_DIR="/media/hddb/mario/data/medsyn/pretrained/fid/"
SAMPLES_DIR="/media/hddb/mario/data/medsyn/hyperparameter_synthetic_samples"
OUTPUT_DIR="/media/hddb/mario/results/medsyn"
DEVICE="cuda:0"
COUNTS=1000

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

echo "==================================================="
echo "Starting FID jobs on GPU 0"
echo "==================================================="

# Job 1: gamma1_temp10
echo "[1/4] Processing gamma1_temp10 on $DEVICE..."
python medsyn/analysis/ddpm_performance/metrics/fid.py \
    --ground-truth "$GT_PATH" \
    --model gamma1_temp10 "$SAMPLES_DIR/gamma1_temp10_synth_samples.npz" \
    --output "$OUTPUT_DIR/fid_gamma1_temp10.csv" \
    --counts $COUNTS \
    --weights-dir "$WEIGHTS_DIR" \
    --verbose \
    --device "$DEVICE"

# Job 2: gamma1_temp15
echo "[2/4] Processing gamma1_temp15 on $DEVICE..."
python medsyn/analysis/ddpm_performance/metrics/fid.py \
    --ground-truth "$GT_PATH" \
    --model gamma1_temp15 "$SAMPLES_DIR/gamma1_temp15_synth_samples.npz" \
    --output "$OUTPUT_DIR/fid_gamma1_temp15.csv" \
    --counts $COUNTS \
    --weights-dir "$WEIGHTS_DIR" \
    --verbose \
    --device "$DEVICE"

# Job 3: gamma1_temp20
echo "[3/4] Processing gamma1_temp20 on $DEVICE..."
python medsyn/analysis/ddpm_performance/metrics/fid.py \
    --ground-truth "$GT_PATH" \
    --model gamma1_temp20 "$SAMPLES_DIR/gamma1_temp20_synth_samples.npz" \
    --output "$OUTPUT_DIR/fid_gamma1_temp20.csv" \
    --counts $COUNTS \
    --weights-dir "$WEIGHTS_DIR" \
    --verbose \
    --device "$DEVICE"

# Job 4: gamma5_temp10
echo "[4/4] Processing gamma5_temp10 on $DEVICE..."
python medsyn/analysis/ddpm_performance/metrics/fid.py \
    --ground-truth "$GT_PATH" \
    --model gamma5_temp10 "$SAMPLES_DIR/gamma5_temp10_synth_samples.npz" \
    --output "$OUTPUT_DIR/fid_gamma5_temp10.csv" \
    --counts $COUNTS \
    --weights-dir "$WEIGHTS_DIR" \
    --verbose \
    --device "$DEVICE"

echo "==================================================="
echo "All jobs on GPU 0 completed successfully."
echo "==================================================="
