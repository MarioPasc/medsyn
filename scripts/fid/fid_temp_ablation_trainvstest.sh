#!/bin/bash

# Exit on error
set -e

# Ablation study FID computation

# Common variables
GT_PATH="/media/hddb/mario/data/medsyn/merged.npz"
WEIGHTS_DIR="/media/hddb/mario/data/medsyn/pretrained/fid/"
SAMPLES_DIR="/media/hddb/mario/data/medsyn/ablation_study"
# Split to use for real ground truth samples (train/val/test)
SPLIT="test"
OUTPUT_DIR="/media/hddb/mario/results/medsyn/not_considering_minSNR/fid/$SPLIT"
DEVICE="cuda:1"
# Using counts=2048 for statistically stable FID (Heusel et al. recommend >=2048 samples)
# With 10K synthetic samples per class: floor(10000/2048) = 4 subsets
# Total samples used: 4 * 2048 = 8192, discarded: 10000 - 8192 = 1808 per class
COUNTS=2048


# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

echo "==================================================="
echo "Starting FID jobs for ablation study on $DEVICE"
echo "Ground truth split: $SPLIT"
echo "Counts per subset: $COUNTS"
echo "==================================================="


python medsyn/analysis/ddpm_performance/metrics/fid.py \
    --ground-truth $GT_PATH \
    --split $SPLIT \
    --model temp05 "$SAMPLES_DIR/not_considering_minSNR/temp05_inference.npz" \
    --output "$OUTPUT_DIR/temp05_fid.csv" \
    --counts $COUNTS \
    --weights-dir "$WEIGHTS_DIR" \
    --verbose \
    --device $DEVICE

python medsyn/analysis/ddpm_performance/metrics/fid.py \
    --ground-truth $GT_PATH \
    --split $SPLIT \
    --model temp10 "$SAMPLES_DIR/not_considering_minSNR/temp10_inference.npz" \
    --output "$OUTPUT_DIR/temp10_fid.csv" \
    --counts $COUNTS \
    --weights-dir "$WEIGHTS_DIR" \
    --verbose \
    --device $DEVICE

python medsyn/analysis/ddpm_performance/metrics/fid.py \
    --ground-truth $GT_PATH \
    --split $SPLIT \
    --model temp15 "$SAMPLES_DIR/not_considering_minSNR/temp15_inference.npz" \
    --output "$OUTPUT_DIR/temp15_fid.csv" \
    --counts $COUNTS \
    --weights-dir "$WEIGHTS_DIR" \
    --verbose \
    --device $DEVICE

python medsyn/analysis/ddpm_performance/metrics/fid.py \
    --ground-truth $GT_PATH \
    --split $SPLIT \
    --model temp20 "$SAMPLES_DIR/not_considering_minSNR/temp20_inference.npz" \
    --output "$OUTPUT_DIR/temp20_fid.csv" \
    --counts $COUNTS \
    --weights-dir "$WEIGHTS_DIR" \
    --verbose \
    --device $DEVICE

python medsyn/analysis/ddpm_performance/metrics/fid.py \
    --ground-truth $GT_PATH \
    --split $SPLIT \
    --model temp25 "$SAMPLES_DIR/not_considering_minSNR/temp25_inference.npz" \
    --output "$OUTPUT_DIR/temp25_fid.csv" \
    --counts $COUNTS \
    --weights-dir "$WEIGHTS_DIR" \
    --verbose \
    --device $DEVICE

python medsyn/analysis/ddpm_performance/metrics/fid.py \
    --ground-truth $GT_PATH \
    --split $SPLIT \
    --model temp30 "$SAMPLES_DIR/not_considering_minSNR/temp30_inference.npz" \
    --output "$OUTPUT_DIR/temp30_fid.csv" \
    --counts $COUNTS \
    --weights-dir "$WEIGHTS_DIR" \
    --verbose \
    --device $DEVICE 

echo "==================================================="
echo "All ablation study FID jobs completed successfully."
echo "==================================================="
