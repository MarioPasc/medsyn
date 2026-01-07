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
mkdir -p "$OUTPUT_DIR/not_considering_minSNR/fid"

echo "==================================================="
echo "Starting FID jobs for ablation study on $DEVICE"
echo "==================================================="


python medsyn/analysis/ddpm_performance/metrics/fid.py \
    --ground-truth $GT_PATH \
    --model temp05 "$SAMPLES_DIR/not_considering_minSNR/temp05_inference.npz" \
    --output "$OUTPUT_DIR/not_considering_minSNR/fid/temp05_fid.csv" \
    --counts $COUNTS \
    --weights-dir "$WEIGHTS_DIR" \
    --verbose \
    --device $DEVICE 

python medsyn/analysis/ddpm_performance/metrics/fid.py \
    --ground-truth $GT_PATH \
    --model temp10 "$SAMPLES_DIR/not_considering_minSNR/temp10_inference.npz" \
    --output "$OUTPUT_DIR/not_considering_minSNR/fid/temp10_fid.csv" \
    --counts $COUNTS \
    --weights-dir "$WEIGHTS_DIR" \
    --verbose \
    --device $DEVICE 

python medsyn/analysis/ddpm_performance/metrics/fid.py \
    --ground-truth $GT_PATH \
    --model temp15 "$SAMPLES_DIR/not_considering_minSNR/temp15_inference.npz" \
    --output "$OUTPUT_DIR/not_considering_minSNR/fid/temp15_fid.csv" \
    --counts $COUNTS \
    --weights-dir "$WEIGHTS_DIR" \
    --verbose \
    --device $DEVICE 

python medsyn/analysis/ddpm_performance/metrics/fid.py \
    --ground-truth $GT_PATH \
    --model temp20 "$SAMPLES_DIR/not_considering_minSNR/temp20_inference.npz" \
    --output "$OUTPUT_DIR/not_considering_minSNR/fid/temp20_fid.csv" \
    --counts $COUNTS \
    --weights-dir "$WEIGHTS_DIR" \
    --verbose \
    --device $DEVICE 

python medsyn/analysis/ddpm_performance/metrics/fid.py \
    --ground-truth $GT_PATH \
    --model temp25 "$SAMPLES_DIR/not_considering_minSNR/temp25_inference.npz" \
    --output "$OUTPUT_DIR/not_considering_minSNR/fid/temp25_fid.csv" \
    --counts $COUNTS \
    --weights-dir "$WEIGHTS_DIR" \
    --verbose \
    --device $DEVICE 

python medsyn/analysis/ddpm_performance/metrics/fid.py \
    --ground-truth $GT_PATH \
    --model temp30 "$SAMPLES_DIR/not_considering_minSNR/temp30_inference.npz" \
    --output "$OUTPUT_DIR/not_considering_minSNR/fid/temp30_fid.csv" \
    --counts $COUNTS \
    --weights-dir "$WEIGHTS_DIR" \
    --verbose \
    --device $DEVICE 

echo "==================================================="
echo "All ablation study FID jobs completed successfully."
echo "==================================================="
