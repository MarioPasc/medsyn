#!/usr/bin/env bash
set -euo pipefail

# ========================================================================
# DISTDIFF LOCAL MULTI-GPU PIPELINE FOR PATHMNIST
# ========================================================================
# This script runs the DistDiff pipeline on a local machine with multiple GPUs
# (simpler version without SLURM/LocalScratch complexity)
#
# IMPORTANT: Activate the medsyn-distdiff environment before running!
#   conda activate medsyn-distdiff
#
# This script requires transformers==4.19.2 which conflicts with ccDDPM.
# See ENVIRONMENTS.md for details on setting up separate environments.
#
# Pipeline:
#   1. Train guide model on original NPZ data
#   2. Generate synthetic data in parallel across GPUs
#   3. Train classifier on original + synthetic data
#
# Usage:
#   conda activate medsyn-distdiff
#   bash scripts/distdiff_local_job.sh [config_file.yaml] [num_gpus]
#
# Example:
#   conda activate medsyn-distdiff
#   bash scripts/distdiff_local_job.sh config/distdiff_pathmnist.yaml 4
# ========================================================================

echo "================================================================================"
echo "🎯 DistDiff Local Pipeline for PathMNIST"
echo "================================================================================"

# ---------- Arguments ----------
CONFIG_FILE="${1:-config/distdiff_pathmnist.yaml}"
NUM_GPUS="${2:-4}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    echo "Usage: $0 [config_file.yaml] [num_gpus]"
    exit 1
fi

echo "Config: $CONFIG_FILE"
echo "Number of GPUs: $NUM_GPUS"
echo ""

# ---------- Parse config ----------
NPZ_PATH=$(grep "npz_path:" "$CONFIG_FILE" | head -1 | awk '{print $2}')
OUTPUT_DIR=$(grep "output_dir:" "$CONFIG_FILE" | grep -v "^#" | head -1 | awk '{print $2}')

echo "NPZ Dataset: $NPZ_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo ""

# Verify dataset exists
if [ ! -f "$NPZ_PATH" ]; then
    echo "❌ Error: NPZ dataset not found: $NPZ_PATH"
    exit 1
fi

# Create output directories
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints/guide_model"
SYNTH_DATA_DIR="${OUTPUT_DIR}/synthetic_data"
EXPANDED_CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints/classifier_on_expanded"
LOGS_DIR="${OUTPUT_DIR}/logs"

mkdir -p "${CHECKPOINT_DIR}" "${SYNTH_DATA_DIR}" "${EXPANDED_CHECKPOINT_DIR}" "${LOGS_DIR}"

# ---------- Environment check ----------
echo "================================================================================"
echo "🐍 Environment Check"
echo "================================================================================"

if ! command -v python &> /dev/null; then
    echo "❌ Error: python not found. Please activate conda environment 'medsyn-distdiff'"
    echo "   Run: conda activate medsyn-distdiff"
    exit 1
fi

echo "[python] $(which python)"
python -c "import sys; print('Python', sys.version.split()[0])"
python -c "import torch; print('PyTorch', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('CUDA devices:', torch.cuda.device_count())"

# Verify DistDiff dependencies
echo ""
echo "[env] Verifying DistDiff dependencies..."
python -c "import transformers; print('Transformers:', transformers.__version__)"
if ! python -c "import transformers; assert transformers.__version__ == '4.19.2'" 2>/dev/null; then
    echo "⚠️  WARNING: Expected transformers==4.19.2, but found different version"
    echo "   DistDiff may not work correctly. Please install with:"
    echo "   conda activate medsyn-distdiff"
    echo "   pip install -e \".[distdiff]\""
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi
python -c "import timm; print('TIMM:', timm.__version__)" || echo "⚠️  WARNING: TIMM not found"
python -c "import open_clip; print('OpenCLIP: OK')" || echo "⚠️  WARNING: OpenCLIP not found"
python -c "import accelerate; print('Accelerate:', accelerate.__version__)" || echo "⚠️  WARNING: Accelerate not found"
echo ""

# Verify GPU count
AVAILABLE_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
if [ "$AVAILABLE_GPUS" -lt "$NUM_GPUS" ]; then
    echo "⚠️  Warning: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available"
    echo "   Using $AVAILABLE_GPUS GPUs instead"
    NUM_GPUS=$AVAILABLE_GPUS
fi

# ---------- GPU information ----------
echo "================================================================================"
echo "🖥️  GPU Information"
echo "================================================================================"
nvidia-smi || true
echo ""

# ========================================================================
# STAGE 1: TRAIN GUIDE MODEL
# ========================================================================
echo ""
echo "================================================================================"
echo "🏋️  STAGE 1: Training Guide Model on Original Data"
echo "================================================================================"

echo "[stage1] Training guide model (ResNet50) on NPZ dataset..."
echo "[stage1] Dataset: ${NPZ_PATH}"
echo "[stage1] Checkpoint: ${CHECKPOINT_DIR}"
echo ""

# Use first GPU for guide model training
# PathMNIST paper: use CLIP-ViT-B/32 baseline (not ResNet50)
CUDA_VISIBLE_DEVICES=0 python medsyn/models/distdiff/train.py \
    -a open_clip_vit_b32 \
    -d pathmnist_npz \
    --data_dir "${NPZ_PATH}" \
    --checkpoint "${CHECKPOINT_DIR}" \
    --manualSeed 23102003 \
    --train-batch-size 64 \
    --val-batch-size 64 \
    --lr 0.1 \
    --epochs 100 \
    2>&1 | tee "${LOGS_DIR}/stage1_train_guide.log"

STAGE1_EXIT_CODE=${PIPESTATUS[0]}

if [ ${STAGE1_EXIT_CODE} -ne 0 ]; then
    echo "❌ Stage 1 failed with exit code ${STAGE1_EXIT_CODE}"
    exit ${STAGE1_EXIT_CODE}
fi

echo "✅ Stage 1 completed successfully"
echo ""

# Verify guide model checkpoint
GUIDE_MODEL_PATH="${CHECKPOINT_DIR}/model_best.pth.tar"
if [ ! -f "${GUIDE_MODEL_PATH}" ]; then
    echo "❌ Error: Guide model checkpoint not found at ${GUIDE_MODEL_PATH}"
    exit 1
fi
echo "✓ Guide model checkpoint verified: ${GUIDE_MODEL_PATH}"
echo ""

# ========================================================================
# STAGE 2: GENERATE SYNTHETIC DATA (PARALLEL)
# ========================================================================
echo ""
echo "================================================================================"
echo "🎨 STAGE 2: Generating Synthetic Data (Parallel across ${NUM_GPUS} GPUs)"
echo "================================================================================"

echo "[stage2] Generating 5x expanded dataset using Stable Diffusion..."
echo "[stage2] Guide model: ${GUIDE_MODEL_PATH}"
echo "[stage2] Output: ${SYNTH_DATA_DIR}"
echo "[stage2] Running ${NUM_GPUS} parallel generation jobs (1 per GPU)..."
echo ""

# Launch parallel generation jobs
PIDS=()
for (( split=0; split<NUM_GPUS; split++ )); do
    echo "  → Launching generation job ${split}/${NUM_GPUS} on GPU ${split}..."

    CUDA_VISIBLE_DEVICES=${split} python medsyn/models/distdiff/generate_data.py \
        --guidance_type=transform_guidance \
        -a open_clip_vit_b32 \
        -d pathmnist_npz \
        --data_dir "${NPZ_PATH}" \
        --output_dir "${SYNTH_DATA_DIR}/split_${split}" \
        --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
        --gradient_checkpointing \
        --K 5 \
        --train_batch_size 1 \
        --optimize_targets "global_prototype-local_prototype" \
        --strength 0.2 \
        --num_images_per_prompt 5 \
        --guidance_step 10 \
        --guidance_period 2 \
        --encoder_weight_path "${GUIDE_MODEL_PATH}" \
        --guidance_scale 7.5 \
        --constraint_value 0.2 \
        --rho 10.0 \
        --total_split ${NUM_GPUS} \
        --split ${split} \
        > "${LOGS_DIR}/generation_split_${split}.log" 2>&1 &

    PIDS+=($!)
done

# Wait for all generation jobs to complete
echo ""
echo "  Waiting for all ${NUM_GPUS} generation jobs to complete..."
echo "  (This may take several hours depending on dataset size and GPU speed)"
echo ""

# Monitor progress
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    wait $pid
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "❌ Generation job $i (PID $pid) failed with exit code $exit_code"
        # Kill remaining jobs
        for remaining_pid in "${PIDS[@]}"; do
            kill $remaining_pid 2>/dev/null || true
        done
        exit $exit_code
    else
        echo "  ✓ Generation job $i completed successfully"
    fi
done

echo ""
echo "✅ Stage 2 completed successfully"
echo ""

# Verify synthetic data
SYNTH_COUNT=0
for (( split=0; split<NUM_GPUS; split++ )); do
    if [ -d "${SYNTH_DATA_DIR}/split_${split}" ]; then
        SYNTH_COUNT=$((SYNTH_COUNT + 1))
    fi
done

if [ ${SYNTH_COUNT} -ne ${NUM_GPUS} ]; then
    echo "⚠️  Warning: Only ${SYNTH_COUNT}/${NUM_GPUS} synthetic data splits found"
fi

echo "✓ Synthetic data generation verified (${SYNTH_COUNT}/${NUM_GPUS} splits)"
echo ""

# ========================================================================
# STAGE 3: TRAIN CLASSIFIER ON EXPANDED DATA
# ========================================================================
echo ""
echo "================================================================================"
echo "🎓 STAGE 3: Training Classifier on Original + Synthetic Data"
echo "================================================================================"

echo "[stage3] Training classifier on concatenated dataset..."
echo "[stage3] Original data: ${NPZ_PATH}"
echo "[stage3] Synthetic data splits: ${NUM_GPUS}"
echo "[stage3] Checkpoint: ${EXPANDED_CHECKPOINT_DIR}"
echo ""

# Build data_expanded_dir arguments
EXPANDED_DIRS=""
for (( split=0; split<NUM_GPUS; split++ )); do
    EXPANDED_DIRS="${EXPANDED_DIRS} ${SYNTH_DATA_DIR}/split_${split}"
done

# Use single GPU for final training
# PathMNIST paper: use CLIP-ViT-B/32 baseline (not ResNet50)
CUDA_VISIBLE_DEVICES=0 python medsyn/models/distdiff/train_expanded_data_concat_original.py \
    -d pathmnist_npz \
    --data_dir "${NPZ_PATH}" \
    --data_expanded_dir ${EXPANDED_DIRS} \
    --checkpoint "${EXPANDED_CHECKPOINT_DIR}" \
    -a open_clip_vit_b32 \
    --manualSeed 23102003 \
    --train-batch-size 64 \
    --val-batch-size 64 \
    --lr 0.1 \
    --epochs 100 \
    2>&1 | tee "${LOGS_DIR}/stage3_train_expanded.log"

STAGE3_EXIT_CODE=${PIPESTATUS[0]}

if [ ${STAGE3_EXIT_CODE} -ne 0 ]; then
    echo "❌ Stage 3 failed with exit code ${STAGE3_EXIT_CODE}"
    exit ${STAGE3_EXIT_CODE}
fi

echo "✅ Stage 3 completed successfully"
echo ""

# ========================================================================
# SUMMARY
# ========================================================================
echo ""
echo "================================================================================"
echo "🎉 DistDiff Pipeline Complete!"
echo "================================================================================"
echo "Dataset: PathMNIST (NPZ format)"
echo "Output location: ${OUTPUT_DIR}"
echo ""
echo "Pipeline Summary:"
echo "  ✅ Stage 1: Guide model trained"
echo "  ✅ Stage 2: Synthetic data generated (5x expansion, ${NUM_GPUS} GPUs)"
echo "  ✅ Stage 3: Classifier trained on expanded data"
echo ""
echo "Checkpoints:"
echo "  • Guide model: ${CHECKPOINT_DIR}/"
echo "  • Expanded classifier: ${EXPANDED_CHECKPOINT_DIR}/"
echo ""
echo "Synthetic data: ${SYNTH_DATA_DIR}/"
echo "Logs: ${LOGS_DIR}/"
echo "================================================================================"
echo ""
echo "To evaluate the classifier:"
echo "  CUDA_VISIBLE_DEVICES=0 python medsyn/models/distdiff/train_expanded_data_concat_original.py \\"
echo "    -d pathmnist_npz \\"
echo "    --data_dir ${NPZ_PATH} \\"
echo "    --resume ${EXPANDED_CHECKPOINT_DIR}/model_best.pth.tar \\"
echo "    --evaluate"
echo ""

exit 0
