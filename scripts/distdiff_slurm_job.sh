#!/usr/bin/env bash
#SBATCH -J distdiff_pathmnist
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --constraint=dgx
#SBATCH --gres=gpu:4
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail

# ========================================================================
# DISTDIFF MULTI-STAGE PIPELINE FOR PATHMNIST
# ========================================================================
# Pipeline:
#   1. Train guide model on original NPZ data
#   2. Generate synthetic data in parallel (4 GPUs)
#   3. Train classifier on original + synthetic data
#
# This script follows the picasso_parallel_job.sh pattern for consistency
# ========================================================================

echo "================================================================================"
echo "🎯 DistDiff Multi-Stage Pipeline for PathMNIST"
echo "================================================================================"

# ---------- Configuration ----------
CONFIG_FILE="${1:-config/distdiff_pathmnist.yaml}"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    echo "Usage: sbatch $0 [config_file.yaml]"
    exit 1
fi

echo "Using config: $CONFIG_FILE"
echo ""

# Parse config file for key paths (simple YAML parsing)
NPZ_PATH=$(grep "npz_path:" "$CONFIG_FILE" | head -1 | awk '{print $2}')
OUTPUT_DIR=$(grep "output_dir:" "$CONFIG_FILE" | grep -v "^#" | head -1 | awk '{print $2}')
REPO_SRC=$(grep "repo_src:" "$CONFIG_FILE" | awk '{print $2}')
RESULTS_DST=$(grep "results_dst:" "$CONFIG_FILE" | awk '{print $2}')
CACHE_DIR=$(grep "cache_dir:" "$CONFIG_FILE" | grep -v "^#" | head -1 | awk '{print $2}')

# Handle null cache_dir
if [ "$CACHE_DIR" = "null" ] || [ -z "$CACHE_DIR" ]; then
    CACHE_DIR_ARG=""
    echo "Cache dir: Using default HuggingFace cache"
else
    CACHE_DIR_ARG="--cache_dir ${CACHE_DIR}"
    echo "Cache dir: ${CACHE_DIR}"
fi

echo "NPZ Dataset: $NPZ_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo ""

# ---------- LocalScratch layout ----------
MYLOCALSCRATCH="${LOCALSCRATCH%/}/${USER}/${SLURM_JOB_ID}"
WORKDIR="${MYLOCALSCRATCH}/work"
REPO_DIR="${WORKDIR}/medsyn"
DATA_DIR="${WORKDIR}/datasets"
OUT_DIR="${WORKDIR}/distdiff_output"

mkdir -p "${WORKDIR}" "${DATA_DIR}" "${OUT_DIR}"
echo "Localscratch workdir: ${WORKDIR}"
echo ""

# ---------- 1) Repo to localscratch ----------
echo "================================================================================"
echo "📦 Staging Repository to LocalScratch"
echo "================================================================================"

if [ -d "${REPO_DIR}" ] && [ -f "${REPO_DIR}/pyproject.toml" ]; then
    echo "[repo] Found at ${REPO_DIR}. Skip copy."
else
    echo "[repo] Copying from ${REPO_SRC} ..."
    mkdir -p "${REPO_DIR}"
    rsync -a "${REPO_SRC}/" "${REPO_DIR}/"
fi
echo ""

# ---------- 2) Dataset to localscratch ----------
echo "================================================================================"
echo "📦 Staging Dataset to LocalScratch"
echo "================================================================================"

DATA_DST="${DATA_DIR}/pathmnist.npz"
if [ -f "${DATA_DST}" ]; then
    echo "[data] Found at ${DATA_DST}. Skip copy."
else
    echo "[data] Copying ${NPZ_PATH} -> ${DATA_DST}"
    rsync -a "${NPZ_PATH}" "${DATA_DST}"
fi
echo ""

# ---------- 3) Load conda and activate environment ----------
echo "================================================================================"
echo "🐍 Activating Conda Environment"
echo "================================================================================"

module_loaded=0
for m in miniconda3 Miniconda3 anaconda3 Anaconda3; do
    if module avail 2>/dev/null | grep -qi "^${m}[[:space:]]"; then
        module load "$m" && module_loaded=1 && break
    fi
done

if [ "$module_loaded" -eq 0 ]; then
    echo "[env] No conda module loaded; assuming conda already in PATH."
fi

# Activate medsyn-distdiff environment (NOT medsyn!)
# DistDiff requires transformers==4.19.2 which may conflict with ccDDPM
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh" || true
    conda activate medsyn-distdiff 2>/dev/null || source activate medsyn-distdiff
else
    source activate medsyn-distdiff
fi

echo "[python] $(which python)"
python -c "import sys; print('Python', sys.version.split()[0])"
python -c "import torch; print('PyTorch', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
python -c "import torch; print('CUDA devices:', torch.cuda.device_count())"

# Verify DistDiff dependencies
echo "[env] Verifying DistDiff dependencies..."
python -c "import transformers; print('Transformers:', transformers.__version__)"
if ! python -c "import transformers; assert transformers.__version__ == '4.19.2'" 2>/dev/null; then
    echo "⚠️  WARNING: Expected transformers==4.19.2, but found different version"
    echo "   This may cause issues with DistDiff. Please use medsyn-distdiff environment."
fi
python -c "import timm; print('TIMM:', timm.__version__)" || echo "⚠️  TIMM not found"
python -c "import open_clip; print('OpenCLIP: OK')" || echo "⚠️  OpenCLIP not found"
echo ""

# ---------- 4) Display GPU information ----------
echo "================================================================================"
echo "🖥️  GPU Information"
echo "================================================================================"
nvidia-smi || true
echo ""

# ---------- 5) Change to repo directory ----------
cd "${REPO_DIR}"

# ---------- 6) Set environment variables ----------
export DATASET_PATH="${DATA_DST}"
export OUTPUT_DIR="${OUT_DIR}"
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 4 GPUs
export NCCL_DEBUG=INFO
export IS_SUPERCOMPUTER=1

# ========================================================================
# STAGE 1: TRAIN GUIDE MODEL
# ========================================================================
echo ""
echo "================================================================================"
echo "🏋️  STAGE 1: Training Guide Model on Original Data"
echo "================================================================================"

CHECKPOINT_DIR="${OUT_DIR}/checkpoints/guide_model"
mkdir -p "${CHECKPOINT_DIR}"

echo "[stage1] Training guide model (ResNet50) on NPZ dataset..."
echo "[stage1] Dataset: ${DATA_DST}"
echo "[stage1] Checkpoint: ${CHECKPOINT_DIR}"
echo ""

# Use only first GPU for guide model training
# PathMNIST paper: use CLIP-ViT-B/32 baseline (not ResNet50)
CUDA_VISIBLE_DEVICES=0 python medsyn/models/distdiff/train.py \
    -a open_clip_vit_b32 \
    -d pathmnist_npz \
    --data_dir "${DATA_DST}" \
    --checkpoint "${CHECKPOINT_DIR}" \
    --manualSeed 23102003 \
    --train-batch-size 64 \
    --val-batch-size 64 \
    --lr 0.1 \
    --epochs 100

STAGE1_EXIT_CODE=$?

if [ ${STAGE1_EXIT_CODE} -ne 0 ]; then
    echo "❌ Stage 1 failed with exit code ${STAGE1_EXIT_CODE}"
    exit ${STAGE1_EXIT_CODE}
fi

echo "✅ Stage 1 completed successfully"
echo ""

# Verify guide model checkpoint exists
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
echo "🎨 STAGE 2: Generating Synthetic Data (Parallel across 4 GPUs)"
echo "================================================================================"

SYNTH_DATA_DIR="${OUT_DIR}/synthetic_data"
mkdir -p "${SYNTH_DATA_DIR}"

echo "[stage2] Generating 5x expanded dataset using Stable Diffusion..."
echo "[stage2] Guide model: ${GUIDE_MODEL_PATH}"
echo "[stage2] Output: ${SYNTH_DATA_DIR}"
echo "[stage2] Running 4 parallel generation jobs (1 per GPU)..."
echo ""

# Launch 4 parallel generation jobs (one per GPU)
# Each job processes 1/4 of the dataset
for split in 0 1 2 3; do
    echo "  → Launching generation job ${split}/4 on GPU ${split}..."

    CUDA_VISIBLE_DEVICES=${split} python medsyn/models/distdiff/generate_data.py \
        --guidance_type=transform_guidance \
        -a open_clip_vit_b32 \
        -d pathmnist_npz \
        --data_dir "${DATA_DST}" \
        --output_dir "${SYNTH_DATA_DIR}/split_${split}" \
        --pretrained_model_name_or_path "CompVis/stable-diffusion-v1-4" \
        ${CACHE_DIR_ARG} \
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
        --total_split 4 \
        --split ${split} \
        > "${OUT_DIR}/logs/generation_split_${split}.log" 2>&1 &
done

# Wait for all generation jobs to complete
echo ""
echo "  Waiting for all 4 generation jobs to complete..."
wait

STAGE2_EXIT_CODE=$?

if [ ${STAGE2_EXIT_CODE} -ne 0 ]; then
    echo "❌ Stage 2 failed with exit code ${STAGE2_EXIT_CODE}"
    exit ${STAGE2_EXIT_CODE}
fi

echo "✅ Stage 2 completed successfully"
echo ""

# Verify synthetic data was generated
SYNTH_COUNT=0
for split in 0 1 2 3; do
    if [ -d "${SYNTH_DATA_DIR}/split_${split}" ]; then
        SYNTH_COUNT=$((SYNTH_COUNT + 1))
    fi
done

if [ ${SYNTH_COUNT} -ne 4 ]; then
    echo "⚠️  Warning: Only ${SYNTH_COUNT}/4 synthetic data splits found"
fi

echo "✓ Synthetic data generation verified (${SYNTH_COUNT}/4 splits)"
echo ""

# ========================================================================
# STAGE 3: TRAIN CLASSIFIER ON EXPANDED DATA
# ========================================================================
echo ""
echo "================================================================================"
echo "🎓 STAGE 3: Training Classifier on Original + Synthetic Data"
echo "================================================================================"

EXPANDED_CHECKPOINT_DIR="${OUT_DIR}/checkpoints/classifier_on_expanded"
mkdir -p "${EXPANDED_CHECKPOINT_DIR}"

echo "[stage3] Training classifier on concatenated dataset..."
echo "[stage3] Original data: ${DATA_DST}"
echo "[stage3] Synthetic data: ${SYNTH_DATA_DIR}/split_{0,1,2,3}"
echo "[stage3] Checkpoint: ${EXPANDED_CHECKPOINT_DIR}"
echo ""

# Use single GPU for final training
# PathMNIST paper: use CLIP-ViT-B/32 baseline (not ResNet50)
CUDA_VISIBLE_DEVICES=0 python medsyn/models/distdiff/train_expanded_data_concat_original.py \
    -d pathmnist_npz \
    --data_dir "${DATA_DST}" \
    --data_expanded_dir "${SYNTH_DATA_DIR}/split_0" "${SYNTH_DATA_DIR}/split_1" "${SYNTH_DATA_DIR}/split_2" "${SYNTH_DATA_DIR}/split_3" \
    --checkpoint "${EXPANDED_CHECKPOINT_DIR}" \
    -a open_clip_vit_b32 \
    --manualSeed 23102003 \
    --train-batch-size 64 \
    --val-batch-size 64 \
    --lr 0.1 \
    --epochs 100

STAGE3_EXIT_CODE=$?

if [ ${STAGE3_EXIT_CODE} -ne 0 ]; then
    echo "❌ Stage 3 failed with exit code ${STAGE3_EXIT_CODE}"
    exit ${STAGE3_EXIT_CODE}
fi

echo "✅ Stage 3 completed successfully"
echo ""

# ========================================================================
# RESULTS SYNCHRONIZATION
# ========================================================================
echo ""
echo "================================================================================"
echo "📤 Syncing Results Back to Permanent Storage"
echo "================================================================================"

echo "Copying results from ${OUT_DIR} to ${RESULTS_DST}..."
mkdir -p "${RESULTS_DST}"
rsync -av "${OUT_DIR}/" "${RESULTS_DST}/"

echo "✅ Results successfully copied to ${RESULTS_DST}"
echo ""

# ========================================================================
# CLEANUP
# ========================================================================
echo ""
echo "================================================================================"
echo "🧹 Cleaning Up LocalScratch"
echo "================================================================================"

if cd "${LOCALSCRATCH%/}/${USER}"; then
    [ -n "${MYLOCALSCRATCH:-}" ] && rm -rf --one-file-system "${MYLOCALSCRATCH}"
    echo "✓ Cleanup complete"
fi
echo ""

# ========================================================================
# SUMMARY
# ========================================================================
echo ""
echo "================================================================================"
echo "🎉 DistDiff Pipeline Complete!"
echo "================================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Dataset: PathMNIST (NPZ format)"
echo "Results location: ${RESULTS_DST}"
echo ""
echo "Pipeline Summary:"
echo "  ✅ Stage 1: Guide model trained"
echo "  ✅ Stage 2: Synthetic data generated (5x expansion)"
echo "  ✅ Stage 3: Classifier trained on expanded data"
echo ""
echo "Checkpoints:"
echo "  • Guide model: ${RESULTS_DST}/checkpoints/guide_model/"
echo "  • Expanded classifier: ${RESULTS_DST}/checkpoints/classifier_on_expanded/"
echo ""
echo "Synthetic data: ${RESULTS_DST}/synthetic_data/"
echo "================================================================================"

exit 0
