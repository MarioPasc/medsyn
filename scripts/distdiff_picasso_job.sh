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

# ========================================================================
# ERROR HANDLING: Always copy results back on failure
# ========================================================================
RESULTS_DST="/mnt/home/users/tic_163_uma/mpascual/fscratch/results/PathMNIST_DistDiff"
OUT_DIR="/mnt/home/users/tic_163_uma/mpascual/fscratch/results/PathMNIST_DistDiff"

cleanup_and_sync() {
    local exit_code=$?
    echo ""
    echo "================================================================================"
    if [ ${exit_code} -ne 0 ]; then
        echo "❌ Job failed with exit code ${exit_code}"
    else
        echo "✅ Job completed successfully"
    fi
    echo "================================================================================"

    # Always sync results back, even on failure
    if [ -n "${RESULTS_DST}" ] && [ -n "${OUT_DIR}" ] && [ -d "${OUT_DIR}" ]; then
        echo ""
        echo "📦 Syncing results/logs back to permanent storage..."
        mkdir -p "${RESULTS_DST}"

        # Check for NPZ files before rsync
        echo ""
        echo "Checking for NPZ files in localscratch before sync..."
        NPZ_COUNT=$(find "${OUT_DIR}/synthetic_data" -name "*.npz" 2>/dev/null | wc -l)
        if [ ${NPZ_COUNT} -gt 0 ]; then
            echo "✓ Found ${NPZ_COUNT} NPZ files to sync:"
            find "${OUT_DIR}/synthetic_data" -name "*.npz" -exec ls -lh {} \; 2>/dev/null
        else
            echo "⚠️  No NPZ files found in ${OUT_DIR}/synthetic_data"
        fi

        # Copy everything: checkpoints, logs, synthetic data (including NPZ files), and error traces
        echo ""
        echo "[sync] Copying from ${OUT_DIR} to ${RESULTS_DST}"
        echo "       This includes NPZ files, checkpoints, and logs"
        rsync -av --progress "${OUT_DIR}/" "${RESULTS_DST}/" 2>&1 || echo "⚠️  rsync failed, but continuing..."

        echo ""
        echo "Synced contents:"
        ls -lah "${RESULTS_DST}" 2>/dev/null || echo "  (could not list)"

        # Verify NPZ files were synced
        echo ""
        echo "Verifying NPZ files in permanent storage..."
        NPZ_SYNCED=$(find "${RESULTS_DST}/synthetic_data" -name "*.npz" 2>/dev/null | wc -l)
        if [ ${NPZ_SYNCED} -gt 0 ]; then
            echo "✓ ${NPZ_SYNCED} NPZ files successfully synced to permanent storage:"
            find "${RESULTS_DST}/synthetic_data" -name "*.npz" -exec ls -lh {} \; 2>/dev/null
        else
            echo "❌ WARNING: No NPZ files found in ${RESULTS_DST}/synthetic_data"
            echo "   Expected ${NPZ_COUNT} NPZ files but found 0 after rsync"
        fi

        if [ ${exit_code} -ne 0 ]; then
            echo ""
            echo "================================================================================"
            echo "🔍 Debug Information"
            echo "================================================================================"
            echo "Check the following for error details:"
            echo "  • SLURM logs: ${SLURM_JOB_NAME}.${SLURM_JOB_ID}.{out,err}"
            echo "  • Results dir: ${RESULTS_DST}"
            echo "  • Generation logs: ${RESULTS_DST}/logs/generation_split_*.log"
            echo "  • Python logs: ${RESULTS_DST}/logs/*.log"
            echo "================================================================================"
        fi
    else
        echo "⚠️  Cannot sync results: RESULTS_DST or OUT_DIR not set or not accessible"
    fi

    # Cleanup localscratch
    echo ""
    echo "🧹 Cleaning up localscratch..."
    if [ -n "${MYLOCALSCRATCH:-}" ] && cd "${LOCALSCRATCH%/}/${USER}" 2>/dev/null; then
        rm -rf --one-file-system "${MYLOCALSCRATCH}" 2>/dev/null || true
        echo "✓ Cleanup complete"
    fi

    exit ${exit_code}
}

# Register cleanup function to run on exit (success or failure)
trap cleanup_and_sync EXIT

echo "================================================================================"
echo "🎯 DistDiff Multi-Stage Pipeline for PathMNIST"
echo "================================================================================"

# ---------- Configuration ----------
CONFIG_FILE="${1:-/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/medsyn/config/distdiff_pathmnist.yaml}"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    echo "Usage: sbatch $0 [config_file.yaml]"
    exit 1
fi

echo "Using config: $CONFIG_FILE"
echo ""

# Parse config file for all parameters (YAML parsing)
echo "Parsing configuration from: $CONFIG_FILE"

# Data paths
NPZ_PATH=$(grep "npz_path:" "$CONFIG_FILE" | head -1 | awk '{print $2}')
OUTPUT_DIR=$(grep "output_dir:" "$CONFIG_FILE" | grep -v "^#" | head -1 | awk '{print $2}')
REPO_SRC=$(grep "repo_src:" "$CONFIG_FILE" | awk '{print $2}')
RESULTS_DST=$(grep "results_dst:" "$CONFIG_FILE" | awk '{print $2}')
CACHE_DIR=$(grep "cache_dir:" "$CONFIG_FILE" | grep -v "^#" | head -1 | awk '{print $2}')

# Model configuration
MODEL_ARCH=$(grep "arch:" "$CONFIG_FILE" | grep -v "^#" | head -1 | awk '{print $2}')
MODEL_PRETRAINED=$(grep "pretrained:" "$CONFIG_FILE" | grep -v "^#" | head -1 | awk '{print $2}')

# Training parameters
SEED=$(grep "seed:" "$CONFIG_FILE" | grep -v "^#" | head -1 | awk '{print $2}')
TRAIN_BATCH_SIZE=$(grep "train_batch_size:" "$CONFIG_FILE" | grep -v "^#" | head -1 | awk '{print $2}')
VAL_BATCH_SIZE=$(grep "val_batch_size:" "$CONFIG_FILE" | grep -v "^#" | head -1 | awk '{print $2}')
LEARNING_RATE=$(grep "^[[:space:]]*lr:" "$CONFIG_FILE" | head -1 | awk '{print $2}')
MOMENTUM=$(grep "momentum:" "$CONFIG_FILE" | grep -v "^#" | head -1 | awk '{print $2}')
WEIGHT_DECAY=$(grep "weight_decay:" "$CONFIG_FILE" | grep -v "^#" | head -1 | awk '{print $2}')
EPOCHS=$(grep "epochs:" "$CONFIG_FILE" | grep -v "^#" | head -1 | awk '{print $2}')
NUM_WORKERS=$(grep "num_workers:" "$CONFIG_FILE" | grep -v "^#" | head -1 | awk '{print $2}')

# Expansion parameters
EXPAND_FACTOR=$(grep "expand_factor:" "$CONFIG_FILE" | grep -v "^#" | head -1 | awk '{print $2}')
NUM_GPUS=$(grep "num_gpus:" "$CONFIG_FILE" | grep -v "^#" | head -1 | awk '{print $2}')
K_PROTOTYPES=$(grep "^[[:space:]]*K:" "$CONFIG_FILE" | head -1 | awk '{print $2}')
PRETRAINED_MODEL=$(grep "pretrained_model:" "$CONFIG_FILE" | grep -v "^#" | head -1 | awk '{print $2}' | tr -d '"')
NUM_IMAGES_PER_PROMPT=$(grep "num_images_per_prompt:" "$CONFIG_FILE" | grep -v "^#" | head -1 | awk '{print $2}')
STRENGTH=$(grep "strength:" "$CONFIG_FILE" | grep -v "^#" | head -1 | awk '{print $2}')
GUIDANCE_SCALE=$(grep "guidance_scale:" "$CONFIG_FILE" | grep -v "^#" | head -1 | awk '{print $2}')
GUIDANCE_TYPE=$(grep "guidance_type:" "$CONFIG_FILE" | grep -v "^#" | head -1 | awk '{print $2}')
OPTIMIZE_TARGETS=$(grep "optimize_targets:" "$CONFIG_FILE" | grep -v "^#" | head -1 | awk '{print $2}')
GUIDANCE_STEP=$(grep "guidance_step:" "$CONFIG_FILE" | grep -v "^#" | head -1 | awk '{print $2}')
GUIDANCE_PERIOD=$(grep "guidance_period:" "$CONFIG_FILE" | grep -v "^#" | head -1 | awk '{print $2}')
CONSTRAINT_VALUE=$(grep "constraint_value:" "$CONFIG_FILE" | grep -v "^#" | head -1 | awk '{print $2}')
RHO=$(grep "rho:" "$CONFIG_FILE" | grep -v "^#" | head -1 | awk '{print $2}')
GRADIENT_CHECKPOINTING=$(grep "gradient_checkpointing:" "$CONFIG_FILE" | grep -v "^#" | head -1 | awk '{print $2}')

# Handle null cache_dir
if [ "$CACHE_DIR" = "null" ] || [ -z "$CACHE_DIR" ]; then
    CACHE_DIR_ARG=""
    LOCAL_FILES_ONLY_FLAG=""
    echo "Cache dir: Using default HuggingFace cache"
    # Don't set offline mode if using default cache (may need internet)
    export HF_HUB_OFFLINE=0
    export TRANSFORMERS_OFFLINE=0
    export HF_DATASETS_OFFLINE=0
else
    CACHE_DIR_ARG="--cache_dir ${CACHE_DIR}"
    # Enable offline mode when using a custom cache directory (HPC without internet)
    LOCAL_FILES_ONLY_FLAG="--local_files_only"
    echo "Cache dir: ${CACHE_DIR}"
    echo "Offline mode: ENABLED (--local_files_only)"
    
    # Set environment variables for offline mode (affects all HuggingFace/Transformers libraries)
    # These ensure no network requests are made even if local_files_only is not set
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_DATASETS_OFFLINE=1
    
    # CRITICAL: Set HF_HOME to point to your cache directory
    # This ensures all HuggingFace libraries look here first
    export HF_HOME="${CACHE_DIR}"
    export TRANSFORMERS_CACHE="${CACHE_DIR}"
    export HF_DATASETS_CACHE="${CACHE_DIR}"
    
    echo "Environment: HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, HF_DATASETS_OFFLINE=1"
    echo "Environment: HF_HOME=${HF_HOME}"
fi

# Handle gradient checkpointing flag
if [ "$GRADIENT_CHECKPOINTING" = "true" ]; then
    GRAD_CKPT_FLAG="--gradient_checkpointing"
else
    GRAD_CKPT_FLAG=""
fi

# Handle pretrained flag for guide model
if [ "$MODEL_PRETRAINED" = "true" ]; then
    PRETRAINED_FLAG="--pretrained"
    echo "Pretrained weights: ENABLED (using LAION2B weights for ViT)"
else
    PRETRAINED_FLAG=""
    echo "Pretrained weights: DISABLED (random initialization)"
fi

# Set default values for any missing parameters
SEED=${SEED:-23102003}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-64}
LEARNING_RATE=${LEARNING_RATE:-0.1}
EPOCHS=${EPOCHS:-100}
NUM_GPUS=${NUM_GPUS:-4}
MODEL_ARCH=${MODEL_ARCH:-open_clip_vit_b32}
K_PROTOTYPES=${K_PROTOTYPES:-5}
NUM_IMAGES_PER_PROMPT=${NUM_IMAGES_PER_PROMPT:-5}
STRENGTH=${STRENGTH:-0.2}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-7.5}
GUIDANCE_TYPE=${GUIDANCE_TYPE:-transform_guidance}
GUIDANCE_STEP=${GUIDANCE_STEP:-10}
GUIDANCE_PERIOD=${GUIDANCE_PERIOD:-2}
CONSTRAINT_VALUE=${CONSTRAINT_VALUE:-0.2}
RHO=${RHO:-10.0}
PRETRAINED_MODEL=${PRETRAINED_MODEL:-CompVis/stable-diffusion-v1-4}

echo ""
echo "Configuration Summary:"
echo "  NPZ Dataset: $NPZ_PATH"
echo "  Output Directory: $OUTPUT_DIR"
echo "  Model Architecture: $MODEL_ARCH"
echo "  Seed: $SEED"
echo "  Training: ${EPOCHS} epochs, batch=${TRAIN_BATCH_SIZE}, lr=${LEARNING_RATE}"
echo "  Expansion: ${EXPAND_FACTOR}x, ${NUM_GPUS} GPUs, K=${K_PROTOTYPES}"
echo "  Guidance: type=${GUIDANCE_TYPE}, step=${GUIDANCE_STEP}, period=${GUIDANCE_PERIOD}"
echo "  Optimization: constraint=${CONSTRAINT_VALUE}, rho=${RHO}, strength=${STRENGTH}"
echo "  Diffusion Model: $PRETRAINED_MODEL"
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

echo "[repo] Copying from ${REPO_SRC} ..."
mkdir -p "${REPO_DIR}"
rsync -a "${REPO_SRC}/" "${REPO_DIR}/"
echo ""

# ---------- 2) Dataset to localscratch ----------
echo "================================================================================"
echo "📦 Staging Dataset to LocalScratch"
echo "================================================================================"

DATA_DST="${DATA_DIR}/pathmnist.npz"
echo "[data] Copying ${NPZ_PATH} -> ${DATA_DST}"
rsync -a "${NPZ_PATH}" "${DATA_DST}"
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
# NOTE: Do NOT export CUDA_VISIBLE_DEVICES globally - each parallel job will set it
# export CUDA_VISIBLE_DEVICES=0,1,2,3  # DISABLED - causes all processes to use GPU 0
export CUDA_DEVICE_ORDER=PCI_BUS_ID  # Ensure consistent GPU ordering
export NCCL_DEBUG=INFO
export IS_SUPERCOMPUTER=1

# IMPORTANT: If using cache_dir, verify it exists and contains required models
if [ "$CACHE_DIR" != "null" ] && [ -n "$CACHE_DIR" ]; then
    if [ ! -d "${CACHE_DIR}" ]; then
        echo "⚠️  WARNING: CACHE_DIR does not exist: ${CACHE_DIR}"
        echo "   OpenCLIP and Stable Diffusion will fail to load!"
        echo "   Please ensure pretrained models are downloaded to this path."
        exit 1
    fi
    
    echo "✓ Cache directory verified: ${CACHE_DIR}"
    echo "  Contents:"
    ls -la "${CACHE_DIR}" 2>/dev/null | head -10
    echo ""
fi

# ========================================================================
# STAGE 1: TRAIN GUIDE MODEL (or skip if already trained)
# ========================================================================
echo ""
echo "================================================================================"
echo "🏋️  STAGE 1: Training Guide Model on Original Data"
echo "================================================================================"

CHECKPOINT_DIR="${OUT_DIR}/checkpoints/guide_model"
mkdir -p "${CHECKPOINT_DIR}"

# Check if a pre-trained guide model exists in permanent storage
EXISTING_GUIDE_MODEL="${RESULTS_DST}/checkpoints/guide_model/model_best.pth.tar"
EXISTING_RESULTS_YAML="${RESULTS_DST}/checkpoints/guide_model/results.yaml"

SKIP_STAGE1=false
if [ -f "${EXISTING_GUIDE_MODEL}" ] && [ -f "${EXISTING_RESULTS_YAML}" ]; then
    echo ""
    echo "================================================================================"
    echo "📦 Found existing guide model in permanent storage!"
    echo "================================================================================"
    echo "  • Model: ${EXISTING_GUIDE_MODEL}"
    echo "  • Results: ${EXISTING_RESULTS_YAML}"
    echo ""
    echo "Copying pre-trained guide model to localscratch..."

    # Copy the entire guide_model directory to preserve all files
    rsync -av "${RESULTS_DST}/checkpoints/guide_model/" "${CHECKPOINT_DIR}/"

    if [ $? -eq 0 ] && [ -f "${CHECKPOINT_DIR}/model_best.pth.tar" ]; then
        echo ""
        echo "✅ Successfully copied pre-trained guide model"
        echo "⏭️  SKIPPING Stage 1 training (using existing model)"
        echo ""

        # Display previous training results
        echo "Previous training results:"
        cat "${CHECKPOINT_DIR}/results.yaml" 2>/dev/null || echo "  (could not read results.yaml)"
        echo ""

        SKIP_STAGE1=true
    else
        echo "⚠️  Failed to copy guide model, will train from scratch"
        SKIP_STAGE1=false
    fi
fi

if [ "${SKIP_STAGE1}" = false ]; then
    echo "[stage1] Training guide model (${MODEL_ARCH}) on NPZ dataset..."
    echo "[stage1] Dataset: ${DATA_DST}"
    echo "[stage1] Checkpoint: ${CHECKPOINT_DIR}"
    echo ""

    # Use only first GPU for guide model training
    # All parameters read from config file
    CUDA_VISIBLE_DEVICES=0 python medsyn/models/distdiff/train.py \
        -a "${MODEL_ARCH}" \
        -d pathmnist_npz \
        --data_dir "${DATA_DST}" \
        --checkpoint "${CHECKPOINT_DIR}" \
        --manualSeed "${SEED}" \
        --train-batch-size "${TRAIN_BATCH_SIZE}" \
        --val-batch-size "${VAL_BATCH_SIZE}" \
        --lr "${LEARNING_RATE}" \
        --epochs "${EPOCHS}" \
        ${CACHE_DIR_ARG} \
        ${PRETRAINED_FLAG}

    STAGE1_EXIT_CODE=$?

    if [ ${STAGE1_EXIT_CODE} -ne 0 ]; then
        echo "❌ Stage 1 failed with exit code ${STAGE1_EXIT_CODE}"
        exit ${STAGE1_EXIT_CODE}
    fi

    echo "✅ Stage 1 completed successfully"
    echo ""
fi

# Verify guide model checkpoint exists and is valid
GUIDE_MODEL_PATH="${CHECKPOINT_DIR}/model_best.pth.tar"
if [ ! -f "${GUIDE_MODEL_PATH}" ]; then
    echo "❌ Error: Guide model checkpoint not found at ${GUIDE_MODEL_PATH}"
    exit 1
fi

# Verify checkpoint is not empty/corrupted (should be at least 1MB for a valid model)
CKPT_SIZE=$(stat -c%s "${GUIDE_MODEL_PATH}" 2>/dev/null || echo "0")
if [ "${CKPT_SIZE}" -lt 1000000 ]; then
    echo "❌ Error: Guide model checkpoint appears corrupted (size: ${CKPT_SIZE} bytes)"
    echo "   Expected at least 1MB for a valid model checkpoint"
    exit 1
fi
CKPT_SIZE_MB=$(echo "scale=2; ${CKPT_SIZE} / 1048576" | bc)
echo "✓ Guide model checkpoint verified: ${GUIDE_MODEL_PATH} (${CKPT_SIZE_MB} MB)"
echo ""

# ========================================================================
# STAGE 2: GENERATE SYNTHETIC DATA (PARALLEL)
# ========================================================================
echo ""
echo "================================================================================"
echo "🎨 STAGE 2: Generating Synthetic Data (Parallel across ${NUM_GPUS} GPUs)"
echo "================================================================================"

SYNTH_DATA_DIR="${OUT_DIR}/synthetic_data"
mkdir -p "${SYNTH_DATA_DIR}"
mkdir -p "${OUT_DIR}/logs"

echo "[stage2] Generating ${EXPAND_FACTOR}x expanded dataset using Stable Diffusion..."
echo "[stage2] Guide model: ${GUIDE_MODEL_PATH}"
echo "[stage2] Output: ${SYNTH_DATA_DIR}"
echo "[stage2] Running ${NUM_GPUS} parallel generation jobs (1 per GPU)..."
echo ""

# Launch parallel generation jobs (one per GPU)
# Each job processes 1/NUM_GPUS of the dataset
# All parameters read from config file
# IMPORTANT: Using subshells to ensure proper GPU isolation
for split in $(seq 0 $((NUM_GPUS - 1))); do
    echo "  → Launching generation job ${split}/${NUM_GPUS} on GPU ${split}..."

    # Use subshell to isolate CUDA_VISIBLE_DEVICES per process
    (
        export CUDA_VISIBLE_DEVICES=${split}

        python medsyn/models/distdiff/generate_data.py \
            --guidance_type="${GUIDANCE_TYPE}" \
            -a "${MODEL_ARCH}" \
            -d pathmnist_npz \
            --data_dir "${DATA_DST}" \
            --output_dir "${SYNTH_DATA_DIR}/split_${split}" \
            --pretrained_model_name_or_path "${PRETRAINED_MODEL}" \
            ${CACHE_DIR_ARG} \
            ${LOCAL_FILES_ONLY_FLAG} \
            ${GRAD_CKPT_FLAG} \
            --K "${K_PROTOTYPES}" \
            --train_batch_size 1 \
            --optimize_targets "${OPTIMIZE_TARGETS}" \
            --strength "${STRENGTH}" \
            --num_images_per_prompt "${NUM_IMAGES_PER_PROMPT}" \
            --guidance_step "${GUIDANCE_STEP}" \
            --guidance_period "${GUIDANCE_PERIOD}" \
            --encoder_weight_path "${GUIDE_MODEL_PATH}" \
            --guidance_scale "${GUIDANCE_SCALE}" \
            --constraint_value "${CONSTRAINT_VALUE}" \
            --rho "${RHO}" \
            --total_split "${NUM_GPUS}" \
            --split ${split} \
            > "${OUT_DIR}/logs/generation_split_${split}.log" 2>&1
    ) &
done

# Wait for all generation jobs to complete
echo ""
echo "  Waiting for all ${NUM_GPUS} generation jobs to complete..."
wait

STAGE2_EXIT_CODE=$?

if [ ${STAGE2_EXIT_CODE} -ne 0 ]; then
    echo "❌ Stage 2 failed with exit code ${STAGE2_EXIT_CODE}"
    exit ${STAGE2_EXIT_CODE}
fi

echo "✅ Stage 2 completed successfully"
echo ""

# ========================================================================
# STAGE 2.5: COMPRESS PNG OUTPUT TO NPZ FORMAT
# ========================================================================
echo ""
echo "================================================================================"
echo "📦 STAGE 2.5: Compressing PNG Output to NPZ Format"
echo "================================================================================"
echo "[compress] Compressing generated images to NPZ for efficient transfer..."
echo ""

COMPRESS_SUCCESS=true
for split in $(seq 0 $((NUM_GPUS - 1))); do
    SPLIT_DIR="${SYNTH_DATA_DIR}/split_${split}"
    NPZ_OUTPUT="${SYNTH_DATA_DIR}/split_${split}.npz"

    if [ -d "${SPLIT_DIR}" ]; then
        echo "  → Compressing split ${split}..."

        python medsyn/models/distdiff/compress_split.py \
            --split_dir "${SPLIT_DIR}" \
            --output "${NPZ_OUTPUT}"
            # Note: Not deleting PNGs automatically to preserve data for Stage 3
            # You can manually delete PNG directories after confirming NPZ files are valid

        COMPRESS_EXIT=$?
        if [ ${COMPRESS_EXIT} -ne 0 ]; then
            echo "⚠️  Warning: Compression failed for split ${split}"
            COMPRESS_SUCCESS=false
        else
            echo "✓ Split ${split} compressed to NPZ ($(du -h "${NPZ_OUTPUT}" | cut -f1))"
        fi
    else
        echo "⚠️  Warning: Split directory not found: ${SPLIT_DIR}"
        COMPRESS_SUCCESS=false
    fi
done

if [ "${COMPRESS_SUCCESS}" = true ]; then
    echo ""
    echo "✅ All splits compressed successfully"
    echo ""
    echo "NPZ files created:"
    ls -lh "${SYNTH_DATA_DIR}"/*.npz 2>/dev/null || echo "  (no NPZ files found)"
else
    echo ""
    echo "⚠️  Some splits failed to compress, but continuing..."
fi
echo ""

# Verify synthetic data was generated (check for NPZ files or PNG directories)
SYNTH_COUNT=0
for split in $(seq 0 $((NUM_GPUS - 1))); do
    NPZ_FILE="${SYNTH_DATA_DIR}/split_${split}.npz"
    SPLIT_DIR="${SYNTH_DATA_DIR}/split_${split}"

    # Check if NPZ file exists (preferred)
    if [ -f "${NPZ_FILE}" ]; then
        NPZ_SIZE=$(stat -c%s "${NPZ_FILE}" 2>/dev/null || echo "0")
        if [ "${NPZ_SIZE}" -gt 1000000 ]; then  # At least 1MB
            SYNTH_COUNT=$((SYNTH_COUNT + 1))
        else
            echo "⚠️  Warning: split_${split}.npz exists but appears corrupted (size: ${NPZ_SIZE} bytes)"
        fi
    # Fallback: check if directory exists with PNG files
    elif [ -d "${SPLIT_DIR}" ]; then
        CLASS_COUNT=$(find "${SPLIT_DIR}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
        if [ ${CLASS_COUNT} -gt 0 ]; then
            SYNTH_COUNT=$((SYNTH_COUNT + 1))
        else
            echo "⚠️  Warning: split_${split} directory exists but is empty (no class directories)"
        fi
    fi
done

if [ ${SYNTH_COUNT} -ne ${NUM_GPUS} ]; then
    echo ""
    echo "================================================================================"
    echo "❌ ERROR: Synthetic data generation failed"
    echo "================================================================================"
    echo "Expected: ${NUM_GPUS} synthetic data splits"
    echo "Found:    ${SYNTH_COUNT} splits"
    echo ""
    echo "Generation output directories:"
    ls -la "${SYNTH_DATA_DIR}" 2>/dev/null || echo "  (directory not accessible)"
    echo ""
    echo "Check generation logs for errors:"
    for split in $(seq 0 $((NUM_GPUS - 1))); do
        log_file="${OUT_DIR}/logs/generation_split_${split}.log"
        if [ -f "${log_file}" ]; then
            echo ""
            echo "=== Last 30 lines of generation_split_${split}.log ==="
            tail -n 30 "${log_file}"
        fi
    done
    echo "================================================================================"
    exit 1
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

EXPANDED_CHECKPOINT_DIR="${OUT_DIR}/checkpoints/classifier_on_expanded"
mkdir -p "${EXPANDED_CHECKPOINT_DIR}"

echo "[stage3] Training classifier on concatenated dataset..."
echo "[stage3] Original data: ${DATA_DST}"

# Dynamically build list of synthetic data directories
SYNTH_DIRS=""
for split in $(seq 0 $((NUM_GPUS - 1))); do
    SYNTH_DIRS="${SYNTH_DIRS} ${SYNTH_DATA_DIR}/split_${split}"
done
echo "[stage3] Synthetic data:${SYNTH_DIRS}"
echo "[stage3] Checkpoint: ${EXPANDED_CHECKPOINT_DIR}"
echo ""

# Use single GPU for final training
# All parameters read from config file
CUDA_VISIBLE_DEVICES=0 python medsyn/models/distdiff/train_expanded_data_concat_original.py \
    -d pathmnist_npz \
    --data_dir "${DATA_DST}" \
    --data_expanded_dir ${SYNTH_DIRS} \
    --checkpoint "${EXPANDED_CHECKPOINT_DIR}" \
    -a "${MODEL_ARCH}" \
    --manualSeed "${SEED}" \
    --train-batch-size "${TRAIN_BATCH_SIZE}" \
    --val-batch-size "${VAL_BATCH_SIZE}" \
    --lr "${LEARNING_RATE}" \
    --epochs "${EPOCHS}"

STAGE3_EXIT_CODE=$?

if [ ${STAGE3_EXIT_CODE} -ne 0 ]; then
    echo "❌ Stage 3 failed with exit code ${STAGE3_EXIT_CODE}"
    exit ${STAGE3_EXIT_CODE}
fi

echo "✅ Stage 3 completed successfully"
echo ""

# ========================================================================
# PIPELINE COMPLETE
# ========================================================================
# Note: Results sync and cleanup handled by trap on EXIT
echo ""
echo "================================================================================"
echo "🎉 DistDiff Pipeline Complete!"
echo "================================================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Dataset: PathMNIST (NPZ format)"
echo ""
echo "Configuration Used:"
echo "  • Model: ${MODEL_ARCH}"
echo "  • Training: ${EPOCHS} epochs, lr=${LEARNING_RATE}, batch=${TRAIN_BATCH_SIZE}"
echo "  • Expansion: ${EXPAND_FACTOR}x using ${NUM_GPUS} GPUs"
echo "  • Guidance: ${GUIDANCE_TYPE}, step=${GUIDANCE_STEP}, period=${GUIDANCE_PERIOD}"
echo "  • Optimization: constraint=${CONSTRAINT_VALUE}, rho=${RHO}"
echo ""
echo "Pipeline Summary:"
echo "  ✅ Stage 1: Guide model trained (${MODEL_ARCH})"
echo "  ✅ Stage 2: Synthetic data generated (${EXPAND_FACTOR}x expansion, ${NUM_GPUS} parallel jobs)"
echo "  ✅ Stage 2.5: Synthetic data compressed to NPZ format (${NUM_GPUS} files)"
echo "  ✅ Stage 3: Classifier trained on expanded data"
echo ""
echo "Results will be synced to: ${RESULTS_DST}"
echo "  • Guide model: checkpoints/guide_model/"
echo "  • Expanded classifier: checkpoints/classifier_on_expanded/"
echo "  • Synthetic data PNG: synthetic_data/split_*/"
echo "  • Synthetic data NPZ: synthetic_data/split_*.npz (compressed format)"
echo "  • Logs: logs/"
echo "================================================================================"
echo ""
echo "IMPORTANT: Retrieving NPZ Files"
echo "================================================================================"
echo "The NPZ files will be automatically synced on job exit to:"
echo "  ${RESULTS_DST}/synthetic_data/"
echo ""
echo "To copy NPZ files to your local machine:"
echo "  scp user@picasso:${RESULTS_DST}/synthetic_data/split_*.npz /local/path/"
echo ""
echo "Or use rsync for better transfer:"
echo "  rsync -avz --progress user@picasso:${RESULTS_DST}/synthetic_data/*.npz /local/path/"
echo "================================================================================"
echo ""
echo "Note: Result synchronization and cleanup will run automatically on exit."
echo "Check the sync output above to verify NPZ files were transferred successfully."
echo "================================================================================"

# Exit with success - trap will handle sync and cleanup
exit 0
