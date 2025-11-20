#!/usr/bin/env bash
# Verification script for DistDiff PathMNIST paper alignment
# Run this to ensure all settings match the paper specifications

set -euo pipefail

echo "================================================================================"
echo "🔍 DistDiff PathMNIST Paper Alignment Verification"
echo "================================================================================"
echo ""

ERRORS=0
WARNINGS=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    ((ERRORS++))
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNINGS++))
}

# Check 1: Prompt template case
echo "1. Checking prompt template..."
TEMPLATE=$(python -c "from medsyn.models.distdiff.dataloader import CUSTOM_TEMPLATES; print(CUSTOM_TEMPLATES['pathmnist_npz'])" 2>/dev/null || echo "ERROR")

if [ "$TEMPLATE" = "ERROR" ]; then
    check_fail "Could not import dataloader (is environment activated?)"
elif [ "$TEMPLATE" = "A COLON PATHOLOGICAL IMAGE OF {}." ]; then
    check_pass "Prompt template is UPPERCASE (paper specification)"
elif [ "$TEMPLATE" = "a colon pathological image of {}." ]; then
    check_fail "Prompt template is lowercase (should be UPPERCASE per paper)"
else
    check_warn "Unexpected prompt template: $TEMPLATE"
fi
echo ""

# Check 2: Config file - strength parameter
echo "2. Checking noise strength parameter..."
if [ -f "config/distdiff_pathmnist.yaml" ]; then
    STRENGTH=$(grep -E "^\s*strength:" config/distdiff_pathmnist.yaml | awk '{print $2}' | head -1)
    if [ "$STRENGTH" = "0.2" ]; then
        check_pass "Noise strength is 0.2 (paper specification)"
    elif [ "$STRENGTH" = "0.5" ]; then
        check_fail "Noise strength is 0.5 (should be 0.2 per paper)"
    else
        check_warn "Unexpected noise strength: $STRENGTH"
    fi
else
    check_warn "Config file not found: config/distdiff_pathmnist.yaml"
fi
echo ""

# Check 3: Config file - guidance_step parameter
echo "3. Checking guidance step (M) parameter..."
if [ -f "config/distdiff_pathmnist.yaml" ]; then
    GUIDANCE_STEP=$(grep -E "^\s*guidance_step:" config/distdiff_pathmnist.yaml | awk '{print $2}' | head -1)
    if [ "$GUIDANCE_STEP" = "10" ]; then
        check_pass "Guidance step is 10 (paper specification: M=10)"
    elif [ "$GUIDANCE_STEP" = "20" ]; then
        check_fail "Guidance step is 20 (should be 10 per paper)"
    else
        check_warn "Unexpected guidance step: $GUIDANCE_STEP"
    fi
else
    check_warn "Config file not found: config/distdiff_pathmnist.yaml"
fi
echo ""

# Check 4: Config file - model architecture
echo "4. Checking baseline model architecture..."
if [ -f "config/distdiff_pathmnist.yaml" ]; then
    ARCH=$(grep -E "^\s*arch:" config/distdiff_pathmnist.yaml | awk '{print $2}' | head -1)
    if [ "$ARCH" = "open_clip_vit_b32" ]; then
        check_pass "Model architecture is open_clip_vit_b32 (paper: CLIP-ViT-B/32)"
    elif [ "$ARCH" = "resnet50" ]; then
        check_fail "Model architecture is resnet50 (should be open_clip_vit_b32 per paper)"
    else
        check_warn "Unexpected architecture: $ARCH"
    fi
else
    check_warn "Config file not found: config/distdiff_pathmnist.yaml"
fi
echo ""

# Check 5: Config file - Stable Diffusion model
echo "5. Checking Stable Diffusion model..."
if [ -f "config/distdiff_pathmnist.yaml" ]; then
    SD_MODEL=$(grep -E "^\s*pretrained_model:" config/distdiff_pathmnist.yaml | awk '{print $2}' | tr -d '"')
    if [[ "$SD_MODEL" == *"stable-diffusion-v1-4"* ]]; then
        check_pass "Using Stable Diffusion v1-4 (paper specification)"
    else
        check_warn "Using non-standard SD model: $SD_MODEL"
    fi
else
    check_warn "Config file not found: config/distdiff_pathmnist.yaml"
fi
echo ""

# Check 6: SLURM script - strength in generation
echo "6. Checking SLURM script generation parameters..."
if [ -f "scripts/distdiff_slurm_job.sh" ]; then
    if grep -q "strength 0.2" scripts/distdiff_slurm_job.sh; then
        check_pass "SLURM script uses strength 0.2"
    elif grep -q "strength 0.5" scripts/distdiff_slurm_job.sh; then
        check_fail "SLURM script uses strength 0.5 (should be 0.2)"
    else
        check_warn "Could not verify strength parameter in SLURM script"
    fi

    if grep -q "guidance_step 10" scripts/distdiff_slurm_job.sh; then
        check_pass "SLURM script uses guidance_step 10"
    elif grep -q "guidance_step 20" scripts/distdiff_slurm_job.sh; then
        check_fail "SLURM script uses guidance_step 20 (should be 10)"
    else
        check_warn "Could not verify guidance_step in SLURM script"
    fi

    if grep -q "open_clip_vit_b32" scripts/distdiff_slurm_job.sh; then
        check_pass "SLURM script uses open_clip_vit_b32"
    elif grep -q "\-a resnet50" scripts/distdiff_slurm_job.sh; then
        check_fail "SLURM script uses resnet50 (should be open_clip_vit_b32)"
    else
        check_warn "Could not verify architecture in SLURM script"
    fi
else
    check_warn "SLURM script not found: scripts/distdiff_slurm_job.sh"
fi
echo ""

# Check 7: Local script - parameters
echo "7. Checking local script generation parameters..."
if [ -f "scripts/distdiff_local_job.sh" ]; then
    if grep -q "strength 0.2" scripts/distdiff_local_job.sh; then
        check_pass "Local script uses strength 0.2"
    elif grep -q "strength 0.5" scripts/distdiff_local_job.sh; then
        check_fail "Local script uses strength 0.5 (should be 0.2)"
    else
        check_warn "Could not verify strength parameter in local script"
    fi

    if grep -q "guidance_step 10" scripts/distdiff_local_job.sh; then
        check_pass "Local script uses guidance_step 10"
    elif grep -q "guidance_step 20" scripts/distdiff_local_job.sh; then
        check_fail "Local script uses guidance_step 20 (should be 10)"
    else
        check_warn "Could not verify guidance_step in local script"
    fi

    if grep -q "open_clip_vit_b32" scripts/distdiff_local_job.sh; then
        check_pass "Local script uses open_clip_vit_b32"
    elif grep -q "\-a resnet50" scripts/distdiff_local_job.sh; then
        check_fail "Local script uses resnet50 (should be open_clip_vit_b32)"
    else
        check_warn "Could not verify architecture in local script"
    fi
else
    check_warn "Local script not found: scripts/distdiff_local_job.sh"
fi
echo ""

# Check 8: Environment dependencies
echo "8. Checking environment dependencies..."
if python -c "import open_clip" 2>/dev/null; then
    check_pass "open_clip installed (required for CLIP-ViT-B/32)"
else
    check_fail "open_clip not installed (required for CLIP-ViT-B/32)"
    echo "   Install with: pip install -e \".[distdiff]\""
fi

if python -c "import timm" 2>/dev/null; then
    check_pass "timm installed (required for DistDiff)"
else
    check_fail "timm not installed (required for DistDiff)"
    echo "   Install with: pip install -e \".[distdiff]\""
fi
echo ""

# Summary
echo "================================================================================"
echo "📊 Verification Summary"
echo "================================================================================"

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed!${NC}"
    echo "Your DistDiff configuration matches the PathMNIST paper specifications."
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠️  $WARNINGS warning(s) found${NC}"
    echo "Configuration mostly correct but some non-critical issues detected."
    exit 0
else
    echo -e "${RED}❌ $ERRORS error(s) and $WARNINGS warning(s) found${NC}"
    echo ""
    echo "Please review the errors above and:"
    echo "  1. Check DISTDIFF_PATHMNIST_PAPER_ALIGNMENT.md for details"
    echo "  2. Verify all changes have been applied"
    echo "  3. Rerun this verification script"
    exit 1
fi
