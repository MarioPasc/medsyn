#!/bin/bash
# Quick test runner for all embeddings tests
# Run this to verify all fixes are working

echo "================================================================================"
echo "Running All Embeddings Tests"
echo "================================================================================"
echo ""

PYTHON=~/.conda/envs/medsyn/bin/python
FAILED=0

# Test 1
echo "[1/5] Running test_class_embed.py..."
if $PYTHON test_class_embed.py > /dev/null 2>&1; then
    echo "  ✅ PASSED"
else
    echo "  ❌ FAILED"
    FAILED=$((FAILED+1))
fi
echo ""

# Test 2
echo "[2/5] Running test_embeddings_edge_cases.py..."
if $PYTHON test_embeddings_edge_cases.py > /dev/null 2>&1; then
    echo "  ✅ PASSED"
else
    echo "  ❌ FAILED"
    FAILED=$((FAILED+1))
fi
echo ""

# Test 3
echo "[3/5] Running diagnose_layer_outputs.py..."
if $PYTHON diagnose_layer_outputs.py > /dev/null 2>&1; then
    echo "  ✅ PASSED"
else
    echo "  ❌ FAILED"
    FAILED=$((FAILED+1))
fi
echo ""

# Test 4
echo "[4/5] Running verify_fixes.py..."
if $PYTHON verify_fixes.py > /dev/null 2>&1; then
    echo "  ✅ PASSED"
else
    echo "  ❌ FAILED"
    FAILED=$((FAILED+1))
fi
echo ""

# Test 5
echo "[5/5] Running test_embeddings_integration.py..."
if $PYTHON test_embeddings_integration.py > /dev/null 2>&1; then
    echo "  ✅ PASSED"
else
    echo "  ❌ FAILED"
    FAILED=$((FAILED+1))
fi
echo ""

echo "================================================================================"
if [ $FAILED -eq 0 ]; then
    echo "✅ ALL TESTS PASSED (5/5)"
    echo "The embeddings code is ready for use!"
else
    echo "❌ $FAILED TEST(S) FAILED"
    echo "Please review the output above."
    exit 1
fi
echo "================================================================================"
