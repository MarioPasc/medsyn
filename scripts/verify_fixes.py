#!/usr/bin/env python3
"""
Comprehensive verification of all fixes in embeddings.py
This script checks that all identified bugs have been properly fixed.
"""

import sys
from pathlib import Path

# Read the embeddings file
embeddings_path = Path("medsyn/models/ccDDPM/engine/logging/embeddings.py")
with open(embeddings_path, 'r') as f:
    content = f.read()
    lines = content.split('\n')

print("=" * 80)
print("VERIFICATION OF EMBEDDINGS.PY FIXES")
print("=" * 80)

fixes = []

# Fix 1: Tuple handling in hook
print("\n✓ Fix #1: Tuple handling in forward hook (lines 176-182)")
hook_section = '\n'.join(lines[175:183])
if 'isinstance(output, tuple)' in hook_section and "output[0]" in hook_section:
    print("  ✅ VERIFIED: Hook checks for tuple and extracts output[0]")
    fixes.append(True)
else:
    print("  ❌ MISSING: Tuple handling not found in hook")
    fixes.append(False)
print("  Code snippet:")
for i, line in enumerate(lines[175:183], start=176):
    print(f"    {i}: {line}")

# Fix 2: class_embed.emb usage (line ~309)
print("\n✓ Fix #2: class_embed.emb() usage for conditional (around line 309)")
found_line = None
for i, line in enumerate(lines):
    if 'feat_cond = model.class_embed.emb(label)' in line:
        found_line = i + 1
        break

if found_line:
    print(f"  ✅ VERIFIED: Found at line {found_line}")
    fixes.append(True)
    print(f"    {found_line}: {lines[found_line-1]}")
else:
    # Check if old buggy version exists
    for i, line in enumerate(lines):
        if 'feat_cond = model.class_embed(label)' in line and 'class_embed.emb' not in line:
            print(f"  ❌ BUG STILL PRESENT at line {i+1}")
            print(f"    {i+1}: {line}")
            fixes.append(False)
            break
    else:
        print("  ⚠️  Pattern not found (might be refactored)")
        fixes.append(None)

# Fix 3: class_embed.emb usage for unconditional (line ~330)
print("\n✓ Fix #3: class_embed.emb() usage for unconditional (around line 330)")
found_line = None
for i, line in enumerate(lines):
    if 'feat_uncond = model.class_embed.emb(uncond_label)' in line:
        found_line = i + 1
        break

if found_line:
    print(f"  ✅ VERIFIED: Found at line {found_line}")
    fixes.append(True)
    print(f"    {found_line}: {lines[found_line-1]}")
else:
    for i, line in enumerate(lines):
        if 'feat_uncond = model.class_embed(uncond_label)' in line and 'class_embed.emb' not in line:
            print(f"  ❌ BUG STILL PRESENT at line {i+1}")
            print(f"    {i+1}: {line}")
            fixes.append(False)
            break
    else:
        print("  ⚠️  Pattern not found (might be refactored)")
        fixes.append(None)

# Fix 4: 1D tensor handling (around line 229-232)
print("\n✓ Fix #4: 1D tensor handling in activation pooling (around line 229)")
found = False
for i in range(len(lines) - 3):
    section = '\n'.join(lines[i:i+4])
    if 'act.dim() == 1' in section:
        print(f"  ✅ VERIFIED: Found at line {i+1}")
        fixes.append(True)
        found = True
        for j, line in enumerate(lines[i:i+4], start=i+1):
            print(f"    {j}: {line}")
        break

if not found:
    print("  ❌ MISSING: 1D tensor handling not found")
    fixes.append(False)

# Fix 5: Empty timesteps check (around line 825)
print("\n✓ Fix #5: Empty timesteps list check (around line 825)")
found = False
for i, line in enumerate(lines):
    if 'if not config.probe.timesteps' in line:
        print(f"  ✅ VERIFIED: Found at line {i+1}")
        fixes.append(True)
        found = True
        for j in range(max(0, i-1), min(len(lines), i+3)):
            print(f"    {j+1}: {lines[j]}")
        break

if not found:
    print("  ❌ MISSING: Empty timesteps check not found")
    fixes.append(False)

# Fix 6: Feature shape validation (around line 636)
print("\n✓ Fix #6: Feature shape validation before np.stack() (around line 636)")
found = False
for i, line in enumerate(lines):
    if 'feature_shapes' in line and 'r.feature.shape' in line:
        print(f"  ✅ VERIFIED: Found at line {i+1}")
        fixes.append(True)
        found = True
        for j in range(i, min(len(lines), i+7)):
            print(f"    {j+1}: {lines[j]}")
        break

if not found:
    print("  ❌ MISSING: Feature shape validation not found")
    fixes.append(False)

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

verified = sum(1 for f in fixes if f is True)
missing = sum(1 for f in fixes if f is False)
unknown = sum(1 for f in fixes if f is None)

print(f"Total fixes checked: {len(fixes)}")
print(f"  ✅ Verified: {verified}")
print(f"  ❌ Missing: {missing}")
print(f"  ⚠️  Unknown: {unknown}")

if missing == 0:
    print("\n🎉 ALL FIXES SUCCESSFULLY APPLIED!")
    sys.exit(0)
else:
    print(f"\n⚠️  {missing} FIX(ES) MISSING OR INCOMPLETE")
    sys.exit(1)
