# Embeddings Code Debugging Checklist

## 🔍 Critical Issues to Check

### 1. Tuple/List Return Values (HIGH PRIORITY - ALREADY HIT)
- [x] **Line 176-182**: Hook function handling tuple outputs ✓ FIXED
- [ ] **Line 309**: `model.class_embed(label)` - could return tuple?
- [ ] **Line 330**: `model.class_embed(uncond_label)` - could return tuple?
- [ ] **Line 558**: `model.class_embed.emb.weight` - attribute chain safe?

### 2. Tensor Shape Assumptions (HIGH PRIORITY)
- [ ] **Line 214-231**: Shape handling assumes act.dim() returns 2, 3, or 4 - what about 1D or 5D+?
- [ ] **Line 287**: `x0 = sample["pixel_values"].unsqueeze(0)` - what if already has batch dim?
- [ ] **Line 629**: `features = np.stack([r.feature for r in records])` - assumes all same shape
- [ ] **Line 568-572**: Shape consistency check for embeddings trajectory

### 3. Device Consistency (MEDIUM PRIORITY)
- [ ] **Line 288**: `label = torch.tensor([class_id], device=device)` ✓ correct
- [ ] **Line 292**: `t = torch.tensor([t_val], device=device, dtype=torch.long)` ✓ correct
- [ ] **Line 295**: `noise = torch.randn_like(x0)` ✓ inherits device
- [ ] **Line 199**: Fallback tensor creation uses correct device ✓
- [ ] **Line 209**: Fallback tensor creation uses correct device ✓

### 4. None/Empty Value Handling (HIGH PRIORITY)
- [ ] **Line 207-209**: Empty features dict - returns zeros ✓
- [ ] **Line 299**: Empty layer_names list - handled with [""] ✓
- [ ] **Line 618-620**: Empty records list - logged but could break np.stack
- [ ] **Line 828**: Filtered list < 10 samples - no clustering (good)
- [ ] **Line 564**: `prev_emb` could be None - handled ✓

### 5. Attribute Access Safety (MEDIUM PRIORITY)
- [ ] **Line 493-524**: Multiple `getattr()` calls could return None
- [ ] **Line 558**: `model.class_embed.emb.weight` - assumes structure exists
- [ ] **Line 309, 330**: Direct access to `model.class_embed` - exists?

### 6. Type Conversions (LOW PRIORITY)
- [ ] **Line 318**: `.cpu().numpy().flatten()` - safe
- [ ] **Line 339**: `.cpu().numpy().flatten()` - safe
- [ ] **Line 72**: `.tolist()` for JSON - safe
- [ ] **Line 117**: `sample["labels"].item()` - assumes 0-d or 1-d tensor

### 7. Index Out of Bounds (MEDIUM PRIORITY)
- [ ] **Line 186-193**: Layer navigation with splits - could fail
- [ ] **Line 820**: `config.probe.timesteps[len(config.probe.timesteps) // 2]` - assumes non-empty
- [ ] **Line 821**: `config.probe.layer_names[0]` - assumes non-empty

### 8. Division by Zero / Math Edge Cases (LOW PRIORITY)
- [ ] **Line 216**: `.mean(dim=[2, 3])` - safe with 4D
- [ ] **Line 225**: `.mean(dim=1)` - safe with 3D
- [ ] **Line 433-438**: Clustering scores - sklearn handles edge cases

### 9. Data Type Mismatches (LOW PRIORITY)
- [ ] **Line 292**: `dtype=torch.long` for timesteps ✓
- [ ] **Line 288**: Label tensor dtype not specified - could cause issues?

### 10. Model State Issues (MEDIUM PRIORITY)
- [ ] **Line 270**: `model.eval()` - good
- [ ] **Line 283**: `with torch.no_grad()` - good
- [ ] **Line 203**: `with torch.no_grad()` in hook - good

## 🚨 Most Likely Issues (Priority Order)

1. **model.class_embed() returning tuple** - Similar to the fixed issue
2. **Empty layer_names accessing [0]** - IndexError
3. **Empty timesteps list accessing middle element** - IndexError
4. **Incorrect sample format** - KeyError on "pixel_values" or "labels"
5. **model.class_embed structure** - AttributeError if doesn't exist
