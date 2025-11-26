# DistDiff Parallel Generation Bug Fix

## Date
2025-11-26

## Issue
When running parallel image generation using 4 GPUs (via `scripts/distdiff_picasso_job.sh`), only `split_0/` was populated with images. The other 3 GPU processes (split_1, split_2, split_3) were running but not saving any output.

## Root Cause
The `generate_data.py` script uses HuggingFace's Accelerate library, which is designed for distributed training. When multiple independent processes are launched (one per GPU using background jobs with `&`), Accelerate detects them and designates only ONE process as the "main process" (typically split_0).

Critical code sections guarded by `accelerator.is_main_process`:
1. **Output directory creation** (line ~941) - Only split_0 created its output directory
2. **Image saving** (line ~1311) - Only split_0 saved generated images to disk
3. **Tracker initialization** (line ~1150) - Only split_0 initialized experiment tracking

All 4 processes performed GPU computation (denoising loops), but splits 1-3 discarded results instead of saving them.

## Solution Implemented
**Option 1: Remove Accelerator main process checks for independent processes**

Modified `medsyn/models/distdiff/generate_data.py` to allow all processes to act independently:

### Changes Made

#### 1. Output Directory Creation (line ~941)
**BEFORE:**
```python
if accelerator.is_main_process:
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.logging_dir, exist_ok=True)
```

**AFTER:**
```python
# Allow all parallel processes to create their own output directories
if args.output_dir is not None:
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.logging_dir, exist_ok=True)
```

#### 2. Image Saving (line ~1311)
**BEFORE:**
```python
                    if accelerator.is_main_process:
                        image = decoded_original_latent
                        image = image_processor.postprocess(...)

                        for i in range(len(batch["image_paths"])):
                            # ... save images ...
```

**AFTER:**
```python
                    # Allow all parallel processes to save their split's images
                    image = decoded_original_latent
                    image = image_processor.postprocess(...)

                    for i in range(len(batch["image_paths"])):
                        # ... save images ...
```

#### 3. Tracker Initialization (line ~1150)
**BEFORE:**
```python
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        tracker_config.pop("validation_images")
        accelerator.init_trackers("dreambooth", config=tracker_config)
```

**AFTER:**
```python
    # Allow each process to track independently (each split has its own tracker)
    tracker_config = vars(copy.deepcopy(args))
    tracker_config.pop("validation_images")
    accelerator.init_trackers("dreambooth", config=tracker_config)
```

#### 4. Progress Logging (lines ~1284, 1294, 1300)
**BEFORE:**
```python
if accelerator.is_main_process: logging.info("Guidance timesteps: ...")
```

**AFTER:**
```python
# Changed to is_local_main_process to avoid excessive logging while allowing all processes to log
if accelerator.is_local_main_process: logging.info("Guidance timesteps: ...")
```

**Note:** `is_local_main_process` is less restrictive and appropriate for logging when processes run independently.

## How to Revert Changes

If this fix causes issues, revert to the original behavior:

### Step 1: Restore Original generate_data.py

```bash
cd /home/mpascual/research/code/medsyn
git diff medsyn/models/distdiff/generate_data.py > /tmp/parallel_fix.patch
git checkout medsyn/models/distdiff/generate_data.py
```

### Step 2: Manual Reversion (if not using git)

Replace the modified sections with the original code:

#### Revert Output Directory Creation (~line 941)
```python
if accelerator.is_main_process:
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.logging_dir, exist_ok=True)
```

#### Revert Image Saving (~line 1311)
```python
                    if accelerator.is_main_process:
                        image = decoded_original_latent
                        image = image_processor.postprocess(image, output_type="pt",
                                                            do_denormalize=[True] * image.shape[0])

                        for i in range(len(batch["image_paths"])):
                            image_file_path = os.path.basename(batch["image_paths"][i]).split(".")[0]
                            class_name = batch["class_names"][i]
                            path = f'{args.output_dir}/{class_name}/{image_file_path}_expand_{image_i}.png'

                            try:
                                os.makedirs(os.path.dirname(path), exist_ok=True)
                                save_image([image[i]], path)
                                generated_count += 1

                                if generated_count <= 5 or generated_count % 100 == 0:
                                    logger.info(f"Saved image {generated_count}: {path}")
                            except Exception as e:
                                logger.error(f"Failed to save image to {path}: {e}")
                                raise
```

#### Revert Tracker Initialization (~line 1150)
```python
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        tracker_config.pop("validation_images")
        accelerator.init_trackers("dreambooth", config=tracker_config)
```

#### Revert Logging Checks (~lines 1284, 1294, 1300)
```python
if accelerator.is_main_process: logging.info("Guidance timesteps: %s", ", ".join([str(x) for x in guide_timesteps]))
# ... (similar for other logging statements)
```

## Alternative Solutions (Not Implemented)

### Option 2: Force Single-Process Mode
Initialize Accelerator to treat each process as independent:
```python
accelerator = Accelerator(
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    mixed_precision=args.mixed_precision,
    log_with=args.report_to,
    project_config=accelerator_project_config,
    # Force single-process mode
    cpu=False,
)
```
Set environment variable in job script:
```bash
export ACCELERATE_TORCH_DEVICE=cuda
```

### Option 3: Proper Distributed Training
Refactor to use `accelerate launch` with proper distributed setup. Would require changes to bash script and more extensive refactoring.

## Testing

To verify the fix works:

1. Submit the job with 4 GPUs
2. Check that all 4 splits generate data:
   ```bash
   ls -la $RESULTS_DST/synthetic_data/
   # Should show: split_0/ split_1/ split_2/ split_3/
   ```

3. Verify each split contains class directories with images:
   ```bash
   for split in {0..3}; do
     echo "=== Split $split ==="
     ls $RESULTS_DST/synthetic_data/split_${split}/ | wc -l
   done
   ```

4. Check generation logs:
   ```bash
   tail -n 50 $RESULTS_DST/logs/generation_split_*.log
   ```

## Known Limitations

- Each process will create its own tracker (if using wandb/tensorboard), resulting in 4 separate experiment runs
- Logging may be more verbose with all processes logging independently
- No coordination between processes (but this is expected for data-parallel generation)

## References

- Job script: `scripts/distdiff_picasso_job.sh` (lines 444-471)
- Generation script: `medsyn/models/distdiff/generate_data.py`
- HuggingFace Accelerate docs: https://huggingface.co/docs/accelerate/
