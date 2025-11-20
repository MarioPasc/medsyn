#!/usr/bin/env python3
"""
medsyn/cli/generate_ccDDPM_parallel.py

Parallel multi-GPU image generation for ccDDPM models.

This script parallelizes image generation across N GPUs using a work queue approach.
Each generation task (split, class, num_samples) is processed independently on an available GPU.

Features:
- Multi-GPU parallel generation using work queue
- Efficient resource utilization (no idle GPUs)
- Order-independent generation (any split/class can be processed first)
- Progress tracking across all workers
- Aggregated JSON and NPZ outputs
- Professional logging with per-worker logs

Author: M.Pascual-González
Date: 2025
"""
from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
import yaml

# Import from single-GPU version
from medsyn.cli.generate_ccDDPM import (
    load_model_and_scheduler,
    generate_images_for_class,
    build_json_index,
    save_split_npz,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(processName)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class GenerationTask(NamedTuple):
    """A single generation task for one class in one split."""
    split_name: str
    class_id: int
    num_samples: int


class GenerationResult(NamedTuple):
    """Result from a completed generation task."""
    split_name: str
    class_id: int
    samples_metadata: dict[str, dict[str, Any]]
    images_array: np.ndarray
    labels_array: np.ndarray
    success: bool
    error_msg: str = ""


def worker_process(
    gpu_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    config_path: Path,
    checkpoint_path: Path,
    output_base_path: Path,
    create_visualization: bool,
    total_images_in_job: int,
    batch_size: int = 4,
) -> None:
    """
    Worker process that generates images on a specific GPU.

    Each worker:
    1. Loads the model once on its assigned GPU
    2. Pulls tasks from the queue
    3. Generates images for each task
    4. Puts results in the result queue
    5. Continues until queue is empty

    Args:
        gpu_id: CUDA device ID for this worker
        task_queue: Queue of GenerationTask objects
        result_queue: Queue to put GenerationResult objects
        config_path: Path to config YAML
        checkpoint_path: Path to model checkpoint
        output_base_path: Base output directory
        create_visualization: Whether to create denoising visualizations
        total_images_in_job: Total images across all tasks (for global ETA calculation)
        batch_size: Batch size for generation (default: 4)
    """
    # Set up logging for this worker
    worker_logger = logging.getLogger(f"Worker-GPU{gpu_id}")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        f"[%(asctime)s] GPU{gpu_id} - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    worker_logger.addHandler(handler)
    worker_logger.setLevel(logging.INFO)

    try:
        # Set this process to use the assigned GPU
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")

        worker_logger.info(f"Initializing on GPU {gpu_id}")
        worker_logger.info(f"  GPU Name: {torch.cuda.get_device_name(gpu_id)}")
        worker_logger.info(f"  GPU Memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1e9:.2f} GB")

        # Load model once per worker
        worker_logger.info("Loading model and scheduler...")
        model, scheduler, cfg = load_model_and_scheduler(
            config_path, checkpoint_path, device
        )
        worker_logger.info("Model loaded successfully")

        tasks_completed = 0

        # Process tasks from queue
        while True:
            try:
                # Get task with timeout to allow checking for empty queue
                task = task_queue.get(timeout=1)

                if task is None:  # Poison pill to signal completion
                    worker_logger.info(f"Received stop signal. Completed {tasks_completed} tasks.")
                    break

                worker_logger.info(
                    f"Starting task: {task.split_name}/class_{task.class_id} "
                    f"({task.num_samples} samples)"
                )

                task_start_time = time.time()

                # Create split-specific output directory
                split_output_dir = output_base_path / task.split_name
                split_output_dir.mkdir(parents=True, exist_ok=True)

                # Generate images for this class
                try:
                    samples_metadata, images_array, labels_array = generate_images_for_class(
                        model=model,
                        scheduler=scheduler,
                        class_id=task.class_id,
                        num_samples=task.num_samples,
                        output_dir=split_output_dir,
                        config=cfg,
                        device=device,
                        create_visualization=create_visualization,
                        batch_size=batch_size,
                        total_images_in_job=total_images_in_job,
                        images_completed_before_this_class=0,  # Conservative estimate for parallel execution
                    )

                    task_duration = time.time() - task_start_time
                    samples_per_sec = task.num_samples / task_duration if task_duration > 0 else 0

                    worker_logger.info(
                        f"Completed: {task.split_name}/class_{task.class_id} "
                        f"in {task_duration:.1f}s ({samples_per_sec:.2f} samples/s)"
                    )

                    # Put result in queue
                    result = GenerationResult(
                        split_name=task.split_name,
                        class_id=task.class_id,
                        samples_metadata=samples_metadata,
                        images_array=images_array,
                        labels_array=labels_array,
                        success=True,
                    )
                    result_queue.put(result)

                    tasks_completed += 1

                except Exception as e:
                    worker_logger.error(
                        f"Failed to generate {task.split_name}/class_{task.class_id}: {e}",
                        exc_info=True
                    )

                    # Put error result
                    result = GenerationResult(
                        split_name=task.split_name,
                        class_id=task.class_id,
                        samples_metadata={},
                        images_array=np.array([]),
                        labels_array=np.array([]),
                        success=False,
                        error_msg=str(e),
                    )
                    result_queue.put(result)

            except mp.queues.Empty:
                # Queue is empty, check if we should continue
                continue

    except Exception as e:
        worker_logger.error(f"Worker process failed: {e}", exc_info=True)
        raise

    finally:
        worker_logger.info(f"Worker shutting down. Total tasks completed: {tasks_completed}")


def parse_generation_config_parallel(config_path: Path) -> tuple[dict[str, dict[int, int]], Path, Path]:
    """
    Parse the generation configuration from YAML.
    Same format as single-GPU version.
    """
    logger.info("Parsing generation configuration from: %s", config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if "generate" not in config:
        raise ValueError("'generate' section not found in configuration file")

    gen_config = config["generate"]

    # Parse checkpoint path
    if "checkpoint" not in gen_config:
        raise ValueError("'checkpoint' path not specified in generate section")

    checkpoint_path = Path(gen_config["checkpoint"]).expanduser().resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    logger.info("Found checkpoint file: %s", checkpoint_path)

    # Parse npz_with_synth_images section
    if "npz_with_synth_images" not in gen_config:
        raise ValueError("'npz_with_synth_images' section not found in generate section")

    npz_config = gen_config["npz_with_synth_images"]

    # Parse save_to path
    if "save_to" not in npz_config:
        raise ValueError("'save_to' path not specified in npz_with_synth_images section")

    output_base_path = Path(npz_config["save_to"]).expanduser().resolve()

    logger.info("Output base path: %s", output_base_path)

    # Parse splits (train, val, test)
    split_to_class_to_samples: dict[str, dict[int, int]] = {}

    for split_name in ["train", "val", "test"]:
        if split_name not in npz_config:
            logger.debug(f"Split '{split_name}' not found in config, skipping...")
            continue

        split_config = npz_config[split_name]
        if "classes" not in split_config:
            logger.warning(f"'classes' not found in {split_name} section, skipping...")
            continue

        classes_dict = split_config["classes"]
        if not isinstance(classes_dict, dict):
            raise ValueError(f"'classes' in {split_name} must be a dictionary mapping class_id -> num_samples")

        # Convert to int keys and validate
        class_to_samples: dict[int, int] = {}
        for class_id, num_samples in classes_dict.items():
            class_id_int = int(class_id)
            num_samples_int = int(num_samples)
            if num_samples_int <= 0:
                raise ValueError(
                    f"Number of samples must be positive, got {num_samples_int} for class {class_id_int} in split {split_name}"
                )
            class_to_samples[class_id_int] = num_samples_int

        if class_to_samples:
            split_to_class_to_samples[split_name] = class_to_samples
            logger.info("Loaded split '%s' with %d classes and %d total samples",
                       split_name, len(class_to_samples), sum(class_to_samples.values()))

    if not split_to_class_to_samples:
        raise ValueError("No valid splits/classes specified in generate.npz_with_synth_images")

    logger.info("Configuration parsed successfully: %d splits, %d total classes",
               len(split_to_class_to_samples),
               sum(len(classes) for classes in split_to_class_to_samples.values()))

    return split_to_class_to_samples, checkpoint_path, output_base_path


def create_task_queue(split_to_class_to_samples: dict[str, dict[int, int]]) -> list[GenerationTask]:
    """
    Create list of all generation tasks.

    Each task represents generating images for one class in one split.
    Tasks are independent and can be executed in any order.

    Args:
        split_to_class_to_samples: Nested dict of split -> class -> num_samples

    Returns:
        List of GenerationTask objects
    """
    tasks = []

    for split_name, class_to_samples in split_to_class_to_samples.items():
        for class_id, num_samples in class_to_samples.items():
            task = GenerationTask(
                split_name=split_name,
                class_id=class_id,
                num_samples=num_samples,
            )
            tasks.append(task)

    logger.info(f"Created {len(tasks)} generation tasks")

    return tasks


def aggregate_results(
    results: list[GenerationResult],
    output_base_path: Path,
    dataset_name: str = "PathMNIST",
) -> None:
    """
    Aggregate results from all workers into JSON and NPZ files.

    Args:
        results: List of GenerationResult objects from all workers
        output_base_path: Base output directory
        dataset_name: Dataset name for JSON index
    """
    logger.info("Aggregating results from all workers...")

    # Group results by split
    split_to_results: dict[str, list[GenerationResult]] = defaultdict(list)

    for result in results:
        if result.success:
            split_to_results[result.split_name].append(result)
        else:
            logger.error(f"Skipping failed task: {result.split_name}/class_{result.class_id}: {result.error_msg}")

    # Process each split
    for split_name, split_results in split_to_results.items():
        logger.info(f"Processing split '{split_name}' with {len(split_results)} successful tasks")

        split_output_dir = output_base_path / split_name

        # Collect all samples, images, and labels
        all_samples: dict[int, dict[str, dict[str, Any]]] = {}
        all_images: dict[int, np.ndarray] = {}
        all_labels: dict[int, np.ndarray] = {}

        for result in sorted(split_results, key=lambda r: r.class_id):
            all_samples[result.class_id] = result.samples_metadata
            all_images[result.class_id] = result.images_array
            all_labels[result.class_id] = result.labels_array

            logger.debug(f"  Class {result.class_id}: {len(result.images_array)} samples")

        # Build and save JSON index
        logger.info(f"Building JSON index for split '{split_name}'...")
        json_index = build_json_index(
            all_samples,
            dataset_name=dataset_name,
            split_name=split_name,
        )

        json_path = split_output_dir / f"{dataset_name.lower()}_{split_name}_index.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_index, f, ensure_ascii=False, indent=2)

        logger.info(f"JSON index saved to: {json_path}")

        # Save NPZ file
        logger.info(f"Saving NPZ file for split '{split_name}'...")
        npz_path = split_output_dir / f"{dataset_name.lower()}_{split_name}_synth.npz"
        save_split_npz(
            all_images=all_images,
            all_labels=all_labels,
            split_name=split_name,
            output_path=npz_path,
        )

        total_samples = sum(len(arr) for arr in all_images.values())
        logger.info(f"Split '{split_name}' aggregation complete:")
        logger.info(f"  Total samples: {total_samples}")
        logger.info(f"  JSON index: {json_path}")
        logger.info(f"  NPZ file: {npz_path}")


def main() -> None:
    """
    Entry point for parallel ccDDPM image generation.
    """
    parser = argparse.ArgumentParser(
        description="Generate synthetic images with trained ccDDPM (parallel multi-GPU)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use 4 GPUs
  python -m medsyn.cli.generate_ccDDPM_parallel config/medsyn_cfg.yaml --num-gpus 4

  # Use specific GPUs
  CUDA_VISIBLE_DEVICES=0,1,3 python -m medsyn.cli.generate_ccDDPM_parallel config/medsyn_cfg.yaml --num-gpus 3

  # Disable visualizations for faster generation
  python -m medsyn.cli.generate_ccDDPM_parallel config/medsyn_cfg.yaml --num-gpus 8 --no-visualizations
        """,
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: all available)",
    )
    parser.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Disable denoising process visualizations",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="PathMNIST",
        help="Dataset name for JSON index (default: PathMNIST)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of images to generate in parallel per batch (default: 4, recommended: 2-8)",
    )
    args = parser.parse_args()

    config_path = Path(args.config)

    if not config_path.exists():
        logger.error("Config file not found: %s", config_path)
        sys.exit(1)

    print("=" * 80)
    print("   Class-Conditioned DDPM - Parallel Multi-GPU Image Generation")
    print("=" * 80)
    logger.info("Configuration file: %s", config_path.absolute())

    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. This script requires GPU(s).")
        sys.exit(1)

    # Determine number of GPUs to use
    num_gpus_available = torch.cuda.device_count()
    if args.num_gpus is None:
        num_gpus = num_gpus_available
    else:
        num_gpus = min(args.num_gpus, num_gpus_available)

    if num_gpus <= 0:
        logger.error("No GPUs available or --num-gpus set to 0")
        sys.exit(1)

    logger.info(f"Using {num_gpus} GPU(s) out of {num_gpus_available} available")
    for i in range(num_gpus):
        logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    try:
        # Parse generation configuration
        logger.info("Step 1: Parsing generation configuration")
        split_to_class_to_samples, checkpoint_path, output_base_path = parse_generation_config_parallel(config_path)

        # Create task queue
        logger.info("Step 2: Creating task queue")
        tasks = create_task_queue(split_to_class_to_samples)

        # Display configuration
        print("\nGeneration Configuration:")
        print(f"  Checkpoint: {checkpoint_path}")
        print(f"  Output directory: {output_base_path}")
        print(f"  Number of GPUs: {num_gpus}")
        print(f"  Total tasks: {len(tasks)}")
        print(f"  Visualizations: {'Disabled' if args.no_visualizations else 'Enabled'}")

        print("\nSamples per split and class:")
        total_samples = 0
        for split_name in sorted(split_to_class_to_samples.keys()):
            class_to_samples = split_to_class_to_samples[split_name]
            split_total = sum(class_to_samples.values())
            print(f"  {split_name.capitalize()} split: {split_total} samples")
            for class_id in sorted(class_to_samples.keys()):
                num_samples = class_to_samples[class_id]
                print(f"    Class {class_id}: {num_samples} samples")
            total_samples += split_total
        print(f"  Grand total: {total_samples} images")

        output_base_path.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print("Starting parallel generation...")
        print("=" * 80 + "\n")

        # Create task and result queues
        mp.set_start_method('spawn', force=True)
        task_queue = mp.Queue()
        result_queue = mp.Queue()

        # Fill task queue
        for task in tasks:
            task_queue.put(task)

        # Add poison pills (one per worker)
        for _ in range(num_gpus):
            task_queue.put(None)

        logger.info("Task queue created with %d tasks", len(tasks))

        # Calculate total images in job for global ETA tracking
        total_images_in_job = sum(task.num_samples for task in tasks)

        # Start worker processes
        logger.info("Starting %d worker processes...", num_gpus)
        workers = []

        start_time = time.time()

        for gpu_id in range(num_gpus):
            worker = mp.Process(
                target=worker_process,
                args=(
                    gpu_id,
                    task_queue,
                    result_queue,
                    config_path,
                    checkpoint_path,
                    output_base_path,
                    not args.no_visualizations,
                    total_images_in_job,
                    args.batch_size,
                ),
                name=f"Worker-GPU{gpu_id}",
            )
            worker.start()
            workers.append(worker)
            logger.info(f"Started worker for GPU {gpu_id} (PID: {worker.pid})")

        # Monitor progress
        results = []
        completed_tasks = 0

        logger.info("Waiting for workers to complete...")

        while completed_tasks < len(tasks):
            try:
                result = result_queue.get(timeout=1)
                results.append(result)
                completed_tasks += 1

                progress_pct = (completed_tasks / len(tasks)) * 100
                logger.info(
                    f"Progress: {completed_tasks}/{len(tasks)} ({progress_pct:.1f}%) - "
                    f"Latest: {result.split_name}/class_{result.class_id}"
                )

            except mp.queues.Empty:
                # Check if any workers have died
                for worker in workers:
                    if not worker.is_alive() and worker.exitcode != 0:
                        logger.error(f"Worker {worker.name} died unexpectedly with code {worker.exitcode}")
                        raise RuntimeError(f"Worker process failed: {worker.name}")

        # Wait for all workers to finish
        logger.info("All tasks completed. Waiting for workers to shut down...")
        for worker in workers:
            worker.join(timeout=30)
            if worker.is_alive():
                logger.warning(f"Worker {worker.name} did not shut down gracefully, terminating...")
                worker.terminate()
                worker.join()

        elapsed_time = time.time() - start_time
        samples_per_second = total_samples / elapsed_time if elapsed_time > 0 else 0

        logger.info("=" * 60)
        logger.info("All workers completed successfully")
        logger.info(f"  Total time: {elapsed_time:.1f}s")
        logger.info(f"  Total samples: {total_samples}")
        logger.info(f"  Throughput: {samples_per_second:.2f} samples/s")
        logger.info("=" * 60)

        # Aggregate results
        print("\n" + "=" * 80)
        print("Aggregating results from all workers...")
        print("=" * 80 + "\n")

        logger.info("Step 3: Aggregating results")
        aggregate_results(results, output_base_path, args.dataset_name)

        # Summary
        print("\n" + "=" * 80)
        print("Generation completed successfully!")
        print("=" * 80)
        logger.info("=" * 60)
        logger.info("PARALLEL GENERATION PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info("Total images generated: %d", total_samples)
        logger.info("Total time: %.1f seconds", elapsed_time)
        logger.info("Throughput: %.2f samples/second", samples_per_second)
        logger.info("Splits processed: %s", ', '.join(sorted(split_to_class_to_samples.keys())))
        logger.info("Output directory: %s", output_base_path)

        print(f"  Total images generated: {total_samples}")
        print(f"  Total time: {elapsed_time:.1f} seconds")
        print(f"  Throughput: {samples_per_second:.2f} samples/second")
        print(f"  Output directory: {output_base_path}")
        print(f"  Splits processed: {', '.join(sorted(split_to_class_to_samples.keys()))}")

        if not args.no_visualizations:
            num_vis = sum(1 for _ in output_base_path.rglob("denoising_process_*.png"))
            print(f"  Denoising visualizations: {num_vis}")
            logger.info("Denoising visualizations created: %d", num_vis)

        print("=" * 80 + "\n")
        logger.info("=" * 60)

    except Exception as e:
        print("\n" + "=" * 80)
        print("Generation failed!")
        print("=" * 80)
        logger.error("=" * 60)
        logger.error("PARALLEL GENERATION PIPELINE FAILED")
        logger.error("=" * 60)
        logger.error("Error: %s", e, exc_info=True)
        logger.error("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
