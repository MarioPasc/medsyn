#!/usr/bin/env python3
"""
CLI script to execute all classification experiments for a list of models.
"""

import argparse
import logging
import subprocess
import sys
import yaml
from pathlib import Path
from typing import List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

EXPERIMENT_FOLDERS = [
    "cfg_medsyn",
    "distdiff",
    "only_real",
    "real_traditional_augmentation"
]

CLIP_IMAGE_SIZE = 224

def run_command(command: List[str]):
    """Run a shell command."""
    logger.info(f"Running command: {' '.join(command)}")
    try:
        # Use subprocess.run to let the command inherit stdout/stderr
        # This preserves TTY for tqdm so it updates in-place instead of printing new lines
        result = subprocess.run(
            command,
            check=False
        )
        
        if result.returncode != 0:
            logger.error(f"Command failed with return code {result.returncode}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Failed to execute command: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Execute all classification experiments for given models"
    )
    
    parser.add_argument(
        "--config_root",
        type=str,
        default="config/classification",
        help="Root directory for classification configs"
    )
    
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="List of models to run (e.g., resnet50 clip_vit_b16)"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing them"
    )

    parser.add_argument(
        "--skip_experiments",
        type=str,
        nargs="+",
        default=[],
        help="List of experiment folders to skip (e.g., cfg_medsyn distdiff)"
    )

    parser.add_argument(
        "--device",
        type=str,
        help="Override device for all experiments (e.g., cuda:0)"
    )
    
    args = parser.parse_args()
    
    config_root = Path(args.config_root).resolve()
    if not config_root.exists():
        logger.error(f"Config root not found: {config_root}")
        sys.exit(1)
        
    models_dir = config_root / "models"
    experiments_dir = config_root / "experiments"
    
    # Validate models
    for model in args.models:
        model_config = models_dir / f"{model}.yaml"
        if not model_config.exists():
            logger.error(f"Model config not found: {model_config}")
            sys.exit(1)
            
    # Execute experiments
    failed_experiments = []
    
    for model in args.models:
        logger.info(f"\n{'='*80}")
        logger.info(f"Running experiments for model: {model}")
        logger.info(f"{'='*80}")
        
        for folder in EXPERIMENT_FOLDERS:
            if folder in args.skip_experiments:
                logger.info(f"Skipping experiment folder: {folder}")
                continue

            folder_path = experiments_dir / folder
            if not folder_path.exists():
                logger.warning(f"Experiment folder not found: {folder_path}")
                continue
                
            # Find all YAML files in the folder
            yaml_files = sorted(list(folder_path.glob("*.yaml")))
            
            if not yaml_files:
                logger.warning(f"No YAML files found in {folder_path}")
                continue
                
            for yaml_file in yaml_files:
                logger.info(f"\nRunning experiment: {folder}/{yaml_file.name}")
                
                # Read YAML to determine experiment name and output dir
                try:
                    with open(yaml_file, 'r') as f:
                        config = yaml.safe_load(f)
                    
                    exp_config = config.get('experiment', {})
                    exp_name = exp_config.get('name', '')
                    output_dir = exp_config.get('output_dir', '')
                    
                    # Dynamic replacement
                    if "{classifier}" in exp_name:
                        exp_name = exp_name.replace("{classifier}", model)
                    
                    if "{experiment_name}" in output_dir:
                        output_dir = output_dir.replace("{experiment_name}", exp_name)
                        
                except Exception as e:
                    logger.warning(f"Could not parse YAML for dynamic naming: {e}")
                    exp_name = None
                    output_dir = None

                # Construct command
                cmd = [
                    sys.executable,
                    "-m", "medsyn.cli.train_classifier",
                    str(yaml_file),
                    "--model", model
                ]
                
                if exp_name:
                    cmd.extend(["--experiment_name", exp_name])
                
                if output_dir:
                    cmd.extend(["--output_dir", output_dir])
                
                if args.device:
                    cmd.extend(["--device", args.device])

                if model == "clip_vit_b16":
                    cmd.extend(["--image_size", str(CLIP_IMAGE_SIZE)])
                
                if args.dry_run:
                    print(f"Would run: {' '.join(cmd)}")
                else:
                    success = run_command(cmd)
                    if not success:
                        failed_experiments.append(f"{model} - {folder}/{yaml_file.name}")
                        
    if failed_experiments:
        logger.error("\nThe following experiments failed:")
        for exp in failed_experiments:
            logger.error(f"  - {exp}")
        sys.exit(1)
    else:
        logger.info("\nAll experiments completed successfully!")

if __name__ == "__main__":
    main()
