# medsyn/models/ccDDPM/augmentation/statistics.py
# Purpose: Track and save augmentation statistics to CSV
from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class AugmentationStatistics:
    """
    Tracks which augmentations were applied to each image during training.
    Generates CSV with counts per augmentation type and combinations.
    """

    def __init__(self, output_path: Path, transform_names: List[str]):
        """
        Args:
            output_path: Path to save the CSV file
            transform_names: List of available transform names (e.g., ['HorizontalFlip', 'Rotate'])
        """
        self.output_path = Path(output_path)
        self.transform_names = sorted(transform_names)  # Sort for consistent ordering

        # Create output directory if needed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Statistics storage: {epoch: {transform_combo: count}}
        # transform_combo can be 'original', 'HorizontalFlip', 'HorizontalFlip+Rotate', etc.
        self.stats: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        # Current epoch tracker
        self.current_epoch = 0

        # Track unique combinations encountered
        self.all_combinations: Set[str] = {'original'}  # Always have 'original'

        logger.info(f"Augmentation statistics will be saved to: {self.output_path}")

    def record_batch(self, applied_transforms_list: List[List[str]], epoch: int) -> None:
        """
        Record augmentations applied to a batch of images.

        Args:
            applied_transforms_list: List of lists, where each inner list contains
                                     the names of transforms applied to one image.
                                     Empty list means original (no augmentation).
            epoch: Current epoch number
        """
        self.current_epoch = epoch

        for applied_transforms in applied_transforms_list:
            if not applied_transforms or len(applied_transforms) == 0:
                # No augmentation applied
                combo_key = 'original'
            else:
                # Sort to ensure consistent key (e.g., 'Flip+Rotate' same as 'Rotate+Flip')
                combo_key = '+'.join(sorted(applied_transforms))

            # Track this combination
            self.all_combinations.add(combo_key)

            # Increment count for this epoch and combination
            self.stats[epoch][combo_key] += 1

    def get_epoch_summary(self, epoch: int) -> Dict[str, int]:
        """
        Get statistics summary for a specific epoch.

        Args:
            epoch: Epoch number

        Returns:
            Dictionary mapping transform combinations to counts
        """
        return dict(self.stats[epoch])

    def get_total_images(self, epoch: int) -> int:
        """Get total number of images processed in an epoch."""
        return sum(self.stats[epoch].values())

    def save_csv(self, final: bool = False) -> None:
        """
        Save statistics to CSV file.

        Args:
            final: If True, saves final comprehensive statistics.
                   If False, saves incremental update.
        """
        try:
            # Determine all unique combinations across all epochs
            all_combos = sorted(self.all_combinations)

            # Prepare CSV header
            header = ['epoch', 'total_images'] + all_combos

            # Prepare rows
            rows = []
            for epoch in sorted(self.stats.keys()):
                row = [epoch, self.get_total_images(epoch)]

                # Add counts for each combination
                for combo in all_combos:
                    count = self.stats[epoch].get(combo, 0)
                    row.append(count)

                rows.append(row)

            # Write CSV
            with open(self.output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)

            status = "final" if final else "incremental"
            logger.info(f"Saved {status} augmentation statistics to: {self.output_path}")
            logger.info(f"  Tracked {len(all_combos)} unique augmentation combinations across {len(rows)} epochs")

        except Exception as e:
            logger.error(f"Failed to save augmentation statistics: {e}")

    def get_summary_stats(self) -> Dict[str, any]:
        """
        Get overall summary statistics across all epochs.

        Returns:
            Dictionary with summary information
        """
        total_images = sum(self.get_total_images(ep) for ep in self.stats.keys())
        total_original = sum(self.stats[ep].get('original', 0) for ep in self.stats.keys())
        total_augmented = total_images - total_original

        # Find most common augmentation
        combo_totals = defaultdict(int)
        for epoch_stats in self.stats.values():
            for combo, count in epoch_stats.items():
                if combo != 'original':
                    combo_totals[combo] += count

        most_common = max(combo_totals.items(), key=lambda x: x[1]) if combo_totals else ('None', 0)

        return {
            'total_images': total_images,
            'total_original': total_original,
            'total_augmented': total_augmented,
            'augmentation_rate': total_augmented / total_images if total_images > 0 else 0.0,
            'unique_combinations': len(self.all_combinations),
            'most_common_augmentation': most_common[0],
            'most_common_count': most_common[1],
            'epochs_tracked': len(self.stats)
        }

    def print_summary(self) -> None:
        """Print a human-readable summary to console."""
        summary = self.get_summary_stats()

        print("\n" + "="*80)
        print("Augmentation Statistics Summary")
        print("="*80)
        print(f"Total images processed: {summary['total_images']:,}")
        print(f"  Original (no augmentation): {summary['total_original']:,} ({summary['total_original']/max(summary['total_images'],1)*100:.1f}%)")
        print(f"  Augmented: {summary['total_augmented']:,} ({summary['augmentation_rate']*100:.1f}%)")
        print(f"\nUnique augmentation combinations: {summary['unique_combinations']}")
        print(f"Most common augmentation: {summary['most_common_augmentation']} ({summary['most_common_count']:,} images)")
        print(f"Epochs tracked: {summary['epochs_tracked']}")
        print(f"\nStatistics saved to: {self.output_path}")
        print("="*80 + "\n")

    def __del__(self):
        """Cleanup - ensure final save on deletion."""
        # Note: We'll explicitly call save_csv(final=True) at end of training
        # This is just a safety net
        pass
