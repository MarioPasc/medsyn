# medsyn/data/registry.py
"""
Central registry for all MedMNIST datasets with class mappings and metadata.

This module provides a unified interface for accessing dataset information
and preparing datasets from the MedMNIST collection.
"""
from __future__ import annotations
from typing import Dict, List, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from medsyn.data.config import ProjectCfg
    from medsyn.data.pathmnist import SplitDatasets

# Type alias for supported datasets
DatasetFlag = Literal["pathmnist", "bloodmnist", "dermamnist"]

# Dataset information registry
DATASET_INFO: Dict[str, Dict] = {
    "pathmnist": {
        "num_classes": 9,
        "class_names": [
            "adipose",
            "background",
            "debris",
            "lymphocytes",
            "mucus",
            "smooth_muscle",
            "normal_colon_mucosa",
            "cancer_associated_stroma",
            "colorectal_adenocarcinoma_epithelium",
        ],
        "medmnist_class": "PathMNIST",
        "description": "Colon Pathology - 9 types of colorectal cancer tissues",
    },
    "bloodmnist": {
        "num_classes": 8,
        "class_names": [
            "basophil",
            "eosinophil",
            "erythroblast",
            "immature_granulocyte",
            "lymphocyte",
            "monocyte",
            "neutrophil",
            "platelet",
        ],
        "medmnist_class": "BloodMNIST",
        "description": "Blood Cell Microscopy - 8 types of blood cells",
    },
    "dermamnist": {
        "num_classes": 7,
        "class_names": [
            "actinic_keratoses",
            "basal_cell_carcinoma",
            "benign_keratosis",
            "dermatofibroma",
            "melanoma",
            "melanocytic_nevi",
            "vascular_lesions",
        ],
        "medmnist_class": "DermaMNIST",
        "description": "Dermatoscopy - 7 types of skin lesions",
    },
}


def get_supported_datasets() -> List[str]:
    """Return list of supported dataset flags."""
    return list(DATASET_INFO.keys())


def get_dataset_info(dataset_name: str) -> Dict:
    """
    Get dataset metadata for a given dataset name.

    Args:
        dataset_name: Dataset flag (e.g., 'pathmnist', 'bloodmnist', 'dermamnist')

    Returns:
        Dictionary containing num_classes, class_names, medmnist_class, description

    Raises:
        ValueError: If dataset_name is not supported
    """
    if dataset_name not in DATASET_INFO:
        supported = ", ".join(get_supported_datasets())
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. Supported datasets: {supported}"
        )
    return DATASET_INFO[dataset_name]


def get_num_classes(dataset_name: str) -> int:
    """Get the number of classes for a dataset."""
    return get_dataset_info(dataset_name)["num_classes"]


def get_class_names(dataset_name: str) -> List[str]:
    """Get the list of class names for a dataset."""
    return get_dataset_info(dataset_name)["class_names"]


def get_class_map(dataset_name: str) -> Dict[int, str]:
    """
    Get label-to-class-name mapping for a dataset.

    Args:
        dataset_name: Dataset flag

    Returns:
        Dictionary mapping label indices to class names
    """
    class_names = get_class_names(dataset_name)
    return {i: name for i, name in enumerate(class_names)}


def prepare_dataset(dataset_name: str, cfg: "ProjectCfg") -> "SplitDatasets":
    """
    Prepare a MedMNIST dataset with stratified reduction.

    This is the main entry point for dataset preparation. It dispatches
    to the appropriate loader based on dataset_name.

    Args:
        dataset_name: Dataset flag ('pathmnist', 'bloodmnist', 'dermamnist')
        cfg: Project configuration

    Returns:
        SplitDatasets with train/val/test and the index mapping

    Raises:
        ValueError: If dataset_name is not supported
    """
    # Validate dataset name
    if dataset_name not in DATASET_INFO:
        supported = ", ".join(get_supported_datasets())
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. Supported datasets: {supported}"
        )

    # Import and dispatch to appropriate loader
    from medsyn.data.base import prepare_medmnist

    return prepare_medmnist(dataset_name, cfg)
