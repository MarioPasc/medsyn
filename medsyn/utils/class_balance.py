"""
Utilidad para analizar y mostrar el balanceo de clases en los datasets de MedSyn.
"""
from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
from collections import Counter
import json
import logging
import numpy as np

logger = logging.getLogger(__name__)


def load_balance_from_config(config_path: str | Path) -> Dict[str, Dict[int, int]]:
    """
    Carga el balanceo de clases usando la configuración de MedSyn.
    
    Args:
        config_path: Ruta al archivo de configuración YAML.
        
    Returns:
        Diccionario con el conteo de clases por split.
    """
    try:
        from medsyn.data.config import load_config
        cfg = load_config(config_path)
        
        # Prioritize NPZ file if configured
        if cfg.data.postprocess_npz.enabled and cfg.data.postprocess_npz.npz_path:
            npz_path = Path(cfg.data.postprocess_npz.npz_path)
            if npz_path.exists():
                logger.info(f"Using NPZ file: {npz_path}")
                return analyze_class_balance_npz(npz_path)
            else:
                logger.warning(f"NPZ file not found: {npz_path}, falling back to JSON")
        
        # Fallback to JSON if available
        if cfg.data.save_png.enabled and cfg.data.save_png.index_json:
            json_path = Path(cfg.data.save_png.index_json)
            if json_path.exists():
                logger.info(f"Using JSON file: {json_path}")
                return analyze_class_balance_json(json_path)
            else:
                logger.error(f"JSON file not found: {json_path}")
                raise FileNotFoundError(f"JSON file not found: {json_path}")
        
        raise ValueError("No valid data source configured (neither NPZ nor JSON enabled/found)")
        
    except ImportError:
        logger.error("No se pudo importar load_config. Asegúrate de que el módulo data.config existe.")
        raise
    except Exception as e:
        logger.error(f"Error al cargar configuración: {e}")
        raise


def analyze_class_balance_npz(npz_path: str | Path) -> Dict[str, Dict[int, int]]:
    """
    Analiza el balanceo de clases desde un archivo NPZ.
    
    Args:
        npz_path: Ruta al archivo NPZ creado por _create_custom_npz.
        
    Returns:
        Diccionario con el conteo de clases por split:
        {
            "train": {0: 100, 1: 150, ...},
            "val": {0: 20, 1: 30, ...},
            "test": {0: 50, 1: 60, ...}
        }
    """
    npz_path = Path(npz_path)
    
    if not npz_path.exists():
        raise FileNotFoundError(f"El archivo NPZ no existe: {npz_path}")
    
    logger.info(f"Cargando datos desde NPZ: {npz_path}")
    
    data = np.load(str(npz_path))
    
    balance: Dict[str, Dict[int, int]] = {}
    
    for split in ["train", "val", "test"]:
        labels_key = f"{split}_labels"
        
        if labels_key not in data:
            logger.warning(f"Split '{split}' no encontrado en el NPZ")
            balance[split] = {}
            continue
        
        labels = data[labels_key].flatten()  # Ensure 1D array
        label_counts = Counter(labels.tolist())
        balance[split] = dict(label_counts)
        
        logger.info(f"Split '{split}': {len(labels)} muestras, {len(balance[split])} clases")
    
    return balance


def analyze_class_balance(json_path: str | Path) -> Dict[str, Dict[int, int]]:
    """
    Analiza el balanceo de clases desde un archivo JSON de índices.
    DEPRECATED: Use analyze_class_balance_json instead.
    
    Args:
        json_path: Ruta al archivo JSON generado por el módulo data.
        
    Returns:
        Diccionario con el conteo de clases por split.
    """
    return analyze_class_balance_json(json_path)


def analyze_class_balance_json(json_path: str | Path) -> Dict[str, Dict[int, int]]:
    """
    Analiza el balanceo de clases desde un archivo JSON de índices.
    
    Args:
        json_path: Ruta al archivo JSON generado por el módulo data.
        
    Returns:
        Diccionario con el conteo de clases por split:
        {
            "train": {0: 100, 1: 150, ...},
            "val": {0: 20, 1: 30, ...},
            "test": {0: 50, 1: 60, ...}
        }
    """
    json_path = Path(json_path)
    
    if not json_path.exists():
        raise FileNotFoundError(f"El archivo JSON no existe: {json_path}")
    
    logger.info(f"Cargando índices desde {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Extraer el dataset (asumiendo estructura {"PathMNIST": {...}} o similar)
    # Buscar el primer nivel que contenga splits
    if "PathMNIST" in data:
        dataset_data = data["PathMNIST"]
    else:
        # Asumir que data ya contiene los splits directamente
        dataset_data = data
    
    balance: Dict[str, Dict[int, int]] = {}
    
    for split in ["train", "val", "test"]:
        if split not in dataset_data:
            logger.warning(f"Split '{split}' no encontrado en el JSON")
            balance[split] = {}
            continue
        
        split_data = dataset_data[split]
        labels = [entry["label"] for entry in split_data.values()]
        balance[split] = dict(Counter(labels))
    
    return balance


def print_synthesis_requirements(balance: Dict[int, int], split_name: str) -> None:
    """
    Imprime una tabla con los requerimientos de síntesis para balancear un split.
    
    Args:
        balance: Diccionario con el conteo de clases para el split (e.g., {0: 100, 1: 150, ...}).
        split_name: Nombre del split (e.g., "train", "val").
    """
    if not balance:
        print(f"\nNo hay datos para el split '{split_name}'")
        return
    
    classes = sorted(balance.keys())
    max_count = max(balance.values())
    
    # Calcular anchos de columna
    class_width = max(len("Class"), max(len(str(c)) for c in classes))
    real_width = max(len("Real"), len(str(max_count)))
    synth_width = max(len("Synth"), len(str(max_count)))
    total_width = max(len("Total"), len(str(max_count)))
    
    # Imprimir encabezado
    print("\n" + "=" * 70)
    print(f"SYNTHESIS REQUIREMENTS - {split_name.upper()} FOLD")
    print("=" * 70)
    print(f"Target: {max_count} images per class (maximum class count)")
    print("-" * 70)
    
    header = f"{'Class':<{class_width}} | {'Real':>{real_width}} | {'Synth':>{synth_width}} | {'Total':>{total_width}}"
    print(header)
    print("-" * 70)
    
    # Imprimir filas de datos
    total_real = 0
    total_synth = 0
    
    for cls in classes:
        real_count = balance[cls]
        synth_needed = max_count - real_count
        total_count = max_count
        
        total_real += real_count
        total_synth += synth_needed
        
        row = f"{cls:<{class_width}} | {real_count:>{real_width}} | {synth_needed:>{synth_width}} | {total_count:>{total_width}}"
        print(row)
    
    # Imprimir totales
    print("-" * 70)
    total_row = f"{'TOTAL':<{class_width}} | {total_real:>{real_width}} | {total_synth:>{synth_width}} | {total_real + total_synth:>{total_width}}"
    print(total_row)
    print("=" * 70)
    
    # Estadísticas
    n_classes = len(classes)
    print(f"\nSummary:")
    print(f"  Classes: {n_classes}")
    print(f"  Real images: {total_real}")
    print(f"  Synthetic images needed: {total_synth}")
    print(f"  Total after balancing: {total_real + total_synth}")
    print(f"  Avg synthetic per class: {total_synth / n_classes:.1f}")
    print()


def print_class_balance_table(balance: Dict[str, Dict[int, int]]) -> None:
    """
    Imprime una tabla formateada con el balanceo de clases.
    
    Args:
        balance: Diccionario con el conteo de clases por split.
    """
    # Obtener todas las clases únicas
    all_classes: set[int] = set()
    for split_counts in balance.values():
        all_classes.update(split_counts.keys())
    
    classes = sorted(all_classes)
    
    if not classes:
        print("No se encontraron clases en los datos.")
        return
    
    # Calcular anchos de columna
    class_width = max(len("Class"), max(len(str(c)) for c in classes))
    split_widths = {}
    
    for split in ["train", "val", "test"]:
        if split in balance:
            max_count = max(balance[split].values()) if balance[split] else 0
            split_widths[split] = max(len(split.capitalize()), len(str(max_count)), 5)
    
    # Imprimir encabezado
    print("\n" + "=" * 80)
    print("BALANCEO DE CLASES POR SPLIT")
    print("=" * 80)
    
    # Construir línea de encabezado
    header = f"{'Class':<{class_width}}"
    for split in ["train", "val", "test"]:
        if split in balance:
            header += f" | {split.capitalize():>{split_widths[split]}}"
    header += " | Total"
    
    print(header)
    print("-" * len(header))
    
    # Imprimir filas de datos
    total_per_split = {split: 0 for split in balance.keys()}
    
    for cls in classes:
        row = f"{cls:<{class_width}}"
        row_total = 0
        
        for split in ["train", "val", "test"]:
            if split in balance:
                count = balance[split].get(cls, 0)
                row += f" | {count:>{split_widths[split]}}"
                total_per_split[split] += count
                row_total += count
        
        row += f" | {row_total}"
        print(row)
    
    # Imprimir totales
    print("-" * len(header))
    total_row = f"{'TOTAL':<{class_width}}"
    grand_total = 0
    
    for split in ["train", "val", "test"]:
        if split in balance:
            total = total_per_split[split]
            total_row += f" | {total:>{split_widths[split]}}"
            grand_total += total
    
    total_row += f" | {grand_total}"
    print(total_row)
    print("=" * 80)
    
    # Imprimir estadísticas adicionales
    print("\nESTADÍSTICAS:")
    for split in ["train", "val", "test"]:
        if split in balance and balance[split]:
            counts = list(balance[split].values())
            n_classes = len(balance[split])
            total = sum(counts)
            avg = total / n_classes if n_classes > 0 else 0
            min_count = min(counts) if counts else 0
            max_count = max(counts) if counts else 0
            
            print(f"  {split.capitalize():5s}: {total:5d} muestras | "
                  f"{n_classes} clases | "
                  f"avg={avg:.1f} | min={min_count} | max={max_count}")
    
    print()
    
    # Imprimir tablas de requerimientos de síntesis para train y val
    if "train" in balance:
        print_synthesis_requirements(balance["train"], "train")
    
    if "val" in balance:
        print_synthesis_requirements(balance["val"], "val")


def show_class_balance(json_path: str | Path) -> None:
    """
    Función principal que analiza y muestra el balanceo de clases desde JSON.
    
    Args:
        json_path: Ruta al archivo JSON de índices.
    """
    try:
        balance = analyze_class_balance_json(json_path)
        print_class_balance_table(balance)
    except Exception as e:
        logger.error(f"Error al analizar balanceo de clases: {e}")
        raise


def show_class_balance_npz(npz_path: str | Path) -> None:
    """
    Función principal que analiza y muestra el balanceo de clases desde NPZ.
    
    Args:
        npz_path: Ruta al archivo NPZ de datos.
    """
    try:
        balance = analyze_class_balance_npz(npz_path)
        print_class_balance_table(balance)
    except Exception as e:
        logger.error(f"Error al analizar balanceo de clases: {e}")
        raise


if __name__ == "__main__":
    # Ejemplo de uso
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analizar y mostrar el balanceo de clases de MedSyn datasets"
    )
    parser.add_argument(
        "--json",
        type=str,
        help="Ruta al archivo JSON de índices"
    )
    parser.add_argument(
        "--npz",
        type=str,
        help="Ruta al archivo NPZ de datos (formato custom de MedSyn)"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Ruta al archivo de configuración YAML (alternativa a --json/--npz)"
    )
    
    args = parser.parse_args()
    
    if args.config:
        balance = load_balance_from_config(args.config)
        print_class_balance_table(balance)
    elif args.npz:
        show_class_balance_npz(args.npz)
    elif args.json:
        show_class_balance(args.json)
    else:
        # Usar ruta por defecto del config
        default_config = "/home/mpascual/research/code/medsyn/config/medsyn_cfg.yaml"
        if Path(default_config).exists():
            print(f"Using default config: {default_config}")
            balance = load_balance_from_config(default_config)
            print_class_balance_table(balance)
        else:
            parser.print_help()
            sys.exit(1)
