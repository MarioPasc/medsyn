"""
Utilidad para analizar y mostrar el balanceo de clases en los datasets de MedSyn.
"""
from __future__ import annotations
from typing import Dict, Any
from pathlib import Path
from collections import Counter
import json
import logging

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
        return analyze_class_balance(cfg.data.index_json)
    except ImportError:
        logger.error("No se pudo importar load_config. Asegúrate de que el módulo data.config existe.")
        raise
    except Exception as e:
        logger.error(f"Error al cargar configuración: {e}")
        raise


def analyze_class_balance(json_path: str | Path) -> Dict[str, Dict[int, int]]:
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


def show_class_balance(json_path: str | Path) -> None:
    """
    Función principal que analiza y muestra el balanceo de clases.
    
    Args:
        json_path: Ruta al archivo JSON de índices.
    """
    try:
        balance = analyze_class_balance(json_path)
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
        "--config",
        type=str,
        help="Ruta al archivo de configuración YAML (alternativa a --json)"
    )
    
    args = parser.parse_args()
    
    if args.config:
        balance = load_balance_from_config(args.config)
        print_class_balance_table(balance)
    elif args.json:
        show_class_balance(args.json)
    else:
        # Usar ruta por defecto del config
        json_file = "/home/mpascual/research/datasets/medsyn/PathMNIST/indexes/pathmnist_index.json"
        show_class_balance(json_file)
