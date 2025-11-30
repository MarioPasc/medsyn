#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_metrics.py

Resumen:
    Lee un CSV de entrenamiento/validación y calcula estadísticas agregadas
    (media y desviación típica) para todas las columnas numéricas en las que
    tenga sentido. Detecta splits por prefijos o sufijos comunes:
    {train, tr, training} y {val, valid, validation}.

Uso:
    python aggregate_metrics.py --csv /ruta/training_metrics_batch4.csv
    # Opcional: guardar resultados
    python aggregate_metrics.py --csv training.csv --out results_agg.json

Salida en consola (ejemplo):
    [TRAIN] SSIM = 0.548 ± 0.012  (n=40)
    [VAL]   PSNR = 21.59 ± 0.31 dB (n=40)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

# ----------------------------- Configuración ---------------------------------


@dataclass(frozen=True)
class SplitConfig:
    """Configuración de detección de splits en nombres de columnas."""
    train_aliases: Tuple[str, ...] = ("train", "tr", "training")
    val_aliases: Tuple[str, ...] = ("val", "valid", "validation")
    # patrones: prefijo_*, *_sufijo
    # Se aplican en minúsculas sobre el nombre normalizado de la columna.


class ColumnParsingError(Exception):
    """Error al parsear columnas y asignarlas a métricas/splits."""


# ----------------------------- Utilidades ------------------------------------


def setup_logging(verbosity: int) -> None:
    """Configura logging según nivel de verbosidad."""
    level = logging.WARNING if verbosity == 0 else logging.INFO if verbosity == 1 else logging.DEBUG
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def load_csv(csv_path: Path) -> pd.DataFrame:
    """Carga el CSV en un DataFrame."""
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe el fichero: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("El CSV está vacío.")
    return df


def normalize_name(name: str) -> str:
    """Normaliza nombres de columnas a minúsculas y sin espacios."""
    return re.sub(r"\s+", "_", name.strip().lower())


def detect_split_and_metric(
    col: str,
    cfg: SplitConfig,
) -> Optional[Tuple[str, str]]:
    """
    Dada una columna, intenta separar split y nombre de métrica.
    Devuelve (split, metric) si identifica un split, si no, None.

    Reglas:
      - Prefijo:  <split>_<metric>  (p.ej., train_ssim, val-psnr)
      - Sufijo:   <metric>_<split>  (p.ej., ssim_train, psnr.val)
      - Separadores admitidos: '_', '-', '.'
    """
    c = normalize_name(col)
    parts = re.split(r"[_\-.]", c)

    # Prefijo
    if parts and parts[0] in cfg.train_aliases + cfg.val_aliases:
        split = "train" if parts[0] in cfg.train_aliases else "val"
        metric = "_".join(parts[1:]).strip("_- .")
        if metric:
            return split, metric

    # Sufijo
    if parts and parts[-1] in cfg.train_aliases + cfg.val_aliases:
        split = "train" if parts[-1] in cfg.train_aliases else "val"
        metric = "_".join(parts[:-1]).strip("_- .")
        if metric:
            return split, metric

    return None


def is_numeric_series(s: pd.Series) -> bool:
    """Devuelve True si la serie es numérica y tiene al menos 1 valor no nulo."""
    return pd.api.types.is_numeric_dtype(s) and s.notna().sum() > 0


def summarize(series: pd.Series) -> Tuple[float, float, int]:
    """Devuelve (media, std, n) con ddof=1 si n>1, en float."""
    x = pd.to_numeric(series.dropna(), errors="coerce")
    x = x[~x.isna()]
    n = int(x.shape[0])
    if n == 0:
        return (float("nan"), float("nan"), 0)
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if n > 1 else float("nan")
    return mean, std, n


def unit_for_metric(metric_name: str) -> str:
    """
    Asigna unidades conocidas por métrica. PSNR -> dB; resto sin unidad.
    Se usa el nombre de métrica normalizado.
    """
    m = normalize_name(metric_name)
    if "psnr" in m:
        return " dB"
    return ""


def pretty(mu: float, sd: float) -> str:
    """Formatea 'mu ± sd' con precisión adaptativa."""
    if math.isnan(mu):
        return "nan ± nan"
    # Precisión: 3 decimales por defecto para valores típicos de métricas
    if math.isnan(sd):
        return f"{mu:.3f} ± nan"
    return f"{mu:.3f} ± {sd:.3f}"


def aggregate_metrics(
    df: pd.DataFrame,
    cfg: SplitConfig,
    include_global_numeric: bool = False,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Recorre columnas, detecta split y métrica, y agrega media/std/n por split.

    Devuelve un dict:
    {
      "train": { "ssim": {"mean":..., "std":..., "n":...}, "psnr": {...}, ... },
      "val":   { ... }
    }

    Si include_global_numeric=True, también agrega columnas numéricas sin split bajo "global".
    """
    results: Dict[str, Dict[str, Dict[str, float]]] = {"train": {}, "val": {}}
    global_bucket: Dict[str, Dict[str, float]] = {}

    for col in df.columns:
        parsed = detect_split_and_metric(col, cfg)
        s = df[col]
        if parsed is None:
            # opcional: columnas numéricas sin split explícito
            if include_global_numeric and is_numeric_series(s):
                mu, sd, n = summarize(s)
                global_bucket[normalize_name(col)] = {"mean": mu, "std": sd, "n": n}
            continue

        split, metric = parsed
        if not is_numeric_series(s):
            logging.debug("Se ignora columna no numérica: %s", col)
            continue

        mu, sd, n = summarize(s)
        results[split][metric] = {"mean": mu, "std": sd, "n": n}

    if include_global_numeric and global_bucket:
        results["global"] = global_bucket  # type: ignore[assignment]

    return results


def print_formatted(results: Mapping[str, Mapping[str, Mapping[str, float]]]) -> None:
    """Imprime resultados en el formato solicitado."""
    ordering = ("train", "val", "global")
    for split in ordering:
        if split not in results:
            continue
        # alineación visual
        tag = "[TRAIN]" if split == "train" else "[VAL]  " if split == "val" else "[ALL]  "
        for metric, stats in sorted(results[split].items()):
            mu = stats["mean"]
            sd = stats["std"]
            n = int(stats["n"])
            unit = unit_for_metric(metric)
            # Métricas habituales como PSNR/SSIM conservar mayúsculas estándar
            metric_label = metric.upper() if metric.lower() in {"psnr", "ssim"} else metric
            print(f"{tag} {metric_label} = {pretty(mu, sd)}{unit} (n={n})")


def save_results(
    results: Mapping[str, Mapping[str, Mapping[str, float]]],
    out_path: Optional[Path],
) -> None:
    """Guarda resultados como JSON si se solicita."""
    if out_path is None:
        return
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ------------------------------- CLI -----------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    """Construye el parser de argumentos para CLI."""
    p = argparse.ArgumentParser(description="Agrega métricas media±std por split y métrica.")
    p.add_argument("--csv", type=Path, required=True, help="Ruta al CSV de métricas.")
    p.add_argument(
        "--include-global",
        action="store_true",
        help="Incluir columnas numéricas sin split bajo el grupo 'global'.",
    )
    p.add_argument("--out", type=Path, default=None, help="Guardar agregados en JSON.")
    p.add_argument("-v", "--verbose", action="count", default=1, help="Aumentar verbosidad.")
    return p


def main() -> None:
    """Punto de entrada del script."""
    args = build_argparser().parse_args()
    setup_logging(args.verbose)

    cfg = SplitConfig()

    df = load_csv(args.csv)
    results = aggregate_metrics(df, cfg, include_global_numeric=bool(args.include_global))
    if not any(results.get(k) for k in ("train", "val", "global")):
        raise ColumnParsingError(
            "No se detectaron columnas numéricas con split. "
            "Asegura nombres como 'train_ssim', 'val_psnr', 'psnr_train', 'ssim.val', etc."
        )

    print_formatted(results)
    save_results(results, args.out)


if __name__ == "__main__":
    main()
