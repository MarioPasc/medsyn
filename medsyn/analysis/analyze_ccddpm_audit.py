# analyze_ccddpm_audit.py
# -*- coding: utf-8 -*-
"""
Procesa y visualiza resultados de audit_ccddpm_privacy.*:
- audit_summary.json
- memorization_suspects.csv
- mia_roc.csv
- audit_ccddpm_privacy.npz

Requisitos: python>=3.9, numpy, pandas, matplotlib, pillow (opcional)
Uso:
  python analyze_ccddpm_audit.py --root . --out figuras
"""

from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LOG = logging.getLogger("ccddpm.audit.analyze")

def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def safe_read_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        LOG.info(f"OK: leído JSON {p}")
        return data
    except Exception as e:
        LOG.error(f"ERROR al leer JSON {p}: {e}")
        return None

def safe_read_csv(p: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(p)
        LOG.info(f"OK: leído CSV {p} con {len(df):,} filas y {len(df.columns)} columnas")
        return df
    except Exception as e:
        LOG.error(f"ERROR al leer CSV {p}: {e}")
        return None

def safe_read_npz(p: Path) -> Optional[Dict[str, np.ndarray]]:
    try:
        data = np.load(p, allow_pickle=True)
        keys = list(data.keys())
        LOG.info(f"OK: leído NPZ {p} con claves: {keys}")
        return {k: data[k] for k in keys}
    except Exception as e:
        LOG.error(f"ERROR al leer NPZ {p}: {e}")
        return None

def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    LOG.info(f"Directorio de figuras: {outdir}")

# --------------------- Visualizaciones ---------------------

def plot_lpips_ssim_scatter(df: pd.DataFrame, outdir: Path, lpips_th: float, ssim_th: float) -> bool:
    req = {"lpips_train","ssim_train","lpips_test","ssim_test"}
    if not req.issubset(df.columns):
        LOG.warning("Faltan columnas para scatter LPIPS-SSIM. Omitiendo.")
        return False
    try:
        fig, ax = plt.subplots(figsize=(6,5))
        ax.scatter(df["lpips_train"], df["ssim_train"], s=6, alpha=0.5, label="NN train")
        ax.scatter(df["lpips_test"],  df["ssim_test"],  s=6, alpha=0.5, label="NN test")
        ax.axvline(lpips_th, linestyle="--", linewidth=1)
        ax.axhline(ssim_th,  linestyle="--", linewidth=1)
        ax.set_xlabel("LPIPS"); ax.set_ylabel("SSIM"); ax.set_title("LPIPS vs SSIM (train/test NN)")
        ax.legend(loc="lower left", fontsize=8)
        png = outdir / "scatter_lpips_ssim.png"
        pdf = outdir / "scatter_lpips_ssim.pdf"
        fig.tight_layout(); fig.savefig(png, dpi=200); fig.savefig(pdf)
        plt.close(fig)
        LOG.info(f"OK: scatter LPIPS-SSIM guardado en {png} y {pdf}")
        return True
    except Exception as e:
        LOG.error(f"ERROR al generar scatter LPIPS-SSIM: {e}")
        return False

def plot_nn_distance_hist(df: pd.DataFrame, outdir: Path) -> bool:
    req = {"nn_train_d","nn_test_d"}
    if not req.issubset(df.columns):
        LOG.warning("Faltan columnas nn_*_d para histogramas. Omitiendo.")
        return False
    try:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.hist(df["nn_train_d"], bins=50, alpha=0.6, label="NN train")
        ax.hist(df["nn_test_d"],  bins=50, alpha=0.6, label="NN test")
        ax.set_xlabel("Distancia de embedding (1 - cos_sim)"); ax.set_ylabel("Frecuencia")
        ax.set_title("Distribución de distancias a NN")
        ax.legend()
        png = outdir / "hist_nn_distance.png"
        fig.tight_layout(); fig.savefig(png, dpi=200); plt.close(fig)
        LOG.info(f"OK: histograma NN distances guardado en {png}")
        return True
    except Exception as e:
        LOG.error(f"ERROR al generar histograma NN distances: {e}")
        return False

def plot_suspects_by_class(df: pd.DataFrame, outdir: Path) -> bool:
    req = {"class","flag_suspect"}
    if not req.issubset(df.columns):
        LOG.warning("Faltan columnas 'class' o 'flag_suspect'. Omitiendo barras por clase.")
        return False
    try:
        counts = df.groupby("class")["flag_suspect"].sum()
        fig, ax = plt.subplots(figsize=(7,4))
        counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("Clase"); ax.set_ylabel("# sospechosos"); ax.set_title("Sospechosos por clase")
        png = outdir / "bar_suspects_by_class.png"
        fig.tight_layout(); fig.savefig(png, dpi=200); plt.close(fig)
        LOG.info(f"OK: barras sospechosos por clase guardado en {png}")
        return True
    except Exception as e:
        LOG.error(f"ERROR al generar barras por clase: {e}")
        return False

def plot_roc(roc_df: Optional[pd.DataFrame], npz: Optional[Dict[str,np.ndarray]], outdir: Path) -> bool:
    try:
        if roc_df is not None and {"fpr","tpr"}.issubset(roc_df.columns):
            fpr = roc_df["fpr"].to_numpy()
            tpr = roc_df["tpr"].to_numpy()
            auc = None
        elif npz is not None and {"mia_fpr","mia_tpr"}.issubset(npz.keys()):
            fpr = npz["mia_fpr"]; tpr = npz["mia_tpr"]
            auc = float(npz.get("mia_auc", [np.nan])[0]) if "mia_auc" in npz else None
        else:
            LOG.warning("No hay datos de ROC. Omitiendo ROC.")
            return False
        fig, ax = plt.subplots(figsize=(5,5))
        ax.plot(fpr, tpr, lw=2, label=f"ROC{'' if auc is None else f' (AUC={auc:.3f})'}")
        ax.plot([0,1], [0,1], "--", lw=1, color="gray")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("Membership Inference ROC")
        ax.legend(loc="lower right")
        png = outdir / "roc_mia.png"
        pdf = outdir / "roc_mia.pdf"
        fig.tight_layout(); fig.savefig(png, dpi=200); fig.savefig(pdf); plt.close(fig)
        LOG.info(f"OK: ROC guardado en {png} y {pdf}")
        return True
    except Exception as e:
        LOG.error(f"ERROR al generar ROC: {e}")
        return False

def plot_loss_hist(npz: Optional[Dict[str,np.ndarray]], outdir: Path) -> bool:
    if npz is None or not {"mia_loss_in","mia_loss_out"}.issubset(npz.keys()):
        LOG.warning("No hay pérdidas por muestra en NPZ. Omitiendo histogramas de pérdidas.")
        return False
    try:
        pin, pout = npz["mia_loss_in"].astype(float), npz["mia_loss_out"].astype(float)
        fig, ax = plt.subplots(figsize=(6,4))
        ax.hist(pin, bins=50, alpha=0.6, label="Miembros (train)")
        ax.hist(pout, bins=50, alpha=0.6, label="No-miembros (test)")
        ax.set_xlabel("Pérdida de denoising por muestra"); ax.set_ylabel("Frecuencia")
        ax.set_title("Distribución de pérdidas por muestra")
        ax.legend()
        png = outdir / "hist_mia_losses.png"
        fig.tight_layout(); fig.savefig(png, dpi=200); plt.close(fig)
        LOG.info(f"OK: hist pérdidas MIA guardado en {png}")
        return True
    except Exception as e:
        LOG.error(f"ERROR al generar hist pérdidas MIA: {e}")
        return False

# --------------------- Flujo principal ---------------------

def main(root: Path, outdir: Path) -> None:
    ensure_outdir(outdir)

    # Rutas
    p_summary = root / "audit_summary.json"
    p_suspects = root / "memorization_suspects.csv"
    p_roc = root / "mia_roc.csv"
    p_npz = root / "audit_ccddpm_privacy.npz"

    # Lecturas
    summary = safe_read_json(p_summary) if p_summary.exists() else None
    suspects_df = safe_read_csv(p_suspects) if p_suspects.exists() else None
    roc_df = safe_read_csv(p_roc) if p_roc.exists() else None
    npz = safe_read_npz(p_npz) if p_npz.exists() else None

    # Validaciones básicas
    if summary is not None:
        mem = summary.get("memorization", {})
        mia = summary.get("membership_inference", {})
        LOG.info(f"Resumen: sospechosos={mem.get('num_suspicious','?')}, rate={mem.get('rate','?')}, "
                 f"LPIPS_th={mem.get('lpips_threshold','?')}, SSIM_th={mem.get('ssim_threshold','?')}; "
                 f"MIA AUC={mia.get('auc','?')}")
        lp_th = float(mem.get("lpips_threshold", 0.12))
        ss_th = float(mem.get("ssim_threshold", 0.90))
    else:
        lp_th, ss_th = 0.12, 0.90
        LOG.warning("Sin audit_summary.json: usando umbrales por defecto LPIPS=0.12, SSIM=0.90.")

    # Visualizaciones
    ok_any = False
    if suspects_df is not None:
        ok_any |= plot_lpips_ssim_scatter(suspects_df, outdir, lp_th, ss_th)
        ok_any |= plot_nn_distance_hist(suspects_df, outdir)
        ok_any |= plot_suspects_by_class(suspects_df, outdir)
    else:
        LOG.warning("No se generarán gráficos de memorization al faltar memorization_suspects.csv.")

    ok_any |= plot_roc(roc_df, npz, outdir)
    ok_any |= plot_loss_hist(npz, outdir)

    if ok_any:
        LOG.info("Éxito: al menos una visualización generada correctamente. Proceda al análisis.")
    else:
        LOG.error("No se generó ninguna visualización. Revise rutas y formatos de entrada.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."), help="Directorio donde están los ficheros de auditoría")
    parser.add_argument("--out", type=Path, default=Path("figuras"), help="Directorio de salida de figuras")
    parser.add_argument("--log", type=str, default="INFO", help="Nivel de log: DEBUG, INFO, WARNING, ERROR")
    args = parser.parse_args()
    setup_logging(args.log)
    main(args.root, args.out)
