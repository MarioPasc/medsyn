# analyze_ccddpm_audit.py
# -*- coding: utf-8 -*-
"""
Genera visualizaciones publication-ready de auditoría de privacidad de ccDDPM.

Lee la configuración desde config/audit_ccddpm_privacy.yaml y genera:
1. Panel 1x4 (train vs test): [NN Distance Hist] [MIA Loss Hist] [LPIPS-SSIM Scatter] [ROC MIA]
2. Visualización 2x3 de muestras sospechosas usando idx_synth y nn_train_idx

Archivos procesados:
- config/audit_ccddpm_privacy.yaml: Configuración principal (ubicaciones de datos)
- audit_summary.json: Resumen de resultados
- memorization_suspects.csv: Muestras sospechosas con índices
- mia_roc.csv: Curva ROC de MIA
- audit_ccddpm_privacy.npz: Datos de auditoría (métricas, distancias NN, etc.)
- merged.npz: Imágenes (train/val, reales+sintéticas) según config (data->postprocess_npz->npz_path)

Requisitos:
  python>=3.9, numpy, pandas, matplotlib, scienceplots (opcional), pyyaml, sklearn

Uso:
  # Usar directorio de visualizaciones del config (io->outdir_visualizations)
  python medsyn/utils/analyze_ccddpm_audit.py --config config/audit_ccddpm_privacy.yaml

  # O especificar directorio de salida manualmente
  python medsyn/utils/analyze_ccddpm_audit.py --config config/audit_ccddpm_privacy.yaml --out figuras
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
from matplotlib.gridspec import GridSpec
from sklearn.metrics import auc as compute_auc
import yaml

# Configuración publication-ready con scienceplots
try:
    plt.style.use(['science', 'ieee'])
    SCIENCEPLOTS_AVAILABLE = True
except:
    SCIENCEPLOTS_AVAILABLE = False
    LOG = logging.getLogger("ccddpm.audit.analyze")
    LOG.warning("scienceplots no disponible. Usando estilo matplotlib por defecto.")

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

def safe_read_yaml(p: Path) -> Optional[Dict[str, Any]]:
    """Lee archivo YAML de configuración."""
    try:
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        LOG.info(f"OK: leído YAML {p}")
        return data
    except Exception as e:
        LOG.error(f"ERROR al leer YAML {p}: {e}")
        return None

def load_images_from_config(config_yaml: Path) -> Optional[Dict[str, Any]]:
    """
    Carga imágenes desde el NPZ postprocesado y el NPZ de auditoría.

    Proceso:
    1. Lee config_yaml (audit_ccddpm_privacy.yaml)
    2. Extrae io->yaml para obtener la ruta del YAML principal
    3. Lee ese YAML y extrae data->postprocess_npz->npz_path
    4. Carga el NPZ postprocesado (merged.npz) con train_images, val_images/test_images, etc.
    5. Extrae las imágenes sintéticas usando la máscara train_is_synth

    Nota: Detecta automáticamente si usar 'test_images' o 'val_images' como conjunto de evaluación.
    Las imágenes sintéticas se extraen del conjunto de entrenamiento usando train_is_synth.

    Returns:
        Dict con claves:
        - 'synth_images': imágenes sintéticas extraídas del train (usando train_is_synth)
        - 'train_images': todas las imágenes de train (reales + sintéticas del NPZ postprocesado)
        - 'test_images': todas las imágenes de test/val (del NPZ postprocesado)
        - 'train_is_synth': máscara booleana indicando cuáles del train son sintéticas
        - 'test_is_synth': máscara booleana indicando cuáles del test/val son sintéticas
        - 'test_split_name': 'test' o 'val' (indica qué split se usó)
        - 'data_npz_path': ruta al NPZ postprocesado
        - 'audit_dir': directorio de auditoría
    """
    try:
        # 1. Leer configuración de auditoría
        LOG.info(f"[1/5] Leyendo configuración de auditoría desde {config_yaml}...")
        audit_cfg = safe_read_yaml(config_yaml)
        if audit_cfg is None:
            return None

        # 2. Extraer io->yaml
        io_cfg = audit_cfg.get("io", {})
        main_yaml_path = Path(io_cfg.get("yaml", ""))
        audit_dir = Path(io_cfg.get("outdir", ""))

        if not main_yaml_path.exists():
            LOG.error(f"Archivo de configuración principal no encontrado: {main_yaml_path}")
            return None

        LOG.info(f"[2/5] Leyendo configuración principal desde {main_yaml_path}...")

        # 3. Leer configuración principal y extraer data->postprocess_npz->npz_path
        main_cfg = safe_read_yaml(main_yaml_path)
        if main_cfg is None:
            return None

        data_cfg = main_cfg.get("data", {})
        postprocess_npz_cfg = data_cfg.get("postprocess_npz", {})
        npz_path = Path(postprocess_npz_cfg.get("npz_path", ""))

        if not npz_path.exists():
            LOG.error(f"NPZ postprocesado no encontrado: {npz_path}")
            LOG.error("Asegúrese de que data->postprocess_npz->npz_path esté configurado correctamente.")
            return None

        # 4. Cargar NPZ postprocesado (merged.npz)
        LOG.info(f"[3/5] Cargando NPZ postprocesado desde {npz_path}...")
        data_npz = np.load(npz_path, allow_pickle=True)

        # Extraer imágenes y máscaras por split
        # Estructura flexible: puede ser train/test o train/val
        available_keys = list(data_npz.keys())
        LOG.info(f"  Claves disponibles en NPZ: {available_keys}")

        train_images = data_npz.get("train_images", None)
        train_is_synth = data_npz.get("train_is_synth", None)

        # Detectar automáticamente si usar test o val como conjunto de evaluación
        if "test_images" in data_npz:
            test_images = data_npz.get("test_images")
            test_is_synth = data_npz.get("test_is_synth", None)
            test_split_name = "test"
            LOG.info(f"  ✓ Usando 'test' como conjunto de evaluación")
        elif "val_images" in data_npz:
            test_images = data_npz.get("val_images")
            test_is_synth = data_npz.get("val_is_synth", None)
            test_split_name = "val"
            LOG.info(f"  ✓ Usando 'val' como conjunto de evaluación")
        else:
            test_images = None
            test_is_synth = None
            test_split_name = None

        if train_images is None:
            LOG.error(f"No se encontró 'train_images' en {npz_path}")
            return None

        if test_images is None:
            LOG.error(f"No se encontró 'test_images' ni 'val_images' en {npz_path}")
            return None

        LOG.info(f"  ✓ Train images: {train_images.shape}")
        LOG.info(f"  ✓ {test_split_name.capitalize()} images: {test_images.shape}")
        if train_is_synth is not None:
            LOG.info(f"  ✓ Train is_synth: {train_is_synth.shape} (sintéticas: {train_is_synth.sum()})")
        if test_is_synth is not None:
            LOG.info(f"  ✓ {test_split_name.capitalize()} is_synth: {test_is_synth.shape} (sintéticas: {test_is_synth.sum()})")

        # 5. Extraer imágenes sintéticas del NPZ postprocesado
        LOG.info(f"[4/5] Extrayendo imágenes sintéticas del NPZ postprocesado...")
        synth_images = None

        if train_is_synth is not None and train_is_synth.sum() > 0:
            # Extraer solo las imágenes sintéticas usando la máscara
            synth_images = train_images[train_is_synth]
            LOG.info(f"  ✓ Imágenes sintéticas extraídas: {synth_images.shape}")
            LOG.info(f"  ✓ Total sintéticas en train: {train_is_synth.sum()}")
        else:
            LOG.warning(f"No se encontraron imágenes sintéticas en el conjunto de entrenamiento")
            LOG.warning(f"  train_is_synth: {train_is_synth is not None}, suma: {train_is_synth.sum() if train_is_synth is not None else 0}")

        # Construir resultado
        result = {
            "synth_images": synth_images,
            "train_images": train_images,
            "test_images": test_images,
            "train_is_synth": train_is_synth,
            "test_is_synth": test_is_synth,
            "test_split_name": test_split_name,  # 'test' o 'val'
            "data_npz_path": npz_path,
            "audit_dir": audit_dir
        }

        LOG.info(f"[5/5] ✓ Carga de imágenes completada correctamente")
        if synth_images is not None:
            LOG.info(f"       Resumen: {synth_images.shape[0]} sintéticas, {train_images.shape[0]} train total, {test_images.shape[0]} {test_split_name} total")
        return result

    except Exception as e:
        LOG.error(f"ERROR al cargar imágenes desde configuración: {e}")
        import traceback
        traceback.print_exc()
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

def plot_publication_panel(
    df: Optional[pd.DataFrame],
    roc_df: Optional[pd.DataFrame],
    npz: Optional[Dict[str, np.ndarray]],
    outdir: Path,
    lpips_th: float = 0.12,
    ssim_th: float = 0.90
) -> bool:
    """
    Genera panel publication-ready de 1 fila x 4 columnas:
    - [0] Histograma de distancias NN (train vs test)
    - [1] Histograma de MIA losses (train vs test)
    - [2] Scatter LPIPS vs SSIM (train vs test)
    - [3] ROC MIA

    Train y test se comparan en el mismo plot usando los mismos colores consistentes.
    """
    try:
        # Verificar datos disponibles
        has_nn_dist = df is not None and {"nn_train_d", "nn_test_d"}.issubset(df.columns)
        has_suspects = df is not None and {"lpips_train", "ssim_train", "lpips_test", "ssim_test"}.issubset(df.columns)
        has_mia = npz is not None and {"mia_loss_in", "mia_loss_out"}.issubset(npz.keys())
        has_roc = (roc_df is not None and {"fpr", "tpr"}.issubset(roc_df.columns)) or \
                  (npz is not None and {"mia_fpr", "mia_tpr"}.issubset(npz.keys()))

        if not (has_suspects or has_mia or has_roc or has_nn_dist):
            LOG.warning("No hay suficientes datos para generar el panel publication-ready.")
            return False

        # Colores consistentes para train y test
        COLOR_TRAIN = '#2E86AB'  # Azul
        COLOR_TEST = '#F18F01'   # Naranja

        # Crear figura con GridSpec 1x4
        fig = plt.figure(figsize=(20, 5))
        gs = GridSpec(1, 4, figure=fig, hspace=0.25, wspace=0.35)

        # ==================== [0] Histograma NN Distance ====================
        ax_nn_dist = fig.add_subplot(gs[0, 0])
        if has_nn_dist:
            ax_nn_dist.hist(df["nn_train_d"], bins=40, alpha=0.65, color=COLOR_TRAIN,
                           edgecolor='black', linewidth=0.5, label='Train')
            ax_nn_dist.hist(df["nn_test_d"], bins=40, alpha=0.65, color=COLOR_TEST,
                           edgecolor='black', linewidth=0.5, label='Test')
            ax_nn_dist.set_xlabel('NN Distance (1 - cosine similarity)')
            ax_nn_dist.set_ylabel('Frequency')
            ax_nn_dist.set_title('Nearest Neighbor Distance Distribution')
            # Leyenda fuera del plot, debajo del xlabel
            ax_nn_dist.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                            ncol=2, fontsize=9, framealpha=0.9)
            ax_nn_dist.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        else:
            ax_nn_dist.text(0.5, 0.5, 'No NN distance data', ha='center', va='center',
                           transform=ax_nn_dist.transAxes)
            ax_nn_dist.set_title('Nearest Neighbor Distance Distribution')

        # ==================== [1] Histograma MIA Losses ====================
        ax_mia_hist = fig.add_subplot(gs[0, 1])
        if has_mia:
            mia_loss_train = npz["mia_loss_in"].astype(float)
            mia_loss_test = npz["mia_loss_out"].astype(float)

            ax_mia_hist.hist(mia_loss_train, bins=40, alpha=0.65, color=COLOR_TRAIN,
                            edgecolor='black', linewidth=0.5, label='Train (Members)')
            ax_mia_hist.hist(mia_loss_test, bins=40, alpha=0.65, color=COLOR_TEST,
                            edgecolor='black', linewidth=0.5, label='Test (Non-members)')
            ax_mia_hist.set_xlabel('MIA Loss (per sample)')
            ax_mia_hist.set_ylabel('Frequency')
            ax_mia_hist.set_title('MIA Loss Distribution')
            # Leyenda fuera del plot, debajo del xlabel
            ax_mia_hist.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                             ncol=2, fontsize=9, framealpha=0.9)
            ax_mia_hist.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        else:
            ax_mia_hist.text(0.5, 0.5, 'No MIA data', ha='center', va='center',
                            transform=ax_mia_hist.transAxes)
            ax_mia_hist.set_title('MIA Loss Distribution')

        # ==================== [2] Scatter LPIPS vs SSIM ====================
        ax_scatter = fig.add_subplot(gs[0, 2])
        if has_suspects:
            ax_scatter.scatter(df["lpips_train"], df["ssim_train"],
                             s=15, alpha=0.6, color=COLOR_TRAIN, edgecolors='none', label='Train')
            ax_scatter.scatter(df["lpips_test"], df["ssim_test"],
                             s=15, alpha=0.6, color=COLOR_TEST, edgecolors='none', label='Test')

            # Líneas de umbral
            ax_scatter.axvline(lpips_th, linestyle='--', linewidth=1.5, color='#E63946',
                             label=f'LPIPS th={lpips_th}')
            ax_scatter.axhline(ssim_th, linestyle='--', linewidth=1.5, color='#6A4C93',
                             label=f'SSIM th={ssim_th}')

            ax_scatter.set_xlabel('LPIPS')
            ax_scatter.set_ylabel('SSIM')
            ax_scatter.set_title('LPIPS vs SSIM')
            # Leyenda fuera del plot, debajo del xlabel
            ax_scatter.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                            ncol=2, fontsize=8, framealpha=0.9)
            ax_scatter.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        else:
            ax_scatter.text(0.5, 0.5, 'No suspects data', ha='center', va='center',
                           transform=ax_scatter.transAxes)
            ax_scatter.set_title('LPIPS vs SSIM')

        # ==================== [3] ROC MIA ====================
        ax_roc = fig.add_subplot(gs[0, 3])
        if has_roc:
            if roc_df is not None and {"fpr", "tpr"}.issubset(roc_df.columns):
                fpr = roc_df["fpr"].to_numpy()
                tpr = roc_df["tpr"].to_numpy()
                auc_val = compute_auc(fpr, tpr)
            elif npz is not None and {"mia_fpr", "mia_tpr"}.issubset(npz.keys()):
                fpr = npz["mia_fpr"]
                tpr = npz["mia_tpr"]
                auc_val = float(npz.get("mia_auc", [np.nan])[0]) if "mia_auc" in npz else compute_auc(fpr, tpr)

            ax_roc.plot(fpr, tpr, linewidth=2.5, color='#06A77D', label=f'MIA (AUC={auc_val:.3f})')
            ax_roc.plot([0, 1], [0, 1], '--', linewidth=1.5, color='gray', alpha=0.5, label='Random')
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('Membership Inference Attack ROC')
            # Leyenda fuera del plot, debajo del xlabel
            ax_roc.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                        ncol=2, fontsize=9, framealpha=0.9)
            ax_roc.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax_roc.set_xlim([0, 1])
            ax_roc.set_ylim([0, 1])
        else:
            ax_roc.text(0.5, 0.5, 'No ROC data', ha='center', va='center',
                       transform=ax_roc.transAxes)
            ax_roc.set_title('Membership Inference Attack ROC')

        # Guardar figura (bbox_inches='tight' ajusta automáticamente para incluir las leyendas)
        png = outdir / "publication_panel_1x4.png"
        pdf = outdir / "publication_panel_1x4.pdf"
        plt.tight_layout()  # Ajustar layout para evitar superposición
        fig.savefig(png, dpi=300, bbox_inches='tight')
        fig.savefig(pdf, bbox_inches='tight')
        plt.close(fig)

        LOG.info(f"OK: Panel publication-ready 1x4 guardado en {png} y {pdf}")
        return True

    except Exception as e:
        LOG.error(f"ERROR al generar panel publication-ready: {e}")
        import traceback
        traceback.print_exc()
        return False

def plot_suspect_samples(
    df: Optional[pd.DataFrame],
    npz: Optional[Dict[str, np.ndarray]],
    outdir: Path,
    n_suspects: int = 3
) -> bool:
    """
    Genera visualización de muestras sospechosas en panel 2x3:
    - Fila 1: n_suspects muestras sospechosas de train
    - Fila 2: n_suspects muestras sospechosas de test

    Cada subgráfico muestra la imagen original y su reconstrucción más cercana,
    junto con métricas LPIPS y SSIM.
    """
    try:
        if df is None or not {"flag_suspect"}.issubset(df.columns):
            LOG.warning("No hay datos de sospechosos para visualizar muestras.")
            return False

        # Filtrar sospechosos
        suspects = df[df["flag_suspect"] == True].copy()
        if len(suspects) == 0:
            LOG.warning("No hay muestras marcadas como sospechosas.")
            return False

        # Separar por conjunto (si hay columna que lo indique, sino usar NN)
        # Asumimos que hay información sobre train/test en las columnas nn_train_* y nn_test_*
        has_train_test_sep = {"nn_train_d", "nn_test_d"}.issubset(df.columns)

        if has_train_test_sep:
            # Encontrar top sospechosos para train (menor distancia train, alta SSIM train, bajo LPIPS train)
            suspects_train = suspects.copy()
            suspects_train["suspect_score_train"] = suspects_train["ssim_train"] - suspects_train["lpips_train"]
            suspects_train = suspects_train.nlargest(n_suspects, "suspect_score_train")

            # Encontrar top sospechosos para test
            suspects_test = suspects.copy()
            suspects_test["suspect_score_test"] = suspects_test["ssim_test"] - suspects_test["lpips_test"]
            suspects_test = suspects_test.nlargest(n_suspects, "suspect_score_test")
        else:
            # Si no hay separación, tomar los top n_suspects en general
            suspects["suspect_score"] = suspects.get("ssim_train", 0) - suspects.get("lpips_train", 1)
            suspects_sorted = suspects.nlargest(min(n_suspects * 2, len(suspects)), "suspect_score")
            suspects_train = suspects_sorted.iloc[:n_suspects]
            suspects_test = suspects_sorted.iloc[n_suspects:n_suspects*2] if len(suspects_sorted) > n_suspects else suspects_sorted.iloc[:n_suspects]

        # Crear figura
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(2, n_suspects, figure=fig, hspace=0.35, wspace=0.3)

        # ==================== FILA 1: TRAIN SUSPECTS ====================
        for i, (idx, row) in enumerate(suspects_train.iterrows()):
            if i >= n_suspects:
                break

            ax = fig.add_subplot(gs[0, i])

            # Crear visualización de información de la muestra
            info_text = f"Sample {row.get('sample_id', idx)}\n"
            info_text += f"Class: {row.get('class', 'N/A')}\n\n"
            info_text += f"Train NN Metrics:\n"
            info_text += f"  LPIPS: {row.get('lpips_train', 0):.4f}\n"
            info_text += f"  SSIM: {row.get('ssim_train', 0):.4f}\n"
            info_text += f"  Distance: {row.get('nn_train_d', 0):.4f}\n"

            # Si hay imágenes en el NPZ, las mostraríamos aquí
            # Por ahora, mostrar información textual
            ax.text(0.5, 0.5, info_text, ha='center', va='center',
                   transform=ax.transAxes, fontsize=9, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax.set_title(f'Train Suspect #{i+1}', fontsize=10, fontweight='bold')
            ax.axis('off')

        # ==================== FILA 2: TEST SUSPECTS ====================
        for i, (idx, row) in enumerate(suspects_test.iterrows()):
            if i >= n_suspects:
                break

            ax = fig.add_subplot(gs[1, i])

            # Crear visualización de información de la muestra
            info_text = f"Sample {row.get('sample_id', idx)}\n"
            info_text += f"Class: {row.get('class', 'N/A')}\n\n"
            info_text += f"Test NN Metrics:\n"
            info_text += f"  LPIPS: {row.get('lpips_test', 0):.4f}\n"
            info_text += f"  SSIM: {row.get('ssim_test', 0):.4f}\n"
            info_text += f"  Distance: {row.get('nn_test_d', 0):.4f}\n"

            ax.text(0.5, 0.5, info_text, ha='center', va='center',
                   transform=ax.transAxes, fontsize=9, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            ax.set_title(f'Test Suspect #{i+1}', fontsize=10, fontweight='bold')
            ax.axis('off')

        # Guardar figura
        png = outdir / "suspect_samples_comparison.png"
        pdf = outdir / "suspect_samples_comparison.pdf"
        fig.savefig(png, dpi=300, bbox_inches='tight')
        fig.savefig(pdf, bbox_inches='tight')
        plt.close(fig)

        LOG.info(f"OK: Comparación de muestras sospechosas guardada en {png} y {pdf}")
        return True

    except Exception as e:
        LOG.error(f"ERROR al generar comparación de muestras sospechosas: {e}")
        import traceback
        traceback.print_exc()
        return False

def plot_suspect_samples_with_images(
    df: Optional[pd.DataFrame],
    images_dict: Optional[Dict[str, Any]],
    outdir: Path,
    n_suspects: int = 3
) -> bool:
    """
    Visualiza muestras sospechosas usando idx_synth y nn_train_idx.
    Panel 2x3: cada celda muestra la imagen generada y su vecino más cercano lado a lado.

    Args:
        df: DataFrame con columnas idx_synth, nn_train_idx, lpips_train, ssim_train, etc.
        images_dict: Dict con 'synth_images', 'train_images', 'test_images'
        outdir: Directorio de salida
        n_suspects: Número de sospechosos a visualizar
    """
    try:
        if df is None or not {"flag_suspect"}.issubset(df.columns):
            LOG.warning("No hay datos de sospechosos para visualizar muestras con imágenes.")
            return False

        if images_dict is None:
            LOG.warning("No se proporcionaron imágenes.")
            return plot_suspect_samples(df, None, outdir, n_suspects)

        suspects = df[df["flag_suspect"] == True].copy()
        if len(suspects) == 0:
            LOG.warning("No hay muestras marcadas como sospechosas.")
            return False

        # Extraer imágenes del dict
        synth_images = images_dict.get("synth_images", None)
        train_images = images_dict.get("train_images", None)
        test_images = images_dict.get("test_images", None)

        if synth_images is None or train_images is None:
            LOG.warning("Faltan imágenes sintéticas o de entrenamiento.")
            return plot_suspect_samples(df, None, outdir, n_suspects)

        # Verificar que tenemos las columnas necesarias
        required_cols = {"idx_synth", "nn_train_idx", "lpips_train", "ssim_train", "nn_train_d"}
        if not required_cols.issubset(df.columns):
            LOG.warning(f"Faltan columnas requeridas. Necesarias: {required_cols}, Disponibles: {df.columns.tolist()}")
            return False

        # Separar sospechosos train/test
        has_test_cols = {"nn_test_idx", "lpips_test", "ssim_test", "nn_test_d"}.issubset(df.columns)

        # Top sospechosos de train (alto SSIM, bajo LPIPS)
        suspects_train = suspects.copy()
        suspects_train["suspect_score_train"] = suspects_train["ssim_train"] - suspects_train["lpips_train"]
        suspects_train = suspects_train.nlargest(n_suspects, "suspect_score_train")

        # Top sospechosos de test
        if has_test_cols and test_images is not None:
            suspects_test = suspects.copy()
            suspects_test["suspect_score_test"] = suspects_test["ssim_test"] - suspects_test["lpips_test"]
            suspects_test = suspects_test.nlargest(n_suspects, "suspect_score_test")
        else:
            # Si no hay test, usar los siguientes sospechosos de train
            all_sorted = suspects.copy()
            all_sorted["suspect_score_train"] = all_sorted["ssim_train"] - all_sorted["lpips_train"]
            all_sorted = all_sorted.nlargest(n_suspects * 2, "suspect_score_train")
            suspects_test = all_sorted.iloc[n_suspects:n_suspects*2] if len(all_sorted) > n_suspects else all_sorted.iloc[:n_suspects]

        # Crear figura con subgrid para cada par de imágenes
        # Cada celda contendrá 2 imágenes (generada | vecina)
        fig = plt.figure(figsize=(6 * n_suspects, 10))
        # GridSpec: 2 filas, n_suspects*2 columnas (cada par ocupa 2 columnas adyacentes)
        gs = GridSpec(2, n_suspects * 2, figure=fig, hspace=0.4, wspace=0.1)

        # Función auxiliar para normalizar y mostrar imagen
        def normalize_image(img):
            """Normaliza imagen para visualización."""
            if img is None:
                return None

            img = img.copy()
            # Si está en rango [-1, 1], llevar a [0, 1]
            if img.min() < 0:
                img = (img + 1) / 2

            # Si está en [0, 1], llevar a [0, 255]
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

            # Transponer si es CHW -> HWC
            if len(img.shape) == 3 and img.shape[0] in [1, 3]:
                img = np.transpose(img, (1, 2, 0))

            # Squeeze si es grayscale con canal
            if len(img.shape) == 3 and img.shape[-1] == 1:
                img = img.squeeze(-1)

            return img

        def show_image_pair(row, col_pair, gen_img, nn_img, title, lpips, ssim, distance):
            """Muestra par de imágenes lado a lado."""
            # col_pair es el índice del par (0, 1, 2, ...)
            # Cada par ocupa 2 columnas: col_pair*2 y col_pair*2+1
            ax_gen = fig.add_subplot(gs[row, col_pair * 2])
            ax_nn = fig.add_subplot(gs[row, col_pair * 2 + 1])

            # Mostrar imagen generada
            gen_img_show = normalize_image(gen_img)
            if gen_img_show is not None:
                cmap = 'gray' if len(gen_img_show.shape) == 2 else None
                ax_gen.imshow(gen_img_show, cmap=cmap)
                ax_gen.set_title('Generated', fontsize=9)
            else:
                ax_gen.text(0.5, 0.5, 'No image', ha='center', va='center', transform=ax_gen.transAxes)
            ax_gen.axis('off')

            # Mostrar imagen vecina
            nn_img_show = normalize_image(nn_img)
            if nn_img_show is not None:
                cmap = 'gray' if len(nn_img_show.shape) == 2 else None
                ax_nn.imshow(nn_img_show, cmap=cmap)
                ax_nn.set_title('Nearest Neighbor', fontsize=9)
            else:
                ax_nn.text(0.5, 0.5, 'No image', ha='center', va='center', transform=ax_nn.transAxes)
            ax_nn.axis('off')

            # Añadir métricas como suptitle del par
            # Crear un título sobre ambas imágenes
            metrics_text = f"{title}\nLPIPS: {lpips:.4f} | SSIM: {ssim:.4f} | NN Dist: {distance:.4f}"
            # Usar fig.text para colocar el título sobre el par de imágenes
            # Calcular posición X centrada sobre el par
            x_center = (col_pair * 2 + 1) / (n_suspects * 2)
            # Posiciones Y ajustadas para hspace=0.4
            y_pos = 0.96 if row == 0 else 0.48
            fig.text(x_center, y_pos, metrics_text, ha='center', va='top',
                    fontsize=10, fontweight='bold', transform=fig.transFigure)

        # ==================== FILA 1: TRAIN SUSPECTS ====================
        for i, (idx, suspect_row) in enumerate(suspects_train.iterrows()):
            if i >= n_suspects:
                break

            # Usar idx_synth para imagen generada
            idx_synth = int(suspect_row.get('idx_synth', -1))
            gen_img = synth_images[idx_synth] if idx_synth >= 0 and idx_synth < len(synth_images) else None

            # Usar nn_train_idx para imagen vecina
            nn_train_idx = int(suspect_row.get('nn_train_idx', -1))
            nn_img = train_images[nn_train_idx] if nn_train_idx >= 0 and nn_train_idx < len(train_images) else None

            show_image_pair(
                0, i, gen_img, nn_img, f"Train Suspect #{i+1}",
                suspect_row.get('lpips_train', 0), suspect_row.get('ssim_train', 0), suspect_row.get('nn_train_d', 0)
            )

        # ==================== FILA 2: TEST SUSPECTS ====================
        for i, (idx, suspect_row) in enumerate(suspects_test.iterrows()):
            if i >= n_suspects:
                break

            # Usar idx_synth para imagen generada
            idx_synth = int(suspect_row.get('idx_synth', -1))
            gen_img = synth_images[idx_synth] if idx_synth >= 0 and idx_synth < len(synth_images) else None

            # Usar nn_test_idx si existe, sino nn_train_idx
            if has_test_cols and test_images is not None:
                nn_test_idx = int(suspect_row.get('nn_test_idx', -1))
                nn_img = test_images[nn_test_idx] if nn_test_idx >= 0 and nn_test_idx < len(test_images) else None
                lpips = suspect_row.get('lpips_test', 0)
                ssim = suspect_row.get('ssim_test', 0)
                distance = suspect_row.get('nn_test_d', 0)
                title = f"Test Suspect #{i+1}"
            else:
                nn_train_idx = int(suspect_row.get('nn_train_idx', -1))
                nn_img = train_images[nn_train_idx] if nn_train_idx >= 0 and nn_train_idx < len(train_images) else None
                lpips = suspect_row.get('lpips_train', 0)
                ssim = suspect_row.get('ssim_train', 0)
                distance = suspect_row.get('nn_train_d', 0)
                title = f"Suspect #{i+1+n_suspects}"

            show_image_pair(1, i, gen_img, nn_img, title, lpips, ssim, distance)

        # Guardar
        png = outdir / "suspect_samples_images.png"
        pdf = outdir / "suspect_samples_images.pdf"
        fig.savefig(png, dpi=300, bbox_inches='tight')
        fig.savefig(pdf, bbox_inches='tight')
        plt.close(fig)

        LOG.info(f"OK: Muestras sospechosas con imágenes guardadas en {png} y {pdf}")
        return True

    except Exception as e:
        LOG.error(f"ERROR al generar visualización de muestras sospechosas con imágenes: {e}")
        import traceback
        traceback.print_exc()
        # Fallback a versión sin imágenes
        return plot_suspect_samples(df, None, outdir, n_suspects)

# --------------------- Flujo principal ---------------------

def main(config_yaml: Path, outdir: Optional[Path] = None) -> None:
    """
    Función principal que genera visualizaciones publication-ready.

    Args:
        config_yaml: Ruta al archivo de configuración audit_ccddpm_privacy.yaml
        outdir: Directorio de salida para las figuras (opcional, usa io->outdir_visualizations del config si no se especifica)
    """
    # Leer configuración de auditoría
    LOG.info(f"Leyendo configuración desde {config_yaml}...")
    audit_cfg = safe_read_yaml(config_yaml)
    if audit_cfg is None:
        LOG.error("No se pudo leer la configuración de auditoría. Abortando.")
        return

    # Obtener directorios desde configuración
    io_cfg = audit_cfg.get("io", {})
    audit_dir = Path(io_cfg.get("outdir", ""))

    # Usar outdir_visualizations del config si está disponible, sino el argumento outdir
    config_viz_dir = io_cfg.get("outdir_visualizations", None)
    if config_viz_dir is not None:
        outdir = Path(config_viz_dir)
        LOG.info(f"✓ Usando directorio de visualizaciones desde config: {outdir}")
    elif outdir is None:
        # Si no hay outdir_visualizations en el config ni argumento --out, usar audit_dir/visualizations
        outdir = audit_dir / "visualizations"
        LOG.warning(f"No se especificó directorio de salida. Usando por defecto: {outdir}")
    else:
        LOG.info(f"✓ Usando directorio de visualizaciones desde argumento --out: {outdir}")

    ensure_outdir(outdir)

    if not audit_dir.exists():
        LOG.error(f"Directorio de auditoría no encontrado: {audit_dir}")
        return

    LOG.info(f"Directorio de auditoría: {audit_dir}")

    # Cargar imágenes desde configuración
    LOG.info("Cargando imágenes desde configuración...")
    images_dict = load_images_from_config(config_yaml)

    # Rutas de archivos de auditoría
    p_summary = audit_dir / "audit_summary.json"
    p_suspects = audit_dir / "memorization_suspects.csv"
    p_roc = audit_dir / "mia_roc.csv"
    p_npz = audit_dir / "audit_ccddpm_privacy.npz"

    # Lecturas
    LOG.info("Leyendo archivos de auditoría...")
    summary = safe_read_json(p_summary) if p_summary.exists() else None
    suspects_df = safe_read_csv(p_suspects) if p_suspects.exists() else None
    roc_df = safe_read_csv(p_roc) if p_roc.exists() else None
    npz = safe_read_npz(p_npz) if p_npz.exists() else None

    # Obtener umbrales desde configuración o summary
    nearest_neighbor_cfg = audit_cfg.get("nearest_neighbor", {})
    lp_th = float(nearest_neighbor_cfg.get("lpips_threshold", 0.12))
    ss_th = float(nearest_neighbor_cfg.get("ssim_threshold", 0.90))

    # Validaciones básicas y logging
    if summary is not None:
        mem = summary.get("memorization", {})
        mia = summary.get("membership_inference", {})
        LOG.info("=" * 80)
        LOG.info("RESUMEN DE AUDITORÍA:")
        LOG.info(f"  Sospechosos: {mem.get('num_suspicious','?')}")
        LOG.info(f"  Tasa de sospechosos: {mem.get('rate','?')}")
        LOG.info(f"  LPIPS threshold: {mem.get('lpips_threshold','?')}")
        LOG.info(f"  SSIM threshold: {mem.get('ssim_threshold','?')}")
        LOG.info(f"  MIA AUC: {mia.get('auc','?')}")
        LOG.info("=" * 80)

        # Actualizar umbrales si están en el summary
        lp_th = float(mem.get("lpips_threshold", lp_th))
        ss_th = float(mem.get("ssim_threshold", ss_th))
    else:
        LOG.warning("Sin audit_summary.json: usando umbrales de configuración YAML.")

    # Visualizaciones publication-ready
    ok_any = False

    # Panel principal 1x4 (publication-ready)
    LOG.info("=" * 80)
    LOG.info("Generando panel publication-ready 1x4...")
    ok_any |= plot_publication_panel(suspects_df, roc_df, npz, outdir, lp_th, ss_th)

    # Visualización de muestras sospechosas (publication-ready)
    if suspects_df is not None and images_dict is not None:
        LOG.info("Generando visualización de muestras sospechosas con imágenes...")
        ok_any |= plot_suspect_samples_with_images(suspects_df, images_dict, outdir, n_suspects=3)
    elif suspects_df is not None:
        LOG.warning("No se pudieron cargar imágenes. Generando visualización sin imágenes...")
        ok_any |= plot_suspect_samples(suspects_df, npz, outdir, n_suspects=3)

    # Visualizaciones individuales (legacy, mantenidas para compatibilidad)
    LOG.info("=" * 80)
    LOG.info("Generando visualizaciones individuales (legacy)...")
    if suspects_df is not None:
        ok_any |= plot_lpips_ssim_scatter(suspects_df, outdir, lp_th, ss_th)
        ok_any |= plot_nn_distance_hist(suspects_df, outdir)
        ok_any |= plot_suspects_by_class(suspects_df, outdir)
    else:
        LOG.warning("No se generarán gráficos de memorization al faltar memorization_suspects.csv.")

    ok_any |= plot_roc(roc_df, npz, outdir)
    ok_any |= plot_loss_hist(npz, outdir)

    if ok_any:
        LOG.info("=" * 80)
        LOG.info("ÉXITO: Visualizaciones generadas correctamente.")
        LOG.info(f"Archivos principales:")
        LOG.info(f"  - {outdir / 'publication_panel_1x4.png'}")
        LOG.info(f"  - {outdir / 'publication_panel_1x4.pdf'}")
        LOG.info(f"  - {outdir / 'suspect_samples_images.png'}")
        LOG.info(f"  - {outdir / 'suspect_samples_images.pdf'}")
        LOG.info("=" * 80)
    else:
        LOG.error("No se generó ninguna visualización. Revise rutas y formatos de entrada.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera visualizaciones publication-ready de auditoría de privacidad de ccDDPM"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/audit_ccddpm_privacy.yaml"),
        help="Ruta al archivo de configuración audit_ccddpm_privacy.yaml"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Directorio de salida de figuras (opcional, usa io->outdir_visualizations del config por defecto)"
    )
    parser.add_argument(
        "--log",
        type=str,
        default="INFO",
        help="Nivel de log: DEBUG, INFO, WARNING, ERROR"
    )
    args = parser.parse_args()
    setup_logging(args.log)
    main(args.config, args.out)
