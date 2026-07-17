"""
Métricas del experimento_18 (cluster). Lee el CSV unido (_FULL) de un modelo y
calcula accuracy/kappa/F1 por variable frente al ground truth.

Uso:
    python metrics.py [ruta_csv_FULL] [etiqueta_modelo]

Por defecto lee results/18-Experimento-18_03_2026_resultados_modelo_2024_scrape_FULL.csv
y escribe metrics/metrics_<basename>.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    f1_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    mean_absolute_error,
)

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(
    BASE_DIR, "results",
    "18-Experimento-18_03_2026_resultados_modelo_2024_scrape_FULL.csv",
)

ARCHIVO_INPUT = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
ETIQUETA_MODELO = sys.argv[2] if len(sys.argv) > 2 else "modelo"

METRICS_DIR = os.path.join(BASE_DIR, "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)
ARCHIVO_OUTPUT = os.path.join(METRICS_DIR, f"metrics_{os.path.basename(ARCHIVO_INPUT)}")

NUMERIC_COLS = ['Caracteres', 'numero_fotografias', 'numero_declaraciones']


# ==========================================
# 2. LIMPIEZA
# ==========================================
def clean_val(x):
    """
    Normaliza valores mixtos a string limpio para comparar categorías.
    Maneja: 1, 1.0, '1', y el formato español con coma decimal '1,0' -> '1'.
    """
    if pd.isna(x) or str(x).strip() == "":
        return "N/A"
    s = str(x).strip().strip("'\"")   # quita comillas envolventes ('1,0')
    s = s.replace(",", ".")            # coma decimal española -> punto
    if s.endswith('.0'):
        s = s[:-2]
    return s


# ==========================================
# 3. PROCESAMIENTO
# ==========================================
print(f"Cargando datos de: {ARCHIVO_INPUT}")
try:
    df = pd.read_csv(ARCHIVO_INPUT)
except FileNotFoundError:
    sys.exit(f"Error: No se encuentra el archivo {ARCHIVO_INPUT}")

pairs = [
    ('lenguaje_sexista', 'modelo_lenguaje_sexista'),
    ('masc_generico', 'modelo_masc_generico'),
    ('sexismo_discurso', 'modelo_sexismo_discurso'),
    ('asimetria_mujer_hombre', 'modelo_asimetria_mujer_hombre'),
    ('denominacion_sexualizada', 'modelo_denominacion_sexualizada'),
]

metrics_list = []
print("Calculando métricas detalladas...")

for true_col, pred_col in pairs:
    if true_col not in df.columns or pred_col not in df.columns:
        metrics_list.append({'Modelo': ETIQUETA_MODELO, 'Variable': true_col, 'Tipo': 'ERROR', 'Accuracy': 0})
        continue

    temp_df = df[[true_col, pred_col]].dropna()
    if len(temp_df) == 0:
        metrics_list.append({'Modelo': ETIQUETA_MODELO, 'Variable': true_col, 'Tipo': 'VACÍO', 'Accuracy': 0})
        continue

    if true_col in NUMERIC_COLS:
        y_true = pd.to_numeric(temp_df[true_col], errors='coerce').fillna(0)
        y_pred = pd.to_numeric(temp_df[pred_col], errors='coerce').fillna(0)
        mae = mean_absolute_error(y_true, y_pred)
        sd_error = np.std(y_true - y_pred)
        metrics_list.append({
            'Modelo': ETIQUETA_MODELO, 'Variable': true_col, 'Tipo': 'Numérica',
            'N_Muestras': len(temp_df), 'Accuracy': None, 'Kappa': None,
            'F1_Macro': None, 'F1_Weighted': None,
            'MAE': round(mae, 4), 'SD_Error': round(sd_error, 4),
        })
    else:
        y_true = temp_df[true_col].apply(clean_val)
        y_pred = temp_df[pred_col].apply(clean_val)
        acc = accuracy_score(y_true, y_pred)
        # Accuracy del baseline tonto (predecir siempre la clase mayoritaria real):
        # contexto imprescindible en variables desbalanceadas.
        baseline_acc = y_true.value_counts(normalize=True).max()
        try:
            kappa = cohen_kappa_score(y_true, y_pred)
        except Exception:
            kappa = 0
        try:
            bal_acc = balanced_accuracy_score(y_true, y_pred)   # media de recalls por clase
        except Exception:
            bal_acc = 0
        try:
            mcc = matthews_corrcoef(y_true, y_pred)             # robusto ante desbalanceo
        except Exception:
            mcc = 0
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Precision/Recall/F1 de la clase positiva "2" (=Sí), que es la difícil
        # en las variables desbalanceadas (asimetria, denominacion).
        p2, r2, f2, sup2 = precision_recall_fscore_support(
            y_true, y_pred, labels=["2"], average=None, zero_division=0)

        metrics_list.append({
            'Modelo': ETIQUETA_MODELO, 'Variable': true_col, 'Tipo': 'Categórica',
            'N_Muestras': len(y_true), 'Accuracy': round(acc, 4),
            'Baseline_Acc': round(float(baseline_acc), 4), 'Balanced_Acc': round(bal_acc, 4),
            'Kappa': round(kappa, 4), 'MCC': round(mcc, 4),
            'F1_Macro': round(f1_macro, 4), 'F1_Weighted': round(f1_weighted, 4),
            'Prec_Si': round(float(p2[0]), 4), 'Recall_Si': round(float(r2[0]), 4),
            'F1_Si': round(float(f2[0]), 4), 'N_Si_real': int(sup2[0]),
            'MAE': None, 'SD_Error': None,
        })

# ==========================================
# 4. GUARDADO
# ==========================================
df_metrics = pd.DataFrame(metrics_list)
cols_order = ['Modelo', 'Variable', 'Tipo', 'N_Muestras', 'Accuracy', 'Baseline_Acc',
              'Balanced_Acc', 'Kappa', 'MCC', 'F1_Macro', 'F1_Weighted',
              'Prec_Si', 'Recall_Si', 'F1_Si', 'N_Si_real', 'MAE', 'SD_Error']
cols_final = [c for c in cols_order if c in df_metrics.columns] + \
             [c for c in df_metrics.columns if c not in cols_order]
df_metrics = df_metrics[cols_final]

print(df_metrics.to_markdown(index=False))
df_metrics.to_csv(ARCHIVO_OUTPUT, index=False)
print(f"\nArchivo de métricas guardado en: {ARCHIVO_OUTPUT}")
