"""
Métricas detalladas del experimento_12 (versión cluster).

Misma lógica que Experimentos/experiments/experimento_12/metrics.py,
pero con rutas adaptadas a esta carpeta y configurables por CLI.

Por defecto lee el CSV unido que genera merge_shards.py y escribe las
métricas en ./metrics/ dentro de esta carpeta.

Uso:
    python3 metrics.py                     # usa el FULL por defecto
    python3 metrics.py --input ruta.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    mean_absolute_error,
)

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NOMBRE_EXPERIMENTO = "12-Experimento-12_03_2026"
NOMBRE_ARCHIVO = "12-Experimento-12_03_2026_resultados_modelo_2024_scrape_FULL.csv"

parser = argparse.ArgumentParser(description="Métricas exp12 (cluster)")
parser.add_argument("--input", default=os.path.join(BASE_DIR, "results", NOMBRE_ARCHIVO),
                    help="CSV de resultados (por defecto el _FULL unido con merge_shards.py)")
parser.add_argument("--output-dir", default=os.path.join(BASE_DIR, "metrics"),
                    help="Carpeta donde guardar el CSV de métricas")
args = parser.parse_args()

ARCHIVO_INPUT = args.input
os.makedirs(args.output_dir, exist_ok=True)
ARCHIVO_OUTPUT = os.path.join(args.output_dir, f"metrics_{os.path.basename(ARCHIVO_INPUT)}")

# Columnas que son puramente numéricas (conteo)
NUMERIC_COLS = ['Caracteres', 'numero_fotografias', 'numero_declaraciones']

# ==========================================
# 2. FUNCIONES DE RECODIFICACIÓN Y LIMPIEZA
# ==========================================

def clean_val(x):
    """Normaliza valores mixtos (1, 1.0, '1') a string limpio para comparar categorías."""
    if pd.isna(x) or str(x).strip() == "":
        return "N/A"
    s = str(x).strip()
    if s.endswith('.0'):
        s = s[:-2]
    return s

# --- Funciones de Traducción (Modelo -> Realidad) ---

def traducir_titular(val):
    try: x = int(float(val))
    except: return 1
    if x == 1: return 2       # Hombre
    if x == 2: return 3       # Mujer
    if x in [3, 32, 33]: return 4  # Mixto
    if x in [4, 41, 42]: return 5  # Neutro
    return 1

def traducir_protagonista(val):
    try: x = int(float(val))
    except: return 4
    if x == 1: return 1       # Hombre
    if x == 2: return 2       # Mujer
    if x in [3, 32, 33]: return 3  # Mixto
    return 4                  # Neutro

def traducir_periodista(val):
    try: x = int(float(val))
    except: return 4
    if x in [1, 2, 3, 6, 7]: return x
    if x == 0: return 4
    if x in [4, 5]: return 5
    return 4

# ==========================================
# 3. PROCESAMIENTO
# ==========================================

print(f"Cargando datos de: {ARCHIVO_INPUT}")
try:
    df = pd.read_csv(ARCHIVO_INPUT)
except FileNotFoundError:
    raise SystemExit(
        f"Error: No se encuentra el archivo {ARCHIVO_INPUT}\n"
        "¿Has unido los shards con 'python3 merge_shards.py ./results'?"
    )

# --- APLICAR RECODIFICACIÓN AL VUELO ---
print("Aplicando recodificación de variables complejas...")
df['modelo_recod_nombre_propio_titular'] = df['modelo_nombre_propio_titular'].apply(traducir_titular)
df['modelo_recod_cla_genero_prota'] = df['modelo_cla_genero_prota'].apply(traducir_protagonista)
df['modelo_recod_genero_periodista'] = df['modelo_genero_periodista'].apply(traducir_periodista)

# Pares de variables a comparar
pairs = [
    ('nombre_propio_titular', 'modelo_recod_nombre_propio_titular'),  # Usamos la recodificada
    ('cita_en_titulo', 'modelo_cita_en_titulo'),
    ('cla_genero_prota', 'modelo_recod_cla_genero_prota'),          # Usamos la recodificada
    ('genero_periodista', 'modelo_recod_genero_periodista'),        # Usamos la recodificada
]

metrics_list = []

print("Calculando métricas detalladas...")

for true_col, pred_col in pairs:
    # 1. Validar existencia
    if true_col not in df.columns or pred_col not in df.columns:
        metrics_list.append({'Experimento': NOMBRE_EXPERIMENTO, 'Variable': true_col, 'Tipo': 'ERROR', 'Accuracy': 0})
        continue

    # 2. Filtrar filas inválidas (NaN en Realidad o Predicción)
    temp_df = df[[true_col, pred_col]].dropna()

    if len(temp_df) == 0:
        metrics_list.append({'Experimento': NOMBRE_EXPERIMENTO, 'Variable': true_col, 'Tipo': 'VACÍO', 'Accuracy': 0})
        continue

    # --- CASO NUMÉRICO (Regresión) ---
    if true_col in NUMERIC_COLS:
        try:
            y_true = pd.to_numeric(temp_df[true_col], errors='coerce').fillna(0)
            y_pred = pd.to_numeric(temp_df[pred_col], errors='coerce').fillna(0)

            mae = mean_absolute_error(y_true, y_pred)
            errores = y_true - y_pred
            sd_error = np.std(errores)

            metrics_list.append({
                'Experimento': NOMBRE_EXPERIMENTO,
                'Variable': true_col,
                'Tipo': 'Numérica',
                'N_Muestras': len(temp_df),
                'Accuracy': None,
                'Kappa': None,
                'F1_Micro': None,
                'F1_Macro': None,
                'F1_Weighted': None,
                'MAE': round(mae, 4),
                'SD_Error': round(sd_error, 4)
            })
        except Exception as e:
            print(f"Error en numérica {true_col}: {e}")

    # --- CASO CATEGÓRICO (Clasificación) ---
    else:
        y_true = temp_df[true_col].apply(clean_val)
        y_pred = temp_df[pred_col].apply(clean_val)

        if len(y_true) > 0:
            acc = accuracy_score(y_true, y_pred)

            try:
                kappa = cohen_kappa_score(y_true, y_pred)
            except:
                kappa = 0

            f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
            f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
            f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

            metrics_list.append({
                'Experimento': NOMBRE_EXPERIMENTO,
                'Variable': true_col,
                'Tipo': 'Categórica',
                'N_Muestras': len(y_true),
                'Accuracy': round(acc, 4),
                'Kappa': round(kappa, 4),
                'F1_Micro': round(f1_micro, 4),
                'F1_Macro': round(f1_macro, 4),
                'F1_Weighted': round(f1_weighted, 4),
                'MAE': None,
                'SD_Error': None
            })

# ==========================================
# 4. GUARDADO
# ==========================================

df_metrics = pd.DataFrame(metrics_list)

cols_order = ['Experimento', 'Variable', 'Tipo', 'N_Muestras', 'Accuracy', 'Kappa',
              'F1_Macro', 'F1_Weighted', 'MAE', 'SD_Error']
cols_final = [c for c in cols_order if c in df_metrics.columns] + [c for c in df_metrics.columns if c not in cols_order]

df_metrics = df_metrics[cols_final]

print(df_metrics.to_markdown(index=False))
df_metrics.to_csv(ARCHIVO_OUTPUT, index=False)
print(f"\nArchivo de métricas guardado exitosamente en: {ARCHIVO_OUTPUT}")
