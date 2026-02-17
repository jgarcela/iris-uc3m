import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    cohen_kappa_score, 
    mean_absolute_error
)
import os

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================

NOMBRE_EXPERIMENTO = "Experimento-1_02-2026"
NOMBRE_ARCHIVO = 'Experimento-1_02-2026_resultados_modelo_2024_scrape.csv'
ARCHIVO_INPUT = f'../../results/{NOMBRE_ARCHIVO}'
ARCHIVO_OUTPUT = f'../../metrics/metrics_{NOMBRE_ARCHIVO}'

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
# Las incluimos aquí para asegurar que las columnas "recod" existan en memoria

def traducir_titular(val):
    try: x = int(float(val))
    except: return 1
    if x == 1: return 2       # Hombre
    if x == 2: return 3       # Mujer
    if x in [3, 32, 33]: return 4 # Mixto
    if x in [4, 41, 42]: return 5 # Neutro
    return 1

def traducir_protagonista(val):
    try: x = int(float(val))
    except: return 4
    if x == 1: return 1       # Hombre
    if x == 2: return 2       # Mujer
    if x in [3, 32, 33]: return 3 # Mixto
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
    print("Error: No se encuentra el archivo.")
    exit()

# --- APLICAR RECODIFICACIÓN AL VUELO ---
print("Aplicando recodificación de variables complejas...")
df['modelo_recod_nombre_propio_titular'] = df['modelo_nombre_propio_titular'].apply(traducir_titular)
df['modelo_recod_cla_genero_prota'] = df['modelo_cla_genero_prota'].apply(traducir_protagonista)
df['modelo_recod_genero_periodista'] = df['modelo_genero_periodista'].apply(traducir_periodista)

# Pares de variables a comparar
pairs = [
    ('Medio_num', 'modelo_Medio_num'),
    ('Fecha', 'modelo_Fecha'),
    ('año', 'modelo_año'),
    ('no_MES', 'modelo_no_MES'),
    ('Caracteres', 'modelo_Caracteres'),
    ('Titular', 'modelo_Titular'),
    ('nombre_propio_titular', 'modelo_recod_nombre_propio_titular'), # Usamos la recodificada
    ('cita_en_titulo', 'modelo_cita_en_titulo'),
    ('cla_genero_prota', 'modelo_recod_cla_genero_prota'),         # Usamos la recodificada
    ('genero_periodista', 'modelo_recod_genero_periodista'),       # Usamos la recodificada
    ('Tema_recodificado', 'modelo_tema_codigo'),
    ('ia_tema_central', 'modelo_ia_tema_central_codigo'),
    ('significado_ia', 'modelo_ia_significado_codigo'),
    ('menciona_ia', 'modelo_ia_menciona_codigo'),
    ('referencia_politicas_genero', 'modelo_politicas_genero_codigo'),
    ('denuncia_desigualdad_genero', 'modelo_denuncia_desigualdad_codigo'),
    ('mujeres_racializadas_noticias', 'modelo_mujeres_racializadas_codigo'),
    ('mujeres_con_discapacidad_noticias', 'modelo_mujeres_con_discapacidad_codigo'),
    ('mujeres_generacionalidad_noticias', 'modelo_mujeres_generacionalidad_codigo'),
    ('tiene_fotografias', 'modelo_tiene_fotografias_codigo'),
    ('numero_fotografias', 'modelo_numero_fotografias'),
    ('utiliza_fuente', 'modelo_utiliza_fuente_codigo'),
    ('numero_declaraciones', 'modelo_numero_declaraciones')
]

metrics_list = []

print("Calculando métricas detalladas...")

for true_col, pred_col in pairs:
    # 1. Validar existencia
    if true_col not in df.columns or pred_col not in df.columns:
        metrics_list.append({'Experimento': NOMBRE_EXPERIMENTO, 'Variable': true_col, 'Tipo': 'ERROR', 'Accuracy': 0})
        continue

    # 2. Filtrar filas inválidas (NaN en Realidad o Predicción)
    # Importante: Convertimos a string primero para evitar errores de tipo
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
            sd_error = np.std(errores) # Desviación estándar del error
            
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
        # Limpieza estricta
        y_true = temp_df[true_col].apply(clean_val)
        y_pred = temp_df[pred_col].apply(clean_val)
        
        # Filtramos si la realidad es 0 (asumiendo que 0 significa "No aplica" en tu manual original)
        # Si tu manual usa 0 como valor válido, comenta estas dos líneas:
        # mask_valid = y_true != "0"
        # y_true, y_pred = y_true[mask_valid], y_pred[mask_valid]

        if len(y_true) > 0:
            # Cálculos
            acc = accuracy_score(y_true, y_pred)
            
            # Kappa (Mide concordancia descontando el azar)
            # Manejamos excepción si solo hay una clase
            try:
                kappa = cohen_kappa_score(y_true, y_pred)
            except:
                kappa = 0

            # F1 Scores (Diferentes formas de promediar)
            # Micro: Global (igual a Accuracy en multiclase balanceada)
            # Macro: Promedio simple por clase (da igual peso a clases pequeñas)
            # Weighted: Promedio ponderado por número de ejemplos reales
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

# Reordenar columnas para que se vea bonito
cols_order = ['Experimento', 'Variable', 'Tipo', 'N_Muestras', 'Accuracy', 'Kappa', 
              'F1_Macro', 'F1_Weighted', 'MAE', 'SD_Error']
# Añadimos las columnas que falten en el orden (por si acaso)
cols_final = [c for c in cols_order if c in df_metrics.columns] + [c for c in df_metrics.columns if c not in cols_order]

df_metrics = df_metrics[cols_final]

print(df_metrics.to_markdown(index=False))
df_metrics.to_csv(ARCHIVO_OUTPUT, index=False)
print(f"\nArchivo de métricas guardado exitosamente en: {ARCHIVO_OUTPUT}")