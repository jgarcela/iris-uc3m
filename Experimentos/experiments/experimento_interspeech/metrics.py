import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    cohen_kappa_score, 
    mean_absolute_error
)
import os
import datetime
import uuid

# ==========================================
# 1. FUNCIONES AUXILIARES Y VARIABLES
# ==========================================

def clean_val(x):
    """Normaliza valores mixtos (1, 1.0, '1') a string limpio para comparar categorías."""
    if pd.isna(x) or str(x).strip() == "":
        return "N/A"
    s = str(x).strip()
    if s.endswith('.0'):
        s = s[:-2]
    return s

# Columnas que deben tratarse como numéricas para regresión (MAE, SD)
# NUMERIC_COLS = [
#     'Medio_num', 
#     'año', 
#     'no_MES', 
#     'Caracteres', 
#     'numero_fotografias', 
#     'numero_declaraciones'
# ]

# Pares de variables a comparar (Real vs Predicción)
PAIRS = [
    ('lenguaje_sexista', 'modelo_lenguaje_sexista'),
    ('masc_generico', 'modelo_masc_generico'),
    ('hombre_denominar_humanidad', 'modelo_hombre_denominar_humanidad'),
    ('uso_dual_zorr', 'modelo_uso_dual_zorr'),
    ('uso_cargo_mujer', 'modelo_uso_cargo_mujer'),
    ('sexismo_social', 'modelo_sexismo_social'),
    ('androcentrismo', 'modelo_androcentrismo'), 
    ('mencion_nombre_investigadora', 'modelo_mencion_nombre_investigadora'),
    ('asimetria_mujer_hombre', 'modelo_asimetria_mujer_hombre'),         
    ('infatilizacion', 'modelo_infatilizacion'),       
    ('denominacion_sexualizada', 'modelo_denominacion_sexualizada'),
    ('denominacion_redundante', 'modelo_denominacion_redundante'),
    ('denominacion_dependiente', 'modelo_denominacion_dependiente'),
    ('criterios_excepcion', 'modelo_criterios_excepcion'),
    ('comparacion_mujer_hombre', 'modelo_comparacion_mujer_hombre'),
]

# ==========================================
# 2. FUNCIÓN PRINCIPAL DE MÉTRICAS Y SUMMARY
# ==========================================

def generar_metricas_y_summary(archivo_input, archivo_output, nombre_experimento, modelo_usado, prompt_file):
    """
    Lee los resultados de un experimento, calcula las métricas por variable,
    guarda el CSV de métricas detalladas y añade un log global en runs.csv.
    """
    print(f"\n--- Calculando métricas para: {nombre_experimento} ---")
    print(f"Cargando datos de: {archivo_input}")
    
    try:
        df = pd.read_csv(archivo_input)
    except FileNotFoundError:
        print(f"Error: No se encuentra el archivo {archivo_input}.")
        return

    metrics_list = []

    for true_col, pred_col in PAIRS:
        # 1. Validar existencia
        if true_col not in df.columns or pred_col not in df.columns:
            metrics_list.append({'Experimento': nombre_experimento, 'Variable': true_col, 'Tipo': 'ERROR', 'Accuracy': 0})
            continue

        # 2. Filtrar filas inválidas (NaN en Realidad o Predicción)
        temp_df = df[[true_col, pred_col]].dropna()
        
        if len(temp_df) == 0:
            metrics_list.append({'Experimento': nombre_experimento, 'Variable': true_col, 'Tipo': 'VACÍO', 'Accuracy': 0})
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
                    'Experimento': nombre_experimento,
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
                print(f"Error en variable numérica {true_col}: {e}")

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
                    'Experimento': nombre_experimento,
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
    # 3. GUARDADO DEL CSV DE MÉTRICAS DETALLADAS
    # ==========================================
    df_metrics = pd.DataFrame(metrics_list)
    cols_order = ['Experimento', 'Variable', 'Tipo', 'N_Muestras', 'Accuracy', 'Kappa', 
                  'F1_Macro', 'F1_Weighted', 'MAE', 'SD_Error']
    cols_final = [c for c in cols_order if c in df_metrics.columns] + [c for c in df_metrics.columns if c not in cols_order]
    df_metrics = df_metrics[cols_final]

    # Asegurar que el directorio base de archivo_output existe
    directorio_metricas = os.path.dirname(archivo_output)
    if directorio_metricas:
        os.makedirs(directorio_metricas, exist_ok=True)
        
    df_metrics.to_csv(archivo_output, index=False)
    print(f"Archivo de métricas detalladas guardado en: {archivo_output}")

    # ==========================================
    # 4. GENERACIÓN Y GUARDADO DEL SUMMARY (runs.csv)
    # ==========================================
    print("Generando summary del run...")

    run_id = f"run_{str(uuid.uuid4())[:8]}"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n_textos = len(df)
    
    # Extraer el tiempo de la columna si existe
    if 'modelo_tiempo_procesamiento_seg' in df.columns:
        tiempo_total_seg = round(df['modelo_tiempo_procesamiento_seg'].sum(), 2)
    else:
        tiempo_total_seg = 0.0

    def obtener_media(columna):
        if columna in df_metrics.columns:
            return round(pd.to_numeric(df_metrics[columna], errors='coerce').mean(), 4)
        return None

    # Excluimos variables que fallaron para no alterar el contador real
    variables_validas = df_metrics[~df_metrics['Tipo'].isin(['ERROR', 'VACÍO'])]
    n_variables = len(variables_validas)

    run_summary = {
        'experimento': nombre_experimento,
        'run_id': run_id,
        'timestamp': timestamp,
        'modelo': modelo_usado,
        'prompt_file': prompt_file,
        'n_textos': n_textos,
        'n_variables': n_variables,
        'accuracy': obtener_media('Accuracy'),
        'kappa': obtener_media('Kappa'),
        'f1_micro': obtener_media('F1_Micro'),
        'f1_macro': obtener_media('F1_Macro'),
        'f1_weighted': obtener_media('F1_Weighted'),
        'mae': obtener_media('MAE'),
        'sd': obtener_media('SD_Error'),
        'tiempo_total_seg': tiempo_total_seg
    }

    # Ruta de nuestro dashboard de runs
    archivo_runs = '../../metrics/runs.csv' 
    os.makedirs(os.path.dirname(archivo_runs), exist_ok=True)

    df_run = pd.DataFrame([run_summary])
    
    # Añadir o crear el csv de runs
    if os.path.exists(archivo_runs):
        df_run.to_csv(archivo_runs, mode='a', header=False, index=False, encoding='utf-8')
    else:
        df_run.to_csv(archivo_runs, mode='w', header=True, index=False, encoding='utf-8')

    print(f"Summary del run [{run_id}] añadido exitosamente a: {archivo_runs}\n")

# ==========================================
# 5. EJECUCIÓN DIRECTA (MODO MANUAL/DEBUG)
# ==========================================
# Si ejecutas 'python metrics.py' directamente, usará estos valores de prueba.
if __name__ == "__main__":
    NOMBRE_EXPERIMENTO = "Experimento-1_02-2026"
    NOMBRE_ARCHIVO = 'Experimento-1_02-2026_resultados_modelo_2024_scrape.csv'
    ARCHIVO_INPUT = f'../../results/{NOMBRE_ARCHIVO}'
    ARCHIVO_OUTPUT = f'../../metrics/metrics_{NOMBRE_ARCHIVO}'
    
    generar_metricas_y_summary(
        archivo_input=ARCHIVO_INPUT,
        archivo_output=ARCHIVO_OUTPUT,
        nombre_experimento=NOMBRE_EXPERIMENTO,
        modelo_usado="Test-Model",
        prompt_file="Test-Prompt.txt"
    )