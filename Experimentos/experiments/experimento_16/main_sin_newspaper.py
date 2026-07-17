
import pandas as pd
from tqdm import tqdm
import sys
import os
import time

# Añade la carpeta raíz 'Experimentos' al path
EXPERIMENTOS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
EXPERIMENTO_DIR = os.path.dirname(__file__)
sys.path.append(EXPERIMENTOS_DIR)
sys.path.append(EXPERIMENTO_DIR)

# Rutas absolutas (siempre apuntan a Experimentos/, independiente del CWD)
RUTA_VARIABLES_JSON = os.path.join(EXPERIMENTOS_DIR, "variables.json")

from clasificador_skills import clasificar_articulo, MODEL  # noqa: E402

# ==========================================
# 1. CONFIGURACIÓN DE ARCHIVOS DE SALIDA
# ==========================================
FOLDER = "../../results"
PREFFIX = "16-Experimento-16_05_2026"
SUFFIX = "scrape"
nombre_output = f"{FOLDER}/{PREFFIX}_resultados_modelo_2024_{SUFFIX}_skills.csv"

if not os.path.exists(FOLDER):
    os.makedirs(FOLDER)

COLUMNA_ID = 'IdNoticia'

# ==========================================
# 2. CARGA DE DATOS Y ESTADO DE REANUDACIÓN
# ==========================================

ruta_archivo = f"../../../data/2026_02_10_imio_def_todo_envio_heidy.xlsx - 2026_02_09_imio_def_todo_clara_{SUFFIX}.csv"

print(f"Modelo: {MODEL} + Claude Skills (skills/orquestador + skills/<var>/SKILL.md)")
print("Cargando datos originales...")
try:
    data = pd.read_csv(ruta_archivo)
except FileNotFoundError:
    print("No se encuentra el archivo de datos.")
    exit()

df_2024 = data[data['año'] == 2024].copy()
n_samples = min(1000, len(df_2024))
df_procesar = df_2024.sample(n=n_samples, random_state=42)

ids_procesados = set()
es_primera_vez = True

if os.path.exists(nombre_output):
    try:
        df_salida = pd.read_csv(nombre_output)
        if COLUMNA_ID in df_salida.columns:
            ids_procesados = set(df_salida[COLUMNA_ID].dropna().astype(str).unique())
            es_primera_vez = False
            print(f"-> Archivo previo detectado. {len(ids_procesados)} registros ya procesados encontrados.")
        else:
            print(f"-> Advertencia: No se encontró la columna '{COLUMNA_ID}' en el archivo de salida.")
    except pd.errors.EmptyDataError:
        print("-> El archivo de salida existe pero está vacío. Comenzando desde cero.")

total_antes = len(df_procesar)
df_procesar = df_procesar[~df_procesar[COLUMNA_ID].astype(str).isin(ids_procesados)]
total_despues = len(df_procesar)

print(f"Iniciando: quedan {total_despues} noticias por procesar (se omitieron {total_antes - total_despues}).")

if df_procesar.empty:
    print("Todas las filas han sido procesadas. Saliendo del script.")
    exit()

# ==========================================
# 3. FUNCIÓN DE PROCESAMIENTO
# ==========================================

def _expandir_resultado(resultado, prefijo: str) -> dict:
    return {
        prefijo: resultado.codigo,
        f"{prefijo}_explicacion": resultado.explicacion,
        f"{prefijo}_evidencias": " | ".join(resultado.evidencias) if resultado.evidencias else "",
    }


def procesar_fila(row):
    """
    Clasifica las 5 variables del experimento 16 con Claude Skills
    (una llamada por artículo: SKILL.md + codebook desde variables.json).
    """
    texto = str(row["contenido_articulo"]) if pd.notna(row["contenido_articulo"]) else ""

    clasificaciones = clasificar_articulo(texto, ruta_json=RUTA_VARIABLES_JSON)

    resultados = {}
    mapeo_prefijos = {
        "lenguaje_sexista": "modelo_lenguaje_sexista",
        "masc_generico": "modelo_masc_generico",
        "sexismo_discurso": "modelo_sexismo_discurso",
        "asimetria_mujer_hombre": "modelo_asimetria_mujer_hombre",
        "denominacion_sexualizada": "modelo_denominacion_sexualizada",
    }
    for nombre_var, prefijo in mapeo_prefijos.items():
        resultados.update(_expandir_resultado(clasificaciones[nombre_var], prefijo))

    return resultados

# ==========================================
# 4. BUCLE PRINCIPAL Y GUARDADO INCREMENTAL
# ==========================================

print(f"Los datos se guardarán en tiempo real en: {nombre_output}")

for index, row in tqdm(df_procesar.iterrows(), total=df_procesar.shape[0]):

    start_time = time.time()
    res_fila = procesar_fila(row)
    duration = time.time() - start_time

    fila_completa = row.to_dict()
    fila_completa.update(res_fila)
    fila_completa['modelo_tiempo_procesamiento_seg'] = duration

    df_temp = pd.DataFrame([fila_completa])

    if es_primera_vez:
        df_temp.to_csv(nombre_output, index=False, mode='w', encoding='utf-8')
        es_primera_vez = False
    else:
        df_temp.to_csv(nombre_output, index=False, mode='a', header=False, encoding='utf-8')

print(f"Proceso finalizado. Archivo completado: {nombre_output}")
