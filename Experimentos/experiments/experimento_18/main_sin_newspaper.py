
import pandas as pd
from newspaper import Article, Config
from tqdm import tqdm
import sys
import os
import time

# Añade la carpeta raíz 'Experimentos' al path
EXPERIMENTOS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(EXPERIMENTOS_DIR)

# Rutas absolutas (siempre apuntan a Experimentos/, independiente del CWD)
RUTA_VARIABLES_JSON = os.path.join(EXPERIMENTOS_DIR, "variables.json")
RUTA_TEMPLATE = os.path.join(EXPERIMENTOS_DIR, "prompts", "prompt_clara.md")

import variables # Importamos archivo variables.py
import utils # Importamos archivo utils.py

# ==========================================
# 1. CONFIGURACIÓN DE ARCHIVOS DE SALIDA
# ==========================================
# Definimos el nombre del archivo ANTES de cargar los datos para poder comprobar el progreso
FOLDER = "../../results"
PREFFIX = "18-Experimento-18_03_2026"
SUFFIX = "scrape"
nombre_output = f"{FOLDER}/{PREFFIX}_resultados_modelo_2024_{SUFFIX}.csv"

if not os.path.exists(FOLDER):
    os.makedirs(FOLDER)

# IMPORTANTE: Escribe aquí el nombre de la columna que identifica de forma única tu fila 
# (Ejemplo: 'id', 'no_Pagina_url', 'Titular', etc.)
COLUMNA_ID = 'IdNoticia' # <--- ¡CAMBIA ESTO por tu columna real!

# ==========================================
# 2. CARGA DE DATOS Y ESTADO DE REANUDACIÓN
# ==========================================

# Configuración de Newspaper
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
config = Config()
config.browser_user_agent = user_agent
config.request_timeout = 10

# Cargar CSV original
ruta_archivo = f"../../../data/2026_02_10_imio_def_todo_envio_heidy.xlsx - 2026_02_09_imio_def_todo_clara_{SUFFIX}.csv"

print("Cargando datos originales...")
try:
    data = pd.read_csv(ruta_archivo)
except FileNotFoundError:
    print("No se encuentra el archivo de datos.")
    exit()

# Filtrar año 2024 y generar el muestreo base
df_2024 = data[data['año'] == 2024].copy()
n_samples = min(1000, len(df_2024))
# Importante: al usar random_state=42 siempre obtenemos el MISMO set de 1000.
df_procesar = df_2024.sample(n=n_samples, random_state=42)

# --- NUEVA LÓGICA DE REANUDACIÓN ---
ids_procesados = set()
es_primera_vez = True

if os.path.exists(nombre_output):
    try:
        # Leemos el archivo donde estamos guardando los resultados
        df_salida = pd.read_csv(nombre_output)
        if COLUMNA_ID in df_salida.columns:
            # Obtenemos los IDs ya procesados
            ids_procesados = set(df_salida[COLUMNA_ID].dropna().astype(str).unique())
            es_primera_vez = False # Para que el modo de escritura sea 'a' (append) y sin cabecera
            print(f"-> Archivo previo detectado. {len(ids_procesados)} registros ya procesados encontrados.")
        else:
            print(f"-> Advertencia: No se encontró la columna '{COLUMNA_ID}' en el archivo de salida.")
    except pd.errors.EmptyDataError:
        print("-> El archivo de salida existe pero está vacío. Comenzando desde cero.")

# Filtramos las filas que ya están procesadas
total_antes = len(df_procesar)
df_procesar = df_procesar[~df_procesar[COLUMNA_ID].astype(str).isin(ids_procesados)]
total_despues = len(df_procesar)

print(f"Iniciando: quedan {total_despues} noticias por procesar (se omitieron {total_antes - total_despues}).")

# Si ya se procesaron todos, terminamos la ejecución
if df_procesar.empty:
    print("Todas las filas han sido procesadas. Saliendo del script.")
    exit()

# ==========================================
# 3. FUNCIÓN DE PROCESAMIENTO
# ==========================================

def _expandir_resultado(resultado, prefijo: str) -> dict:
    """
    Convierte un objeto Pydantic (con campos codigo/explicacion/evidencias)
    en 3 columnas: '{prefijo}' (código), '{prefijo}_explicacion' y '{prefijo}_evidencias'.
    """
    return {
        prefijo: resultado.codigo,
        f"{prefijo}_explicacion": resultado.explicacion,
        f"{prefijo}_evidencias": " | ".join(resultado.evidencias) if resultado.evidencias else "",
    }


def procesar_fila(row):
    """
    Toma una fila del DF, descarga la noticia y aplica las variables.
    Devuelve un diccionario con los campos 'modelo_..._codigo', 'modelo_..._explicacion'
    y 'modelo_..._evidencias' por cada variable analizada.
    """
    resultados = {}
    
    # Extraer textos básicos para pasar a las funciones (y evitar error NaN = float)
    titulo = str(row["Titular"]) if pd.notna(row["Titular"]) else ""
    texto = str(row["contenido_articulo"]) if pd.notna(row["contenido_articulo"]) else ""
    authors = str(row["no_Autor"]) if pd.notna(row["no_Autor"]) else ""

    # --- APLICACIÓN DE VARIABLES ---

    # 25. Lenguaje Sexista
    lenguaje_sexista = variables.clasificar_var_lenguaje_sexista(texto, ruta_json=RUTA_VARIABLES_JSON, ruta_template=RUTA_TEMPLATE, modelo='gpt-5-nano')
    resultados.update(_expandir_resultado(lenguaje_sexista, 'modelo_lenguaje_sexista'))

    # 26. Masc Generico
    masc_generico = variables.clasificar_var_masc_generico(texto, ruta_json=RUTA_VARIABLES_JSON, ruta_template=RUTA_TEMPLATE, modelo='gpt-5-nano')
    resultados.update(_expandir_resultado(masc_generico, 'modelo_masc_generico'))

    # 30. Sexismo Discurso
    sexismo_discurso = variables.clasificar_var_sexismo_discurso(texto, ruta_json=RUTA_VARIABLES_JSON, ruta_template=RUTA_TEMPLATE, modelo='gpt-5-nano')
    resultados.update(_expandir_resultado(sexismo_discurso, 'modelo_sexismo_discurso'))

    # 33. Asimetria Mujer Hombre
    asimetria_mujer_hombre = variables.clasificar_var_asimetria_mujer_hombre(texto, ruta_json=RUTA_VARIABLES_JSON, ruta_template=RUTA_TEMPLATE, modelo='gpt-5-nano')
    resultados.update(_expandir_resultado(asimetria_mujer_hombre, 'modelo_asimetria_mujer_hombre'))

    # 35. Denominacion Sexualizada
    denominacion_sexualizada = variables.clasificar_var_denominacion_sexualizada(texto, ruta_json=RUTA_VARIABLES_JSON, ruta_template=RUTA_TEMPLATE, modelo='gpt-5-nano')
    resultados.update(_expandir_resultado(denominacion_sexualizada, 'modelo_denominacion_sexualizada'))

    return resultados

# ==========================================
# 4. BUCLE PRINCIPAL Y GUARDADO INCREMENTAL
# ==========================================

print(f"Los datos se guardarán en tiempo real en: {nombre_output}")

for index, row in tqdm(df_procesar.iterrows(), total=df_procesar.shape[0]):

    # --- INICIO CRONÓMETRO ---
    start_time = time.time()

    # 1. Procesamos la fila
    res_fila = procesar_fila(row)

    # --- FIN CRONÓMETRO ---
    end_time = time.time()
    duration = end_time - start_time
    
    # 2. Unimos resultados
    fila_completa = row.to_dict()
    fila_completa.update(res_fila)

    # --- GUARDAR TIEMPO ---
    fila_completa['modelo_tiempo_procesamiento_seg'] = duration
    
    # 3. Convertimos ESTA fila a un DataFrame temporal
    df_temp = pd.DataFrame([fila_completa])
    
    # 4. Guardamos en el CSV
    if es_primera_vez:
        df_temp.to_csv(nombre_output, index=False, mode='w', encoding='utf-8')
        es_primera_vez = False
    else:
        df_temp.to_csv(nombre_output, index=False, mode='a', header=False, encoding='utf-8')

print(f"Proceso finalizado. Archivo completado: {nombre_output}")