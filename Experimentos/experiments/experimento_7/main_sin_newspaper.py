
import pandas as pd
from newspaper import Article, Config
from tqdm import tqdm
import sys
import os
import time

# Añade la carpeta raíz 'Experimentos' al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import variables # Importamos archivo variables.py
import utils # Importamos archivo utils.py

# ==========================================
# 1. CONFIGURACIÓN DE ARCHIVOS DE SALIDA
# ==========================================
# Definimos el nombre del archivo ANTES de cargar los datos para poder comprobar el progreso
FOLDER = "../../results"
PREFFIX = "7-Experimento-7_03_2026"
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

def procesar_fila(row):
    """
    Toma una fila del DF, descarga la noticia y aplica las variables.
    Devuelve un diccionario con los campos 'modelo_...'.
    """
    resultados = {}
    
    # Extraer textos básicos para pasar a las funciones (y evitar error NaN = float)
    titulo = str(row["Titular"]) if pd.notna(row["Titular"]) else ""
    texto = str(row["contenido_articulo"]) if pd.notna(row["contenido_articulo"]) else ""
    authors = str(row["no_Autor"]) if pd.notna(row["no_Autor"]) else ""

    # --- APLICACIÓN DE VARIABLES ---
    # 7a. Nombre Propio Titular (Lista)
    np_titular = variables.clasificar_var_nombre_propio_titular_list_e3(titulo=titulo, modelo='deepseek-r1:1.5b')
    resultados['modelo_nombre_propio_titular_nombres'] = str(np_titular.nombres)
    resultados['modelo_nombre_propio_titular_valores'] = str(np_titular.valores)
    
    # 7b. Género Nombre Propio Titular (Cálculo sobre la lista)
    resultados['modelo_nombre_propio_titular'] = variables.clasificar_var_nombre_propio_titular(np_titular.valores)

    # 9a. Protagonistas Cuerpo
    protas = variables.clasificar_var_cla_genero_prota_list_e3(texto_noticia=texto, modelo='deepseek-r1:1.5b')
    resultados['modelo_cla_genero_prota_nombres'] = str(protas.nombres)
    resultados['modelo_cla_genero_prota_valores'] = str(protas.valores)

    # 9b. Género Protagonistas
    resultados['modelo_cla_genero_prota'] = variables.clasificar_var_cla_genero_prota(protas.valores)

    # 10. Periodista
    nombre_periodista = variables.clasificar_var_nombre_periodista_authors(authors)
    resultados['modelo_nombre_periodista'] = nombre_periodista

    # 11. Género Periodista (Autoría)
    nombre_medio = resultados.get('modelo_Medio_nombre', 'Desconocido')
    resultados['modelo_genero_periodista'] = variables.clasificar_var_genero_periodista_e3(nombre_periodista=nombre_periodista, nombre_medio=nombre_medio, modelo='deepseek-r1:1.5b')

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