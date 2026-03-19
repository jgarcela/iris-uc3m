import pandas as pd
from newspaper import Article, Config
from tqdm import tqdm
import sys
import os

# Añade la carpeta raíz 'Experimentos' al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import variables # Importamos archivo variables.py
import utils # Importamos archivo utils.py
import time
from typing import Optional, Any, List
import os

# ==========================================
# 1. CONFIGURACIÓN Y CARGA DE DATOS
# ==========================================

# Configuración de Newspaper para evitar bloqueos básicos
user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
config = Config()
config.browser_user_agent = user_agent
config.request_timeout = 10

# Cargar CSV
SUFFIX = "scrape"
# Ajusta la ruta a tu archivo real
ruta_archivo = f"../../../data/2026_02_10_imio_def_todo_envio_heidy.xlsx - 2026_02_09_imio_def_todo_clara_{SUFFIX}.csv"

print("Cargando datos...")
try:
    data = pd.read_csv(ruta_archivo)
except FileNotFoundError:
    print("No se encuentra el archivo, usando ruta alternativa de prueba o generando dummy data.")
    # data = pd.read_csv("ruta_alternativa.csv") 
    exit()

# Filtrar año 2024
df_2024 = data[data['año'] == 2024].copy()

# Muestreo de 1000 (o el total si hay menos de 1000)
n_samples = min(1000, len(df_2024))
df_procesar = df_2024.sample(n=n_samples, random_state=42)

print(f"Procesando {len(df_procesar)} noticias del año 2024...")

# ==========================================
# 2. FUNCIÓN DE PROCESAMIENTO
# ==========================================

def procesar_fila(row):
    """
    Toma una fila del DF, descarga la noticia y aplica las variables.
    Devuelve un diccionario con los campos 'modelo_...'.
    """
    url = row.get('no_Pagina_url', '')
    if not isinstance(url, str) or len(url) < 5:
        return {} # URL inválida
    
    # Inicializar resultado
    resultados = {}
    
    # 1. Descargar Artículo
    articulo = Article(url, config=config)
    try:
        articulo.download()
        articulo.parse()
        # articulo.nlp() # Opcional si usas keywords nativas de newspaper
    except Exception as e:
        # Si falla la descarga, devolvemos diccionario vacío o con error
        resultados['modelo_error_descarga'] = str(e)
        return resultados

    # Extraer textos básicos para pasar a las funciones
    # titulo = row["Titular"]
    # texto = row["contenido_articulo"]
    titulo = variables.clasificar_var_titular(articulo) or ""
    texto = articulo.text or ""

    # --- APLICACIÓN DE VARIABLES ---

    # 7a. Nombre Propio Titular (Lista)
    np_titular = variables.clasificar_var_nombre_propio_titular_list(titulo)
    resultados['modelo_nombre_propio_titular_nombres'] = str(np_titular.nombres)
    resultados['modelo_nombre_propio_titular_valores'] = str(np_titular.valores)
    
    # 7b. Género Nombre Propio Titular (Cálculo sobre la lista)
    resultados['modelo_nombre_propio_titular'] = variables.clasificar_var_nombre_propio_titular(np_titular.valores)

    # 8. Cita Titular
    cita = variables.clasificar_var_cita_titular(titulo)
    resultados['modelo_cita_en_titulo'] = cita.tipo
    resultados['modelo_cita_en_titulo_texto'] = cita.cita

    # 9a. Protagonistas Cuerpo
    protas = variables.clasificar_var_cla_genero_prota_list(texto)
    resultados['modelo_cla_genero_prota_nombres'] = str(protas.nombres)
    resultados['modelo_cla_genero_prota_valores'] = str(protas.valores)

    # 9b. Género Protagonistas
    resultados['modelo_cla_genero_prota'] = variables.clasificar_var_cla_genero_prota(protas.valores)

    # 10. Periodista
    nombre_periodista = variables.clasificar_var_nombre_periodista(articulo)
    resultados['modelo_nombre_periodista'] = nombre_periodista

    # 11. Género Periodista (Autoría)
    nombre_medio = resultados.get('modelo_Medio_nombre', 'Desconocido')
    resultados['modelo_genero_periodista'] = variables.clasificar_var_genero_periodista(nombre_periodista, nombre_medio)


    return resultados

# ==========================================
# 3. BUCLE PRINCIPAL Y GUARDADO INCREMENTAL
# ==========================================

# Definimos el nombre del archivo ANTES de empezar
FOLDER = "../../results"
PREFFIX = "Experimento-2" + "_" + "02-2026"
nombre_output = f"{FOLDER}/{PREFFIX}_resultados_modelo_2024_{SUFFIX}.csv"

# --- IMPORTANTE: Crear la carpeta si no existe para evitar error ---
if not os.path.exists(FOLDER):
    os.makedirs(FOLDER)

# Variable para controlar si escribimos la cabecera (header)
es_primera_vez = True

# Si el archivo ya existe de una ejecución anterior fallida, podrías querer borrarlo o seguir.
# Por seguridad, aquí no lo borramos, pero ten cuidado de no duplicar si lo lanzas dos veces.

print(f"Iniciando procesamiento. Los datos se guardarán en tiempo real en: {nombre_output}")

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
        # La primera vez: mode='w' (write) crea el archivo y pone cabeceras
        df_temp.to_csv(nombre_output, index=False, mode='w', encoding='utf-8')
        es_primera_vez = False
    else:
        # Las siguientes veces: mode='a' (append) añade al final SIN cabeceras
        df_temp.to_csv(nombre_output, index=False, mode='a', header=False, encoding='utf-8')

print(f"Proceso finalizado. Archivo completado: {nombre_output}")