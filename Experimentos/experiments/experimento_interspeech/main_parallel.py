import pandas as pd
from newspaper import Article, Config
from tqdm import tqdm
import variables # Importamos archivo variables.py
import utils # Importamos archivo utils.py
import time
from typing import Optional, Any, List
import os
import glob # <-- Importamos glob para escanear carpetas

# Importamos la función que generamos en metrics.py
from metrics import generar_metricas_y_summary

import threading # <-- Para evitar errores al escribir en el CSV
from concurrent.futures import ThreadPoolExecutor, as_completed # <-- Para el paralelismo

# ==========================================
# 0. CONFIGURACIÓN DEL GRID SEARCH Y MÉTRICAS
# ==========================================

# 1. Define aquí tu lista de modelos
MODELOS = [
    # "gemma3:1b",
    # "llama3.2:1b",
    # "qwen3:1.7b",
    "gemma3:4b",
    # "llama3.1:8b",
    # "qwen3:8b"
]

# 2. Leemos automáticamente todos los prompts de la carpeta
CARPETA_PROMPTS = "../../prompts/interspeech"
lista_prompts = glob.glob(f"{CARPETA_PROMPTS}/*.md") + glob.glob(f"{CARPETA_PROMPTS}/*.txt")

if not lista_prompts:
    print(f"⚠️ Error: No se encontraron archivos .md o .txt en la carpeta '{CARPETA_PROMPTS}/'.")
    exit()

print(f"🤖 Modelos cargados: {len(MODELOS)}")
print(f"📄 Prompts encontrados: {len(lista_prompts)} -> {lista_prompts}")
print(f"🔄 Se ejecutarán un total de {len(MODELOS) * len(lista_prompts)} experimentos.\n")


# Configuración de paralelismo
MAX_WORKERS = 4  # Con un M4 y 24GB, 4 o 5 es el "punto dulce" para no saturar la memoria

# 3. Mapeo para las Matrices de Confusión 
# IMPORTANTE: Reemplaza los valores de la derecha por el nombre EXACTO de la columna en tu CSV original.
MAPEO_COLUMNAS_REALES = {
    "modelo_lenguaje_sexista_codigo": "lenguaje_sexista", 
    "modelo_masc_generico_codigo": "masc_generico",
    "modelo_hombre_denominar_humanidad_codigo": "hombre_denominar_humanidad",
    "modelo_uso_dual_zorr_codigo": "uso_dual_zorr",
    "modelo_uso_cargo_mujer_codigo": "uso_cargo_mujer",
    "modelo_sexismo_social_codigo": "sexismo_social",
    "modelo_androcentrismo_codigo": "androcentrismo",
    "modelo_mencion_nombre_investigadora_codigo": "mencion_nombre_investigadora",
    "modelo_asimetria_mujer_hombre_codigo": "asimetria_mujer_hombre",
    "modelo_infatilizacion_codigo": "infantilizacion",
    "modelo_denominacion_sexualizada_codigo": "denominacion_sexualizada",
    "modelo_denominacion_redundante_codigo": "denominacion_redundante",
    "modelo_denominacion_dependiente_codigo": "denominacion_dependiente",
    "modelo_criterios_excepcion_codigo": "criterios_excepcion",
    "modelo_comparacion_mujer_hombre_codigo": "comparacion_mujer_hombre"
}

# ==========================================
# 1. CONFIGURACIÓN Y CARGA DE DATOS
# ==========================================

user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
config = Config()
config.browser_user_agent = user_agent
config.request_timeout = 10

SUFFIX = "scrape"
ruta_archivo = f"../../../data/2026_02_10_imio_def_todo_envio_heidy.xlsx - 2026_02_09_imio_def_todo_clara_{SUFFIX}.csv"

print("Cargando datos...")
try:
    data = pd.read_csv(ruta_archivo)
except FileNotFoundError:
    print("No se encuentra el archivo, usando ruta alternativa de prueba o generando dummy data.")
    exit()

df_2024 = data[data['año'] == 2024].copy()

n_samples = min(1000, len(df_2024))
df_procesar = df_2024.sample(n=n_samples, random_state=42)

print(f"Procesando {len(df_procesar)} noticias del año 2024...")

# ==========================================
# 2. FUNCIÓN DE PROCESAMIENTO
# ==========================================

def procesar_fila(row, modelo_actual, prompt_actual):
    """
    Descarga la noticia y aplica las variables. Captura errores individuales
    para evitar que el script falle.
    """
    url = row.get('no_Pagina_url', '')
    if not isinstance(url, str) or len(url) < 5:
        return {'modelo_error_descarga': 'URL inválida', 'modelo_fallo': True}
    
    resultados = {}
    errores_variables = [] # Lista para capturar qué variables fallan
    
    # articulo = Article(url, config=config)
    # try:
    #     articulo.download()
    #     articulo.parse()
    # except Exception as e:
    #     resultados['modelo_error_descarga'] = str(e)
    #     resultados['modelo_fallo'] = True
    #     return resultados

    # texto = articulo.text or ""
    texto = row.get('contenido_articulo', '')

    # --- LISTA DE FUNCIONES A EVALUAR ---
    # Esto hace el código más limpio y fácil de proteger con try-except
    funciones_variables = [
        ("lenguaje_sexista", variables.clasificar_var_lenguaje_sexista),
        ("masc_generico", variables.clasificar_var_masc_generico),
        ("hombre_denominar_humanidad", variables.clasificar_var_hombre_denominar_humanidad),
        ("uso_dual_zorr", variables.clasificar_var_uso_dual_zorr),
        ("uso_cargo_mujer", variables.clasificar_var_uso_cargo_mujer),
        ("sexismo_social", variables.clasificar_var_sexismo_discurso),
        ("androcentrismo", variables.clasificar_var_androcentrismo),
        ("mencion_nombre_investigadora", variables.clasificar_var_mencion_nombre_investigadora),
        ("asimetria_mujer_hombre", variables.clasificar_var_asimetria_mujer_hombre),
        ("infatilizacion", variables.clasificar_var_diminutivos_infantilizacion),
        ("denominacion_sexualizada", variables.clasificar_var_denominacion_sexualizada),
        ("denominacion_redundante", variables.clasificar_var_denominacion_redundante),
        ("denominacion_dependiente", variables.clasificar_var_denominacion_dependiente),
        ("criterios_excepcion", variables.clasificar_var_criterios_excepcion),
        ("comparacion_mujer_hombre", variables.clasificar_var_comparacion_mujer_hombre)
    ]

    # --- APLICACIÓN DE VARIABLES CON MANEJO DE ERRORES ---
    for nombre_var, funcion in funciones_variables:
        try:
            res = funcion(texto_articulo=texto, ruta_json="variables.json", ruta_template=prompt_actual, modelo=modelo_actual)
            resultados[f'modelo_{nombre_var}_codigo'] = res.codigo
            resultados[f'modelo_{nombre_var}_explicacion'] = res.explicacion
            resultados[f'modelo_{nombre_var}_evidencias'] = res.evidencias
        except Exception as e:
            # Si el modelo falla, guardamos el error y continuamos con la siguiente variable
            error_msg = str(e).replace('\n', ' ')
            errores_variables.append(f"{nombre_var}: {error_msg}")
            resultados[f'modelo_{nombre_var}_codigo'] = "ERROR"
            resultados[f'modelo_{nombre_var}_explicacion'] = "Fallo en la inferencia."
            resultados[f'modelo_{nombre_var}_evidencias'] = []

    # Guardamos si hubo algún fallo y el detalle para cruzarlo con el ID de la fila original
    if errores_variables:
        resultados['modelo_fallo'] = True
        resultados['modelo_errores'] = " | ".join(errores_variables)
    else:
        resultados['modelo_fallo'] = False
        resultados['modelo_errores'] = ""

    return resultados

# ==========================================
# 3. FUNCIÓN PARA MATRICES DE CONFUSIÓN
# ==========================================

def guardar_matrices_confusion(archivo_resultados, nombre_exp, carpeta_salida):
    """
    Lee los resultados, compara con la columna real y guarda matrices de confusión en CSV.
    """
    try:
        df = pd.read_csv(archivo_resultados)
    except Exception as e:
        print(f"⚠️ No se pudo leer {archivo_resultados} para generar matrices: {e}")
        return

    for col_predicha, col_real in MAPEO_COLUMNAS_REALES.items():
        if col_predicha in df.columns and col_real in df.columns:
            # Limpiamos NaNs para que salgan en la matriz si existen
            y_real = df[col_real].fillna("N/A").astype(str)
            y_pred = df[col_predicha].fillna("N/A").astype(str)
            
            # Crear matriz cruzada
            matriz = pd.crosstab(y_real, y_pred, rownames=['Real'], colnames=['Predicho'])
            
            # Guardar a CSV
            ruta_matriz = f"{carpeta_salida}/CM_{nombre_exp}_{col_predicha}.csv"
            matriz.to_csv(ruta_matriz)
        else:
            # Si la columna real no existe en el DataFrame original, saltamos
            pass

# ==========================================
# 4. PARALLEL
# ==========================================

# Lock para escribir en el archivo de forma segura
csv_lock = threading.Lock()

def tarea_hilo(row, modelo_actual, prompt_actual, nombre_output):
    """Encapsula el proceso de una fila para el ThreadPool"""
    start_time = time.time()
    res_fila = procesar_fila(row, modelo_actual, prompt_actual)
    duration = time.time() - start_time
    
    fila_completa = row.to_dict()
    fila_completa.update(res_fila)
    fila_completa['modelo_tiempo_procesamiento_seg'] = duration
    
    df_temp = pd.DataFrame([fila_completa])
    
    # Escritura segura con Lock
    with csv_lock:
        header_necesario = not os.path.exists(nombre_output)
        df_temp.to_csv(nombre_output, index=False, mode='a', header=header_necesario, encoding='utf-8')


# ==========================================
# 5. BUCLE PRINCIPAL (GRID SEARCH)
# ==========================================

FOLDER_RESULTS = "../../results"
FOLDER_METRICS = "../../metrics"
FOLDER_METRICS_DETAILS = "../../metrics/details"
FOLDER_MATRICES = "../../metrics/confusion_matrices" # <-- Nueva carpeta

os.makedirs(FOLDER_RESULTS, exist_ok=True)
os.makedirs(FOLDER_METRICS, exist_ok=True)
os.makedirs(FOLDER_METRICS_DETAILS, exist_ok=True)
os.makedirs(FOLDER_MATRICES, exist_ok=True) # <-- Aseguramos que exista

for modelo_actual in MODELOS:
    for prompt_actual in lista_prompts:
        
        modelo_limpio = modelo_actual.replace(':', '_').replace('-', '_').replace('/', '_')
        prompt_limpio = os.path.splitext(os.path.basename(prompt_actual))[0]
        nombre_exp = f"Experimento-Interspeech-{modelo_limpio}_{prompt_limpio}"

        print(f"\n{'='*60}")
        print(f"🚀 INICIANDO EXPERIMENTO: {nombre_exp}")
        print(f"\n🚀 EJECUTANDO EN PARALELO ({MAX_WORKERS} hilos): {nombre_exp}")
        print(f"   🤖 Modelo: {modelo_actual}")
        print(f"   📄 Prompt: {prompt_actual}")
        print(f"{'='*60}")

        nombre_output = f"{FOLDER_RESULTS}/{nombre_exp}_resultados_2024_{SUFFIX}.csv"
        archivo_metricas = f"{FOLDER_METRICS_DETAILS}/metrics_{nombre_exp}_2024.csv"

        es_primera_vez = True

        print(f"Los datos se guardarán en tiempo real en: {nombre_output}")

       # Ejecución en paralelo de las filas del DataFrame
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Creamos la lista de tareas
            futures = [
                executor.submit(tarea_hilo, row, modelo_actual, prompt_actual, nombre_output) 
                for _, row in df_procesar.iterrows()
            ]
            
            # tqdm envuelve as_completed para ver el progreso real
            for _ in tqdm(as_completed(futures), total=len(futures), desc=f"Procesando {prompt_limpio}"):
                pass

        # Métricas al finalizar el experimento
        print(f"📊 Generando métricas para {nombre_exp}...")
        generar_metricas_y_summary(
            archivo_input=nombre_output,
            archivo_output=f"{FOLDER_METRICS_DETAILS}/metrics_{nombre_exp}_2024.csv",
            nombre_experimento=nombre_exp,
            modelo_usado=modelo_actual,
            prompt_file=prompt_actual
        )

        # 4.2 GENERAR MATRICES DE CONFUSIÓN
        print("📊 Generando matrices de confusión...")
        guardar_matrices_confusion(nombre_output, nombre_exp, FOLDER_MATRICES)

print("\n🎉 TODOS LOS EXPERIMENTOS DEL GRID SEARCH HAN FINALIZADO.")