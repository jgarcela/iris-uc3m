"""
Experimento 12 adaptado para el cluster TSC (fase CLIENTE).

Ataca a un servidor Ollama levantado por launch_process.py (llm.json).
Pensado para lanzarse una vez por servidor/GPU, repartiendo el dataset en shards.

Todo se configura por variables de entorno o argumentos CLI (ver --help).

Ejemplo (un cliente contra un servidor):
    OLLAMA_HOST=bastet07:11434 \
    EXPERIMENTOS_DIR=/export/usuarios01/jggomez/iris/Experimentos \
    DATA_CSV=/export/usuarios01/jggomez/iris/data/2026_02_10_..._scrape.csv \
    python main_cluster.py --shard 0 --n-shards 4 --workers 4

Para 4 servidores lanzarías 4 procesos, cambiando OLLAMA_HOST y --shard (0..3).
"""

import argparse
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm


# ==========================================
# 0. CONFIGURACIÓN (env + CLI)
# ==========================================
def parse_args():
    p = argparse.ArgumentParser(description="Experimento 12 - cliente cluster (sharded)")
    p.add_argument("--experimentos-dir", default=os.environ.get("EXPERIMENTOS_DIR"),
                   help="Ruta a la carpeta 'Experimentos' (con variables.py, utils.py, variables.json, prompts/). Env: EXPERIMENTOS_DIR")
    p.add_argument("--data", default=os.environ.get("DATA_CSV"),
                   help="Ruta al CSV de datos (ya scrapeado). Env: DATA_CSV")
    p.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "./results"),
                   help="Carpeta de salida de resultados. Env: OUTPUT_DIR")
    p.add_argument("--model", default=os.environ.get("MODELO", "gemma4:e4b"),
                   help="Modelo de Ollama. Env: MODELO")
    p.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST"),
                   help="host:puerto del servidor Ollama (ej. bastet07:11434). Env: OLLAMA_HOST")
    p.add_argument("--shard", type=int, default=int(os.environ.get("SHARD", 0)),
                   help="Índice de este shard (0-based). Env: SHARD")
    p.add_argument("--n-shards", type=int, default=int(os.environ.get("N_SHARDS", 1)),
                   help="Número total de shards (= nº de servidores). Env: N_SHARDS")
    p.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", 1)),
                   help="Peticiones concurrentes DENTRO de este shard/servidor. Env: WORKERS")
    p.add_argument("--limit", type=int, default=int(os.environ.get("LIMIT", 0)) or None,
                   help="Procesa como mucho N artículos de este shard (para pruebas rápidas). Env: LIMIT")
    return p.parse_args()


args = parse_args()

if not args.experimentos_dir:
    sys.exit("ERROR: define --experimentos-dir o la variable de entorno EXPERIMENTOS_DIR.")
if not args.data:
    sys.exit("ERROR: define --data o la variable de entorno DATA_CSV.")
if not (0 <= args.shard < args.n_shards):
    sys.exit(f"ERROR: --shard ({args.shard}) debe estar en [0, {args.n_shards}).")

# El cliente Ollama lee OLLAMA_HOST del entorno; lo fijamos aquí por si vino por CLI.
if args.ollama_host:
    os.environ["OLLAMA_HOST"] = args.ollama_host

EXPERIMENTOS_DIR = os.path.abspath(args.experimentos_dir)
sys.path.append(EXPERIMENTOS_DIR)
RUTA_VARIABLES_JSON = os.path.join(EXPERIMENTOS_DIR, "variables.json")
RUTA_TEMPLATE = os.path.join(EXPERIMENTOS_DIR, "prompts", "prompt_clara.md")

import variables  # noqa: E402  (import tras ajustar sys.path)
import utils       # noqa: E402  (se importa para que variables lo tenga disponible)
import ollama      # noqa: E402  (para instrumentar el tiempo real de inferencia)

# --- Instrumentación: tiempo REAL de inferencia por artículo ---
# Ollama devuelve 'total_duration' (ns) = tiempo que el servidor dedica a CADA
# petición, SIN incluir la espera en cola. Envolvemos ollama.chat para acumular
# ese tiempo por hilo; así medimos el cómputo real de cada artículo aunque haya
# concurrencia (--workers > 1), separándolo de la latencia (que sí incluye cola).
_tls = threading.local()
_orig_ollama_chat = ollama.chat

def _timed_chat(*a, **kw):
    resp = _orig_ollama_chat(*a, **kw)
    try:
        dur = resp.get("total_duration") if isinstance(resp, dict) else getattr(resp, "total_duration", None)
    except Exception:
        dur = None
    if dur:
        _tls.model_ns = getattr(_tls, "model_ns", 0) + dur
    return resp

ollama.chat = _timed_chat  # utils.consultar_ollama llama a ollama.chat -> queda instrumentado

MODELO = args.model
COLUMNA_ID = "IdNoticia"

print(f"🖥️  OLLAMA_HOST : {os.environ.get('OLLAMA_HOST', '(por defecto localhost:11434)')}")
print(f"🤖 Modelo      : {MODELO}")
print(f"🔀 Shard       : {args.shard} / {args.n_shards}   (workers={args.workers})")
print(f"📂 Experimentos: {EXPERIMENTOS_DIR}")
print(f"📄 Datos       : {args.data}")


# ==========================================
# 1. SALIDA (un fichero por shard, para no pisarse entre clientes)
# ==========================================
os.makedirs(args.output_dir, exist_ok=True)
PREFFIX = "12-Experimento-12_03_2026"
nombre_output = os.path.join(
    args.output_dir,
    f"{PREFFIX}_resultados_modelo_2024_scrape_shard{args.shard}de{args.n_shards}.csv",
)


# ==========================================
# 2. CARGA DE DATOS + SELECCIÓN DE SHARD + REANUDACIÓN
# ==========================================
print("Cargando datos originales...")
try:
    data = pd.read_csv(args.data)
except FileNotFoundError:
    sys.exit(f"No se encuentra el archivo de datos: {args.data}")

# Mismo muestreo determinista que el experimento original (random_state=42).
df_2024 = data[data["año"] == 2024].copy()
n_samples = min(1000, len(df_2024))
df_procesar = df_2024.sample(n=n_samples, random_state=42).reset_index(drop=True)

# Reparto en shards de forma determinista: cada fila va al shard (i % n_shards).
df_procesar = df_procesar[df_procesar.index % args.n_shards == args.shard].copy()
print(f"-> A este shard le tocan {len(df_procesar)} artículos de {n_samples}.")

# Reanudación: saltar IDs ya guardados en el CSV de ESTE shard.
ids_procesados = set()
es_primera_vez = True
if os.path.exists(nombre_output):
    try:
        df_salida = pd.read_csv(nombre_output)
        if COLUMNA_ID in df_salida.columns:
            ids_procesados = set(df_salida[COLUMNA_ID].dropna().astype(str).unique())
            es_primera_vez = False
            print(f"-> Archivo previo detectado: {len(ids_procesados)} ya procesados.")
    except pd.errors.EmptyDataError:
        print("-> El archivo de salida existe pero está vacío. Empezando de cero.")

total_antes = len(df_procesar)
df_procesar = df_procesar[~df_procesar[COLUMNA_ID].astype(str).isin(ids_procesados)]
print(f"Quedan {len(df_procesar)} por procesar (se omitieron {total_antes - len(df_procesar)}).")

if args.limit:
    df_procesar = df_procesar.head(args.limit)
    print(f"-> LIMIT activo: solo se procesarán {len(df_procesar)} (prueba rápida).")

if df_procesar.empty:
    print("Nada que procesar en este shard. Saliendo.")
    sys.exit(0)


# ==========================================
# 3. FUNCIÓN DE PROCESAMIENTO (idéntica al exp12, con MODELO parametrizado)
# ==========================================
def procesar_fila(row):
    resultados = {}

    titulo = str(row["Titular"]) if pd.notna(row["Titular"]) else ""
    texto = str(row["contenido_articulo"]) if pd.notna(row["contenido_articulo"]) else ""
    authors = str(row["no_Autor"]) if pd.notna(row["no_Autor"]) else ""

    # 7a. Nombre Propio Titular (Lista)
    np_titular = variables.clasificar_var_nombre_propio_titular_list_e1(titulo=titulo, modelo=MODELO)
    resultados["modelo_nombre_propio_titular_nombres"] = str(np_titular.nombres)
    resultados["modelo_nombre_propio_titular_valores"] = str(np_titular.valores)

    # 7b. Género Nombre Propio Titular
    resultados["modelo_nombre_propio_titular"] = variables.clasificar_var_nombre_propio_titular(np_titular.valores)

    # 9a. Protagonistas Cuerpo
    protas = variables.clasificar_var_cla_genero_prota_list_e1(texto_noticia=texto, modelo=MODELO)
    resultados["modelo_cla_genero_prota_nombres"] = str(protas.nombres)
    resultados["modelo_cla_genero_prota_valores"] = str(protas.valores)

    # 9b. Género Protagonistas
    resultados["modelo_cla_genero_prota"] = variables.clasificar_var_cla_genero_prota(protas.valores)

    # 10. Periodista
    nombre_periodista = variables.clasificar_var_nombre_periodista_authors(authors)
    resultados["modelo_nombre_periodista"] = nombre_periodista

    # 11. Género Periodista (Autoría)
    nombre_medio = resultados.get("modelo_Medio_nombre", "Desconocido")
    resultados["modelo_genero_periodista"] = variables.clasificar_var_genero_periodista_e1(
        nombre_periodista=nombre_periodista, nombre_medio=nombre_medio, modelo=MODELO
    )

    return resultados


# ==========================================
# 4. BUCLE PRINCIPAL (concurrencia opcional + guardado incremental con lock)
# ==========================================
print(f"Los datos se guardarán en tiempo real en: {nombre_output}")

_write_lock = threading.Lock()
_state = {"primera": es_primera_vez}


def _guardar(fila_completa):
    df_temp = pd.DataFrame([fila_completa])
    with _write_lock:
        if _state["primera"]:
            df_temp.to_csv(nombre_output, index=False, mode="w", encoding="utf-8")
            _state["primera"] = False
        else:
            df_temp.to_csv(nombre_output, index=False, mode="a", header=False, encoding="utf-8")


def _trabajo(row):
    start = time.time()
    _tls.model_ns = 0                      # reinicia el acumulador de este hilo
    res_fila = procesar_fila(row)
    duration = time.time() - start         # latencia (incluye espera en cola si hay workers>1)
    model_real = getattr(_tls, "model_ns", 0) / 1e9   # cómputo real del modelo (sin cola)
    fila_completa = row.to_dict()
    fila_completa.update(res_fila)
    fila_completa["modelo_tiempo_procesamiento_seg"] = duration
    fila_completa["modelo_tiempo_modelo_real_seg"] = model_real
    _guardar(fila_completa)


filas = [row for _, row in df_procesar.iterrows()]

_wall0 = time.time()
if args.workers <= 1:
    for row in tqdm(filas, total=len(filas)):
        _trabajo(row)
else:
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futuros = [ex.submit(_trabajo, row) for row in filas]
        for _ in tqdm(as_completed(futuros), total=len(futuros)):
            pass
_wall = time.time() - _wall0

# --- Resumen de throughput (tiempo de pared real de este shard) ---
n = len(filas)
print("\n" + "=" * 50)
print(" RESUMEN DE TIEMPOS (este shard)")
print("=" * 50)
print(f"  Artículos procesados : {n}")
print(f"  Workers              : {args.workers}")
print(f"  Tiempo de pared      : {_wall/60:.1f} min ({_wall:.0f} s)")
if n:
    print(f"  Throughput real      : {_wall/n:.1f} s/artículo  |  {n/(_wall/3600):.0f} artículos/hora")
print("  (tiempo REAL de inferencia por artículo -> columna 'modelo_tiempo_modelo_real_seg')")
print(f"\nProceso finalizado. Archivo completado: {nombre_output}")
