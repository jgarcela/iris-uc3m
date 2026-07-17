"""
Experimento 18 adaptado para el cluster TSC (fase CLIENTE).

Mismas 5 variables de SEXISMO que el experimento_13 (lenguaje sexista, masc
genérico, sexismo discurso, asimetría mujer/hombre, denominación sexualizada),
pero el experimento_18 compara MODELOS DE API además del Ollama local. Como
`utils.consultar_ollama` enruta por el ID del modelo, el mismo runner sirve para:
  - OpenAI  (gpt-4o-mini, gpt-4.1, gpt-5-mini, ...) -> OPENAI_API_KEY
  - Anthropic (claude-*)                            -> ANTHROPIC_API_KEY
  - Gemini  (gemini-2.5-*)                          -> GEMINI_API_KEY / GOOGLE_API_KEY
  - Ollama local (gemma4:e4b, ...)                  -> OLLAMA_HOST

Cada variable devuelve código + explicación + evidencias (columnas
'modelo_<var>', '_explicacion', '_evidencias'). Se reparte el dataset en shards
y se paraleliza con --workers.

Ejemplo (API OpenAI, sin servidor Ollama):
    OPENAI_API_KEY=sk-... \
    python main_cluster.py --model gpt-4o-mini --shard 0 --n-shards 1 --workers 8 \
      --experimentos-dir /ruta/Experimentos --data /ruta/...scrape.csv

Ejemplo (Ollama local en la granja):
    OLLAMA_HOST=bastet07:11434 \
    python main_cluster.py --model gemma4:e4b --shard 0 --n-shards 2 --workers 4 \
      --experimentos-dir /ruta/Experimentos --data /ruta/...scrape.csv
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
    p = argparse.ArgumentParser(description="Experimento 18 - cliente cluster (sharded, multi-proveedor)")
    p.add_argument("--experimentos-dir", default=os.environ.get("EXPERIMENTOS_DIR"),
                   help="Ruta a la carpeta 'Experimentos' (con variables.py, utils.py, variables.json, prompts/). Env: EXPERIMENTOS_DIR")
    p.add_argument("--data", default=os.environ.get("DATA_CSV"),
                   help="Ruta al CSV de datos (ya scrapeado). Env: DATA_CSV")
    p.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "./results"),
                   help="Carpeta de salida de resultados. Env: OUTPUT_DIR")
    p.add_argument("--model", default=os.environ.get("MODELO", "gpt-4o-mini"),
                   help="Modelo a usar (OpenAI/Claude/Gemini/Ollama). Env: MODELO")
    p.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST"),
                   help="host:puerto del servidor Ollama (solo para modelos locales). Env: OLLAMA_HOST")
    p.add_argument("--shard", type=int, default=int(os.environ.get("SHARD", 0)),
                   help="Índice de este shard (0-based). Env: SHARD")
    p.add_argument("--n-shards", type=int, default=int(os.environ.get("N_SHARDS", 1)),
                   help="Número total de shards. Env: N_SHARDS")
    p.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", 8)),
                   help="Peticiones concurrentes DENTRO de este shard. Env: WORKERS "
                        "(8 va bien para APIs; baja a ~4 para Ollama local)")
    p.add_argument("--limit", type=int, default=int(os.environ.get("LIMIT", 0)) or None,
                   help="Procesa como mucho N artículos de este shard (prueba rápida). Env: LIMIT")
    p.add_argument("--year", type=int, default=int(os.environ.get("YEAR", 0)),
                   help="Filtra por año (ej. 2024). 0 = TODOS los años. Env: YEAR")
    p.add_argument("--n-samples", type=int, default=int(os.environ.get("N_SAMPLES", 0)),
                   help="Muestrea N artículos (random_state=42). 0 = TODA la base sin muestreo. Env: N_SAMPLES")
    p.add_argument("--only-labeled", action="store_true",
                   default=os.environ.get("ONLY_LABELED", "").lower() in ("1", "true", "yes"),
                   help="Procesa SOLO artículos con etiqueta real en las 5 variables (para poder medir). Env: ONLY_LABELED=1")
    p.add_argument("--variables-json", default=os.environ.get("VARIABLES_JSON"),
                   help="Ruta a un variables.json alternativo (p. ej. recalibrado). "
                        "Por defecto usa <experimentos-dir>/variables.json. Env: VARIABLES_JSON")
    p.add_argument("--vars", default=os.environ.get("VARS", ""),
                   help="Lista separada por comas de variables a procesar. Vacío = las 5. "
                        "Opciones: lenguaje_sexista,masc_generico,sexismo_discurso,"
                        "asimetria_mujer_hombre,denominacion_sexualizada. Env: VARS")
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
RUTA_VARIABLES_JSON = os.path.abspath(args.variables_json) if args.variables_json \
    else os.path.join(EXPERIMENTOS_DIR, "variables.json")
RUTA_TEMPLATE = os.path.join(EXPERIMENTOS_DIR, "prompts", "prompt_clara.md")

import variables  # noqa: E402  (import tras ajustar sys.path)
import utils       # noqa: E402  (se importa para que variables lo tenga disponible)
import ollama      # noqa: E402  (para instrumentar el tiempo real de inferencia local)

# --- Instrumentación: tiempo REAL de inferencia por artículo (solo Ollama) ---
# Ollama devuelve 'total_duration' (ns) por petición. Para modelos de API esta
# métrica no aplica (queda en 0) y el tiempo de pared/latencia se mide igual.
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

# Selección de variables a procesar (--vars). Vacío = las 5, en su orden canónico.
_ORDEN_VARS = ["lenguaje_sexista", "masc_generico", "sexismo_discurso",
               "asimetria_mujer_hombre", "denominacion_sexualizada"]
if args.vars.strip():
    pedidas = [v.strip() for v in args.vars.split(",") if v.strip()]
    desconocidas = [v for v in pedidas if v not in _ORDEN_VARS]
    if desconocidas:
        sys.exit(f"ERROR: variables no reconocidas en --vars: {desconocidas}. Válidas: {_ORDEN_VARS}")
    VARS_A_PROCESAR = [v for v in _ORDEN_VARS if v in pedidas]  # respeta orden canónico
else:
    VARS_A_PROCESAR = list(_ORDEN_VARS)

print(f"🤖 Modelo      : {MODELO}")
print(f"🎯 Variables   : {', '.join(VARS_A_PROCESAR)}")
print(f"🖥️  OLLAMA_HOST : {os.environ.get('OLLAMA_HOST', '(N/A para API; localhost:11434 para local)')}")
print(f"🔀 Shard       : {args.shard} / {args.n_shards}   (workers={args.workers})")
print(f"📂 Experimentos: {EXPERIMENTOS_DIR}")
print(f"📑 variables   : {RUTA_VARIABLES_JSON}")
print(f"📄 Datos       : {args.data}")


# ==========================================
# 1. SALIDA (un fichero por shard, para no pisarse entre clientes)
# ==========================================
os.makedirs(args.output_dir, exist_ok=True)
PREFFIX = "18-Experimento-18_03_2026"
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

# Filtro de año opcional (0 = todos los años -> toda la base, >7k artículos).
if args.year:
    df_base = data[data["año"] == args.year].copy()
    print(f"-> Filtrando año {args.year}: {len(df_base)} artículos.")
else:
    df_base = data.copy()
    print(f"-> Sin filtro de año: {len(df_base)} artículos (toda la base).")

# Solo artículos con etiqueta real (ground truth) en las 5 variables de sexismo.
# Útil para medir métricas sin gastar en artículos no anotados.
_VARS_GT = ["lenguaje_sexista", "masc_generico", "sexismo_discurso",
            "asimetria_mujer_hombre", "denominacion_sexualizada"]
if args.only_labeled:
    presentes = [c for c in _VARS_GT if c in df_base.columns]
    antes = len(df_base)
    df_base = df_base[df_base[presentes].notna().any(axis=1)].copy()
    print(f"-> --only-labeled: {len(df_base)} artículos etiquetados (de {antes}).")

# Muestreo determinista opcional (0 = sin muestreo -> se procesa toda la base).
if args.n_samples and args.n_samples < len(df_base):
    df_procesar = df_base.sample(n=args.n_samples, random_state=42).reset_index(drop=True)
    print(f"-> Muestreo determinista (random_state=42): {len(df_procesar)} artículos.")
else:
    df_procesar = df_base.reset_index(drop=True)

# Reparto en shards de forma determinista: cada fila va al shard (i % n_shards).
total_dataset = len(df_procesar)
df_procesar = df_procesar[df_procesar.index % args.n_shards == args.shard].copy()
print(f"-> A este shard le tocan {len(df_procesar)} artículos de {total_dataset}.")

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
# 3. FUNCIÓN DE PROCESAMIENTO (5 variables de sexismo)
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


# Mapa variable -> función clasificadora. Permite seleccionar cuáles procesar (--vars).
_CLASIFICADORES = {
    "lenguaje_sexista": variables.clasificar_var_lenguaje_sexista,
    "masc_generico": variables.clasificar_var_masc_generico,
    "sexismo_discurso": variables.clasificar_var_sexismo_discurso,
    "asimetria_mujer_hombre": variables.clasificar_var_asimetria_mujer_hombre,
    "denominacion_sexualizada": variables.clasificar_var_denominacion_sexualizada,
}


def procesar_fila(row):
    resultados = {}

    texto = str(row["contenido_articulo"]) if pd.notna(row["contenido_articulo"]) else ""

    for nombre in VARS_A_PROCESAR:
        clasificador = _CLASIFICADORES[nombre]
        res = clasificador(
            texto, ruta_json=RUTA_VARIABLES_JSON, ruta_template=RUTA_TEMPLATE, modelo=MODELO)
        resultados.update(_expandir_resultado(res, f"modelo_{nombre}"))

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
    _tls.model_ns = 0
    res_fila = procesar_fila(row)
    duration = time.time() - start
    model_real = getattr(_tls, "model_ns", 0) / 1e9
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
print(f"  Modelo               : {MODELO}")
print(f"  Artículos procesados : {n}")
print(f"  Workers              : {args.workers}")
print(f"  Tiempo de pared      : {_wall/60:.1f} min ({_wall:.0f} s)")
if n:
    print(f"  Throughput real      : {_wall/n:.1f} s/artículo  |  {n/(_wall/3600):.0f} artículos/hora")
print("  (para Ollama local, tiempo REAL de inferencia -> 'modelo_tiempo_modelo_real_seg')")
print(f"\nProceso finalizado. Archivo completado: {nombre_output}")
