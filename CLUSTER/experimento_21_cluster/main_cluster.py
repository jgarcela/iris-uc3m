"""
Experimento 21 (Agent Skills) adaptado al cluster TSC — fase CLIENTE.

Corre la arquitectura de agentes especializados del Experimento 21 sobre la
granja Ollama, con las mismas 5 variables de sexismo. Dos niveles seleccionables:

  · B1 (por defecto)  → Agent Skills: progressive disclosure + herramientas.
  · B0 (--baseline)   → metodología inyectada en el prompt, sin herramientas.

Reutiliza el esqueleto sharded del experimento_18_bis (shards, --workers,
reanudación por IdNoticia, OLLAMA_HOST, instrumentación de tiempo real de Ollama)
pero llama al código de agentes de experimento_21_agentskills. La salida usa el
mismo esquema (modelo_<var>, _explicacion, _evidencias) para que metrics.py del
Exp 21 funcione sin cambios; añade columnas de traza del agente.

Prompt caching desactivado (config canónica del benchmark; Ollama lo ignora igual).

Ejemplo (granja Ollama, B1 con skills):
    OLLAMA_HOST=bastet07:11434 \
    python main_cluster.py --model gemma4:e4b --shard 0 --n-shards 2 --workers 4 \
      --experimentos-dir /ruta/Experimentos \
      --agente-dir /ruta/Experimentos/experiments/experimento_21_agentskills \
      --data /ruta/...scrape.csv --only-labeled

Ejemplo (B0 baseline sin skills):
    OLLAMA_HOST=bastet07:11434 \
    python main_cluster.py --model gemma4:e4b --baseline --shard 0 --n-shards 2 \
      --workers 4 --experimentos-dir /ruta/Experimentos \
      --agente-dir /ruta/Experimentos/experiments/experimento_21_agentskills \
      --data /ruta/...scrape.csv --only-labeled
"""

import argparse
import json
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
    p = argparse.ArgumentParser(description="Experimento 21 - cliente cluster (Agent Skills, sharded)")
    p.add_argument("--experimentos-dir", default=os.environ.get("EXPERIMENTOS_DIR"),
                   help="Carpeta 'Experimentos' (utils.py, variables.json, prompts/, methodology/). Env: EXPERIMENTOS_DIR")
    p.add_argument("--agente-dir", default=os.environ.get("AGENTE_DIR"),
                   help="Carpeta experimento_21_agentskills (agente.py, tools.py, guias.py, skills/). Env: AGENTE_DIR")
    p.add_argument("--data", default=os.environ.get("DATA_CSV"),
                   help="CSV del corpus (con contenido_articulo e IdNoticia). Env: DATA_CSV")
    p.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "./results"),
                   help="Carpeta de salida. Env: OUTPUT_DIR")
    p.add_argument("--model", default=os.environ.get("MODELO", "gemma4:e4b"),
                   help="Modelo (Ollama en la granja). Env: MODELO")
    p.add_argument("--baseline", action="store_true",
                   default=os.environ.get("BASELINE", "").lower() in ("1", "true", "yes"),
                   help="Nivel B0: metodología inyectada, sin skills. Por defecto B1. Env: BASELINE")
    p.add_argument("--sin-resumenes-guias", action="store_true",
                   default=os.environ.get("SIN_RESUMENES_GUIAS", "").lower() in ("1", "true", "yes"),
                   help="Ablación B1: no listar las skills-resumen de guías. Env: SIN_RESUMENES_GUIAS")
    p.add_argument("--sin-consultar-guia", action="store_true",
                   default=os.environ.get("SIN_CONSULTAR_GUIA", "").lower() in ("1", "true", "yes"),
                   help="Ablación B1: desactivar la tool RAG en vivo CONSULTAR_GUIA. Env: SIN_CONSULTAR_GUIA")
    p.add_argument("--ollama-host", default=os.environ.get("OLLAMA_HOST"),
                   help="host:puerto del servidor Ollama de la granja. Env: OLLAMA_HOST")
    p.add_argument("--shard", type=int, default=int(os.environ.get("SHARD", 0)),
                   help="Índice de este shard (0-based). Env: SHARD")
    p.add_argument("--n-shards", type=int, default=int(os.environ.get("N_SHARDS", 1)),
                   help="Número total de shards. Env: N_SHARDS")
    p.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", 4)),
                   help="Peticiones concurrentes dentro de este shard. Env: WORKERS")
    p.add_argument("--limit", type=int, default=int(os.environ.get("LIMIT", 0)) or None,
                   help="Procesa como mucho N artículos (prueba rápida). Env: LIMIT")
    p.add_argument("--year", type=int, default=int(os.environ.get("YEAR", 0)),
                   help="Filtra por año (0 = todos). Env: YEAR")
    p.add_argument("--only-labeled", action="store_true",
                   default=os.environ.get("ONLY_LABELED", "").lower() in ("1", "true", "yes"),
                   help="Solo artículos con GT en las 5 variables. Env: ONLY_LABELED")
    return p.parse_args()


args = parse_args()

if not args.experimentos_dir:
    sys.exit("ERROR: define --experimentos-dir o EXPERIMENTOS_DIR.")
if not args.agente_dir:
    sys.exit("ERROR: define --agente-dir o AGENTE_DIR (carpeta experimento_21_agentskills).")
if not args.data:
    sys.exit("ERROR: define --data o DATA_CSV.")
if not (0 <= args.shard < args.n_shards):
    sys.exit(f"ERROR: --shard ({args.shard}) debe estar en [0, {args.n_shards}).")

# El cliente Ollama lee OLLAMA_HOST del entorno; lo fijamos por si vino por CLI.
if args.ollama_host:
    os.environ["OLLAMA_HOST"] = args.ollama_host

EXPERIMENTOS_DIR = os.path.abspath(args.experimentos_dir)
AGENTE_DIR = os.path.abspath(args.agente_dir)
for _p in (EXPERIMENTOS_DIR, AGENTE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ollama       # noqa: E402  (instrumentar el tiempo real de inferencia)
import agente       # noqa: E402  (código de agentes del Exp 21)

# Config canónica del benchmark: sin prompt caching (Ollama lo ignora igualmente).
agente.USAR_PROMPT_CACHE = False

# Ablaciones B1 (solo aplican en nivel B1; en B0 no hay catálogo ni tools).
if args.sin_resumenes_guias:
    agente.INCLUIR_RESUMENES_GUIAS = False
if args.sin_consultar_guia:
    agente.HABILITAR_CONSULTAR_GUIA = False

# --- Instrumentación: tiempo REAL de inferencia por artículo (Ollama) ---
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


ollama.chat = _timed_chat  # utils.consultar_ollama -> ollama.chat queda instrumentado

MODELO = args.model
COLUMNA_ID = "IdNoticia"
NIVEL = "b0" if args.baseline else "b1"
# Sufijo de ablación para que B1-completo y B1-mínimo no compartan nombre de fichero.
if not args.baseline:
    if args.sin_resumenes_guias and args.sin_consultar_guia:
        NIVEL = "b1min"          # solo skills (sin resúmenes ni tool)
    elif args.sin_resumenes_guias:
        NIVEL = "b1sinres"
    elif args.sin_consultar_guia:
        NIVEL = "b1singuia"
_VARS = ["lenguaje_sexista", "masc_generico", "sexismo_discurso",
         "asimetria_mujer_hombre", "denominacion_sexualizada"]
_clasificar = agente.clasificar_variable_baseline if args.baseline else agente.clasificar_variable

print(f"🤖 Modelo      : {MODELO}")
_abl = (" · sin resúmenes" if args.sin_resumenes_guias else "") + \
       (" · sin CONSULTAR_GUIA" if args.sin_consultar_guia else "")
print(f"🧩 Nivel       : {NIVEL.upper()}  "
      f"({'baseline sin skills' if args.baseline else 'Agent Skills'+_abl})")
print(f"🖥️  OLLAMA_HOST : {os.environ.get('OLLAMA_HOST', '(localhost:11434)')}")
print(f"🔀 Shard       : {args.shard} / {args.n_shards}   (workers={args.workers})")
print(f"📂 Agente dir  : {AGENTE_DIR}")


# ==========================================
# 1. SALIDA (un fichero por shard)
# ==========================================
os.makedirs(args.output_dir, exist_ok=True)
PREFFIX = f"21{NIVEL}-Experimento-21_{NIVEL}_{MODELO.replace(':', '_').replace('/', '_')}"
nombre_output = os.path.join(
    args.output_dir, f"{PREFFIX}_shard{args.shard}de{args.n_shards}.csv")


# ==========================================
# 2. CARGA + SHARD + REANUDACIÓN
# ==========================================
print("Cargando datos...")
try:
    data = pd.read_csv(args.data)
except FileNotFoundError:
    sys.exit(f"No se encuentra el archivo de datos: {args.data}")

df_base = data[data["año"] == args.year].copy() if args.year else data.copy()

if args.only_labeled:
    presentes = [c for c in _VARS if c in df_base.columns]
    antes = len(df_base)
    df_base = df_base[df_base[presentes].notna().all(axis=1)].copy()
    print(f"-> --only-labeled: {len(df_base)} artículos con GT en las 5 (de {antes}).")

df_procesar = df_base.reset_index(drop=True)
total_dataset = len(df_procesar)
df_procesar = df_procesar[df_procesar.index % args.n_shards == args.shard].copy()
print(f"-> A este shard le tocan {len(df_procesar)} artículos de {total_dataset}.")

ids_procesados = set()
es_primera_vez = True
if os.path.exists(nombre_output):
    try:
        df_salida = pd.read_csv(nombre_output)
        if COLUMNA_ID in df_salida.columns:
            ids_procesados = set(df_salida[COLUMNA_ID].dropna().astype(str).unique())
            es_primera_vez = False
            print(f"-> Reanudación: {len(ids_procesados)} ya procesados.")
    except pd.errors.EmptyDataError:
        print("-> Archivo de salida vacío. Empezando de cero.")

total_antes = len(df_procesar)
df_procesar = df_procesar[~df_procesar[COLUMNA_ID].astype(str).isin(ids_procesados)]
print(f"Quedan {len(df_procesar)} por procesar (se omitieron {total_antes - len(df_procesar)}).")

if args.limit:
    df_procesar = df_procesar.head(args.limit)
    print(f"-> LIMIT activo: solo {len(df_procesar)}.")

if df_procesar.empty:
    print("Nada que procesar en este shard. Saliendo.")
    sys.exit(0)


# ==========================================
# 3. PROCESAMIENTO (5 agentes por artículo)
# ==========================================
def procesar_fila(row):
    texto = str(row["contenido_articulo"]) if pd.notna(row["contenido_articulo"]) else ""
    out = {}
    n_err = 0
    for variable in _VARS:
        res, traza = _clasificar(variable, texto, modelo=MODELO)
        out[f"modelo_{variable}"] = res["codigo"]
        out[f"modelo_{variable}_explicacion"] = res["explicacion"]
        out[f"modelo_{variable}_evidencias"] = " | ".join(res["evidencias"]) if res["evidencias"] else ""
        out[f"{variable}_n_tools"] = traza["n_tools"]
        out[f"{variable}_colapso_b0"] = int(traza["colapso_b0"])
        out[f"{variable}_error"] = traza.get("error") or ""
        if traza.get("error"):
            n_err += 1
    out["n_variables_error"] = n_err
    return out


# ==========================================
# 4. BUCLE PRINCIPAL (concurrencia + guardado incremental con lock)
# ==========================================
print(f"Guardado en tiempo real en: {nombre_output}")
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
    fila = row.to_dict()
    fila.update(res_fila)
    fila["modelo_tiempo_procesamiento_seg"] = time.time() - start
    fila["modelo_tiempo_modelo_real_seg"] = getattr(_tls, "model_ns", 0) / 1e9
    _guardar(fila)


filas = [row for _, row in df_procesar.iterrows()]

# Pre-vuelo: procesa el primer artículo en serie. Si fallan las 5 variables, el
# servidor Ollama no está sirviendo el modelo (host/puerto sin el modelo cargado):
# se aborta ANTES de escribir filas basura (todas codigo=1 / sin_final).
_pf = procesar_fila(filas[0])
if _pf.get("n_variables_error", 0) >= len(_VARS):
    sys.exit(
        f"\n❌ ABORTADO (pre-vuelo): el primer artículo falló en las {len(_VARS)} "
        f"variables. Revisa que OLLAMA_HOST={os.environ.get('OLLAMA_HOST')} sirve "
        f"'{MODELO}' (curl http://$OLLAMA_HOST/api/tags | grep {MODELO.split(':')[0]}).\n"
        "No se ha escrito nada.")
_fila0 = filas[0].to_dict(); _fila0.update(_pf)
_fila0["modelo_tiempo_procesamiento_seg"] = 0.0
_fila0["modelo_tiempo_modelo_real_seg"] = 0.0
_guardar(_fila0)
filas = filas[1:]  # el primero ya está guardado

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

n = len(filas)
print("\n" + "=" * 50)
print(f" RESUMEN — Exp 21 {NIVEL.upper()} · {MODELO} · shard {args.shard}/{args.n_shards}")
print("=" * 50)
print(f"  Artículos      : {n}")
print(f"  Tiempo de pared: {_wall/60:.1f} min")
if n:
    print(f"  Throughput     : {_wall/n:.1f} s/artículo | {n/(_wall/3600):.0f} art/hora")
print(f"\nArchivo completado: {nombre_output}")
