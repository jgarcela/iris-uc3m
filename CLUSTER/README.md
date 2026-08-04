# Ejecutar experimentos IRIS en el cluster TSC (Ollama en la granja)

Runbook completo para lanzar los análisis (experimento_12) usando los servidores
Ollama de la granja de GPUs del DTSC, atacándolos como cliente desde tu equipo.

Basado en la puesta en marcha real: incluye los problemas que aparecieron y cómo
resolverlos.

---

## 1. Arquitectura

Dos roles en dos sitios:

```
[queron: tu PC del lab]                    [amaterasu: máquina de acceso]
   CLIENTE                                    LANZADOR
   main_cluster.py  --- HTTP --->  Ollama en bastet0X:puerto  <--- launch_process.py
   (reparte los 1000                (servidores en la granja)      (los levanta vía SLURM)
    artículos en shards)
```

- **Servidores Ollama**: los levanta `launch_process.py -c llm.json` **desde amaterasu**.
  Se ejecutan en nodos GPU de la granja (bastet07‑09, rtx3090), cada uno en un puerto
  (11434‑11437), y pueden caer en máquinas distintas.
- **Cliente**: `main_cluster.py` corre **en queron**, manda peticiones HTTP a los
  servidores y guarda los resultados. queron alcanza la granja por red (sin túneles).

Todo el código y los datos del cliente se quedan en queron. En amaterasu solo van
3 ficheros: `launch_process.py`, `Scheduler.py`, `llm.json`.

---

## 2. Acceso (una vez)

- Máquina de acceso: **amaterasu.tsc.uc3m.es** (kusanagi está apagado). Grupo de
  Procesado Multimedia.
- Usuario: **jggomez** (`whoami` para confirmar). HOME: `/export/usuarios01/jggomez`.
- El acceso es por **clave SSH registrada por el admin** (Harold). No hay contraseña.

```bash
ssh jggomez@amaterasu.tsc.uc3m.es
```

---

## 3. Preparar el lanzador en amaterasu (una vez)

Desde **queron**, sube los 3 ficheros del lanzador (deben ir juntos):

```bash
ssh jggomez@amaterasu.tsc.uc3m.es 'mkdir -p ~/ollama_launcher'
rsync -avz \
  CLUSTER/cluster_tsc/bin/launch_process.py \
  CLUSTER/cluster_tsc/lib/Scheduler.py \
  CLUSTER/llm.json \
  jggomez@amaterasu.tsc.uc3m.es:~/ollama_launcher/
```

En **amaterasu**, crea la carpeta de logs:

```bash
mkdir -p /export/usuarios01/$USER/logs_llm
```

El `llm.json` ya está configurado (usuario jggomez, cola `gpus`, 4 puertos, rtx3090,
y un directorio de modelos por puerto: `/data/tmp/jggomez/models_{port}`).

---

## 4. Levantar los servidores Ollama (cada sesión)

En **amaterasu**, dentro de un `screen` (para poder dejarlo y volver):

```bash
# Asegúrate de NO estar ya dentro de un screen:
echo $STY                 # si imprime algo, sal con: exit
screen -S ollama
cd ~/ollama_launcher
python3 launch_process.py -c llm.json
```

Para salir dejándolo vivo: **Ctrl+A** y luego **D**. Para volver: `screen -r ollama`.
NO lo mates con `screen -X ... quit` (deja los jobs huérfanos, ver §8).

Comprueba dónde cayó cada servidor:

```bash
squeue -u $USER -o "%.10i %.30j %.2t %R"          # nodo de cada job
grep -H "OLLAMA_HOST:http" ~/logs_llm/*.err       # puerto de cada log
```

---

## 5. Verificar servidores y modelo (desde queron)

Qué servidores están vivos Y alcanzables (ajusta host:puerto a lo del §4):

```bash
for hp in bastet07:11434 bastet07:11435 bastet08:11436 bastet09:11437; do
  echo -n "$hp -> "; curl -s -m5 http://$hp/api/tags >/dev/null && echo OK || echo NO; done
```

Descarga el modelo en cada servidor vivo (directorio por puerto → uno a uno):

```bash
for hp in bastet07:11434 bastet07:11435 bastet08:11436; do
  echo "=== $hp ==="
  OLLAMA_HOST=$hp ollama pull gemma4:e4b     # o qwen3:8b
  OLLAMA_HOST=$hp ollama list
done
```

Modelos disponibles y tamaño: `gemma4:e4b` (9.6 GB), `qwen3:8b` (5.2 GB, más rápido).
Puedes tener los dos a la vez; eliges cuál con `--model` en el cliente.

---

## 6. Lanzar el análisis (cliente, en queron)

**Regla de oro**: `--n-shards` = nº de servidores vivos, y un shard por servidor.
Usa siempre el python del venv (`.venv`), que tiene las dependencias.

Prueba rápida (1 shard, 20 artículos):

```bash
cd /home/jggomez/Desktop/IRIS/iris-uc3m
OLLAMA_HOST=bastet07:11434 .venv/bin/python3 CLUSTER/experimento_12_cluster/main_cluster.py \
  --model gemma4:e4b --shard 0 --n-shards 3 --workers 4 --limit 20 \
  --experimentos-dir "$PWD/Experimentos" \
  --data "$PWD/data/2026_02_10_imio_def_todo_envio_heidy.xlsx - 2026_02_09_imio_def_todo_clara_scrape.csv" \
  --output-dir "$PWD/CLUSTER/experimento_12_cluster/results"
```

Corrida completa (ejemplo con 3 servidores → 3 shards):

```bash
cd /home/jggomez/Desktop/IRIS/iris-uc3m
DATA="$PWD/data/2026_02_10_imio_def_todo_envio_heidy.xlsx - 2026_02_09_imio_def_todo_clara_scrape.csv"
EXP="$PWD/Experimentos"
OUT="$PWD/CLUSTER/experimento_12_cluster/results"
PY="$PWD/.venv/bin/python3"
CLI=CLUSTER/experimento_12_cluster
M=gemma4:e4b     # o qwen3:8b (si usas qwen, cambia OUT a results_qwen)

OLLAMA_HOST=bastet07:11434 $PY $CLI/main_cluster.py --model $M --shard 0 --n-shards 3 --workers 4 --experimentos-dir "$EXP" --data "$DATA" --output-dir "$OUT" > $CLI/cli0.log 2>&1 &
OLLAMA_HOST=bastet07:11435 $PY $CLI/main_cluster.py --model $M --shard 1 --n-shards 3 --workers 4 --experimentos-dir "$EXP" --data "$DATA" --output-dir "$OUT" > $CLI/cli1.log 2>&1 &
OLLAMA_HOST=bastet08:11436 $PY $CLI/main_cluster.py --model $M --shard 2 --n-shards 3 --workers 4 --experimentos-dir "$EXP" --data "$DATA" --output-dir "$OUT" > $CLI/cli2.log 2>&1 &
```

Seguir el avance: `tail -n3 CLUSTER/experimento_12_cluster/cli*.log` y `jobs -l`.

Reanudación: si un shard se corta, relanza el mismo comando; salta los `IdNoticia`
ya guardados en su CSV. El primer artículo de cada shard va lento (carga del modelo
en GPU): es normal.

---

## 7. Unir resultados y métricas (en queron, con el venv)

```bash
cd /home/jggomez/Desktop/IRIS/iris-uc3m
PY=.venv/bin/python3
$PY CLUSTER/experimento_12_cluster/merge_shards.py CLUSTER/experimento_12_cluster/results
$PY CLUSTER/experimento_12_cluster/recod_genero.py        # accuracy/F1 por variable
$PY CLUSTER/experimento_12_cluster/metrics.py             # kappa/F1 detallado -> metrics/
```

Como el muestreo es determinista (`random_state=42`), los 1000 del cluster son los
MISMOS que los del local → comparación 1:1 (solo válida si usas el mismo modelo).

Tiempos: cada fila lleva `modelo_tiempo_procesamiento_seg` (latencia, incluye cola) y
`modelo_tiempo_modelo_real_seg` (cómputo real del modelo, sin cola). Cada shard imprime
un "RESUMEN DE TIEMPOS" con el throughput real al terminar.

---

## 8. APAGAR al terminar (IMPRESCINDIBLE)

No dejes servidores ociosos (Harold: es un sistema compartido).

```bash
# 1. clientes (en queron)
pkill -f main_cluster.py

# 2. servidores (en amaterasu): vuelve al screen y Ctrl+C
screen -r ollama          # Ctrl+C -> el launcher cancela sus jobs

# 3. verifica que NO queda nada tuyo
squeue -u $USER           # debe salir VACÍO
scancel -u $USER          # si quedó algo colgado, cancélalo
```

⚠️ Si matas el screen con `quit` en vez de Ctrl+C, el launcher muere pero **los jobs
Ollama quedan huérfanos y siguen corriendo**. En ese caso límpialos a mano con
`scancel -u $USER`.

---

## 9. Problemas conocidos

**Varios servidores caen en el mismo nodo y solo arrancan 2‑3.**
La granja tiene 3 nodos rtx3090 (bastet07‑09) para 4 servidores. Cuando 2+ caen en el
mismo nodo, se pelean por la misma GPU (todos con `CUDA_VISIBLE_DEVICES=0`, mismo
`pci_id`) y los que no la consiguen cascan con *"CUDA device busy or unavailable"* en
la fase de GPU discovery. Resultado: no siempre salen los 4.
→ Usa tantos shards como servidores vivos veas en el §5. Pendiente de consultar a
Harold cómo dar una GPU dedicada a cada `ollama serve`.

**Un servidor "RUNNING" en squeue pero no responde a curl.**
El job (bash) sigue vivo pero el `ollama serve` de dentro crasheó (ver su `.err` en
`~/logs_llm/`). No lo uses; míralo con `tail ~/logs_llm/llm_eval_*.err`.

**El modelo se re-descarga al relanzar.**
Usamos un directorio por puerto (`models_{port}`) y `/data/tmp` es local a cada nodo;
si el servidor cae en otro nodo/puerto, no encuentra el modelo y lo baja otra vez. Es
el precio de evitar un *race condition* al inicializar el directorio compartido.

**`ModuleNotFoundError: sklearn` / `ollama`.**
Estás usando el python del sistema. Usa el del venv: `.venv/bin/python3`.

**No entro al screen ("Attached").**
Está enganchado en otra sesión: `screen -d -r ollama` (fuerza). Si estás dentro de un
screen, sal antes (`exit`).

---

## 10. Ficheros

```
CLUSTER/
├── llm.json                     # config del lanzador (4 servidores Ollama, rtx3090)
├── llm.json.skel                # plantilla original de Harold
├── cluster_tsc/                 # librería del cluster (launch_process.py, Scheduler.py, manual)
└── experimento_12_cluster/      # cliente adaptado + utilidades
    ├── main_cluster.py          # cliente con sharding + medición de tiempos
    ├── run_clients.sh           # lanza los shards en paralelo
    ├── merge_shards.py          # une los CSV por shard
    ├── recod_genero.py          # recodificación + accuracy/F1
    ├── metrics.py               # métricas detalladas (kappa/F1) -> metrics/
    ├── requirements.txt         # dependencias del cliente
    ├── 1_subir.sh / 2_setup.sh / 3_parar.sh
    └── README.md                # detalle del cliente
```
