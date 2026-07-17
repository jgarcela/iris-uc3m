# Experimento 12 en el cluster TSC

Adaptación del `experimento_12` (fase **cliente**) para correr contra los servidores
Ollama que levanta `launch_process.py` con `llm.json`.

- No modifica el `experimento_12` original.
- Reparte los 1000 artículos en **shards** (uno por servidor/GPU).
- Selecciona modelo, `OLLAMA_HOST` y rutas por **variables de entorno / CLI**.
- Guarda un CSV por shard (los clientes no se pisan) y reanuda si se corta.

## Flujo completo

### 0. Subir archivos y crear el entorno (una vez)
Los scripts de apoyo y los datos deben estar en un **volumen compartido** del
cluster (visible desde la máquina de acceso y los nodos), no en tu PC local.
El manual recomienda `/export/usuarios01/$USER` o `/export/clusterdata/$USER`.

```bash
# Desde tu PC (queron): subir Experimentos/ y data/
rsync -av /home/jggomez/Desktop/IRIS/iris-uc3m/Experimentos/ \
      jggomez@ceres.tsc.uc3m.es:/export/usuarios01/jggomez/iris/Experimentos/
rsync -av "/home/jggomez/Desktop/IRIS/iris-uc3m/data/" \
      jggomez@ceres.tsc.uc3m.es:/export/usuarios01/jggomez/iris/data/
# subir también esta carpeta (el cliente)
rsync -av /home/jggomez/Desktop/IRIS/iris-uc3m/CLUSTER/experimento_12_cluster/ \
      jggomez@ceres.tsc.uc3m.es:/export/usuarios01/jggomez/iris/experimento_12_cluster/

# En la máquina de acceso: venv en volumen compartido con las deps
python3 -m venv /export/clusterdata/jggomez/venvs/iris
source /export/clusterdata/jggomez/venvs/iris/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> `variables.py`/`utils.py` importan `variables` desde `EXPERIMENTOS_DIR` (vía
> `sys.path`) y leen `variables.json` + `prompts/prompt_clara.md` de ahí. Los
> datos se leen de `DATA_CSV`. Ambas rutas deben apuntar al volumen compartido.

### 1. Levantar los servidores Ollama (fase servidor)
Desde el directorio con `launch_process.py`, `Scheduler.py` y `llm.json`:
```bash
launch_process.py -c llm.json
```
Levanta 4 `ollama serve` en bastet07-09, puertos 11434-11437.
Mira `stdout.log` / `logdir` para saber **en qué máquina cayó cada puerto**.

### 2. Descargar el modelo (una vez por máquina)
Los modelos van a `/data/tmp/$USER/models` (ver `llm.json`). Contra cada servidor:
```bash
OLLAMA_HOST=bastet07:11434 /opt/ollama/bin/ollama pull gemma4:e4b
```
> Si `/data/tmp` es local a cada nodo, repite el pull contra un servidor de
> cada máquina (bastet07, bastet08, bastet09). Confírmalo con Harold.

### 3. Prueba rápida (1 shard, pocos artículos)
```bash
OLLAMA_HOST=bastet07:11434 \
EXPERIMENTOS_DIR=/export/usuarios01/jggomez/iris/Experimentos \
DATA_CSV="/export/usuarios01/jggomez/iris/data/..._scrape.csv" \
python main_cluster.py --shard 0 --n-shards 4 --workers 4 --limit 20
```

### 4. Corrida completa (4 clientes, 4 shards)
Edita los `host:puerto` en `run_clients.sh` con los reales y:
```bash
bash run_clients.sh
```

### 5. Unir resultados
```bash
python merge_shards.py "$OUTPUT_DIR"
```

## Monitorizar y PARAR (no dejar procesos dormidos)

Lanza el launcher dentro de `tmux`/`screen` para poder pararlo con `Ctrl+C`:
```bash
tmux new -s ollama
launch_process.py -c llm.json     # Ctrl+C aquí cancela los 4 servidores
```

Comandos útiles:
```bash
squeue -u $USER                   # tus jobs (debe quedar VACÍO al terminar)
scontrol show job JOBID           # detalle de un job
scancel --name "LLM Evaluation"   # cancelar los servidores Ollama
scancel -u $USER                  # cancelar TODOS tus jobs
pkill -f main_cluster.py          # matar los clientes
```

Al acabar, ejecuta `bash 3_parar.sh` (para clientes + jobs) y comprueba que
`squeue -u $USER` sale vacío.

## Parámetros (todos por env o CLI)

| CLI | Env | Def. | Qué es |
|-----|-----|------|--------|
| `--experimentos-dir` | `EXPERIMENTOS_DIR` | — | Carpeta `Experimentos/` (variables.py, utils.py, ...) |
| `--data` | `DATA_CSV` | — | CSV de datos ya scrapeado |
| `--output-dir` | `OUTPUT_DIR` | `./results` | Salida |
| `--model` | `MODELO` | `gemma4:e4b` | Modelo Ollama |
| `--ollama-host` | `OLLAMA_HOST` | localhost:11434 | Servidor a atacar |
| `--shard` | `SHARD` | 0 | Índice de shard (0-based) |
| `--n-shards` | `N_SHARDS` | 1 | Nº total de shards |
| `--workers` | `WORKERS` | 1 | Concurrencia dentro del shard |
| `--limit` | `LIMIT` | — | Máx. artículos (pruebas) |

## Validar contra local
Como el muestreo es determinista (`random_state=42`), al unir los shards obtienes
los mismos 1000 artículos que en local. Compara el CSV `_FULL` con tu resultado
local del exp12 para verificar que el entorno del cluster da resultados equivalentes.
