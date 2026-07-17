#!/usr/bin/env bash
# Lanza 4 clientes exp12, uno por servidor Ollama (un shard cada uno).
# Rellena los host:puerto reales que veas en los logs de launch_process.py.
set -euo pipefail

# --- CONFIGURA ESTO ---
export EXPERIMENTOS_DIR="/export/usuarios01/jggomez/iris/Experimentos"
export DATA_CSV="/export/usuarios01/jggomez/iris/data/2026_02_10_imio_def_todo_envio_heidy.xlsx - 2026_02_09_imio_def_todo_clara_scrape.csv"
export OUTPUT_DIR="/export/usuarios01/jggomez/iris/results_cluster"
export MODELO="gemma4:e4b"
WORKERS=4          # peticiones concurrentes por servidor
N_SHARDS=4

# host:puerto de cada servidor Ollama (míralos en stdout.log / logdir del launcher)
SERVERS=(
  "bastet07:11434"
  "bastet07:11435"
  "bastet08:11436"
  "bastet09:11437"
)
# ----------------------

mkdir -p "$OUTPUT_DIR" logs
for i in "${!SERVERS[@]}"; do
  host="${SERVERS[$i]}"
  echo "Lanzando shard $i -> $host"
  OLLAMA_HOST="$host" python main_cluster.py \
      --shard "$i" --n-shards "$N_SHARDS" --workers "$WORKERS" \
      > "logs/cliente_shard${i}.log" 2>&1 &
done

wait
echo "Todos los shards han terminado. Resultados en: $OUTPUT_DIR"
