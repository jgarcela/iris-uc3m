#!/usr/bin/env bash
# Lanza B0 (baseline sin skills) del Exp 21 repartido en shards, uno por servidor
# Ollama de la granja. Se ejecuta EN QUERON tras haber levantado los Ollama en
# amaterasu (launch_process.py -c llm.json).
#
# Uso:
#   ./lanzar_b0.sh host1:puerto host2:puerto [host3:puerto ...]
# Ej (4 servidores del llm.json):
#   ./lanzar_b0.sh bastet07:11434 bastet07:11435 bastet08:11436 bastet08:11437
set -u
cd "$(dirname "$0")" || exit 1

EXP=/home/jggomez/Desktop/IRIS/iris-uc3m/Experimentos
AGE=$EXP/experiments/experimento_21_agentskills
DATA="/home/jggomez/Desktop/IRIS/iris-uc3m/data/2026_02_10_imio_def_todo_envio_heidy.xlsx - 2026_02_09_imio_def_todo_clara_scrape.csv"
PY=/home/jggomez/Desktop/IRIS/iris-uc3m/.venv/bin/python
OUT=./results_b0

[ $# -lt 1 ] && { echo "Uso: ./lanzar_b0.sh host1:puerto [host2:puerto ...]"; exit 1; }
N=$#
echo "Lanzando B0 en $N shards (--only-labeled, workers=4). Salida: $OUT"
i=0
for hp in "$@"; do
  log="/tmp/exp21_b0_shard${i}.log"
  OLLAMA_HOST="$hp" nohup "$PY" main_cluster.py \
      --model gemma4:e4b --baseline \
      --shard "$i" --n-shards "$N" --workers 4 --only-labeled \
      --experimentos-dir "$EXP" --agente-dir "$AGE" --data "$DATA" \
      --output-dir "$OUT" > "$log" 2>&1 &
  echo "  shard $i/$N → $hp   (log: $log)"
  i=$((i+1))
  sleep 2
done
echo
echo "Seguimiento:  tail -f /tmp/exp21_b0_shard*.log"
echo "Al terminar:  $PY merge_shards.py --input-dir $OUT --output $OUT/FULL.csv"
echo "Métricas:     $PY \"$AGE/metrics.py\" --pred $OUT/FULL.csv --corpus \"$DATA\""
