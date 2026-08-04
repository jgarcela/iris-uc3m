#!/usr/bin/env bash
# Lanza B1 de gemma (Agent Skills) repartido en shards por la granja Ollama.
# Dos modos de ablación:
#   completo  → B1 completo (resúmenes + tool CONSULTAR_GUIA)
#   minimo    → B1-mínimo (solo skills: sin resúmenes, sin tool)
#
# Se ejecuta EN QUERON tras levantar los Ollama en amaterasu.
#
# Uso:
#   ./lanzar_b1_ablacion.sh <completo|minimo> host1:puerto host2:puerto [...]
# Ej:
#   ./lanzar_b1_ablacion.sh completo bastet07:11434 bastet07:11435 bastet08:11436
#   ./lanzar_b1_ablacion.sh minimo   bastet07:11434 bastet07:11435 bastet08:11436
set -u
cd "$(dirname "$0")" || exit 1

MODO="${1:-}"; shift || true
case "$MODO" in
  completo) FLAGS="";                                            OUT=./results_b1_completo ;;
  minimo)   FLAGS="--sin-resumenes-guias --sin-consultar-guia"; OUT=./results_b1_minimo ;;
  sinres)   FLAGS="--sin-resumenes-guias";                      OUT=./results_b1_sinres ;;   # solo tool
  singuia)  FLAGS="--sin-consultar-guia";                       OUT=./results_b1_singuia ;;  # solo resúmenes
  *) echo "Uso: ./lanzar_b1_ablacion.sh <completo|minimo|sinres|singuia> host:puerto [...]"; exit 1 ;;
esac
[ $# -lt 1 ] && { echo "Faltan hosts. Uso: ./lanzar_b1_ablacion.sh $MODO host:puerto [...]"; exit 1; }

EXP=/home/jggomez/Desktop/IRIS/iris-uc3m/Experimentos
AGE=$EXP/experiments/experimento_21_agentskills
DATA="/home/jggomez/Desktop/IRIS/iris-uc3m/data/2026_02_10_imio_def_todo_envio_heidy.xlsx - 2026_02_09_imio_def_todo_clara_scrape.csv"
PY=/home/jggomez/Desktop/IRIS/iris-uc3m/.venv/bin/python

N=$#
echo "Lanzando B1-$MODO en $N shards (workers=4, --only-labeled). Salida: $OUT"
echo "Flags de ablación: ${FLAGS:-（ninguno = B1 completo）}"
i=0
for hp in "$@"; do
  log="/tmp/exp21_b1${MODO}_shard${i}.log"
  OLLAMA_HOST="$hp" nohup "$PY" main_cluster.py \
      --model gemma4:e4b $FLAGS \
      --shard "$i" --n-shards "$N" --workers 4 --only-labeled \
      --experimentos-dir "$EXP" --agente-dir "$AGE" --data "$DATA" \
      --output-dir "$OUT" > "$log" 2>&1 &
  echo "  shard $i/$N → $hp   (log: $log)"
  i=$((i+1))
  sleep 2
done
echo
echo "Seguimiento:  tail -f /tmp/exp21_b1${MODO}_shard*.log"
echo "Al terminar:  $PY merge_shards.py --input-dir $OUT --output $OUT/FULL.csv"
echo "Métricas:     $PY \"$AGE/metrics.py\" --pred $OUT/FULL.csv --corpus \"$DATA\""
