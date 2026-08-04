#!/usr/bin/env bash
# Espera a que terminen los 4 shards de B1-sinres y entonces:
#   1) fusiona sinres,
#   2) lanza B1-singuia (solo resúmenes) en los mismos servidores.
# Pensado para dejarlo en segundo plano:  nohup ./encadenar_singuia.sh &
set -u
cd "$(dirname "$0")" || exit 1

HOSTS=(bastet10:11434 bastet10:11435 bastet10:11436 bastet10:11437)
PY=/home/jggomez/Desktop/IRIS/iris-uc3m/.venv/bin/python
DATA="/home/jggomez/Desktop/IRIS/iris-uc3m/data/2026_02_10_imio_def_todo_envio_heidy.xlsx - 2026_02_09_imio_def_todo_clara_scrape.csv"

log(){ echo "[$(date '+%H:%M:%S')] $*"; }

log "Esperando a que terminen los 4 shards de B1-sinres…"
while :; do
  done=$(grep -lE "Archivo completado" /tmp/exp21_b1sinres_shard*.log 2>/dev/null | wc -l)
  alive=$(pgrep -f "main_cluster.py.*results_b1_sinres" | grep -vc pgrep 2>/dev/null || echo 0)
  # terminado = los 4 logs con "Archivo completado" y ningún proceso sinres vivo
  if [ "$done" -ge 4 ] && [ "$alive" -eq 0 ]; then break; fi
  # aborto de seguridad: si no queda ningún proceso vivo pero no están los 4 -> avisar y salir
  if [ "$alive" -eq 0 ] && [ "$done" -lt 4 ]; then
    log "⚠️  sinres sin procesos vivos pero solo $done/4 completados. Revisa logs. NO lanzo singuia."
    exit 1
  fi
  sleep 60
done
log "✅ B1-sinres completado."

log "Fusionando sinres…"
"$PY" merge_shards.py --input-dir results_b1_sinres --output results_b1_sinres/FULL.csv

# Verificar servidores antes de lanzar el último
for hp in "${HOSTS[@]}"; do
  curl -s -m5 "http://$hp/api/tags" | grep -q gemma4:e4b || { log "⚠️  $hp no responde con gemma. Aborto."; exit 1; }
done

log "🚀 Lanzando B1-singuia (solo resúmenes)…"
./lanzar_b1_ablacion.sh singuia "${HOSTS[@]}"
log "singuia lanzado. Sigue con:  watch -n 15 ./estado_b1.sh singuia"
