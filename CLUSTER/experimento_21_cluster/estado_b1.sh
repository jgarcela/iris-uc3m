#!/usr/bin/env bash
# Estado de los shards de B1 en el cluster.
# Uso:  watch -n 15 ./estado_b1.sh [completo|minimo]
cd "$(dirname "$0")" || exit 1
MODO="${1:-completo}"
OUT="./results_b1_${MODO}"
TOTAL=1313   # artículos etiquetados (--only-labeled); repartidos entre los shards

printf "B1-%s   (objetivo total ~%d artículos)\n" "$MODO" "$TOTAL"
printf "%-8s %-10s %-6s %s\n" "SHARD" "PROGRESO" "ETA" "ESTADO"
printf '%.0s-' {1..56}; echo
suma=0
for log in /tmp/exp21_b1${MODO}_shard*.log; do
  [ -f "$log" ] || continue
  s=$(echo "$log" | grep -oE 'shard[0-9]+' | grep -oE '[0-9]+')
  # progreso REAL = tqdm (contar filas con wc -l falla: las celdas tienen saltos de línea)
  line=$(tr '\r' '\n' < "$log" | grep -E '[0-9]+/[0-9]+' | tail -1)
  tq=$(echo "$line" | grep -oE '[0-9]+/[0-9]+' | tail -1)
  eta=$(echo "$line" | grep -oE '<[0-9:]+' | tr -d '<' | tail -1)
  hechos=$(echo "$tq" | cut -d/ -f1); [ -z "$hechos" ] && hechos=0
  suma=$((suma + hechos))
  if grep -qE "Archivo completado|RESUMEN DE TIEMPOS|Hecho" "$log" 2>/dev/null; then est="✅ HECHO"
  elif pgrep -f "main_cluster.py.*--shard ${s} " >/dev/null 2>&1; then est="▶ corriendo"
  elif tr '\r' '\n' < "$log" | grep -qiE "ABORT|Traceback|CUDA"; then est="❌ revisar log"
  else est="⏸ parado"; fi
  printf "%-8s %-10s %-6s %s\n" "$s" "${tq:-0/?}" "${eta:-–}" "$est"
done
printf '%.0s-' {1..56}; echo
printf "TOTAL procesados: %d / ~%d  (%d%%)\n" "$suma" "$TOTAL" "$(( suma*100/TOTAL ))"
