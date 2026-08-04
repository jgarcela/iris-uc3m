#!/usr/bin/env bash
# Estado de los shards del Exp 21 en el cluster. Uso: watch -n 10 ./estado.sh
cd "$(dirname "$0")" || exit 1
PY=/home/jggomez/Desktop/IRIS/iris-uc3m/.venv/bin/python
OUT=results_b0

printf '%s   Exp 21 B0 · gemma4:e4b · granja TSC\n\n' "$(date '+%H:%M:%S')"

filas() {  # cuenta filas reales (las explicaciones llevan saltos de línea)
  [ -s "$1" ] || { echo 0; return; }
  "$PY" -c "import csv,sys
try:
 print(sum(1 for _ in csv.DictReader(open(sys.argv[1],encoding='utf-8'))))
except Exception: print(0)" "$1" 2>/dev/null || echo 0
}
con_error() {  # filas con alguna variable fallida
  [ -s "$1" ] || { echo 0; return; }
  "$PY" -c "import pandas as pd,sys
try:
 d=pd.read_csv(sys.argv[1]); print(int((d['n_variables_error']>0).sum()) if 'n_variables_error' in d else 0)
except Exception: print(0)" "$1" 2>/dev/null || echo 0
}

TOT=0; ERRTOT=0
for i in 0 1 2; do
  f="$OUT/21b0-Experimento-21_b0_gemma4_e4b_shard${i}de3.csv"
  log="/tmp/exp21_b0_shard${i}.log"
  n=$(filas "$f"); e=$(con_error "$f")
  TOT=$((TOT+n)); ERRTOT=$((ERRTOT+e))

  # estado del proceso
  if pgrep -f "shard $i --n-shards 3" >/dev/null || pgrep -f "shard ${i} " >/dev/null 2>&1; then est="corriendo"; else est="parado"; fi
  # ETA de tqdm
  eta=$(tr '\r' '\n' < "$log" 2>/dev/null | grep -o '\[[0-9:]*<[^]]*\]' | tail -1)
  # aborto por pre-vuelo
  grep -q "ABORTADO (pre-vuelo)" "$log" 2>/dev/null && est="⛔ ABORTADO (servidor sin modelo)"

  bar=""; pct=$(( n * 20 / 439 )); [ $pct -gt 20 ] && pct=20
  for ((k=0;k<pct;k++)); do bar+="#"; done; for ((k=pct;k<20;k++)); do bar+="."; done

  flag=""; [ "$e" -gt 5 ] && flag="  ⚠ $e con error"
  printf '  shard %d  [%s] %3d/439  %-10s %s%s\n' "$i" "$bar" "$n" "$est" "$eta" "$flag"
done

printf '\n  TOTAL: %d/1315 filas' "$TOT"
[ "$ERRTOT" -gt 0 ] && printf '   (⚠ %d filas con error — revisar antes de fusionar)' "$ERRTOT"
printf '\n'
