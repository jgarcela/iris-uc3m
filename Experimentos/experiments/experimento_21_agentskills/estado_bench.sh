#!/usr/bin/env bash
# Estado de las corridas del Exp 21.
#   ./estado_bench.sh          → nivel B1 (results/bench_*)
#   ./estado_bench.sh b0       → nivel B0 (results/b0_*)
# En vivo:  watch -n 10 ./estado_bench.sh b0
cd "$(dirname "$0")" || exit 1
PREFIJO="${1:-bench}"

# Nº de filas reales del corpus (los artículos llevan saltos de línea → wc -l no vale)
total=$(python3 -c "
import csv,sys
try:
    with open(sys.argv[1], encoding='utf-8') as f: print(sum(1 for _ in csv.DictReader(f)))
except Exception: print(0)
" "${CORPUS:-real1315_corpus.csv}" 2>/dev/null)
[ -z "$total" ] || [ "$total" -lt 1 ] && total=1313

if [ "$PREFIJO" = "b0" ]; then
  titulo="Eje B — nivel B0 (metodología inyectada, sin skills)"
else
  titulo="Eje A/B1 — benchmark con skills"
fi
printf '%s   %s · Exp 21 (N=%d)\n\n' "$(date '+%H:%M:%S')" "$titulo" "$total"

shopt -s nullglob
for d in results/"${PREFIJO}"_*; do
  f=$(ls "$d"/*.csv 2>/dev/null | head -1)
  # Filas reales (las explicaciones llevan saltos de línea → wc -l no vale)
  n=0
  [ -n "$f" ] && [ -s "$f" ] && n=$(python3 -c "
import csv,sys
with open(sys.argv[1], encoding='utf-8') as fh: print(sum(1 for _ in csv.DictReader(fh)))
" "$f" 2>/dev/null || echo 0)
  pct=$(( n * 100 / (total > 0 ? total : 1) ))
  barra=""; for ((i=0; i<pct/5; i++)); do barra+="#"; done
  for ((i=pct/5; i<20; i++)); do barra+="."; done

  modelo="${d#results/${PREFIJO}_}"
  if pgrep -f "output-dir $d" > /dev/null; then estado="corriendo"
  elif [ "$n" -ge "$total" ]; then estado="✔ ok"
  else estado="⚠ parada"; fi

  coste=""
  if [ -n "$f" ] && [ -s "$f" ]; then
    coste=$(python3 - "$f" <<'PY' 2>/dev/null
import csv, sys
t = n = 0.0
with open(sys.argv[1], encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        try:
            t += float(r.get("coste_articulo_usd") or 0); n += 1
        except ValueError:
            pass
print(f"${t:.2f} (proy.1315 ${t/n*1315:,.1f})" if n else "")
PY
)
  fi
  printf '  %-26s [%s] %4d/%d  %-9s %s\n' "$modelo" "$barra" "$n" "$total" "$estado" "$coste"
done
