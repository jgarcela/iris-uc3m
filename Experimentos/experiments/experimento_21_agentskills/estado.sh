#!/usr/bin/env bash
# Estado de las corridas A (sin cache) y B (con cache) del Experimento 21.
# Uso en vivo:  watch -n 10 ./estado.sh
cd "$(dirname "$0")" || exit 1

printf '%s   Exp 21 — 100 artículos · gpt-4o-mini\n\n' "$(date '+%H:%M:%S')"

estado_de() {  # $1 = etiqueta, $2 = carpeta results, $3 = log
  local f="results/$2/exp21_gpt-4o-mini.csv"
  local n=0
  [ -s "$f" ] && n=$(( $(wc -l < "$f") - 1 ))
  local barra="" i
  for ((i = 0; i < n / 5; i++)); do barra+="#"; done
  for ((i = n / 5; i < 20; i++)); do barra+="."; done

  local eta="—"
  [ -f "$3" ] && eta=$(tr '\r' '\n' < "$3" | grep -o '\[[^]]*\]' | tail -1)

  if pgrep -f "output-dir results/$2" > /dev/null; then
    printf '  %-14s [%s] %3d/100   %s\n' "$1" "$barra" "$n" "$eta"
  else
    if [ "$n" -ge 100 ]; then
      printf '  %-14s [%s] %3d/100   ✔ COMPLETADA\n' "$1" "$barra" "$n"
    else
      printf '  %-14s [%s] %3d/100   ⚠ PARADA (relanzar para reanudar)\n' "$1" "$barra" "$n"
    fi
  fi
}

estado_de "A sin cache" A_sin_cache /tmp/runA.log
estado_de "B con cache" B_con_cache /tmp/runB.log

printf '\n  Coste acumulado hasta ahora:\n'
for c in A_sin_cache B_con_cache; do
  f="results/$c/exp21_gpt-4o-mini.csv"
  if [ -s "$f" ]; then
    python3 - "$f" "$c" <<'PY' 2>/dev/null
import csv, sys
ruta, etiqueta = sys.argv[1], sys.argv[2]
total = tok = 0.0
n = 0
with open(ruta, encoding="utf-8") as fh:
    for fila in csv.DictReader(fh):
        try:
            total += float(fila.get("coste_articulo_usd") or 0)
            tok += float(fila.get("tokens_articulo") or 0)
            n += 1
        except ValueError:
            pass
if n:
    print(f"    {etiqueta:<12} ${total:.4f}  ({tok/n:,.0f} tok/art · proy. 1315: ${total/n*1315:,.2f})")
PY
  fi
done
