#!/usr/bin/env bash
# Estado de las 6 ablaciones (sin-resúmenes / sin-consultar-guía × 3 modelos API).
# Uso:  watch -n 15 ./estado_abl.sh
cd "$(dirname "$0")" || exit 1
TOTAL=1313
printf "%-34s %-9s %-10s %s\n" "CORRIDA" "PROGRESO" "ETA" "ESTADO"
printf '%.0s-' {1..70}; echo
for l in /tmp/abl_*.log; do
  [ -f "$l" ] || continue
  name=$(basename "$l" .log)
  slug=${name#abl_}
  proc=$(pgrep -af "main.py.*results/${name#abl_}" 2>/dev/null | grep -v pgrep)
  # última línea de progreso de tqdm
  line=$(tr '\r' '\n' < "$l" | grep -E "[0-9]+/$TOTAL" | tail -1)
  prog=$(echo "$line" | grep -oE "[0-9]+/$TOTAL" | tail -1)
  eta=$(echo "$line" | grep -oE '<[0-9:]+' | tr -d '<' | tail -1)
  if tr '\r' '\n' < "$l" | grep -q "Hecho →"; then
    estado="✅ HECHO"; prog="$TOTAL/$TOTAL"
  elif tr '\r' '\n' < "$l" | grep -qiE "ABORTADO|Traceback"; then
    estado="❌ ERROR"
  elif pgrep -f "main.py.*${name#abl_}" >/dev/null 2>&1; then
    estado="▶ corriendo"
  else
    estado="⏸ parado"
  fi
  printf "%-34s %-9s %-10s %s\n" "$slug" "${prog:-0/$TOTAL}" "${eta:-–}" "$estado"
done
