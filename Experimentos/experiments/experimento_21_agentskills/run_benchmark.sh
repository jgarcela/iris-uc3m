#!/usr/bin/env bash
# Eje A — Benchmark multi-proveedor del Experimento 21.
#
# Config canónica (decidida por ablaciones, ver DIARIO_TFM):
#   · prompt caching  OFF   (+87% coste, 0 ganancia)
#   · reasoning       OFF    (todos los proveedores, comparabilidad)
#   · skills-resumen  ON
#   · variables.json  estricto  (umbral bajo = ablación aparte)
#
# Uso:
#   ./run_benchmark.sh <corpus.csv> [modelo ...]              → B1 (con skills)
#   BASELINE=1 ./run_benchmark.sh <corpus.csv> [modelo ...]   → B0 (sin skills)
# Ej:
#   ./run_benchmark.sh real1315_corpus.csv
#   BASELINE=1 ./run_benchmark.sh real1315_corpus.csv
set -u
cd "$(dirname "$0")" || exit 1

PY=../../../.venv/bin/python
CORPUS="${1:?Uso: ./run_benchmark.sh <corpus.csv> [modelo ...]}"
shift

MODELOS=("$@")
if [ ${#MODELOS[@]} -eq 0 ]; then
  # Claude fuera del set por defecto: sin caching automático cuesta ~8x el resto
  # (~$69 vs ~$7-9 en las 1313). Añadir a mano si se decide incluirlo.
  MODELOS=(gpt-4o-mini gpt-5.4-nano gemini-3.1-flash-lite)
fi

# 1. Skills desde el JSON estricto (deja constancia del origen en cada SKILL.md)
echo "== Regenerando skills desde variables.json (estricto) =="
$PY generar_skills.py --json ../../variables.json || exit 1
echo

# 2. Una corrida por modelo, en paralelo (cada una con su carpeta y su log)
if [ "${BASELINE:-0}" = "1" ]; then
  NIVEL="b0"; EXTRA="--baseline"; echo "== Nivel B0: metodología inyectada, sin skills =="
else
  NIVEL="bench"; EXTRA=""
fi

for m in "${MODELOS[@]}"; do
  slug="${m//[^A-Za-z0-9._-]/_}"
  out="results/${NIVEL}_${slug}"
  log="/tmp/${NIVEL}_${slug}.log"
  echo "== Lanzando $m → $out =="
  nohup $PY main.py --input "$CORPUS" --modelo "$m" \
      --output-dir "$out" --sin-cache $EXTRA > "$log" 2>&1 &
  sleep 2
done

echo
echo "Lanzados ${#MODELOS[@]} modelos. Seguimiento:"
echo "  watch -n 10 ./estado_bench.sh"
echo "Métricas al terminar:"
echo "  for d in results/bench_*; do echo \"== \$d\"; $PY metrics.py --pred \"\$d\"/*.csv; done"
