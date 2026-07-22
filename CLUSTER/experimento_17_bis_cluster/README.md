# Experimento 17 bis (cluster) — gpt-5-nano + `variables_umbral_bajo.json`

Misma tarea que el Exp 17 (5 variables de sexismo, multi-proveedor), pero con
**umbral amplio** en V25/V26 vía `--variables-json …/variables_umbral_bajo.json`.

Sirve para la ablación **estricto (Exp 17) vs umbral bajo (este bis)** con
`gpt-5-nano`, análoga a Exp 18 vs 18 bis con `gpt-4o-mini`.

## Corrida recomendada (mismas 1 315 anotadas que Exp 17)

```bash
cd /home/jggomez/Desktop/IRIS/iris-uc3m
DATA="$PWD/data/2026_02_10_imio_def_todo_envio_heidy.xlsx - 2026_02_09_imio_def_todo_clara_scrape.csv"
EXP="$PWD/Experimentos"
PY="$PWD/.venv/bin/python3"
CLI=CLUSTER/experimento_17_bis_cluster
M=gpt-5-nano
OUT="$PWD/$CLI/results/${M}_umbral_bajo"
VARS_JSON="$EXP/variables_umbral_bajo.json"

export OPENAI_API_KEY=...   # requerida

# Prueba rápida
$PY $CLI/main_cluster.py --model "$M" --shard 0 --n-shards 1 --workers 8 --limit 20 \
  --only-labeled --variables-json "$VARS_JSON" \
  --experimentos-dir "$EXP" --data "$DATA" --output-dir "$OUT"

# Corrida completa (1 315 etiquetadas)
$PY $CLI/main_cluster.py --model "$M" --shard 0 --n-shards 1 --workers 8 \
  --only-labeled --variables-json "$VARS_JSON" \
  --experimentos-dir "$EXP" --data "$DATA" --output-dir "$OUT" \
  > "$CLI/run_${M}_umbral_bajo.log" 2>&1 &
```

## Merge + métricas

```bash
$PY $CLI/merge_shards.py "$OUT"
$PY $CLI/metrics.py \
  "$OUT/17bis-Experimento-17bis_03_2026_resultados_modelo_2024_scrape_FULL.csv" \
  "${M}-umbral"
```

Comparar con:
`CLUSTER/experimento_17_cluster/metrics/metrics_17-…FULL.csv` (`gpt-5-nano` estricto).
