# Experimento 18 bis (cluster) — comparativa multi-modelo de las 5 variables de sexismo

Mismas 5 variables que el experimento_13 (lenguaje sexista, masc genérico, sexismo
discurso, asimetría mujer/hombre, denominación sexualizada), pero pensado para
**comparar modelos de distintos proveedores**. `main_cluster.py` enruta por el ID
del modelo (vía `utils.consultar_ollama`), así que el MISMO script vale para:

| Proveedor | Ejemplos `--model` | Credencial |
|-----------|--------------------|------------|
| OpenAI    | `gpt-4o-mini`, `gpt-4.1`, `gpt-5-mini` | `OPENAI_API_KEY` |
| Anthropic | `claude-haiku-4-5-20251001`, `claude-sonnet-4-6` | `ANTHROPIC_API_KEY` |
| Gemini    | `gemini-2.5-flash`, `gemini-2.5-pro` | `GEMINI_API_KEY` (o `GOOGLE_API_KEY`) |
| Ollama local (granja) | `gemma4:e4b` | `OLLAMA_HOST=bastet0X:PUERTO` |

El muestreo es determinista (`random_state=42`) → los 1000 artículos de 2024 son los
MISMOS para todos los modelos, así que la comparación es 1:1.

> Consejo: da a cada modelo su propio `--output-dir` (p. ej. `results/gpt-4o-mini/`)
> para no mezclar shards de modelos distintos.

## 1. Modelos de API (NO necesitan la granja Ollama)

```bash
cd /home/jggomez/Desktop/IRIS/iris-uc3m
DATA="$PWD/data/2026_02_10_imio_def_todo_envio_heidy.xlsx - 2026_02_09_imio_def_todo_clara_scrape.csv"
EXP="$PWD/Experimentos"
PY="$PWD/.venv/bin/python3"
CLI=CLUSTER/experimento_18_cluster
M=gpt-4o-mini                      # o gpt-4.1 / claude-sonnet-4-6 / gemini-2.5-flash ...
OUT="$PWD/$CLI/results/$M"

export OPENAI_API_KEY=...          # la credencial del proveedor que toque

# Prueba rápida (20 artículos)
$PY $CLI/main_cluster.py --model "$M" --shard 0 --n-shards 1 --workers 8 --limit 20 \
  --experimentos-dir "$EXP" --data "$DATA" --output-dir "$OUT"

# Corrida completa (1 solo shard vale para API; sube --workers para ir más rápido)
$PY $CLI/main_cluster.py --model "$M" --shard 0 --n-shards 1 --workers 8 \
  --experimentos-dir "$EXP" --data "$DATA" --output-dir "$OUT"
```

Nota: con modelos de razonamiento (`gpt-5-*`) `utils` omite `temperature`
automáticamente; no hace falta configurar nada.

## 2. Modelo local en la granja (igual que exp_13)

Levanta los servidores Ollama según el README general de `CLUSTER/` y luego, con un
shard por servidor vivo:

```bash
M=gemma4:e4b
OUT="$PWD/$CLI/results/$M"
OLLAMA_HOST=bastet07:11434 $PY $CLI/main_cluster.py --model $M --shard 0 --n-shards 2 --workers 4 --experimentos-dir "$EXP" --data "$DATA" --output-dir "$OUT" > $CLI/cli0.log 2>&1 &
OLLAMA_HOST=bastet07:11435 $PY $CLI/main_cluster.py --model $M --shard 1 --n-shards 2 --workers 4 --experimentos-dir "$EXP" --data "$DATA" --output-dir "$OUT" > $CLI/cli1.log 2>&1 &
```

## 3. Unir shards y calcular métricas

```bash
$PY $CLI/merge_shards.py "$OUT"
$PY $CLI/metrics.py "$OUT/18bis-Experimento-18bis_03_2026_resultados_modelo_2024_scrape_FULL.csv" "$M"
```

`metrics.py` acepta la ruta del CSV `_FULL` y una etiqueta de modelo, así generas una
tabla de métricas por modelo y luego las comparas entre sí.

## Ficheros
```
experimento_18_cluster/
├── main_cluster.py   # cliente sharded multi-proveedor (5 variables de sexismo)
├── merge_shards.py   # une los CSV por shard -> _FULL
├── metrics.py        # accuracy/kappa/F1 por variable (ruta y etiqueta por argv)
├── results/          # salidas (recomendado: una subcarpeta por modelo)
└── README.md
```
