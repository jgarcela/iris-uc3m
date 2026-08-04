# Experimento 21 en el cluster TSC — Agent Skills con `gemma4:e4b`

Corre la arquitectura de agentes del Exp 21 sobre la granja Ollama, en dos
niveles comparables con el benchmark de API (Gemini/OpenAI):

- **B0** (`--baseline`): metodología inyectada en el prompt, sin herramientas.
- **B1** (por defecto): Agent Skills (progressive disclosure + tools + RAG).

Reutiliza la infraestructura del cluster (ver [`../README.md`](../README.md)):
`launch_process.py -c llm.json` levanta los Ollama en la granja desde amaterasu;
este `main_cluster.py` corre en **queron** y les manda peticiones por HTTP.

Salida compatible con `Experimentos/experiments/experimento_21_agentskills/metrics.py`
(columnas `modelo_<var>` + traza `<var>_n_tools`, `<var>_colapso_b0`, `<var>_error`).

## Variables de entorno comunes

```bash
export EXP=/home/jggomez/Desktop/IRIS/iris-uc3m/Experimentos
export AGE=$EXP/experiments/experimento_21_agentskills
export DATA="/home/jggomez/Desktop/IRIS/iris-uc3m/data/2026_02_10_imio_def_todo_envio_heidy.xlsx - 2026_02_09_imio_def_todo_clara_scrape.csv"
```

> **Importante:** genera antes las skills con el JSON canónico (estricto), para
> que coincida con el benchmark de API:
> `cd $AGE && python3 generar_skills.py --json ../../variables.json`

## Lanzamiento (ejemplo con 2 shards sobre 2 servidores Ollama)

### B0 — baseline sin skills (1 llamada/variable, rápido)

```bash
# shard 0 → bastet07 ; shard 1 → bastet08 (ajusta host:puerto a tu llm.json)
OLLAMA_HOST=bastet07:11434 python main_cluster.py --model gemma4:e4b --baseline \
  --shard 0 --n-shards 2 --workers 4 --only-labeled \
  --experimentos-dir "$EXP" --agente-dir "$AGE" --data "$DATA" \
  --output-dir ./results_b0 &

OLLAMA_HOST=bastet08:11434 python main_cluster.py --model gemma4:e4b --baseline \
  --shard 1 --n-shards 2 --workers 4 --only-labeled \
  --experimentos-dir "$EXP" --agente-dir "$AGE" --data "$DATA" \
  --output-dir ./results_b0 &
```

### B1 — Agent Skills (≈10 llamadas/variable, MUCHO más pesado)

Igual pero sin `--baseline` y con `--output-dir ./results_b1`. Conviene subir
`--n-shards` (más servidores) porque B1 multiplica el número de llamadas.

## Prueba rápida antes de lanzar en serio

```bash
LIMIT=2 OLLAMA_HOST=bastet07:11434 python main_cluster.py --model gemma4:e4b \
  --baseline --shard 0 --n-shards 1 --workers 1 --only-labeled --limit 2 \
  --experimentos-dir "$EXP" --agente-dir "$AGE" --data "$DATA" --output-dir /tmp/t21
```

## Fusionar shards y calcular métricas

```bash
python merge_shards.py --input-dir ./results_b0 --output ./results_b0/FULL.csv
# métricas contra el GT (reutiliza el metrics.py del Exp 21):
python "$AGE/metrics.py" --pred ./results_b0/FULL.csv --corpus "$DATA"
```

## Notas de rendimiento

- **B0** es viable: ~1 llamada por variable. Con `--only-labeled` son 1.313
  artículos; repartidos en N shards × workers, cuestión de horas.
- **B1 es caro en cómputo**: cada agente hace hasta ~10 llamadas. Sobre modelos
  locales esto es el orden de decenas de segundos por variable. Lanza B1 con el
  máximo de shards/servidores disponibles y espera tiempos largos. Este coste
  computacional es, en sí mismo, un resultado a reportar (Sección de coste del
  Cap. 5).
- Ambos niveles son **reanudables** por `IdNoticia`: si se corta, relanza el
  mismo comando y continúa.
- Prompt caching queda desactivado (`agente.USAR_PROMPT_CACHE = False`); Ollama
  lo ignora igualmente.
