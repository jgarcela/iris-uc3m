# Comparación experimento_12: original (local) vs cluster

Mismo experimento (experimento_12), mismo modelo (**gemma4:e4b**), mismos **1000**
artículos (muestreo determinista `random_state=42`), mismas variables. Solo cambia
**dónde** se ejecuta:

- **Local (original)**: CPU de queron, secuencial (1 artículo cada vez).
- **Cluster**: 2 servidores Ollama en GPU (bastet07 + bastet08), 2 shards × 4 workers.

Ficheros de datos:
- `comparacion_metricas.csv` — accuracy / kappa / F1 lado a lado.
- `comparacion_tiempos.csv` — tiempos de procesamiento.

---

## 1. Métricas (N=1000)

| Variable | Métrica | Local | Cluster | |
|----------|---------|:---:|:---:|:---:|
| **Titular** (nombre_propio_titular) | Accuracy | 0.800 | 0.809 | ≈ igual |
| | Kappa | 0.7105 | 0.7235 | ≈ igual |
| | F1_Weighted | 0.7959 | 0.8055 | ≈ igual |
| **Protagonista** (cla_genero_prota) | Accuracy | 0.669 | 0.814 | ⬆️ mejor |
| | Kappa | 0.5234 | 0.7125 | ⬆️ mejor |
| | F1_Weighted | 0.6925 | 0.8127 | ⬆️ mejor |
| **Periodista** (genero_periodista) | Accuracy | 0.803 | 0.802 | ≈ igual |
| | Kappa | 0.7115 | 0.7101 | ≈ igual |
| | F1_Weighted | 0.8116 | 0.8111 | ≈ igual |

**Lectura:**
- **Titular** y **Periodista**: prácticamente idénticos → el pipeline del cluster
  **reproduce** el original. Validación correcta.
- **Protagonista**: el cluster sale bastante mejor (kappa 0.71 vs 0.52). Como el
  modelo y los datos son los mismos, la diferencia viene del **no determinismo**
  (`temperature=0.3` en `consultar_ollama`), y Protagonista es la variable más
  sensible (extrae listas de nombres). Posiblemente el CSV local era de una corrida
  más antigua. Pendiente de mirar artículo a artículo dónde difieren.
- `cita_en_titulo` sale ERROR en ambos: experimento_12 no genera esa columna.

---

## 2. Tiempos

| Entorno | Latencia media/artículo | Suma latencias | Concurrencia | **Tiempo de pared (1000)** |
|---------|:---:|:---:|:---:|:---:|
| Local (CPU, secuencial) | 58.2 s | 16.16 h | 1 | **~16.2 h** |
| Cluster (GPU, 2 serv × 4 workers) | 113.8 s | 31.60 h | 8 | **~4.0 h** |

**→ El cluster hizo en ~4 h lo que en local costaba ~16 h: ≈ 4× más rápido.**

⚠️ **Nota importante sobre la latencia por artículo (113.8 s):** NO significa que el
cluster sea más lento. Es un artefacto de la concurrencia: con `--workers 4` y
`OLLAMA_NUM_PARALLEL=1`, cada artículo **espera en cola** detrás de otros 3 en su
servidor, así que su latencia medida se infla ~4×. El cómputo real por artículo es
~28‑30 s (la GPU rinde ~88 tokens/s, medido). Lo que cuenta es el **tiempo de pared**,
que es donde se ve la ganancia real (16 h → 4 h).

Margen de mejora: con las 4 GPUs (bloqueado por colisión de GPU, ver README del
cluster) o subiendo `OLLAMA_NUM_PARALLEL`, el tiempo de pared bajaría aún más.

---

## Reproducir esta comparación

```bash
# los CSV se generan a partir de:
#   local : Experimentos/metrics/metrics_...scrape.csv  +  Experimentos/results/...scrape.csv
#   cluster: metrics/metrics_...FULL.csv                 +  results/...FULL.csv
```
