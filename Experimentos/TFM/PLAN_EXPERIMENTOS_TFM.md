# Plan de experimentos del TFM

Documento **canónico** de qué experimentos entran en la memoria (y cuáles no).
 complementary: `DIARIO_TFM.md` (día a día), `ESTADO_LATEX.md` (huecos del `.tex`), notebook `Experimentos.ipynb` (números).

> **Idea central**  
> Evaluar LLMs en la detección automática de **5 variables de lenguaje sexista** (elegidas entre 15 por recomendación de expertas), con un **benchmark multi-proveedor** (local + Gemini + Claude + OpenAI) y, como **core**, comparar ese baseline frente a una **arquitectura de agentes / Agent Skills** sobre los mismos modelos y el mismo corpus.

---

## 1. Objeto de estudio

### Corpus y evaluación

| Elemento | Valor de trabajo |
|----------|------------------|
| Corpus completo | IMIO / IRIS_IAMEDIA scrapeado (`…clara_scrape.csv`): **~7 115** noticias (2017–2021 + 2024) |
| Inferencia (corrida TFM) | **Toda la BBDD** (~7k+): mismo conjunto para todos los modelos |
| Ground truth / métricas | Subconjunto **anotado en las 5 variables**: **1 315** noticias (las 5 etiquetadas a la vez) |
| Cómo filtrar GT en cluster | `--only-labeled` en `main_cluster.py` (útil para pruebas de métricas; la corrida oficial procesa todo) |
| Métricas | Accuracy, Cohen’s Kappa, F1 macro (por variable) sobre las 1 315 — scripts `metrics.py` |
| Pipeline | Texto scrapeado; salida código + explicación + evidencias |

**Importante:** el benchmark compara modelos sobre la **misma** corrida a escala full (~7k). Las tablas de acuerdo con expertas se calculan solo donde hay etiqueta (≈1.3k). El resto de predicciones sirve para análisis descriptivo / PoC editorial, no para kappa/F1.

> Nota: corridas previas del cluster usaron a menudo un muestreo de 1000 arts. de 2024 (`random_state=42`). Eso queda como **piloto**; el diseño TFM es full BBDD + evaluación en las 1 315 anotadas.

### Las 15 variables de lenguaje (inventario completo)

Definidas en `Experimentos/variables.json` (V25–V39), inspiradas en Sainz de Baranda, García Meseguer, Lledó, Bengoechea, etc.

| Cód. | Clave | En el TFM |
|------|-------|-----------|
| V25 | `lenguaje_sexista` | **Sí (core)** |
| V26 | `masc_generico` | **Sí (core)** |
| V27 | `hombre_denominar_humanidad` | No (contexto / anexo) |
| V28 | `uso_dual_zorr` | No |
| V29 | `uso_cargo_mujer` | No |
| V30 | `sexismo_discurso` | **Sí (core)** |
| V31 | `androcentrismo` | No |
| V32 | `mencion_nombre_investigadora` | No |
| V33 | `asimetria_mujer_hombre` | **Sí (core)** |
| V34 | `disminutivos_infantilizacion` | No |
| V35 | `denominacion_sexualizada` | **Sí (core)** |
| V36 | `denominacion_redundante` | No |
| V37 | `denominacion_dependiente` | No |
| V38 | `criterios_excepcion` | No |
| V39 | `comparacion_mujer_hombre` | No |

**Selección de las 5:** criterio de expertas en comunicación / perspectiva de género (proyecto IRIS). En la memoria: justificar la selección; no hace falta reportar las 15 como experimentos principales (sí como marco teórico / codebook).

---

## 2. Ejes experimentales del TFM

Dos ejes ortogonales sobre las **mismas 5 variables** y el **mismo muestreo**:

```
                    ┌─────────────────────────┐
                    │  EJE A — Benchmark      │
                    │  modelo × proveedor     │
                    │  (prompt / JSON baseline)│
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │  EJE B — Core TFM       │
                    │  Agentes + Agent Skills │
                    │  (misma tarea, otro     │
                    │   empaquetado method.)  │
                    └─────────────────────────┘
```

### Eje A — Benchmark de modelos (baseline)

Comparar proveedores en zero/few-shot con la metodología operativa en `variables.json` (+ plantilla prompt Clara / e3), **sin** skills/orquestador.

| Familia | Modelos objetivo | Notas |
|---------|------------------|-------|
| Local (Ollama / granja TSC) | `gemma4:e4b`, `qwen3:8b` | Al menos uno; ideal ambos |
| OpenAI | p.ej. `gpt-4o-mini` (y/o `gpt-5-nano` si se mantiene) | Elegir 1–2 y fijarlos en la memoria |
| Anthropic | p.ej. `claude-haiku-4-5-20251001` | Representante Claude |
| Google | p.ej. `gemini-3.1-flash-lite` o `gemini-2.5-flash` | Representante Gemini |

**Condiciones a fijar (evitar manzanas con naranjas):**

- Mismos artículos (**toda la BBDD ~7k**), mismas 5 variables.
- **JSON principal: `variables_umbral_bajo.json`** (alineado al GT / anotación experta en V25–V26).
- Ablación documentada: estricto (`variables.json`) vs umbral bajo en al menos 2 modelos (`gpt-4o-mini` Exp 18/18bis, `gpt-5-nano` Exp 17/17bis).
- Misma plantilla de prompt baseline.
- Un `--output-dir` por modelo.
- Métricas siempre sobre el mismo subconjunto GT (**1 315** anotadas).

### Eje B — Core: agentes y Agent Skills

Misma tarea de clasificación, cambiando la **arquitectura de inferencia**:

| Condición | Descripción | Ancla en repo |
|-----------|-------------|----------------|
| B0 Baseline | Una llamada / variable (o pipeline actual `variables.py`) con prompt + codebook | Exp 13–15, 17–19 |
| B1 Skills | Metodología en `SKILL.md` (orquestador + skill por variable) inyectada al system prompt | Exp 16 (`clasificador_skills.py`, `skills/`) |
| B2 Agentes (ampliar) | Arquitectura multi-agente (roles: detección, explicación, revisión…) sobre los mismos modelos | Parcialmente explorado en `pruebas_skills_*`; **pendiente formalizar como experimento TFM** |

**Hipótesis del core:** empaquetar la metodología experta en skills/agentes mejora coherencia, trazabilidad y/o acuerdo con el GT frente al baseline del Eje A, sin cambiar el codebook.

Skills actuales (Exp 16):

```
experimento_16/skills/
├── orquestador/SKILL.md
├── lenguaje_sexista/SKILL.md      # V25
├── masc_generico/SKILL.md         # V26
├── sexismo_discurso/SKILL.md      # V30
├── asimetria_mujer_hombre/SKILL.md # V33
└── denominacion_sexualizada/SKILL.md # V35
```

**Objetivo de cobertura Eje B:** repetir B1 (y si da tiempo B2) no solo en Claude, sino también en **local + OpenAI + Gemini**, para que el core sea comparable al benchmark.

---

## 3. Matriz TFM (qué cuenta y estado)

Leyenda estado: ✅ hecho / datos en disco · 🟡 parcial · ⬜ pendiente · — fuera de alcance TFM

### 3.1 Eje A — Baseline por modelo

| Modelo | Proveedor | Exp / carpeta de referencia | Estado | Notas |
|--------|-----------|-----------------------------|--------|-------|
| `gemma4:e4b` | Local | Exp 13 · `CLUSTER/experimento_13_cluster` | ✅ | FULL en results |
| `qwen3:8b` | Local | Exp 14 | 🟡 | Scripts listos; confirmar FULL + métricas en memoria |
| `claude-haiku-4-5-…` | Anthropic | Exp 15 | 🟡 | Baseline Claude; confirmar métricas publicables |
| `gpt-5-nano` / `gpt-4o-mini` | OpenAI | Exp 17–18 · cluster results | 🟡 | Hay carpetas `results/`; unificar qué ID entra en la tabla final |
| `gemini-3.1-flash-lite` | Google | Exp 19 | 🟡 | Hay run con `umbral_bajo`; alinear condición con el resto |

### 3.2 Eje B — Skills / agentes

| Condición | Modelo(s) | Referencia | Estado |
|-----------|-----------|------------|--------|
| B1 Skills | Claude Haiku | Exp 16 | 🟡 | Existe; falta contraste limpio 15 vs 16 en Cap. 5 |
| B1 Skills | Gemma / Qwen | — | ⬜ | **Prioritario** para el core multi-modelo |
| B1 Skills | OpenAI | — | ⬜ | |
| B1 Skills | Gemini | — | ⬜ | |
| B2 Multi-agente | ≥1 API + ≥1 local | diseñar a partir de `pruebas_skills_*` | ⬜ | Definir diseño antes de correr a escala |

### 3.3 Ablaciones opcionales (solo si aportan a la narrativa)

| Ablación | ¿TFM? | Comentario |
|----------|-------|------------|
| `variables.json` vs `variables_umbral_bajo.json` | **Sí (corta)** | Principal = umbral bajo. Ablación en gpt-4o-mini + gpt-5-nano: V25/V26 ↑ recall/acc/kappa; V30/V33/V35 ≈ sin cambio |
| Prompt e1/e2/e3 en vars de **nombres** (Exp 1–12) | **No como eje principal** | Trabajo exploratorio previo; mención breve en metodología si se quiere |
| Exp 20 (duplicado Gemini) | No | Fusionar con 19 o eliminar del notebook |

---

## 4. Qué NO entra como experimento principal del TFM

Aunque exista en `Experimentos/experiments/`:

| Bloque | Exps | Motivo |
|--------|------|--------|
| Variables de género en nombres / periodista | 1–12 | Otro objeto de estudio; no son las 5 de lenguaje elegidas |
| Barrido amplio de modelos locales (mistral, deepseek, llama…) | 4–7 | Exploración; el TFM fija gemma ± qwen |
| Interspeech (15 vars) | `experimento_interspeech` | Marco / paper paralelo; no sustituye el benchmark TFM |
| RAG en generación excel | histórico 2025–01 | Fuera del diseño actual HITL + skills |

Se pueden citar como **trabajo preliminar** o **descartado**, no como tablas centrales del Cap. 5.

---

## 5. Diseño mínimo publicable (MVP memoria)

Para que Cap. 5 cierre con una historia coherente:

1. **Método** — BBDD completa (~7 115) para inferencia; evaluación en las **1 315** anotadas; prompt/JSON; cluster + multi-proveedor.
2. **Tabla benchmark (Eje A)** — 5 variables × {gemma, qwen?, claude, openai, gemini} con kappa/F1 **sobre las 1 315**.
3. **Tabla core (Eje B)** — al menos **baseline vs skills** en el mismo modelo (Claude Exp 15 vs 16) + idealmente skills en 1 local y 1 API más (mismas 1 315).
4. **Discusión** — ¿skills ayudan? ¿local competitivo? coste/latencia a escala 7k; cobertura de anotación (~18 % de la BBDD).

Checklist operativo:

- [ ] Fijar IDs exactos de modelo que aparecerán en el LaTeX (una fila por familia).
- [x] JSON principal = **umbral bajo**; estricto solo como ablación (2 modelos).
- [ ] Re-ejecutar / completar corridas benchmark/skills con `--variables-json …/variables_umbral_bajo.json` (full BBDD o al menos las 1 315).
- [ ] Calcular métricas solo en las 1 315 anotadas; documentar N en cada tabla.
- [ ] Completar contraste Exp 15 vs 16 (y skills en más proveedores).
- [ ] Decidir si B2 multi-agente entra en esta entrega o queda como línea futura.
- [ ] Actualizar Cap. 5 del LaTeX según este plan (no según Exp 1–12).

---

## 6. Mapa repo → experimento TFM

| Rol TFM | Rutas |
|---------|-------|
| Definiciones 15→5 | `Experimentos/variables.json` |
| Clasificación baseline | `Experimentos/variables.py`, `utils.py` |
| Baseline local sexismo | `experiments/experimento_13`, `_14` + `CLUSTER/experimento_13_cluster` |
| Baseline Claude | `experiments/experimento_15` |
| Skills (core) | `experiments/experimento_16/` (+ `skills/`) |
| Baseline OpenAI / Gemini | `CLUSTER/experimento_17_cluster` … `_19_cluster` |
| Infra granja | `CLUSTER/README.md`, `CLUSTER/cluster_tsc/` |
| Números / figuras | `Experimentos/Experimentos.ipynb` |

---

## 7. Narrativa sugerida para el LaTeX (Cap. 5)

1. **Marco:** de 15 variables a 5 (expertas).  
2. **Benchmark:** rendimiento por proveedor (Eje A).  
3. **Core:** efecto de Agent Skills / agentes (Eje B) sobre el mejor o sobre todos.  
4. **Implicaciones:** transferencia editorial HITL, coste local vs API, limitaciones.

---

## 8. Decisiones abiertas (rellenar juntos)

| # | Pregunta | Decisión |
|---|----------|----------|
| 1 | ¿Gemma y Qwen, o solo uno en la tabla final? | |
| 2 | OpenAI: ¿`gpt-4o-mini`, `gpt-5-nano`, o ambos? | |
| 3 | Gemini: ¿flash-lite u otro ID? | |
| 4 | ¿Tabla principal con `variables.json` o umbral bajo? | **Umbral bajo** (2026-07-20). Estricto = ablación Cap. 5/discusión. |
| 5 | ¿B2 multi-agente en esta entrega o “líneas futuras”? | |
| 6 | ¿Skills en todos los proveedores o solo contraste Claude + 1 local? | |
| 7 | ¿Las corridas full 7k se hacen siempre, o en algún modelo solo se evalúa en las 1 315 por coste? | |
