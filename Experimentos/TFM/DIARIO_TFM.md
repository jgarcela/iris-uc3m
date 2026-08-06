# Diario de trabajo — TFM IRIS / UC3M

Registro diario de cambios, experimentos y decisiones técnicas.
Sirve como memoria operativa para redactar la memoria del TFM (metodología, ablaciones, resultados y limitaciones).

> **Cómo usarlo**
> - Una entrada por día (más reciente arriba).
> - Por cada cambio: **qué**, **dónde** (experimento / ficheros), **por qué**, **resultado / siguiente paso**.
> - Si un experimento se re-ejecuta, anotar modelo, JSON de variables, muestra y métricas clave.
> - Al escribir el TFM: filtrar por experimento o por tema (prompts, cluster, skills, umbral, etc.).

---

## Índice rápido de experimentos

| Exp | Mes | Enfoque | Modelo(s) destacados | Notas |
|-----|-----|---------|----------------------|-------|
| 1 | 02/2026 | Baseline variables generales (prompt_e1) | — | Primer pipeline sobre corpus IMIO |
| 2 | 02/2026 | Retoque prompts variables de género | — | Kappa/F1 no mejoran |
| 3 | 03/2026 | Sync estricta listas + `gender_guesser` | — | Foco en género de nombres |
| 4 | 03/2026 | Igual que 3, otro modelo | `llama3.1:8b` | Comparativa local |
| 5 | 03/2026 | Igual, otro modelo | `qwen3:8b` | Referencia local fuerte |
| 6 | 03/2026 | Subconjunto vars género | `mistral:7b` | |
| 7 | 03/2026 | Subconjunto vars género | `deepseek-r1:1.5` | |
| 8 | 03/2026 | Vars género, prompt_e2 | (como exp 5) | Ablación de prompt |
| 9 | 03/2026 | Vars género, prompt_e1 | (como exp 5) | Ablación de prompt |
| 10 | 05/2026 | Vars género, API | `claude-haiku-4-5` | |
| 11 | 05/2026 | Vars género, API | `claude-opus-4-7` | |
| 12 | 05/2026 | Vars género, local granja | `gemma4:e4b` | Primera corrida cluster TSC |
| 13 | 05/2026 | 5 vars sexismo | `gemma4:e4b` | Pivot a variables de lenguaje sexista |
| 14 | 05/2026 | 5 vars sexismo | `qwen3:8b` | |
| 15 | 05/2026 | 5 vars sexismo | `claude-haiku-4-5` | Prompt Clara / e3 |
| 16 | 05/2026 | 5 vars sexismo + skills (~~core~~) | `claude-haiku-4-5` | **DESCARTADO**: no es Agent Skills real → sustituido por Exp 21 |
| 17 | 07/2026 | 5 vars sexismo, multi-proveedor | `gpt-5-nano` | Cluster + APIs; umbral bajo |
| 18 | 07/2026 | 5 vars sexismo, multi-proveedor | `gpt-4o-mini` | Cluster + APIs; umbral bajo |
| 18 bis | 07/2026 | Variante cluster de 18 | configurable | `--variables-json` |
| 19 | 07/2026 | 5 vars sexismo, multi-proveedor | `gemini-3.1-flash-lite` | Cluster + APIs; umbral bajo |
| 20 | 07/2026 | (borrador en notebook) | `gemini-3.1-flash-lite` | Revisar si es entrada real o duplicado |
| 21 | 07/2026 | **Core Eje B — Agent Skills** (5 agentes especializados) | multi-proveedor | Sustituye al 16. **B0 vs B1 sobre 1 313**: las skills NO mejoran κ y cuestan 2.2× |
| Interspeech | 02/2026 | Set amplio vars sexismo (15+) | — | Línea paralela / paper |

**5 variables de sexismo (exp 13+):** `lenguaje_sexista`, `masc_generico`, `sexismo_discurso`, `asimetria_mujer_hombre`, `denominacion_sexualizada`.

---

## Entradas diarias

### 2026-08-04 — gemma B1 en cluster, ablación cruzada, equipo_ia y redacción Cap 3–5

**Contexto:** cerrar la parte experimental que faltaba (modelo local B1 y desglose de componentes), añadir el análisis del modelo como anotador, y reescribir el Cap 5 entero + consistencia Cap 3–4 + limpieza del TFG anterior en el `.tex`.

#### Experimentos nuevos

| # | Qué | Rutas | Resultado |
|---|-----|-------|-----------|
| 1 | **Ablación por componentes (5 brazos × 4 modelos)** | flags `--sin-consultar-guia` y `--sin-resumenes-guias` en `agente.py`/`main.py` | La **tool RAG en vivo nunca ayuda** (peor o casi peor κ en los 4 modelos); el mecanismo de skills solo mejora a **gemini**. Mejor config = skills + resúmenes **sin tool** |
| 2 | **gemma B1 en cluster (5 brazos)** | `CLUSTER/experimento_21_cluster/` (RTX 4090, sharded), flags añadidos a `main_cluster.py`; `lanzar_b1_ablacion.sh`, `estado_b1.sh` | gemma B0 tiene el **mayor κ de control** de los 4 (+0,061); B1 lo empeora (como los gpt), no como gemini. ~54 s/artículo, ~5 h/variante con 4 GPU |
| 3 | **El sistema como tercer anotador** | `experimento_21_agentskills/equipo_ia.py` (B0 y B1) | Los 3 modelos caen **fuera del rango humano** (extremo permisivo). κ intra-IA 0,014–0,084: **no hay "equipo IA"**, cada modelo es idiosincrásico. B1 los hace aún más dispares |

#### Redacción (LaTeX)

- **Cap 5 reescrito**: `iris_experimento21.tex` → **`experimentacion.tex`**; añadidos modelo local (gemma), descomposición 5 brazos × 4 modelos, y "el sistema como anotadora"; quitada la ablación de configuración (caching/umbral). `results_and_discussion.tex` (intro) y `discussion.tex` reescritos.
- **Consistencia Cap 3–4**: `eval` = evaluación zero-shot sobre las 1.313 (fuera dev/test); `rag` con nota de que la tool es prescindible; jerarquía de `iris_analisis_experto` y dedup del bloque "definición ejecutable".
- **qwen3 y claude eliminados de todo el TFM** (tablas, prosa, menciones de Anthropic).
- **Limpieza `.tex`**: 42 ficheros residuales del TFG viejo borrados; `main.tex` sin `\iffalse` ni `%\input`/`%\section` comentados; intros de capítulo en español.
- **REUNION_DIRECTORA.md** ampliado (métricas por variable, por anotador Indexa/UCM3 e intra-equipo, infra-detección, ablación cruzada).

#### Decisiones

- **exp22 (confianza) queda FUERA del TFM.** El experimento existe (`experimento_22_confianza/`) pero no se redacta.
- **Estructura Cap 5** = intro + "Experimentación" (experimentacion.tex) + "Discusión". No se crea capítulo aparte de experimentación.
- Nota de infra: gemma B0 se corrió en RTX 3090 y B1 en RTX 4090 por disponibilidad; la GPU afecta al throughput, no a las salidas.

#### Pendiente

- [ ] Cap. 6 (conclusiones + líneas futuras) — sigue siendo #SeAcabó.
- [ ] Abstract: alinear con lo hecho (Agent Skills, B0/B1); ahora describe la propuesta antigua.
- [ ] Backup de `results_b1_*` del cluster (git los ignora).

#### Commits del día (git)

- `9d68d09` — Exp 21: ablación 4 modelos × 5 brazos, exp22 confianza y redacción TFM (Caps. 3 y 5).
- `981cded` — TFM: reescritura del Cap 5, consistencia Cap 3-4 y limpieza del TFG anterior.

---

### 2026-07-21 — Exp 21 a escala: Eje A + core B0 vs B1 sobre las 1 313

**Contexto:** cerrar el Exp 21 con corridas a escala real y ablaciones controladas.

#### Resultados principales (N = 1 313, 3 proveedores, sin reasoning, sin caching)

**Eje B — core (B0 metodología inyectada vs B1 Agent Skills):**

| Modelo | B0 acc / κ | B1 acc / κ | Δκ | coste B1/B0 |
|--------|-----------|-----------|-----|-------------|
| gemini-3.1-flash-lite | 0.600 / +0.026 | 0.661 / **+0.066** | **+0.040** | 2.3× |
| gpt-4o-mini | 0.564 / +0.033 | 0.612 / +0.014 | −0.019 | 2.7× |
| gpt-5.4-nano | 0.571 / +0.052 | 0.569 / +0.004 | −0.048 | 1.9× |
| **media** | 0.578 / **+0.037** | **0.614** / +0.028 | −0.009 | 2.2× |

- **Las Agent Skills NO mejoran el acuerdo.** κ medio peor (+0.028 vs +0.037) a 2.2× coste. Solo Gemini mejora.
- **Accuracy y κ apuntan en direcciones opuestas** → reportar solo accuracy llevaría a la conclusión contraria.
- Coste total: B1 $30.91 + B0 $13.83 = **$44.74**.

**Ablaciones (todas sobre gpt-4o-mini, N=100 salvo indicación):**

| Ablación | Efecto |
|---|---|
| Prompt caching (`IRIS_CACHE_BREAK`) | +87 % coste, κ sin cambio. Además **altera el comportamiento**: parte el prompt en roles system/user (OpenAI/Gemini) y duplica el uso de tools. En Anthropic no cambia roles. OpenAI ya cachea automáticamente (59 % aciertos sin hacer nada) |
| Skills-resumen de guías en catálogo | −20 % tokens al quitarlas; sin efecto claro en κ |
| `variables.json` vs `umbral_bajo` | Umbral bajo sube accuracy (0.644→0.688) **destruyendo señal**: lift de `masc_generico` 1.25 → 1.01. κ no mejora de forma real |

**Diagnóstico (análisis de lift = precisión ÷ tasa base):**
- Solo `sexismo_discurso` tiene señal consistente (lift ≈ 2).
- `lenguaje_sexista` lift ≈ 1.0 → **sin capacidad discriminativa**; su GT (81 % positivos) sugiere criterio experto mucho más amplio que la regla de inversión de la skill → desajuste de constructo.
- `denominacion_sexualizada` no marca positivos.
- Infra-detección transversal (posible causa: exigencia de evidencia literal).

#### Decisiones

- **Config canónica:** sin caching, sin reasoning (todos los proveedores), `variables.json` estricto, evaluación en las 1 313.
- **Reasoning desactivado** en `utils.py`: OpenAI GPT-5.4+ `reasoning_effort='none'`, GPT-5 antiguos `'minimal'`, Gemini `thinking_budget=0`, Claude sin `thinking`.
- **Claude fuera del benchmark** por coste (~$69 en B1: Anthropic no tiene caching automático). Viable en B0 por ~$17 si se decide incluirlo.
- **Locales fuera de alcance sin cluster**: ~8 min/artículo → ~175 h.
- **Sin doble codificación** en el corpus (una anotadora por noticia) → declarar como limitación; no se puede estimar el techo de acuerdo.
- Revisar §8.4 (umbral bajo como tabla principal): sus ganancias podrían ser el mismo artefacto de marginales.

#### Cambios de código

| # | Qué | Rutas |
|---|-----|-------|
| 1 | Nivel B0 (baseline sin skills) | `agente.py::clasificar_variable_baseline`, `main.py --baseline` |
| 2 | Flags de ablación | `--sin-cache`, `--sin-resumenes-guias` |
| 3 | Tokens/coste por artículo y variable | `costes.py`, `main.py`, `metrics.py` |
| 4 | Corta-circuitos ante fallos de saldo/API | `main.py` (`MAX_FALLOS_SEGUIDOS`) + columnas `<var>_error` |
| 5 | Precios verificados + `gpt-5.4-nano` | `costes.py`, `utils.py` |
| 6 | Runners y seguimiento | `run_benchmark.sh`, `estado_bench.sh [b0]` |
| 7 | Trazabilidad del JSON en las skills | `generar_skills.py` (`origen_json` en frontmatter) |

#### Pendiente

- [ ] Cap. 5 con estas tablas; Cap. 4 con la arquitectura.
- [ ] Decidir si entra Claude en B0 (~$17).
- [ ] Portar Exp 21 al cluster TSC para el eje local (o declararlo fuera de alcance).

---

### 2026-07-20 (tarde-2) — Core Eje B: Exp 21 Agent Skills (nuevo, limpio)

**Contexto:** los experimentos de skills previos (Exp 16, `pruebas_skills_*`) están **mal** como "Agent Skills" (sin progressive disclosure; ablación 15-vs-16 confundida con el nº de llamadas). Se descartan y se rehace el core desde cero.

#### Cambios / trabajo del día

| # | Qué | Rutas | Detalle |
|---|-----|-------|---------|
| 1 | Diseño canónico del core | `TFM/ARQUITECTURA_AGENTES_SKILLS.md` | Terminología skill≠agente, auditoría, niveles B0/B1/B2, stack |
| 2 | Experimento nuevo | `experiments/experimento_21_agentskills/` | 5 agentes especializados (1 por variable), 5 llamadas (comparable a B0=Exp15) |
| 3 | Skills | `.../skills/` | 5 de variable (regeneradas desde `variables.json`, frontera limpia) + 3 auxiliares (`guia_regla_inversion`, `guia_lenguaje_inclusivo`, `verificar_evidencias`) |
| 4 | Runtime | `tools.py`, `agente.py`, `main.py`, `generar_skills.py` | Bucle ReAct de texto sobre `utils.consultar_ollama` (uniforme 4 proveedores, sin LangChain) |
| 5 | Guías de lenguaje promovidas | `pruebas_skills_ollama/methodology/` → `Experimentos/methodology/` | `git mv` de todo (md+pdf+docx+tesis, ~31 MB) + `methodology_manifest.json`; defaults de `analyze_article.py` actualizados |

#### Decisiones / notas

- **Descartados:** Exp 16 y `pruebas_skills_*` (no son Agent Skills reales).
- **Agente especializado = 1 por variable** con progressive disclosure (ve `name`+`description`, carga `SKILL.md` bajo demanda). Colapso a B0 (no usa tools) se registra como métrica.
- Ablación limpia: B0 (Exp 15) y B1 (Exp 21) ambos a **5 llamadas**; única diferencia = metodología inyectada vs cargada.
- Corrige bugs del 16: evidencias vs texto real, `str(dict)` en frontera, JSON por variable (un fallo no arrastra las 5).

#### Pendiente / siguiente

- [ ] Corrida real B1 por proveedor (Claude/OpenAI/Gemini/local) + tasa de uso de tools.
- [ ] `metrics.py` sobre las 1 315; tabla B0 vs B1.
- [ ] Prompt caching a escala 7k.

---

### 2026-07-20 (tarde) — Exp 17 bis + decisión: umbral bajo como principal

**Contexto:** ablación estricto vs umbral en `gpt-5-nano`; cerrar criterio del JSON para el TFM.

#### Cambios / trabajo del día

| # | Qué | Rutas | Detalle |
|---|-----|-------|---------|
| 1 | Runner 17 bis | `CLUSTER/experimento_17_bis_cluster/` | `gpt-5-nano` + `variables_umbral_bajo.json`, N=1315 |
| 2 | Métricas 17 bis | `metrics/…17bis…FULL.csv` | V25/V26: Acc +0.45/+0.23, Kappa +0.19/+0.10 vs Exp 17 estricto |
| 3 | **Decisión TFM** | `PLAN_EXPERIMENTOS_TFM.md` §8.4 | Tabla principal = **umbral bajo**; estricto = ablación (2 modelos) |

#### Notas

- Efecto umbral replicado en `gpt-4o-mini` y `gpt-5-nano` (solo V25/V26; resto ≈ igual).
- En memoria: declarar desvío respecto a regla de inversión estricta de las guías.

---

### 2026-07-20 — LaTeX en el repo + plan canónico de experimentos TFM

**Contexto:** traer la memoria LaTeX junto a los experimentos; definir qué experiments cuentan para el TFM (no todo lo de la carpeta).

#### Cambios / trabajo del día

| # | Qué | Experimentos / rutas | Detalle | Por qué / impacto TFM |
|---|-----|----------------------|---------|------------------------|
| 1 | Carpeta TFM en el repo | `Experimentos/TFM/` | Contiene zip + fuentes descomprimidas | Trabajo LaTeX y experimentos en el mismo sitio |
| 2 | Fuentes LaTeX | `TFM___JORGE_GARCELA_N_GO_MEZ/Plantilla_TFG_ingles_2019/` | Plantilla UC3M; Caps. 1–2 ya orientados a IRIS | Base de la memoria |
| 3 | Diario movido aquí | `Experimentos/TFM/DIARIO_TFM.md` | Antes en `Experimentos/` | Todo el material de redacción junto |
| 4 | Checklist estado LaTeX | `Experimentos/TFM/ESTADO_LATEX.md` | Mapa cap. ↔ experimentos + bugs `\input` | Guía de qué reescribir primero |
| 5 | Plan de experimentos TFM | `Experimentos/TFM/PLAN_EXPERIMENTOS_TFM.md` | 5 vars (de 15) + benchmark multi-proveedor + core agents/skills | Alcance oficial Cap. 5; Exp 1–12 fuera del eje principal |

#### Decisiones / notas

- Cap. 1 (intro Bindi/IRIS) y Cap. 2 (estado del arte) están **alineados** con el TFM actual.
- Cap. 3–5 arrastran mucho **texto del TFG anterior** (#SeAcabó, Análisis General / Insultos, ML/DL, corpus Twitter).
- **Alcance experimental TFM:** 5 vars de lenguaje (V25, V26, V30, V33, V35) + benchmark (gemma/qwen, Gemini, Claude, OpenAI) + core agents/skills. Exp 1–12 = preliminar, no tablas centrales.
- **Muestra:** inferencia sobre **toda la BBDD (~7 115)**; evaluación vs expertas en las **1 315** anotadas en las 5 vars (no el piloto de 1000×2024).
- Cap. 4 (`agents`, `prompts`, `rag`, `eval`, `poc`) son **stubs idénticos**.
- Varios `\input{Plantilla_TFG_ingles_2019/...}` rotos al compilar desde esa carpeta.
- El `.zip` queda fuera de git (`*.zip` en `.gitignore`).

#### Pendiente / siguiente

- [ ] Cerrar decisiones abiertas del §8 en `PLAN_EXPERIMENTOS_TFM.md` (IDs de modelo, umbral, alcance B2).
- [ ] Completar matriz: skills en local + OpenAI + Gemini (no solo Claude Exp 16).
- [ ] Arreglar rutas `\input` y reescribir Cap. 3–5 según el plan.
- [ ] Seguir `ESTADO_LATEX.md` + `PLAN_EXPERIMENTOS_TFM.md`.

---

### 2026-07-17 — Puesta al día del repo, cluster y experimentos 17–19

**Contexto:** consolidar trabajo acumulado (exp 9–19), runners del cluster TSC y variantes con `variables_umbral_bajo.json`.

#### Cambios / trabajo del día

| # | Qué | Experimentos / rutas | Detalle | Por qué / impacto TFM |
|---|-----|----------------------|---------|------------------------|
| 1 | Commit masivo de experimentos 9–19 | `Experimentos/experiments/experimento_{9..19}/` | Scripts `main`, `main_sin_newspaper`, `metrics`, dedup, `recod_genero` | Deja trazable la secuencia de ablaciones modelo/prompt/skills |
| 2 | Notebook actualizado | `Experimentos/Experimentos.ipynb` | Secciones markdown Exp 9–20 + Interspeech | Fuente narrativa para el capítulo de experimentos |
| 3 | Utils multi-proveedor | `Experimentos/utils.py` | Enrutado OpenAI / Anthropic / Gemini / Ollama por ID de modelo | Misma pipeline → comparación 1:1 entre proveedores |
| 4 | Variables umbral bajo | `Experimentos/variables_umbral_bajo.json` (+ cambios en `variables.json` / `variables.py`) | Definiciones más permisivas / umbral bajo para detección | Hipótesis: subir recall en vars de sexismo |
| 5 | Skills (exp 16) | `Experimentos/experiments/experimento_16/skills/` | Orquestador + skill por variable; metodología inspirada en guías CSD / tesis Clara | Comparar prompt monolítico (15) vs skills (16) |
| 6 | Pruebas skills Ollama/Claude + methodology | `pruebas_skills_*`, `methodology/` | Material de apoyo (guías, tesis Sainz de Baranda, etc.) | Justificación metodológica en el TFM |
| 7 | Cluster runners | `CLUSTER/experimento_{12,13,17,18,18_bis,19}_cluster/` | `main_cluster.py` sharded, `merge_shards.py`, `metrics.py`, READMEs | Escalado a 1000 arts. 2024 en granja GPU + APIs |
| 8 | Runbook cluster | `CLUSTER/README.md` | Arquitectura queron ↔ amaterasu ↔ Ollama en bastet | Reproducibilidad infraestructura |
| 9 | Resultados parciales en disco | `CLUSTER/experimento_17_cluster/results/gpt-5-nano`, `…/18/…/gpt-4o-mini`, `…/19/…/gemini-3.1-flash-lite_umbral_bajo` | Corridas multi-modelo | Material de tablas comparativas |
| 10 | Diario TFM | `Experimentos/TFM/DIARIO_TFM.md` | Plantilla + entrada + índice (ubicación actual) | Memoria para redacción |

#### Decisiones / notas

- Muestreo determinista (`random_state=42`) en runners cluster → mismos 1000 artículos 2024 entre modelos.
- Exp 17/18/19 en notebook marcan prompt como **«umbral bajo variables + prompt_e3»**.
- Exp 18_bis y 18/19 cluster aceptan `--variables-json` para elegir `variables.json` vs `variables_umbral_bajo.json`.
- Exp 20 en el notebook parece duplicado conceptual de 19: confirmar en próximos días si se mantiene o se redefine.

#### Pendiente / siguiente

- [ ] Completar métricas y tablas Exp 17–19 (umbral bajo vs estándar).
- [ ] Aclarar rol de Exp 20 y 18 bis en la narrativa del TFM.
- [ ] Rellenar RESULTADOS/CONCLUSIÓN en el notebook para Exp 13–19.
- [ ] Ir añadiendo aquí cada cambio futuro el mismo día que ocurra.

#### Commits del día (git)

- `e15251b` — Add experiments 9-19, multi-provider utils, and umbral_bajo variables.
- `d172af4` — Track CLUSTER runners and experimento_5 copy.

---

### 2026-07-16 — Cluster exp 17–18 y variantes locales

**Fuentes:** fechas de modificación en disco (`experimento_17`, `18`, `18_bis_cluster`).

| # | Qué | Experimentos | Detalle |
|---|-----|--------------|---------|
| 1 | Scripts locales exp 17–18 | `experiments/experimento_17`, `experimento_18` | Pipeline 5 vars sexismo |
| 2 | Cluster 17 | `CLUSTER/experimento_17_cluster` | Runner multi-proveedor + log `run_gpt-4o-mini.log` |
| 3 | Cluster 18 / 18 bis | `CLUSTER/experimento_18_cluster`, `experimento_18_bis_cluster` | Misma base; bis con merge/metrics etiquetados 18bis |

---

### 2026-07-15 — Cluster experimento 13

| # | Qué | Experimentos | Detalle |
|---|-----|--------------|---------|
| 1 | Cliente cluster exp 13 | `CLUSTER/experimento_13_cluster` | 5 vars sexismo sobre granja Ollama |

---

### 2026-07-14 — Cluster experimento 12 + config LLM

| # | Qué | Experimentos | Detalle |
|---|-----|--------------|---------|
| 1 | Cliente cluster exp 12 | `CLUSTER/experimento_12_cluster` | Vars género con `gemma4` / `qwen3`; logs `gemma_cli*`, `qwen_cli*` |
| 2 | Config lanzador | `CLUSTER/llm.json` | Puertos/modelos para `launch_process.py` en amaterasu |

---

### 2026-07-13 — Bootstrap cluster TSC

| # | Qué | Experimentos | Detalle |
|---|-----|--------------|---------|
| 1 | Código lanzador granja | `CLUSTER/cluster_tsc/` | `launch_process.py`, `Scheduler.py`, skel `llm.json` |
| 2 | Inicio documentación cluster | `CLUSTER/` | Base del runbook |

---

## Retroactiva breve (antes de este diario)

No hay entradas día a día previas; resumen a partir del notebook y del historial git para no perder el hilo al escribir el TFM.

### 2026-05 — Exp 10–16 (API + skills + pivot a sexismo)

- **10–12:** mismas vars de género que la línea 5–9; modelos Claude / Gemma.
- **13–15:** cambio de objeto de estudio a las 5 variables de sexismo (alineadas con Interspeech / tesis).
- **16:** empaquetado en Agent Skills; metodología en `variables.json` + docs en `methodology/`.

### 2026-03 — Exp 3–9 (género de nombres, ablación modelo/prompt)

- Énfasis en sincronización nombre↔código y `gender_guesser`.
- Barrido de modelos locales (llama, qwen, mistral, deepseek) y de prompts (e1/e2/e3).

### 2026-02 — Exp 1–2 + Interspeech

- **Exp 1:** baseline variables del codebook IMIO.
- **Exp 2:** mejora de prompts de género; resultados insuficientes → motiva Exp 3.
- **Interspeech:** inventario amplio de variables de sexismo (línea paper).

### 2026-01 — Pipeline de datos

- `generacion_excel`: scrapeo con newspaper + Gemma; limpieza autor/mes; validación códigos vía `config.ini`; eliminación de RAG en ese flujo.

### 2025 (pre-experimentos numerados)

- Pruebas de declaraciones, RAG, agentes Ollama, embeddings, pipeline por categorías de variables (`tst.py`, agentes FCA, etc.). Base conceptual del sistema actual.

---

## Plantilla para copiar (nueva entrada)

```markdown
### YYYY-MM-DD — Título corto

**Contexto:** (1–2 frases)

#### Cambios / trabajo del día

| # | Qué | Experimentos / rutas | Detalle | Por qué / impacto TFM |
|---|-----|----------------------|---------|------------------------|
| 1 | | | | |

#### Decisiones / notas

- …

#### Resultados (si aplica)

- Modelo:
- Variables JSON:
- N artículos / año:
- Métricas clave (accuracy / kappa / F1):
- Observaciones cualitativas:

#### Pendiente / siguiente

- [ ] …

#### Commits / PRs

- `hash` — mensaje
```

---

## Mapa de carpetas útiles

```
Experimentos/
├── Experimentos.ipynb            ← narrativa + métricas por experimento
├── variables.json / variables_umbral_bajo.json
├── variables.py / utils.py
├── experiments/experimento_N/
├── Informe/
└── TFM/
    ├── DIARIO_TFM.md             ← este diario
    ├── PLAN_EXPERIMENTOS_TFM.md  ← qué experimentos entran en el TFM
    ├── ESTADO_LATEX.md           ← checklist LaTeX ↔ experimentos
    └── TFM___JORGE_GARCELA_N_GO_MEZ/Plantilla_TFG_ingles_2019/  ← .tex

CLUSTER/
├── README.md
├── cluster_tsc/
└── experimento_N_cluster/
```
