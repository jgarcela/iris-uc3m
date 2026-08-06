# Arquitectura de agentes y Agent Skills — Eje B del TFM

Documento **canónico de diseño** del core (Eje B). Complementa:
[`PLAN_EXPERIMENTOS_TFM.md`](PLAN_EXPERIMENTOS_TFM.md) (alcance), [`ESTADO_LATEX.md`](ESTADO_LATEX.md) (huecos `.tex`), [`DIARIO_TFM.md`](DIARIO_TFM.md) (día a día).

> ⚠️ **ESTADO 2026-08-04 — el diseño de las secciones 0–4 sigue vigente; el plan y las hipótesis (§5–7) ya están ejecutados.** Lo que cambió respecto a lo planeado:
> - **Ejecutado en `experimento_21_agentskills/`** sobre **4 modelos**: gemma4:e4b (local) + gpt-4o-mini, gpt-5.4-nano, gemini-3.1-flash-lite. **Claude y qwen3 fuera.**
> - **Resultado (contradice la hipótesis de §5):** las Agent Skills **solo mejoran a gemini** (y ahí, casi todo en `masc_generico`); en gpt empeoran y en el local igualan al control.
> - **La tool `CONSULTAR_GUIA` (RAG en vivo) nunca ayuda** (ablación 5 brazos × 4 modelos); mejor config = skills + resúmenes sin tool.
> - **Prompt caching descartado** de la config canónica (no ayuda, altera el comportamiento) — no es una "mejora a incorporar".
> - Añadido no previsto: **el sistema como tercer anotador** (`equipo_ia.py`) → los modelos caen fuera del rango humano; no forman "equipo".
> - **B2 multi-agente**: confirmado como línea futura. **Exp 22 (confianza): fuera del TFM.**
>
> Redacción final en `chapters/5 .../experimentacion.tex`. Verdad del `.tex`: `ESTADO_LATEX.md`.

> **Tarea invariante:** clasificar 5 variables de lenguaje sexista (V25 `lenguaje_sexista`, V26 `masc_generico`, V30 `sexismo_discurso`, V33 `asimetria_mujer_hombre`, V35 `denominacion_sexualizada`) sobre el mismo corpus (~7 115), evaluando en las 1 315 anotadas. Lo único que cambia entre niveles es **cómo se estructura la inferencia**.

---

## 0. Terminología (para que la memoria aguante la defensa)

| Término | Qué es | Papel |
|---------|--------|-------|
| **Skill** | Conocimiento empaquetado en un `SKILL.md` (metodología, ejemplos, fronteras). Frontmatter `name` + `description`. | **Pasivo**: se carga, no decide. |
| **Progressive disclosure** | El modelo ve solo `name`+`description`; carga el cuerpo del `SKILL.md` **bajo demanda** vía una tool. | Mecanismo que define "Agent Skills". |
| **Agente especializado** | Modelo + tools + **bucle de decisión** con un objetivo acotado (una variable). Decide qué skills leer y cuándo verificar. | **Activo**: ejecuta acciones. |

**Regla de honestidad del TFM:** "1 skill por variable" **no** es un agente. Lo que sí lo es: **1 agente especializado por variable que carga su skill (y skills auxiliares) bajo demanda**. Si un agente nunca usa una tool, colapsa al baseline — y esa tasa de uso es un resultado a reportar, no un fallo.

---

## 1. Auditoría de lo existente (punto de partida)

| Artefacto | Qué es realmente | Veredicto |
|-----------|------------------|-----------|
| Exp 16 [`clasificador_skills.py`](../experiments/experimento_16/clasificador_skills.py) | **1 llamada** con orquestador + 5 `SKILL.md` **concatenados** en el system prompt | **No es agente ni Agent Skills**: sin progressive disclosure; `description:` es metadato muerto. Es prompting estructurado. |
| Exp 15 [`main_sin_newspaper.py`](../experiments/experimento_15/main_sin_newspaper.py) | **5 llamadas**, una por variable, metodología inyectada | Sirve tal cual como **B0 Baseline**. |
| [`pruebas_skills_ollama/agent_skills.py`](../pruebas_skills_ollama/agent_skills.py) | Agente LangChain con `list_skills`/`read_skill` → progressive disclosure **correcto** | Patrón válido, pero **demo genérico**: no clasifica, no da JSON, no evalúa. |

### Confound que invalida "baseline vs skills" tal cual
Exp 15 = **5 llamadas** (aisladas); Exp 16 = **1 llamada** (conjunta). Comparar 15 vs 16 mezcla *dos* cambios (empaquetado **y** nº de llamadas) → el delta es inatribuible. **Decisión TFM: igualar todo a 5 llamadas** (una por variable) en B0 y B1.

### Bugs a corregir antes de reusar
- Evidencias: `texto.replace('"',"'")` altera el texto pero `_verificar_evidencias` comprueba contra el original → descarta evidencias válidas.
- 1 llamada / 5 variables → un fallo de JSON pone las 5 a `codigo=1` (sesga métricas a la clase negativa).
- `frontera_con_otras` se serializa con `str(dict)` → `repr` de diccionario Python dentro del `SKILL.md`.
- Sin prompt caching → coste innecesario a escala 7k.

---

## 2. Niveles experimentales (Eje B)

Todos: **5 llamadas por artículo, una por variable, mismo JSON de salida** `{codigo, explicacion, evidencias}`.

| Nivel | Metodología llega... | Tools | Diferencia única vs anterior |
|-------|----------------------|-------|------------------------------|
| **B0 Baseline** | **inyectada** en el prompt (= Exp 15) | ninguna | — |
| **B1 Agent Skills** | el agente ve `name`+`description` y **carga** su `SKILL.md` vía `read_skill` (+ auxiliares opcionales) | `list_skills`, `read_skill` | progressive disclosure + skills auxiliares |
| **B2 Multi-agente** *(alcance: solo B1 esta entrega; B2 = línea futura)* | roles detector → verificador de evidencias → árbitro | tools B1 + handoff | orquestación entre roles |

Delta B1−B0 = **efecto neto de las Agent Skills** (sin confound de nº de llamadas).

---

## 3. Diseño de B1 — 5 agentes especializados

```
Artículo ──> 5 agentes especializados (5 llamadas independientes):

  Agente V25 ─┐   cada agente:
  Agente V26 ─┤     system: "eres experta SOLO en <variable>. La metodología no
  Agente V30 ─┤              está aquí: cárgala con read_skill('<variable>')."
  Agente V33 ─┤     tools:  list_skills(), read_skill(id)
  Agente V35 ─┘     bucle:  leer su skill → (opc.) leer guía si duda
                            → (opc.) verificar_evidencias → cerrar JSON
```

### Dos tipos de skill

| Tipo | Skills | Uso |
|------|--------|-----|
| **De variable** (metodología) | `lenguaje_sexista`, `masc_generico`, `sexismo_discurso`, `asimetria_mujer_hombre`, `denominacion_sexualizada` | cada agente carga la suya (obligatorio) |
| **Auxiliares** (compartidas) | `guia_regla_inversion`, `guia_lenguaje_inclusivo` (Sainz de Baranda / CSD), `verificar_evidencias` | cualquier agente las carga **bajo demanda** según el `description:` |

Aquí el `description:` del frontmatter **por fin decide**: el agente lee las descripciones disponibles y elige si necesita la guía o el verificador. Eso es lo que legitima el nombre "Agent Skills".

### Layout en disco (propuesto)
```
experiments/experimento_21_agentskills/
├── skills/
│   ├── lenguaje_sexista/SKILL.md        # V25  (regenerable desde variables.json)
│   ├── masc_generico/SKILL.md           # V26
│   ├── sexismo_discurso/SKILL.md        # V30
│   ├── asimetria_mujer_hombre/SKILL.md  # V33
│   ├── denominacion_sexualizada/SKILL.md# V35
│   ├── guia_regla_inversion/SKILL.md    # auxiliar compartida
│   ├── guia_lenguaje_inclusivo/SKILL.md # auxiliar (guías methodology/)
│   └── verificar_evidencias/SKILL.md    # auxiliar
├── agente.py            # bucle tool-use por variable (1 agente = 1 variable)
├── tools.py             # list_skills / read_skill / verificar_evidencias
├── main.py              # itera corpus → 5 agentes → JSON por artículo
└── metrics.py           # kappa/F1 sobre las 1 315 (reusa el de exp 15/16)
```

---

## 4. Stack: bucle tool-use nativo multi-proveedor (no LangChain)

**Decisión:** implementar el agente como **bucle de acciones en texto (estilo ReAct)** sobre `consultar_ollama` de [`utils.py`](../utils.py), **no** LangChain ni tool-calling nativo.

Motivo técnico: `consultar_ollama` es **texto→texto** (no expone la API de tools de cada proveedor). En vez de forzar tool-calling nativo (frágil en Ollama pequeños, distinto por proveedor), el agente emite acciones como texto y el runtime las enruta:
```
LEER_SKILL: <id>          → se le devuelve el cuerpo del SKILL.md
CONSULTAR_GUIA: <query>   → RAG sobre methodology/ → pasajes literales de las guías expertas
VERIFICAR: ["..","..."]   → se verifica que son literales del texto
FINAL: {json}             → veredicto de la variable
```

`CONSULTAR_GUIA` (`guias.py`) indexa con TF-IDF las guías reales de [`methodology/`](../methodology/)
(las del `methodology_manifest.json`; la tesis no se indexa) y devuelve pasajes con cita. Con esto
las guías se **usan en inferencia**, no sólo como fuente offline del codebook.

Justificación:
1. **Uniforme en los 4 proveedores** (mismo protocolo Claude/OpenAI/Gemini/Ollama) → comparabilidad 1:1 con el Eje A, que ya usa `utils.py`.
2. Evita `langchain_classic` (legacy) y el tool-calling frágil en locales pequeños.
3. Progressive disclosure real: el modelo sólo ve `name`+`description` (`list_skills`) y decide qué cargar.
4. Permite **medir la tasa de uso de tools** por modelo (métrica del core).

Bucle (`agente.py`): hasta **8 iteraciones** (`MAX_ITERS`); si el modelo emite `FINAL` sin haber usado ninguna tool → **colapso a B0**, se registra en la traza. Implementado en `experiments/experimento_21_agentskills/`.

Mejoras heredadas del Exp 16 a incorporar:
- Prompt caching en el system prompt (Anthropic) para las corridas a 7k.
- Verificar evidencias contra el **mismo** texto que ve el modelo (no el original alterado).
- JSON por variable → un fallo afecta a 1 variable, no a 5.

---

## 5. Métricas del core (además de kappa/F1)

Sobre las 1 315 anotadas, por modelo y variable:
- **Acuerdo con GT:** accuracy, Cohen's Kappa, F1 macro (igual que Eje A → comparables).
- **Trazabilidad:** % de predicciones con evidencia literal válida (ataca el bug de evidencias).
- **Comportamiento del agente:** nº medio de tools por artículo, % de artículos que cargan guía/verificador, % de colapso a B0.
- **Coste/latencia:** tokens y tiempo por artículo (local vs API a escala 7k).

**Hipótesis del core:** B1 mejora acuerdo y/o trazabilidad frente a B0 sin cambiar el codebook; el efecto es mayor en modelos que **usan** las tools (esperable: más en API que en locales pequeños).

---

## 6. Plan de implementación (orden)

> **Nota:** los experimentos de skills previos (Exp 16, `pruebas_skills_*`) quedan **descartados**. El core vive en `experiments/experimento_21_agentskills/`.

1. **[este doc]** Diseño B0/B1, agentes y skills. ✅
2. Skills auxiliares (`guia_*`, `verificar_evidencias`) + regenerar las 5 de variable con `frontera_con_otras` limpia. ✅
3. `tools.py` + `agente.py`: 1 agente especializado end-to-end (bucle validado con mock). ✅
4. `main.py`: 5 agentes sobre corpus, escritura incremental + traza. ✅ (falta corrida real)
5. Multi-proveedor (Claude, OpenAI, Gemini, local) reusando `utils.py`; medir tasa de uso de tools. ⬜
6. Métricas sobre las 1 315; tabla B0 vs B1 por modelo (`metrics.py`). ⬜
7. Prompt caching del bloque de skills a escala 7k. ⬜
8. Redactar Cap. 4 (arquitectura) y Cap. 5 (resultados core) del LaTeX. ⬜

## 7. Decisiones (cerradas)

| # | Pregunta | Decisión |
|---|----------|----------|
| 1 | ¿Nº de experimento para B1? | **Exp 21** (`experimento_21_agentskills/`). |
| 2 | ¿Cuántas guías de `methodology/` como skill auxiliar? | 3 auxiliares (`guia_regla_inversion`, `guia_lenguaje_inclusivo`, `verificar_evidencias`) + resúmenes de guías descubribles; la tool `CONSULTAR_GUIA` resultó prescindible. |
| 3 | ¿B1 sobre los 5 modelos o subconjunto? | **4 modelos**: gemma local + gpt-4o-mini, gpt-5.4-nano, gemini. **Claude y qwen fuera** (coste / alcance). |
| 4 | ¿`verificar_evidencias` como tool o post-proceso? | **Tool** (el agente decide); su uso se registra en la traza. |
| 5 | ¿B2 multi-agente? | **Línea futura** (no entra en esta entrega). |
