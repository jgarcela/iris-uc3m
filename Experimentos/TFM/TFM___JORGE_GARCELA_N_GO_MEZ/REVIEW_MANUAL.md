# Review manual — comentarios del repaso

Comentarios de Jorge sobre la memoria.
Por ahora cubre parte del **Capítulo 3**; se irán añadiendo más secciones.

> ✅ **Tanda §3.1–§3.1.5 aplicada (2026-08-04).** Ver "Estado" bajo cada punto.
>
> ⚠️ **Verificar metadatos de guías en `bibliografia.bib`.** Se crearon 9 entradas
> para las guías de `methodology/`. Con datos de portada fiables:
> `csd_deporte2022`, `csd_diagnostico2022`, `yufera2023` (Irene Yúfera, Madrid 2023),
> `guerrero_malaga` (Susana Guerrero, U. Málaga), `iortv_mujer_violencia`
> (Observatorio Igualdad RTVE), `coe_violenciasexual` (UE–Consejo de Europa).
> **Autor/año a confirmar** (los puse por inferencia, marcados `[s.f.]`):
> `guia_igualdad_rtve`, `guias_no_sexista` (¿Instituto de la Mujer?),
> `guia_actuacion_lecturafacil` (¿Ministerio de Igualdad?).

**Leyenda de acciones:**
📚 añadir referencia / footnote · ➕ añadir contenido · ✏️ reescribir / aclarar (no se entiende) · 📍 ubicación por decidir

---

## General (ubicación por decidir) 📍

- **Web del proyecto IRIS.** Mencionar que, en el marco del proyecto IRIS, se ha
  desarrollado la web <https://dei.inf.uc3m.es/iris/> (se añadirá alguna captura)
  y que el sistema de IA que hay detrás corresponde a esta investigación.
  *(Falta decidir en qué sección encaja.)*

---

## §3.1 — Análisis experto y perspectiva de género

- 📚 Añadir footnote con:
  - Research portal UC3M: <https://researchportal.uc3m.es/display/act565092>
  - Web desarrollada: <https://dei.inf.uc3m.es/iris/>
- 📚 En la frase *«A diferencia de aproximaciones que abordan la detección de
  discurso de odio sobre mensajes breves en redes sociales»*, añadir la referencia
  al TFG previo:
  > Jorge Garcelán Gómez (2024). *Detection of Gender-Based Hate Speech in Social
  > Media via Natural Language Processing: An Analysis of the #SeAcabó Movement.*

---

## §3.1.1 — Fundamentación metodológica

- 📚 En *«…operacionaliza criterios procedentes de guías institucionales de
  lenguaje no sexista elaboradas por…»*, añadir referencias a **todas** las guías
  de `Experimentos/methodology/`.
- ✏️ **No se entiende** — reescribir más claro:
  > «Conviene subrayar que el sexismo lingüístico opera sobre la forma del mensaje
  > y no sobre su fondo. Una pieza puede presentar un contenido no sexista
  > expresado mediante formas sexistas, o a la inversa. Esta distinción justifica
  > que el libro de códigos separe explícitamente las variables de forma de
  > aquellas de contenido…»
- ➕ Explicar que el análisis se organiza en **bloques de variables** (≈40 en
  total), de los cuales este trabajo aborda solo el de Lenguaje:
  - **Contenido general:** autoría, género de la autoría, personas mencionadas y
    su género, tema, etc.
  - **Lenguaje:** 15 variables; aquí se reduce a 5 (ver §3.1.5), aunque es
    extrapolable a más.
  - **Imágenes:** número de imágenes en el artículo, personas y género.
  - **Fuentes de información:** declaraciones, fuente, género de la fuente, etc.

---

## §3.1.2 — Procedimiento de anotación

- 📚 Añadir footnote (aquí o donde encaje mejor) con:
  <https://www.uc3m.es/ss/Satellite/INST-EstudiosGenero/es/TextoDosColumnas/1371460773414/Analisis_de_las_informaciones_sobre_IA_en_medios_digitales>
- ✏️ **No se entiende** — reescribir más claro:
  > «La unidad de análisis es, por tanto, el artículo: cada variable recibe un
  > único veredicto por pieza, con independencia del número de ocurrencias del
  > fenómeno en el texto. Esta decisión, coherente con la práctica habitual en
  > análisis de contenido, tiene implicaciones relevantes para la interpretación
  > de los resultados, pues basta una sola ocurrencia para que una pieza extensa
  > quede codificada como positiva.»

---

## §3.1.4 — Fuente de datos / subconjunto de evaluación

- 📚 Añadir footnote (aquí o donde encaje) con:
  <https://www.uc3m.es/ss/Satellite/INST-EstudiosGenero/es/TextoDosColumnas/1371460773414/Analisis_de_las_informaciones_sobre_IA_en_medios_digitales>
- ✏️ Aclarar qué significa **«extensión suficiente»**:
  > «De ellas, 1.313 disponen de texto recuperado con extensión suficiente para el
  > análisis, cifra sobre la que se calculan todas las métricas del
  > Capítulo~\ref{results_and_discussion}.»

---

## §3.1.5 — Variables del análisis

- ➕ En *«Las cinco variables seleccionadas se definen del modo siguiente»*,
  indicar que las definiciones **se basan en las guías de lenguaje y en el
  conocimiento experto**.
- ➕ Lo mismo en el apartado *«De la definición experta a la especificación
  ejecutable»*.

---

---

## 🔑 Puntos transversales a propagar por toda la memoria

*(Ideas importantes que deben aparecer en varios sitios: abstract, intro, resultados, conclusiones, future work.)*

- **Salida cuantitativa + cualitativa (aportación clave).** El sistema no solo
  produce la **etiqueta** (plano cuantitativo, validable contra el GT), sino
  también el **marcado de evidencias literales en el texto** + una **explicación**
  del veredicto (plano cualitativo). Esto es lo que un análisis de contenido
  experto necesita en sus dos planos. La parte cualitativa **no se puede validar**
  (no hay marcado experto de referencia) y queda como **línea futura**.
  → Añadir/reforzar en: **abstract**, **intro/objetivos**, **§3.1.2** (ya está),
  **discusión** (valor HITL) y **conclusiones + future work**.

- **Comparación multiproveedor: local vs API (objetivo metodológico core).**
  Evaluar ambas vías —modelos locales de pesos abiertos vs API comerciales— sobre
  **la misma tarea y el mismo corpus**, con una tubería agnóstica al proveedor, es
  uno de los objetivos del TFM. Conecta con el hallazgo del Cap 5: **gemma local
  (gratis, privado) iguala o supera a las API en κ de control**.
  → Reforzar en: **abstract**, **intro/objetivos**, **§3.2** (ya está),
  **resultados** (Eje A) y **conclusiones**.

- **Complejidad de la tarea (naturaleza del corpus).** A diferencia de los corpus
  de redes sociales de la detección de discurso de odio, aquí son **textos
  extensos** (mediana ≈ 5.881 caracteres), **redactados por profesionales** y
  **sometidos a procesos editoriales**: el fenómeno es **sesgo sutil**, no agresión
  explícita. Eso **eleva la complejidad** y ayuda a explicar el bajo acuerdo
  absoluto (no es que el modelo falle, es que la tarea es intrínsecamente difícil).
  → Reforzar en: **abstract**, **intro**, **§3.1** corpus (ya está la frase),
  **discusión** (junto al techo de la tarea) y **conclusiones**.

---

## Pendientes (con estado)

Leyenda: ⬜ por hacer · 🔄 en curso · ✅ hecho · ⏸️ depende de terceros

| # | Pendiente | Estado | Nota |
|---|-----------|:------:|------|
| 1 | **Referencias de literatura en Cap. 3** | ✅ | Añadidas: **zero-shot / LLM** (`radford2019`, `brown2020` en `ai_solution`; `ziems2024css`, `hollauer_plastics` en `zero_shot`), Transformer (`vaswani2017`), aprendizaje profundo (`lecun2015deep`), κ de Cohen (`cohen1960`), guías institucionales (9 entradas), TFG previo (`garcelan2024`), corpus/análisis experto (`sainzdebaranda2026`). ⚠️ Verificar metadatos de 3 guías `[s.f.]` (ver #3). |
| 1b | **Estado del arte de Agent Skills en Cap. 2** | ⬜ | Revisar, añadir referencias y estado del arte de *Agent Skills* / agentes LLM (progressive disclosure, tool-use, ReAct) en el Cap. 2, que sustente el Cap. 4. |
| 1c | **Referencias en Cap. 4 (agentes)** | ✅ | Añadidas en `agents.tex`: **ReAct** (`yao2023react`), **progressive disclosure** (`nielsen1994`), **degradación con contexto largo** (`liu2023lost`) y **alucinaciones** (`huang2025hallucination`, versión ACM TOIS). |
| 1d | **Referencia de TF-IDF (RAG)** | ✅ | En `rag.tex`: **RAG** (`lewis2020rag`, NeurIPS 2020) y **TF-IDF** (`sparckjones1972`, J. Documentation). |
| 2 | Captura de la web IRIS (`dei.inf.uc3m.es/iris`) | ⏸️ | La insertas tú; el texto ya la introduce (§3.1) |
| 3 | Metadatos de 3 guías `[s.f.]` en el `.bib` | ⏸️ | `guia_igualdad_rtve`, `guias_no_sexista`, `guia_actuacion_lecturafacil`: confirmar autor/año |
| 4 | Propagar los 3 puntos transversales | ⬜ | Cuanti+cuali · complejidad de la tarea · local vs API → abstract, intro, discusión, conclusiones |
| 5 | Alinear **abstract** con lo hecho | ⬜ | Hoy describe la propuesta antigua (Chain-of-Thought, "entrenar", RAG como eje) |
| 6 | Reescribir **Cap. 6** (conclusiones + future work) | ⬜ | Sigue siendo #SeAcabó; aquí cierran los puntos transversales |
| 7 | Repaso manual de Caps. 4 y 5 | ⬜ | Tras terminar el de Cap. 3 |
| 8 | Compilar en Overleaf y revisar refs/figuras | ⬜ | Comprobar `\ref` y que las figuras PDF están en `imagenes/` |
| 9 | **§3.2.2 — comentar más `gemma4:e4b`** | ⬜ | Tras revisar referencias de Cap 3/4: valorar explicar mejor el modelo local (p. ej. *effective 4B params* / arquitectura), ya que es el modelo protagonista del hallazgo "local ≥ API". |

---

## Estado de aplicación (2026-08-04)

| Comentario | Estado |
|---|---|
| General — web IRIS + sistema IA | ✅ Añadido en §3.1 (intro): frase + footnote a `dei.inf.uc3m.es/iris`. Falta la **captura** (la pones tú). |
| §3.1 — footnote researchportal + web | ✅ Footnote a `researchportal.uc3m.es/display/act565092` y a la web |
| §3.1 — ref TFG SeAcabó | ✅ `\cite{garcelan2024}` (entrada nueva en el `.bib`) |
| §3.1.1 — referencias a las guías | ✅ `\cite{...}` a las 9 guías (⚠️ verificar metadatos de 3, ver arriba) |
| §3.1.1 — reescribir forma/fondo | ✅ Reescrito con ejemplo concreto |
| §3.1.1 — bloques de ~40 variables | ✅ Añadida lista (Contenido general / Lenguaje / Imágenes / Fuentes) |
| §3.1.2 — footnote InfoIA | ✅ `\cite{sainzdebaranda2026}` (es esa memoria; ya estaba en el `.bib`) |
| §3.1.2 — reescribir unidad de análisis | ✅ Reescrito más claro |
| §3.1.4 — footnote InfoIA | ✅ `\cite{sainzdebaranda2026}` en la intro del corpus |
| §3.1.4 — aclarar "extensión suficiente" | ✅ Aclarado: las 2 piezas sin cuerpo recuperado se excluyen |
| §3.1.5 — variables basadas en guías + experto | ✅ Añadido en los dos apartados |

**Pendiente tuyo:** la captura de la web; confirmar los metadatos de las 3 guías marcadas `[s.f.]`.

---

## Estado de aplicación — referencias Cap 3 y Cap 4 (2026-08-06)

Bloque de referencias (#1, #1c, #1d) cerrado paso a paso con confirmación.

| Cita | Ubicación | Estado |
|---|---|---|
| `radford2019`, `brown2020` | `ai_solution.tex` (aprendizaje profundo / zero-shot) | ✅ |
| `ziems2024css`, `hollauer_plastics` | `zero_shot.tex` (supervisión por libro de códigos) | ✅ |
| `vaswani2017` | `llms_models.tex` (Transformer) | ✅ |
| `lecun2015deep` | `ai_solution.tex` | ✅ |
| `cohen1960` | `iris_variables.tex` (κ de Cohen) | ✅ |
| `yao2023react` | `agents.tex` (bucle ReAct) | ✅ |
| `nielsen1994` | `agents.tex` (progressive disclosure) | ✅ |
| `liu2023lost`, `huang2025hallucination` | `agents.tex` (especialización / contexto largo) | ✅ |
| `lewis2020rag` | `rag.tex` (RAG — ref. canónica) | ✅ |
| `sparckjones1972` | `rag.tex` (TF-IDF) | ✅ |

**Nota:** se priorizó venue publicado sobre arXiv (`huang2025hallucination` → ACM TOIS; `lewis2020rag` → NeurIPS; `hollauer_plastics` → AAAI Symposium Series). Sin metadatos inventados.

*(Pendiente: más secciones de Caps. 3-4-5 cuando continúes el repaso.)*
