
# Reunión con la directora — estado del TFM

**Fecha:** 2026-07-29 · **Autor:** Jorge Garcelán

---

## Titular en una frase

> He construido y evaluado con rigor una **arquitectura de agentes especializados con Agent Skills** para detectar sexismo lingüístico en prensa. El resultado no es un simple "funciona / no funciona": su eficacia **depende del modelo**, y el análisis destapa que el cuello de botella está en la **tarea** (criterio experto amplio y variable), no en la arquitectura.

---

## 1. Objeto de estudio

Clasificar cada pieza periodística según **5 variables de lenguaje sexista**
(elegidas entre 15 por criterio experto). Enfoque *zero-shot*: la metodología del
libro de códigos se da en el *prompt*, sin entrenamiento.

| Cód. | Variable                     | Qué capta                                 | Prevalencia «Sí» |
| ----- | ---------------------------- | ------------------------------------------ | :-----------------: |
| V25   | `lenguaje_sexista`         | Sexismo de forma; categoría paraguas      |        0,81        |
| V26   | `masc_generico`            | Masculino genérico como pretendido neutro |        0,81        |
| V30   | `sexismo_discurso`         | Sexismo de fondo (lo que se cuenta)        |        0,43        |
| V33   | `asimetria_mujer_hombre`   | Trato asimétrico mujer/hombre en la pieza |        0,06        |
| V35   | `denominacion_sexualizada` | Mujer nombrada por físico/rol familiar    |        0,10        |

**Corpus:** ~7.115 piezas (IMIO/IRIS); **1.313 anotadas** en las 5 variables
(evaluación). Partición **dev (300) / test (1.013)**.

> Ojo a las prevalencias: V25/V26 al 81 % → la exactitud engaña; hay que usar
> **κ de Cohen** y **F1 macro**.

---

## 2. Sistema construido (la contribución)

- **5 agentes especializados** (uno por variable) con *progressive disclosure*:
  el agente ve solo `nombre + descripción` de las skills y carga la metodología
  bajo demanda.
- **Herramientas**: leer skill, **consultar guía** (RAG sobre las guías reales de
  lenguaje no sexista) y **verificar evidencias**.
- **Tubería multi-proveedor** (mismo código enruta a local y a las APIs) →
  comparación 1:1 entre modelos.
- **Salida trazable** (código + explicación + evidencias literales) para el marco
  *human-in-the-loop*.

Comparación **sin sesgo**: **B1** (Agent Skills, el sistema) frente a **B0**
(control: misma metodología inyectada en el *prompt*, sin herramientas). Ambos:
5 llamadas por artículo, mismo formato.

---

## 3. Modelos evaluados

| Proveedor                  | Modelo                    | Acceso               | Estado                |
| -------------------------- | ------------------------- | -------------------- | --------------------- |
| Local (Ollama, granja TSC) | `gemma4:e4b`            | GPU propia · gratis | ✅ B0                 |
| Local (Ollama)             | `qwen3:8b`              | GPU propia           | ⬜ opcional           |
| OpenAI                     | `gpt-4o-mini`           | API                  | ✅ B0 + B1            |
| OpenAI                     | `gpt-5.4-nano`          | API                  | ✅ B0 + B1            |
| Google                     | `gemini-3.1-flash-lite` | API                  | ✅ B0 + B1            |
| Anthropic                  | `claude-haiku-4-5`      | API                  | ⬜ excluido por coste |

---

## 4. Resultados clave

### 4.1 Eje A — Benchmark (nivel B0), medias sobre 5 variables · N≈1.300

| Modelo                       |    Exactitud    |    F1 macro    |   **κ**   |
| ---------------------------- | :-------------: | :-------------: | :--------------: |
| gemini-3.1-flash-lite        |      0,600      |      0,388      |      +0,026      |
| gpt-4o-mini                  |      0,564      |      0,410      |      +0,033      |
| gpt-5.4-nano                 |      0,571      |      0,430      |      +0,052      |
| **gemma4:e4b (local)** | **0,598** | **0,422** | **+0,062** |

> **El modelo local, gratis, tiene el κ medio más alto.** Mensaje para IRIS:
> privacidad + coste cero **sin** perder rendimiento frente a las APIs.

### 4.2 Eje B (core) — B0 (control) vs B1 (Agent Skills) · N=1.313

| Modelo                |      B0 κ      |      B1 κ      |       Δκ       |                          |
| --------------------- | :--------------: | :--------------: | :--------------: | :-----------------------: |
| gemini-3.1-flash-lite |      +0,026      | **+0,066** | **+0,040** |     ✅ mejora (×2,5)     |
| gpt-4o-mini           |      +0,033      |      +0,014      |     −0,019     |            ✗            |
| gpt-5.4-nano          |      +0,052      |      +0,004      |     −0,048     |            ✗            |
| **Media**       | **+0,037** |      +0,028      |     −0,009     | coste B1 =**2,2×** |

> Las Agent Skills **aportan cuando el modelo explota las herramientas (gemini)**
> y estorban cuando las sobrecarga (gpt). No es útil ni inútil "en abstracto".
>
> #### **Explicar esto --->> sobreajuste a la pregunta?? saturación?? sobreespecificación??**

### 4.2-bis Métricas por variable × 5 brazos (κ, los 4 modelos)

La media esconde que **cada variable se comporta distinto**. κ por variable en los
cinco brazos de ablación (brazos sin la tool RAG primero):

**gemini-3.1-flash-lite:**

| Variable | B0 | B1 mín | solo resúm | solo tool | completo |
|----------|:---:|:---:|:---:|:---:|:---:|
| `lenguaje_sexista` | −0,000 | −0,001 | +0,011 | +0,014 | +0,015 |
| **`masc_generico`** | +0,073 | +0,238 | **+0,266** | +0,225 | +0,261 |
| `sexismo_discurso` | +0,004 | +0,002 | +0,006 | −0,002 | −0,000 |
| `asimetria_mujer_hombre` | +0,036 | +0,048 | +0,045 | +0,060 | +0,045 |
| `denominacion_sexualizada` | +0,019 | +0,018 | +0,048 | +0,016 | +0,011 |

**gemma4:e4b (local):**

| Variable | B0 | B1 mín | solo resúm | solo tool | completo |
|----------|:---:|:---:|:---:|:---:|:---:|
| `lenguaje_sexista` | −0,005 | +0,001 | −0,000 | −0,002 | +0,004 |
| `masc_generico` | +0,101 | +0,128 | +0,124 | +0,097 | +0,109 |
| `sexismo_discurso` | +0,002 | −0,009 | +0,007 | −0,012 | −0,006 |
| `asimetria_mujer_hombre` | +0,054 | +0,124 | +0,098 | +0,054 | +0,092 |
| **`denominacion_sexualizada`** | **+0,154** | +0,080 | +0,078 | +0,051 | +0,015 |

**gpt-4o-mini:**

| Variable | B0 | B1 mín | solo resúm | solo tool | completo |
|----------|:---:|:---:|:---:|:---:|:---:|
| `lenguaje_sexista` | −0,013 | −0,005 | −0,006 | −0,045 | −0,038 |
| `masc_generico` | +0,023 | +0,051 | +0,020 | +0,032 | +0,034 |
| `sexismo_discurso` | +0,002 | +0,011 | −0,003 | +0,016 | +0,011 |
| `asimetria_mujer_hombre` | +0,019 | −0,001 | +0,032 | +0,021 | +0,003 |
| `denominacion_sexualizada` | +0,136 | +0,070 | +0,075 | +0,115 | +0,061 |

**gpt-5.4-nano:**

| Variable | B0 | B1 mín | solo resúm | solo tool | completo |
|----------|:---:|:---:|:---:|:---:|:---:|
| `lenguaje_sexista` | +0,001 | +0,001 | +0,000 | +0,000 | +0,000 |
| `masc_generico` | +0,103 | +0,016 | +0,021 | +0,006 | +0,016 |
| `sexismo_discurso` | +0,007 | −0,002 | +0,018 | +0,044 | +0,004 |
| `asimetria_mujer_hombre` | +0,026 | −0,012 | +0,010 | −0,004 | −0,012 |
| `denominacion_sexualizada` | +0,123 | −0,003 | +0,025 | +0,016 | +0,014 |

> **Lo que revela el desglose completo:**
> - **`lenguaje_sexista`** (paraguas, 81 % "Sí"): κ≈0 en **todo** modelo y **todo** brazo. Nadie acuerda — la categoría agregadora problemática (§6, decisión 3).
> - **`masc_generico`**: **toda** la mejora de gemini es aquí (0,073 → 0,24-0,27) y la produce el **mecanismo de skills** (ya en B1-mínimo salta a 0,238); las guías no añaden. Ninguna otra variable ni modelo lo replica.
> - **`sexismo_discurso`**: κ≈0 en todos los brazos — coherente con que los equipos lo anotan 5,7× distinto (§4.3). Techo de la tarea puro, ninguna configuración lo arregla.
> - **`denominacion_sexualizada`**: el caso más claro de **daño de las skills**. En gpt/gemma parte alto en B0 (0,12-0,15) y **se degrada monótonamente** al añadir maquinaria — en gemma: 0,154 → 0,080 → 0,078 → 0,051 → **0,015** (completo). La tool es la que más resta.
> - **`asimetria_mujer_hombre`**: gemma mejora con B1-mínimo (0,054 → 0,124), único punto donde el mecanismo ayuda al modelo local.

### 4.3 El acuerdo depende de QUIÉN anotó (modelo = gemini B1)

Dos equipos codificaron el corpus: **Indexa** (548 piezas, coders 1-5) y **UCM3**
(765 piezas, coders 3-6). El modelo concuerda **mucho más con Indexa**:

| Equipo |  n  | Exactitud |  F1  |   **κ**   |
| ------ | :-: | :-------: | :---: | :--------------: |
| Indexa | 548 |   0,741   | 0,503 | **+0,118** |
| UCM3   | 765 |   0,604   | 0,398 |      +0,043      |

**La causa está en cómo anota cada equipo, no en el modelo.** Mira la prevalencia
de "Sí" por equipo y variable — la misma variable se codifica de forma distinta:

| Variable                       |  % "Sí" Indexa  |   % "Sí" UCM3   |      Ratio      | κ Indexa | κ UCM3 |
| ------------------------------ | :--------------: | :--------------: | :-------------: | :-------: | :-----: |
| `lenguaje_sexista`           |      74,6 %      |      85,5 %      |      1,1×      |  +0,026  | +0,009 |
| `masc_generico`              |      71,4 %      |      88,0 %      |      1,2×      |  +0,277  | +0,223 |
| **`sexismo_discurso`** | **11,5 %** | **65,2 %** | **5,7×** |  +0,065  | −0,003 |
| `asimetria_mujer_hombre`     |      3,3 %      |      8,2 %      |      2,5×      |  +0,202  | −0,027 |
| `denominacion_sexualizada`   |      6,6 %      |      12,4 %      |      1,9×      |  +0,019  | +0,013 |

> **El caso `sexismo_discurso` lo dice todo:** Indexa marca sexismo de fondo en el
> 11,5 % de las piezas; UCM3, en el 65,2 %. Casi seis veces más. Eso **no es
> ruido del modelo**: son dos criterios humanos incompatibles sobre la misma
> variable. La variabilidad entre equipos **iguala o supera** la diferencia entre
> modelos → evidencia directa del **techo de la tarea**, sin doble codificación.

### 4.3-bis Heterogeneidad DENTRO de cada equipo (modelo = gemini B1) --> OpenAI B1

El desacuerdo no es solo entre equipos: también entre personas. Por anotador
individual (κ vs **gemini** B1):

| Anotador         |    Piezas    |     % "Sí"     |       acc       |    **κ**    |
| ---------------- | :-----------: | :--------------: | :-------------: | :----------------: |
| Indexa 1         |      127      |      40,8 %      |      0,672      |       +0,254       |
| Indexa 2         |      128      |      32,5 %      |      0,761      |       +0,364       |
| Indexa 3         |      231      |      28,9 %      |      0,772      |       +0,346       |
| Indexa 4         |      12      |      28,3 %      |      0,733      |       +0,265       |
| Indexa 5         |      50      |      39,6 %      |      0,724      |       +0,360       |
| **UCM3 3** | **363** | **63,9 %** | **0,504** |  **+0,166**  |
| UCM3 4           |      166      |      46,5 %      |      0,663      |       +0,293       |
| UCM3 5           |      235      |      37,1 %      |      0,718      |       +0,313       |
| UCM3 6           |       1       |        —        |       —       | *(n=1, ignorar)* |

| Equipo | κ min | κ max | Rango | Lectura                                  |
| ------ | :----: | :----: | :---: | ---------------------------------------- |
| Indexa | +0,254 | +0,364 | 0,110 | **compacto** (criterio compartido) |
| UCM3   | +0,166 | +0,313 | 0,147 | **disperso**                       |

> **Indexa está calibrado entre sí** (los 5 coders marcan 29-41 % "Sí", κ pareja).
> **UCM3 no**, y el problema tiene nombre: **UCM3-3** codifica el 28 % del corpus,
> marca 63,9 % "Sí" y es con quien menos concuerda el modelo (κ 0,166), mientras
> UCM3-5 se comporta casi como Indexa. **Buena parte del gap entre equipos lo
> produce un solo anotador.** La heterogeneidad es entre personas → refuerza el
> techo de la tarea y justifica la doble codificación (§6.1).
>
> *Aviso: estas κ están agrupadas por las 5 variables (pooling) → sirven para
> comparar coders entre sí, no como cifra absoluta (el κ por-variable ronda 0,06).*

### 4.4 Infra-detección sistemática (los 3 modelos coinciden, la humana no)

De las **4.503 celdas** donde los tres modelos (gemini, gpt-4o-mini, gpt-5.4-nano)
dan **el mismo veredicto**, aciertan 3.064. En las **1.439 que fallan**, el error
es casi siempre en la misma dirección:

| Variable                     |    Unánimes    |    Aciertan    | **3M=No / Hum=Sí** | 3M=Sí / Hum=No |
| ---------------------------- | :-------------: | :-------------: | :-----------------------: | :-------------: |
| `lenguaje_sexista`         |       694       |       122       |       **572**       |        0        |
| `masc_generico`            |       372       |       147       |       **223**       |        2        |
| `sexismo_discurso`         |      1.064      |       610       |       **454**       |        0        |
| `asimetria_mujer_hombre`   |      1.088      |      1.022      |            65            |        1        |
| `denominacion_sexualizada` |      1.285      |      1.163      |            122            |        0        |
| **TOTAL**              | **4.503** | **3.064** |      **1.436**      |   **3**   |

> **1.436 vs 3.** Cuando los tres modelos se equivocan al unísono, el 99,8 % de las
> veces es "modelos dicen No, la humana dice Sí". Los modelos aplican un criterio
> **sistemáticamente más estrecho** que las expertas — infra-detectan sexismo. No
> es indecisión: lo hacen con alta confianza (ver §4.5).
> **Scraping descartado** como causa (solo 3,7 % de textos sospechosos; excluirlos
> no mueve las métricas).

### 4.5 Confianza y calibración (exp22 · gpt-4o-mini)

Pedimos al modelo, además del veredicto, una **probabilidad de "Sí" (0-100)** para
estudiar si "sabe cuándo no sabe" y si filtrando por confianza mejora el acuerdo.

| Aspecto                                           | Resultado                                                                               |
| ------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Distribución de la probabilidad                  | **Bimodal** (casi todo 0 o 100): el modelo casi nunca duda                        |
| Calibración                                      | **Mal calibrado**: alta confianza ≠ mayor acierto                                |
| En los desacuerdos con la humana                  | **Confiadamente equivocado** (mediana de probabilidad ≈ 0 donde ella dice "Sí") |
| Umbral óptimo (dev→test,**por variable**) | κ medio**0,027 → 0,064** (mejora modesta)                                       |
| Umbral agrupando variables (κ=0,227)             | **Descartado**: era un espejismo de agrupación                                   |

> Conclusión: el desacuerdo es de **criterio**, no de incertidumbre. El modelo no
> está indeciso en los casos que falla; está seguro y equivocado → la confianza
> auto-declarada **no sirve** como filtro de calidad tal cual. Ajustar umbral por
> variable recupera algo, pero poco.

### 4.6 ¿De dónde viene el efecto de las skills? (ablación, 4 modelos × 5 brazos)

Descompuse B1 en sus ingredientes para ver qué parte importa. Cinco brazos, todo
fijo salvo qué se le da al agente. **κ medio (5 variables) por modelo:**

| Brazo | Tool RAG | Resúm. | gemini | gpt-4o-mini | gpt-5.4-nano | gemma (local) |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| B0 (control) | ✗ | ✗ | +0,026 | +0,033 | +0,052 | +0,061 |
| B1 mínimo (solo skills) | ✗ | ✗ | +0,061 | +0,025 | +0,000 | +0,065 |
| B1 solo resúmenes | ✗ | ✅ | **+0,075** | +0,024 | +0,015 | +0,061 |
| B1 solo tool | ✅ | ✗ | +0,063 | +0,028 | +0,012 | +0,038 |
| B1 completo | ✅ | ✅ | +0,066 | +0,014 | +0,004 | +0,043 |

> **Dos conclusiones que sí generalizan:**
>
> 1. **La tool RAG en vivo (`CONSULTAR_GUIA`) nunca ayuda; a menudo hace daño.** En
>    los cuatro modelos, la mejor variante **no** lleva la tool. En gemma se ve
>    clarísimo: sin tool κ≈0,061-0,065, con tool cae a 0,038-0,043. El TF-IDF sobre
>    textos cortos mete ruido, no señal.
> 2. **El mecanismo de skills solo aporta en gemini.** Ahí, empaquetar la
>    metodología como skill verificable (mínimo) ya recupera ~85 % de la mejora
>    sobre B0. En los otros tres, B1 iguala (gemma) o empeora (gpt) al control.
>
> **La mejor configuración universal es "skills + resúmenes, sin tool"** (columna
> `B1 solo resúmenes`): más simple y barata que el B1 completo, y nunca peor.

### 4.7 Aportación metodológica (transversal)

- **Exactitud y κ se contradicen**: reportar solo exactitud concluiría lo
  contrario. Demostrado 3 veces (umbral, caching, skills).
- Incluso detecté un **espejismo de agrupación** (el κ se infla al juntar
  variables con prevalencias distintas) en mi propio análisis.

---

## 5. Limitaciones (a decir yo antes que ella)

- **Acuerdo absoluto bajo** (κ ≈ 0 en 4 de 5 variables), pero explicado.
- **Sin doble codificación** → no se separa el error del modelo de la subjetividad
  de la anotación.
- `gemma` **solo en B0**: B1 en local es ~10× más lento (decenas de horas).
- Los N por variable difieren en gemma (enmascaré ~1,8 % de fallos técnicos).

---

## 6. Decisiones que le pido a la directora

1. **¿Doble-codificar una submuestra (50-100 noticias)?** Es lo único que no puedo
   hacer yo y lo que más blindaría el argumento del techo. ¿Lo arregla el proyecto?
2. **¿Cierro alcance o amplío?** Tengo 4 modelos, B0/B1, ablaciones y confianza.
   ¿Suficiente para redactar, o quiere gemma-B1 / Claude / más?
3. **¿El criterio de V25 (categoría agregadora) es intencionado?** El codebook se
   contradice (regla de inversión vs cláusula agregadora); afecta a la validez.

---

## 7. Acordado en la reunión (giro del marco)

La directora reorienta la lectura de los resultados: **el eje no es acierto/fallo
del modelo, sino qué revela el desacuerdo sobre la tarea y sobre quien anota.**

### 7.1 Reencuadre de la discusión

Lo que hasta ahora contaba como "el modelo falla" pasa a leerse como evidencia de
que la variable es intrínsecamente disputada. El desacuerdo entre Indexa y UCM3
(§4.3) y entre anotadoras de un mismo equipo (§4.3-bis) no es ruido a limpiar: es
el hallazgo. Detrás de la formación recibida hay creencias y sesgos propios que
mueven el criterio, y `sexismo_discurso` (11,5 % vs 65,2 %) lo enseña sin
ambigüedad. Si los modelos reproducen ese mismo patrón de dispersión, la
conclusión no es que estén rotos, sino que la tarea no tiene un suelo estable
contra el que medirlos.

Tres consecuencias de redacción:

- **En positivo.** La IA llega hasta cierto punto y ahí se topa con algo difícil
  de capturar incluso para expertas. Eso es un resultado, no una excusa.
- **Efecto espejo.** El modelo se parece a quien le pregunta: refleja el marco,
  el vocabulario y los supuestos del *prompt* y del codebook. La infra-detección
  sistemática de §4.4 encaja aquí — un criterio estrecho heredado, no un error
  aleatorio.
- **Menos tabla de aciertos, más discusión.** Los κ bajos sostienen el argumento
  en vez de debilitarlo; hay que presentarlos así desde el principio del capítulo
  de resultados.

### 7.2 Análisis cualitativo (acción concreta)

Seleccionar **10 piezas** al final del análisis y pedir que las anoten. Criterio
de selección propuesto: casos donde los tres modelos coinciden en "No" y la
humana dice "Sí" (§4.4), más algún caso de máxima divergencia entre equipos en
`sexismo_discurso`. El objetivo es ilustrar *por qué* se discrepa, con texto
concreto delante — no medir nada.

### 7.3 El modelo como tercer equipo anotador — **probado**

Tratar cada modelo como **una anotadora más** y montar un tercer equipo "IA"
junto a Indexa y UCM3. Cambia la pregunta de "¿acierta el modelo?" a "¿dónde cae
el modelo en el espacio de criterios humanos?". Implementado en
`experimento_21_agentskills/equipo_ia.py` (nivel B1).

**Restricción que condiciona el diseño:** cada noticia la anotó **una sola
persona** (1.313 noticias = 1.313 filas). No hay solape humano-humano, así que la
κ entre personas **no es calculable**. El equipo IA es el único con solape total
(cada modelo anota las 1.313), y por tanto el único cuya cohesión interna sí se
puede medir. Lo que sí compara a todo el mundo sin solape es la **prevalencia de
"Sí"**.

#### 7.3.1 Espacio de criterios (% "Sí" medio sobre las 5 variables)

| Coder                        | Equipo | n    | leng_sex | masc_gen | sex_disc | asimetría | denom_sex | **media** |
| ---------------------------- | ------ | ---- | :------: | :------: | :------: | :-------: | :-------: | :-------------: |
| UCM3 3                       | UCM3   | 363  |  0,986  |  0,975  |  0,978  |   0,041   |   0,215   | **0,639** |
| UCM3 4                       | UCM3   | 166  |  0,831  |  0,855  |  0,283  |   0,271   |   0,084   |      0,465      |
| Indexa 1                     | Indexa | 127  |  0,772  |  0,756  |  0,307  |   0,087   |   0,118   |      0,408      |
| Indexa 5                     | Indexa | 50   |  0,940  |  0,920  |  0,100  |   0,020   |   0,000   |      0,396      |
| UCM3 5                       | UCM3   | 235  |  0,668  |  0,753  |  0,409  |   0,013   |   0,013   |      0,371      |
| Indexa 2                     | Indexa | 128  |  0,742  |  0,727  |  0,062  |   0,031   |   0,062   |      0,325      |
| Indexa 3                     | Indexa | 231  |  0,701  |  0,649  |  0,039  |   0,009   |   0,048   |      0,289      |
| Indexa 4                     | Indexa | 12   |  0,583  |  0,500  |  0,167  |   0,000   |   0,167   |      0,283      |
| **gpt-4o-mini**        | **IA** | 1313 |  0,446  |  0,191  |  0,107  |   0,144   |   0,012   | **0,180** |
| **gemini-3.1-f-lite**  | **IA** | 1313 |  0,038  |  0,669  |  0,020  |   0,039   |   0,009   | **0,155** |
| **gpt-5.4-nano**       | **IA** | 1313 |  0,001  |  0,097  |  0,112  |   0,007   |   0,001   | **0,043** |

> **Los tres modelos ocupan el extremo permisivo, por debajo de la humana más
> permisiva.** No es que el modelo caiga "entre" Indexa y UCM3: cae fuera del
> rango humano completo. Es la lectura limpia de la infra-detección de §4.4.

Como cada humano vio noticias distintas, la prevalencia mezcla criterio con
muestra. Usando el modelo de instrumento común (mismo modelo, mismas noticias que
vio cada coder), la **Δ prevalencia humano − IA** aísla el criterio y el orden se
mantiene: UCM3-3 **+0,510**, UCM3-4 +0,328, Indexa 5 +0,291, Indexa 1 +0,275,
UCM3-5 +0,254, Indexa 2 +0,191, Indexa 3 +0,177. El sesgo de composición no
explica el gap.

#### 7.3.2 El hallazgo inesperado: **el equipo IA no es un equipo**

κ pareada entre modelos, con solape total sobre las 1.313 piezas:

| Par                          | leng_sex | masc_gen | sex_disc | asimetría | denom_sex | **κ media** |
| ---------------------------- | :------: | :------: | :------: | :-------: | :-------: | :---------------: |
| gemini ↔ gpt-4o-mini        |  0,040  |  0,055  |  0,015  |   0,086   |   0,062   | **0,052** |
| gemini ↔ gpt-5.4-nano       |  −0,001 |  0,012  |  0,037  |   0,022   |  −0,001  | **0,014** |
| gpt-4o-mini ↔ gpt-5.4-nano  |  0,002  |  0,071  |  0,300  |   0,048   |  −0,001  | **0,084** |

> **Los modelos concuerdan entre sí tan poco como con las humanas** (κ 0,014-0,084,
> frente a κ humano-modelo de hasta 0,157). Un "equipo IA" con criterio compartido
> **no existe**: cada modelo es idiosincrásico. Esto desmonta la lectura fácil de
> "la IA tiene un sesgo" — son tres sesgos distintos que solo coinciden en la
> dirección (todos permisivos), no en los casos concretos.
>
> *Cautela: con prevalencias tan extremas (gpt-5.4-nano marca "Sí" en el 0,1 % de
> `lenguaje_sexista`) la κ se hunde por construcción. Reportar junto a la
> prevalencia, nunca sola.*

#### 7.3.3 Qué aporta al argumento

1. Sitúa a modelos y humanas en **una sola escala** sin necesitar doble
   codificación — resuelve parcialmente la limitación de §5.
2. Convierte "el modelo falla" en "el modelo es sistemáticamente más permisivo
   que cualquier anotadora", que es una afirmación **descriptiva y defendible**.
3. Refuerza §7.1: la variabilidad humana (0,283 → 0,639 de prevalencia media) es
   del mismo orden que la distancia humano-modelo. El techo es la tarea.
4. Da la selección natural de las **10 piezas** de §7.2: los casos donde UCM3-3 y
   los modelos están en las antípodas sobre `sexismo_discurso` (0,978 vs 0,020).

#### 7.3.4 Réplica en B0: el patrón no lo induce la arquitectura

Repitiendo todo sin skills (`equipo_ia.py --nivel b0`), los modelos siguen en el
extremo permisivo (media 0,060 / 0,113 / 0,224 frente al rango humano 0,283-0,639)
y la Δ ajustada apenas se mueve (UCM3-3 +0,505 en B0 vs +0,510 en B1). **El sesgo
permisivo es del modelo, no del sistema de agentes.**

Un matiz que sí cambia, y conviene mirarlo:

| κ media intra-IA            |  B0  |  B1  |
| --------------------------- | :---: | :---: |
| gemini ↔ gpt-4o-mini       | 0,079 | 0,052 |
| gemini ↔ gpt-5.4-nano      | 0,067 | 0,014 |
| gpt-4o-mini ↔ gpt-5.4-nano | 0,280 | 0,084 |

> Las Agent Skills **reducen** el acuerdo entre modelos en los tres pares. Dar
> herramientas y metodología cargable no los converge hacia un criterio común: los
> vuelve **más idiosincrásicos**, cada uno explotando las skills a su manera. Encaja
> con §4.2 (gemini mejora, los gpt empeoran) y con el efecto espejo de §7.1 — la
> skill amplifica lo que cada modelo ya traía.

**Pendiente:** decidir si va como §5 de resultados (mi recomendación: sí, cierra
el capítulo con el reencuadre de §7.1) o como línea futura en conclusiones.

---

## 8. Estado de la memoria

| Capítulo                            | Estado                                                                   |
| ------------------------------------ | ------------------------------------------------------------------------ |
| 1 Introducción · 2 Estado del arte | ✅ alineados con IRIS                                                    |
| 3 Metodología                       | ✅ reescrito (corpus, variables, solución IA, modelos, zero-shot)       |
| 4 Implementación                    | ✅ reescrito (arquitectura, skills, RAG, evaluación, PoC) + figuras     |
| 5 Resultados                         | ✅ Exp 21 (B0/B1, divergencia, lift, ablaciones, anotadores) + 4 figuras |
| 5 Confianza (exp22)                  | ⬜ por escribir (datos listos)                                           |
| 6 Conclusiones                       | ⬜ por escribir (aquí van las líneas futuras)                          |
