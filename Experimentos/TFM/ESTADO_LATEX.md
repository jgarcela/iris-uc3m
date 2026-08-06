# Estado del LaTeX vs experimentos IRIS

Checklist para ir cerrando el TFM.
Raíz LaTeX: `TFM___JORGE_GARCELA_N_GO_MEZ/Plantilla_TFG_ingles_2019/`
Fuente experimental: `Experimentos/experiments/`, `CLUSTER/`, `Experimentos.ipynb`

Leyenda: ✅ alineado IRIS · 🟡 parcial / pulir · ❌ residual TFG anterior · ⬜ por escribir

Última actualización: **2026-08-04**.

---

## Mapa capítulo → contenido real

| Cap. | Sección LaTeX | Estado | Nota | Prioridad |
|------|---------------|--------|------|-----------|
| 1 | Introducción y subsecciones | ✅ | Alineado IRIS (Bindi, HITL, IRIS_IAMEDIA) | Baja (pulir) |
| 2 | Rep. mujeres / Detección sesgos / Justificación | ✅ | Estado del arte alineado | Baja |
| 3 | Análisis experto (Fundamentación / Anotación / Limitación) | ✅ | Reescrito; codebook IRIS, regla de inversión, sin doble codificación | — |
| 3 | Fuente de datos (`iris_corpus`) | ✅ | Corpus IMIO 7.115 → 1.315 anotadas → **1.313** métricas | — |
| 3 | Variables (`iris_variables`) | ✅ | 15 vars, 5 evaluadas, prevalencias, escala 1/2/3 | — |
| 3 | Solución IA / LLMs / modelos / zero-shot | ✅ | 4 modelos (gemma + 3 API); qwen/claude fuera; pipeline y salida trazable | — |
| 4 | Arquitectura agéntica (`agents`) | ✅ | B0/B1, progressive disclosure, ReAct, agentes especializados | — |
| 4 | Prompts / RAG / Evaluación / PoC | ✅ | `eval` = zero-shot sobre 1.313 (sin dev/test); `rag` con nota de prescindible | — |
| 5 | Intro (`results_and_discussion`) | ✅ | Reescrita con la clave de lectura (desacuerdo = hallazgo) | — |
| 5 | Experimentación (`experimentacion`) | ✅ | B0/B1, divergencia acc-κ, lift, **modelo local gemma**, **descomposición 5 brazos × 4 modelos**, errores, **equipo_ia**, coste | — |
| 5 | Discusión (`discussion`) | ✅ | Síntesis transversal + implicaciones IRIS | — |
| 6 | Conclusiones / líneas futuras | ❌ | **Aún #SeAcabó del TFG viejo** | **Alta** |
| Anexos | Presupuesto / Código / AI Act | 🟡 | Activos; revisar contenido | Media |

---

## Estado técnico del `.tex` principal

Resueltos en la sesión de reescritura (2026-08-02 → 04):

- ✅ Rutas `\input` con prefijo duplicado corregidas (Cap 5 intro/discusión, anexos ya no aplican).
- ✅ Label duplicado `sec:discussion` resuelto (bloque viejo eliminado).
- ✅ `main.tex` **sin bloques `\iffalse`** ni `%\input`/`%\section` comentados del TFG viejo (Cap 2 hate-speech, Cap 4 preprocessing/ML/DL, Cap 5 residual, anexos zero-shot/ML).
- ✅ Intros de capítulo (Metodología, Implementación) en español; "Anthropic" quitado.
- ✅ Fichero `iris_experimento21.tex` renombrado a `experimentacion.tex` (etiqueta `sec:iris_experimentacion`).
- ✅ 42 `.tex` residuales borrados + imágenes huérfanas (gemeco, hatemedia1, creativecommons).

Pendiente de comprobar al compilar en Overleaf:

- [ ] Que `experimentacion.tex` referencia `subsec:limitacion_fiabilidad` (definido en `iris_analisis_experto`) — verificar en compilación.
- [ ] Figuras PDF (`fig_b0b1_metricas.pdf`, `fig_divergencia.pdf`) presentes en `imagenes/`.
- [ ] Abstract y dedicatoria: aún en **rojo/placeholder** y describen la propuesta antigua (Chain-of-Thought, "entrenar", RAG como eje). Alinear con lo hecho (Agent Skills, B0/B1, hallazgos).

---

## Hallazgos del Cap 5 (para no perder el hilo)

- **Las Agent Skills solo mejoran a gemini**, y ahí casi todo el efecto está en `masc_generico`. En gpt empeoran; en gemma local igualan al control.
- **La tool RAG en vivo (`CONSULTAR_GUIA`) nunca ayuda** (demostrado en los 4 modelos); mejor config = skills + resúmenes sin tool.
- **Divergencia exactitud–κ**: reportar solo exactitud llevaría a la conclusión contraria.
- **Techo de la tarea**: heterogeneidad entre equipos (Indexa vs UCM3, 5,7× en `sexismo_discurso`) e intra-equipo; el sistema como tercer anotador cae fuera del rango humano (infra-detección).
- **Modelo local gratis (gemma4:e4b)** tiene el mayor κ de control de los 4.
- **exp22 (confianza) excluido** del TFM por decisión propia.

---

## Checklist de trabajo restante

- [x] Reescribir Cap. 3 (datos IMIO + variables IRIS)
- [x] Reescribir Cap. 4 (arquitectura, prompts, RAG, eval, PoC)
- [x] Reescribir Cap. 5 (experimentación + resultados + discusión, con gemma, ablación y equipo_ia)
- [x] Limpiar `main.tex` y ficheros del TFG anterior
- [x] Quitar qwen3 y claude de todo el TFM
- [ ] **Reescribir Cap. 6** (conclusiones + líneas futuras)
- [ ] Alinear **abstract** con lo realmente hecho
- [ ] Pulir Cap. 1 (rojo/RTVE en dedicatoria y resumen)
- [ ] Compilar en Overleaf y revisar refs/figuras

---

## Estructura de carpetas

```
Experimentos/TFM/
├── DIARIO_TFM.md                 ← bitácora diaria
├── ESTADO_LATEX.md               ← este checklist
├── REUNION_DIRECTORA.md          ← notas y tablas para la reunión
├── TFM___JORGE_GARCELA_N_GO_MEZ/ ← fuentes LaTeX (editar aquí)
└── TFM___JORGE_GARCELA_N_GO_MEZ.zip  ← backup (ignorado por git: *.zip)
```
