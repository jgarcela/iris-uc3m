# Estado del LaTeX vs experimentos IRIS

Checklist para ir cerrando el TFM juntos.
Raíz LaTeX: `TFM___JORGE_GARCELA_N_GO_MEZ/Plantilla_TFG_ingles_2019/`
Fuente experimental: `Experimentos/experiments/`, `CLUSTER/`, `Experimentos.ipynb`

Leyenda: ✅ alineado IRIS · 🟡 parcial / mezcla · ❌ residual TFG anterior · ⬜ por escribir

---

## Mapa capítulo → contenido real

| Cap. | Sección LaTeX | Estado | Qué debería contar (repo) | Prioridad |
|------|---------------|--------|---------------------------|-----------|
| 1 | Introducción | ✅ | Caso Bindi, HITL, IRIS_IAMEDIA como GT | Baja (pulir) |
| 1 | Objetivos y motivación | ✅ | — | Baja |
| 1 | PT / planificación | 🟡 | Revisar si refleja cluster + exp 13–19 | Media |
| 1 | Estructura / marco / impacto / riesgos / contexto | 🟡 | Contexto IRIS OK; revisar coherencia | Baja |
| 2 | Rep. mujeres en medios | ✅ | Estado del arte | Baja |
| 2 | Detección sesgos | ✅ | — | Baja |
| 2 | Justificación solución | ✅ | — | Baja |
| 3 | Análisis experto | ❌ | Aún habla de #SeAcabó / Análisis General / Insultos | **Alta** |
| 3 | Fuente de datos | ❌ | Corpus Twitter Alba Adá; debe ser corpus IMIO / Clara | **Alta** |
| 3 | Variables del análisis | 🟡 | Revisar vs `variables.json` + codebook IRIS | **Alta** |
| 3 | Solución IA / LLMs / GenAI / zero-shot | 🟡 | Texto corto o genérico; alinear con pipeline actual | Media |
| 4 | Agentes | ❌ | Stub copiado (split train/test ML); no describe arquitectura real | **Alta** |
| 4 | Prompts | ❌ | Mismo stub que agents | **Alta** |
| 4 | RAG | ❌ | Mismo stub; además RAG se eliminó del flujo excel | Media |
| 4 | Evaluación | ❌ | Stub; métricas reales en `metrics.py` + notebook | **Alta** |
| 4 | PoC editorial | ❌ | Stub | Media |
| 5 | Resultados (intro) | ❌ | Tabla 1090 exps. / Análisis General / Insultos | **Alta** |
| 5 | «Análisis General» + insultos_llms | ❌ | Residual TFG; sustituir por Exp 1–19 | **Alta** |
| 5 | Discusión | ❌ | Discusión antigua | **Alta** |
| 6 | Conclusiones / líneas futuras | 🟡 | Revisar si hablan del TFG o del IRIS | Media |
| Anexos | Prompts zeroshot ageneral/cnegativo/insultos | ❌ | Sustituir por prompts e1/e2/e3 + skills | Media |

---

## Bugs técnicos del `.tex` principal

Al compilar desde `Plantilla_TFG_ingles_2019/`, estos `\input` están mal (prefijo duplicado):

- `Plantilla_TFG_ingles_2019/chapters/3 methodology/data_source`
- `Plantilla_TFG_ingles_2019/chapters/3 methodology/eda_targetvariables`
- `Plantilla_TFG_ingles_2019/chapters/5 results and discussion/results_and_discussion`
- `Plantilla_TFG_ingles_2019/chapters/5 results and discussion/discussion`
- Varios `\input{Plantilla_TFG_ingles_2019/appendix/...}`

Deben ser relativos tipo `chapters/...` y `appendix/...` (como el resto).

También: `introduction.tex` incluye imagen con ruta `Plantilla_TFG_ingles_2019/imagenes/...` pero `\graphicspath{{imagenes/}}` → conviene `bindi_investigadores.png` a secas.

Labels duplicados: `\label{zero_shot}` aparece dos veces (genai_models y zero_shot).

---

## Qué experimentos meter en Cap. 5

**Fuente de verdad del alcance:** [`PLAN_EXPERIMENTOS_TFM.md`](PLAN_EXPERIMENTOS_TFM.md)  
(no meter Exp 1–12 como eje principal).

1. **Eje A — Benchmark multi-proveedor** (5 vars, baseline prompt/JSON): local (`gemma4:e4b` ± `qwen3:8b`), Claude, OpenAI, Gemini.  
2. **Eje B — Core:** Agent Skills / agentes vs baseline (Exp 16 vs 15; extender skills a más proveedores).  
3. **Infra (Cap. 4 / anexo):** cluster TSC, `utils` multi-proveedor.

Detalle día a día: `DIARIO_TFM.md`. Tablas/números: `Experimentos.ipynb`.
---

## Checklist de trabajo (orden sugerido)

- [ ] Arreglar rutas `\input` / figuras / labels duplicados (compilar sin error)
- [ ] Reescribir Cap. 3: datos IMIO + variables IRIS (quitar #SeAcabó)
- [ ] Reescribir Cap. 4: pipeline real (utils multi-proveedor, prompts, skills, cluster; decidir qué decir de RAG)
- [ ] Reescribir Cap. 5: estructura Línea A/B/C + tablas desde notebook/CLUSTER
- [ ] Alinear Cap. 6 con resultados IRIS
- [ ] Limpiar anexos de prompts del TFG anterior
- [ ] Decidir si Exp 20 y 18 bis entran en la memoria

---

## Estructura de carpetas

```
Experimentos/TFM/
├── DIARIO_TFM.md                 ← bitácora diaria
├── ESTADO_LATEX.md               ← este checklist
├── TFM___JORGE_GARCELA_N_GO_MEZ/ ← fuentes LaTeX (editar aquí)
└── TFM___JORGE_GARCELA_N_GO_MEZ.zip  ← backup (ignorado por git: *.zip)
```
