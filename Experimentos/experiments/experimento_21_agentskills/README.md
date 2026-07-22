# Experimento 21 — Agent Skills (Eje B / core del TFM)

Implementación **nueva y limpia** del core de agentes del TFM. **Sustituye** a los
experimentos de skills previos (Exp 16 y `pruebas_skills_*`), que quedan descartados
por no ser Agent Skills reales (sin progressive disclosure; ablación confundida con
el nº de llamadas). Diseño canónico: [`../../TFM/ARQUITECTURA_AGENTES_SKILLS.md`](../../TFM/ARQUITECTURA_AGENTES_SKILLS.md).

## Idea
5 **agentes especializados** (uno por variable). Cada agente:
- Ve sólo `name`+`description` de sus skills (progressive disclosure).
- Carga bajo demanda su `SKILL.md` de metodología y, si duda, skills auxiliares.
- Decide por acciones de texto (estilo ReAct) → uniforme en Claude/OpenAI/Gemini/Ollama vía `utils.py` (sin LangChain, sin tool-calling nativo frágil en locales).

**5 llamadas por artículo** (una por variable) → comparable 1:1 con el baseline B0 (Exp 15),
que también hace 5 llamadas. El único cambio B0→B1 es *cómo llega la metodología*
(inyectada vs cargada) → el delta kappa/F1 mide el efecto neto de las skills.

## Estructura
```
skills/<variable>/SKILL.md      # V25,V26,V30,V33,V35 — generadas desde variables.json
skills/guia_regla_inversion/    # auxiliar compartida
skills/guia_lenguaje_inclusivo/ # auxiliar (guías expertas)
skills/verificar_evidencias/    # auxiliar (trazabilidad HITL)
tools.py        # list_skills / read_skill / verificar_evidencias
guias.py        # RAG ligero (TF-IDF) sobre Experimentos/methodology/ → tool consultar_guia
agente.py       # bucle de decisión de UN agente (una variable)
generar_skills.py  # regenera las 5 skills de variable desde variables.json
main.py         # itera corpus → 5 agentes → CSV (código + explicación + evidencias + traza)
```

## Guías expertas en runtime
El agente puede recuperar pasajes **literales** de las guías reales de
[`Experimentos/methodology/`](../../methodology/) (Sainz de Baranda, CSD, guías de lenguaje
inclusivo…) con la acción `CONSULTAR_GUIA: <consulta>`. `guias.py` indexa los `.md` del
`methodology_manifest.json` (TF-IDF en memoria, sin dependencias, la tesis no se indexa) y
devuelve los pasajes con cita de fuente + sección. Así las guías se usan **de verdad**, no
sólo como inspiración del codebook.

## Uso
```bash
# 1) (re)generar las skills de variable
python3 generar_skills.py

# 2) probar un agente sobre un texto
python3 agente.py lenguaje_sexista "Los testigos son dos bomberos y dos mujeres." -v

# 3) corrida sobre corpus
python3 main.py --input ../../<corpus>.csv --modelo claude-haiku-4-5-20251001 \
    --output-dir results/claude-haiku --limit 20
```
(Ejecutar con el intérprete del proyecto, p.ej. `../../../.venv/bin/python`.)

## Traza (métricas de comportamiento del agente)
Por variable, `main.py` guarda `n_tools`, `skills` cargadas y `colapso_b0`
(el agente cerró sin usar ninguna skill → equivalente a B0). Permite reportar
**cuánto usan realmente las tools** los distintos proveedores.

## Estado
- [x] Skills (5 variable + 3 auxiliares), tools, agente, main, generador.
- [x] Bucle validado end-to-end (mock) y tools/progressive disclosure verificados.
- [ ] Corrida real por proveedor + métricas sobre las 1 315 (`metrics.py` — reusar exp 15/16).
- [ ] Prompt caching del bloque de skills a escala 7k.
