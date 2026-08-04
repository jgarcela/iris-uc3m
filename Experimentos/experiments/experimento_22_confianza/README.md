# Experimento 22 — Confianza / P(Sí)

Estudia la **señal de confianza** del modelo, no la arquitectura. Cada variable se
clasifica en una única llamada (como el B0 del Exp 21) pidiendo además `prob_si`
(0-100): la probabilidad de que la variable esté presente. Nace del hallazgo del
Exp 21 (los modelos **infra-detectan**): con una probabilidad continua se puede
medir calibración y **mover el umbral** para recuperar acuerdo con el criterio
experto amplio.

**No modifica el Exp 21.** Reutiliza por importación (solo lectura) su metodología
(`tools.read_skill`), su tabla de costes (`costes.py`) y el enrutado multi-proveedor
(`utils.py`). Todo lo propio vive aquí.

## Ficheros
```
clasificador.py     # B0 + prob_si (prompt y parseo propios)
main.py             # runner sobre el corpus (reanudable, coste por artículo)
dev_test_split.csv  # 300 dev / 1013 test (determinista, random_state=42)
```

## Diseño (imprescindible: dev/test)
El barrido de umbral **ajusta un corte mirando datos** → hay que hacerlo en **dev**
y reportar en **test** (una sola vez). Sin esto sería overfitting al test.

## Lanzar (3 modelos, B0 con confianza)
```bash
for m in gemini-3.1-flash-lite gpt-4o-mini gpt-5.4-nano; do
  nohup ../../../.venv/bin/python main.py \
    --input ../experimento_21_agentskills/real1315_corpus.csv \
    --modelo $m --only-labeled --output-dir results/$m > /tmp/exp22_$m.log 2>&1 &
done
```
Coste ~$14 total · ~2-3 h en paralelo (1 llamada/variable). Reanudable por IdNoticia.

## Análisis previstos (`analisis.py`, pendiente)
1. **Calibración** — curva de fiabilidad (prob declarada vs acierto real), en dev.
2. **Barrido de umbral** — τ que maximiza F1/κ en dev; se reporta en test.
   ¿Bajar el corte recupera el acuerdo con la anotación (que es muy inclusiva)?
3. **Confianza en los desacuerdos** — ¿los casos modelo=No/humano=Sí son de baja
   confianza (frontera → techo bajo) o alta (punto ciego / anotación cuestionable)?

Salida compatible con `experimento_21_agentskills/metrics.py` (columnas `modelo_<var>`).
