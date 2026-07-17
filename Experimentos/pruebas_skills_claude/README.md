# Anotador de variables sobre texto periodístico

Pipeline agéntico para codificar texto periodístico contra un codebook de variables y etiquetas. Devuelve JSON estructurado con spans literales, etiquetas validadas, justificaciones y resumen agregado.

## Estructura

```
anotador-periodismo/
├── skill/
│   └── SKILL.md                  # Instrucciones para el agente (portable)
├── schemas/
│   └── anotacion.schema.json     # Schema de validación de la salida
├── codebooks/
│   └── ejemplo_genero_politica.yaml  # Codebook de ejemplo (sustitúyelo)
├── scripts/
│   └── anotador.py               # Pipeline ejecutable
├── examples/
│   └── articulo.txt              # Texto de prueba
└── README.md
```

## Arquitectura del pipeline

```
texto + codebook
      ↓
[Fase 1] Extracción ─── Claude (con SKILL.md como system prompt)
      ↓ JSON candidato
[Fase 2] Verificación de spans ─── determinista, str.find()
      ↓ anotaciones con span literal confirmado
[Fase 3] Validación contra codebook ─── determinista, set membership
      ↓ anotaciones con (variable, etiqueta) válidas
[Fase 4] Validación de schema ─── determinista, jsonschema
      ↓
ResultadoAnotacion final
```

Las fases 2-4 son deterministas y baratas. Detectan tres clases de error que los LLMs cometen sistemáticamente:

1. **Alucinación de spans**: el modelo cita "ministra visiblemente nerviosa" cuando el texto dice "claramente nerviosa". `str.find()` lo descarta.
2. **Etiquetas inventadas**: el modelo crea categorías nuevas no declaradas. La validación contra el catálogo del codebook las elimina.
3. **Estructura malformada**: campos faltantes, tipos incorrectos. JSON Schema lo señala.

Solo la Fase 1 consume tokens. Si quieres el patrón multi-agente con extractor → codificador → verificador como tres llamadas separadas, basta con descomponer `fase_1_extraer` en tres prompts encadenados.

## Uso

### Setup

```bash
pip install anthropic pyyaml jsonschema
export ANTHROPIC_API_KEY=sk-ant-...
```

### Ejecutar

```bash
python scripts/anotador.py \
    --texto examples/articulo.txt \
    --codebook codebooks/ejemplo_genero_politica.yaml \
    --salida examples/articulo.anotado.json \
    --html examples/articulo.anotado.html
```

La salida estándar reporta anotaciones válidas, spans descartados, etiquetas inválidas y errores de schema. El JSON queda en `--salida` y opcionalmente un HTML con highlights en `--html`.

### Como librería

```python
from pathlib import Path
import yaml
from scripts.anotador import anotar

texto = Path("mi_articulo.txt").read_text(encoding="utf-8")
codebook = yaml.safe_load(Path("mi_codebook.yaml").read_text(encoding="utf-8"))

resultado = anotar(texto, codebook)

print(resultado.num_anotaciones)
print(resultado.es_valido)
print(resultado.json_anotaciones)
```

## Sustituir el codebook

El codebook YAML tiene una estructura mínima:

```yaml
variables:
  - nombre: mi_variable
    descripcion: Qué captura.
    etiquetas: [etq1, etq2, etq3]
    multi_span: true              # ¿puede aparecer varias veces?
    depende_de: otra_variable     # opcional
    criterio: |
      Instrucciones operativas para el codificador.
    ejemplos_positivos:
      - {span: "ejemplo del texto", etiqueta: etq1, motivo: "por qué"}
```

No tienes que cambiar nada del código. El script lee tu YAML y lo serializa al prompt. Solo asegúrate de:

- Cada variable tiene un `nombre` único, una lista cerrada de `etiquetas` y un `criterio` operativo.
- Si una variable depende de otra, decláralo con `depende_de` y declárala después en orden.
- Los ejemplos positivos ayudan mucho. Tres a cinco por variable es lo óptimo.

## Cuándo este patrón no basta

Cambia a multi-agente paralelo (un agente por variable) si:

- Tu codebook tiene **20+ variables**.
- Una sola pasada produce contexto excesivo (texto largo + codebook largo).
- Las variables son muy heterogéneas (unas léxicas, otras de framing global, otras de cita).

Cambia a fine-tuning si:

- Tienes **>500 textos anotados manualmente** como gold standard.
- El codebook está estabilizado y no va a cambiar.
- Te importa el coste por inferencia más que la flexibilidad.

Para investigación exploratoria con codebook en evolución, este patrón pipeline es lo correcto.

## Métricas que conviene medir desde el día uno

Cuando empieces a tener anotaciones humanas para comparar:

- **Recall por variable**: cuántas ocurrencias reales captura el agente.
- **Precision por variable**: cuántas de sus anotaciones son correctas.
- **Acuerdo etiqueta**: dado un span correctamente identificado, ¿asignó la etiqueta correcta?
- **Acuerdo de span**: jaccard entre spans humanos y de agente para la misma ocurrencia.
- **Asimetría detectada**: ¿reproduce el agente el sesgo o lo identifica?

Esto último es crucial. El paper de Pastorino et al. (2024, arXiv:2402.11621) muestra que GPT-4 confunde lenguaje emocional con framing. Si tu codebook tiene variables sensibles a esto (`frame_dominante`, `descriptor_sexista`), mide el falso positivo en oraciones neutras.

## Adaptaciones al ecosistema Claude

- **Como skill en Claude.ai/Code**: copia `skill/SKILL.md` a `~/.claude/skills/anotador-periodismo/SKILL.md`. Al pedirle a Claude que anote un texto contra un codebook, cargará la skill automáticamente.
- **Como Agent Skill por API**: sube el directorio `skill/` con la API de Skills (beta). Después invocas el modelo con `skills: [{"type": "custom", "skill_id": "..."}]`.
- **Con Claude Code**: añade el repo como plugin marketplace. La skill se activará cuando detecte tareas de anotación.
