---
name: anotador-periodismo
description: Anota un texto periodístico contra un codebook de variables y etiquetas. Identifica spans literales, asigna etiquetas, justifica cada decisión y devuelve un JSON estructurado más versión marcada del texto. Úsalo cuando recibas un texto periodístico junto con un codebook (variables + etiquetas cerradas) y se te pida codificarlo.
---

# Anotador de variables sobre texto periodístico

## Cuándo activarte

El usuario aporta:

1. Un **texto periodístico** (artículo, reportaje, transcripción)
2. Un **codebook**: lista de variables, cada una con sus etiquetas cerradas, un criterio operativo y opcionalmente ejemplos

El usuario pide identificar las ocurrencias de las variables, asignarles etiqueta, justificar la decisión y devolver el texto marcado.

No te actives si solo se pide resumir, traducir, opinar o reescribir. Esta skill es estrictamente para anotación contra un esquema predefinido.

## Principios irrenunciables

Cuatro reglas que no se negocian:

1. **Solo spans literales**. El campo `texto` de cada anotación debe ser una subcadena exacta del documento, copiada carácter por carácter incluyendo mayúsculas y signos. Si no puedes encontrarlo con `texto_original.find(span)`, no lo emitas.
2. **Etiquetas cerradas**. Solo etiquetas declaradas en el codebook. Si dudas entre dos, elige la más conservadora y bájala la confianza. Si ninguna aplica, marca la variable como ausente para ese span en lugar de inventar etiqueta.
3. **Trazabilidad por oración**. Numera las oraciones del texto (S1, S2…) antes de empezar y cita en cada anotación la oración de la que procede.
4. **Una pasada por variable, no por mención**. Recorre el texto una vez para cada variable del codebook. Esto evita el sesgo de saliencia (no anotes solo las menciones obvias).

## Flujo en tres fases

### Fase 1 — preparación

1. Lee el codebook completo. Confirma que entiendes cada variable, sus etiquetas, su criterio y si admite multi-span.
2. Segmenta el texto en oraciones numeradas. Mantén un mapa `S<n>` → texto literal de la oración.
3. Si el codebook trae ejemplos, cárgalos como referencia pero **no los copies** en la salida.

### Fase 2 — anotación variable por variable

Para cada variable del codebook:

1. Recorre el texto buscando ocurrencias del fenómeno que define el criterio de la variable.
2. Para cada ocurrencia candidata:
   - Identifica el span mínimo suficiente: el fragmento más corto que captura el fenómeno sin truncarlo.
   - Asigna una etiqueta del conjunto cerrado.
   - Escribe una justificación de una a dos frases que cite el criterio del codebook y, si aplica, contraste con la asimetría de género (por ejemplo: «descriptor que rara vez se aplica a un hombre en posición equivalente»).
   - Asigna confianza en `[0, 1]`. Usa 0.95+ solo cuando el span y la etiqueta son inequívocos. Baja a 0.6-0.8 cuando hay ambigüedad razonable.
3. Si la variable admite multi-span, no fusiones ocurrencias separadas. Cada aparición es una anotación independiente.

### Fase 3 — verificación

Antes de devolver el resultado:

1. Para cada anotación, ejecuta mentalmente `texto_original.find(span)`. Si no encuentras el span, **descarta la anotación**.
2. Comprueba que toda etiqueta esté en el conjunto declarado por el codebook.
3. Detecta solapamientos sospechosos: si dos anotaciones de la misma variable cubren spans superpuestos, conserva la más específica.
4. Genera el bloque `resumen` con conteos por variable y etiqueta.

## Esquema de salida

Devuelve **únicamente** este JSON, sin texto previo ni posterior, sin code fences:

```json
{
  "documento_id": "<id si se proporciona, si no null>",
  "oraciones": [
    {"id": "S1", "texto": "..."},
    {"id": "S2", "texto": "..."}
  ],
  "anotaciones": [
    {
      "id": "a1",
      "variable": "<nombre de variable del codebook>",
      "etiqueta": "<etiqueta del conjunto cerrado>",
      "texto": "<span literal copiado del original>",
      "oracion_id": "S<n>",
      "explicacion": "<una a dos frases citando el criterio>",
      "confianza": 0.0
    }
  ],
  "resumen": {
    "total_anotaciones": 0,
    "por_variable": {"<variable>": {"<etiqueta>": 0}},
    "asimetrias_detectadas": []
  }
}
```

El campo `asimetrias_detectadas` es opcional y útil para género: lista hallazgos del tipo «todos los descriptores de apariencia se aplicaron a personas etiquetadas como F».

## Ejemplo abreviado

Codebook (resumido):

```yaml
- nombre: atribucion_genero
  etiquetas: [F, M, no_binario, colectivo, indeterminado]
- nombre: descriptor_sexista
  etiquetas: [apariencia_emocion, rol_familiar, edad, tono_voz, vestimenta, ninguno]
```

Texto: `La ministra, madre de dos hijos, presentó el plan.`

Salida correcta:

```json
{
  "anotaciones": [
    {"id": "a1", "variable": "atribucion_genero", "etiqueta": "F",
     "texto": "ministra", "oracion_id": "S1",
     "explicacion": "Sustantivo morfológicamente femenino.", "confianza": 0.99},
    {"id": "a2", "variable": "descriptor_sexista", "etiqueta": "rol_familiar",
     "texto": "madre de dos hijos", "oracion_id": "S1",
     "explicacion": "Mención de maternidad sin relevancia para el plan presentado. Descriptor que rara vez se aplica a un ministro varón en contexto equivalente.", "confianza": 0.94}
  ]
}
```

## Errores frecuentes a evitar

- Inventar spans aproximados ("ministra nerviosa" cuando el texto dice "claramente nerviosa")
- Crear etiquetas no declaradas en el codebook
- Anotar solo las menciones más obvias y omitir las sutiles
- Justificar con paráfrasis del span en vez de citar el criterio del codebook
- Fusionar ocurrencias separadas en un único span largo
- Sobreusar la etiqueta `ninguno` o `indeterminado` por comodidad

## Cuando el codebook tiene variables interdependientes

Si una variable depende del valor de otra (por ejemplo, `descriptor_sexista` solo cuenta cuando la mención previa es de una persona con `atribucion_genero` asignada), procesa las variables en el orden declarado en el codebook y reutiliza las anotaciones previas como contexto para las posteriores.
