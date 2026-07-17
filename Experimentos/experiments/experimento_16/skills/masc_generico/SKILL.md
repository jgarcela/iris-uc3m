---
name: masc_generico
description: Clasifica la variable masc_generico (V26) en artículos periodísticos. Ejecuta la metodología paso a paso antes de asignar codigo.
---

# Variable masc_generico (V26)

Skill del Experimento 16. Clasifica **`masc_generico`** a nivel de documento.

## Inspiración metodológica
La metodología operativa de esta skill está codificada en `variables.json` y se **inspira** en las guías de `Experimentos/pruebas_skills_ollama/methodology/` (tesis Sainz de Baranda, guías de lenguaje inclusivo y no sexista, CSD, etc.). Esas guías son referencia bibliográfica; en clasificación se ejecutan los pasos declarados en la sección Metodología de esta skill.

## Definición
Uso del género gramatical masculino como 'género no marcado' para designar a individuos de ambos sexos. No es siempre sexista: lo es cuando genera ambigüedad, invisibiliza la presencia de las mujeres, o produce una representación mental que las excluye y subordina al modelo masculino.

**Criterio operativo:** El masculino genérico es problemático cuando: (a) la inversión a femenino genérico produce extrañeza (no consideraríamos 'las profesoras' como inclusivo para hombres); (b) en el mismo texto aparece después una marca explícita que demuestra que no era inclusivo (salto semántico, ver V25); (c) se refiere a un colectivo donde la presencia de mujeres es relevante y se las invisibiliza.

## Metodología (ejecutar en orden, sin saltar pasos)
**Paso 1 — Localizar genericos**
Identifica todos los sustantivos masculinos plurales que pretenden incluir a ambos sexos: 'los profesores', 'los investigadores', 'los alumnos', 'los nadadores'.

**Paso 2 — Test de invisibilizacion**
Pregúntate: ¿este masculino, en este contexto, evoca a hombres y mujeres por igual? ¿O el lector medio entenderá 'solo hombres'? Pista: si el texto luego añade 'y las mujeres también', es claro caso de no-inclusión.

**Paso 3 — Test de inversion**
Reescribe con femenino genérico: '¿Las profesoras del CSIC pidieron una reunión?'. Si el femenino no funcionaría como inclusivo, el masculino tampoco lo es.

**Paso 4 — Alternativas disponibles**
¿Existe forma no marcada (profesorado, alumnado, ciudadanía)? Si sí y no se usa, el masculino genérico está invisibilizando deliberadamente.

## Códigos posibles
1 = No
2 = Sí

## Ejemplos donde SÍ aplica
1. «Este año los tenistas españoles llegan bien posicionados... y las chicas también»
   Razón: Caso ejemplar de salto semántico. 'Tenistas' se reveló no inclusivo.
2. «Los investigadores del CSIC firmaron el manifiesto»
   Razón: Existe 'el personal investigador' o 'investigadoras e investigadores'. Usar solo masculino invisibiliza.
3. «Convención sobre los Derechos del Niño»
   Razón: Caso paradigmático. 'Del Niño' invisibiliza a las niñas. Existía 'de la Infancia' como alternativa neutra.

## Contraejemplos (NO marcar)
1. «El profesorado universitario debate la nueva ley»
   No aplica: Sustantivo colectivo no marcado por género gramatical. Solución correcta del masculino genérico.
2. «Los juegos olímpicos masculinos comienzan mañana»
   No aplica: El adjetivo 'masculinos' hace explícito que no pretende ser genérico. Es uso marcado, correcto.
3. «Las ministras del Gobierno se reunieron con sus homólogas europeas»
   No aplica: Femenino específico, refleja realidad: las ministras existentes son mujeres.

## Caso límite documentado
Texto: «Los ciudadanos tendrán que renovar sus DNI antes de junio»
Decisión: Discutible — depende del contexto institucional
Explicación: En contexto administrativo genérico, 'ciudadanos' se usa como término jurídico que históricamente incluye a todas las personas. Si el texto procede de fuente oficial (BOE, gobierno) con esta convención, el sesgo es del sistema institucional. Marcar V26 si hay alternativa neutra disponible ('la ciudadanía').

## Fronteras con otras variables
{'v25_lenguaje_sexista': 'V26 es un subtipo de V25. Si activas V26, casi siempre activas también V25.', 'v27_hombre_humanidad': "V26 cubre todos los masculinos genéricos plurales ('los investigadores'). V27 es específicamente el uso de 'hombre' (singular o plural) con valor genérico ('los derechos del hombre').", 'v31_androcentrismo': 'V26 es un mecanismo concreto. V31 es la visión del mundo subyacente. Una noticia puede tener V26 sin ser globalmente androcéntrica.'}

## Salida esperada para esta variable

Tras aplicar la metodología, aporta en el JSON unificado:

- Clave: `masc_generico`
- Campos: `codigo`, `explicacion` (citando pasos aplicados), `evidencias` (literales o `[]` si codigo=1)
