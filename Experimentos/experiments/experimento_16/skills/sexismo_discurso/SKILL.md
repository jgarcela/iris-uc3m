---
name: sexismo_discurso
description: Clasifica la variable sexismo_discurso (V30) en artículos periodísticos. Ejecuta la metodología paso a paso antes de asignar codigo.
---

# Variable sexismo_discurso (V30)

Skill del Experimento 16. Clasifica **`sexismo_discurso`** a nivel de documento.

## Inspiración metodológica
La metodología operativa de esta skill está codificada en `variables.json` y se **inspira** en las guías de `Experimentos/pruebas_skills_ollama/methodology/` (tesis Sainz de Baranda, guías de lenguaje inclusivo y no sexista, CSD, etc.). Esas guías son referencia bibliográfica; en clasificación se ejecutan los pasos declarados en la sección Metodología de esta skill.

## Definición
Sexismo social o cultural. A diferencia del lingüístico (V25), opera sobre el FONDO del mensaje: la realidad que el texto describe, normaliza o presenta como natural. Las palabras pueden estar gramaticalmente bien elegidas, pero lo que se cuenta refleja o legitima asimetrías de poder, estereotipos o desigualdad estructural.

**Criterio operativo:** Existe sexismo social cuando el texto: (a) describe asimetrías de poder entre hombres y mujeres sin marcarlas como problemáticas (las normaliza); (b) presenta estereotipos de género como hechos naturales; (c) invisibiliza estructuralmente a las mujeres en ámbitos donde están presentes; o (d) atribuye roles sociales según sexo sin cuestionarlos. Distinguir de la denuncia crítica (V17): si el texto SEÑALA la desigualdad como problema, NO es V30; si la presenta como dato neutro o como hecho legítimo, SÍ es V30.

## Metodología (ejecutar en orden, sin saltar pasos)
**Paso 1 — Separar forma y fondo**
Ignora momentáneamente las palabras concretas. Pregúntate: ¿qué realidad está retratando este texto? ¿Qué relaciones, roles o jerarquías muestra?

**Paso 2 — Evaluar actitud textual**
El texto presenta esa realidad como (a) un problema a denunciar [→ NO es V30, posiblemente V17], (b) un dato informativo neutro [→ ZONA GRIS], (c) algo natural o legítimo [→ V30].

**Paso 3 — Test de naturalizacion**
Pregúntate: si la realidad descrita fuese al revés (mujeres ocupando el rol de poder o autoridad), ¿el texto sonaría natural o forzado? Si la inversión social produce extrañeza, hay sexismo social en el original.

**Paso 4 — Zonas grises**
El dato neutro sobre una asimetría real (ej. 'el Consejo lo componen 20 hombres y 2 mujeres') es V30 si SE PRESENTA sin contexto crítico — el lector recibe el dato como 'esto es así'. Si va acompañado de análisis sobre por qué, es denuncia (V17), no V30.

## Códigos posibles
1 = No
2 = Sí

## Ejemplos donde SÍ aplica
1. «El Consejo está compuesto por 20 varones y 2 mujeres»
   Razón: Dato presentado sin marcaje crítico. La asimetría se naturaliza por la mera enunciación.
2. «Su marido ingresó en el hospital tras el accidente; ella lo cuidó día y noche»
   Razón: Asigna implícitamente el rol cuidador a la esposa como hecho natural.
3. «Las mujeres tienen una función biológica que las hace más adecuadas para la crianza»
   Razón: Presenta como verdad biológica un constructo cultural.

## Contraejemplos (NO marcar)
1. «Solo 2 de los 22 miembros del Consejo son mujeres, un dato que evidencia la brecha que persiste en los órganos de poder corporativo»
   No aplica: Mismo dato que el primer ejemplo positivo, pero AQUÍ se enmarca explícitamente como brecha problemática. Esto es V17 (denuncia), no V30.
2. «La ministra de Defensa presentó el nuevo plan estratégico ante el Congreso»
   No aplica: Hecho informativo sin asimetrías presentadas, sin estereotipo, sin naturalización. Es una noticia neutra.
3. «El estudio del INE muestra que la brecha salarial se redujo dos puntos respecto al año anterior»
   No aplica: Reporta la asimetría como dato medido en un contexto de seguimiento, no la legitima.

## Caso límite documentado
Texto: «El Papa Francisco recibió en audiencia a la familia real»
Decisión: NO es V30
Explicación: Describe una situación factual (Papa = varón por institución, familia real = mixta). No naturaliza asimetría que no esté en la propia realidad institucional descrita. Una noticia sobre las restricciones del cardenalato a varones sería distinto: la PRESENTACIÓN importa.

## Fronteras con otras variables
{'v25_lenguaje_sexista': 'V25 trata el cómo se DICE (palabras). V30 trata el qué se CUENTA (realidad). Pueden coexistir o no.', 'v17_denuncia_desigualdad': 'V17 y V30 son polos opuestos sobre el mismo material. Si el texto DENUNCIA la asimetría → V17. Si la NORMALIZA o la describe como hecho natural → V30. Un texto bien escrito sobre desigualdad puede activar V17 y NO V30.', 'v31_androcentrismo': "V31 sitúa al varón como centro/referente. V30 describe asimetrías de poder. V31 es un MECANISMO; V30 es el CONTENIDO. Una noticia que solo cita expertos varones es V31; una noticia que dice 'las mujeres prefieren X por su naturaleza' es V30."}

## Salida esperada para esta variable

Tras aplicar la metodología, aporta en el JSON unificado:

- Clave: `sexismo_discurso`
- Campos: `codigo`, `explicacion` (citando pasos aplicados), `evidencias` (literales o `[]` si codigo=1)
