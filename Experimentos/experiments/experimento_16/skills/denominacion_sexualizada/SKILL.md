---
name: denominacion_sexualizada
description: Clasifica la variable denominacion_sexualizada (V35) en artículos periodísticos. Ejecuta la metodología paso a paso antes de asignar codigo.
---

# Variable denominacion_sexualizada (V35)

Skill del Experimento 16. Clasifica **`denominacion_sexualizada`** a nivel de documento.

## Inspiración metodológica
La metodología operativa de esta skill está codificada en `variables.json` y se **inspira** en las guías de `Experimentos/pruebas_skills_ollama/methodology/` (tesis Sainz de Baranda, guías de lenguaje inclusivo y no sexista, CSD, etc.). Esas guías son referencia bibliográfica; en clasificación se ejecutan los pasos declarados en la sección Metodología de esta skill.

## Definición
Identificar a mujeres exclusivamente por su sexo biológico, condición corporal o roles familiares y reproductivos, eclipsando su identidad profesional, cívica o pública. La mujer queda reducida a su categoría sexual/reproductiva, no a su persona o función social.

**Criterio operativo:** Existe V35 cuando: (a) la mujer es identificada con 'mujer' como sustantivo principal en contexto donde el cargo profesional sería esperable ('dos bomberos y dos mujeres' en lugar de 'cuatro bomberos/as'); (b) la denominación principal es por rol biológico/reproductivo en contexto profesional ('madre de oro'); (c) se usan apelativos sexualizadores ('lolitas', 'bellezas') que reducen a la mujer a objeto.

## Metodología (ejecutar en orden, sin saltar pasos)
**Paso 1 — Localizar referencias a mujeres**
Identifica todas las menciones a mujeres y la categoría principal por la que se las nombra (profesión, cargo, rol familiar, sexo biológico, etc.).

**Paso 2 — Test de reduccion**
¿La denominación reduce a la mujer a su sexo, cuerpo, o rol reproductivo cuando el contexto requería identificación profesional o cívica?

**Paso 3 — Test de inversion**
¿Se usaría una categoría equivalente para hombres en el mismo contexto? 'Un padre de oro' por 'campeón olímpico' no se usaría: la maternidad como denominación profesional es asimétrica.

**Paso 4 — Contexto de uso**
Una noticia sobre maternidad puede legítimamente identificar a mujeres como madres. Una noticia sobre deporte que identifica a una deportista solo como madre activa V35.

## Códigos posibles
1 = No
2 = Sí

## Ejemplos donde SÍ aplica
1. «Los testigos son dos bomberos y dos mujeres»
   Razón: Las mujeres son identificadas por su sexo cuando el contexto requería 'bomberas' o 'cuatro testigos'.
2. «Maialen Chourraut, una madre de oro»
   Razón: Deportista olímpica identificada por su rol familiar reproductivo. Eclipsa identidad profesional.
3. «Lolitas para endulzar la sección de moda»
   Razón: Apelativo sexualizador ('lolitas' = niñas-sexualizadas, referencia a Nabokov) aplicado a modelos profesionales.

## Contraejemplos (NO marcar)
1. «La empresaria Ana Botín presentó los resultados anuales»
   No aplica: Identificación profesional ('empresaria'), no biológica ni familiar.
2. «La madre de la víctima testificó ante el tribunal»
   No aplica: Rol familiar relevante para la noticia (testificó por ser madre de la víctima), no sustituye identidad profesional.
3. «Las nadadoras españolas, entre ellas tres madres recientes, prepararon los Juegos»
   No aplica: Identifica profesionalmente como 'nadadoras' y añade información biográfica relevante para el contexto deportivo (compatibilidad maternidad/élite). No reduce ni eclipsa.

## Caso límite documentado
Texto: «La esposa del nuevo CEO acudió a la presentación»
Decisión: Depende del contexto. SÍ V37 si solo se la identifica así
Explicación: Si 'la esposa' es la única identificación de una mujer protagonista, activa V37 (dependiente). Si la esposa es accesoria al evento (acompañante meramente mencionada), puede ser información contextual no problemática. La frontera está en la centralidad narrativa.

## Fronteras con otras variables
{'v28_uso_dual_zorr': "V28 trata pares léxicos con asimetría intrínseca ('zorra/zorro'). V35 trata la elección de categoría sustantiva ('madre' vs 'campeona'). Pueden coexistir.", 'v34_infantilizacion': "Si la denominación es además infantilizadora ('chicas', 'nenas'), activa también V34.", 'v37_dependiente': "V37 = nombrada por relación familiar con un VARÓN concreto. V35 = nombrada por rol biológico-familiar genérico. 'La señora de Karembeu' = V37. 'Una madre' = V35."}

## Salida esperada para esta variable

Tras aplicar la metodología, aporta en el JSON unificado:

- Clave: `denominacion_sexualizada`
- Campos: `codigo`, `explicacion` (citando pasos aplicados), `evidencias` (literales o `[]` si codigo=1)
