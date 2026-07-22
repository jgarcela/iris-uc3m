---
name: denominacion_sexualizada
description: Metodología para clasificar «denominacion_sexualizada» (V35) en un artículo. Cárgala antes de emitir el veredicto de esta variable.
origen_json: variables.json
---

# Variable denominacion_sexualizada (V35)

> Generada desde `variables.json`. Regenerar con `generar_skills.py --json <ruta>`.

Clasifica **`denominacion_sexualizada`** a nivel de documento.

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
   Por qué NO es {nombre}: Identificación profesional ('empresaria'), no biológica ni familiar.
2. «La madre de la víctima testificó ante el tribunal»
   Por qué NO es {nombre}: Rol familiar relevante para la noticia (testificó por ser madre de la víctima), no sustituye identidad profesional.
3. «Las nadadoras españolas, entre ellas tres madres recientes, prepararon los Juegos»
   Por qué NO es {nombre}: Identifica profesionalmente como 'nadadoras' y añade información biográfica relevante para el contexto deportivo (compatibilidad maternidad/élite). No reduce ni eclipsa.

## Caso límite documentado
Texto difícil: «La esposa del nuevo CEO acudió a la presentación»
Decisión: Depende del contexto. SÍ V37 si solo se la identifica así
Explicación: Si 'la esposa' es la única identificación de una mujer protagonista, activa V37 (dependiente). Si la esposa es accesoria al evento (acompañante meramente mencionada), puede ser información contextual no problemática. La frontera está en la centralidad narrativa.

## Fronteras con otras variables
- **v28_uso_dual_zorr:** V28 trata pares léxicos con asimetría intrínseca ('zorra/zorro'). V35 trata la elección de categoría sustantiva ('madre' vs 'campeona'). Pueden coexistir.
- **v34_infantilizacion:** Si la denominación es además infantilizadora ('chicas', 'nenas'), activa también V34.
- **v37_dependiente:** V37 = nombrada por relación familiar con un VARÓN concreto. V35 = nombrada por rol biológico-familiar genérico. 'La señora de Karembeu' = V37. 'Una madre' = V35.

## Salida
Tras aplicar la metodología emite: `FINAL: {"codigo": <n>, "explicacion": "...cita pasos...", "evidencias": [literales o []]}`.
