---
name: asimetria_mujer_hombre
description: Clasifica la variable asimetria_mujer_hombre (V33) en artículos periodísticos. Ejecuta la metodología paso a paso antes de asignar codigo.
---

# Variable asimetria_mujer_hombre (V33)

Skill del Experimento 16. Clasifica **`asimetria_mujer_hombre`** a nivel de documento.

## Inspiración metodológica
La metodología operativa de esta skill está codificada en `variables.json` y se **inspira** en las guías de `Experimentos/pruebas_skills_ollama/methodology/` (tesis Sainz de Baranda, guías de lenguaje inclusivo y no sexista, CSD, etc.). Esas guías son referencia bibliográfica; en clasificación se ejecutan los pasos declarados en la sección Metodología de esta skill.

## Definición
Uso desigual del tratamiento nominal en un mismo texto: las mujeres son referidas por nombre de pila (familiarización) mientras los hombres son referidos por apellido o nombre completo (formalización). Esta asimetría infantiliza, familiariza o subordina a la mujer respecto al hombre del mismo texto.

**Criterio operativo:** Existe V33 cuando, en el mismo texto, hay una asimetría detectable: hombres tratados como 'apellido' o 'señor X' y mujeres tratadas como 'nombre de pila' o 'señora X' sin apellido. Test: aplica la inversión: ¿si llamáramos a Sánchez 'Pedro' y a Ayuso 'señora Ayuso' o 'Díaz Ayuso', sonaría natural? Si la inversión produce extrañeza, había asimetría sexista.

## Metodología (ejecutar en orden, sin saltar pasos)
**Paso 1 — Enumerar referencias a personas**
Lista todas las personas mencionadas y la forma exacta en que se las menciona en cada ocurrencia.

**Paso 2 — Clasificar tipo referencia**
Por persona: nombre de pila, apellido, nombre+apellido, título+apellido. Anota la fórmula predominante para cada uno/a.

**Paso 3 — Comparar intra texto**
¿Las mujeres tienen fórmula sistemáticamente menos formal que los hombres del mismo texto?

**Paso 4 — Descartar convenciones legitimas**
En ciertos contextos (deporte juvenil, perfiles personales) puede ser legítimo el nombre de pila para todas las personas. Solo activa V33 si la asimetría es entre sexos del mismo texto.

## Códigos posibles
1 = No
2 = Sí

## Ejemplos donde SÍ aplica
1. «Presidieron el acto la presidenta, Inmaculada, y el vicepresidente, Emilio Núñez»
   Razón: Ella nombrada con nombre de pila ('Inmaculada'), él con nombre y apellido ('Emilio Núñez').
2. «Carolina se enfrentará en cuartos a Carlos Alcaraz»
   Razón: Ella por nombre de pila ('Carolina' = Carolina Marín), él con nombre+apellido.
3. «Don Carlos Ramos asistió acompañado de Matilde»
   Razón: Tratamiento formal ('Don', nombre+apellido) para él; coloquial (solo nombre de pila) para ella.

## Contraejemplos (NO marcar)
1. «Sánchez y Ayuso debatieron en el Congreso»
   No aplica: Apellido para ambos. Tratamiento simétrico.
2. «Pedro y María explicaron su decisión»
   No aplica: Nombre de pila para ambos. Simetría conservada (probable contexto informal o familiar).
3. «La presidenta Sánchez recibió al primer ministro Macron»
   No aplica: Apellido para ambos en función de su cargo. Asimetría aparente pero simétrica por convención diplomática.

## Caso límite documentado
Texto: «La Vigara firmó por el Real Madrid»
Decisión: SÍ es V33 si en el mismo texto hay hombres con nombre+apellido
Explicación: El uso de artículo + apellido ('la Vigara') es marca de informalidad/familiaridad sobre figuras femeninas en deporte que no se aplica al uso masculino contemporáneo. Lo opuesto sería decir 'el Sánchez' al presidente, que sonaría irrespetuoso. Caso histórico bien documentado en prensa deportiva (Sainz de Baranda, 2013).

## Fronteras con otras variables
{'v32_sin_nombre': 'V32 = mujer SIN nombre. V33 = mujer CON nombre pero solo de pila vs hombre con apellido. Distintos niveles.', 'v34_infantilizacion': "Si el nombre de pila es además diminutivo o forma infantil ('Caro' por Carolina, 'Lola' por Dolores en cargo profesional), activa V34 además de V33.", 'v37_dependiente': "V37 = nombrada por relación con varón ('la mujer de X'). V33 = nombrada por nombre de pila propio pero asimétricamente."}

## Salida esperada para esta variable

Tras aplicar la metodología, aporta en el JSON unificado:

- Clave: `asimetria_mujer_hombre`
- Campos: `codigo`, `explicacion` (citando pasos aplicados), `evidencias` (literales o `[]` si codigo=1)
