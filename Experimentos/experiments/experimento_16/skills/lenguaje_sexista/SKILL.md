---
name: lenguaje_sexista
description: Clasifica la variable lenguaje_sexista (V25) en artículos periodísticos. Ejecuta la metodología paso a paso antes de asignar codigo.
---

# Variable lenguaje_sexista (V25)

Skill del Experimento 16. Clasifica **`lenguaje_sexista`** a nivel de documento.

## Inspiración metodológica
La metodología operativa de esta skill está codificada en `variables.json` y se **inspira** en las guías de `Experimentos/pruebas_skills_ollama/methodology/` (tesis Sainz de Baranda, guías de lenguaje inclusivo y no sexista, CSD, etc.). Esas guías son referencia bibliográfica; en clasificación se ejecutan los pasos declarados en la sección Metodología de esta skill.

## Definición
Uso discriminatorio del lenguaje por razón de sexo. El sexismo lingüístico opera sobre la FORMA del mensaje (palabras o estructuras gramaticales elegidas), no sobre su fondo. Un mensaje puede tener contenido no sexista pero estar expresado con formas sexistas, o al revés (ver V30 para el fondo).

**Criterio operativo:** Existe sexismo lingüístico cuando, al aplicar la regla de inversión (sustituir referencias femeninas por masculinas y viceversa), el resultado produce extrañeza, incoherencia o sería inaceptable. Si la inversión no produce nada raro, no hay sexismo lingüístico aunque el mensaje hable de desigualdad.

**Salto semantico:** Subcaso especial de sexismo lingüístico. Ocurre cuando un masculino usado como genérico se revela posteriormente no inclusivo al aparecer una marca explícita de mujer. La frase 'expone' por sí sola que el masculino nunca fue inclusivo. Ejemplo: 'los tenistas españoles llegan bien posicionados... y las chicas también' (el segundo 'y' demuestra que 'tenistas' no incluía a las chicas).

## Metodología (ejecutar en orden, sin saltar pasos)
**Paso 1 — Identificacion**
Localiza pasajes donde aparezca alguna mujer (sustantivo, nombre, pronombre, adjetivo concordado) y observa cómo se la trata gramaticalmente respecto a los hombres del mismo texto.

**Paso 2 — Regla de inversion**
Toma el pasaje candidato y reescríbelo invirtiendo el género (femeninos→masculinos, masculinos→femeninos). Pregúntate: ¿esta frase invertida se publicaría con naturalidad? ¿O suena extraña, ridícula, irrespetuosa? Si lo segundo, hay sexismo en la versión original.

**Paso 3 — Deteccion salto semantico**
Busca específicamente la estructura: [colectivo masculino] + [aparece marca de mujer]. Si encuentras un masculino que se presentó como inclusivo y luego se demuestra que no lo era, hay salto semántico (etiqueta 3).

**Paso 4 — Diferenciacion con v30**
Si el sexismo está en la FORMA (palabras concretas elegibles de otra manera), es V25. Si está en el FONDO del mensaje (lo que se cuenta), es V30.

## Códigos posibles
1 = No
2 = Sí
3 = Sí; además se observa un salto semántico

## Ejemplos donde SÍ aplica
1. «Los testigos son dos bomberos y dos mujeres» → etiqueta 2
   Razón: Asimetría léxica: profesión vs sexo biológico
2. «Maialen Chourraut, una madre de oro» → etiqueta 2
   Razón: Identificación por rol familiar eclipsando profesional
3. «Este año los tenistas españoles llegan bien posicionados... y las chicas también» → etiqueta 3
   Razón: Salto semántico ejemplar

## Contraejemplos (NO marcar)
1. «La brecha salarial de género se redujo un 2% en 2024 según un informe del INE»
   No aplica: Habla DE desigualdad pero sin formas sexistas. La regla de inversión ('La brecha salarial entre hombres se redujo...') no produce extrañeza porque el contenido es informativo. Esto sería V17 (denuncia desigualdad) si describe el problema, no V25.
2. «El equipo de bomberos rescató a tres personas atrapadas en el incendio»
   No aplica: 'Bomberos' funciona como colectivo profesional inclusivo; 'personas' es no marcado. La inversión ('Las personas rescatadas por el equipo de bomberas') no genera extrañeza.
3. «La presidenta del Congreso clausuró la sesión»
   No aplica: Cargo correctamente feminizado, sin asimetría. Inversión: 'El presidente del Congreso clausuró la sesión'.

## Caso límite documentado
Texto: «Las nadadoras españolas han ganado tres medallas»
Decisión: NO es V25
Explicación: 'Nadadoras' está bien feminizado. La inversión ('Los nadadores españoles han ganado tres medallas') tampoco produciría extrañeza. Es un texto sin sexismo lingüístico, aunque hable solo de mujeres.

## Fronteras con otras variables
{'v26_masc_generico': "V25 es la categoría 'paraguas' que incluye TODO sexismo lingüístico. V26 es un subtipo específico: usar el masculino como pretendido genérico. Si la única forma sexista detectada es masc_generico, marca AMBAS V25 y V26.", 'v30_sexismo_discurso': 'V25 = problema en las PALABRAS elegidas (forma). V30 = problema en lo QUE SE CUENTA (fondo). Una frase puede ser V25 sin V30, V30 sin V25, o ambas.', 'v25_vs_subtipos': 'V25 actúa como agregador. Si detectas V26, V27, V28, V29, V31-V39, también deberías marcar V25 porque todas son subtipos de sexismo lingüístico (excepto V30 que es de fondo).'}

## Salida esperada para esta variable

Tras aplicar la metodología, aporta en el JSON unificado:

- Clave: `lenguaje_sexista`
- Campos: `codigo`, `explicacion` (citando pasos aplicados), `evidencias` (literales o `[]` si codigo=1)
