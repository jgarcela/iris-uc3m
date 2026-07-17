# Skill 2 — Detección y etiquetado

Sobre el artículo y el catálogo de variables ya cargados:

1. Recorre las variables del bloque lenguaje (códigos **25–39**) que sean evaluables con el texto dado.
2. Para cada variable, asigna **una etiqueta** coherente con los **valores posibles** del catálogo (texto exacto o código numérico si el catálogo lo usa; sé consistente en todo el JSON).
3. Si no hay evidencia suficiente para una variable, indícalo con valor neutro según el catálogo (p. ej. "No" o el primer nivel de no detección) y `notas` breves.

Salida parcial esperada en el JSON unificado: objeto **`labels`** cuyas claves sean el **código** de variable (string, p. ej. `"25"`) y el valor incluya al menos `valor_etiqueta` y `notas`.
