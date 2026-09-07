# Análisis del Capítulo 5 (métricas por variable)

Recomputación de todas las métricas del Capítulo 5 **por variable**, sin promediar
entre variables, tras los comentarios de las tutoras (07/09/2026).

## Ficheros

- `calc.py` — script que genera todo, reproducible.
- `metricas_por_variable.csv` — B0 y B1, por modelo y variable: exactitud, precisión,
  recall, F1 de la clase positiva, F1 macro, kappa, prevalencia real y predicha, y
  la matriz de confusión (TP, FP, FN, TN).
- `ablacion_por_variable.csv` — lo mismo para las cuatro configuraciones de B1
  (bench, abl_minimo, abl_singuia, abl_sinres).
- `kappa_entre_modelos_por_variable.csv` — acuerdo entre pares de modelos, por
  variable y nivel.
- `voto_mayoria.csv` — voto por mayoría de los cuatro modelos frente al ground truth.
- `_todas_metricas.csv` — todo junto.
- `tablas.md` — las tablas en markdown.

## Criterios

- Binarización: código 1 = No, códigos 2 y 3 = Sí. Clase positiva = Sí.
- N = 1.313 piezas con anotación experta.
- Predicciones fallidas excluidas del cálculo. En `gemma4:e4b` nivel B0 hay 118
  (76 en V25). Contarlas como negativo en lugar de excluirlas altera kappa como
  mucho en 0,005 (V35), de modo que no afecta a las conclusiones.
