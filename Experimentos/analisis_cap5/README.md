# Análisis del Capítulo 5 (métricas por variable)

Recomputación de todas las métricas del Capítulo 5 **por variable**, sin promediar
entre variables, tras los comentarios de las tutoras (07/09/2026).

## Cálculo

- `calc.py` — lee las predicciones crudas y el corpus anotado y escribe los CSV de
  métricas de este directorio. Es la fuente de las cifras del capítulo.
- `tab.py` — vuelca esos CSV a `tablas.md` para revisarlos de un vistazo.

## Datos

- `metricas_por_variable.csv` — B0 y B1, por modelo y variable: exactitud, precisión,
  recall, F1 de la clase positiva, F1 macro, kappa, prevalencia real y predicha, y
  la matriz de confusión (TP, FP, FN, TN).
- `ablacion_por_variable.csv` — lo mismo para las cuatro configuraciones de B1
  (bench, abl_minimo, abl_singuia, abl_sinres).
- `kappa_entre_modelos_por_variable.csv` — acuerdo entre pares de modelos, por
  variable y nivel.
- `equipos_por_variable.csv` — prevalencia anotada, recall y kappa de cada modelo,
  separando los dos equipos de anotación (Indexa y UCM3).
- `voto_mayoria.csv` — voto por mayoría de los cuatro modelos frente al ground truth.
- `coste_por_variable.csv` — coste de inferencia en USD por nivel, modelo y variable,
  tomado de los ficheros de coste de cada ejecución (no lo produce `calc.py`).
- `_todas_metricas.csv` — todo junto.
- `tablas.md` — los mismos datos en markdown, para lectura rápida.

## Generadores de tablas LaTeX

Escriben directamente en `chapters/5 results and discussion/`. Las tablas que producen
no se editan a mano.

- `gen_tablas_porvariable.py` → `tabla_var_v25.tex` … `tabla_var_v35.tex` (una por
  variable, B0 frente a B1).
- `gen_tablas_ablacion.py` → `tablas_ablacion.tex`.
- `gen_tabla_equipos.py` → `tablas_equipos.tex`.
- `gen_tabla_intraia.py` → `tabla_intraia.tex`.
- `gen_tabla_prev_modelo.py` → `tabla_prev_modelo.tex` (prevalencia de Sí predicha por
  cada modelo en B1).
- `gen_tabla_sintesis.py` → `tabla_sintesis.tex` (una fila por variable con el mejor
  modelo, su mejor configuración, el coste y si admite apoyo automático).
- `gen_tabla_coste_modelo.py` → `tabla_coste_modelo.tex` (coste en B1 desglosado por
  variable y modelo, solo los tres accesibles por API).

## Criterios

- Binarización: código 1 = No, códigos 2 y 3 = Sí. Clase positiva = Sí.
- N = 1.313 piezas con anotación experta, las mismas en todas las celdas.
- Las respuestas que un modelo no llegó a emitir en formato válido se contabilizan
  como negativas. En `gemma4:e4b` nivel B0 son 118 (76 en V25), y es el único caso
  con un número apreciable. Excluirlas en lugar de contarlas como negativo altera
  kappa como mucho en 0,005 (V35), de modo que no afecta a las conclusiones.
