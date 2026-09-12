# TFM: pendientes

Tareas transversales que se dejan para una pasada final, cuando el texto ya no vaya
a cambiar. No abordarlas mientras se estén aplicando comentarios de las tutoras.

Última actualización: 12/09/2026

## Revisión final

- [ ] **Figuras: tamaño de letra.** Las gráficas tienen la fuente demasiado pequeña.
      El script matplotlib que las genera no está en el repositorio, así que hay que
      localizarlo o regenerarlas.
- [ ] **Figuras: tamaño y colocación** de las imágenes en general.
- [x] **Figura del acuerdo frente al coste**: rehecha el 12/09/2026 como
      `fig_rendimiento_coste_blanco.pdf`, sobre F1 macro en lugar de $\kappa$.
      Actualizados el pie y el párrafo que la comenta. `fig_acuerdo_por_dolar_blanco.pdf`
      queda sin uso y puede borrarse de `imagenes/`.
- [x] **Seccion "Donde sirve hoy el sistema" retirada** (12/09/2026). Se quito primero
      todo lo promediado (la tabla de B0 frente a B1 por metrica media y los dos
      parrafos que la comentaban) y despues la seccion entera, porque su lectura de
      conjunto ("de las cinco variables solo una admite hoy apoyo automatico") resultaba
      demasiado categorica. El capitulo cierra ahora con el desacoplamiento entre
      exactitud y acuerdo. Consecuencias:
      - `tabla_sintesis.tex` y su generador quedan sin uso.
      - `fig_b0b1_barras_blanco.pdf` queda sin uso y puede borrarse de `imagenes/`,
        igual que `fig_acuerdo_por_dolar_blanco.pdf`.
      - **Pendiente**: la estimacion de coste de despliegue (8,77 USD asignando el mejor
        modelo a cada variable, frente a 13,16 con `gemini` y 6,89 con `gpt-4o-mini`)
        solo estaba ahi. Valorar si recuperarla en la discusion (5.2), que es donde el
        texto la remite ahora.
- [x] **Figura de la Tabla 5.14** (`fig_coste_variable_blanco.pdf`): añadida el
      12/09/2026 acompañando a la tabla, no sustituyéndola, porque el texto cita
      cifras por modelo (3,43, 2,13, 2,46, 0,98) que la figura no rotula.
- [ ] **Mover el código a `github.com/jgarcela/tfm-iris-uc3m`** (repo creado el
      06/09/2026, vacío por ahora). El Apéndice B ya lo enlaza. Solo el código: el
      corpus y las anotaciones de IRIS_IAMEDIA e InfoIA no se difunden con él.
- [ ] **Glosario** (`chapters/glosario.tex`): repasar que la lista esté completa y
      que el tratamiento tipográfico que declara se cumpla en todo el documento.
      Ya corregidos así `codebook` (se traduce a "libro de códigos") y
      `skill` / `Agent Skills` (cursiva y mayúsculas).

## Contenido

- [ ] **Comentario de la "ceguera" (Carmen).** Reencuadrar el trabajo de
      "divergencia de opiniones" a "dificultad de detección / ceguera patriarcal".
      Afecta al abstract, a la discusión del recall y a las implicaciones para IRIS.
      Empezado en la introducción del Cap 5, falta el resto. Los términos "ceguera
      patriarcal" y "espejismo de neutralidad androcéntrica" están sin citar.
- [ ] **`NEW_REUNION.md`**: los nueve puntos de la reunión con las tutoras.
- [ ] **Comentarios de las tutoras en la Discusión (5.2).** Jorge todavía no los ha
      pasado, así que `discussion.tex` no está cerrado. Lo mismo vale para el Cap 6,
      que además arrastra el encuadre antiguo ("el techo lo pone la tarea") y hay que
      poner al día con las conclusiones del Cap 5 reordenado.
- [ ] **Rehacer el abstract** al final, una vez reordenado el Cap 5.
- [ ] Apellido de Harold en los agradecimientos.
- [ ] Decidir el título definitivo del TFM.

## Entrega

- [ ] Compilar en Overleaf y subir las figuras que están en `.gitignore`.
