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
      - Se retira tambien la subseccion del desacoplamiento entre exactitud y acuerdo,
        por el mismo motivo (su figura promediaba las cinco variables) y porque el
        fenomeno ya se describe en V25 y V33 y se discute en 5.2.2. El capitulo 5.1
        cierra ahora con el origen de la infradeteccion.
      - `fig_b0b1_barras_blanco.pdf` y `fig_b0b1_slope_blanco.pdf` quedan sin uso y
        pueden borrarse de `imagenes/`, igual que `fig_acuerdo_por_dolar_blanco.pdf`.
      - **Resuelto el 12/09/2026**: 5.2.2 ("Exactitud, kappa y prevalencia") se amplia
        con los dos casos que evidencian el desacoplamiento, gpt-4o-mini en V25 y
        gpt-5.4-nano en V33, con sus cifras.
      - **Resuelto el 12/09/2026**: la estimacion de coste de despliegue (8,77 USD
        asignando el mejor modelo a cada variable, frente a 13,16 con `gemini` y 6,89
        con `gpt-4o-mini`) se recupera en 5.2.4, en el parrafo que ya recomendaba
        tratar cada variable por separado. Cifras reverificadas contra
        `coste_por_variable.csv` y `metricas_por_variable.csv`.
- [x] **Figura de la Tabla 5.14** (`fig_coste_variable_blanco.pdf`): añadida el
      12/09/2026 acompañando a la tabla, no sustituyéndola, porque el texto cita
      cifras por modelo (3,43, 2,13, 2,46, 0,98) que la figura no rotula.
- [x] **Citas de 5.2 contrastadas contra la fuente** (12/09/2026, arXiv 2306.00176 y
      2302.10724). Pangakis et al. cuadra literal ("nine of the 27 tasks had either
      precision or recall below 0.5"). Kocon et al. estaba mal atribuido: el 25,5 % es
      la perdida media sobre las 25 tareas, no la de las subjetivas. Corregido, y se
      anade el dato que si sostiene la afirmacion, que al descartar las ocho tareas de
      emociones la perdida cae al 12,8 %.
- [x] **Comentarios de Carmen a 5.2** (30/08/2026), atendidos el 12/09/2026:
      - "El techo lo pone la tarea" no estaba demostrado. El titulo ya era "Dos limites,
        uno de la tarea y otro de los modelos". Se anade ahora el reparto por variable.
        Ojo, su hipotesis (que valdria en las de poca prevalencia) NO la sostienen los
        datos: el unico sitio donde esta documentado es sexismo_discurso, de prevalencia
        media (0,428), con 53,7 puntos de brecha entre equipos. En V25 y V26 los equipos
        se separan menos de 20 puntos, y en V33 y V35 unos 5. Conviene confirmarselo.
      - El "mayor acuerdo de control" del modelo local ya no se afirma. Se anade ademas
        que su ventaja en asimetria_mujer_hombre es de una sola pieza, y se apoya la
        recomendacion en masc_generico.
- [x] **Limitaciones** (12/09/2026). Nota anterior equivocada: el TFM SI las tiene, en
      un parrafo de `conclusions.tex` (Cap 6). Se le anaden las dos que faltaban de 5.1,
      los 81 y 131 positivos de V33 y V35, y que el salto semantico no es evaluable
      porque la anotacion no usa el valor 3 en ninguna pieza.
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
