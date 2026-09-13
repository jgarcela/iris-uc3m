# TFM: pendientes

Tareas transversales que se dejan para una pasada final, cuando el texto ya no vaya
a cambiar. No abordarlas mientras se estén aplicando comentarios de las tutoras.

Última actualización: 12/09/2026

## Revisión final

- [x] **Bibliografia del libro de codigos** (12/09/2026). Se anaden las seis
      referencias canonicas que `variables.json` declara y que faltaban (Garcia
      Meseguer 1977 y 1994, Lledo Cunill 1992, Sainz de Baranda 2014, Bengoechea 2015,
      Bengoechea y Calero 2003), citadas variable a variable en `iris_variables.tex`
      segun el mapa del propio libro de codigos. Metadatos verificados contra fuentes
      externas, los de `variables.json` eran correctos. No confundir con el bloque de
      guias institucionales, que son las nueve del RAG (`methodology_manifest.json`) y
      se citan en `iris_analisis_experto.tex`.
- [x] **Figuras: tamaño de letra.** Revisadas por Jorge el 13/09/2026.
- [x] **Figuras: tamaño y colocación.** Revisadas por Jorge el 13/09/2026.
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
- [ ] **Descartado por falta de tiempo (12/09/2026), de `REUNION_DIRECTORA.md`.** Tres
      compromisos de la reunion del 29/07 que no han llegado a la memoria. Se dejan
      fuera a proposito, no por olvido:
      1. **Analisis de confianza y calibracion (exp22).** Los datos estan en
         `experiments/experimento_22_confianza/`. La probabilidad declarada es bimodal,
         el modelo esta mal calibrado y en los desacuerdos esta confiadamente
         equivocado. Version barata si hace falta: tres frases en el parrafo de
         limitaciones del Cap 6 diciendo que se exploro y no dio un filtro util.
      2. **Las diez piezas del analisis cualitativo.** Se acordo seleccionar 10 y que
         las expertas las anotasen. La memoria lleva una (5.1.5.4) y es lectura propia.
         Depende de terceros, no es viable ahora.
      3. **La contradiccion del libro de codigos en V25** (regla de inversion frente a
         clausula agregadora). Era la decision 3 que se le planteaba a la directora y
         no consta si se resolvio. No se menciona en la memoria.
- [ ] **Mover el código a `github.com/jgarcela/tfm-iris-uc3m`** (repo creado el
      06/09/2026, vacío por ahora). El Apéndice B ya lo enlaza. Solo el código: el
      corpus y las anotaciones de IRIS_IAMEDIA e InfoIA no se difunden con él.
- [x] **Glosario** (12/09/2026). Se retira la frase que prometia glosar cada termino
      en su primera aparicion, porque solo se cumplia en 2 de 12 casos. Se unifica
      "revelacion progresiva" (el cuerpo decia "divulgacion" en un sitio). Se anaden
      GPU y JSON a la tabla de acronimos, expandidos ademas en su primera aparicion, y
      golden standard y Ollama a la de terminos. Comprobado que VP, FP, VN y FN si se
      definen en eval.tex y que los hsize de las dos tablas suman 3,00.

- [x] **Comentario de la "ceguera" (Carmen)** (12/09/2026). Reencuadrado en los dos
      unicos sitios donde sobrevivia el enfoque de "divergencia de opiniones", la
      entradilla del Cap 5 y 5.2.3: el bajo acuerdo se explica ahora por la dificultad
      de percibir el fenomeno, tambien para quien esta formada, y no por criterios
      enfrentados. **Los dos terminos siguen sin citar**, por decision de Jorge. No he
      encontrado fuente que los acune y no procede inventarla.
- [x] **`NEW_REUNION.md`** (12/09/2026): 8 de sus 10 puntos hechos (kappa fuera como
      metrica de referencia, sin conclusiones categoricas, promedios fuera y analisis
      por variable con coste por variable, el recall bajo discutido, la literatura de
      que los modelos deben mejorar, el ejemplo de noticia analizada, el enfasis en el
      plano cualitativo y por que no esta, y FP frente a FN). Queda solo el tamano de
      las figuras, que ya tiene entrada propia arriba.
- [x] **Comentarios de Carmen a la Discusion (5.2)** (12/09/2026). Los dos atendidos.
      El titulo "El techo lo pone la tarea" ya no existe y 5.2.3 reparte el desajuste
      entre los dos limites. **Ojo al trasladarselo**: su hipotesis de que el techo de
      la tarea valdria en las variables de poca prevalencia NO la sostienen los datos,
      ya que el unico sitio donde esta documentado es sexismo_discurso, de prevalencia
      media (0,428). El Cap 6 y el resumen quedan tambien al dia.
- [x] **Abstract rehecho** el 12/09/2026, con el Cap 5 ya cerrado.
- [x] Apellido de Harold: confirmado el 12/09/2026, es Harold Molina, tal como ya
      figura en la dedicatoria. Sin cambios.
- [x] Título del TFM: el usuario confirma el 12/09/2026 que se queda como está.

## Entrega

- [x] Compilar en Overleaf y subir las figuras que están en `.gitignore` (hecho por Jorge, 12/09/2026).
