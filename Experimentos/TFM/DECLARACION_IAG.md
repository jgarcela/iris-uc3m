# Declaración de uso de IAG en el TFM

Texto para pegar en el formulario `TFM_declaracion_uso_IAG_es_2026.pdf`. Redactado en primera persona.

---

## Parte 1

Tal como está marcado en el PDF: SÍ he usado datos confidenciales con autorización (el corpus y las anotaciones de IRIS_IAMEDIA e InfoIA), SÍ materiales protegidos al amparo de las excepciones legales (noticias publicadas, como fragmentos con fines de investigación), NO datos de carácter personal, y SÍ he respetado los términos de uso.

---

## Parte 2: declaración de uso técnico

**Declaro haber hecho uso del sistema de IAG Claude (Anthropic), a través de la herramienta Claude Code, durante la redacción y revisión de este trabajo, para:**

### Documentación y redacción

**Soporte a la reflexión.** La he usado como interlocutora para decidir cómo organizar el capítulo de resultados y cómo enfocar la discusión: qué secciones sobraban, en qué orden contar los resultados y qué conclusiones podía sostener con los datos que tenía. Por ejemplo, cuando dudaba de si mantener una sección de síntesis con métricas promediadas, le pedí que me dijera qué dependía de ella y qué alternativas tenía, y con eso decidí retirarla. Cuando recibí los comentarios de mis tutoras, le pedí que los contrastara con los datos antes de decidir cómo responder a cada uno.

**Revisión o reescritura de párrafos redactados previamente.** Sobre todo en los capítulos de resultados y conclusiones y en el resumen. Le pedía que reescribiera párrafos que me sonaban poco académicos o demasiado categóricos, que aplicara unas reglas de estilo que fijé al principio (registro académico, siglas en inglés desarrolladas en su primera aparición, fechas en un formato único, uso coherente del femenino para "la agente"), y que propagara al resto de la memoria los cambios de enfoque que iba tomando con mis tutoras, para que la introducción, el resumen y las conclusiones no se quedaran con la versión anterior. También me ayudó a recortar el resumen para que cupiera en una página sin perder ninguna de las conclusiones.

**Búsqueda de información o respuesta a preguntas concretas.** Dudas de LaTeX (tablas que no cuadraban, referencias rotas, tamaños de fuente, cómo se incrustan los metadatos del PDF), de métricas de clasificación (por qué la exactitud y el coeficiente kappa pueden moverse en sentidos opuestos cuando las clases están desequilibradas), y de normativa, como qué artículo del Reglamento de Inteligencia Artificial excluye a los sistemas de investigación científica.

**Búsqueda y resumen de bibliografía.** No la he usado para buscar bibliografía nueva. Sí para contrastar referencias ya citadas con su fuente original y comprobar que las cifras que atribuía a esos trabajos eran las correctas; en un caso detectó que había atribuido a un artículo un dato que no decía exactamente eso, y lo corregí. También para completar y verificar los metadatos de las referencias del libro de códigos del proyecto antes de incorporarlas a la bibliografía, y para mantener el fichero de bibliografía coherente (claves citadas que faltaban, campos incompletos).

**Glosario.** Me ayudó a construir y revisar el glosario de términos en inglés y de acrónimos, comprobando qué términos se usaban realmente en la memoria y cuáles faltaban o sobraban.

**Traducción.** No.

**Revisión final.** Al terminar, le pedí una revisión completa de la memoria, capítulo por capítulo, buscando cifras que no cuadraran con los datos, afirmaciones que fueran más allá de lo demostrado, contradicciones entre capítulos y problemas de estilo. Los hallazgos los revisé uno a uno y decidí cuáles aplicar y cuáles no.

### Desarrollar contenido específico

**Programación.** Me ha ayudado con los scripts en Python que calculan las métricas a partir de las predicciones de los modelos y generan las tablas de la memoria, y con partes del código de experimentación y de su ejecución en el clúster de GPU.

**Generación de esquemas o imágenes.** Las gráficas de resultados y los diagramas de la memoria los ha generado la herramienta a partir de los datos y las tablas. Previamente creé unos archivos de estilo con los colores, las tipografías y la estética de la página web de IRIS, para que todas las figuras tuvieran coherencia entre sí y con el proyecto.

**Procesos de optimización.** Depuración de errores de LaTeX, de código Python y de coherencia del documento tras reestructurar secciones.

**Tratamiento de datos.** Ha sido el uso más útil: recalcular desde los datos crudos las cifras que cito en la memoria, comprobar que las afirmaciones del texto las sostienen, y hacer análisis complementarios que no tenía previstos, como la dispersión de la anotación entre personas o la combinación de varios modelos.

**Inspiración de ideas.** Algunas propuestas de análisis surgieron en la conversación con la herramienta y las incorporé después de verificarlas con los datos. También le pedí ideas para organizar las secciones y los contenidos de la memoria, buscando un flujo narrativo coherente y fácil de seguir, de manera que cada capítulo prepare el siguiente y las conclusiones se apoyen en lo ya mostrado.

**Otros usos.** Mantenimiento de la documentación de trabajo en ficheros Markdown: la lista de tareas pendientes con sus fechas límite y recordatorios, el registro de los cambios aplicados en cada sesión, las notas de las reuniones con mis tutoras y el seguimiento de sus comentarios hasta darlos por resueltos, y el informe consolidado de la revisión final de la memoria, con cada hallazgo, su estado y la decisión tomada.

**Además, declaro haber usado el modelo GPT-4o-mini (OpenAI) para generar los resúmenes de guías de lenguaje no sexista que el sistema evaluado carga como parte de su metodología.** Los cuatro modelos de lenguaje que se evalúan en la memoria son el objeto de estudio del trabajo y no se declaran aquí como herramientas de apoyo.

---

## Parte 3: reflexión sobre utilidad

A mi modo de verlo, la cuestión con la IA no es si se usa, sino de qué modo y con qué objetivo.

Lo que más me ha servido no ha sido escribir, sino comprobar, y tenerla como asistente durante todo el proceso. Un trabajo con cinco variables, cuatro modelos y varias configuraciones produce cientos de cifras, y mantenerlas coherentes entre tablas, texto y conclusiones es difícil de hacer a mano, y llevaría bastante tiempo. Con la IA podía recalcularlas todas desde los datos y obtener cuáles no cuadraban o qué afirmaciones iban más allá de lo que los datos permitían. Otra ayuda que no esperaba tanto ha sido la de mantener al día la documentación de trabajo: una lista de tareas pendientes con fechas, las notas de las reuniones con mis tutoras y el seguimiento de sus comentarios hasta cerrarlos, y un registro de lo que iba cambiando en cada sesión. Con un trabajo de meses y muchas versiones, eso me ha evitado perder cosas por el camino. También me ha ayudado a mantener la coherencia del documento entero. Cuando mis tutoras me pidieron cambiar ciertos enfoques, había cambios que tocaban varias secciones de la memoria y me permitió poder detectar rápidamente si me había dejado algo sin cambiar. En cuanto a las conclusiones, me ha servido más para descartar que para producir: varias de las primeras versiones eran demasiado rotundas, y fue el contraste con los datos el que me obligó a matizarlas.

En cuanto a las debilidades, es indudable que comete errores y no avisa: en la revisión final aparecieron frases imprecisas que había escrito ella misma unos días antes. Por eso no he incorporado nada a la memoria sin leerlo y contrastarlo antes.

En general, me ha servido en todo el desarrollo y el proceso de este TFM a entender nuevos conceptos, errores de redacción, mantener coherencia de la memoria y a trabajar con un asistente "24h al día" que me permitía seguir aprendiendo de una manera muy sencilla y cómoda.
