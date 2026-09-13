# Declaración de uso de IAG en el TFM (borrador para el formulario de la UC3M)

Redactado en primera persona para pegarlo en el formulario `TFM_declaracion_uso_IAG_es_2026.pdf`. Todo lo que aquí se afirma está respaldado por el historial del repositorio (commits con coautoría, fechas, ficheros) o por el propio código del sistema. Los puntos marcados **[CONFIRMAR]** son decisiones o datos que solo Jorge puede cerrar.

---

## Parte 1: declaración sobre comportamiento legal, ético y responsable

Tal como está marcado en el PDF:

1. Datos de carácter confidencial: **SÍ, con autorización de los interesados.** (Corpus y anotaciones de los proyectos IRIS_IAMEDIA e InfoIA, no públicos, cedidos por el equipo del proyecto.)
2. Materiales protegidos por derechos de autoría: **SÍ**, con autorización o al amparo de las excepciones legales. (Piezas periodísticas publicadas, empleadas como fragmentos con fines de investigación; guías institucionales de lenguaje no sexista de acceso público.)
3. Datos de carácter personal: **NO.**
4. Respeto a los términos de uso y principios éticos: **SÍ.**

**[CONFIRMAR con las tutoras, punto 3]** Las piezas periodísticas del corpus contienen nombres de personas reales (personas protagonistas de noticias publicadas), y esas piezas se enviaron a las API de OpenAI y Google durante los experimentos. Las identidades de las personas anotadoras están seudonimizadas en el corpus (Indexa 1-5, UCM3 3-6) y no se facilitaron nombres. Conviene que el "NO" del punto 3 lo valides con Carmen y Teresa a la luz de lo primero, porque son ellas quienes conocen las condiciones de cesión del corpus.

---

## Parte 2: declaración de uso técnico

### Declaración A

**Declaro haber hecho uso del sistema de IAG Claude Code (Anthropic), con los modelos Claude Opus 4.8, Claude Opus 5 y Claude Fable 5.1, entre el 22/07/2026 y el 13/09/2026, para:**

#### Documentación y redacción

**Soporte a la reflexión en relación con el desarrollo del trabajo.**
He utilizado la herramienta como interlocutora para decidir la estructura del capítulo de resultados y el encuadre de la discusión. Por ejemplo, empleando el prompt "vamos con 5.1.6, aquí quizás lo dejaría solo en dónde sirve hoy el sistema pero quitaría todo lo de métricas promediadas", teniendo como interacción un análisis de qué dependía de la sección (referencias cruzadas desde otros capítulos, tablas huérfanas) y una propuesta con dos alternativas, tras la cual decidí retirar la sección y reubicar la estimación de coste de despliegue en la discusión. Del mismo modo, tras recibir los comentarios de las tutoras sobre la discusión, pedí que se contrastasen contra los datos: la herramienta comprobó que la hipótesis de uno de los comentarios (que el techo de la tarea valdría en las variables de poca prevalencia) no la sostenían los datos, y reformulé la sección en consecuencia.

**Revisión o reescritura de párrafos redactados previamente.**
Uso intensivo en los capítulos 5 y 6 y en el resumen. Por ejemplo, empleando el prompt "esto no es para un TFM (ten cuidado eh): 'la magnitud de lo que está en juego'", teniendo como interacción la reescritura del párrafo en registro académico. También he empleado la herramienta para propagar a los capítulos 1, 2, 3 y 6 y al resumen las decisiones tomadas al reescribir el capítulo 5 (por ejemplo, sustituir "el techo lo pone la tarea" por el reparto entre dos límites, o retirar el coeficiente kappa como métrica de referencia), y para una revisión final de todo el documento con siete agentes de revisión en paralelo, cuyos hallazgos quedaron recogidos en un informe que después apliqué de forma selectiva.

**Búsqueda de información o respuesta a preguntas concretas.**
Preguntas de LaTeX (por qué el tipo monoespaciado salía más grande dentro de las notas al pie, cómo cuadrar los factores `\hsize` de `tabularx`, por qué una referencia mostraba "??"), de métricas (la paradoja de kappa con prevalencias extremas, por qué exactitud y kappa se mueven en sentidos opuestos) y de interpretación de resultados (de dónde salía el factor 2,2 de coste entre los dos niveles del sistema).

**Búsqueda de bibliografía.**
No la he empleado para localizar bibliografía nueva. Sí para **contrastar** referencias ya citadas contra su fuente: descargó los PDF de ElSherief et al. (2021), Röttger et al. (2021), Pangakis et al. (2023) y Kocoń et al. (2023) y comprobó las cifras atribuidas. En el caso de Kocoń et al. detectó que la memoria atribuía a las tareas subjetivas una pérdida (25,5 %) que en el artículo es la media de todas las tareas, y la corregí con el dato que sí sostiene la afirmación (12,8 % al descartar las tareas de emociones). También verificó contra fuentes externas los metadatos de las seis referencias canónicas del libro de códigos antes de incorporarlas a la bibliografía.

**Resumen de bibliografía consultada.**
Solo en el sentido anterior: extracción de los pasajes concretos de esos artículos que respaldan (o no) una cifra citada. No he pedido resúmenes de artículos para sustituir su lectura.

**Traducción de textos consultados.**
**[CONFIRMAR]** No, salvo que Jorge lo haya usado fuera de las sesiones registradas.

#### Desarrollar contenido específico

**Asistencia en el desarrollo de líneas de código (programación).**
Dos usos distintos. (a) Los scripts de análisis y de generación de tablas del capítulo 5 (`Experimentos/analisis_cap5/`, diecisiete commits): scripts en Python que recalculan las métricas desde las predicciones de los modelos y emiten las tablas LaTeX, de modo que ninguna tabla de resultados se edita a mano. Por ejemplo, empleando el prompt "quiero una tabla como la 5.14 pero con B0 también, porque se habla de que 'en el nivel de control el gasto de cada modelo es prácticamente el mismo' y eso no se ve en ningún lado", teniendo como interacción la modificación del generador para emitir los dos niveles con cabeceras agrupadas. (b) Partes del código de experimentación (`experimento_21_agentskills`, `experimento_22_confianza`, ejecución en el clúster de GPU): dos de los siete commits del código de experimentos y dos de los tres del clúster llevan coautoría de la herramienta. **[CONFIRMAR]** qué partes concretas del código de la arquitectura de agentes se escribieron con asistencia y cuáles sin ella, porque el historial anterior al 22/07/2026 no lo registra.

**Generación de esquemas, imágenes, audios o vídeos.**
Las figuras de resultados (coste por variable, rendimiento frente a coste) las generé yo con matplotlib a partir de las cifras que la herramienta me preparó a petición mía ("dame los datos y métricas necesarias para rehacerla yo"). **[CONFIRMAR]** si los diagramas conceptuales (arquitectura B0/B1, anatomía de la skill, pipeline, enjambre de agentes) se hicieron con asistencia de IAG; si es así, hay que indicarlo en el pie de cada figura como pide el formulario.

**Procesos de optimización.**
Depuración de LaTeX (una redefinición de `\texttt` que entraba en recursión, tablas que desbordaban el margen, factores `\hsize` que no sumaban el número de columnas) y de la coherencia del documento (referencias cruzadas rotas tras reestructurar secciones, etiquetas duplicadas entre versiones de un fichero).

**Tratamiento de datos: recogida, análisis, cruce de datos.**
Uso central. La herramienta recalculó desde los ficheros CSV y las predicciones crudas todas las cifras citadas en el capítulo de resultados (unas 400 en la revisión final), detectó varias afirmaciones que los datos no sostenían (por ejemplo, que una configuración era "en ningún caso peor" cuando lo era en 7 de 20 celdas, o que la recuperación en vivo "nunca compensa" cuando mejora el F1 en 9 de 20) y calculó análisis que no estaban previstos: la dispersión de la anotación persona a persona dentro de cada equipo, el consenso de los cuatro modelos frente a la anotación, la estrategia de unión de modelos, y la comparación del tiempo de cómputo del modelo local entre los dos niveles del sistema.

**Inspiración de ideas en el proceso creativo.**
Varias propuestas de análisis surgieron de la herramienta y las incorporé tras verificarlas: tratar la dispersión entre personas anotadoras (y no solo entre equipos) como evidencia del techo de la tarea, probar la unión de modelos como alternativa al voto por mayoría, y seleccionar una pieza concreta del corpus para mostrar por dentro el razonamiento que conduce a un fallo de detección.

**Otros usos vinculados a la generación de puntos concretos del desarrollo específico del trabajo.**
Redacción de los mensajes de commit del repositorio y mantenimiento de la lista de tareas pendientes (`PENDIENTES.md`).

### Declaración B

**Declaro haber hecho uso del sistema de IAG GPT-4o-mini (OpenAI), a través de su API, para generar contenido que forma parte del propio sistema evaluado:** los cinco resúmenes de guías institucionales de lenguaje no sexista que la arquitectura carga como *skills* (`generar_resumenes_guias.py`). Cada resumen (250-450 palabras) se generó a partir de pasajes recuperados de las guías originales mediante cinco consultas fijas, y no fue revisado manualmente. Este uso se documenta en la memoria como parte de la implementación (Capítulo 4). **[CONFIRMAR]** que la memoria lo dice ya de forma explícita: la revisión final detectó que no se explicaba (hallazgo B1 de `REVISION_COMPLETA.md`) y está pendiente de redactar.

### Nota sobre los modelos evaluados

Los cuatro modelos de lenguaje que la memoria evalúa (gpt-4o-mini y gpt-5.4-nano de OpenAI, gemini-3.1-flash-lite de Google y gemma4:e4b ejecutado en local) son el **objeto de estudio** del trabajo, no herramientas de apoyo a su elaboración. Su uso, su coste y sus resultados se describen íntegramente en los capítulos 3, 4 y 5, y no se declaran aquí como asistencia a la redacción.

---

## Parte 3: reflexión sobre utilidad

**[BORRADOR en primera persona. Es el apartado más personal del formulario y conviene que lo reescribas con tu voz. Lo que sigue es fiel a lo que ha ocurrido en este trabajo.]**

La mayor fortaleza que he encontrado no ha sido la redacción, sino la verificación. Un trabajo con cinco variables, cuatro modelos, dos niveles y cuatro configuraciones produce cientos de cifras, y mantenerlas coherentes entre tablas, texto, discusión, conclusiones y resumen es un problema de escala que una persona sola resuelve mal. La herramienta pudo recalcular cada cifra citada desde los datos crudos y señalar las que no cuadraban o las afirmaciones que iban más allá de lo que los datos permitían. Varias conclusiones de versiones anteriores de la memoria eran demasiado categóricas, y fue ese contraste sistemático contra los datos el que obligó a matizarlas.

La segunda fortaleza ha sido la coherencia entre partes distantes del documento. Cuando las tutoras pidieron cambiar el encuadre de la discusión, ese cambio afectaba a la introducción, al resumen y a las conclusiones, y la herramienta localizó cada resto del encuadre antiguo que yo habría pasado por alto.

Las debilidades son de dos tipos. La primera es que la herramienta también introduce errores, y no los anuncia: en la revisión final aparecieron afirmaciones imprecisas que había escrito ella misma unos días antes ("sin apenas pagarlo en precisión", "el desacuerdo mide distancia de criterio") y que contradecían decisiones posteriores. Eso obliga a un flujo de trabajo en el que nada se acepta sin verificarse, y en el que la última revisión no puede hacerla quien escribió el texto, sea persona o modelo. La segunda es la tendencia a sobreafirmar y a uniformar el estilo: hubo que fijar reglas explícitas de registro académico y corregir repetidamente formulaciones que sonaban a veredicto donde los datos solo permitían un matiz.

En el proceso de aprendizaje me ha servido sobre todo para entender mejor las métricas de clasificación con clases desequilibradas (la paradoja de kappa, el desacoplamiento entre exactitud y acuerdo) y algunos detalles de LaTeX que de otro modo habría resuelto por ensayo y error. En la extracción de conclusiones, su aportación ha sido negativa en el mejor sentido: ha servido más para descartar conclusiones que para producirlas, y las que quedan en la memoria son las que sobrevivieron a ese contraste.

Hay, por último, una circunstancia particular de este trabajo que conviene dejar dicha. Su objeto de estudio son los propios modelos de lenguaje, y una de sus conclusiones es que esos modelos detectan de forma sistemática menos sexismo que las personas expertas. Haber empleado un modelo de lenguaje para redactar y revisar esa conclusión no la invalida, pero sí me ha hecho aplicarle a la herramienta la misma cautela que la memoria recomienda para el sistema que evalúa: como asistente que ordena y señala, no como decisor.
