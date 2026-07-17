# Estado del arte: arquitecturas de agentes, habilidades de agentes para periodismo y sesgos de género

**Informe de investigación** · Mayo 2026

---

## Resumen ejecutivo

Este informe sintetiza el estado del arte en la intersección de tres campos que evolucionan a gran velocidad: (1) arquitecturas de agentes basados en LLM, (2) su aplicación específica a flujos de trabajo periodísticos, y (3) la detección y mitigación de sesgos de género en medios y modelos. La literatura revela un campo en transición rápida desde sistemas mono-agente reactivos hacia arquitecturas multi-agente colaborativas, con un creciente reconocimiento de que el problema del sesgo —especialmente el de género— no se resuelve solo con técnicas de "debiasing" del modelo subyacente, sino que requiere ingeniería arquitectónica explícita: agentes especializados, herramientas de detección como tools, mecanismos de verificación humano-en-el-bucle, y "skills" portables que codifiquen el conocimiento profesional.

Para tu proyecto de app de política española, las implicaciones son directas: las decisiones de arquitectura (qué agentes, qué memoria, qué herramientas, qué skills) determinan más la calidad y equidad del producto que la elección del modelo base.

---

## 1. Arquitecturas de agentes: marco general

### 1.1 La definición actual de "agente"

Un agente LLM moderno se entiende como un sistema que combina un modelo fundacional con razonamiento, planificación, memoria y uso de herramientas, sirviendo de interfaz entre la intención en lenguaje natural y la computación en el mundo real (Xu, 2026, *AI Agent Systems: Architectures, Applications, and Evaluation*, arXiv:2601.01743). La taxonomía unificada más reciente descompone los agentes en seis dimensiones modulares:

1. **Componentes núcleo**: percepción, memoria, acción, perfilado
2. **Arquitectura cognitiva**: planificación, reflexión
3. **Aprendizaje**
4. **Sistemas multi-agente**
5. **Entornos**
6. **Evaluación**

Esta vista "architecture-first" (arXiv:2601.12560, *Agentic Artificial Intelligence: Architectures, Taxonomies, and Evaluation of Large Language Model Agents*, 2026) supera enfoques anteriores que agrupaban el campo por aplicaciones o paradigmas. Es la lente más útil para diseñar sistemas como el tuyo.

### 1.2 Patrones de diseño dominantes

#### Patrones mono-agente

- **ReAct** (Yao et al., 2022): intercala razonamiento y acción. Sigue siendo el patrón más prevalente. El agente genera razonamiento sobre un problema, interactúa con el entorno (APIs, búsqueda web), y usa los resultados para informar el siguiente paso, creando un bucle de retroalimentación que reduce alucinaciones.
- **Reflexion** (Shinn et al., 2023): añade auto-reflexión mediante feedback lingüístico. Usa un evaluador LLM para dar feedback sobre el estado de éxito, trayectoria actual y memoria persistente. Mejora la tasa de éxito y reduce alucinaciones frente a ReAct, pero es susceptible a mínimos locales no óptimos y limita la memoria al tamaño de la ventana de contexto (arXiv:2404.11584).
- **LATS** (Language Agent Tree Search): combina razonamiento, búsqueda en árbol y reflexión. Más potente pero computacionalmente caro.

#### Patrones multi-agente

Los sistemas multi-agente abordan tareas complejas mediante coordinación entre agentes especializados. Los frameworks principales en 2026 son:

- **LangGraph**: máquina de estados como grafo dirigido. Ofrece la granularidad de control más fina, ideal para producción con flujos de ejecución precisos. Es el único de los tres principales con persistencia durable y "human-in-the-loop" como capacidades de primera clase.
- **CrewAI**: centrado en role-playing y delegación de tareas. Velocidad de desarrollo más rápida con menor flexibilidad. Funciona bien cuando la estructura de salida se conoce de antemano (monitor → analizar → sintetizar → reportar).
- **AutoGen / AG2**: arquitectura multi-agente conversacional, basada en GroupChat. Excelente para escenarios que requieren negociación profunda entre agentes; mayor curva de aprendizaje. Microsoft ha desplazado el foco al Agent Framework, así que AutoGen está en modo mantenimiento (DEV Community, 2026).

Existen también MetaGPT, AgentVerse, DyLAN y otros frameworks orientados a colaboración estructurada.

### 1.3 Memoria de agentes

Du (2026, *Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers*, arXiv:2603.07670) formaliza la memoria como un bucle "write-manage-read" acoplado a percepción y acción. Identifica cinco familias de mecanismos:

1. Compresión residente en contexto
2. Almacenes aumentados por recuperación
3. Auto-mejora reflexiva
4. Contexto virtual jerárquico
5. Gestión por política aprendida

Para una app cívica como la tuya, la memoria es crítica: el perfil de alineación política del usuario depende de poder recordar votaciones, preferencias y contexto a través de sesiones.

### 1.4 La transición model-native

Una tendencia clave (arXiv:2510.16720, *Beyond Pipelines: A Survey of the Paradigm Shift toward Model-Native Agentic AI*, 2025) es el paso de capacidades agénticas externamente programadas (planificación, uso de herramientas, memoria) a comportamientos aprendidos extremo-a-extremo mediante refuerzo. Esto reorganiza el debate: en el futuro próximo, parte del orquestador "vive" dentro del modelo.

---

## 2. Habilidades de agentes (Agent Skills): el formato emergente

### 2.1 Qué son los Agent Skills

Anthropic introdujo en 2025 los **Agent Skills** como formato abierto para extender capacidades de agentes con conocimiento especializado y flujos de trabajo. Una skill es una carpeta que contiene un archivo `SKILL.md` con metadatos (nombre, descripción) e instrucciones que indican al agente cómo realizar una tarea específica. Pueden incluir scripts, materiales de referencia, plantillas y otros recursos (Anthropic Engineering, *Equipping agents for the real world with Agent Skills*, 2025).

Estructura mínima:

```
mi-skill/
├── SKILL.md       # Requerido: metadatos + instrucciones
├── scripts/       # Opcional: código ejecutable
├── references/    # Opcional: documentación
├── assets/        # Opcional: plantillas, recursos
└── ...
```

El mecanismo clave es la **divulgación progresiva**: el agente precarga solo nombre y descripción de cada skill instalada al inicio. La skill completa entra en contexto solo cuando el agente detecta una tarea relevante. Esto permite mantener miles de skills disponibles sin saturar el contexto.

### 2.2 La especificación abierta

La especificación se ha publicado como estándar abierto en `agentskills.io`, lo que significa que las skills creadas no quedan atadas a Claude. Funcionan en cualquier plataforma compatible. Existe un SDK Python de referencia.

### 2.3 Skills aplicadas a periodismo: el caso Hagar

Nick Hagar (Northwestern University) ha publicado una serie de tres estudios de caso aplicando Agent Skills a periodismo investigativo (*Generative AI in the Newsroom*, 2026):

#### Caso 1: Replicación de investigación de datos

Replicó la investigación de MuckRock/WHRO sobre descertificaciones policiales en Virginia usando Claude Code en menos de una hora con 20 minutos de revisión humana. Construyó tres skills:

1. **Python Runner**: elimina fricción de configuración del entorno usando `uv` para resolución automática de dependencias.
2. **Structured Data Preprocessing**: codifica el principio de que la limpieza de datos en periodismo debe ser transparente, reproducible y guiada por humano. Exige columnas de procedencia trazando cada fila a su archivo y número de fila origen. Requiere un flujo de cinco fases: cargar, auditar, reportar, transformar (solo tras aprobación humana), validar. Antes de transformar, Claude debe producir un informe de calidad de datos que marque problemas, proponga correcciones y exponga decisiones que requieren juicio humano.
3. **Structured Data Analysis**: captura la idea de que los hallazgos investigativos deben ser defendibles bajo escrutinio. Un modelo sofisticado que identifica "patrones sospechosos" es inútil si no puedes explicar la metodología en lenguaje claro.

Hallazgo crítico: sin las skills, Claude convirtió silenciosamente una fecha errónea ("10/04/0222") en una fecha inválida y reportó cifras incorrectas. Con las skills, el informe de calidad de datos marcó el typo explícitamente. Sin skills no habría detectado el error hasta mucho más tarde, si alguna vez.

#### Caso 2: Documentos desordenados (OCR + LLM)

Hagar advierte un fallo modal único de los LLM: la transcripción mediante LLM puede *inventar* texto de formas que el OCR tradicional nunca haría. Un OCR convencional puede leer mal un carácter; no inventará una frase. Esto requiere supervisión humana cuidadosa.

#### Caso 3: Generación de tipsheets investigativos

Probó la skill contra siete investigaciones premiadas con el Pulitzer. Aproximadamente la mitad de las pistas que reporteros humanos encontraron fueron recuperadas; un 20 % se perdieron por completo. Fue inesperadamente fuerte con formatos desordenados de datos y débil cuando las pistas requerían contexto externo o expertise de dominio.

### 2.4 El repositorio de Joe Amditis (Center for Cooperative Media)

Joe Amditis mantiene `jamditis/claude-skills-journalism`, un repositorio público con **31 skills y 11 hooks** para periodismo, incluyendo:

- Verificación de fuentes (método SIFT, búsqueda inversa de imagen, análisis de cuentas en redes sociales)
- FOIA (solicitudes de registros públicos, sistemas de seguimiento, apelaciones)
- Análisis de datos para redacciones
- Aplicación de AP Style
- Investigación previa a entrevista, marcos de preguntas
- Plantillas de pitch para distintos géneros periodísticos

Es referencia obligatoria para diseñar tu propio catálogo de skills cívicas.

### 2.5 Observación clave: skills > tools

En *Skills Beat Tools* (Substack, 2026), Hagar compara dar a Claude Code (a) un sistema RAG bien diseñado con embeddings versus (b) una skill markdown describiendo cómo navegar una biblioteca de Zotero. La skill produjo resultados mejores. La conclusión es contraintuitiva: una "instrucción profesional bien escrita" supera con frecuencia a infraestructura más sofisticada.

---

## 3. Aplicación de agentes a periodismo: estado del arte

### 3.1 Sistemas multi-agente para producción de noticias

#### AI-Press (Yang et al., 2024, arXiv:2410.07561)

AI-Press es probablemente el sistema multi-agente más completo publicado para producción periodística. Combina colaboración multi-agente con Retrieval-Augmented Generation (RAG). Sigue el modelo de "news flow" de Reuters (recolección, procesamiento, publicación, retroalimentación) y asigna agentes a roles editoriales: recuperación de material multi-fuente, creación de contenido, revisión, pulido. Incluye un sistema de simulación de feedback que genera retroalimentación pública según distribuciones demográficas. El uso de RAG mitiga las alucinaciones y mejora la precisión.

Limitaciones explícitas reconocidas: profesionalismo y juicio ético en la generación de noticias, y dificultad para predecir feedback público antes de la publicación.

#### TeleFlash (Maltezos, Kyrychenko, Knuutila, 2025, arXiv:2510.01193)

Caso de estudio reciente y muy relevante metodológicamente. Sistema desarrollado en la Universidad de Helsinki para apoyar a periodistas que cubren el conflicto Rusia-Ucrania mediante el monitoreo de canales de Telegram.

**Arquitectura modular** con cuatro componentes:

1. Módulo de recolección que interactúa con la API de Telegram
2. Módulo de filtrado con regex personalizables (modificables por periodistas sin programar)
3. Módulo de resumen impulsado por LLM (GPT)
4. Módulo de distribución a Slack para colaboración

Monitorea 170+ canales, almacena en PostgreSQL, ejecuta a las 6:00 AM diarias procesando las 24h previas. Genera reportes en inglés y finés con citas explícitas a IDs de mensajes.

**Hallazgos metodológicos clave** (12 entrevistas semiestructuradas con periodistas):

- Telegram destacó como fuente principal de actualizaciones en tiempo real, pues los posts aparecen antes que en wires tradicionales.
- Los periodistas insistieron: ningún sistema debe ser una "caja negra"; toda información debe incluir citas a posts originales para verificación contextual previa a publicación.
- El sistema satisfizo el filtrado por relevancia, resumen, colaboración y manejo de sobrecarga; falló en verificación de fuentes, detección de credibilidad e integración multi-plataforma.

El paper articula una taxonomía pragmática de necesidades periodísticas que cualquier sistema agéntico debería abordar (ver Tabla 1 del paper).

### 3.2 Discusión pública sobre la "redacción agéntica"

Pete Pachal (Media Copilot, *Fast Company*, 2026) argumenta que la pregunta clave no es si los agentes pueden escribir noticias, sino qué pasa con la redacción cuando todos trabajan agénticamente. Reporteros y editores han necesitado dominar muchos sistemas (CMS, project management, SEO, redes sociales). Los agentes pueden encargarse de toda esta infraestructura para que los humanos se concentren en el reporteo y la edición. La controversia surge cuando el modelo se aplica a la escritura misma (caso de The Plain Dealer en Cleveland).

### 3.3 Necesidades periodísticas y arquitectura

Síntesis de TeleFlash y los estudios de Hagar:

| Necesidad periodística | Capacidad arquitectónica requerida |
|---|---|
| Filtrado de información relevante | Tool calling + filtros parametrizables |
| Monitoreo en tiempo real | Scheduling + colas asíncronas |
| Manejo multilingüe | Modelos multilingües + glosarios de dominio |
| Resumen de grandes volúmenes | LLM con prompt engineering controlado |
| Colaboración y compartición | Integración con plataformas (Slack, etc.) |
| Verificación de fuentes | Cross-checking multi-fuente, evaluación de credibilidad |
| Trazabilidad / procedencia | Citation as first-class output |
| Integración multi-plataforma | API gateways, conectores |
| Feedback adaptativo | Memoria persistente + aprendizaje en línea |

---

## 4. Sesgos de género: el problema y su tratamiento computacional

### 4.1 Tipología de sesgos de género en IA

La revisión sistemática de Nadeem, Marjanovic & Lasswell (2023, *Journal of Telecommunications and the Digital Economy*) sintetiza 35 artículos clave y categoriza tres tipos de sesgo de género en IA: **societal, técnico e individual**. Las causas dominantes son societales y socio-técnicas; las estrategias más frecuentes para superarlos son debiasing, diseño de datasets y sensibilidad de género.

Caja Moya & Quiroga Rodríguez (2025, *AI and Ethics*, "Deconstructing gender bias in AGI: mitigating discriminatory architectures in general intelligence", https://doi.org/10.1007/s43681-025-00818-1) trazan el sesgo de género en IA a cuatro dimensiones interconectadas:

1. Representación sesgada en datasets de entrenamiento
2. Prácticas subjetivas de etiquetado
3. Decisiones algorítmicas no reguladas
4. Sesgos emergentes en patrones de interacción usuario-IA

### 4.2 La revisión crítica de 10 años (2025)

Un paper especialmente relevante: *Bias is a Math Problem, AI Bias is a Technical Problem: 10-year Literature Review of AI/LLM Bias Research* (arXiv:2508.11067, 2025) realiza tres hallazgos críticos:

1. Los investigadores estudian "bias" sin definir cómo lo conceptualizan.
2. Se restringen a sesgo de género, ignorando interseccionalidades.
3. No incluyen información sobre cómo los diseñadores de sistemas reales pueden implementar técnicas de debiasing.

Este es el "last mile gap" (Cabitza et al., 2020): los avances de debiasing académicos rara vez se implementan en sistemas reales. Crucialmente, los autores señalan que los métodos de debiasing se estudian en contextos mono-modelo, **sin considerar cómo los sistemas reales podrían implementarse en arquitecturas multi-agente con interacciones complejas de modelos**. Esto abre directamente el espacio que tu proyecto puede ocupar.

### 4.3 Sesgo de género en idiomas marcados gramaticalmente (español)

Trabajo crucial para tu contexto: Robles Carrillo & Magán Hervás (2024, *Leveraging Large Language Models to Measure Gender Representation Bias in Gendered Language Corpora*, arXiv:2406.13677). Es la primera propuesta de usar LLMs para identificar y clasificar sustantivos y pronombres gendereizados en español. Tras evaluación empírica en cuatro datasets de referencia, encuentran disparidad masculino:femenino entre 4:1 (Europarl) y 5–6:1 (WMT-News).

El dataset Europarl exhibe la menor disparidad, alineado con el compromiso institucional con la igualdad de género. Esta metodología es directamente aplicable a corpora de noticias políticas españolas y al BOE.

### 4.4 Detección computacional de sesgo en noticias

#### El Gender Gap Tracker (Asr, Mazraeh, Lopes, Gautam, Gonzales, Rao, Taboada, 2021, *PLoS ONE* 16(1): e0245533)

Es el sistema de referencia. Recolecta diariamente artículos de siete medios canadienses anglófonos, aplica NLP para identificar quién es mencionado y quién es citado por género, y publica resultados en dashboard público. La metodología combina modelos preentrenados de spaCy, reglas lingüísticas y matching de frases personalizado. Ha tenido extensiones:

- **Radar de Parité** (Soumah, Rao, Eibl, Taboada, 2023): versión francesa.
- **Gender bias in the news: a scalable topic modelling and visualization framework** (Rao & Taboada, 2021, *Frontiers in AI* 4(82)).

Toda la metodología está disponible en GitHub bajo GPL-3.0 (`sfu-discourse-lab/GenderGapTracker`) y es directamente adaptable al español. La extracción de citas mediante árboles sintácticos sigue trabajos previos (Pouliquen et al., 2007; van Atteveldt et al.; Krestel et al.).

#### Decoding News Bias (arXiv:2501.02482, 2025)

Estudio multi-bias que construye un dataset utilizando LLMs (GPT-4o-mini para anotación) y evalúa varios modelos (DistilBERT, ALBERT, XLNet) en detección de sesgos múltiples. Encuentran que modelos pequeños sufren especialmente con sesgos de género y sensacionalismo, y que el desbalance de clases es un problema persistente.

#### ViLBias (arXiv:2412.17052, 2024)

Benchmark VQA-style multimodal para sesgo en medios. 40k pares texto-imagen únicos, anotados híbridamente (LLM + revisión humana). Cubre framing ideológico explícito hasta manipulaciones visuales sutiles (recorte, puesta en escena, imaginería emotiva). Crucial porque el sesgo de género en medios opera frecuentemente a través de imagen, no solo texto.

### 4.5 Framing de género en política

#### Multi-Modal Framing Analysis of News (Arora, Yadav, Antoniak, Belongie, Augenstein, 2025, arXiv:2503.20960)

Primer estudio computacional de framing multimodal en noticias. Dataset de 500k artículos estadounidenses con frames automáticos validados con anotaciones humanas. Encuentra diferencias significativas entre cómo los frames se usan en imágenes versus textos, según orientación política y tema.

#### Decoding News Narratives (Pastorino, Sivakumar, Moosavi, 2024, arXiv:2402.11621)

Evaluación sistemática de GPT-3.5/4, FLAN-T5 y Llama 3 en detección de framing. Muestra que el rendimiento es altamente sensible al diseño del prompt y propenso a errores sistemáticos en casos ambiguos. GPT-4 exhibe mejor generalización cross-dominio pero tiende a confundir lenguaje emocional con framing. Introducen un dataset out-of-domain y muestran que el consenso entre múltiples modelos es señal útil para identificar anotaciones disputadas.

#### Frame In, Frame Out (arXiv:2505.05406, 2025)

Pregunta directa: ¿generan los LLMs titulares de noticias más sesgados que los humanos? Encuentra que la mayoría de los datasets de detección de framing se centran en política, dejando un hueco en la evaluación de framing interpretativo en otros dominios donde se espera neutralidad.

#### Benchmarking Gender and Political Bias in LLMs (arXiv:2509.06164, 2025)

Trabajo especialmente relevante para Europa. Estudios recientes han revelado sesgo de género persistente en el Parlamento Europeo: en debates parlamentarios, ciertos subgrupos (mujeres, miembros junior, representantes de estados pequeños) reciben desproporcionadamente menos atención y visibilidad (Walter et al., 2023). El sesgo de género persiste también en la cobertura política de noticias, con disparidades sistemáticas en elección de palabras, sentimiento y framing entre líneas ideológicas, incluso cuando se eliminan marcadores explícitos de género (Davis et al., 2022). El paper introduce **EuroParlVote dataset** y muestra que los LLMs comerciales como GPT-4o superan a alternativas open-weight en robustez y equidad para predicción de comportamiento parlamentario.

### 4.6 GMMP: el monitor longitudinal de referencia

El **Global Media Monitoring Project** es el estudio longitudinal más largo y grande sobre género en medios mundiales (1995-2025, ediciones quinquenales). El más reciente (7ª edición, mayo 2025) celebra 30 años post-Beijing.

Hallazgos clave 2025 (UN Women, *On Parallel Tracks: News Media and Gender Equality*, 2025):

- Solo el **26 %** de los sujetos y fuentes en noticias son mujeres globalmente.
- En 1995 el porcentaje era 17 %; en 2015, 24 %. Es decir, 9 puntos en 30 años.
- Las mujeres siguen siendo mucho más probables de aparecer en historias escritas por reporteras que por reporteros.
- **Solo 2 de cada 100 historias** retan los estereotipos de género de manera explícita; el periodismo que desafía estereotipos está en declive.

Este es el contexto factual sobre el que cualquier herramienta de detección de sesgo en español debe operar.

### 4.7 SESGO: evaluación específica para español

**SESGO: Spanish Evaluation of Stereotypical Generative Outputs** (arXiv:2509.03329, 2025) es el primer framework de evaluación culturalmente situado para modelos en español en contextos latinoamericanos. Construye sobre BBQ (Bias Benchmark for QA) usando preguntas subespecificadas para evaluar si los modelos apuntan desproporcionadamente a ciertos grupos demográficos en escenarios ambiguos. Cubre raza, clase, género y origen nacional con estereotipos extensamente documentados de contextos latinoamericanos.

### 4.8 E.D.I.A. Toolkit (Fundación Vía Libre, Argentina)

**E.D.I.A.** (Estereotipos y Discriminación en Inteligencia Artificial), de Alonso Alemany, Benotti & Busaniche (Vía Libre, Argentina, 2024) es un toolkit que permite a personas sin expertise técnico, pero con experiencia vivida, explorar, caracterizar y auditar sesgos y estereotipos en modelos de lenguaje. Adaptación en español de la metodología de Bolukbasi et al. (2016) considerando: género gramaticalmente marcado, listas de palabras adaptadas al español, asociaciones diferenciales (por ejemplo, "enfermera" se asocia mucho más al extremo femenino que "enfermero" al masculino). Es directamente relevante para tu contexto.

---

## 5. La intersección: agentes para detectar sesgo de género

Esta es la frontera más reciente y donde tu proyecto tiene espacio para innovar.

### 5.1 Bias-Aware Agent (arXiv:2503.21237, 2025)

Trabajo seminal en el espacio. Introduce un framework agéntico que **usa la detección de sesgo como una herramienta** del agente, no como post-procesamiento separado. Arquitectura de alto nivel:

1. El usuario envía consulta al agente.
2. El agente procesa, razona y decide qué fuentes consultar.
3. Recupera contenido de un vector store (RAG).
4. Aplica un detector de sesgo (modelo de clasificación) sobre el contenido recuperado.
5. Resalta los sesgos en cada fuente, mostrando cuánto sesgo carga cada fuente y cómo afecta a la respuesta global.

Contribución clave: el sistema da transparencia explícita sobre qué fuentes se usaron y cuánto sesgo cargan. Esto operacionaliza el principio periodístico de procedencia de manera auditable.

### 5.2 Structured Reasoning for Fairness (Huang & Fan, 2025, arXiv:2503.00355)

Framework multi-agente que **identifica sesgos sistemáticamente**:

1. Disuelve cada afirmación como hecho u opinión
2. Asigna una puntuación de intensidad de sesgo
3. Provee justificaciones concisas y factuales

Evaluado en 1500 muestras del dataset WikiNPOV, alcanza 84.9 % de precisión, mejora del 13.0 % sobre baseline zero-shot. La contribución metodológica clave es modelar explícitamente hecho-vs-opinión antes de cuantificar sesgo. Esto es directamente trasladable a cobertura de discurso político en España.

### 5.3 Mitigating Bias in Queer Representation (arXiv:2411.07656, 2024)

Pipeline colaborativo de agentes con agentes especializados para detección de sesgo y optimización, enfocado en uso inclusivo de pronombres. Evaluado en el dataset Tango, mejora 32.6 puntos porcentuales sobre GPT-4o en clasificación correcta de pronombres inclusivos. Demuestra que arquitectura multi-agente especializada supera ampliamente al modelo base en tareas de equidad lingüística.

### 5.4 Generative agents en fact-checking colaborativo (arXiv:2504.19940, 2025)

Resultados notables: las "multitudes de agentes" superan a multitudes humanas en clasificación de veracidad, exhiben mayor consistencia interna y muestran **menor susceptibilidad a sesgos sociales y cognitivos**. Comparados con humanos, los agentes confían más sistemáticamente en criterios informativos como precisión e informatividad, sugiriendo un proceso de toma de decisiones más estructurado. Hallazgo importante: los agentes pueden ser *menos* sesgados que humanos en ciertos contextos de evaluación, no necesariamente más.

### 5.5 La advertencia: amplificación de diferencias de género

*Diverse, but Divisive: LLMs Can Exaggerate Gender Differences in Opinion Related to Harms of Misinformation* (arXiv:2401.16558, 2024) hace un hallazgo contraintuitivo importante: GPT-3.5-Turbo refleja diferencias de género observadas empíricamente en opinión pero **amplifica el grado de estas diferencias**. Implicaciones para fact-checkers, diseñadores de algoritmos y uso de crowd-workers como anotadores: usar LLMs para tareas que involucran percepciones de género puede polarizar más de lo que refleja la realidad.

### 5.6 BiasScanner

BiasScanner (referenciado en arXiv:2501.02482) es una aplicación que aprovecha un LLM preentrenado para detectar oraciones sesgadas en artículos de noticias y proporciona explicaciones de sus decisiones. Es un ejemplo concreto del patrón "LLM-como-herramienta-de-auditoría" que podrías incorporar como sub-agente en tu sistema.

---

## 6. Sesgo de género en periodismo: contexto español

### 6.1 La situación documentada

El estudio bibliométrico de Olle (UOC, 2024, *Periodismo e inteligencia artificial generativa 2024*) incluye dato muy revelador: en el conjunto de 105 medios de 46 países encuestados, el sesgo de género en las propias respuestas (es decir, en quién participa en la encuesta sobre IA en periodismo) es **58.3 % hombres**. Esto sugiere que hasta la conversación sobre IA en redacciones está sesgada en su composición.

Los principales sesgos identificados en uso de IA por periodistas en español (Yahoo Noticias / Maldita.es, 2025):

- **Sesgos de género**: la IA puede adaptar respuestas según género percibido del usuario y reproducir ideas sexistas.
- **Sesgos racistas**.
- **Sesgo de adulación** (sycophancy): la IA da la razón al usuario, reforzando creencias y aislando de otros puntos de vista.
- **Sesgo de equidistancia**: chatbots evitan posicionarse incluso con evidencia abrumadora.
- **Sesgo de automatización**: usuarios confían en exceso en respuestas algorítmicas.

### 6.2 Anglocentrismo en LLMs y el español

Estudio reciente sobre IA en América Latina (sinembargo.mx, 2026) revela que cuando se usan modelos de lenguaje en español traducidos del inglés, se ignoran estereotipos, sesgos y contenido dañino profundamente arraigados en contextos locales, **reforzando la "perspectiva anglocéntrica"**. Esto tiene implicaciones críticas para tu producto: usar LLMs en español sin auditoría específica para contexto político español puede importar marcos analíticos angloamericanos.

### 6.3 Plataformas digitales como arquitecturas

Frontiers (2025, *Algorithmic gender representation in digital journalism*) propone el marco AGRP (Algorithmic Gender Representation in Platforms). Construye sobre Gillespie (2010): las plataformas no son intermediarios neutrales sino "**arquitecturas computacionales, económicas y políticas**" que activamente moldean la información que distribuyen. Esto es directamente aplicable a tu app: serás una plataforma que mediará entre legislación, votos y representación política. El diseño arquitectónico es decisión política.

---

## 7. Síntesis: qué significa esto para arquitecturas de agentes en tu proyecto

### 7.1 Decisiones de diseño con base en literatura

| Decisión | Recomendación con base en literatura |
|---|---|
| Modelo de orquestación | Multi-agente especializado por rol > mono-agente generalista (AI-Press, TeleFlash) |
| Framework | LangGraph para producción con human-in-the-loop crítico; CrewAI si la pipeline es secuencial conocida |
| Memoria | Memoria persistente con write-manage-read explícito (Du, 2026) |
| Skills | Catálogo modular con SKILL.md por tarea (Hagar; Amditis) |
| Detección de sesgo | Como tool del agente, no como post-procesamiento (Bias-Aware Agent) |
| Razonamiento sobre sesgo | Disolver hecho vs. opinión antes de puntuar intensidad (Huang & Fan) |
| Verificación | Cross-checking multi-fuente + procedencia explícita (TeleFlash; Skills de Hagar) |
| Evaluación | Endpoint anchoring + rubric-based (Hagar, *Notes on evaluating agentic tools*) |

### 7.2 Arquitectura tentativa para tu app

Una posible arquitectura agéntica para una app cívica sobre política española:

1. **Agente de ingesta legislativa**: monitorea BOE, calendarios parlamentarios, votaciones del Congreso/Senado. Skills: `boe-parser`, `votacion-mapper`, `presupuesto-analyzer`.
2. **Agente de contextualización**: enriquece cada propuesta legislativa con noticias relevantes y contexto histórico. Skills: `prensa-monitor`, `eu-context`.
3. **Agente de auditoría de género**: analiza el lenguaje de la propuesta y la cobertura mediática para detectar sesgos. Tool: detector de sesgo entrenado sobre corpus español. Skills: `gender-bias-detector` (basado en E.D.I.A. + Robles 2024).
4. **Agente de simplificación**: traduce jerga jurídica a lenguaje claro. Skill: `legalese-translator`.
5. **Agente de perfil de usuario**: construye y actualiza el perfil de alineación basándose en el patrón de votos. Memoria persistente.
6. **Agente de visualización**: genera el "Spain virtual" comparando voto ciudadano vs. voto parlamentario. Skill: `provincia-mapper`.

Cada agente puede operar bajo ReAct (razón + acción + observación), con un agente orquestador que aplique reflexión sobre la calidad agregada de la salida.

### 7.3 Riesgos específicos a tu dominio

- **Sesgo de equidistancia**: tu app se posicionará entre partidos. Decisión arquitectónica: ¿el agente "neutral" es realmente neutral o equidistante artificial? Documentar metodológicamente.
- **Sesgo de género en presentación de propuestas**: si los agentes resumen propuestas de forma diferente según género del proponente, replicas el problema.
- **Anglocentrismo importado**: cualquier LLM extranjero usado debe auditarse explícitamente sobre contexto político español.
- **Sycophancy del modelo de perfil**: si el sistema solo refuerza al usuario en su preferencia, contribuye a polarización (uno de los temas centrales del GMMP 2025).
- **Verificabilidad**: cualquier afirmación debe poder rastrearse a fuente primaria (lección de TeleFlash y de las skills de Hagar).

### 7.4 Métricas de equidad propuestas

Adaptando el Gender Gap Tracker al dominio cívico:

- **Ratio de citación parlamentaria por género**: % de citas a parlamentarias mujeres vs. parlamentarios hombres en los resúmenes de propuestas.
- **Ratio de framing**: distribución de frames usados al describir propuestas según género del proponente.
- **Asimetría de descriptores**: análisis cualitativo de adjetivos asociados a parlamentarias vs. parlamentarios.
- **Cobertura de propuestas con perspectiva de género**: % de propuestas que el sistema marca como con/sin impacto de género diferencial.

---

## 8. Bibliografía estructurada

### 8.1 Arquitecturas y habilidades de agentes

- Xu, B. (2026). *AI Agent Systems: Architectures, Applications, and Evaluation*. arXiv:2601.01743.
- (Anonymous). (2026). *Agentic Artificial Intelligence: Architectures, Taxonomies, and Evaluation of Large Language Model Agents*. arXiv:2601.12560.
- (Anonymous). (2024). *The Landscape of Emerging AI Agent Architectures for Reasoning, Planning, and Tool Calling: A Survey*. arXiv:2404.11584.
- Yao, S. et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*.
- Shinn, N. et al. (2023). *Reflexion: Language Agents with Verbal Reinforcement Learning*.
- Du, P. (2026). *Memory for Autonomous LLM Agents: Mechanisms, Evaluation, and Emerging Frontiers*. arXiv:2603.07670.
- (Anonymous). (2025). *Beyond Pipelines: A Survey of the Paradigm Shift toward Model-Native Agentic AI*. arXiv:2510.16720.
- (Anonymous). (2025). *From Language to Action: A Review of Large Language Models as Autonomous Agents and Tool Users*. arXiv:2508.17281.
- Anthropic. (2025). *Equipping agents for the real world with Agent Skills*. https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills
- Agent Skills Standard. (2025). https://agentskills.io
- Anthropic Skills Repository (2025): https://github.com/anthropics/skills

### 8.2 Agentes en periodismo

- Maltezos, V., Kyrychenko, R., & Knuutila, A. (2025). *How can AI agents support journalists' work? An experiment with designing an LLM-driven intelligent reporting system* (TeleFlash). arXiv:2510.01193.
- Yang, X. et al. (2024). *AI-Press: A Multi-Agent News Generating and Feedback Simulation System Powered by Large Language Models*. arXiv:2410.07561.
- Caswell, D., & Dörr, K. (2018). Automated journalism 2.0: Event-driven narratives. *Journalism Practice*, 12(4), 477-496.
- Diakopoulos, N. (2019). *Automating the news: How algorithms are rewriting the media*. Harvard University Press.
- Hagar, N. (2026). *Coding Agents for Investigative Journalism*. Generative AI in the Newsroom.
- Hagar, N. (2026). *Wrangling Messy Documents with Coding Agents*. Generative AI in the Newsroom.
- Hagar, N. (2026). *Building Investigative Tipsheets with Claude Code*. Generative AI in the Newsroom.
- Hagar, N. (2026). *Notes on evaluating agentic tools*. Substack.
- Amditis, J. (2026). *Claude Skills for Journalism* repository. https://github.com/jamditis/claude-skills-journalism
- Veerbeek, J. (2024). *How Teams of AI Agents Could Provide Valuable Leads For Investigative Data Journalism*. Generative AI in the Newsroom.
- Pachal, P. (2026). *What an agentic newsroom will look like*. Media Copilot / Fast Company.

### 8.3 Sesgos de género en IA y NLP

- Caja Moya, C., & Quiroga Rodríguez, E. (2025). Deconstructing gender bias in AGI: mitigating discriminatory architectures in general intelligence. *AI and Ethics*, 5, 5857-5865. https://doi.org/10.1007/s43681-025-00818-1
- Nadeem, A., Marjanovic, O., & Lasswell, B. (2023). Gender Bias in Artificial Intelligence: A Systematic Review of the Literature. *Journal of Telecommunications and the Digital Economy*.
- (Anonymous). (2025). *Bias is a Math Problem, AI Bias is a Technical Problem: 10-year Literature Review of AI/LLM Bias Research*. arXiv:2508.11067.
- Robles Carrillo, M., & Magán Hervás, A. (2024). *Leveraging Large Language Models to Measure Gender Representation Bias in Gendered Language Corpora*. arXiv:2406.13677.
- Bolukbasi, T. et al. (2016). Man is to computer programmer as woman is to homemaker? Debiasing word embeddings.
- Alonso Alemany, L., Benotti, L., & Busaniche, B. (2024). *E.D.I.A.: a democratising toolkit to audit biases and stereotypes in language models*. A+ Alliance / Fundación Vía Libre.
- (Anonymous). (2025). *SESGO: Spanish Evaluation of Stereotypical Generative Outputs*. arXiv:2509.03329.
- Fossa, F., & Sucameli, I. (2022). Gender Bias and Conversational Agents: an ethical perspective on Social Robotics. *Science and Engineering Ethics*, 28(3), 23.

### 8.4 Sesgos en medios y framing

- Asr, F.T., Mazraeh, M., Lopes, A., Gautam, V., Gonzales, J., Rao, P., & Taboada, M. (2021). The Gender Gap Tracker: Using Natural Language Processing to measure gender bias in media. *PLoS ONE*, 16(1): e0245533.
- Rao, P., & Taboada, M. (2021). Gender bias in the news: A scalable topic modelling and visualization framework. *Frontiers in Artificial Intelligence*, 4(82).
- Soumah, V.-G., Rao, P., Eibl, P., & Taboada, M. (2023). *Radar de Parité: An NLP system to measure gender representation in French news stories*. Canadian Conference on AI.
- Pastorino, V., Sivakumar, J.A., & Moosavi, N.S. (2024). *Decoding News Narratives: A Critical Analysis of Large Language Models in Framing Detection*. arXiv:2402.11621.
- Arora, A., Yadav, S., Antoniak, M., Belongie, S., & Augenstein, I. (2025). *Multi-Modal Framing Analysis of News*. arXiv:2503.20960.
- (Anonymous). (2025). *Frame In, Frame Out: Do LLMs Generate More Biased News Headlines than Humans?* arXiv:2505.05406.
- (Anonymous). (2025). *Benchmarking Gender and Political Bias in Large Language Models* (EuroParlVote). arXiv:2509.06164.
- (Anonymous). (2025). *ViLBias: Detecting and Reasoning about Bias in Multimodal Content*. arXiv:2412.17052.
- (Anonymous). (2025). *Decoding News Bias: Multi Bias Detection in News Articles*. arXiv:2501.02482.
- Walter, A. et al. (2023). Sobre invisibilidad de mujeres en el Parlamento Europeo.
- Davis, A. et al. (2022). Sesgo de género persistente en cobertura política.
- Lühiste, M., & Banducci, S. (2016). Invisible Women? Comparing Candidates' News Coverage in Europe.
- Global Media Monitoring Project (GMMP). (1995-2025). Ediciones quinquenales. WACC.
- UN Women (2025). *On Parallel Tracks: News Media and Gender Equality*. GMMP 2025 Highlights.

### 8.5 Agentes y detección de sesgos

- (Anonymous). (2025). *Bias-Aware Agent: Enhancing Fairness in AI-Driven Knowledge Retrieval*. arXiv:2503.21237.
- Huang, T., & Fan, E. (2025). *Structured Reasoning for Fairness: A Multi-Agent Approach to Bias Detection in Textual Data*. arXiv:2503.00355.
- (Anonymous). (2024). *Mitigating Bias in Queer Representation within Large Language Models: A Collaborative Agent Approach*. arXiv:2411.07656.
- (Anonymous). (2025). *Assessing the Potential of Generative Agents in Crowdsourced Fact-Checking*. arXiv:2504.19940.
- (Anonymous). (2024). *Diverse, but Divisive: LLMs Can Exaggerate Gender Differences in Opinion Related to Harms of Misinformation*. arXiv:2401.16558.

### 8.6 Periodismo en español e IA

- Olle, C. (2024). Periodismo e inteligencia artificial generativa 2024. *COMeIN*, UOC.
- Ufarte, M.J., & Manfredi, J.L. (2019, 2020). Estudios sobre IA en periodismo en español.
- Túñez-López, M., Fieiras, C., et al. (2019, 2024). Estudios sobre automatización de noticias en español.
- Fundación Gabo (2025). *La inteligencia artificial en los medios*.
- CIDOB. *Inteligencia artificial y periodismo: una herramienta contra la desinformación*.
- Frontiers (2025). *Algorithmic gender representation in digital journalism* (marco AGRP).

---

## 9. Notas finales

El campo cambia en meses, no años. Esta síntesis captura el estado a mayo de 2026, pero los siguientes meses traerán evoluciones especialmente en:

- Agentes "model-native" donde planificación, herramientas y memoria son aprendidos en el modelo, no orquestados externamente
- Skills generadas por los propios agentes (auto-codificación de patrones de comportamiento reutilizables)
- Benchmarks específicos para sesgo en multi-agente
- Frameworks de evaluación para "calidad" investigativa más allá de endpoints

La intersección que tú propones —agentes para participación cívica con conciencia de género en política española— está esencialmente desocupada en la literatura. Hay todas las piezas (detección de sesgo en español, frameworks multi-agente, skills periodísticas, monitoreo legislativo) pero no hay aún un paper o producto que las componga para el caso cívico-político en español. Esa es tu oportunidad.
