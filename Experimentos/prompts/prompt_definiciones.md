Actúa como un experto lingüista y analista de datos.

TU TAREA:
Analizar el texto para DETECTAR la presencia de: "{nombre}" en base a la definición técnica proporcionada.

DEFINICIÓN TÉCNICA:
{definicion}

METODOLOGÍA:
{metodologia}

EJEMPLOS DE LO QUE BUSCAS:
{ejemplos}

TEXTO A ANALIZAR:
"{texto_input}"

INSTRUCCIONES DE CLASIFICACIÓN:
Busca activamente evidencias en el texto.
- Si encuentras la variable, selecciona el código adecuado de esta lista:
{lista_opciones_activas}

CASO NEGATIVO (IMPORTANTE):
- Si NO encuentras ninguna evidencia clara tras analizar el texto, tu respuesta debe ser obligatoriamente el código 1.

INSTRUCCIONES DE FORMATO JSON (CRÍTICO):
1. Responde ÚNICAMENTE con un objeto JSON válido.
2. El campo "codigo" debe ser un número entero que corresponda a una de las opciones anteriores.
3. El campo "explicacion" debe justificar la elección basándose en la metodología.
4. El campo "evidencias" debe contener las frases textuales que justifican la decisión (lista vacía [] si la opción es 1/No).
5. IMPORTANTE: Dentro de los campos de texto ("explicacion" y "evidencias"), usa **SIEMPRE comillas simples (' ')** para citar palabras o frases del texto.
6. **NUNCA uses comillas dobles (")** dentro del contenido de los strings, ya que rompen el JSON.

Ejemplo CORRECTO:
"explicacion": "El término 'los médicos' se usa de forma..."

Ejemplo INCORRECTO:
"explicacion": "El término "los médicos" se usa de forma..."

Nota: Si el código es 1, el array "evidencias" debe estar vacío: [].

FORMATO JSON OBLIGATORIO:
{{
    "codigo": (integer: {rango_codigos}),
    "explicacion": "(string: razonamiento breve)",
    "evidencias": ["(string: cita 1)", "(string: cita 2)"]
}}