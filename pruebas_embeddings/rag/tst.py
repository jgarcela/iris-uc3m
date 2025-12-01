import configparser
import json
import ast
import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# --- 1. FUNCIÓN DE FILTRADO QUIRÚRGICO (Tu función original) ---
def cargar_definiciones_de_ini(ruta_ini, nombre_seccion):
    if not os.path.exists(ruta_ini):
        return {}

    parser = configparser.ConfigParser()
    parser.read(ruta_ini, encoding="utf-8")
    
    if nombre_seccion not in parser:
        raise ValueError(f"La sección '{nombre_seccion}' no existe en el INI.")

    items_seccion = dict(parser.items(nombre_seccion))
    
    try:
        lista_variables = ast.literal_eval(items_seccion.get('variables', '[]'))
    except:
        return {}

    mapa_default_seccion = {'1': 'No', '2': 'Sí'}
    for k, v in items_seccion.items():
        if k.upper().endswith("_VARS"):
            try: mapa_default_seccion = ast.literal_eval(v)
            except: pass
            break

    definiciones_filtradas = {}

    for var in lista_variables:
        var_upper = var.upper()
        mapa_encontrado = None

        # Búsqueda Exacta
        for k_ini in items_seccion.keys():
            if k_ini.upper() == var_upper:
                val_raw = items_seccion[k_ini]
                try:
                    val_clean = val_raw.replace("’", "'").replace("“", '"').replace("”", '"')
                    mapa_encontrado = ast.literal_eval(val_clean)
                except: pass
                break
        
        # Búsqueda Aproximada
        if not mapa_encontrado:
            for k_ini, v_ini in items_seccion.items():
                if k_ini.upper() == "VARIABLES" or k_ini.upper().endswith("_VARS"): continue
                if var_upper in k_ini.upper(): 
                    try:
                        v_clean = v_ini.replace("’", "'").replace("“", '"').replace("”", '"')
                        mapa_encontrado = ast.literal_eval(v_clean)
                        break
                    except: pass

        if not mapa_encontrado:
            mapa_encontrado = mapa_default_seccion

        definiciones_filtradas[var] = mapa_encontrado

    return definiciones_filtradas

# --- 2. CONFIGURACIÓN DEL LLM ---
llm = ChatOllama(
    model="gemma3:4b", 
    temperature=0,
    format="json" 
)

# --- 3. PROMPT DINÁMICO ---
template = """
Eres un analista de datos periodísticos.
Tu objetivo es clasificar el texto basándote ÚNICAMENTE en las variables de la sección: "{nombre_seccion}".

DEFINICIONES DE CÓDIGOS PARA ESTA SECCIÓN:
{definiciones}

INSTRUCCIONES:
1. Para cada variable listada, selecciona el código numérico.
2. Extrae "evidencia" (cita literal del texto).
   - REGLA: Si el código es negativo (No, No hay, Falso), la evidencia DEBE ser [].
3. Devuelve un JSON estrictamente con este formato:
{{
    "nombre_variable": {{ "codigo": "...", "evidencia": ["..."] }}
}}

NOTICIA:
Titular: {titular}
Texto: {texto}
"""

prompt = ChatPromptTemplate.from_template(template)


def analizar_seccion(titular, texto, nombre_seccion, ruta_ini="config.ini"):
    # 1. Cargar definiciones (Mapas de códigos)
    defs = cargar_definiciones_de_ini(ruta_ini, nombre_seccion)
    
    if not defs:
        return {"error": f"No se encontraron variables para la sección {nombre_seccion}"}
    
    print(f"DEBUG: Variables detectadas: {list(defs.keys())}")

    defs_str = json.dumps(defs, indent=2, ensure_ascii=False)
    
    # 2. Invocar LLM
    chain = prompt | llm
    
    response = chain.invoke({
        "nombre_seccion": nombre_seccion,
        "definiciones": defs_str,
        "titular": titular,
        "texto": texto
    })
    
    try:
        # El LLM nos da: { "variable": { "codigo": "1", "evidencia": [] } }
        json_raw = json.loads(response.content)
    except:
        return {"error": "JSON inválido", "raw": response.content}

    # --- 3. POST-PROCESADO: INYECCIÓN DE ETIQUETAS ---
    # Aquí es donde ocurre la magia para añadir la "etiqueta"
    json_enriquecido = {}

    for variable, datos in json_raw.items():
        # Validar estructura básica
        if not isinstance(datos, dict):
            json_enriquecido[variable] = datos
            continue

        codigo_elegido = str(datos.get("codigo", "")).strip()
        evidencia = datos.get("evidencia", [])

        # Buscar el mapa de esta variable en las definiciones cargadas
        mapa_codigos = defs.get(variable, {})
        
        # Obtener la etiqueta del mapa (Consistencia garantizada con el INI)
        # Si el código no existe en el mapa, ponemos "Valor desconocido"
        etiqueta_texto = mapa_codigos.get(codigo_elegido, "Valor desconocido")

        # Construir el nuevo objeto con el orden deseado
        json_enriquecido[variable] = {
            "codigo": codigo_elegido,
            "etiqueta": etiqueta_texto,  # <--- AQUI ESTÁ TU NUEVO CAMPO
            "evidencia": evidencia
        }

    return json_enriquecido

# --- 4. EJECUCIÓN DE PRUEBA ---

if __name__ == "__main__":
    titular = "Estudio internacional evidencia sesgos de género en IA: la investigadora María López, coautora del informe, critica que no se le dé protagonismo en los titulares"
    texto = """
            La investigadora principal María López (Universidad Complutense) presentó ayer en Madrid los resultados de un estudio internacional
            sobre sesgos de género en modelos de inteligencia artificial. "Los datos indican que las referencias a mujeres aparecen con menos frecuencia",
            explicó López en la rueda de prensa. El estudio señala además que, en muchos medios, los titulares tienden a priorizar los nombres masculinos.
            """

    # 1. Prueba FUENTES
    print("\n--- ANALIZANDO FUENTES ---")
    res_fuentes = analizar_seccion(titular, texto, "FUENTES")
    print(json.dumps(res_fuentes, indent=2, ensure_ascii=False))

    # 2. Prueba LENGUAJE
    print("\n--- ANALIZANDO LENGUAJE ---")
    res_lenguaje = analizar_seccion(titular, texto, "LENGUAJE")
    print(json.dumps(res_lenguaje, indent=2, ensure_ascii=False))